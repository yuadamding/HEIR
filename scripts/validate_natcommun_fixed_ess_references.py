#!/usr/bin/env python3
"""Fixed-support/ESS NatCommun reference-bank sensitivity.

This is a validation-only wrapper around the frozen generative development
runner.  Natural ST/single-cell training arrays are never reweighted.  The
only intervention is a scoped prediction-time adapter that fits state anchors
from query-excluded training references, verifies natural state support, and
matches joint QC/state mass plus componentwise and total ESS for the matched,
same-indication wrong, and pooled-generic reference mixtures.

``prepare`` may read the frozen public folds and raw single-cell QC metadata,
but never a score target.  ``fit-predict`` cannot accept a score target.
``score`` validates every prediction globally before opening any target.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

SCHEMA = "heir.natcommun_fixed_ess_reference_report.v3"
PREPARED_SCHEMA = "heir.natcommun_fixed_ess_reference_prepared.v3"
PREDICTION_SCHEMA = "heir.natcommun_fixed_ess_reference_predictions.v3"
PLAN_SCHEMA = "heir.natcommun_fixed_ess_reference_plan.v3"
SCORE_TARGET_SCHEMA = "heir.natcommun_fixed_ess_reference_score_targets.v3"
AUTHORITY_SCHEMA = "heir.natcommun_natural_state_authority.v3"
HOPTIMUS_REPOSITORY = "bioptimus/H-optimus-1"
HOPTIMUS_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
FROZEN_BASE_SEED = 1729
FROZEN_EPOCHS = 80
FROZEN_BATCH_SIZE = 256
FROZEN_LATENT_DIM = 20
FROZEN_MAX_STATE_COMPONENTS_PER_TYPE = 3
FROZEN_MIN_STATE_COMPONENTS_PER_TYPE = 2
FROZEN_STATE_SUPPORT_CELLS = 4
FROZEN_STATE_SUPPORT_ESS = 4.0
FROZEN_STATE_PROXIMITY_QUANTILE = 0.95
FROZEN_STATE_MAX_PROXIMITY_RATIO = 16.0
FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO = 0.05
FROZEN_STATE_MAX_TILT = 8.0
FROZEN_STATE_MIN_JOINT_RETENTION = 0.50
FROZEN_STATE_MIN_GLOBAL_CELL_COVERAGE = 0.50
FROZEN_STATE_MIN_TYPE_COVERAGE = 0.50
FROZEN_STATE_MIN_RELATIVE_WEIGHT = 1.0e-4
FROZEN_STATE_MAX_WEIGHTED_MEAN_SHIFT_RATIO = 4.0
STATE_BALANCE_METHOD = (
    "query_excluded_training_anchors_natural_support_joint_QC_state_"
    "proximity_tilt_exact_component_ESS"
)
FROZEN_PROTOCOL_SHA256 = "097803da2f84e2f1e93aaaf3f2bc6c95e5be3de9844929c5524964b4c904e2f5"
EXPECTED_HASHES = {
    "source_sha256": "ec37d5717a9b737dfac226ae9267258fb728ee024496a7655bb69a913aa3cf20",
    "projected_source_sha256": "71479f891b5945762e20ec5b91d85bac097230b12ed9192aeacd965be119607f",
    "panel_sha256": "ce0b6b82440d7fccc69f24afccf0c68bb101b85f590e4a35a514309929fbb6ad",
    "development_protocol_sha256": (
        "2cb92b22b6870488a06e64b213e37ffbbdfe3044f1da8fc7442f506915e78197"
    ),
    "development_runner_sha256": "cf27504e25dfd8cd7e8bfe2894efc8b4a8f79306b47bc492d0e61406d20668ce",
    "generative_core_sha256": "55a63f1360e8cc76267e4b00ba8e2167f36259789e9bfdf2aa929c8cadd83b17",
    "prepared_manifest_sha256": "d1b1353abb9ee80c3132fa08a9ebaea3aeeee607aaa11f2c90e7121be02addde",
    "fit_predict_manifest_sha256": (
        "cb7ebdf9e22090a046937204993a7b2aa3ac1ba2d4c434883e43ed45d1e826ca"
    ),
    "development_report_sha256": "bf3144cf22405752488509dbb1a65b573967fe1b14110881020787187828cf29",
}
DEFAULT_BASELINE_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_generative_development")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_fixed_ess_reference_sensitivity_v3")
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[1] / "configs/natcommun_fixed_ess_reference_sensitivity.json"
)
BASELINE_RUNNER = Path(__file__).resolve().parent / "benchmark_natcommun_generative_development.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def _atomic_npz(path: Path, arrays: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar string")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _load_baseline_runner() -> Any:
    name = "_heir_frozen_natcommun_development_runner"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BASELINE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen NatCommun runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ess(weights: object) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(values < 0) or not values.sum() > 0:
        raise ValueError("ESS requires non-negative weights with positive mass")
    normalized = values / values.sum()
    return float(1.0 / np.sum(normalized**2))


def _distance_tilt_weights(
    distances: Sequence[float],
    category_ids: Sequence[int],
    category_masses: Mapping[int, float],
    tilt: float,
) -> np.ndarray:
    """Maximum-entropy QC weights tilted toward a fixed state anchor.

    Unlike the rejected cell-ID concentration rule, this construction is
    invariant to row order and observation renaming.  It preserves every joint
    QC margin exactly and uses only the training-derived anchor distance.
    """

    values = np.asarray(distances, dtype=np.float64)
    categories = np.asarray(category_ids, dtype=np.int64)
    if (
        values.ndim != 1
        or categories.shape != values.shape
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isfinite(float(tilt))
        or float(tilt) < 0
    ):
        raise ValueError("distance-tilt inputs are malformed")
    present = set(categories.tolist())
    if present != set(category_masses) or any(category_masses[key] <= 0 for key in present):
        raise ValueError("every present QC category needs positive target mass")
    if not np.isclose(sum(category_masses.values()), 1.0, atol=1.0e-12):
        raise ValueError("conditional QC masses must sum to one")
    result = np.zeros(len(values), dtype=np.float64)
    for category in sorted(present):
        rows = np.flatnonzero(categories == category)
        local_distance = values[rows]
        span = float(np.max(local_distance) - np.min(local_distance))
        normalized_distance = (
            (local_distance - np.min(local_distance)) / span
            if span > np.finfo(np.float64).eps
            else np.zeros(len(rows), dtype=np.float64)
        )
        logits = -float(tilt) * normalized_distance
        logits -= np.max(logits)
        local = np.exp(np.clip(logits, -700.0, 0.0))
        local /= local.sum()
        result[rows] = float(category_masses[category]) * local
    if np.any(result <= 0) or not np.isclose(result.sum(), 1.0, atol=1.0e-12):
        raise RuntimeError("distance tilt produced invalid positive weights")
    return result


def _distance_tilt_interval(
    distances: Sequence[float],
    category_ids: Sequence[int],
    category_masses: Mapping[int, float],
) -> tuple[float, float]:
    maximum = _ess(_distance_tilt_weights(distances, category_ids, category_masses, 0.0))
    minimum = _ess(
        _distance_tilt_weights(
            distances,
            category_ids,
            category_masses,
            FROZEN_STATE_MAX_TILT,
        )
    )
    if minimum > maximum + 1.0e-9:
        raise RuntimeError("anchor-proximity tilt has a non-monotone ESS interval")
    return float(minimum), float(maximum)


def _distance_tilt_at_exact_ess(
    distances: Sequence[float],
    category_ids: Sequence[int],
    category_masses: Mapping[int, float],
    target_ess: float,
) -> tuple[np.ndarray, float]:
    """Solve a relabel-invariant proximity tilt at an exact feasible ESS."""

    minimum, maximum = _distance_tilt_interval(distances, category_ids, category_masses)
    target = float(target_ess)
    tolerance = 1.0e-9 * max(1.0, target)
    if target < minimum - tolerance or target > maximum + tolerance:
        raise ValueError(
            f"target component ESS {target:.12g} is outside the proximity-limited "
            f"interval [{minimum:.12g}, {maximum:.12g}]"
        )
    if target >= maximum - tolerance:
        weights = _distance_tilt_weights(distances, category_ids, category_masses, 0.0)
        return weights, 0.0
    low, high = 0.0, FROZEN_STATE_MAX_TILT
    for _ in range(100):
        middle = (low + high) / 2.0
        candidate = _distance_tilt_weights(distances, category_ids, category_masses, middle)
        if _ess(candidate) > target:
            low = middle
        else:
            high = middle
    tilt = (low + high) / 2.0
    weights = _distance_tilt_weights(distances, category_ids, category_masses, tilt)
    if not np.isclose(_ess(weights), target, rtol=1.0e-9, atol=1.0e-9):
        raise RuntimeError("proximity tilt did not attain exact component ESS")
    return weights, float(tilt)


def _compute_weight_plan(
    *,
    heldout_donor: str,
    wrong_donor_ids: Sequence[object],
    cell_ids: Sequence[object],
    bank_ids: Sequence[object],
    type_ids: Sequence[object],
    total_umi: Sequence[float],
    n_features: Sequence[float],
    percent_mt: Sequence[float],
    model_type_names: Sequence[object] | None = None,
) -> dict[str, np.ndarray]:
    """Create the target-free coarse-type and QC support plan.

    State strata and all final weights are deliberately deferred until the
    frozen training-only molecular encoder has been fitted.  No held-out ST is
    available in this stage.
    """

    cells = np.asarray(cell_ids).astype(str)
    banks = np.asarray(bank_ids).astype(str)
    types = np.asarray(type_ids).astype(str)
    depth = np.asarray(total_umi, dtype=np.float64)
    features = np.asarray(n_features, dtype=np.float64)
    mitochondrial = np.asarray(percent_mt, dtype=np.float64)
    wrong = tuple(sorted(np.asarray(wrong_donor_ids).astype(str).tolist()))
    expected_banks = (str(heldout_donor), *wrong)
    model_types = tuple(
        sorted(
            set(
                types.tolist()
                if model_type_names is None
                else np.asarray(model_type_names).astype(str).tolist()
            )
        )
    )
    if any(value.shape != cells.shape for value in (banks, types, depth, features, mitochondrial)):
        raise ValueError("reference QC arrays are not cell aligned")
    if len(set(cells.tolist())) != len(cells):
        raise ValueError("reference cell IDs are not unique within a fold")
    if set(banks.tolist()) != set(expected_banks):
        raise ValueError("reference banks do not match matched plus declared wrong donors")
    if not model_types or not set(types[np.isin(banks, np.asarray(wrong))]).issubset(model_types):
        raise ValueError("frozen model type vocabulary excludes a wrong-bank reference type")
    if np.any(~np.isfinite(depth)) or np.any(depth <= 0):
        raise ValueError("reference total UMI must be finite and positive")
    if np.any(~np.isfinite(features)) or np.any(features <= 0):
        raise ValueError("reference n_features must be finite and positive")
    if np.any(~np.isfinite(mitochondrial)) or np.any(mitochondrial < 0):
        raise ValueError("reference percent_mt must be finite and non-negative")

    all_types = tuple(sorted(set(types.tolist())))
    common_types = set.intersection(
        *[set(types[banks == bank].tolist()) for bank in expected_banks]
    )
    if not common_types:
        raise ValueError("matched/wrong banks have no common Level1 type support")
    thresholds: dict[str, tuple[float, float, float]] = {}
    strata = np.full(len(cells), -1, dtype=np.int8)
    for type_name in sorted(common_types):
        pooled = types == type_name
        threshold = (
            float(np.median(depth[pooled])),
            float(np.median(features[pooled])),
            float(np.median(mitochondrial[pooled])),
        )
        thresholds[type_name] = threshold
        strata[pooled] = (
            4 * (depth[pooled] >= threshold[0]).astype(np.int8)
            + 2 * (features[pooled] >= threshold[1]).astype(np.int8)
            + (mitochondrial[pooled] >= threshold[2]).astype(np.int8)
        )

    shared_strata: dict[str, tuple[int, ...]] = {}
    for type_name in sorted(common_types):
        intersection = set.intersection(
            *[
                set(strata[(banks == bank) & (types == type_name)].tolist())
                for bank in expected_banks
            ]
        )
        if intersection:
            shared_strata[type_name] = tuple(sorted(int(value) for value in intersection))
    retained_types = tuple(
        type_name
        for type_name in sorted(shared_strata)
        if all(
            np.count_nonzero(
                (banks == bank)
                & (types == type_name)
                & np.isin(strata, np.asarray(shared_strata[type_name], dtype=np.int8))
            )
            >= FROZEN_MIN_STATE_COMPONENTS_PER_TYPE * FROZEN_STATE_SUPPORT_CELLS
            for bank in expected_banks
        )
    )
    if not retained_types:
        raise ValueError(
            "no common type retains enough shared-QC cells for the frozen state components"
        )
    type_support_diagnostics = []
    for type_name in all_types:
        banks_present = [
            bank for bank in expected_banks if np.any((banks == bank) & (types == type_name))
        ]
        if type_name not in common_types:
            reason = "missing_from_one_or_more_compared_banks"
            status = "excluded_before_QC_support"
            shared = ()
        elif type_name not in shared_strata:
            reason = "no_joint_QC_stratum_present_in_every_compared_bank"
            status = "excluded_at_QC_support"
            shared = ()
        elif type_name not in retained_types:
            reason = "insufficient_shared_QC_cells_for_minimum_state_capacity"
            status = "excluded_at_state_capacity_screen"
            shared = shared_strata[type_name]
        else:
            reason = "eligible_for_training_anchor_and_all_bank_state_evaluation"
            status = "QC_plan_eligible"
            shared = shared_strata[type_name]
        type_support_diagnostics.append(
            {
                "Level1_type": type_name,
                "banks_present": banks_present,
                "status": status,
                "reason": reason,
                "shared_QC_strata": list(shared),
                "input_cells_by_bank": {
                    bank: int(np.count_nonzero((banks == bank) & (types == type_name)))
                    for bank in expected_banks
                },
                "QC_plan_retained_cells_by_bank": {
                    bank: int(
                        np.count_nonzero(
                            (banks == bank)
                            & (types == type_name)
                            & np.isin(strata, np.asarray(shared, dtype=np.int8))
                        )
                    )
                    for bank in expected_banks
                },
            }
        )

    category_types: list[str] = []
    category_strata: list[int] = []
    category_masses: list[float] = []
    for type_name in retained_types:
        local_strata = shared_strata[type_name]
        fractions = []
        for bank in expected_banks:
            local = (banks == bank) & (types == type_name)
            counts = np.asarray(
                [np.count_nonzero(local & (strata == stratum)) for stratum in local_strata],
                dtype=np.float64,
            )
            fractions.append(counts / counts.sum())
        geometric = np.exp(np.mean(np.log(np.vstack(fractions)), axis=0))
        geometric /= geometric.sum()
        for stratum, mass in zip(local_strata, geometric):
            category_types.append(type_name)
            category_strata.append(stratum)
            category_masses.append(float(mass / len(retained_types)))

    category_types_array = np.asarray(category_types)
    category_strata_array = np.asarray(category_strata, dtype=np.int8)
    target_mass_array = np.asarray(category_masses, dtype=np.float64)
    category_lookup = {
        (type_name, int(stratum)): index
        for index, (type_name, stratum) in enumerate(
            zip(category_types_array.tolist(), category_strata_array.tolist())
        )
    }
    category = np.full(len(cells), -1, dtype=np.int16)
    for (type_name, stratum), index in category_lookup.items():
        category[(types == type_name) & (strata == stratum)] = index
    retained = category >= 0
    target_masses = {index: float(value) for index, value in enumerate(target_mass_array)}

    maximum_ess = []
    for bank in expected_banks:
        rows = np.flatnonzero((banks == bank) & retained)
        q = sum(
            mass**2 / np.count_nonzero(category[rows] == index)
            for index, mass in target_masses.items()
        )
        maximum_ess.append(1.0 / q)
    threshold_matrix = np.asarray([thresholds[name] for name in retained_types], dtype=np.float64)
    return {
        "schema": np.asarray(PLAN_SCHEMA),
        "heldout_donor": np.asarray(str(heldout_donor)),
        "wrong_donor_ids": np.asarray(wrong),
        "bank_names": np.asarray(expected_banks),
        "cell_ids": cells,
        "bank_ids": banks,
        "type_ids": types,
        "model_type_names": np.asarray(model_types),
        "adapter_input_expected": np.isin(types, np.asarray(model_types)),
        "total_umi": depth,
        "n_features_rna": features,
        "percent_mt": mitochondrial,
        "joint_stratum": strata,
        "retained": retained,
        "category_index": category,
        "common_type_names": np.asarray(retained_types),
        "all_observed_type_names": np.asarray(all_types),
        "all_bank_pre_QC_common_type_names": np.asarray(sorted(common_types)),
        "type_support_diagnostics_json": np.asarray(
            json.dumps(type_support_diagnostics, sort_keys=True, separators=(",", ":"))
        ),
        "within_type_thresholds": threshold_matrix,
        "threshold_names": np.asarray(["total_umi", "n_features_rna", "percent_mt"]),
        "category_type_ids": category_types_array,
        "category_joint_strata": category_strata_array,
        "category_target_masses": target_mass_array,
        "qc_only_maximum_ess_by_bank": np.asarray(maximum_ess, dtype=np.float64),
        "state_max_components_per_type": np.asarray(
            FROZEN_MAX_STATE_COMPONENTS_PER_TYPE, dtype=np.int64
        ),
        "state_min_components_per_type": np.asarray(
            FROZEN_MIN_STATE_COMPONENTS_PER_TYPE, dtype=np.int64
        ),
        "state_balance_method": np.asarray(STATE_BALANCE_METHOD),
        "state_representation": np.asarray(
            "frozen_training_only_fitted_20D_encoder_with_query_excluded_training_anchor_fit"
        ),
        "state_minimum_natural_component_ess": np.asarray(
            FROZEN_STATE_SUPPORT_ESS, dtype=np.float64
        ),
        "state_minimum_natural_component_cells": np.asarray(
            FROZEN_STATE_SUPPORT_CELLS, dtype=np.int64
        ),
        "state_maximum_proximity_ratio": np.asarray(
            FROZEN_STATE_MAX_PROXIMITY_RATIO, dtype=np.float64
        ),
        "state_minimum_anchor_separation_ratio": np.asarray(
            FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO, dtype=np.float64
        ),
        "state_minimum_global_cell_coverage": np.asarray(
            FROZEN_STATE_MIN_GLOBAL_CELL_COVERAGE, dtype=np.float64
        ),
        "state_minimum_type_coverage": np.asarray(FROZEN_STATE_MIN_TYPE_COVERAGE, dtype=np.float64),
        "hard_subsampling_used": np.asarray(False),
        "support_restriction_used": np.asarray(np.any(~retained)),
    }


def _validate_weight_plan(plan: Mapping[str, np.ndarray], *, donor: str) -> None:
    """Recompute every scientific weighting invariant from a loaded plan."""

    if _scalar_text(plan.get("schema")) != PLAN_SCHEMA:
        raise ValueError(f"fixed-reference plan schema differs for {donor}")
    if _scalar_text(plan.get("heldout_donor")) != donor:
        raise ValueError(f"fixed-reference plan donor differs for {donor}")
    wrong = tuple(sorted(np.asarray(plan.get("wrong_donor_ids", ())).astype(str).tolist()))
    banks_expected = (donor, *wrong)
    bank_names = tuple(np.asarray(plan.get("bank_names", ())).astype(str).tolist())
    if not wrong or len(set(wrong)) != len(wrong) or bank_names != banks_expected:
        raise ValueError(f"fixed-reference plan bank set is malformed for {donor}")

    cells = np.asarray(plan.get("cell_ids", ())).astype(str)
    banks = np.asarray(plan.get("bank_ids", ())).astype(str)
    types = np.asarray(plan.get("type_ids", ())).astype(str)
    depth = np.asarray(plan.get("total_umi", ()), dtype=np.float64)
    feature_count = np.asarray(plan.get("n_features_rna", ()), dtype=np.float64)
    mitochondrial = np.asarray(plan.get("percent_mt", ()), dtype=np.float64)
    category = np.asarray(plan.get("category_index", ()), dtype=np.int64)
    retained = np.asarray(plan.get("retained", ()), dtype=bool)
    adapter_input = np.asarray(plan.get("adapter_input_expected", ()), dtype=bool)
    joint = np.asarray(plan.get("joint_stratum", ()), dtype=np.int64)
    vectors = (
        banks,
        types,
        depth,
        feature_count,
        mitochondrial,
        category,
        retained,
        adapter_input,
        joint,
    )
    if not len(cells) or any(value.shape != cells.shape for value in vectors):
        raise ValueError(f"fixed-reference plan rows are malformed for {donor}")
    if (
        len(set(cells.tolist())) != len(cells)
        or set(banks.tolist()) != set(banks_expected)
        or np.any(~np.isfinite(depth))
        or np.any(depth <= 0)
        or np.any(~np.isfinite(feature_count))
        or np.any(feature_count <= 0)
        or np.any(~np.isfinite(mitochondrial))
        or np.any(mitochondrial < 0)
    ):
        raise ValueError(f"fixed-reference plan cell/bank identities are malformed for {donor}")
    if not np.array_equal(retained, category >= 0):
        raise ValueError(f"fixed-reference retained support changed for {donor}")
    model_types = tuple(np.asarray(plan.get("model_type_names", ())).astype(str).tolist())
    if (
        not model_types
        or model_types != tuple(sorted(set(model_types)))
        or not np.array_equal(adapter_input, np.isin(types, np.asarray(model_types)))
        or np.any(~adapter_input[np.isin(banks, np.asarray(wrong))])
    ):
        raise ValueError(f"fixed-reference model-vocabulary input mask changed for {donor}")
    if bool(np.asarray(plan.get("hard_subsampling_used", True))):
        raise ValueError(
            f"fixed-reference support plan unexpectedly used hard subsampling for {donor}"
        )
    if bool(np.asarray(plan.get("support_restriction_used", False))) != bool(np.any(~retained)):
        raise ValueError(f"fixed-reference support-restriction receipt changed for {donor}")

    category_types = np.asarray(plan.get("category_type_ids", ())).astype(str)
    category_strata = np.asarray(plan.get("category_joint_strata", ()), dtype=np.int64)
    masses = np.asarray(plan.get("category_target_masses", ()), dtype=np.float64)
    category_count = len(masses)
    if (
        not category_count
        or category_types.shape != masses.shape
        or category_strata.shape != masses.shape
        or np.any(~np.isfinite(masses))
        or np.any(masses <= 0)
        or not np.isclose(masses.sum(), 1.0, atol=1.0e-12)
        or set(category[retained].tolist()) != set(range(category_count))
    ):
        raise ValueError(f"fixed-reference category masses are malformed for {donor}")
    common_types = tuple(np.asarray(plan.get("common_type_names", ())).astype(str).tolist())
    if common_types != tuple(sorted(set(category_types.tolist()))):
        raise ValueError(f"fixed-reference common type support changed for {donor}")
    all_types = tuple(sorted(set(types.tolist())))
    all_common = set.intersection(*[set(types[banks == bank].tolist()) for bank in banks_expected])
    if tuple(np.asarray(plan.get("all_observed_type_names", ())).astype(str).tolist()) != all_types:
        raise ValueError(f"fixed-reference original type universe changed for {donor}")
    if tuple(
        np.asarray(plan.get("all_bank_pre_QC_common_type_names", ())).astype(str).tolist()
    ) != tuple(sorted(all_common)):
        raise ValueError(f"fixed-reference pre-QC common type universe changed for {donor}")
    expected_common: list[str] = []
    shared_by_type: dict[str, tuple[int, ...]] = {}
    for type_name in sorted(all_common):
        shared = set.intersection(
            *[
                set(joint[(banks == bank) & (types == type_name)].tolist())
                for bank in banks_expected
            ]
        )
        if shared:
            shared_by_type[type_name] = tuple(sorted(int(value) for value in shared))
        if shared and all(
            np.count_nonzero(
                (banks == bank)
                & (types == type_name)
                & np.isin(joint, np.asarray(sorted(shared), dtype=np.int64))
            )
            >= FROZEN_MIN_STATE_COMPONENTS_PER_TYPE * FROZEN_STATE_SUPPORT_CELLS
            for bank in banks_expected
        ):
            expected_common.append(type_name)
    if common_types != tuple(expected_common):
        raise ValueError(f"fixed-reference state-supported type set changed for {donor}")
    expected_type_support = []
    for type_name in all_types:
        banks_present = [
            bank for bank in banks_expected if np.any((banks == bank) & (types == type_name))
        ]
        if type_name not in all_common:
            reason = "missing_from_one_or_more_compared_banks"
            status = "excluded_before_QC_support"
            shared = ()
        elif type_name not in shared_by_type:
            reason = "no_joint_QC_stratum_present_in_every_compared_bank"
            status = "excluded_at_QC_support"
            shared = ()
        elif type_name not in expected_common:
            reason = "insufficient_shared_QC_cells_for_minimum_state_capacity"
            status = "excluded_at_state_capacity_screen"
            shared = shared_by_type[type_name]
        else:
            reason = "eligible_for_training_anchor_and_all_bank_state_evaluation"
            status = "QC_plan_eligible"
            shared = shared_by_type[type_name]
        expected_type_support.append(
            {
                "Level1_type": type_name,
                "banks_present": banks_present,
                "status": status,
                "reason": reason,
                "shared_QC_strata": list(shared),
                "input_cells_by_bank": {
                    bank: int(np.count_nonzero((banks == bank) & (types == type_name)))
                    for bank in banks_expected
                },
                "QC_plan_retained_cells_by_bank": {
                    bank: int(
                        np.count_nonzero(
                            (banks == bank)
                            & (types == type_name)
                            & np.isin(joint, np.asarray(shared, dtype=np.int64))
                        )
                    )
                    for bank in banks_expected
                },
            }
        )
    try:
        observed_type_support = json.loads(_scalar_text(plan.get("type_support_diagnostics_json")))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"fixed-reference type-support receipt is malformed for {donor}"
        ) from error
    if observed_type_support != expected_type_support:
        raise ValueError(f"fixed-reference global type-support receipt changed for {donor}")
    expected_type_mass = 1.0 / len(common_types)
    if any(
        not np.isclose(masses[category_types == type_name].sum(), expected_type_mass)
        for type_name in common_types
    ):
        raise ValueError(f"fixed-reference uniform type mass changed for {donor}")
    thresholds = np.asarray(plan.get("within_type_thresholds", ()), dtype=np.float64)
    threshold_names = tuple(np.asarray(plan.get("threshold_names", ())).astype(str).tolist())
    if (
        thresholds.shape != (len(common_types), 3)
        or not np.isfinite(thresholds).all()
        or threshold_names != ("total_umi", "n_features_rna", "percent_mt")
        or np.any((joint[retained] < 0) | (joint[retained] > 7))
    ):
        raise ValueError(f"fixed-reference QC strata are malformed for {donor}")
    for type_index, type_name in enumerate(common_types):
        rows = types == type_name
        observed_threshold = np.asarray(
            [np.median(depth[rows]), np.median(feature_count[rows]), np.median(mitochondrial[rows])]
        )
        expected_joint = (
            4 * (depth[rows] >= observed_threshold[0]).astype(np.int64)
            + 2 * (feature_count[rows] >= observed_threshold[1]).astype(np.int64)
            + (mitochondrial[rows] >= observed_threshold[2]).astype(np.int64)
        )
        if not np.array_equal(thresholds[type_index], observed_threshold) or not np.array_equal(
            joint[rows], expected_joint
        ):
            raise ValueError(f"fixed-reference QC thresholds changed for {donor}/{type_name}")
        local_categories = np.flatnonzero(category_types == type_name)
        local_strata = category_strata[local_categories]
        fractions = []
        for bank in banks_expected:
            local = (banks == bank) & (types == type_name)
            counts = np.asarray(
                [np.count_nonzero(local & (joint == value)) for value in local_strata],
                dtype=np.float64,
            )
            if np.any(counts == 0):
                raise ValueError(
                    f"fixed-reference shared QC support changed for {bank}/{type_name}"
                )
            fractions.append(counts / counts.sum())
        geometric = np.exp(np.mean(np.log(np.vstack(fractions)), axis=0))
        geometric /= geometric.sum()
        expected_masses = geometric / len(common_types)
        if not np.allclose(masses[local_categories], expected_masses, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError(f"fixed-reference QC target mass changed for {donor}/{type_name}")
    lookup = {
        (type_name, int(stratum)): index
        for index, (type_name, stratum) in enumerate(
            zip(category_types.tolist(), category_strata.tolist())
        )
    }
    expected_category = np.full(len(cells), -1, dtype=np.int64)
    for key, index in lookup.items():
        expected_category[(types == key[0]) & (joint == key[1])] = index
    if not np.array_equal(category, expected_category):
        raise ValueError(f"fixed-reference QC/category mapping changed for {donor}")

    maximum_components = int(np.asarray(plan.get("state_max_components_per_type", -1)))
    minimum_components = int(np.asarray(plan.get("state_min_components_per_type", -1)))
    if (
        maximum_components != FROZEN_MAX_STATE_COMPONENTS_PER_TYPE
        or minimum_components != FROZEN_MIN_STATE_COMPONENTS_PER_TYPE
        or _scalar_text(plan.get("state_balance_method")) != STATE_BALANCE_METHOD
        or _scalar_text(plan.get("state_representation"))
        != "frozen_training_only_fitted_20D_encoder_with_query_excluded_training_anchor_fit"
        or not np.isclose(
            float(np.asarray(plan.get("state_minimum_natural_component_ess", np.nan))),
            FROZEN_STATE_SUPPORT_ESS,
            atol=1.0e-12,
        )
        or int(np.asarray(plan.get("state_minimum_natural_component_cells", -1)))
        != FROZEN_STATE_SUPPORT_CELLS
        or not np.isclose(
            float(np.asarray(plan.get("state_maximum_proximity_ratio", np.nan))),
            FROZEN_STATE_MAX_PROXIMITY_RATIO,
        )
        or not np.isclose(
            float(np.asarray(plan.get("state_minimum_anchor_separation_ratio", np.nan))),
            FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO,
        )
        or not np.isclose(
            float(np.asarray(plan.get("state_minimum_global_cell_coverage", np.nan))),
            FROZEN_STATE_MIN_GLOBAL_CELL_COVERAGE,
        )
        or not np.isclose(
            float(np.asarray(plan.get("state_minimum_type_coverage", np.nan))),
            FROZEN_STATE_MIN_TYPE_COVERAGE,
        )
    ):
        raise ValueError(f"fixed-reference state-diversity contract changed for {donor}")
    maximum = np.asarray(plan.get("qc_only_maximum_ess_by_bank", ()), dtype=np.float64)
    if maximum.shape != (len(banks_expected),) or not np.isfinite(maximum).all():
        raise ValueError(f"fixed-reference QC-only ESS receipt is malformed for {donor}")
    for bank_index, bank in enumerate(banks_expected):
        rows = np.flatnonzero((banks == bank) & retained)
        if any(np.count_nonzero(category[rows] == index) == 0 for index in range(category_count)):
            raise ValueError(f"fixed-reference category support differs for bank {bank}")
        observed_maximum = 1.0 / sum(
            masses[index] ** 2 / np.count_nonzero(category[rows] == index)
            for index in range(category_count)
        )
        if not np.isclose(maximum[bank_index], observed_maximum, rtol=1.0e-9):
            raise ValueError(f"fixed-reference QC-only ESS changed for {bank}")


def _stable_seed(identifier: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{identifier}".encode()).digest()[:8], "little")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = float(probability) * float(cumulative[-1])
    return float(
        ordered_values[min(np.searchsorted(cumulative, cutoff, side="left"), len(values) - 1)]
    )


def _array_digest(*arrays: object) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(_json_bytes({"dtype": array.dtype.str, "shape": array.shape}))
        if array.dtype.kind in {"U", "S", "O"}:
            digest.update(_json_bytes(array.astype(str).tolist()))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class _AlignedLatentCapture:
    """Read-only hook for the frozen encoder's sealed five-call sequence."""

    def __init__(self, public: Mapping[str, np.ndarray], plan: Mapping[str, np.ndarray]) -> None:
        self.public = public
        self.plan = plan
        self.calls: list[dict[str, object]] = []
        self.train_sc_latent: np.ndarray | None = None
        self.matched_sc_latent: np.ndarray | None = None

    def wrap(self, original: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        expected = (
            ("st", len(self.public["train_st_counts"]), "aligned_train_st"),
            ("scrna", len(self.public["train_sc_counts"]), "aligned_train_sc"),
            ("scrna", len(self.public["matched_sc_counts"]), "aligned_matched_sc"),
            ("st", len(self.public["train_st_counts"]), "unaligned_train_st"),
            ("scrna", len(self.public["train_sc_counts"]), "unaligned_train_sc"),
        )

        def captured(
            module: object,
            counts: np.ndarray,
            *,
            modality: str,
            device: str,
        ) -> np.ndarray:
            index = len(self.calls)
            if index >= len(expected):
                raise RuntimeError("frozen molecular encoder call sequence has extra calls")
            expected_modality, expected_rows, role = expected[index]
            if modality != expected_modality or len(counts) != expected_rows:
                raise RuntimeError(
                    f"frozen molecular encoder call {index} differs from the {role} contract"
                )
            raw_result = original(module, counts, modality=modality, device=device)
            result = np.asarray(raw_result, dtype=np.float64)
            if result.shape != (expected_rows, FROZEN_LATENT_DIM) or not np.isfinite(result).all():
                raise RuntimeError(f"frozen molecular encoder produced malformed {role} latent")
            self.calls.append({"ordinal": index, "role": role, "rows": expected_rows})
            if role == "aligned_train_sc":
                self.train_sc_latent = result.copy()
            elif role == "aligned_matched_sc":
                self.matched_sc_latent = result.copy()
            return raw_result

        return captured

    def plan_latent(self) -> np.ndarray:
        if len(self.calls) != 5 or self.train_sc_latent is None or self.matched_sc_latent is None:
            raise RuntimeError("frozen aligned molecular latent capture is incomplete")
        wrong_index = np.asarray(self.public["wrong_train_sc_index"], dtype=np.int64)
        cells = np.concatenate(
            (
                np.asarray(self.public["matched_sc_cell_ids"]).astype(str),
                np.asarray(self.public["train_sc_cell_ids"]).astype(str)[wrong_index],
            )
        )
        latent = np.concatenate((self.matched_sc_latent, self.train_sc_latent[wrong_index]), axis=0)
        plan_cells = np.asarray(self.plan["cell_ids"]).astype(str)
        if not np.array_equal(cells, plan_cells) or latent.shape != (
            len(plan_cells),
            FROZEN_LATENT_DIM,
        ):
            raise RuntimeError("captured aligned reference latent differs from the frozen plan")
        return latent

    def receipt(self) -> dict[str, object]:
        latent = self.plan_latent()
        return {
            "contract": "frozen_encode_sequence_v1",
            "calls": self.calls,
            "reference_latent_sha256": _array_digest(
                np.asarray(self.plan["cell_ids"]).astype(str), latent
            ),
            "score_target_opened": False,
        }


def _coordinate_key(value: np.ndarray) -> tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(value, dtype=np.float64).tolist())


def _canonical_weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    order = sorted(range(len(matrix)), key=lambda index: _coordinate_key(matrix[index]))
    return np.average(matrix[order], axis=0, weights=mass[order])


def _canonical_weighted_kmeans(
    values: np.ndarray,
    weights: np.ndarray,
    components: int,
    *,
    iterations: int,
) -> np.ndarray:
    """Geometry-only deterministic k-means, invariant to row order and IDs."""

    matrix = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    count = int(components)
    if (
        matrix.ndim != 2
        or mass.shape != (len(matrix),)
        or count < FROZEN_MIN_STATE_COMPONENTS_PER_TYPE
        or count > FROZEN_MAX_STATE_COMPONENTS_PER_TYPE
        or len(matrix) < count
        or np.any(mass <= 0)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("training-anchor k-means inputs are malformed")
    order = sorted(range(len(matrix)), key=lambda index: _coordinate_key(matrix[index]))
    matrix = matrix[order]
    mass = mass[order]
    center = _canonical_weighted_average(matrix, mass)
    distance = np.sum((matrix - center) ** 2, axis=1)
    minimum = float(np.min(distance))
    candidates = np.flatnonzero(np.isclose(distance, minimum, rtol=0.0, atol=1.0e-14))
    first = min(candidates.tolist(), key=lambda index: _coordinate_key(matrix[index]))
    centers = [matrix[first].copy()]
    while len(centers) < count:
        distance = np.min(
            np.sum((matrix[:, None] - np.vstack(centers)[None]) ** 2, axis=2),
            axis=1,
        )
        maximum = float(np.max(distance))
        if maximum <= np.finfo(np.float64).eps:
            raise RuntimeError("query-excluded training anchors collapse")
        candidates = np.flatnonzero(np.isclose(distance, maximum, rtol=1.0e-12, atol=1.0e-14))
        chosen = min(candidates.tolist(), key=lambda index: _coordinate_key(matrix[index]))
        centers.append(matrix[chosen].copy())
    means = np.vstack(centers)
    for _ in range(int(iterations)):
        distance = np.sum((matrix[:, None] - means[None]) ** 2, axis=2)
        assignment = np.argmin(distance, axis=1)
        if set(assignment.tolist()) != set(range(count)):
            raise RuntimeError("query-excluded training anchor loses natural support")
        updated = np.vstack(
            [
                _canonical_weighted_average(
                    matrix[assignment == component],
                    mass[assignment == component],
                )
                for component in range(count)
            ]
        )
        order = sorted(range(count), key=lambda index: _coordinate_key(updated[index]))
        updated = updated[order]
        if np.allclose(updated, means, rtol=0.0, atol=1.0e-12):
            means = updated
            break
        means = updated
    return means


def _geometric_barycenter(fractions: Sequence[np.ndarray]) -> np.ndarray:
    values = np.vstack([np.asarray(value, dtype=np.float64) for value in fractions])
    if values.ndim != 2 or np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("geometric barycenter requires positive finite fractions")
    result = np.exp(np.mean(np.log(values), axis=0))
    result /= result.sum()
    return result


def _natural_state_group(
    rows: np.ndarray,
    distance_ratio: np.ndarray,
) -> dict[str, float | int]:
    if not len(rows):
        return {
            "cells": 0,
            "ESS": 0.0,
            "proximity_quantile_ratio": math.inf,
        }
    local = np.ones(len(rows), dtype=np.float64)
    return {
        "cells": int(len(rows)),
        "ESS": float(_ess(local)),
        "proximity_quantile_ratio": float(
            _weighted_quantile(
                np.asarray(distance_ratio[rows], dtype=np.float64),
                local,
                FROZEN_STATE_PROXIMITY_QUANTILE,
            )
        ),
    }


def _state_group_is_supported(group: Mapping[str, object]) -> bool:
    return bool(
        int(group["cells"]) >= FROZEN_STATE_SUPPORT_CELLS
        and float(group["ESS"]) >= FROZEN_STATE_SUPPORT_ESS - 1.0e-9
        and float(group["proximity_quantile_ratio"]) <= FROZEN_STATE_MAX_PROXIMITY_RATIO + 1.0e-12
    )


def _try_type_state_contract(
    *,
    type_name: str,
    component_count: int,
    values: np.ndarray,
    centered: np.ndarray,
    banks: np.ndarray,
    types: np.ndarray,
    category: np.ndarray,
    retained: np.ndarray,
    anchor_fit_weights: np.ndarray,
    bank_names: tuple[str, ...],
    wrong_banks: tuple[str, ...],
    iterations: int,
    failure_reasons: list[str],
) -> dict[str, object] | None:
    """Build one naturally supported training-anchor contract or return None."""

    def ineligible(reason: str) -> None:
        failure_reasons.append(reason)
        return None

    training_rows = np.flatnonzero((types == type_name) & np.isin(banks, np.asarray(wrong_banks)))
    training_weights = np.zeros(len(training_rows), dtype=np.float64)
    for bank in wrong_banks:
        local = np.flatnonzero(banks[training_rows] == bank)
        if not len(local):
            return ineligible("training_donor_type_missing")
        bank_weights = anchor_fit_weights[training_rows[local]]
        training_weights[local] = bank_weights / bank_weights.sum() / len(wrong_banks)
    try:
        anchors = _canonical_weighted_kmeans(
            centered[training_rows],
            training_weights,
            component_count,
            iterations=iterations,
        )
    except (ValueError, RuntimeError):
        return ineligible("training_only_anchor_fit_failed")
    pooled_center = _canonical_weighted_average(centered[training_rows], training_weights)
    squared_radius = np.sum((centered[training_rows] - pooled_center) ** 2, axis=1)
    scale = max(_weighted_quantile(squared_radius, training_weights, 0.5), 1.0e-8)
    pairwise = np.sum((anchors[:, None] - anchors[None]) ** 2, axis=2)
    separation = float(np.min(pairwise[np.triu_indices(component_count, 1)]) / scale)
    if separation < FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO:
        return ineligible("training_anchor_separation_below_floor")

    type_rows = np.flatnonzero(retained & (types == type_name))
    distance = np.sum(
        (centered[type_rows, None] - anchors[None]) ** 2,
        axis=2,
    )
    ordered = np.sort(distance, axis=1)
    if np.any(np.isclose(ordered[:, 0], ordered[:, 1], rtol=0.0, atol=1.0e-12 * scale)):
        return ineligible("nearest_anchor_assignment_tie")
    assignment = np.argmin(distance, axis=1)
    assigned_ratio = np.min(distance, axis=1) / scale
    full_assignment = np.full(len(values), -1, dtype=np.int16)
    full_distance_ratio = np.full(len(values), np.nan, dtype=np.float64)
    full_assignment[type_rows] = assignment
    full_distance_ratio[type_rows] = assigned_ratio

    natural: list[dict[str, object]] = []
    for bank in bank_names:
        local_means = []
        bank_groups = []
        for component in range(component_count):
            rows = np.flatnonzero(
                retained & (types == type_name) & (banks == bank) & (full_assignment == component)
            )
            group = _natural_state_group(rows, full_distance_ratio)
            bank_groups.append(
                {
                    "bank": bank,
                    "component": component,
                    "stage": "pre_balance",
                    **group,
                }
            )
            if not _state_group_is_supported(group):
                return ineligible(f"natural_support_failed::{bank}::{component}")
            local_means.append(
                _canonical_weighted_average(
                    values[rows],
                    np.ones(len(rows), dtype=np.float64),
                )
            )
        local_means_array = np.vstack(local_means)
        local_pairwise = np.sum(
            (local_means_array[:, None] - local_means_array[None]) ** 2,
            axis=2,
        )
        bank_separation = float(np.min(local_pairwise[np.triu_indices(component_count, 1)]) / scale)
        if bank_separation < FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO:
            return ineligible(f"bank_component_separation_failed::{bank}")
        for group in bank_groups:
            group["bank_component_separation_ratio"] = bank_separation
        natural.extend(bank_groups)

    joint_keep = np.zeros(len(values), dtype=bool)
    component_qc: dict[int, tuple[int, ...]] = {}
    for component in range(component_count):
        shared = set.intersection(
            *[
                set(
                    category[
                        retained
                        & (types == type_name)
                        & (banks == bank)
                        & (full_assignment == component)
                    ].tolist()
                )
                for bank in bank_names
            ]
        )
        shared.discard(-1)
        if not shared:
            return ineligible(f"joint_QC_support_missing::{component}")
        component_qc[component] = tuple(sorted(int(value) for value in shared))
        joint_keep |= (
            retained
            & (types == type_name)
            & (full_assignment == component)
            & np.isin(category, np.asarray(component_qc[component], dtype=np.int64))
        )

    post_support: list[dict[str, object]] = []
    for bank in bank_names:
        type_bank = retained & (types == type_name) & (banks == bank)
        retention = float(np.count_nonzero(joint_keep & type_bank) / np.count_nonzero(type_bank))
        if retention < FROZEN_STATE_MIN_JOINT_RETENTION:
            return ineligible(f"joint_support_retention_failed::{bank}")
        for component in range(component_count):
            rows = np.flatnonzero(
                joint_keep & (types == type_name) & (banks == bank) & (full_assignment == component)
            )
            group = _natural_state_group(rows, full_distance_ratio)
            post_support.append(
                {
                    "bank": bank,
                    "component": component,
                    "stage": "joint_QC_state_support",
                    "type_retention_fraction": retention,
                    **group,
                }
            )
            if not _state_group_is_supported(group):
                return ineligible(f"post_joint_support_failed::{bank}::{component}")

    state_fractions = []
    for bank in bank_names:
        local = joint_keep & (types == type_name) & (banks == bank)
        component_mass = np.asarray(
            [
                np.count_nonzero(local & (full_assignment == component))
                for component in range(component_count)
            ],
            dtype=np.float64,
        )
        state_fractions.append(component_mass / component_mass.sum())
    state_target = _geometric_barycenter(state_fractions)

    component_contracts: list[dict[str, object]] = []
    for component in range(component_count):
        qc_values = component_qc[component]
        qc_fractions = []
        for bank in bank_names:
            rows = joint_keep & (types == type_name) & (banks == bank)
            local = rows & (full_assignment == component)
            fraction = np.asarray(
                [np.count_nonzero(local & (category == value)) for value in qc_values],
                dtype=np.float64,
            )
            qc_fractions.append(fraction / fraction.sum())
        qc_target = _geometric_barycenter(qc_fractions)
        qc_masses = {value: float(qc_target[index]) for index, value in enumerate(qc_values)}
        intervals: dict[str, tuple[float, float]] = {}
        for bank in bank_names:
            rows = np.flatnonzero(
                joint_keep & (types == type_name) & (banks == bank) & (full_assignment == component)
            )
            clipped = np.minimum(
                full_distance_ratio[rows],
                FROZEN_STATE_MAX_PROXIMITY_RATIO,
            )
            intervals[bank] = _distance_tilt_interval(clipped, category[rows], qc_masses)
        lower = max(
            FROZEN_STATE_SUPPORT_ESS,
            max(value[0] for value in intervals.values()),
            len(wrong_banks)
            * max(
                FROZEN_STATE_SUPPORT_ESS,
                max(intervals[bank][0] for bank in wrong_banks),
            ),
        )
        upper = min(value[1] for value in intervals.values())
        if lower > upper + 1.0e-9:
            return ineligible(f"component_ESS_interval_empty::{component}")
        target_ess = float(upper)
        single_conditional: dict[str, np.ndarray] = {}
        generic_conditional: dict[str, np.ndarray] = {}
        weight_diagnostics: list[dict[str, object]] = []
        for bank in bank_names:
            rows = np.flatnonzero(
                joint_keep & (types == type_name) & (banks == bank) & (full_assignment == component)
            )
            clipped = np.minimum(
                full_distance_ratio[rows],
                FROZEN_STATE_MAX_PROXIMITY_RATIO,
            )
            local, tilt = _distance_tilt_at_exact_ess(
                clipped,
                category[rows],
                qc_masses,
                target_ess,
            )
            relative = float(local.min() / local.max())
            if relative < FROZEN_STATE_MIN_RELATIVE_WEIGHT:
                return ineligible(f"single_weight_concentration_failed::{bank}::{component}")
            maximum_entropy = _distance_tilt_weights(clipped, category[rows], qc_masses, 0.0)
            natural_mean = _canonical_weighted_average(values[rows], maximum_entropy)
            weighted_mean = _canonical_weighted_average(values[rows], local)
            mean_shift = float(np.sum((weighted_mean - natural_mean) ** 2) / scale)
            natural_variance = float(
                np.sum(maximum_entropy * np.sum((values[rows] - natural_mean) ** 2, axis=1))
            )
            weighted_variance = float(
                np.sum(local * np.sum((values[rows] - weighted_mean) ** 2, axis=1))
            )
            variance_ratio = (
                weighted_variance / natural_variance
                if natural_variance > np.finfo(np.float64).eps
                else 1.0
            )
            if mean_shift > FROZEN_STATE_MAX_WEIGHTED_MEAN_SHIFT_RATIO:
                return ineligible(f"single_mean_shift_failed::{bank}::{component}")
            full = np.zeros(len(values), dtype=np.float64)
            full[rows] = local
            single_conditional[bank] = full
            weight_diagnostics.append(
                {
                    "bank": bank,
                    "mode": "single",
                    "tilt": tilt,
                    "achieved_component_ESS": float(_ess(local)),
                    "minimum_relative_weight": relative,
                    "maximum_conditional_weight_share": float(local.max()),
                    "effective_cell_fraction": float(_ess(local) / len(rows)),
                    "weighted_mean_shift_ratio": mean_shift,
                    "weighted_to_maximum_entropy_variance_trace_ratio": variance_ratio,
                }
            )
        generic_target = target_ess / len(wrong_banks)
        for bank in wrong_banks:
            rows = np.flatnonzero(
                joint_keep & (types == type_name) & (banks == bank) & (full_assignment == component)
            )
            clipped = np.minimum(
                full_distance_ratio[rows],
                FROZEN_STATE_MAX_PROXIMITY_RATIO,
            )
            local, tilt = _distance_tilt_at_exact_ess(
                clipped,
                category[rows],
                qc_masses,
                generic_target,
            )
            relative = float(local.min() / local.max())
            if relative < FROZEN_STATE_MIN_RELATIVE_WEIGHT:
                return ineligible(f"generic_weight_concentration_failed::{bank}::{component}")
            maximum_entropy = _distance_tilt_weights(clipped, category[rows], qc_masses, 0.0)
            natural_mean = _canonical_weighted_average(values[rows], maximum_entropy)
            weighted_mean = _canonical_weighted_average(values[rows], local)
            mean_shift = float(np.sum((weighted_mean - natural_mean) ** 2) / scale)
            natural_variance = float(
                np.sum(maximum_entropy * np.sum((values[rows] - natural_mean) ** 2, axis=1))
            )
            weighted_variance = float(
                np.sum(local * np.sum((values[rows] - weighted_mean) ** 2, axis=1))
            )
            variance_ratio = (
                weighted_variance / natural_variance
                if natural_variance > np.finfo(np.float64).eps
                else 1.0
            )
            if mean_shift > FROZEN_STATE_MAX_WEIGHTED_MEAN_SHIFT_RATIO:
                return ineligible(f"generic_mean_shift_failed::{bank}::{component}")
            full = np.zeros(len(values), dtype=np.float64)
            full[rows] = local
            generic_conditional[bank] = full
            weight_diagnostics.append(
                {
                    "bank": bank,
                    "mode": "generic_contribution",
                    "tilt": tilt,
                    "achieved_component_ESS": float(_ess(local)),
                    "minimum_relative_weight": relative,
                    "maximum_conditional_weight_share": float(local.max()),
                    "effective_cell_fraction": float(_ess(local) / len(rows)),
                    "weighted_mean_shift_ratio": mean_shift,
                    "weighted_to_maximum_entropy_variance_trace_ratio": variance_ratio,
                }
            )
        component_contracts.append(
            {
                "component": component,
                "anchor": anchors[component],
                "scale": scale,
                "state_mass_within_type": float(state_target[component]),
                "qc_strata": qc_values,
                "qc_target": qc_target,
                "target_ess": target_ess,
                "single_conditional": single_conditional,
                "generic_conditional": generic_conditional,
                "weight_diagnostics": weight_diagnostics,
            }
        )
    return {
        "type_name": type_name,
        "component_count": component_count,
        "anchors": anchors,
        "scale": scale,
        "anchor_separation_ratio": separation,
        "assignment": full_assignment,
        "distance_ratio": full_distance_ratio,
        "joint_keep": joint_keep,
        "natural_support": natural,
        "post_support": post_support,
        "components": component_contracts,
    }


def _fit_state_balance_authority(
    plan: Mapping[str, np.ndarray],
    latent: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, object]:
    """Fit query-excluded anchors, then construct exact joint state/QC weights."""

    donor = _scalar_text(plan["heldout_donor"])
    _validate_weight_plan(plan, donor=donor)
    values = np.asarray(latent, dtype=np.float64)
    cells = np.asarray(plan["cell_ids"]).astype(str)
    banks = np.asarray(plan["bank_ids"]).astype(str)
    types = np.asarray(plan["type_ids"]).astype(str)
    retained = np.asarray(plan["retained"], dtype=bool)
    bank_names = tuple(np.asarray(plan["bank_names"]).astype(str).tolist())
    wrong_banks = tuple(np.asarray(plan["wrong_donor_ids"]).astype(str).tolist())
    if (
        values.shape != (len(cells), FROZEN_LATENT_DIM)
        or not np.isfinite(values).all()
        or bank_names != (donor, *wrong_banks)
        or donor in wrong_banks
    ):
        raise ValueError("target-free state-authority inputs are malformed")
    anchor_fit_weights = np.zeros(len(values), dtype=np.float64)
    centered = np.zeros_like(values)
    for type_name in np.asarray(plan["common_type_names"]).astype(str):
        for bank in bank_names:
            rows = np.flatnonzero((types == type_name) & (banks == bank))
            if not len(rows):
                raise RuntimeError("state-anchor transform lacks a bank/type")
            anchor_fit_weights[rows] = 1.0 / len(rows)
            centered[rows] = values[rows] - _canonical_weighted_average(
                values[rows], anchor_fit_weights[rows]
            )

    contracts: list[dict[str, object]] = []
    unevaluable: list[dict[str, object]] = []
    for type_name in np.asarray(plan["common_type_names"]).astype(str):
        candidate_contracts: dict[int, dict[str, object]] = {}
        candidate_diagnostics: list[dict[str, object]] = []
        for components in range(
            FROZEN_MAX_STATE_COMPONENTS_PER_TYPE,
            FROZEN_MIN_STATE_COMPONENTS_PER_TYPE - 1,
            -1,
        ):
            failure_reasons: list[str] = []
            candidate = _try_type_state_contract(
                type_name=type_name,
                component_count=components,
                values=values,
                centered=centered,
                banks=banks,
                types=types,
                category=np.asarray(plan["category_index"], dtype=np.int64),
                retained=retained,
                anchor_fit_weights=anchor_fit_weights,
                bank_names=bank_names,
                wrong_banks=wrong_banks,
                iterations=iterations,
                failure_reasons=failure_reasons,
            )
            if candidate is None:
                candidate_diagnostics.append(
                    {
                        "K": components,
                        "status": "ineligible",
                        "reason": failure_reasons[-1] if failure_reasons else "unspecified",
                    }
                )
            else:
                candidate_contracts[components] = candidate
                candidate_diagnostics.append(
                    {"K": components, "status": "all_bank_estimable", "reason": "eligible"}
                )
        if not candidate_contracts:
            unevaluable.append(
                {
                    "Level1_type": type_name,
                    "reason": "no_naturally_supported_joint_QC_state_exact_ESS_contract",
                    "component_count_candidates": candidate_diagnostics,
                }
            )
        else:
            contract = candidate_contracts[max(candidate_contracts)]
            contract["component_count_candidates"] = candidate_diagnostics
            contracts.append(contract)
    if not contracts:
        raise RuntimeError("fold has no evaluable naturally supported state type")
    conditional_type_coverage = len(contracts) / len(np.asarray(plan["common_type_names"]))
    global_type_coverage = len(contracts) / len(np.asarray(plan["all_observed_type_names"]))
    if global_type_coverage < FROZEN_STATE_MIN_TYPE_COVERAGE:
        raise RuntimeError("fold has insufficient naturally supported type coverage")

    component_types: list[str] = []
    component_local_ids: list[int] = []
    component_masses: list[float] = []
    component_target_ess: list[float] = []
    component_anchors: list[np.ndarray] = []
    component_scales: list[float] = []
    assignment = np.full(len(values), -1, dtype=np.int16)
    distance_ratio = np.full(len(values), np.nan, dtype=np.float64)
    joint_retained = np.zeros(len(values), dtype=bool)
    single_weights = np.zeros(len(values), dtype=np.float64)
    generic_weights = np.zeros(len(values), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    component_index = 0
    type_mass = 1.0 / len(contracts)
    for contract in contracts:
        type_name = str(contract["type_name"])
        local_assignment = np.asarray(contract["assignment"], dtype=np.int16)
        local_distance = np.asarray(contract["distance_ratio"], dtype=np.float64)
        local_keep = np.asarray(contract["joint_keep"], dtype=bool)
        for component_contract in contract["components"]:
            local_component = int(component_contract["component"])
            local_rows = local_keep & (local_assignment == local_component)
            assignment[local_rows] = component_index
            distance_ratio[local_rows] = local_distance[local_rows]
            joint_retained |= local_rows
            mass = type_mass * float(component_contract["state_mass_within_type"])
            component_types.append(type_name)
            component_local_ids.append(local_component)
            component_masses.append(mass)
            component_target_ess.append(float(component_contract["target_ess"]))
            component_anchors.append(np.asarray(component_contract["anchor"], dtype=np.float64))
            component_scales.append(float(component_contract["scale"]))
            for bank in bank_names:
                single_weights += mass * np.asarray(
                    component_contract["single_conditional"][bank], dtype=np.float64
                )
            for bank in wrong_banks:
                generic_weights += (
                    mass
                    / len(wrong_banks)
                    * np.asarray(
                        component_contract["generic_conditional"][bank],
                        dtype=np.float64,
                    )
                )
            diagnostics.append(
                {
                    "component_index": component_index,
                    "Level1_type": type_name,
                    "training_anchor_id": local_component,
                    "global_mass": mass,
                    "target_component_ESS": float(component_contract["target_ess"]),
                    "QC_strata": list(component_contract["qc_strata"]),
                    "QC_conditional_mass": np.asarray(
                        component_contract["qc_target"], dtype=np.float64
                    ).tolist(),
                    "weight_fits": component_contract["weight_diagnostics"],
                }
            )
            component_index += 1

    component_mass_array = np.asarray(component_masses, dtype=np.float64)
    component_ess_array = np.asarray(component_target_ess, dtype=np.float64)
    expected_total_ess = float(1.0 / np.sum(component_mass_array**2 / component_ess_array))
    for bank in bank_names:
        rows = banks == bank
        if not np.isclose(single_weights[rows].sum(), 1.0, atol=1.0e-10) or not np.isclose(
            _ess(single_weights[rows]),
            expected_total_ess,
            rtol=1.0e-8,
            atol=1.0e-8,
        ):
            raise RuntimeError("single-bank joint state/QC weighting lost exact total ESS")
    if (
        not np.isclose(generic_weights.sum(), 1.0, atol=1.0e-10)
        or not np.isclose(
            _ess(generic_weights),
            expected_total_ess,
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        or np.any(single_weights[joint_retained] <= 0)
        or np.any(single_weights[~joint_retained] != 0)
        or np.any(generic_weights[(banks == donor) | ~joint_retained] != 0)
    ):
        raise RuntimeError("pooled generic joint state/QC weighting is malformed")

    evaluable_types = tuple(str(contract["type_name"]) for contract in contracts)
    support_eligibility_by_bank: dict[str, dict[str, float | int]] = {}
    for bank in bank_names:
        bank_rows = banks == bank
        qc_rows = bank_rows & retained
        evaluable_rows = qc_rows & np.isin(types, np.asarray(evaluable_types))
        joint_rows = bank_rows & joint_retained
        input_cells = int(np.count_nonzero(bank_rows))
        qc_cells = int(np.count_nonzero(qc_rows))
        evaluable_cells = int(np.count_nonzero(evaluable_rows))
        joint_cells = int(np.count_nonzero(joint_rows))
        coverage = joint_cells / input_cells
        if coverage < FROZEN_STATE_MIN_GLOBAL_CELL_COVERAGE:
            raise RuntimeError(f"fold has insufficient global state coverage for {bank}")
        support_eligibility_by_bank[bank] = {
            "input_cells": input_cells,
            "outside_common_type_or_QC_support_cells": input_cells - qc_cells,
            "unevaluable_type_cells_after_QC_support": qc_cells - evaluable_cells,
            "outside_joint_QC_state_support_cells": evaluable_cells - joint_cells,
            "joint_QC_state_retained_cells": joint_cells,
            "retention_fraction_of_input": float(coverage),
            "retention_fraction_of_QC_support": float(joint_cells / qc_cells),
        }
    anchor_values = np.vstack(component_anchors)
    anchor_scales = np.asarray(component_scales, dtype=np.float64)
    return {
        "latent": values,
        "centered": centered,
        "assignment": assignment,
        "distance_ratio": distance_ratio,
        "joint_retained": joint_retained,
        "single_weights": single_weights,
        "generic_weights": generic_weights,
        "component_type_names": np.asarray(component_types),
        "component_local_ids": np.asarray(component_local_ids, dtype=np.int16),
        "component_masses": component_mass_array,
        "component_target_ess": component_ess_array,
        "component_anchors": anchor_values,
        "component_scales": anchor_scales,
        "evaluable_type_names": evaluable_types,
        "conditional_evaluable_type_fraction": float(conditional_type_coverage),
        "global_evaluable_type_fraction": float(global_type_coverage),
        "all_observed_type_names": tuple(
            np.asarray(plan["all_observed_type_names"]).astype(str).tolist()
        ),
        "initial_type_support_diagnostics": json.loads(
            _scalar_text(plan["type_support_diagnostics_json"])
        ),
        "unevaluable_types": unevaluable,
        "type_contracts": contracts,
        "component_diagnostics": diagnostics,
        "single_bank_target_ess": expected_total_ess,
        "generic_pooled_target_ess": expected_total_ess,
        "support_eligibility_by_bank": support_eligibility_by_bank,
        "support_restriction_used": any(
            int(value["joint_QC_state_retained_cells"]) < int(value["input_cells"])
            for value in support_eligibility_by_bank.values()
        ),
        "reference_latent_sha256": _array_digest(cells, values),
        "state_anchor_sha256": _array_digest(
            np.asarray(component_types),
            np.asarray(component_local_ids, dtype=np.int16),
            anchor_values,
            anchor_scales,
        ),
        "anchor_training_weight_sha256": _array_digest(
            np.asarray(wrong_banks),
            np.asarray(plan["common_type_names"]).astype(str),
            *[
                np.asarray(
                    [
                        np.count_nonzero((banks == bank) & (types == type_name))
                        for type_name in np.asarray(plan["common_type_names"]).astype(str)
                    ],
                    dtype=np.int64,
                )
                for bank in wrong_banks
            ],
        ),
        "anchor_training_donor_ids": wrong_banks,
        "seed": int(seed),
        "iterations": int(iterations),
        "score_target_opened": False,
    }


def _state_authority_receipt(
    plan: Mapping[str, np.ndarray],
    state: Mapping[str, object],
    settings: tuple[int, int, float],
) -> dict[str, object]:
    """Serialize the recomputable, target-free state authority."""

    return {
        "schema": AUTHORITY_SCHEMA,
        "method": STATE_BALANCE_METHOD,
        "maximum_components_per_type": FROZEN_MAX_STATE_COMPONENTS_PER_TYPE,
        "minimum_components_per_type": FROZEN_MIN_STATE_COMPONENTS_PER_TYPE,
        "representation": _scalar_text(plan["state_representation"]),
        "reference_latent_sha256": str(state["reference_latent_sha256"]),
        "state_anchor_sha256": str(state["state_anchor_sha256"]),
        "anchor_training_donor_ids": list(state["anchor_training_donor_ids"]),
        "anchor_training_weight_sha256": str(state["anchor_training_weight_sha256"]),
        "heldout_donor_excluded_from_anchor_fit": (
            _scalar_text(plan["heldout_donor"]) not in state["anchor_training_donor_ids"]
        ),
        "component_count_selection": (
            "largest_all_bank_estimable_K_from_training_only_fitted_K3_then_K2_anchors"
        ),
        "evaluable_type_names": list(state["evaluable_type_names"]),
        "all_observed_type_names": list(state["all_observed_type_names"]),
        "conditional_evaluable_type_fraction": float(state["conditional_evaluable_type_fraction"]),
        "global_evaluable_type_fraction": float(state["global_evaluable_type_fraction"]),
        "initial_type_support_diagnostics": list(state["initial_type_support_diagnostics"]),
        "unevaluable_types": list(state["unevaluable_types"]),
        "single_bank_target_ESS": float(state["single_bank_target_ess"]),
        "generic_pooled_target_ESS": float(state["generic_pooled_target_ess"]),
        "component_diagnostics": list(state["component_diagnostics"]),
        "support_eligibility_by_bank": dict(state["support_eligibility_by_bank"]),
        "type_diagnostics": [
            {
                "Level1_type": str(value["type_name"]),
                "component_count": int(value["component_count"]),
                "component_count_candidates": value["component_count_candidates"],
                "anchor_separation_ratio": float(value["anchor_separation_ratio"]),
                "natural_support": value["natural_support"],
                "joint_support": value["post_support"],
            }
            for value in state["type_contracts"]
        ],
        "seed": settings[0],
        "iterations": settings[1],
        "temperature": settings[2],
        "score_target_opened": False,
        "hard_subsampling_used": False,
        "support_restriction_used": bool(state["support_restriction_used"]),
        "all_within_support_weights_positive": True,
    }


def _adapter_call_receipt(
    plan: Mapping[str, np.ndarray],
    state: Mapping[str, object],
    *,
    mode: str,
    donor_ids: tuple[str, ...],
) -> dict[str, object]:
    """Reconstruct one complete adapter call receipt from frozen arrays."""

    banks = np.asarray(plan["bank_ids"]).astype(str)
    types = np.asarray(plan["type_ids"]).astype(str)
    assignment = np.asarray(state["assignment"], dtype=np.int16)
    component_types = np.asarray(state["component_type_names"]).astype(str)
    if mode == "single_bank" and len(donor_ids) == 1:
        weights = np.asarray(state["single_weights"], dtype=np.float64)
    elif mode == "generic_donor_equal" and donor_ids == tuple(state["anchor_training_donor_ids"]):
        weights = np.asarray(state["generic_weights"], dtype=np.float64)
    else:
        raise ValueError("adapter receipt donor/mode contract is malformed")
    selected = np.isin(banks, np.asarray(donor_ids))
    adapter_input = selected & np.asarray(plan["adapter_input_expected"], dtype=bool)
    positive = selected & (weights > 0)
    if np.any(positive & ~adapter_input):
        raise RuntimeError("positive support lies outside the frozen adapter input vocabulary")
    if not np.isclose(weights[selected].sum(), 1.0, atol=1.0e-10):
        raise RuntimeError("adapter receipt weights do not sum to one")

    def groups(*, pooled: bool) -> list[dict[str, object]]:
        group_donors: tuple[str, ...] = ("generic_donor_equal",) if pooled else donor_ids
        result = []
        for group_donor in group_donors:
            donor_rows = selected if pooled else banks == group_donor
            for type_name in sorted(set(component_types.tolist())):
                indices = np.flatnonzero(component_types == type_name)
                components = []
                for component in indices:
                    rows = donor_rows & (types == type_name) & (assignment == component)
                    local = weights[rows]
                    if not len(local) or np.any(local <= 0):
                        raise RuntimeError("adapter receipt component support disappeared")
                    components.append(
                        {
                            "component_index": int(component),
                            "cells": int(np.count_nonzero(rows)),
                            "mass": float(local.sum()),
                            "ESS": float(_ess(local)),
                        }
                    )
                result.append(
                    {
                        "donor": group_donor,
                        "Level1_type": type_name,
                        "component_count": int(len(indices)),
                        "components": components,
                    }
                )
        return result

    source_groups = groups(pooled=False)
    model_groups = groups(pooled=mode == "generic_donor_equal")
    return {
        "mode": mode,
        "donor_ids": list(donor_ids),
        "input_cells": int(np.count_nonzero(adapter_input)),
        "retained_cells": int(np.count_nonzero(positive)),
        "cell_weight_ess": float(_ess(weights[selected])),
        "component_count": int(len(component_types)),
        "state_balance_method": STATE_BALANCE_METHOD,
        "state_components_per_type": {
            type_name: int(np.count_nonzero(component_types == type_name))
            for type_name in sorted(set(component_types.tolist()))
        },
        "reference_latent_sha256": str(state["reference_latent_sha256"]),
        "state_anchor_sha256": str(state["state_anchor_sha256"]),
        "state_groups": source_groups,
        "model_state_groups": model_groups,
        "model_component_richness_matched": True,
        "minimum_state_component_contributing_ESS": min(
            float(component["ESS"]) for group in source_groups for component in group["components"]
        ),
        "minimum_model_component_ESS": min(
            float(component["ESS"]) for group in model_groups for component in group["components"]
        ),
    }


class _WeightedReferenceAdapter:
    """Callable adapter installed only for one frozen fit-predict invocation."""

    def __init__(
        self,
        core: Any,
        plan: Mapping[str, np.ndarray],
        latent_provider: Callable[[], np.ndarray],
    ) -> None:
        self.core = core
        self.plan = plan
        _validate_weight_plan(plan, donor=_scalar_text(plan["heldout_donor"]))
        self.latent_provider = latent_provider
        self.state_authority: dict[str, object] | None = None
        self.state_settings: tuple[int, int, float] | None = None
        cells = np.asarray(plan["cell_ids"]).astype(str)
        self.lookup = {cell: index for index, cell in enumerate(cells.tolist())}
        if len(self.lookup) != len(cells):
            raise ValueError("fixed reference plan has duplicate cells")
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        latent: object,
        donor_ids: Sequence[object] | None = None,
        type_labels: Sequence[object] | None = None,
        observation_ids: Sequence[object] | None = None,
        *,
        components_per_type: int = 4,
        type_ids: Sequence[object] | None = None,
        n_components: int | None = None,
        donor_equal: bool = False,
        seed: int = 17,
        iterations: int = 25,
        temperature: float = 1.0,
        variance_floor: float = 1.0e-4,
        source_modality: str = "snrna",
    ) -> object:
        values = np.asarray(latent, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("reference latent is malformed")
        if donor_ids is None or observation_ids is None:
            raise ValueError("fixed-ESS adapter requires donor and observation IDs")
        if type_labels is not None and type_ids is not None:
            raise ValueError("provide type_labels or type_ids, not both")
        labels = type_ids if type_labels is None else type_labels
        if labels is None:
            raise ValueError("fixed-ESS adapter requires Level1 type IDs")
        donors = np.asarray(donor_ids).astype(str)
        types = np.asarray(labels).astype(str)
        observations = np.asarray(observation_ids).astype(str)
        if any(value.shape != (len(values),) for value in (donors, types, observations)):
            raise ValueError("reference metadata are not latent-row aligned")
        if len(set(observations.tolist())) != len(observations):
            raise ValueError("reference observation IDs are not unique")
        if source_modality not in {"snrna", "single_cell"}:
            raise ValueError("fixed-ESS reference must remain single-cell/nuclear")
        count = int(n_components if n_components is not None else components_per_type)
        if count != FROZEN_MAX_STATE_COMPONENTS_PER_TYPE or iterations < 1 or temperature <= 0:
            raise ValueError("mixture settings differ from the frozen state-balance contract")
        settings = (int(seed), int(iterations), float(temperature))
        if self.state_authority is None:
            self.state_authority = _fit_state_balance_authority(
                self.plan,
                self.latent_provider(),
                seed=int(seed),
                iterations=int(iterations),
            )
            self.state_settings = settings
        elif self.state_settings != settings:
            raise ValueError("reference calls changed the natural-state anchor settings")

        try:
            plan_rows = np.asarray([self.lookup[value] for value in observations], dtype=np.int64)
        except KeyError as error:
            raise ValueError(
                f"reference cell is absent from the frozen weight plan: {error}"
            ) from error
        plan_banks = np.asarray(self.plan["bank_ids"]).astype(str)[plan_rows]
        plan_types = np.asarray(self.plan["type_ids"]).astype(str)[plan_rows]
        if not np.array_equal(plan_banks, donors) or not np.array_equal(plan_types, types):
            raise ValueError("adapter call metadata differ from the frozen weight plan")
        if not np.array_equal(
            values,
            np.asarray(self.state_authority["latent"], dtype=np.float64)[plan_rows],
        ):
            raise ValueError("adapter call latent differs from the captured frozen encoder latent")
        unique_donors = tuple(sorted(set(donors.tolist())))
        wrong = tuple(sorted(np.asarray(self.plan["wrong_donor_ids"]).astype(str).tolist()))
        if len(unique_donors) == 1:
            mode = "single_bank"
            plan_weights = np.asarray(self.state_authority["single_weights"], dtype=np.float64)
        elif unique_donors == wrong and donor_equal:
            mode = "generic_donor_equal"
            plan_weights = np.asarray(self.state_authority["generic_weights"], dtype=np.float64)
        else:
            raise ValueError("adapter received an unregistered reference-bank combination")
        plan_cells = np.asarray(self.plan["cell_ids"]).astype(str)
        plan_bank_ids = np.asarray(self.plan["bank_ids"]).astype(str)
        adapter_input = np.asarray(self.plan["adapter_input_expected"], dtype=bool)
        expected_input_cells = set(
            plan_cells[np.isin(plan_bank_ids, np.asarray(unique_donors)) & adapter_input].tolist()
        )
        if set(observations.tolist()) != expected_input_cells:
            raise ValueError("adapter call differs from the complete registered bank cell set")
        weights = plan_weights[plan_rows]
        keep = weights > 0
        if not np.any(keep):
            raise ValueError("adapter bank has no retained common-support cells")
        values, donors, types, observations, weights = (
            value[keep] for value in (values, donors, types, observations, weights)
        )
        weights = weights / weights.sum()

        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        component_weights: list[float] = []
        out_donors: list[str] = []
        out_types: list[str] = []
        component_ids: list[str] = []
        authority = self.state_authority
        if authority is None:
            raise RuntimeError("natural-state authority was not initialized")
        assignment_all = np.asarray(authority["assignment"], dtype=np.int16)[plan_rows][keep]
        component_types_array = np.asarray(authority["component_type_names"]).astype(str)
        mixture_donor = unique_donors[0] if mode == "single_bank" else "generic_donor_equal"
        for type_name in sorted(set(component_types_array.tolist())):
            component_indices = np.flatnonzero(component_types_array == type_name)
            for component_index in component_indices:
                rows = np.flatnonzero((types == type_name) & (assignment_all == component_index))
                if not len(rows):
                    raise RuntimeError("naturally supported model component disappeared")
                local_weight = weights[rows]
                mass = float(local_weight.sum())
                mean = np.average(values[rows], axis=0, weights=local_weight)
                residual = values[rows] - mean
                variance = np.average(residual**2, axis=0, weights=local_weight)
                means.append(mean)
                variances.append(np.maximum(variance, float(variance_floor)))
                component_weights.append(mass)
                out_donors.append(mixture_donor)
                out_types.append(type_name)
                component_ids.append(
                    f"fixed_state::{mixture_donor}::{type_name}::training_anchor::{component_index}"
                )
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
        digest.update(_json_bytes(observations.tolist()))
        digest.update(np.ascontiguousarray(weights, dtype="<f8").tobytes())
        result = self.core.ReferenceMixture(
            means=np.asarray(means),
            variances=np.asarray(variances),
            weights=np.asarray(component_weights),
            donor_ids=np.asarray(out_donors),
            type_labels=np.asarray(out_types),
            component_ids=np.asarray(component_ids),
            source_observation_ids=tuple(observations.tolist()),
            source_donor_ids=tuple(donors.tolist()),
            source_type_labels=tuple(types.tolist()),
            source_modality=source_modality,
            source_sha256=digest.hexdigest(),
        )
        expected_ess = (
            float(authority["single_bank_target_ess"])
            if mode == "single_bank"
            else float(authority["generic_pooled_target_ess"])
        )
        achieved = _ess(weights)
        if not np.isclose(achieved, expected_ess, rtol=1.0e-8, atol=1.0e-8):
            raise RuntimeError("adapter changed the frozen cell-weight ESS")
        call_receipt = _adapter_call_receipt(
            self.plan,
            authority,
            mode=mode,
            donor_ids=unique_donors,
        )
        if len(result.weights) != int(call_receipt["component_count"]) or not np.allclose(
            result.weights,
            np.asarray(authority["component_masses"], dtype=np.float64),
            rtol=1.0e-9,
            atol=1.0e-11,
        ):
            raise RuntimeError("adapter model-level component richness or masses changed")
        self.calls.append(call_receipt)
        return result

    def authority_receipt(self) -> dict[str, object]:
        if self.state_authority is None or self.state_settings is None:
            raise RuntimeError("natural-state authority was never initialized")
        return _state_authority_receipt(self.plan, self.state_authority, self.state_settings)

    def authority_arrays(self) -> dict[str, np.ndarray]:
        if self.state_authority is None:
            raise RuntimeError("natural-state authority was never initialized")
        return {
            "fixed_reference_state_latent": np.asarray(
                self.state_authority["latent"], dtype=np.float64
            ),
            "fixed_reference_state_component_type_names": np.asarray(
                self.state_authority["component_type_names"]
            ),
            "fixed_reference_state_component_local_ids": np.asarray(
                self.state_authority["component_local_ids"], dtype=np.int16
            ),
            "fixed_reference_state_component_masses": np.asarray(
                self.state_authority["component_masses"], dtype=np.float64
            ),
            "fixed_reference_state_component_target_ESS": np.asarray(
                self.state_authority["component_target_ess"], dtype=np.float64
            ),
            "fixed_reference_state_anchors": np.asarray(
                self.state_authority["component_anchors"], dtype=np.float64
            ),
            "fixed_reference_state_anchor_scales": np.asarray(
                self.state_authority["component_scales"], dtype=np.float64
            ),
            "fixed_reference_state_assignment": np.asarray(
                self.state_authority["assignment"], dtype=np.int16
            ),
            "fixed_reference_state_distance_ratio": np.asarray(
                self.state_authority["distance_ratio"], dtype=np.float64
            ),
            "fixed_reference_joint_QC_state_retained": np.asarray(
                self.state_authority["joint_retained"], dtype=bool
            ),
            "fixed_reference_single_bank_cell_weights": np.asarray(
                self.state_authority["single_weights"], dtype=np.float64
            ),
            "fixed_reference_generic_cell_weights": np.asarray(
                self.state_authority["generic_weights"], dtype=np.float64
            ),
        }


def _semantic_hash(arrays: Mapping[str, object], baseline: Any) -> str:
    return str(baseline._semantic_array_hash(arrays))


def _verify_semantic(path: Path, expected: str, baseline: Any) -> dict[str, np.ndarray]:
    arrays = _load_arrays(path)
    observed = _semantic_hash(arrays, baseline)
    if observed != expected:
        raise ValueError(f"semantic identity mismatch: {path}")
    return arrays


def _baseline_identities(baseline_output: Path) -> dict[str, object]:
    """Verify target-free identities without byte-reading molecular outcomes."""

    files = {
        "development_runner_sha256": BASELINE_RUNNER,
        "generative_core_sha256": Path(__file__).resolve().parents[1]
        / "src/heir/evaluation/generative_fusion.py",
        "prepared_manifest_sha256": baseline_output / "prepared_manifest.json",
        "fit_predict_manifest_sha256": baseline_output / "fit_predict_manifest.json",
    }
    observed = {name: _sha256(path) for name, path in files.items()}
    mismatched = [name for name, value in observed.items() if value != EXPECTED_HASHES[name]]
    if mismatched:
        raise ValueError(f"frozen baseline identities changed: {mismatched}")
    prepared = json.loads((baseline_output / "prepared_manifest.json").read_text())
    bound_target_free = {
        "panel_sha256": Path(str(prepared["panel"])),
        "development_protocol_sha256": Path(str(prepared["protocol"])),
    }
    for name, path in bound_target_free.items():
        observed[name] = _sha256(path)
    mismatched = [name for name, value in observed.items() if value != EXPECTED_HASHES[name]]
    if mismatched:
        raise ValueError(f"frozen baseline identities changed: {mismatched}")
    if str(prepared.get("source_sha256")) != EXPECTED_HASHES["source_sha256"]:
        raise ValueError("frozen prepared manifest source identity changed")
    if str(prepared.get("panel_sha256")) != EXPECTED_HASHES["panel_sha256"]:
        raise ValueError("frozen prepared manifest panel identity changed")
    if str(prepared.get("protocol_sha256")) != EXPECTED_HASHES["development_protocol_sha256"]:
        raise ValueError("frozen prepared manifest protocol identity changed")
    projected = prepared.get("projected_source", {})
    if (
        not isinstance(projected, Mapping)
        or projected.get("used") is not True
        or str(projected.get("sha256")) != EXPECTED_HASHES["projected_source_sha256"]
    ):
        raise ValueError("frozen projected source identity changed")
    # Source/projected/report hashes remain in the receipt, but their bytes are
    # deliberately not read until score-time after all predictions preflight.
    return {
        **EXPECTED_HASHES,
        **observed,
        "source_sha256": str(prepared["source_sha256"]),
    }


def _verify_outcome_bearing_baseline_identities(baseline_output: Path) -> None:
    """Byte-verify molecular/report artifacts only after prediction preflight."""

    baseline_manifest = json.loads(
        (baseline_output / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    paths = {
        "source_sha256": Path(str(baseline_manifest["source"])),
        "projected_source_sha256": baseline_output / "panel_256_projected_counts.npz",
        "development_report_sha256": baseline_output / "report.json",
    }
    mismatched = [name for name, path in paths.items() if _sha256(path) != EXPECTED_HASHES[name]]
    if mismatched:
        raise ValueError(f"outcome-bearing frozen baseline identities changed: {mismatched}")


def _reference_qc(source: Path, baseline: Any) -> dict[str, np.ndarray]:
    with np.load(source, allow_pickle=False) as archive:
        receipt = baseline._json_scalar(archive, "source_receipt_json")
        baseline._validate_encoder_receipt(receipt, synthetic=False)
        required = (
            "sc_cell_ids",
            "sc_donor_ids",
            "sc_total_umi_counts",
            "sc_n_features_rna",
            "sc_percent_mt",
            "sc_percent_ribo",
            "sc_dv200_percent",
            "sc_block_age_months",
        )
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"raw source lacks reference QC: {missing}")
        return {name: np.asarray(archive[name]) for name in required}


def _constant_by_donor(values: np.ndarray, donors: np.ndarray) -> dict[str, list[str]]:
    return {
        donor: sorted(set(np.asarray(values)[donors == donor].astype(str).tolist()))
        for donor in sorted(set(donors.tolist()))
    }


def _validate_protocol(path: Path, protocol: Mapping[str, object]) -> None:
    """Bind every registered protocol statement to the reviewed file bytes."""

    if (
        protocol.get("schema") != "heir.natcommun_fixed_ess_reference_protocol.v3"
        or _sha256(path) != FROZEN_PROTOCOL_SHA256
    ):
        raise ValueError("fixed-ESS protocol differs from the fully frozen v3 contract")


def prepare(args: argparse.Namespace) -> Mapping[str, object]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    _validate_protocol(args.protocol, protocol)
    baseline = _load_baseline_runner()
    identities = _baseline_identities(args.baseline_output)
    baseline_manifest = json.loads(
        (args.baseline_output / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    source = Path(str(baseline_manifest["source"]))
    qc = _reference_qc(source, baseline)
    source_cells = np.asarray(qc["sc_cell_ids"]).astype(str)
    if len(set(source_cells.tolist())) != len(source_cells):
        raise ValueError("raw source reference cell IDs are not unique")
    source_index = {cell: index for index, cell in enumerate(source_cells.tolist())}
    source_donors = np.asarray(qc["sc_donor_ids"]).astype(str)
    dv200 = _constant_by_donor(np.asarray(qc["sc_dv200_percent"]), source_donors)
    age = _constant_by_donor(np.asarray(qc["sc_block_age_months"]), source_donors)
    if any(len(values) != 1 for values in (*dv200.values(), *age.values())):
        raise ValueError("DV200/age are not donor-constant as prespecified")
    ribo = np.asarray(qc["sc_percent_ribo"], dtype=np.float64)
    if not np.all(ribo == 0):
        raise ValueError("percent_ribo is not zero as prespecified")

    folds: dict[str, object] = {}
    score_folds: dict[str, object] = {}
    for donor in baseline_manifest["donors"]:
        donor = str(donor)
        frozen = baseline_manifest["folds"][donor]
        public_path = Path(str(frozen["public_path"]))
        public = _verify_semantic(public_path, str(frozen["public_semantic_sha256"]), baseline)
        baseline.validate_public_fold(public)
        wrong_index = np.asarray(public["wrong_train_sc_index"], dtype=np.int64)
        matched_cells = np.asarray(public["matched_sc_cell_ids"]).astype(str)
        matched_types = np.asarray(public["matched_sc_type_ids"]).astype(str)
        wrong_cells_all = np.asarray(public["train_sc_cell_ids"]).astype(str)[wrong_index]
        wrong_types_all = np.asarray(public["train_sc_type_ids"]).astype(str)[wrong_index]
        wrong_banks_all = np.asarray(public["train_sc_donor_ids"]).astype(str)[wrong_index]
        cells = np.concatenate((matched_cells, wrong_cells_all))
        banks = np.concatenate((np.repeat(donor, len(matched_cells)), wrong_banks_all))
        types = np.concatenate((matched_types, wrong_types_all))
        try:
            rows = np.asarray([source_index[cell] for cell in cells], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"prepared reference cell is absent from raw QC: {error}") from error
        if not np.array_equal(source_donors[rows], banks):
            raise ValueError("raw QC donor IDs differ from the prepared public fold")
        plan = _compute_weight_plan(
            heldout_donor=donor,
            wrong_donor_ids=np.asarray(public["wrong_donor_ids"]),
            cell_ids=cells,
            bank_ids=banks,
            type_ids=types,
            total_umi=np.asarray(qc["sc_total_umi_counts"], dtype=np.float64)[rows],
            n_features=np.asarray(qc["sc_n_features_rna"], dtype=np.float64)[rows],
            percent_mt=np.asarray(qc["sc_percent_mt"], dtype=np.float64)[rows],
            model_type_names=np.asarray(public["train_sc_type_ids"]).astype(str),
        )
        _validate_weight_plan(plan, donor=donor)
        fold_dir = args.output / "folds" / donor
        plan_path = fold_dir / "fixed_ess_plan.npz"
        _atomic_npz(plan_path, plan)
        plan_identity = _semantic_hash(plan, baseline)
        receipt = {
            "schema": PREPARED_SCHEMA,
            "heldout_donor": donor,
            "public_path": str(public_path),
            "public_semantic_sha256": str(frozen["public_semantic_sha256"]),
            "plan_path": str(plan_path),
            "plan_semantic_sha256": plan_identity,
            "all_observed_Level1_types": np.asarray(plan["all_observed_type_names"])
            .astype(str)
            .tolist(),
            "common_Level1_types": np.asarray(plan["common_type_names"]).astype(str).tolist(),
            "initial_type_support_diagnostics": json.loads(
                _scalar_text(plan["type_support_diagnostics_json"])
            ),
            "retained_cells_by_bank": {
                bank: int(
                    np.count_nonzero(
                        (np.asarray(plan["bank_ids"]).astype(str) == bank)
                        & np.asarray(plan["retained"], dtype=bool)
                    )
                )
                for bank in np.asarray(plan["bank_names"]).astype(str)
            },
            "qc_only_maximum_ESS_by_bank": np.asarray(
                plan["qc_only_maximum_ess_by_bank"], dtype=np.float64
            ).tolist(),
            "state_component_range": [
                int(plan["state_min_components_per_type"]),
                int(plan["state_max_components_per_type"]),
            ],
            "state_balance_method": _scalar_text(plan["state_balance_method"]),
            "state_minimum_natural_component_ESS": float(
                plan["state_minimum_natural_component_ess"]
            ),
            "state_anchor_fit": "query_excluded_other_donor_training_references_only",
            "state_weights_deferred_until_frozen_training_encoder_is_fitted": True,
            "support_restriction_used": bool(plan["support_restriction_used"]),
            "hard_subsampling_used": False,
            "score_target_opened": False,
        }
        _atomic_json(fold_dir / "prepare_receipt.json", receipt)
        folds[donor] = receipt
        score_folds[donor] = {
            "heldout_donor": donor,
            "score_target_path": str(Path(str(frozen["score_target_path"])).resolve()),
            "score_target_semantic_sha256": str(frozen["score_target_semantic_sha256"]),
            "public_semantic_sha256": str(frozen["public_semantic_sha256"]),
        }

    manifest = {
        "schema": PREPARED_SCHEMA,
        "analysis_scope": "validation_only_outcome_exposed_non_confirmatory",
        "baseline_output": str(args.baseline_output.resolve()),
        "source": str(source.resolve()),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "baseline_artifact_identities": identities,
        "image_encoder": HOPTIMUS_REPOSITORY,
        "image_encoder_revision": HOPTIMUS_REVISION,
        "uni2_h_run": False,
        "donors": [str(value) for value in baseline_manifest["donors"]],
        "base_seed": FROZEN_BASE_SEED,
        "training_configuration": {
            "epochs": FROZEN_EPOCHS,
            "batch_size": FROZEN_BATCH_SIZE,
            "latent_dim": FROZEN_LATENT_DIM,
            "training_cell_weights": "natural_unchanged",
            "training_ST_weights": "natural_unchanged",
        },
        "reference_intervention": (
            "prediction_time_query_excluded_training_anchors_natural_state_support_"
            "joint_QC_state_mass_and_exact_componentwise_ESS"
        ),
        "hard_subsampling_used": False,
        "support_restriction_used": any(
            bool(folds[donor]["support_restriction_used"]) for donor in baseline_manifest["donors"]
        ),
        "unbalanced_QC_fields": {
            "DV200": {
                "status": "not_balanceable_donor_constant",
                "values_by_donor": dv200,
            },
            "block_age_months": {
                "status": "not_balanceable_donor_constant",
                "values_by_donor": age,
            },
            "percent_ribo": "not_balanceable_all_zero",
        },
        "target_boundary": {
            "prepare_opened_score_target": False,
            "fit_predict_accepts_score_target": False,
            "global_preflight_before_any_score_target": True,
            "outcome_bearing_baseline_byte_verification": (
                "score_only_after_global_prediction_preflight"
            ),
        },
        "resource_limits": {
            "maximum_CPU_threads": 4,
            "maximum_visible_GPUs_used": 1,
            "maximum_GPU_memory_fraction": 0.60,
            "folds_serial": True,
            "swap_permitted": False,
        },
        "folds": folds,
    }
    prepared_path = args.output / "prepared_manifest.json"
    _atomic_json(prepared_path, manifest)
    _atomic_json(
        args.output / "score_target_manifest.json",
        {
            "schema": SCORE_TARGET_SCHEMA,
            "analysis_scope": "validation_only_outcome_exposed_non_confirmatory",
            "prepared_manifest_sha256": _sha256(prepared_path),
            "protocol_sha256": _sha256(args.protocol),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "donors": [str(value) for value in baseline_manifest["donors"]],
            "folds": score_folds,
        },
    )
    return manifest


def _read_prepared(output: Path, protocol: Path) -> dict[str, object]:
    path = output / "prepared_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != PREPARED_SCHEMA:
        raise ValueError("fixed-ESS prepared manifest is missing or malformed")
    serialized = json.dumps(value, sort_keys=True)
    if any(
        f'"{field}"' in serialized
        for field in (
            "score_target_path",
            "score_target_semantic_sha256",
            "heldout_st_counts",
            "heldout_st_library",
            "primary_score_eligible",
        )
    ):
        raise ValueError("fixed-reference fit manifest contains score-target authority")
    if value.get("runner_sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("fixed-ESS runner changed after preparation")
    if value.get("image_encoder") != HOPTIMUS_REPOSITORY or value.get("uni2_h_run") is not False:
        raise ValueError("prepared encoder identity is not frozen H-optimus-1-only")
    if int(value.get("base_seed", -1)) != FROZEN_BASE_SEED:
        raise ValueError("fixed-ESS prepared seed changed")
    prepared_protocol = Path(str(value.get("protocol", "")))
    expected_protocol = protocol.resolve()
    if (
        prepared_protocol != expected_protocol
        or not prepared_protocol.is_file()
        or value.get("protocol_sha256") != _sha256(prepared_protocol)
    ):
        raise ValueError("fixed-ESS protocol path or content changed after preparation")
    _validate_protocol(
        prepared_protocol,
        json.loads(prepared_protocol.read_text(encoding="utf-8")),
    )
    return value


def _read_score_target_manifest(output: Path, prepared: Mapping[str, object]) -> dict[str, object]:
    """Open score-target authority only after every prediction has preflighted."""

    path = output / "score_target_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    donors = [str(item) for item in prepared["donors"]]
    if (
        value.get("schema") != SCORE_TARGET_SCHEMA
        or value.get("prepared_manifest_sha256") != _sha256(output / "prepared_manifest.json")
        or value.get("protocol_sha256") != prepared.get("protocol_sha256")
        or value.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or value.get("donors") != donors
        or set(value.get("folds", {})) != set(donors)
    ):
        raise ValueError("fixed-reference score-target manifest is stale or malformed")
    for donor in donors:
        fold = value["folds"][donor]
        if (
            fold.get("heldout_donor") != donor
            or fold.get("public_semantic_sha256")
            != prepared["folds"][donor]["public_semantic_sha256"]
            or not Path(str(fold.get("score_target_path", ""))).is_absolute()
            or len(str(fold.get("score_target_semantic_sha256", ""))) != 64
        ):
            raise ValueError(f"fixed-reference score-target authority differs for {donor}")
    return value


def _checkpoint_identity(
    *,
    donor: str,
    public_identity: str,
    plan_identity: str,
    fold_seed: int,
    args: argparse.Namespace,
    prepared: Mapping[str, object],
) -> str:
    payload = {
        "schema": PREDICTION_SCHEMA,
        "donor": donor,
        "public_semantic_sha256": public_identity,
        "plan_semantic_sha256": plan_identity,
        "fold_seed": int(fold_seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "latent_dim": int(args.latent_dim),
        "device": str(args.device),
        "protocol_sha256": str(prepared["protocol_sha256"]),
        "sensitivity_runner_sha256": _sha256(Path(__file__).resolve()),
        "prepared_manifest_semantic_sha256": hashlib.sha256(_json_bytes(prepared)).hexdigest(),
        "baseline_artifact_identities": prepared["baseline_artifact_identities"],
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _promote_gate3_support_receipt_precision(
    predictions: dict[str, object],
) -> None:
    """Preserve the frozen Gate-3 threshold decision without float32 aliasing.

    The frozen runner calculates supported composition mass in float64, derives
    eligibility from that value, and then stores only a float32 mass.  A value
    immediately below 0.90 can therefore round to the float32 representation
    of 0.90 and make the otherwise valid receipt fail its own validator.  This
    wrapper recomputes the same diagnostic from the saved H&E composition and
    support mask, verifies the frozen eligibility decision, and stores the mass
    losslessly.  No model rate, posterior, loss, or score field is changed.
    """

    composition = np.asarray(predictions["query_H_composition"])
    supported = np.asarray(predictions["gate3_supported_type_mask"], dtype=bool)
    component_count = np.asarray(predictions["matched_reference_component_count_by_type"])
    if (
        composition.ndim != 2
        or supported.shape != (composition.shape[1],)
        or component_count.shape != supported.shape
        or not np.array_equal(supported, component_count >= 2)
    ):
        raise ValueError("frozen Gate-3 type-support receipt is inconsistent")
    mass64 = np.sum(
        composition * supported[None],
        axis=1,
        dtype=np.float64,
    )
    stored_mass = np.asarray(predictions["gate3_supported_composition_mass"])
    stored_eligible = np.asarray(predictions["gate3_supported_score_eligible"], dtype=bool)
    if (
        stored_mass.shape != mass64.shape
        or stored_eligible.shape != mass64.shape
        or not np.array_equal(stored_mass, mass64.astype(stored_mass.dtype))
        or not np.array_equal(stored_eligible, mass64 >= 0.90)
    ):
        raise ValueError("frozen Gate-3 mass/eligibility receipt changed semantically")
    float32_decision = mass64.astype(np.float32) >= 0.90
    rounding_rows = int(np.count_nonzero(float32_decision != stored_eligible))
    predictions["gate3_supported_composition_mass"] = mass64
    predictions["fixed_reference_gate3_mass_receipt_precision"] = np.asarray(
        "float64_exact_recomputation_from_frozen_H_and_E_composition"
    )
    predictions["fixed_reference_gate3_threshold_rounding_rows"] = np.asarray(
        rounding_rows, dtype=np.int64
    )


def _validate_fixed_prediction(
    *,
    baseline: Any,
    predictions: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    plan: Mapping[str, np.ndarray],
    donor: str,
    epochs: int,
) -> None:
    """Recompute all target-free joint state/QC/ESS invariants."""

    _validate_weight_plan(plan, donor=donor)
    baseline._validate_prediction_artifact(
        predictions,
        public,
        donor=donor,
        epochs=epochs,
    )
    required = {
        "fixed_reference_plan_semantic_sha256",
        "fixed_reference_QC_common_type_names",
        "fixed_reference_common_type_names",
        "fixed_reference_single_bank_cell_weight_ESS",
        "fixed_reference_generic_pooled_cell_weight_ESS",
        "fixed_reference_hard_subsampling_used",
        "fixed_reference_support_restriction_used",
        "fixed_reference_adapter_calls_json",
        "fixed_reference_state_capture_json",
        "fixed_reference_state_authority_json",
        "fixed_reference_state_component_type_names",
        "fixed_reference_state_latent",
        "fixed_reference_state_component_local_ids",
        "fixed_reference_state_component_masses",
        "fixed_reference_state_component_target_ESS",
        "fixed_reference_state_anchors",
        "fixed_reference_state_anchor_scales",
        "fixed_reference_state_assignment",
        "fixed_reference_state_distance_ratio",
        "fixed_reference_joint_QC_state_retained",
        "fixed_reference_single_bank_cell_weights",
        "fixed_reference_generic_cell_weights",
        "fixed_reference_gate3_mass_receipt_precision",
        "fixed_reference_gate3_threshold_rounding_rows",
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"fixed-ESS prediction receipt is incomplete for {donor}: {missing}")
    if _scalar_text(predictions["fixed_reference_plan_semantic_sha256"]) != _semantic_hash(
        plan, baseline
    ):
        raise ValueError(f"fixed-ESS prediction plan identity changed for {donor}")
    if not np.array_equal(
        np.asarray(predictions["fixed_reference_QC_common_type_names"]),
        np.asarray(plan["common_type_names"]),
    ):
        raise ValueError(f"fixed-ESS QC type support changed for {donor}")
    if bool(predictions["fixed_reference_hard_subsampling_used"]):
        raise ValueError(f"hard subsampling was used unexpectedly for {donor}")
    gate3_mass = np.asarray(predictions["gate3_supported_composition_mass"])
    gate3_composition = np.asarray(predictions["query_H_composition"])
    gate3_supported = np.asarray(predictions["gate3_supported_type_mask"], dtype=bool)
    expected_gate3_mass = np.sum(
        gate3_composition * gate3_supported[None], axis=1, dtype=np.float64
    )
    expected_rounding_rows = int(
        np.count_nonzero(
            (expected_gate3_mass.astype(np.float32) >= 0.90) != (expected_gate3_mass >= 0.90)
        )
    )
    if (
        gate3_mass.dtype != np.dtype(np.float64)
        or not np.array_equal(gate3_mass, expected_gate3_mass)
        or _scalar_text(predictions["fixed_reference_gate3_mass_receipt_precision"])
        != "float64_exact_recomputation_from_frozen_H_and_E_composition"
        or int(predictions["fixed_reference_gate3_threshold_rounding_rows"])
        != expected_rounding_rows
    ):
        raise ValueError(f"fixed-ESS Gate-3 precision receipt changed for {donor}")
    try:
        calls = json.loads(_scalar_text(predictions["fixed_reference_adapter_calls_json"]))
        capture = json.loads(_scalar_text(predictions["fixed_reference_state_capture_json"]))
        authority = json.loads(_scalar_text(predictions["fixed_reference_state_authority_json"]))
    except json.JSONDecodeError as error:
        raise ValueError(f"fixed-ESS adapter receipt is malformed for {donor}") from error
    wrong_donors = tuple(sorted(np.asarray(public["wrong_donor_ids"]).astype(str).tolist()))
    wrong_count = len(wrong_donors)
    if tuple(np.asarray(plan["model_type_names"]).astype(str).tolist()) != tuple(
        sorted(set(np.asarray(public["train_sc_type_ids"]).astype(str).tolist()))
    ):
        raise ValueError(f"fixed-reference model type vocabulary changed for {donor}")
    if not isinstance(calls, list) or any(not isinstance(value, Mapping) for value in calls):
        raise ValueError(f"fixed-ESS adapter call receipt is malformed for {donor}")
    cells = np.asarray(plan["cell_ids"]).astype(str)
    banks = np.asarray(plan["bank_ids"]).astype(str)
    qc_category = np.asarray(plan["category_index"], dtype=np.int64)
    state_latent = np.asarray(predictions["fixed_reference_state_latent"], dtype=np.float64)
    component_types = np.asarray(predictions["fixed_reference_state_component_type_names"]).astype(
        str
    )
    component_local_ids = np.asarray(
        predictions["fixed_reference_state_component_local_ids"], dtype=np.int16
    )
    component_masses = np.asarray(
        predictions["fixed_reference_state_component_masses"], dtype=np.float64
    )
    component_target_ess = np.asarray(
        predictions["fixed_reference_state_component_target_ESS"], dtype=np.float64
    )
    anchor_values = np.asarray(predictions["fixed_reference_state_anchors"], dtype=np.float64)
    anchor_scales = np.asarray(predictions["fixed_reference_state_anchor_scales"], dtype=np.float64)
    assignment = np.asarray(predictions["fixed_reference_state_assignment"], dtype=np.int16)
    distance_ratio = np.asarray(
        predictions["fixed_reference_state_distance_ratio"], dtype=np.float64
    )
    joint_retained = np.asarray(predictions["fixed_reference_joint_QC_state_retained"], dtype=bool)
    single_weights = np.asarray(
        predictions["fixed_reference_single_bank_cell_weights"], dtype=np.float64
    )
    generic_weights = np.asarray(
        predictions["fixed_reference_generic_cell_weights"], dtype=np.float64
    )
    component_count = len(component_types)
    vectors = (
        component_local_ids,
        component_masses,
        component_target_ess,
        anchor_scales,
    )
    row_vectors = (assignment, distance_ratio, joint_retained, single_weights, generic_weights)
    if (
        not component_count
        or state_latent.shape != (len(cells), FROZEN_LATENT_DIM)
        or not np.isfinite(state_latent).all()
        or any(value.shape != (component_count,) for value in vectors)
        or anchor_values.shape != (component_count, FROZEN_LATENT_DIM)
        or any(value.shape != (len(cells),) for value in row_vectors)
        or not np.isfinite(anchor_values).all()
        or not np.isfinite(anchor_scales).all()
        or np.any(anchor_scales <= 0)
        or not np.isfinite(component_masses).all()
        or np.any(component_masses <= 0)
        or not np.isclose(component_masses.sum(), 1.0, atol=1.0e-10)
        or not np.isfinite(component_target_ess).all()
        or np.any(component_target_ess <= 0)
        or np.any(~np.isfinite(single_weights))
        or np.any(~np.isfinite(generic_weights))
    ):
        raise ValueError(f"fixed-reference state arrays are malformed for {donor}")
    evaluable_types = tuple(sorted(set(component_types.tolist())))
    if tuple(
        np.asarray(predictions["fixed_reference_common_type_names"]).astype(str).tolist()
    ) != evaluable_types or not set(evaluable_types).issubset(
        set(np.asarray(plan["common_type_names"]).astype(str).tolist())
    ):
        raise ValueError(f"fixed-reference evaluable type support changed for {donor}")
    if any(
        not FROZEN_MIN_STATE_COMPONENTS_PER_TYPE
        <= np.count_nonzero(component_types == type_name)
        <= FROZEN_MAX_STATE_COMPONENTS_PER_TYPE
        for type_name in evaluable_types
    ):
        raise ValueError(f"fixed-reference natural component count changed for {donor}")
    if (
        not np.array_equal(joint_retained, assignment >= 0)
        or np.any(assignment[joint_retained] >= component_count)
        or np.any(~np.isfinite(distance_ratio[joint_retained]))
        or np.any(distance_ratio[joint_retained] < 0)
        or np.any(np.isfinite(distance_ratio[~joint_retained]))
        or np.any(single_weights[joint_retained] <= 0)
        or np.any(single_weights[~joint_retained] != 0)
        or np.any(generic_weights < 0)
        or np.any(generic_weights[(banks != donor) & joint_retained] <= 0)
        or np.any(generic_weights[(banks == donor) | ~joint_retained] != 0)
    ):
        raise ValueError(f"fixed-reference joint state support is malformed for {donor}")
    if _array_digest(cells, state_latent) != capture.get("reference_latent_sha256"):
        raise ValueError(f"fixed-reference captured latent identity changed for {donor}")
    try:
        recomputed_state = _fit_state_balance_authority(
            plan,
            state_latent,
            seed=int(baseline._fold_seed(FROZEN_BASE_SEED, donor)),
            iterations=25,
        )
    except (ValueError, RuntimeError) as error:
        raise ValueError(
            f"fixed-reference state authority is no longer evaluable for {donor}"
        ) from error
    recomputed_arrays = {
        "component_type_names": np.asarray(recomputed_state["component_type_names"]),
        "component_local_ids": np.asarray(recomputed_state["component_local_ids"], dtype=np.int16),
        "component_masses": np.asarray(recomputed_state["component_masses"], dtype=np.float64),
        "component_target_ess": np.asarray(
            recomputed_state["component_target_ess"], dtype=np.float64
        ),
        "component_anchors": np.asarray(recomputed_state["component_anchors"], dtype=np.float64),
        "component_scales": np.asarray(recomputed_state["component_scales"], dtype=np.float64),
        "assignment": np.asarray(recomputed_state["assignment"], dtype=np.int16),
        "distance_ratio": np.asarray(recomputed_state["distance_ratio"], dtype=np.float64),
        "joint_retained": np.asarray(recomputed_state["joint_retained"], dtype=bool),
        "single_weights": np.asarray(recomputed_state["single_weights"], dtype=np.float64),
        "generic_weights": np.asarray(recomputed_state["generic_weights"], dtype=np.float64),
    }
    observed_arrays = {
        "component_type_names": component_types,
        "component_local_ids": component_local_ids,
        "component_masses": component_masses,
        "component_target_ess": component_target_ess,
        "component_anchors": anchor_values,
        "component_scales": anchor_scales,
        "assignment": assignment,
        "distance_ratio": distance_ratio,
        "joint_retained": joint_retained,
        "single_weights": single_weights,
        "generic_weights": generic_weights,
    }
    for name, expected in recomputed_arrays.items():
        observed = observed_arrays[name]
        if expected.dtype.kind in "OUSib":
            matches = np.array_equal(observed, expected)
        else:
            matches = np.allclose(observed, expected, rtol=1.0e-10, atol=1.0e-12, equal_nan=True)
        if not matches:
            raise ValueError(f"fixed-reference recomputed state array changed for {donor}: {name}")
    expected_authority = _state_authority_receipt(
        plan,
        recomputed_state,
        (int(baseline._fold_seed(FROZEN_BASE_SEED, donor)), 25, 1.0),
    )
    if authority != expected_authority:
        raise ValueError(f"fixed-reference recomputed state receipt changed for {donor}")
    if bool(predictions["fixed_reference_support_restriction_used"]) != bool(
        recomputed_state["support_restriction_used"]
    ):
        raise ValueError(f"fixed-reference support-restriction receipt changed for {donor}")
    anchor_hash = _array_digest(
        component_types,
        component_local_ids,
        anchor_values,
        anchor_scales,
    )
    expected_anchor_weight_hash = _array_digest(
        np.asarray(wrong_donors),
        np.asarray(plan["common_type_names"]).astype(str),
        *[
            np.asarray(
                [
                    np.count_nonzero(
                        (banks == bank) & (np.asarray(plan["type_ids"]).astype(str) == type_name)
                    )
                    for type_name in np.asarray(plan["common_type_names"]).astype(str)
                ],
                dtype=np.int64,
            )
            for bank in wrong_donors
        ],
    )
    total_target = float(1.0 / np.sum(component_masses**2 / component_target_ess))
    if not np.isclose(
        float(predictions["fixed_reference_single_bank_cell_weight_ESS"]),
        total_target,
        rtol=1.0e-8,
        atol=1.0e-8,
    ) or not np.isclose(
        float(predictions["fixed_reference_generic_pooled_cell_weight_ESS"]),
        total_target,
        rtol=1.0e-8,
        atol=1.0e-8,
    ):
        raise ValueError(f"fixed-reference derived total ESS changed for {donor}")
    for bank in (donor, *wrong_donors):
        bank_rows = banks == bank
        if not np.isclose(single_weights[bank_rows].sum(), 1.0, atol=1.0e-10) or not np.isclose(
            _ess(single_weights[bank_rows]),
            total_target,
            rtol=1.0e-8,
            atol=1.0e-8,
        ):
            raise ValueError(f"fixed-reference single-bank total ESS changed for {bank}")
        for component in range(component_count):
            rows = bank_rows & (assignment == component)
            if (
                not np.any(rows)
                or not np.isclose(
                    single_weights[rows].sum(),
                    component_masses[component],
                    rtol=1.0e-9,
                    atol=1.0e-11,
                )
                or not np.isclose(
                    _ess(single_weights[rows]),
                    component_target_ess[component],
                    rtol=1.0e-8,
                    atol=1.0e-8,
                )
            ):
                raise ValueError(
                    f"fixed-reference component ESS/mass changed for {bank}/{component}"
                )
    if not np.isclose(generic_weights.sum(), 1.0, atol=1.0e-10) or not np.isclose(
        _ess(generic_weights),
        total_target,
        rtol=1.0e-8,
        atol=1.0e-8,
    ):
        raise ValueError(f"fixed-reference generic total ESS changed for {donor}")
    for component in range(component_count):
        pooled_rows = assignment == component
        if not np.isclose(
            generic_weights[pooled_rows].sum(),
            component_masses[component],
            rtol=1.0e-9,
            atol=1.0e-11,
        ) or not np.isclose(
            _ess(generic_weights[pooled_rows]),
            component_target_ess[component],
            rtol=1.0e-8,
            atol=1.0e-8,
        ):
            raise ValueError(f"fixed-reference generic component ESS changed for {component}")
        expected_qc_categories: tuple[int, ...] | None = None
        expected_qc_mass: np.ndarray | None = None
        for bank in (donor, *wrong_donors):
            rows = (banks == bank) & (assignment == component)
            categories = sorted(set(qc_category[rows].tolist()))
            observed = np.asarray(
                [single_weights[rows & (qc_category == value)].sum() for value in categories]
            )
            observed /= observed.sum()
            if expected_qc_mass is None:
                expected_qc_categories = tuple(categories)
                expected_qc_mass = observed
            elif (
                tuple(categories) != expected_qc_categories
                or len(observed) != len(expected_qc_mass)
                or not np.allclose(observed, expected_qc_mass, rtol=1.0e-9, atol=1.0e-11)
            ):
                raise ValueError(
                    f"fixed-reference joint QC/state mass changed for component {component}"
                )
        if expected_qc_categories is None or expected_qc_mass is None:
            raise ValueError(f"fixed-reference QC mass is absent for component {component}")
        for bank in wrong_donors:
            rows = (banks == bank) & (assignment == component)
            generic_qc_mass = np.asarray(
                [
                    generic_weights[rows & (qc_category == value)].sum()
                    for value in expected_qc_categories
                ],
                dtype=np.float64,
            )
            generic_qc_mass /= generic_qc_mass.sum()
            if (
                not np.allclose(
                    generic_qc_mass,
                    expected_qc_mass,
                    rtol=1.0e-9,
                    atol=1.0e-11,
                )
                or not np.isclose(
                    generic_weights[rows].sum(),
                    component_masses[component] / wrong_count,
                    rtol=1.0e-9,
                    atol=1.0e-11,
                )
                or not np.isclose(
                    _ess(generic_weights[rows]),
                    component_target_ess[component] / wrong_count,
                    rtol=1.0e-8,
                    atol=1.0e-8,
                )
            ):
                raise ValueError(
                    f"fixed-reference donor-equal generic QC/component changed for "
                    f"{bank}/{component}"
                )
    expected_capture_calls = [
        {"ordinal": 0, "role": "aligned_train_st", "rows": len(public["train_st_counts"])},
        {"ordinal": 1, "role": "aligned_train_sc", "rows": len(public["train_sc_counts"])},
        {
            "ordinal": 2,
            "role": "aligned_matched_sc",
            "rows": len(public["matched_sc_counts"]),
        },
        {"ordinal": 3, "role": "unaligned_train_st", "rows": len(public["train_st_counts"])},
        {"ordinal": 4, "role": "unaligned_train_sc", "rows": len(public["train_sc_counts"])},
    ]
    if (
        not isinstance(capture, Mapping)
        or capture.get("contract") != "frozen_encode_sequence_v1"
        or capture.get("score_target_opened") is not False
        or capture.get("calls") != expected_capture_calls
        or len(str(capture.get("reference_latent_sha256", ""))) != 64
        or not isinstance(authority, Mapping)
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("method") != STATE_BALANCE_METHOD
        or authority.get("representation") != _scalar_text(plan["state_representation"])
        or authority.get("maximum_components_per_type") != FROZEN_MAX_STATE_COMPONENTS_PER_TYPE
        or authority.get("minimum_components_per_type") != FROZEN_MIN_STATE_COMPONENTS_PER_TYPE
        or tuple(authority.get("anchor_training_donor_ids", ())) != wrong_donors
        or authority.get("heldout_donor_excluded_from_anchor_fit") is not True
        or authority.get("component_count_selection")
        != "largest_all_bank_estimable_K_from_training_only_fitted_K3_then_K2_anchors"
        or donor in authority.get("anchor_training_donor_ids", ())
        or authority.get("anchor_training_weight_sha256") != expected_anchor_weight_hash
        or tuple(authority.get("evaluable_type_names", ())) != evaluable_types
        or float(authority.get("global_evaluable_type_fraction", 0.0))
        < FROZEN_STATE_MIN_TYPE_COVERAGE
        or authority.get("score_target_opened") is not False
        or authority.get("hard_subsampling_used") is not False
        or authority.get("all_within_support_weights_positive") is not True
        or authority.get("support_restriction_used")
        != bool(recomputed_state["support_restriction_used"])
        or authority.get("reference_latent_sha256") != capture.get("reference_latent_sha256")
        or authority.get("state_anchor_sha256") != anchor_hash
        or not np.isclose(float(authority.get("single_bank_target_ESS", np.nan)), total_target)
        or not np.isclose(float(authority.get("generic_pooled_target_ESS", np.nan)), total_target)
        or authority.get("seed") != int(baseline._fold_seed(FROZEN_BASE_SEED, donor))
        or authority.get("iterations") != 25
        or not np.isclose(float(authority.get("temperature", np.nan)), 1.0)
    ):
        raise ValueError(f"fixed-reference natural-state authority is malformed for {donor}")
    type_diagnostics = authority.get("type_diagnostics")
    component_diagnostics = authority.get("component_diagnostics")
    eligibility = authority.get("support_eligibility_by_bank")
    if (
        not isinstance(type_diagnostics, list)
        or not isinstance(component_diagnostics, list)
        or not isinstance(eligibility, Mapping)
        or len(type_diagnostics) != len(evaluable_types)
        or len(component_diagnostics) != component_count
    ):
        raise ValueError(f"fixed-reference state diagnostics are incomplete for {donor}")
    for bank in (donor, *wrong_donors):
        value = eligibility.get(bank, {})
        expected = recomputed_state["support_eligibility_by_bank"][bank]
        if value != expected or float(value.get("retention_fraction_of_input", 0.0)) < (
            FROZEN_STATE_MIN_GLOBAL_CELL_COVERAGE
        ):
            raise ValueError(f"fixed-reference eligibility receipt changed for {bank}")
    for diagnostic in type_diagnostics:
        if (
            diagnostic.get("Level1_type") not in evaluable_types
            or not FROZEN_MIN_STATE_COMPONENTS_PER_TYPE
            <= int(diagnostic.get("component_count", 0))
            <= FROZEN_MAX_STATE_COMPONENTS_PER_TYPE
            or float(diagnostic.get("anchor_separation_ratio", 0.0))
            < FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO
            or not isinstance(diagnostic.get("component_count_candidates"), list)
            or len(diagnostic.get("component_count_candidates", ()))
            != FROZEN_MAX_STATE_COMPONENTS_PER_TYPE - FROZEN_MIN_STATE_COMPONENTS_PER_TYPE + 1
        ):
            raise ValueError(f"fixed-reference anchor separation changed for {donor}")
        groups = list(diagnostic.get("natural_support", ())) + list(
            diagnostic.get("joint_support", ())
        )
        if not groups or any(
            int(group.get("cells", 0)) < FROZEN_STATE_SUPPORT_CELLS
            or float(group.get("ESS", 0.0)) < FROZEN_STATE_SUPPORT_ESS - 1.0e-9
            or float(group.get("proximity_quantile_ratio", np.inf))
            > FROZEN_STATE_MAX_PROXIMITY_RATIO
            or (
                group.get("stage") == "pre_balance"
                and float(group.get("bank_component_separation_ratio", 0.0))
                < FROZEN_STATE_MIN_ANCHOR_SEPARATION_RATIO
            )
            or (
                group.get("stage") == "joint_QC_state_support"
                and float(group.get("type_retention_fraction", 0.0))
                < FROZEN_STATE_MIN_JOINT_RETENTION
            )
            for group in groups
        ):
            raise ValueError(f"fixed-reference natural state support changed for {donor}")
    for component, diagnostic in enumerate(component_diagnostics):
        if (
            int(diagnostic.get("component_index", -1)) != component
            or diagnostic.get("Level1_type") != component_types[component]
            or not np.isclose(
                float(diagnostic.get("global_mass", np.nan)),
                component_masses[component],
            )
            or not np.isclose(
                float(diagnostic.get("target_component_ESS", np.nan)),
                component_target_ess[component],
            )
        ):
            raise ValueError(f"fixed-reference component diagnostic changed for {donor}")
        fits = diagnostic.get("weight_fits")
        if (
            not isinstance(fits, list)
            or len(fits) != len((donor, *wrong_donors)) + len(wrong_donors)
            or any(
                not 0 <= float(value.get("tilt", np.nan)) <= FROZEN_STATE_MAX_TILT
                or float(value.get("minimum_relative_weight", 0.0))
                < FROZEN_STATE_MIN_RELATIVE_WEIGHT
                or not 0 < float(value.get("maximum_conditional_weight_share", np.nan)) <= 1
                or not 0 < float(value.get("effective_cell_fraction", np.nan)) <= 1
                or float(value.get("weighted_mean_shift_ratio", np.inf))
                > FROZEN_STATE_MAX_WEIGHTED_MEAN_SHIFT_RATIO
                or not 0
                <= float(value.get("weighted_to_maximum_entropy_variance_trace_ratio", np.nan))
                < np.inf
                for value in fits
            )
        ):
            raise ValueError(f"fixed-reference proximity weighting changed for {donor}")
    expected_calls = [
        _adapter_call_receipt(
            plan,
            recomputed_state,
            mode="single_bank",
            donor_ids=(bank,),
        )
        for bank in (donor, *wrong_donors)
    ] + [
        _adapter_call_receipt(
            plan,
            recomputed_state,
            mode="generic_donor_equal",
            donor_ids=wrong_donors,
        )
    ]

    def call_key(value: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
        return (
            str(value.get("mode")),
            tuple(sorted(str(bank) for bank in value.get("donor_ids", ()))),
        )

    observed_by_key = {call_key(value): value for value in calls}
    expected_by_key = {call_key(value): value for value in expected_calls}
    if (
        len(calls) != wrong_count + 2
        or len(observed_by_key) != len(calls)
        or observed_by_key != expected_by_key
    ):
        raise ValueError(f"fixed-reference exact adapter call receipts changed for {donor}")


def fit_predict(args: argparse.Namespace) -> Mapping[str, object]:
    prepared = _read_prepared(args.output, args.protocol)
    baseline = _load_baseline_runner()
    core = baseline._import_core()
    if _sha256(BASELINE_RUNNER) != EXPECTED_HASHES["development_runner_sha256"]:
        raise ValueError("frozen development runner changed before fit-predict")
    if _sha256(Path(core.__file__).resolve()) != EXPECTED_HASHES["generative_core_sha256"]:
        raise ValueError("frozen generative core changed before fit-predict")
    receipts: dict[str, object] = {}
    manifest_path = args.output / "fit_predict_manifest.json"
    for donor_value in prepared["donors"]:
        donor = str(donor_value)
        fold = prepared["folds"][donor]
        public = _verify_semantic(
            Path(str(fold["public_path"])), str(fold["public_semantic_sha256"]), baseline
        )
        baseline.validate_public_fold(public)
        plan = _verify_semantic(
            Path(str(fold["plan_path"])), str(fold["plan_semantic_sha256"]), baseline
        )
        _validate_weight_plan(plan, donor=donor)
        fold_seed = int(baseline._fold_seed(FROZEN_BASE_SEED, donor))
        identity = _checkpoint_identity(
            donor=donor,
            public_identity=str(fold["public_semantic_sha256"]),
            plan_identity=str(fold["plan_semantic_sha256"]),
            fold_seed=fold_seed,
            args=args,
            prepared=prepared,
        )
        fold_dir = args.output / "folds" / donor
        prediction_path = fold_dir / "predictions.npz"
        receipt_path = fold_dir / "fit_predict_receipt.json"
        if args.resume and prediction_path.is_file() and receipt_path.is_file():
            old = json.loads(receipt_path.read_text(encoding="utf-8"))
            if old.get("checkpoint_identity") == identity:
                predictions = _load_arrays(prediction_path)
                if _semantic_hash(predictions, baseline) == old.get("prediction_semantic_sha256"):
                    _validate_fixed_prediction(
                        baseline=baseline,
                        predictions=predictions,
                        public=public,
                        plan=plan,
                        donor=donor,
                        epochs=args.epochs,
                    )
                    receipts[donor] = old
                    continue

        capture = _AlignedLatentCapture(public, plan)
        adapter = _WeightedReferenceAdapter(core, plan, capture.plan_latent)
        original_builder = core.build_reference_mixture
        original_encode = baseline._encode
        core.build_reference_mixture = adapter
        baseline._encode = capture.wrap(original_encode)
        try:
            predictions = dict(
                baseline.fit_predict_one_fold(
                    public,
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    latent_dim=args.latent_dim,
                    seed=fold_seed,
                )
            )
        finally:
            baseline._encode = original_encode
            core.build_reference_mixture = original_builder
        if (
            core.build_reference_mixture is not original_builder
            or baseline._encode is not original_encode
        ):
            raise RuntimeError("fixed-reference adapter/encoder hook was not restored")
        capture_receipt = capture.receipt()
        authority_receipt = adapter.authority_receipt()
        if adapter.state_authority is None:
            raise RuntimeError("natural-state authority was not materialized")
        _promote_gate3_support_receipt_precision(predictions)
        predictions.update(
            {
                "fixed_reference_plan_semantic_sha256": np.asarray(
                    str(fold["plan_semantic_sha256"])
                ),
                "fixed_reference_QC_common_type_names": np.asarray(plan["common_type_names"]),
                "fixed_reference_common_type_names": np.asarray(
                    adapter.state_authority["evaluable_type_names"]
                ),
                "fixed_reference_single_bank_cell_weight_ESS": np.asarray(
                    adapter.state_authority["single_bank_target_ess"], dtype=np.float64
                ),
                "fixed_reference_generic_pooled_cell_weight_ESS": np.asarray(
                    adapter.state_authority["generic_pooled_target_ess"],
                    dtype=np.float64,
                ),
                "fixed_reference_hard_subsampling_used": np.asarray(False),
                "fixed_reference_support_restriction_used": np.asarray(
                    bool(adapter.state_authority["support_restriction_used"])
                ),
                "fixed_reference_training_distribution": np.asarray(
                    "natural_unchanged_prediction_time_reference_adapter_only"
                ),
                "fixed_reference_adapter_calls_json": np.asarray(
                    json.dumps(adapter.calls, sort_keys=True, separators=(",", ":"))
                ),
                "fixed_reference_state_capture_json": np.asarray(
                    json.dumps(capture_receipt, sort_keys=True, separators=(",", ":"))
                ),
                "fixed_reference_state_authority_json": np.asarray(
                    json.dumps(authority_receipt, sort_keys=True, separators=(",", ":"))
                ),
                **adapter.authority_arrays(),
            }
        )
        _validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor=donor,
            epochs=args.epochs,
        )
        _atomic_npz(prediction_path, predictions)
        receipt = {
            "schema": PREDICTION_SCHEMA,
            "heldout_donor": donor,
            "checkpoint_identity": identity,
            "public_semantic_sha256": str(fold["public_semantic_sha256"]),
            "plan_semantic_sha256": str(fold["plan_semantic_sha256"]),
            "prediction_path": str(prediction_path),
            "prediction_semantic_sha256": _semantic_hash(predictions, baseline),
            "score_target_opened": False,
            "training_distribution": "natural_unchanged",
            "reference_intervention": "prediction_time_only",
            "hard_subsampling_used": False,
            "support_restriction_used": bool(adapter.state_authority["support_restriction_used"]),
            "evaluable_type_names": list(adapter.state_authority["evaluable_type_names"]),
            "conditional_evaluable_type_fraction": float(
                adapter.state_authority["conditional_evaluable_type_fraction"]
            ),
            "global_evaluable_type_fraction": float(
                adapter.state_authority["global_evaluable_type_fraction"]
            ),
            "support_eligibility_by_bank": dict(
                adapter.state_authority["support_eligibility_by_bank"]
            ),
            "state_anchor_sha256": str(adapter.state_authority["state_anchor_sha256"]),
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "sensitivity_runner_sha256": _sha256(Path(__file__).resolve()),
            "development_runner_sha256": EXPECTED_HASHES["development_runner_sha256"],
            "generative_core_sha256": EXPECTED_HASHES["generative_core_sha256"],
            "protocol_sha256": str(prepared["protocol_sha256"]),
            "prepared_manifest_semantic_sha256": hashlib.sha256(_json_bytes(prepared)).hexdigest(),
            "artifact_complete": True,
        }
        _atomic_json(receipt_path, receipt)
        receipts[donor] = receipt
        progress = {
            "schema": PREDICTION_SCHEMA,
            "analysis_scope": "validation_only_outcome_exposed_non_confirmatory",
            "folds": receipts,
            "all_folds_complete": len(receipts) == len(prepared["donors"]),
            "score_target_opened": False,
        }
        _atomic_json(manifest_path, progress)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        _assert_zero_process_swap()
    aggregate = {
        "schema": PREDICTION_SCHEMA,
        "analysis_scope": "validation_only_outcome_exposed_non_confirmatory",
        "folds": receipts,
        "all_folds_complete": len(receipts) == len(prepared["donors"]),
        "score_target_opened": False,
    }
    _atomic_json(manifest_path, aggregate)
    return aggregate


def _global_prediction_preflight(
    args: argparse.Namespace,
    prepared: Mapping[str, object],
    baseline: Any,
) -> dict[str, object]:
    """Validate all public/plan/prediction artifacts before any target is opened."""

    path = args.output / "fit_predict_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    donors = tuple(str(value) for value in prepared["donors"])
    if (
        manifest.get("schema") != PREDICTION_SCHEMA
        or manifest.get("all_folds_complete") is not True
        or set(manifest.get("folds", {})) != set(donors)
    ):
        raise ValueError("fixed-ESS prediction manifest is incomplete")
    for donor in donors:
        fold = prepared["folds"][donor]
        receipt = manifest["folds"][donor]
        fold_seed = int(baseline._fold_seed(FROZEN_BASE_SEED, donor))
        checkpoint_identity = _checkpoint_identity(
            donor=donor,
            public_identity=str(fold["public_semantic_sha256"]),
            plan_identity=str(fold["plan_semantic_sha256"]),
            fold_seed=fold_seed,
            args=args,
            prepared=prepared,
        )
        required = {
            "schema": PREDICTION_SCHEMA,
            "heldout_donor": donor,
            "public_semantic_sha256": str(fold["public_semantic_sha256"]),
            "plan_semantic_sha256": str(fold["plan_semantic_sha256"]),
            "checkpoint_identity": checkpoint_identity,
            "sensitivity_runner_sha256": _sha256(Path(__file__).resolve()),
            "development_runner_sha256": EXPECTED_HASHES["development_runner_sha256"],
            "generative_core_sha256": EXPECTED_HASHES["generative_core_sha256"],
            "protocol_sha256": str(prepared["protocol_sha256"]),
            "prepared_manifest_semantic_sha256": hashlib.sha256(_json_bytes(prepared)).hexdigest(),
            "artifact_complete": True,
            "score_target_opened": False,
        }
        mismatched = [name for name, value in required.items() if receipt.get(name) != value]
        if mismatched:
            raise ValueError(f"stale fixed-ESS receipt for {donor}: {mismatched}")
        public = _verify_semantic(
            Path(str(fold["public_path"])), str(fold["public_semantic_sha256"]), baseline
        )
        plan = _verify_semantic(
            Path(str(fold["plan_path"])), str(fold["plan_semantic_sha256"]), baseline
        )
        predictions = _verify_semantic(
            Path(str(receipt["prediction_path"])),
            str(receipt["prediction_semantic_sha256"]),
            baseline,
        )
        baseline.validate_public_fold(public)
        _validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor=donor,
            epochs=args.epochs,
        )
        # Do not retain all folds' 20D reference latents in memory.  The score
        # stage reloads and immediately revalidates one fold at a time.
        del public, plan, predictions
    return manifest


def _validate_target(
    baseline: Any,
    secret: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    donor: str,
) -> None:
    if _scalar_text(secret.get("schema")) != baseline.PREPARED_SCHEMA:
        raise ValueError(f"score target schema differs for {donor}")
    if _scalar_text(secret.get("heldout_donor")) != donor:
        raise ValueError(f"score target donor differs for {donor}")
    aligned = {
        "heldout_spot_ids": "query_spot_ids",
        "heldout_section_ids": "query_section_ids",
        "heldout_indication_ids": "query_indication_ids",
    }
    for secret_name, public_name in aligned.items():
        if not np.array_equal(np.asarray(secret[secret_name]), np.asarray(public[public_name])):
            raise ValueError(f"score target {secret_name} differs for {donor}")
    if not np.array_equal(
        np.asarray(secret["heldout_spot_ids"]), np.asarray(predictions["query_spot_ids"])
    ):
        raise ValueError(f"prediction rows differ from score target for {donor}")
    rows, genes = len(public["query_spot_ids"]), len(public["gene_ids"])
    counts = np.asarray(secret["heldout_st_counts"])
    library = np.asarray(secret["heldout_st_library"])
    eligibility = np.asarray(secret["primary_score_eligible"])
    if counts.shape != (rows, genes):
        raise ValueError(f"score target count shape differs for {donor}")
    if (
        not np.issubdtype(counts.dtype, np.number)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.array_equal(counts, np.floor(counts))
    ):
        raise ValueError(f"score target counts are invalid for {donor}")
    if library.shape != (rows,) or not np.isfinite(library).all() or np.any(library < 0):
        raise ValueError(f"score target library is invalid for {donor}")
    if eligibility.shape != (rows,) or eligibility.dtype != np.dtype(bool):
        raise ValueError(f"score eligibility shape differs for {donor}")
    if not np.any(eligibility) or np.any(library[eligibility] <= 0):
        raise ValueError(f"score eligibility/exposure is invalid for {donor}")


def _score_fixed_fold(
    baseline: Any,
    core: Any,
    secret: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    donor: str,
) -> dict[str, object]:
    _validate_target(baseline, secret, public, predictions, donor=donor)
    state_authority = json.loads(_scalar_text(predictions["fixed_reference_state_authority_json"]))
    keep = np.asarray(secret["primary_score_eligible"], dtype=bool)
    counts = np.asarray(secret["heldout_st_counts"], dtype=np.float32)[keep]
    library = np.asarray(secret["heldout_st_library"], dtype=np.float32)[keep]
    sections = np.asarray(secret["heldout_section_ids"]).astype(str)[keep]
    theta = np.asarray(predictions["training_only_dispersion"], dtype=np.float32)

    def loss(rate: np.ndarray) -> tuple[float, np.ndarray]:
        mean = np.asarray(rate, dtype=np.float32)[keep] * library[:, None]
        rows = baseline._nb_deviance_rows(core, counts, mean, theta)
        return float(baseline._section_macro(rows, sections)), rows

    m3, m3_rows = loss(np.asarray(predictions["rate_M3"]))
    m7, m7_rows = loss(np.asarray(predictions["rate_M7"]))
    candidate_losses = []
    candidate_rows = []
    for rate in np.asarray(predictions["rate_M6_candidates"]):
        value, rows = loss(rate)
        candidate_losses.append(value)
        candidate_rows.append(rows)
    m6_rows = np.mean(np.stack(candidate_rows), axis=0)
    m6 = float(baseline._section_macro(m6_rows, sections))
    if not np.isclose(m6, np.mean(candidate_losses), rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError("equal-mean wrong-donor loss aggregation is inconsistent")
    indications = set(np.asarray(public["query_indication_ids"]).astype(str).tolist())
    if len(indications) != 1:
        raise ValueError(f"query donor spans indications for {donor}")
    return {
        "schema": SCHEMA,
        "heldout_donor": donor,
        "indication": next(iter(indications)),
        "eligible_spots": int(keep.sum()),
        "sections": sorted(set(sections.tolist())),
        "aggregation": "spots_within_section_then_sections_within_donor",
        "mean_NB_deviance": {
            "fixed_M3": m3,
            "equal_mean_wrong_M6": m6,
            "fixed_generic_M7": m7,
        },
        "wrong_donor_ids": np.asarray(predictions["wrong_donor_ids"]).astype(str).tolist(),
        "wrong_candidate_mean_NB_deviance": candidate_losses,
        "paired_improvement": {
            "fixed_M3_vs_equal_mean_wrong_M6": m6 - m3,
            "fixed_M3_vs_fixed_generic_M7": m7 - m3,
        },
        "row_loss_identity_check": {
            "M3_rows": int(len(m3_rows)),
            "M6_rows": int(len(m6_rows)),
            "M7_rows": int(len(m7_rows)),
        },
        "state_balance": {
            "anchor_training_donor_ids": state_authority["anchor_training_donor_ids"],
            "heldout_donor_excluded_from_anchor_fit": state_authority[
                "heldout_donor_excluded_from_anchor_fit"
            ],
            "evaluable_type_names": state_authority["evaluable_type_names"],
            "all_observed_type_names": state_authority["all_observed_type_names"],
            "conditional_evaluable_type_fraction": state_authority[
                "conditional_evaluable_type_fraction"
            ],
            "global_evaluable_type_fraction": state_authority["global_evaluable_type_fraction"],
            "initial_type_support_diagnostics": state_authority["initial_type_support_diagnostics"],
            "unevaluable_types": state_authority["unevaluable_types"],
            "component_count_selection": state_authority["component_count_selection"],
            "single_bank_target_ESS": state_authority["single_bank_target_ESS"],
            "generic_pooled_target_ESS": state_authority["generic_pooled_target_ESS"],
            "support_restriction_used": state_authority["support_restriction_used"],
            "support_eligibility_by_bank": state_authority["support_eligibility_by_bank"],
            "type_diagnostics": state_authority["type_diagnostics"],
            "component_diagnostics": state_authority["component_diagnostics"],
        },
    }


def score(args: argparse.Namespace) -> Mapping[str, object]:
    prepared = _read_prepared(args.output, args.protocol)
    baseline = _load_baseline_runner()
    core = baseline._import_core()
    prediction_manifest = _global_prediction_preflight(args, prepared, baseline)
    _verify_outcome_bearing_baseline_identities(args.baseline_output)
    score_targets = _read_score_target_manifest(args.output, prepared)
    # No score target has been opened above this line.
    fold_reports: dict[str, object] = {}
    for donor_value in prepared["donors"]:
        donor = str(donor_value)
        target_fold = score_targets["folds"][donor]
        prepared_fold = prepared["folds"][donor]
        prediction_receipt = prediction_manifest["folds"][donor]
        public = _verify_semantic(
            Path(str(prepared_fold["public_path"])),
            str(prepared_fold["public_semantic_sha256"]),
            baseline,
        )
        plan = _verify_semantic(
            Path(str(prepared_fold["plan_path"])),
            str(prepared_fold["plan_semantic_sha256"]),
            baseline,
        )
        predictions = _verify_semantic(
            Path(str(prediction_receipt["prediction_path"])),
            str(prediction_receipt["prediction_semantic_sha256"]),
            baseline,
        )
        baseline.validate_public_fold(public)
        # Close the mutation window immediately before opening this target.
        _validate_fixed_prediction(
            baseline=baseline,
            predictions=predictions,
            public=public,
            plan=plan,
            donor=donor,
            epochs=args.epochs,
        )
        secret = _verify_semantic(
            Path(str(target_fold["score_target_path"])),
            str(target_fold["score_target_semantic_sha256"]),
            baseline,
        )
        report = _score_fixed_fold(baseline, core, secret, public, predictions, donor=donor)
        fold_reports[donor] = report
        _atomic_json(args.output / "folds" / donor / "score_report.json", report)

    names = (
        "fixed_M3_vs_equal_mean_wrong_M6",
        "fixed_M3_vs_fixed_generic_M7",
    )
    comparisons: dict[str, object] = {}
    raw_p_values: dict[str, float] = {}
    for name in names:
        effects = np.asarray(
            [fold_reports[donor]["paired_improvement"][name] for donor in prepared["donors"]],
            dtype=np.float64,
        )
        sign_flip = core.exact_sign_flip_test(effects, alternative="greater")
        raw_p_values[name] = float(sign_flip.p_value)
        comparisons[name] = {
            "positive_effect_favors": "fixed_M3",
            "donor_effects": effects.tolist(),
            "mean_improvement": float(np.mean(effects)),
            "median_improvement": float(np.median(effects)),
            "positive_donor_fraction": float(np.mean(effects > 0)),
            "exact_one_sided_sign_flip": dataclasses.asdict(sign_flip),
            "by_indication_mean_improvement": {
                indication: float(
                    np.mean(
                        [
                            fold_reports[donor]["paired_improvement"][name]
                            for donor in prepared["donors"]
                            if fold_reports[donor]["indication"] == indication
                        ]
                    )
                )
                for indication in sorted(
                    {fold_reports[donor]["indication"] for donor in prepared["donors"]}
                )
            },
        }
    adjusted = core.holm_adjust(raw_p_values)
    for name in names:
        comparisons[name]["holm_adjusted_p_value"] = float(adjusted[name])

    report = {
        "schema": SCHEMA,
        "analysis_scope": "validation_only_outcome_exposed_non_confirmatory",
        "evidence_status": ("fixed_reference_natural_state_joint_QC_componentwise_ESS_sensitivity"),
        "can_confirm_scientific_hypothesis": False,
        "image_encoder": HOPTIMUS_REPOSITORY,
        "image_encoder_revision": HOPTIMUS_REVISION,
        "uni2_h_run": False,
        "model_training": "natural_unchanged",
        "reference_intervention": (
            "prediction_time_query_excluded_training_anchors_natural_state_support_"
            "joint_QC_state_mass_and_exact_componentwise_ESS"
        ),
        "hard_subsampling_used": False,
        "support_restriction_used": any(
            fold_reports[donor]["state_balance"]["support_restriction_used"]
            for donor in prepared["donors"]
        ),
        "primary_endpoint": "donor_balanced_heldout_negative_binomial_deviance",
        "global_prediction_preflight_before_any_target": True,
        "artifact_identities": {
            **prepared["baseline_artifact_identities"],
            "fixed_ess_protocol_sha256": str(prepared["protocol_sha256"]),
            "fixed_ess_runner_sha256": _sha256(Path(__file__).resolve()),
            "fixed_ess_prepared_manifest_sha256": _sha256(args.output / "prepared_manifest.json"),
            "fixed_ess_fit_predict_manifest_sha256": _sha256(
                args.output / "fit_predict_manifest.json"
            ),
            "fixed_ess_score_target_manifest_sha256": _sha256(
                args.output / "score_target_manifest.json"
            ),
        },
        "comparisons": comparisons,
        "folds": fold_reports,
        "state_balance_coverage": {
            donor: {
                "evaluable_type_names": fold_reports[donor]["state_balance"][
                    "evaluable_type_names"
                ],
                "unevaluable_types": fold_reports[donor]["state_balance"]["unevaluable_types"],
                "all_observed_type_names": fold_reports[donor]["state_balance"][
                    "all_observed_type_names"
                ],
                "conditional_evaluable_type_fraction": fold_reports[donor]["state_balance"][
                    "conditional_evaluable_type_fraction"
                ],
                "global_evaluable_type_fraction": fold_reports[donor]["state_balance"][
                    "global_evaluable_type_fraction"
                ],
                "initial_type_support_diagnostics": fold_reports[donor]["state_balance"][
                    "initial_type_support_diagnostics"
                ],
                "support_eligibility_by_bank": fold_reports[donor]["state_balance"][
                    "support_eligibility_by_bank"
                ],
            }
            for donor in prepared["donors"]
        },
        "interpretation_boundary": (
            "positive_results_support_reference_personalization_only_for_naturally_"
            "supported_types_and_states_within_the_exposed_NatCommun_development_cohort;_"
            "component_identity_richness_within_type_state_mass_joint_QC_distribution_"
            "componentwise_ESS_"
            "and_total_ESS_are_matched_but_donor_specific_within_component_geometry_"
            "natural_training_asymmetry_and_donor_constant_DV200_age_remain"
        ),
        "limitations": [
            "NatCommun is outcome-exposed development data and is not independent confirmation",
            "the registered suspension reference is annotated as cell and is not verified snRNA",
            "DV200 and block age are donor-constant and cannot be balanced within donor banks",
            "percent_ribo is zero and cannot define a quality stratum",
            "training remains naturally imbalanced; only prediction-time banks are balanced",
            (
                "anchors are fitted from query-excluded training references; matched reference "
                "cells affect only target-free natural-support eligibility and balance targets"
            ),
            (
                "this is a conditional common-support estimand: explicit type/QC/state support "
                "restrictions, not random subsampling, give cells outside support zero weight; "
                "counts and reasons are reported per bank and global cell/type coverage must "
                "remain at least 50 percent"
            ),
            (
                "component identity, richness, within-type state mass, joint QC distribution, "
                "componentwise ESS, and total ESS are matched; donor-specific within-component "
                "locations and covariance remain candidate biological signal"
            ),
            (
                "positive proximity weights are bounded and their minimum relative weight, "
                "maximum share, effective-cell fraction, tilt, weighted-mean displacement, "
                "and variance-trace ratio are reported for every component"
            ),
            "regional Visium spots do not authorize cell-level claims",
        ],
        "prediction_manifest_all_folds_complete": bool(prediction_manifest["all_folds_complete"]),
    }
    _atomic_json(args.output / "report.json", report)
    return report


def _assert_zero_process_swap() -> None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmSwap:") and int(line.split()[1]) != 0:
            raise RuntimeError("process swap usage is non-zero; aborting under frozen resources")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "fit-predict", "score"),
        default="prepare",
        help="run one physical leakage-boundary stage per process",
    )
    parser.add_argument("--baseline-output", type=Path, default=DEFAULT_BASELINE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.60)
    parser.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--latent-dim", type=int, default=FROZEN_LATENT_DIM)
    parser.add_argument("--seed", type=int, default=FROZEN_BASE_SEED)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError("CPU threads must be between one and four")
    if not 0 < args.gpu_memory_fraction <= 0.60:
        raise ValueError("GPU memory fraction must be in (0, 0.60]")
    observed = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
    }
    expected = {
        "seed": FROZEN_BASE_SEED,
        "epochs": FROZEN_EPOCHS,
        "batch_size": FROZEN_BATCH_SIZE,
        "latent_dim": FROZEN_LATENT_DIM,
    }
    mismatched = [name for name in expected if observed[name] != expected[name]]
    if mismatched:
        raise ValueError(f"arguments differ from the frozen model: {mismatched}")
    normalized = str(args.device).casefold().replace("-", "").replace("_", "")
    if "uni2" in normalized:
        raise ValueError("UNI2-h is explicitly prohibited")
    if args.stage == "fit-predict" and args.device != "cuda:0":
        raise ValueError("real fixed-ESS fitting requires the single bounded cuda:0 device")
    if args.stage == "fit-predict":
        visible = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        if len(visible) != 1:
            raise ValueError("fit-predict requires exactly one CUDA_VISIBLE_DEVICES entry")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    baseline = _load_baseline_runner()
    resource_device = args.device if args.stage == "fit-predict" else "cpu"
    baseline.configure_resources(
        cpu_threads=args.cpu_threads,
        gpu_memory_fraction=args.gpu_memory_fraction,
        device=resource_device,
    )
    baseline.seed_everything(args.seed)
    _assert_zero_process_swap()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "fit-predict":
        fit_predict(args)
    else:
        score(args)
    _assert_zero_process_swap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
