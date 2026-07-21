"""Resumable observed and fully refitted-null execution for HEST v2."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from heir.evaluation.neural_nulls import build_neural_null_design
from heir.evaluation.nonlinear_qualification_contract import canonical_sha256, file_sha256
from heir.evaluation.nonlinear_qualification_v2_acceptance import (
    NULL_FAMILIES as ACCEPTANCE_NULL_FAMILIES,
)
from heir.evaluation.nonlinear_qualification_v2_acceptance import (
    ArmPrediction,
    QualificationInputs,
    RegistrationComparison,
    score_qualification,
)
from heir.evaluation.nonlinear_qualification_v2_contract import (
    ANALYSIS_STATUS,
    SELECTED_DONORS,
    SOURCE_SHA256,
    V2_PROTOCOL_SHA256,
)
from heir.evaluation.nonlinear_qualification_v2_data import (
    HestBiologicalData,
)
from heir.evaluation.nonlinear_qualification_v2_execution import (
    atomic_json,
    validate_biological_authorization,
)
from heir.evaluation.nonlinear_qualification_v2_models import (
    FROZEN_TRAINING_GRID,
    REGISTRATION_PAIRS,
    OuterFoldModels,
    TrainingGrid,
    fit_null_outer_fold,
    fit_observed_outer_fold,
    validate_frozen_training_grid,
)

REPORT_SCHEMA = "heir.hest_nonlinear_qualification_report.v2"
OBSERVED_FOLD_SCHEMA = "heir.hest_nonlinear_v2_observed_fold.v1"
NULL_FOLD_SCHEMA = "heir.hest_nonlinear_v2_null_fold.v1"
NULL_PERMUTATION_SCHEMA = "heir.hest_nonlinear_v2_null_permutation.v1"
FULL_RUN_SCHEMA = "heir.hest_nonlinear_v2_full_execution.v1"
MAX_COORDINATE_RANK = 6
FINAL_NULL_PERMUTATIONS = 100
SMOKE_NULL_PERMUTATIONS = 20
SCORED_ARMS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "N0",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "C_blank",
    "C_blank_single",
    "A_star",
)
NEURAL_REFIT_ARMS = (
    "B3",
    "N0",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "C_blank",
    "C_blank_single",
)
CONTROL_ALIASES = {"C_target_removed": "N5"}
NULL_SEEDS = {
    "within_section_type_derangement": 20_260_720,
    "different_spatial_block_reassignment": 21_260_723,
}
MINIMUM_DIFFERENT_BLOCK_CHANGED_FRACTION = 0.95
NULL_P_NAMES = {
    "within_section_type_derangement": "within_section_type_refitted_null_empirical_p",
    "different_spatial_block_reassignment": ("different_spatial_block_refitted_null_empirical_p"),
}
COMPLETE_REFIT_STEPS = (
    "feature_preprocessing",
    "technical_correction",
    "molecular_target_fitting",
    "rank_selection",
    "neural_architecture_selection",
    "training",
    "checkpoint_selection",
    "prediction",
    "scoring",
)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _json(path: Path) -> dict[str, object]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a readable JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def full_run_binding(
    authorization: Mapping[str, object],
    *,
    grid: TrainingGrid = FROZEN_TRAINING_GRID,
    device: str = "cuda",
) -> Mapping[str, object]:
    validate_frozen_training_grid(grid)
    validate_biological_authorization(authorization, device=device)
    value = {
        "schema": FULL_RUN_SCHEMA,
        "protocol_sha256": V2_PROTOCOL_SHA256,
        "registered_source_sha256": SOURCE_SHA256,
        "authorization_sha256": canonical_sha256(authorization),
        "git_commit": authorization["git_commit"],
        "implementation": authorization["implementation"],
        "device": device,
        "training_grid": asdict(grid),
        "scored_arms": list(SCORED_ARMS),
        "control_aliases": dict(CONTROL_ALIASES),
        "registration_pairs": dict(REGISTRATION_PAIRS),
        "null_seeds": dict(NULL_SEEDS),
        "smoke_null_permutations": SMOKE_NULL_PERMUTATIONS,
        "final_null_permutations": FINAL_NULL_PERMUTATIONS,
        "complete_refit_steps": list(COMPLETE_REFIT_STEPS),
    }
    return {**value, "sha256": canonical_sha256(value)}


def _pad(value: np.ndarray, ranks: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    rank_values = np.asarray(ranks, dtype=np.int64)
    if (
        array.ndim != 2
        or rank_values.shape != (len(array),)
        or array.shape[1] > MAX_COORDINATE_RANK
        or np.any(rank_values != array.shape[1])
    ):
        raise ValueError("outer-fold coordinate array or rank is malformed")
    result = np.full((len(array), MAX_COORDINATE_RANK), fill, dtype=np.float64)
    result[:, : array.shape[1]] = array
    return result


def observed_fold_arrays(result: OuterFoldModels) -> Mapping[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "row_indices": np.asarray(result.row_indices, dtype=np.int64),
        "corrected_molecular_truth": np.asarray(result.corrected_molecular_truth, dtype=np.float64),
        "reference_prediction": np.asarray(result.reference_prediction, dtype=np.float64),
    }
    for arm_id in SCORED_ARMS:
        arm = result.arms[arm_id]
        prefix = f"arm__{arm_id}__"
        arrays[prefix + "coordinate_truth"] = _pad(
            arm.coordinate_truth,
            arm.coordinate_ranks,
        )
        arrays[prefix + "coordinate_prediction"] = _pad(
            arm.coordinate_prediction,
            arm.coordinate_ranks,
        )
        arrays[prefix + "coordinate_ranks"] = np.asarray(arm.coordinate_ranks, dtype=np.int64)
        arrays[prefix + "molecular_prediction"] = np.asarray(
            arm.molecular_prediction, dtype=np.float64
        )
        arrays[prefix + "rare_lower"] = _pad(
            arm.rare_lower,
            arm.coordinate_ranks,
            fill=-1.0,
        )
        arrays[prefix + "rare_upper"] = _pad(
            arm.rare_upper,
            arm.coordinate_ranks,
            fill=1.0,
        )
    for comparison_id, comparison in result.registration.items():
        prefix = f"registration__{comparison_id}__"
        arrays[prefix + "truth"] = _pad(
            comparison.truth,
            comparison.coordinate_ranks,
        )
        arrays[prefix + "focal_prediction"] = _pad(
            comparison.focal_prediction,
            comparison.coordinate_ranks,
        )
        arrays[prefix + "comparator_prediction"] = _pad(
            comparison.comparator_prediction,
            comparison.coordinate_ranks,
        )
        arrays[prefix + "coordinate_ranks"] = np.asarray(
            comparison.coordinate_ranks, dtype=np.int64
        )
    return arrays


def _validate_observed_fold(
    receipt_path: Path,
    *,
    heldout_donor: str,
    binding_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, np.ndarray]]:
    receipt = _json(receipt_path)
    artifact_path = Path(str(receipt.get("artifact_path", ""))).resolve()
    outer_refits = receipt.get("neural_outer_refits")
    if (
        not artifact_path.is_file()
        or receipt.get("schema") != OBSERVED_FOLD_SCHEMA
        or receipt.get("heldout_donor") != heldout_donor
        or receipt.get("binding_sha256") != binding_sha256
        or receipt.get("artifact_sha256") != file_sha256(artifact_path)
        or receipt.get("molecular_outcomes_read") is not True
        or receipt.get("authorizes_h_cell") is not False
        or not isinstance(outer_refits, Mapping)
        or set(outer_refits) != set(NEURAL_REFIT_ARMS)
        or any(not isinstance(rows, list) or not rows for rows in outer_refits.values())
    ):
        raise ValueError("observed outer-fold checkpoint identity is invalid")
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"row_indices", "corrected_molecular_truth", "reference_prediction"}
    for arm_id in SCORED_ARMS:
        required.update(
            {
                f"arm__{arm_id}__coordinate_truth",
                f"arm__{arm_id}__coordinate_prediction",
                f"arm__{arm_id}__coordinate_ranks",
                f"arm__{arm_id}__molecular_prediction",
                f"arm__{arm_id}__rare_lower",
                f"arm__{arm_id}__rare_upper",
            }
        )
    for comparison_id in REGISTRATION_PAIRS:
        required.update(
            {
                f"registration__{comparison_id}__truth",
                f"registration__{comparison_id}__focal_prediction",
                f"registration__{comparison_id}__comparator_prediction",
                f"registration__{comparison_id}__coordinate_ranks",
            }
        )
    if set(arrays) != required:
        raise ValueError("observed outer-fold checkpoint array registry changed")
    return receipt, arrays


def run_observed_folds(
    data: HestBiologicalData,
    work_dir: Path,
    binding: Mapping[str, object],
    *,
    grid: TrainingGrid = FROZEN_TRAINING_GRID,
    device: str = "cuda",
    fold_runner: Callable[..., OuterFoldModels] = fit_observed_outer_fold,
) -> tuple[Mapping[str, object], ...]:
    donors = tuple(sorted(set(data.evaluation_donors.tolist())))
    if grid == FROZEN_TRAINING_GRID and len(donors) != 15:
        raise ValueError("production observed execution requires all 15 HEST donors")
    identity = np.arange(len(data.observation_ids), dtype=np.int64)
    receipts = []
    for heldout in donors:
        fold_dir = work_dir / "observed" / "folds" / heldout
        receipt_path = fold_dir / "receipt.json"
        if receipt_path.exists():
            receipt, _ = _validate_observed_fold(
                receipt_path,
                heldout_donor=heldout,
                binding_sha256=str(binding["sha256"]),
            )
            receipts.append(receipt)
            continue
        result = fold_runner(data, heldout, identity, grid=grid, device=device)
        arrays = observed_fold_arrays(result)
        artifact_path = (fold_dir / "predictions.npz").resolve()
        _atomic_npz(artifact_path, arrays)
        receipt = {
            "schema": OBSERVED_FOLD_SCHEMA,
            "analysis_status": ANALYSIS_STATUS,
            "binding_sha256": binding["sha256"],
            "heldout_donor": heldout,
            "rows": len(result.row_indices),
            "artifact_path": str(artifact_path),
            "artifact_sha256": file_sha256(artifact_path),
            "selected_primary_arm": result.selected_primary_arm,
            "selected_single_arm": result.selected_single_arm,
            "selections": result.selections,
            "arm_receipts_sha256": {
                arm_id: canonical_sha256(result.arms[arm_id].receipt) for arm_id in SCORED_ARMS
            },
            "neural_outer_refits": {
                arm_id: result.arms[arm_id].receipt["outer_refit"] for arm_id in NEURAL_REFIT_ARMS
            },
            "molecular_outcomes_read": True,
            "biological_experiment_run": True,
            "engineering_decision_available": False,
            "authorizes_h_cell": False,
            "authorizes_h_intrinsic": False,
            "authorizes_h_ref": False,
            "authorizes_full_heir": False,
        }
        atomic_json(receipt_path, receipt)
        receipts.append(receipt)
    return tuple(receipts)


def _empty_aggregate(rows: int, width: int, *, dtype: object = np.float64) -> np.ndarray:
    return np.empty((rows, width), dtype=dtype)


def aggregate_observed_predictions(
    data: HestBiologicalData,
    receipts: tuple[Mapping[str, object], ...],
    binding: Mapping[str, object],
    *,
    null_p: Optional[Mapping[str, Optional[float]]] = None,
    registration_bootstrap_resamples: int = 20_000,
) -> tuple[QualificationInputs, Mapping[str, object]]:
    rows = len(data.evaluation_indices)
    molecular_width = data.molecular_targets.shape[1]
    assigned = np.zeros(rows, dtype=bool)
    truth = _empty_aggregate(rows, molecular_width)
    reference = _empty_aggregate(rows, molecular_width)
    arm_arrays = {
        arm_id: {
            "coordinate_truth": _empty_aggregate(rows, MAX_COORDINATE_RANK),
            "coordinate_prediction": _empty_aggregate(rows, MAX_COORDINATE_RANK),
            "coordinate_ranks": np.empty(rows, dtype=np.int64),
            "molecular_prediction": _empty_aggregate(rows, molecular_width),
            "rare_lower": _empty_aggregate(rows, MAX_COORDINATE_RANK),
            "rare_upper": _empty_aggregate(rows, MAX_COORDINATE_RANK),
        }
        for arm_id in SCORED_ARMS
    }
    registration_arrays = {
        comparison_id: {
            "truth": _empty_aggregate(rows, MAX_COORDINATE_RANK),
            "focal_prediction": _empty_aggregate(rows, MAX_COORDINATE_RANK),
            "comparator_prediction": _empty_aggregate(rows, MAX_COORDINATE_RANK),
            "coordinate_ranks": np.empty(rows, dtype=np.int64),
        }
        for comparison_id in REGISTRATION_PAIRS
    }
    fold_receipts = []
    for raw_receipt in receipts:
        heldout = str(raw_receipt["heldout_donor"])
        receipt_path = Path(str(raw_receipt["artifact_path"])).parent / "receipt.json"
        receipt, arrays = _validate_observed_fold(
            receipt_path,
            heldout_donor=heldout,
            binding_sha256=str(binding["sha256"]),
        )
        indices = np.asarray(arrays["row_indices"], dtype=np.int64)
        if (
            indices.ndim != 1
            or np.any(indices < 0)
            or np.any(indices >= rows)
            or len(np.unique(indices)) != len(indices)
            or np.any(assigned[indices])
            or not np.all(data.evaluation_donors[indices] == heldout)
        ):
            raise ValueError("observed outer-fold rows overlap or mismatch heldout donor")
        assigned[indices] = True
        truth[indices] = arrays["corrected_molecular_truth"]
        reference[indices] = arrays["reference_prediction"]
        for arm_id in SCORED_ARMS:
            for field in arm_arrays[arm_id]:
                arm_arrays[arm_id][field][indices] = arrays[f"arm__{arm_id}__{field}"]
        for comparison_id in REGISTRATION_PAIRS:
            for field in registration_arrays[comparison_id]:
                registration_arrays[comparison_id][field][indices] = arrays[
                    f"registration__{comparison_id}__{field}"
                ]
        fold_receipts.append(receipt)
    if not assigned.all():
        raise ValueError("observed outer folds do not cover every evaluation row exactly once")
    arms = {
        arm_id: ArmPrediction(
            coordinate_truth=values["coordinate_truth"],
            coordinate_prediction=values["coordinate_prediction"],
            coordinate_ranks=values["coordinate_ranks"],
            molecular_prediction=values["molecular_prediction"],
            rare_lower=values["rare_lower"],
            rare_upper=values["rare_upper"],
        )
        for arm_id, values in arm_arrays.items()
    }
    registration = {
        comparison_id: RegistrationComparison(**values)
        for comparison_id, values in registration_arrays.items()
    }
    p_values = {name: None for name in ACCEPTANCE_NULL_FAMILIES} if null_p is None else dict(null_p)
    inputs = QualificationInputs(
        donor_ids=data.evaluation_donors,
        section_ids=data.evaluation_sections,
        type_labels=data.evaluation_types,
        molecular_truth=truth,
        reference_prediction=reference,
        arms=arms,
        registration=registration,
        quality_mask=np.isin(
            data.evaluation_registration_strata,
            ("best", "intermediate", "near_threshold"),
        ),
        all_rows_mask=np.ones(rows, dtype=bool),
        best_mask=data.evaluation_registration_strata == "best",
        included_donor_ids=SELECTED_DONORS,
        refitted_null_empirical_p=p_values,
    )
    scored = score_qualification(
        inputs,
        registration_bootstrap_resamples=registration_bootstrap_resamples,
    )
    return inputs, {
        "score": scored,
        "fold_receipts": fold_receipts,
        "outer_fold_count": len(fold_receipts),
        "rows": rows,
    }


def _validate_null_fold(
    path: Path,
    *,
    binding_sha256: str,
    family: str,
    permutation_index: int,
    mapping_sha256: str,
    heldout_donor: str,
) -> Mapping[str, object]:
    receipt = _json(path)
    outer_refits = receipt.get("outer_refits")
    if (
        receipt.get("schema") != NULL_FOLD_SCHEMA
        or receipt.get("binding_sha256") != binding_sha256
        or receipt.get("family") != family
        or receipt.get("permutation_index") != permutation_index
        or receipt.get("mapping_sha256") != mapping_sha256
        or receipt.get("heldout_donor") != heldout_donor
        or receipt.get("completed_steps") != list(COMPLETE_REFIT_STEPS)
        or not isinstance(outer_refits, Mapping)
        or set(outer_refits) != {"A_star", "N0"}
        or any(not isinstance(rows, list) or not rows for rows in outer_refits.values())
        or not np.isfinite(float(receipt.get("registered_statistic", np.nan)))
        or receipt.get("molecular_outcomes_read") is not True
    ):
        raise ValueError("refitted-null outer-fold checkpoint identity is invalid")
    return receipt


def _validate_mapping_receipt(
    family: str,
    permutation_index: int,
    receipt: Mapping[str, object],
) -> None:
    changed = float(receipt.get("changed_fraction", np.nan))
    evaluation_changed = float(receipt.get("evaluation_changed_fraction", np.nan))
    expected_seed = NULL_SEEDS[family] + permutation_index * 104_729
    if (
        receipt.get("kind") != family
        or receipt.get("seed") != expected_seed
        or receipt.get("stratification") != ["donor", "section", "fine_type", "pool_role"]
        or not np.isfinite(changed)
        or not np.isfinite(evaluation_changed)
        or int(receipt.get("eligible_rows", 0)) < 1
    ):
        raise RuntimeError("refitted-null mapping receipt differs from the frozen design")
    if family == "within_section_type_derangement":
        if changed != 1.0 or evaluation_changed != 1.0:
            raise RuntimeError("within-stratum image derangement must change every row")
    elif (
        changed < MINIMUM_DIFFERENT_BLOCK_CHANGED_FRACTION
        or evaluation_changed < MINIMUM_DIFFERENT_BLOCK_CHANGED_FRACTION
        or float(receipt.get("cross_block_fraction", np.nan)) != changed
    ):
        raise RuntimeError("different-block mapping does not meet its frozen coverage floor")


def run_refitted_nulls(
    data: HestBiologicalData,
    work_dir: Path,
    binding: Mapping[str, object],
    *,
    observed_statistic: float,
    permutations: int,
    grid: TrainingGrid = FROZEN_TRAINING_GRID,
    device: str = "cuda",
    fold_runner: Callable[..., Mapping[str, object]] = fit_null_outer_fold,
) -> Mapping[str, object]:
    if permutations not in {SMOKE_NULL_PERMUTATIONS, FINAL_NULL_PERMUTATIONS}:
        raise ValueError("refitted null count must be the frozen smoke or final count")
    donors = tuple(sorted(set(data.evaluation_donors.tolist())))
    reports = {}
    for family, seed in NULL_SEEDS.items():
        design = build_neural_null_design(
            family,
            permutations,
            data.donor_ids,
            data.section_ids,
            data.type_labels,
            data.pool_roles,
            data.block_ids,
            data.observation_ids,
            seed=seed,
        )
        values = []
        permutation_receipts = []
        for permutation_index in range(permutations):
            mapping = design.mappings[permutation_index]
            evaluation_rows = data.evaluation_indices
            mapping_receipt = {
                **design.receipts[permutation_index],
                "evaluation_changed_fraction": float(
                    np.mean(mapping[evaluation_rows] != evaluation_rows)
                ),
            }
            _validate_mapping_receipt(family, permutation_index, mapping_receipt)
            permutation_dir = work_dir / "nulls" / family / f"permutation-{permutation_index:03d}"
            fold_receipts = []
            for heldout in donors:
                path = permutation_dir / "folds" / heldout / "receipt.json"
                if path.exists():
                    fold_receipt = _validate_null_fold(
                        path,
                        binding_sha256=str(binding["sha256"]),
                        family=family,
                        permutation_index=permutation_index,
                        mapping_sha256=str(mapping_receipt["mapping_sha256"]),
                        heldout_donor=heldout,
                    )
                else:
                    result = fold_runner(
                        data,
                        heldout,
                        mapping,
                        grid=grid,
                        device=device,
                    )
                    if result.get("completed_steps") != list(COMPLETE_REFIT_STEPS):
                        raise RuntimeError("refitted-null fold omitted a frozen execution step")
                    fold_receipt = {
                        **dict(result),
                        "schema": NULL_FOLD_SCHEMA,
                        "analysis_status": ANALYSIS_STATUS,
                        "binding_sha256": binding["sha256"],
                        "family": family,
                        "permutation_index": permutation_index,
                        "mapping_sha256": mapping_receipt["mapping_sha256"],
                        "molecular_outcomes_read": True,
                        "biological_experiment_run": True,
                        "engineering_decision_available": False,
                        "authorizes_h_cell": False,
                        "authorizes_h_intrinsic": False,
                        "authorizes_h_ref": False,
                        "authorizes_full_heir": False,
                    }
                    atomic_json(path, fold_receipt)
                fold_receipts.append(fold_receipt)
            statistic = float(
                np.mean([float(receipt["registered_statistic"]) for receipt in fold_receipts])
            )
            permutation_receipt = {
                "schema": NULL_PERMUTATION_SCHEMA,
                "analysis_status": ANALYSIS_STATUS,
                "binding_sha256": binding["sha256"],
                "family": family,
                "permutation_index": permutation_index,
                "mapping_receipt": dict(mapping_receipt),
                "registered_statistic": statistic,
                "outer_folds": len(fold_receipts),
                "fold_receipts_sha256": canonical_sha256(fold_receipts),
                "complete_refit": True,
                "molecular_outcomes_read": True,
                "authorizes_h_cell": False,
                "authorizes_h_intrinsic": False,
                "authorizes_h_ref": False,
                "authorizes_full_heir": False,
            }
            atomic_json(permutation_dir / "receipt.json", permutation_receipt)
            values.append(statistic)
            permutation_receipts.append(permutation_receipt)
        array = np.asarray(values, dtype=np.float64)
        reports[family] = {
            "family": family,
            "registered_statistic": ("A_star_minus_N0_donor_type_macro_residual_coordinate_R2"),
            "permutations": permutations,
            "observed_statistic": float(observed_statistic),
            "values": values,
            "empirical_p": float(
                (1 + np.count_nonzero(array >= observed_statistic)) / (1 + permutations)
            ),
            "mapping_set_sha256": design.mapping_set_sha256,
            "permutation_receipts_sha256": canonical_sha256(permutation_receipts),
            "model_refit_for_every_permutation": True,
            "complete_refit_steps": list(COMPLETE_REFIT_STEPS),
        }
    return reports


def _markdown(report: Mapping[str, object]) -> str:
    scored = report["scientific_acceptance"]
    d1 = scored["decisions"]["D1"]
    d2 = scored["decisions"]["D2"]
    lines = [
        "# HEST nonlinear qualification v2",
        "",
        f"- Status: `{report['execution_status']}`",
        f"- Analysis role: `{report['analysis_status']}`",
        f"- Biological rows fitted: `{report['biological_experiment_run']}`",
        f"- Final engineering decision available: `{report['engineering_decision_available']}`",
        f"- D1 prospective estimator design support: `{d1['passed']}`",
        f"- D2 intrinsic-localization follow-up support: `{d2['passed']}`",
        "- H-CELL, H-INTRINSIC, H-REF, and full-HEIR authorization: `false`",
        "",
        "## Interpretation",
        "",
        "This is exposed retrospective estimator-development evidence. Even a D1 or D2 pass can "
        "only motivate a newly registered, unexposed prospective protocol.",
        "",
    ]
    if report.get("refitted_nulls"):
        lines.extend(["## Refitted nulls", ""])
        for family, value in report["refitted_nulls"].items():
            lines.append(
                f"- `{family}`: {value['permutations']} complete refits, "
                f"empirical p = `{value['empirical_p']:.6g}`"
            )
        lines.append("")
    return "\n".join(lines)


def run_full_qualification(
    data: HestBiologicalData,
    authorization: Mapping[str, object],
    *,
    work_dir: Path,
    report_path: Path,
    markdown_path: Path,
    null_permutations: int = FINAL_NULL_PERMUTATIONS,
    grid: TrainingGrid = FROZEN_TRAINING_GRID,
    device: str = "cuda",
    observed_fold_runner: Callable[..., OuterFoldModels] = fit_observed_outer_fold,
    null_fold_runner: Callable[..., Mapping[str, object]] = fit_null_outer_fold,
) -> Mapping[str, object]:
    binding = full_run_binding(authorization, grid=grid, device=device)
    observed_receipts = run_observed_folds(
        data,
        work_dir,
        binding,
        grid=grid,
        device=device,
        fold_runner=observed_fold_runner,
    )
    _, observed = aggregate_observed_predictions(
        data,
        observed_receipts,
        binding,
    )
    observed_statistic = float(observed["score"]["decision_metrics"]["D1"]["A_star_minus_N0_R2"])
    nulls = run_refitted_nulls(
        data,
        work_dir,
        binding,
        observed_statistic=observed_statistic,
        permutations=null_permutations,
        grid=grid,
        device=device,
        fold_runner=null_fold_runner,
    )
    null_p = {NULL_P_NAMES[name]: float(value["empirical_p"]) for name, value in nulls.items()}
    _, final_observed = aggregate_observed_predictions(
        data,
        observed_receipts,
        binding,
        null_p=null_p,
    )
    final = null_permutations == FINAL_NULL_PERMUTATIONS
    report = {
        "schema": REPORT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "execution_status": (
            "complete_final_retrospective_qualification"
            if final
            else "complete_20_permutation_execution_smoke_nonfinal"
        ),
        "binding": binding,
        "authorization": dict(authorization),
        "source_access": dict(data.source_access),
        "observed": {
            "outer_folds": len(observed_receipts),
            "fold_receipts_sha256": canonical_sha256(observed_receipts),
            "control_aliases": dict(CONTROL_ALIASES),
            "selected_primary_arms": {
                str(receipt["heldout_donor"]): str(receipt["selected_primary_arm"])
                for receipt in observed_receipts
            },
            "selected_single_arms": {
                str(receipt["heldout_donor"]): str(receipt["selected_single_arm"])
                for receipt in observed_receipts
            },
            "per_seed_outer_refits": {
                str(receipt["heldout_donor"]): receipt["neural_outer_refits"]
                for receipt in observed_receipts
            },
        },
        "refitted_nulls": nulls,
        "scientific_acceptance": final_observed["score"],
        "supports_prospective_local_context_estimator_design": bool(
            final and final_observed["score"]["decisions"]["D1"]["passed"]
        ),
        "supports_intrinsic_localization_followup": bool(
            final and final_observed["score"]["decisions"]["D2"]["passed"]
        ),
        "biological_experiment_run": True,
        "engineering_decision_available": final,
        "molecular_outcomes_read": True,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }
    atomic_json(report_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=markdown_path.parent,
        prefix=markdown_path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_markdown(report))
        os.replace(temporary, markdown_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return report


__all__ = [
    "COMPLETE_REFIT_STEPS",
    "CONTROL_ALIASES",
    "FINAL_NULL_PERMUTATIONS",
    "FULL_RUN_SCHEMA",
    "NULL_FOLD_SCHEMA",
    "NULL_PERMUTATION_SCHEMA",
    "NULL_SEEDS",
    "MINIMUM_DIFFERENT_BLOCK_CHANGED_FRACTION",
    "NEURAL_REFIT_ARMS",
    "OBSERVED_FOLD_SCHEMA",
    "REPORT_SCHEMA",
    "SCORED_ARMS",
    "SMOKE_NULL_PERMUTATIONS",
    "aggregate_observed_predictions",
    "full_run_binding",
    "observed_fold_arrays",
    "run_full_qualification",
    "run_observed_folds",
    "run_refitted_nulls",
]
