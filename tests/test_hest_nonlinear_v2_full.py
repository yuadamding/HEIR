from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from heir.evaluation import nonlinear_qualification_v2_execution as execution
from heir.evaluation import nonlinear_qualification_v2_full as full
from heir.evaluation.nonlinear_qualification_v2_acceptance import RegistrationComparison
from heir.evaluation.nonlinear_qualification_v2_data import HestBiologicalData
from heir.evaluation.nonlinear_qualification_v2_models import (
    FittedArm,
    OuterFoldModels,
    TrainingGrid,
)
from heir.evaluation.ridge_probe import MolecularTargetFit


def _data() -> HestBiologicalData:
    donor_names = [f"D{index:02d}" for index in range(12)]
    donors = np.repeat(donor_names, 40)
    labels = np.tile(np.repeat([0, 1], 20), len(donor_names))
    rows = len(donors)
    within_type = np.tile(np.arange(20), len(donor_names) * 2)
    return HestBiologicalData(
        source_path=Path("source.npz"),
        observation_ids=np.asarray([f"cell-{index}" for index in range(rows)]),
        donor_ids=donors,
        section_ids=np.asarray([f"{donor}-S" for donor in donors]),
        type_labels=labels,
        type_names=("T0", "T1"),
        pool_roles=np.full(rows, "evaluation", dtype="<U10"),
        block_ids=np.where(within_type % 2 == 0, "a", "b"),
        registration_quality_strata=np.full(rows, "best", dtype="<U16"),
        evaluation_indices=np.arange(rows, dtype=np.int64),
        image_features=np.zeros((rows, 4, 2), dtype=np.float32),
        molecular_targets=np.zeros((rows, 2), dtype=np.float64),
        reference_means=np.zeros((rows, 2), dtype=np.float64),
        technical_covariates=np.zeros((rows, 1), dtype=np.float64),
        control_families={"combined_nonimage": np.zeros((rows, 1), dtype=np.float32)},
        control_feature_names={"combined_nonimage": ("control",)},
        blank_feature=np.zeros((1, 2), dtype=np.float32),
        source_access={"molecular_member_open_count": 1, "molecular_outcomes_read": True},
    )


def _target() -> MolecularTargetFit:
    return MolecularTargetFit(
        technical_means=np.zeros((2, 1)),
        technical_coefficients=np.zeros((2, 1, 2)),
        residual_means=np.zeros((2, 2)),
        bases=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        rank=2,
    )


def _observed_fold(data: HestBiologicalData, heldout: str) -> OuterFoldModels:
    indices = np.flatnonzero(data.evaluation_donors == heldout)
    local = np.tile(np.linspace(-1.0, 1.0, 20), 2)
    truth = np.column_stack((local, local[::-1]))
    ranks = np.full(len(indices), 2, dtype=np.int64)
    lower = np.full_like(truth, -0.7)
    upper = np.full_like(truth, 0.7)
    offsets = {
        "B0": 0.55,
        "B1": 0.50,
        "B2": 0.40,
        "B3": 0.35,
        "N0": 0.50,
        "N1": 0.20,
        "N2": 0.15,
        "N3": 0.20,
        "N4": 0.20,
        "N5": 0.30,
        "N6": 0.12,
        "N7": 0.10,
        "C_blank": 0.50,
        "C_blank_single": 0.50,
        "A_star": 0.10,
    }
    arms = {
        arm_id: FittedArm(
            arm_id=arm_id,
            target=_target(),
            coordinate_truth=truth,
            coordinate_prediction=truth + offset,
            coordinate_ranks=ranks,
            molecular_prediction=truth + offset,
            corrected_molecular_truth=truth,
            rare_lower=lower,
            rare_upper=upper,
            receipt=(
                {"estimator": "per_type_ridge", "arm_id": arm_id}
                if arm_id in {"B0", "B1", "B2"}
                else {
                    "estimator": "neural",
                    "arm_id": arm_id,
                    "outer_refit": [{"seed": 17, "checkpoint_sha256": "a" * 64}],
                }
            ),
        )
        for arm_id, offset in offsets.items()
    }
    registration = {
        comparison_id: RegistrationComparison(
            truth=truth,
            focal_prediction=arms[focal].coordinate_prediction,
            comparator_prediction=arms[comparator].coordinate_prediction,
            coordinate_ranks=ranks,
        )
        for comparison_id, (focal, comparator) in full.REGISTRATION_PAIRS.items()
    }
    return OuterFoldModels(
        heldout_donor=heldout,
        row_indices=indices,
        arms=arms,
        registration=registration,
        selected_primary_arm="N7",
        selected_single_arm="N2",
        selections={"A_star": "N7", "A_single": "N2"},
        corrected_molecular_truth=truth,
        reference_prediction=np.zeros_like(truth),
    )


GRID = TrainingGrid(
    ranks=(2,),
    ridge_alphas=(0.01,),
    weight_decays=(0.01,),
    seeds=(17,),
    maximum_epochs=1,
    batch_size=8,
    patience=1,
    minimum_support=20,
    minimum_variance_ratio=0.0,
    minimum_basis_ceiling_r2=0.0,
)
BINDING = {"sha256": "a" * 64}


def test_full_binding_requires_the_authorized_smoke_device() -> None:
    authorization = {
        "schema": execution.BIOLOGICAL_AUTHORIZATION_SCHEMA,
        "registered_source_sha256": full.SOURCE_SHA256,
        "protocol_sha256": full.V2_PROTOCOL_SHA256,
        "implementation": {"runner.py": "a" * 64},
        "git_commit": "b" * 40,
        "device": "cpu",
        "biological_execution_authorized": True,
        "molecular_outcomes_read": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }
    binding = full.full_run_binding(authorization, device="cpu")
    assert binding["device"] == "cpu"
    with pytest.raises(ValueError, match="production authorization"):
        full.full_run_binding(authorization, device="cuda")

    authorization["authorizes_h_cell"] = True
    with pytest.raises(ValueError, match="production authorization"):
        full.full_run_binding(authorization, device="cpu")


def test_observed_outer_folds_resume_and_aggregate_once(tmp_path: Path) -> None:
    data = _data()
    calls: list[str] = []

    def runner(data, heldout, mapping, **kwargs):
        del mapping, kwargs
        calls.append(heldout)
        return _observed_fold(data, heldout)

    receipts = full.run_observed_folds(
        data,
        tmp_path,
        BINDING,
        grid=GRID,
        device="cpu",
        fold_runner=runner,
    )
    assert len(calls) == 12
    assert len(receipts) == 12

    resumed = full.run_observed_folds(
        data,
        tmp_path,
        BINDING,
        grid=GRID,
        device="cpu",
        fold_runner=runner,
    )
    assert len(calls) == 12
    _, aggregate = full.aggregate_observed_predictions(
        data,
        resumed,
        BINDING,
        registration_bootstrap_resamples=1_000,
    )
    assert aggregate["rows"] == 480
    assert aggregate["outer_fold_count"] == 12
    assert aggregate["score"]["rows"] == 480


def test_observed_resume_rejects_artifact_tampering(tmp_path: Path) -> None:
    data = _data()
    receipts = full.run_observed_folds(
        data,
        tmp_path,
        BINDING,
        grid=GRID,
        device="cpu",
        fold_runner=lambda data, heldout, mapping, **kwargs: _observed_fold(data, heldout),
    )
    artifact = Path(str(receipts[0]["artifact_path"]))
    with artifact.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="checkpoint identity"):
        full.run_observed_folds(
            data,
            tmp_path,
            BINDING,
            grid=GRID,
            device="cpu",
            fold_runner=lambda *args, **kwargs: pytest.fail("must not refit"),
        )


def test_refitted_nulls_resume_each_permutation_and_outer_fold(tmp_path: Path) -> None:
    data = _data()
    calls = 0

    def runner(data, heldout, mapping, **kwargs):
        nonlocal calls
        del data, mapping, kwargs
        calls += 1
        return {
            "heldout_donor": heldout,
            "selected_primary_arm": "N7",
            "registered_statistic": 0.0,
            "outer_refits": {
                "A_star": [{"seed": 17, "checkpoint_sha256": "a" * 64}],
                "N0": [{"seed": 17, "checkpoint_sha256": "b" * 64}],
            },
            "completed_steps": list(full.COMPLETE_REFIT_STEPS),
        }

    result = full.run_refitted_nulls(
        data,
        tmp_path,
        BINDING,
        observed_statistic=0.2,
        permutations=full.SMOKE_NULL_PERMUTATIONS,
        grid=GRID,
        device="cpu",
        fold_runner=runner,
    )
    assert calls == 2 * 20 * 12
    assert all(value["empirical_p"] == pytest.approx(1 / 21) for value in result.values())

    full.run_refitted_nulls(
        data,
        tmp_path,
        BINDING,
        observed_statistic=0.2,
        permutations=full.SMOKE_NULL_PERMUTATIONS,
        grid=GRID,
        device="cpu",
        fold_runner=runner,
    )
    assert calls == 2 * 20 * 12


def test_different_block_null_enforces_the_frozen_coverage_floor() -> None:
    receipt = {
        "kind": "different_spatial_block_reassignment",
        "seed": full.NULL_SEEDS["different_spatial_block_reassignment"],
        "stratification": ["donor", "section", "fine_type", "pool_role"],
        "eligible_rows": 90,
        "changed_fraction": 0.9,
        "evaluation_changed_fraction": 0.9,
        "cross_block_fraction": 0.9,
    }
    with pytest.raises(RuntimeError, match="coverage floor"):
        full._validate_mapping_receipt(
            "different_spatial_block_reassignment",
            0,
            receipt,
        )
