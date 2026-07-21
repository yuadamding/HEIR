"""Outcome-locked preflight and resumable synthetic calibration for HEST v2."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from heir.evaluation.blank_patch_receipt import (
    RECEIPT_SCHEMA as BLANK_RECEIPT_SCHEMA,
)
from heir.evaluation.blank_patch_receipt import (
    SourceIdentity,
    array_sha256,
    inspect_registered_source,
    load_supplement,
)
from heir.evaluation.nonlinear_qualification_contract import canonical_sha256, file_sha256
from heir.evaluation.nonlinear_qualification_v2_acceptance import (
    run_synthetic_trial,
    synthetic_trial_inputs,
)
from heir.evaluation.nonlinear_qualification_v2_contract import (
    ANALYSIS_STATUS,
    BLANK_FEATURE_SHA256,
    BLANK_RECEIPT_SHA256,
    BLANK_SUPPLEMENT_SHA256,
    CALIBRATION_EXPECTATIONS,
    SELECTED_DONORS,
    SOURCE_SHA256,
    TOPOLOGY_RECEIPT_SHA256,
    V2_PROTOCOL_SHA256,
    evaluate_synthetic_calibration,
    validate_manifest,
    validate_protocol,
)
from heir.evaluation.nonlinear_qualification_v2_smoke import (
    COMPLETE_REFIT_STEPS as SMOKE_COMPLETE_REFIT_STEPS,
)
from heir.evaluation.nonlinear_qualification_v2_smoke import (
    run_implementation_smoke,
)
from heir.evaluation.registration_topology import (
    RECEIPT_SCHEMA as TOPOLOGY_RECEIPT_SCHEMA,
)
from heir.evaluation.registration_topology import (
    SOURCE_MEMBER_ALLOWLIST,
    DirectNpzReader,
    TopologyArrays,
    load_topology_arrays,
)

PREFLIGHT_SCHEMA = "heir.hest_nonlinear_v2_preflight.v1"
CHECKPOINT_SCHEMA = "heir.hest_nonlinear_v2_calibration_checkpoint.v1"
CALIBRATION_SCHEMA = "heir.hest_nonlinear_v2_calibration.v1"
BIOLOGICAL_AUTHORIZATION_SCHEMA = "heir.hest_nonlinear_v2_biological_authorization.v1"
IMPLEMENTATION_SMOKE_SCHEMA = "heir.hest_nonlinear_v2_implementation_smoke.v1"
SOURCE_ROWS = 36_121
EVALUATION_ROWS = 18_062
FORMAL_Q3_ROWS = 3_991
PRODUCTION_TRIALS = 200
MOLECULAR_MEMBER = "nucleus_molecular_targets"
CALIBRATION_SOURCE_MEMBER_ALLOWLIST = SOURCE_MEMBER_ALLOWLIST + ("block_ids",)


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a readable JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _git_runtime(repository_root: Path) -> Mapping[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("HEST v2 requires a readable Git execution identity") from error
    return {
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "git_status_entry_count": len(status),
        "python_version": __import__("sys").version.split()[0],
        "numpy_version": np.__version__,
    }


def implementation_hashes(repository_root: Path) -> Mapping[str, str]:
    relative = (
        "scripts/benchmark_hest_nonlinear_qualification_v2.py",
        "src/heir/evaluation/hest_nested_ridge.py",
        "src/heir/evaluation/hierarchical_metrics.py",
        "src/heir/evaluation/neural_checkpoint.py",
        "src/heir/evaluation/neural_control_models.py",
        "src/heir/evaluation/neural_model_selection.py",
        "src/heir/evaluation/neural_nulls.py",
        "src/heir/evaluation/neural_probe.py",
        "src/heir/evaluation/neural_training.py",
        "src/heir/evaluation/nonlinear_qualification_v2_acceptance.py",
        "src/heir/evaluation/nonlinear_qualification_v2_data.py",
        "src/heir/evaluation/nonlinear_qualification_v2_full.py",
        "src/heir/evaluation/nonlinear_qualification_v2_models.py",
        "src/heir/evaluation/nonlinear_qualification_v2_smoke.py",
        "src/heir/evaluation/nonlinear_qualification_v2_contract.py",
        "src/heir/evaluation/nonlinear_qualification_v2_execution.py",
        "src/heir/evaluation/registration_sensitivity_v2.py",
        "src/heir/evaluation/residual_targets.py",
        "src/heir/evaluation/ridge_probe.py",
        "src/heir/evaluation/weighted_basis.py",
    )
    result = {}
    for name in relative:
        path = repository_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = file_sha256(path)
    return result


def _source_member_registry(path: Path) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("registered HEST source is not a readable NPZ archive") from error
    if len(names) != len(set(names)):
        raise ValueError("registered HEST source contains duplicate ZIP members")
    return tuple(name[:-4] for name in names if name.endswith(".npy"))


def _validate_blank_artifacts(
    source_identity: SourceIdentity,
    supplement_path: Path,
    receipt_path: Path,
) -> Mapping[str, object]:
    supplement = load_supplement(supplement_path)
    if (
        supplement.sha256 != BLANK_SUPPLEMENT_SHA256
        or supplement.registered_source_sha256 != SOURCE_SHA256
        or supplement.ordered_observation_ids_sha256 != source_identity.observation_ids_sha256
        or supplement.source_rows != SOURCE_ROWS
        or supplement.feature_sha256 != BLANK_FEATURE_SHA256
    ):
        raise ValueError("blank supplement differs from the v2 protocol")
    if file_sha256(receipt_path) != BLANK_RECEIPT_SHA256:
        raise ValueError("blank receipt bytes differ from the v2 protocol")
    receipt = _json(receipt_path)
    expected = {
        "schema": BLANK_RECEIPT_SCHEMA,
        "status": "passed",
        "registered_source_sha256": SOURCE_SHA256,
        "ordered_observation_ids_sha256": source_identity.observation_ids_sha256,
        "source_rows": SOURCE_ROWS,
        "supplement_sha256": BLANK_SUPPLEMENT_SHA256,
        "feature_sha256": BLANK_FEATURE_SHA256,
        "source_members_opened": ["observation_ids"],
        "molecular_outcomes_read": False,
        "predictions_read": False,
        "biological_experiment_run": False,
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise ValueError("blank receipt content differs from the frozen outcome-blind identity")
    parity = receipt.get("batch_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError("blank receipt lacks passing batch parity")
    return {
        "supplement_sha256": supplement.sha256,
        "receipt_sha256": BLANK_RECEIPT_SHA256,
        "feature_sha256": supplement.feature_sha256,
        "source_members_opened": ["observation_ids"],
        "molecular_outcomes_read": False,
    }


def _validate_topology_receipt(
    path: Path,
    *,
    observation_ids_sha256: str,
) -> Mapping[str, object]:
    if file_sha256(path) != TOPOLOGY_RECEIPT_SHA256:
        raise ValueError("registration-topology receipt bytes differ from the v2 protocol")
    receipt = _json(path)
    selected = receipt.get("selected_band")
    if not isinstance(selected, Mapping):
        raise ValueError("registration-topology receipt has no selected band")
    expected_flags = (
        "molecular_outcomes_read",
        "reference_expression_read",
        "image_features_read",
        "predictions_read",
        "model_reports_read",
    )
    if (
        receipt.get("schema") != TOPOLOGY_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("registered_source_sha256") != SOURCE_SHA256
        or receipt.get("ordered_observation_ids_sha256") != observation_ids_sha256
        or receipt.get("source_rows") != SOURCE_ROWS
        or receipt.get("source_members_opened") != list(SOURCE_MEMBER_ALLOWLIST)
        or receipt.get("selection_status") != "passed"
        or selected.get("quality_band_id") != "Q3"
        or selected.get("selection_rows") != FORMAL_Q3_ROWS
        or tuple(selected.get("included_donor_ids", ())) != SELECTED_DONORS
        or any(receipt.get(name) is not False for name in expected_flags)
    ):
        raise ValueError("registration-topology receipt content differs from v2")
    return {
        "receipt_sha256": TOPOLOGY_RECEIPT_SHA256,
        "source_members_opened": list(SOURCE_MEMBER_ALLOWLIST),
        "selected_quality_band_id": "Q3",
        "selected_rows": FORMAL_Q3_ROWS,
        "included_donor_ids": list(SELECTED_DONORS),
        **{name: False for name in expected_flags},
    }


def preflight(
    *,
    source: Path,
    blank_supplement: Path,
    blank_receipt: Path,
    registration_topology: Path,
    protocol_path: Path,
    manifest_path: Path,
    repository_root: Path,
    require_clean: bool = True,
) -> Mapping[str, object]:
    """Validate every frozen identity while opening no molecular member."""

    protocol = _json(protocol_path)
    if file_sha256(protocol_path) != V2_PROTOCOL_SHA256:
        raise ValueError("v2 protocol bytes differ from the frozen contract")
    validate_protocol(protocol, repository_root=repository_root)
    manifest = _json(manifest_path)
    validate_manifest(
        manifest,
        protocol,
        repository_root=repository_root,
        protocol_file_sha256=V2_PROTOCOL_SHA256,
    )
    source_identity = inspect_registered_source(
        source,
        expected_sha256=SOURCE_SHA256,
        expected_rows=SOURCE_ROWS,
    )
    members = _source_member_registry(source_identity.path)
    if members.count(MOLECULAR_MEMBER) != 1:
        raise ValueError("registered source does not carry exactly one frozen molecular member")
    blank = _validate_blank_artifacts(source_identity, blank_supplement, blank_receipt)
    topology = _validate_topology_receipt(
        registration_topology,
        observation_ids_sha256=source_identity.observation_ids_sha256,
    )
    runtime = _git_runtime(repository_root)
    if require_clean and runtime["git_worktree_dirty"] is not False:
        raise RuntimeError("HEST v2 execution requires a clean committed worktree")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "protocol_sha256": V2_PROTOCOL_SHA256,
        "manifest_sha256": file_sha256(manifest_path),
        "registered_source": {
            "path": str(source_identity.path),
            "sha256": source_identity.sha256,
            "bytes": source_identity.bytes,
            "rows": source_identity.rows,
            "ordered_observation_ids_sha256": source_identity.observation_ids_sha256,
            "source_members_opened": ["observation_ids"],
            "molecular_member_present_not_opened": True,
        },
        "blank_control": blank,
        "registration_topology": topology,
        "implementation": implementation_hashes(repository_root),
        "runtime": runtime,
        "clean_worktree_gate_enforced": require_clean,
        "preflight_passed": True,
        "synthetic_calibration_authorized": bool(
            not require_clean or runtime["git_worktree_dirty"] is False
        ),
        "biological_execution_authorized": False,
        "molecular_outcomes_read": False,
        "biological_experiment_run": False,
        "engineering_decision_available": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


def load_calibration_topology(source: Path) -> tuple[TopologyArrays, Mapping[str, object]]:
    """Read exactly the seven outcome-blind members allowed during calibration."""

    reader = DirectNpzReader(source, allowed_members=CALIBRATION_SOURCE_MEMBER_ALLOWLIST)
    arrays = load_topology_arrays(reader, expected_rows=SOURCE_ROWS)
    block_ids = np.asarray(reader.read("block_ids"))
    if (
        block_ids.shape != (SOURCE_ROWS,)
        or block_ids.dtype.kind not in {"S", "U"}
        or np.any(np.char.str_len(block_ids.astype(str)) == 0)
    ):
        raise ValueError("calibration block IDs must be a complete row-aligned string vector")
    if reader.members_opened != list(CALIBRATION_SOURCE_MEMBER_ALLOWLIST):
        raise RuntimeError("calibration source reader escaped its exact allowlist")
    evaluation = np.char.startswith(arrays.pool_roles, "evaluation")
    if int(evaluation.sum()) != EVALUATION_ROWS:
        raise ValueError("calibration topology evaluation-row count differs from v2")
    quality = evaluation & np.isin(
        arrays.registration_quality_strata,
        ("best", "intermediate", "near_threshold"),
    )
    formal_quality = quality & np.isin(arrays.donor_ids, SELECTED_DONORS)
    if int(formal_quality.sum()) != FORMAL_Q3_ROWS:
        raise ValueError("calibration topology no longer reproduces formal Q3")
    selected = np.flatnonzero(evaluation)
    local = TopologyArrays(
        observation_ids=arrays.observation_ids[selected],
        donor_ids=arrays.donor_ids[selected],
        section_ids=arrays.section_ids[selected],
        type_labels=arrays.type_labels[selected],
        pool_roles=arrays.pool_roles[selected],
        registration_quality_scores=arrays.registration_quality_scores[selected],
        registration_quality_strata=arrays.registration_quality_strata[selected],
    )
    return local, {
        "source_members_opened": list(reader.members_opened),
        "ordered_block_ids_sha256": array_sha256(block_ids.astype(str)),
        "evaluation_rows": EVALUATION_ROWS,
        "formal_Q3_rows": FORMAL_Q3_ROWS,
        "molecular_outcomes_read": False,
        "image_features_read": False,
        "reference_expression_read": False,
    }


def _checkpoint_hash(value: Mapping[str, object]) -> str:
    return canonical_sha256({name: item for name, item in value.items() if name != "sha256"})


def _save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    checkpoint["sha256"] = _checkpoint_hash(checkpoint)
    atomic_json(path, checkpoint)


def _load_checkpoint(path: Path, binding_sha256: str) -> dict[str, object]:
    checkpoint = _json(path)
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("sha256") != _checkpoint_hash(checkpoint)
        or checkpoint.get("binding_sha256") != binding_sha256
        or checkpoint.get("molecular_outcomes_read") is not False
    ):
        raise ValueError("synthetic calibration checkpoint identity is invalid")
    records = checkpoint.get("records")
    if not isinstance(records, Mapping) or set(records) != set(CALIBRATION_EXPECTATIONS):
        raise ValueError("synthetic calibration checkpoint condition registry is invalid")
    binding = checkpoint.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("synthetic calibration checkpoint binding is malformed")
    trial_limit = binding.get("trials_per_condition")
    if isinstance(trial_limit, bool) or not isinstance(trial_limit, int) or trial_limit < 1:
        raise ValueError("synthetic calibration checkpoint trial limit is invalid")
    expected_fields = {
        "condition",
        "trial_index",
        "D1",
        "D2",
        "expected",
        "matches_expectation",
        "decision_metrics_sha256",
        "molecular_outcomes_read",
    }
    for condition, expectation in CALIBRATION_EXPECTATIONS.items():
        local = records[condition]
        if not isinstance(local, list) or len(local) > trial_limit:
            raise ValueError("synthetic calibration checkpoint trial registry is invalid")
        for trial_index, record in enumerate(local):
            if not isinstance(record, Mapping) or set(record) != expected_fields:
                raise ValueError("synthetic calibration checkpoint trial record is malformed")
            decisions = {name: record.get(name) for name in ("D1", "D2")}
            if (
                record.get("condition") != condition
                or record.get("trial_index") != trial_index
                or decisions.keys() != expectation.keys()
                or any(not isinstance(value, bool) for value in decisions.values())
                or record.get("expected") != expectation
                or record.get("matches_expectation") != (decisions == expectation)
                or not isinstance(record.get("decision_metrics_sha256"), str)
                or len(str(record["decision_metrics_sha256"])) != 64
                or record.get("molecular_outcomes_read") is not False
            ):
                raise ValueError("synthetic calibration checkpoint trial record is invalid")
    return checkpoint


def run_calibration_trials(
    topology: TopologyArrays,
    *,
    binding: Mapping[str, object],
    checkpoint_path: Path,
    trials_per_condition: int,
    smoke: bool,
    registration_bootstrap_resamples: int = 20_000,
    trial_runner: Callable[..., Mapping[str, object]] = run_synthetic_trial,
) -> Mapping[str, object]:
    """Run or resume synthetic decision trials and compile a fail-closed receipt."""

    if trials_per_condition < 1:
        raise ValueError("synthetic calibration trial count must be positive")
    if smoke:
        if trials_per_condition >= PRODUCTION_TRIALS:
            raise ValueError("smoke calibration must stay below the production trial count")
    elif trials_per_condition != PRODUCTION_TRIALS:
        raise ValueError("production calibration requires exactly 200 trials per condition")
    binding_value = {
        **dict(binding),
        "trials_per_condition": trials_per_condition,
        "smoke": smoke,
        "registration_bootstrap_resamples": registration_bootstrap_resamples,
    }
    binding_sha256 = canonical_sha256(binding_value)
    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, binding_sha256)
    else:
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "binding": binding_value,
            "binding_sha256": binding_sha256,
            "records": {condition: [] for condition in CALIBRATION_EXPECTATIONS},
            "molecular_outcomes_read": False,
            "biological_experiment_run": False,
        }
        _save_checkpoint(checkpoint_path, checkpoint)
    records = checkpoint["records"]
    if not isinstance(records, dict):
        raise AssertionError("validated calibration checkpoint lost mutable records")
    quality = np.isin(
        topology.registration_quality_strata,
        ("best", "intermediate", "near_threshold"),
    )
    best = topology.registration_quality_strata == "best"
    for condition, expected in CALIBRATION_EXPECTATIONS.items():
        local_records = records[condition]
        if not isinstance(local_records, list) or len(local_records) > trials_per_condition:
            raise ValueError("synthetic calibration checkpoint trial sequence is invalid")
        if any(not isinstance(record, Mapping) for record in local_records):
            raise ValueError("synthetic calibration checkpoint trial record is invalid")
        if [record["trial_index"] for record in local_records] != list(range(len(local_records))):
            raise ValueError("synthetic calibration checkpoint trials are not contiguous")
        for trial_index in range(len(local_records), trials_per_condition):
            inputs = synthetic_trial_inputs(
                topology.donor_ids,
                topology.section_ids,
                topology.type_labels,
                quality,
                best,
                SELECTED_DONORS,
                condition=condition,
                trial_index=trial_index,
            )
            result = trial_runner(
                inputs,
                condition=condition,
                trial_index=trial_index,
                registration_bootstrap_resamples=registration_bootstrap_resamples,
            )
            if any(type(result.get(name)) is not bool for name in ("D1", "D2")):
                raise ValueError("synthetic trial decisions must be booleans")
            decisions = {name: result[name] for name in ("D1", "D2")}
            scored = result.get("scored")
            if not isinstance(scored, Mapping):
                raise ValueError("synthetic trial did not return the shared acceptance report")
            decision_metrics = scored.get("decision_metrics")
            if not isinstance(decision_metrics, Mapping):
                raise ValueError("synthetic trial omitted decision metrics")
            local_records.append(
                {
                    "condition": condition,
                    "trial_index": trial_index,
                    "D1": decisions["D1"],
                    "D2": decisions["D2"],
                    "expected": dict(expected),
                    "matches_expectation": decisions == expected,
                    "decision_metrics_sha256": canonical_sha256(decision_metrics),
                    "molecular_outcomes_read": False,
                }
            )
            _save_checkpoint(checkpoint_path, checkpoint)
    pass_counts = {
        condition: {
            decision: sum(bool(record[decision]) for record in records[condition])
            for decision in ("D1", "D2")
        }
        for condition in CALIBRATION_EXPECTATIONS
    }
    if smoke:
        calibration = {
            "schema": "heir.hest_nonlinear_v2_synthetic_calibration_smoke.v1",
            "trials_per_condition": trials_per_condition,
            "all_trials_match_expectation": all(
                bool(record["matches_expectation"])
                for condition in CALIBRATION_EXPECTATIONS
                for record in records[condition]
            ),
            "calibration_passed": False,
            "HEST_molecular_execution_authorized": False,
            "reason": "reduced_trial_smoke_is_non_authorizing",
        }
    else:
        calibration = evaluate_synthetic_calibration(
            pass_counts,
            trials_per_condition=trials_per_condition,
        )
    return {
        "schema": CALIBRATION_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "binding": binding_value,
        "binding_sha256": binding_sha256,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint["sha256"],
        "conditions": list(CALIBRATION_EXPECTATIONS),
        "trials_per_condition": trials_per_condition,
        "completed_trials": sum(len(records[name]) for name in records),
        "pass_counts": pass_counts,
        "calibration": calibration,
        "production_calibration": not smoke,
        "calibration_passed": bool(calibration["calibration_passed"]),
        "HEST_molecular_execution_authorized": bool(
            not smoke and calibration["HEST_molecular_execution_authorized"]
        ),
        "molecular_outcomes_read": False,
        "biological_experiment_run": False,
        "engineering_decision_available": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


def calibrate(
    *,
    source: Path,
    preflight_receipt: Mapping[str, object],
    checkpoint_path: Path,
    trials_per_condition: int = PRODUCTION_TRIALS,
    smoke: bool = False,
) -> Mapping[str, object]:
    if (
        preflight_receipt.get("schema") != PREFLIGHT_SCHEMA
        or preflight_receipt.get("preflight_passed") is not True
        or preflight_receipt.get("molecular_outcomes_read") is not False
        or preflight_receipt.get("synthetic_calibration_authorized") is not True
    ):
        raise ValueError("synthetic calibration requires a passing outcome-locked preflight")
    topology, topology_receipt = load_calibration_topology(source)
    binding = {
        "protocol_sha256": preflight_receipt["protocol_sha256"],
        "manifest_sha256": preflight_receipt["manifest_sha256"],
        "registered_source_sha256": SOURCE_SHA256,
        "registration_topology_sha256": TOPOLOGY_RECEIPT_SHA256,
        "preflight_sha256": canonical_sha256(preflight_receipt),
        "implementation": preflight_receipt["implementation"],
        "git_commit": preflight_receipt["runtime"]["git_commit"],
        "topology_access": topology_receipt,
    }
    return run_calibration_trials(
        topology,
        binding=binding,
        checkpoint_path=checkpoint_path,
        trials_per_condition=trials_per_condition,
        smoke=smoke,
    )


def implementation_smoke(
    *,
    preflight_receipt: Mapping[str, object],
    checkpoint_dir: Path,
    device: str,
) -> Mapping[str, object]:
    """Run the separate nested-fit and complete-refit smoke without HEST outcomes."""

    if (
        preflight_receipt.get("schema") != PREFLIGHT_SCHEMA
        or preflight_receipt.get("preflight_passed") is not True
        or preflight_receipt.get("molecular_outcomes_read") is not False
        or preflight_receipt.get("runtime", {}).get("git_worktree_dirty") is not False
    ):
        raise ValueError("implementation smoke requires a clean passing preflight")
    binding = {
        "protocol_sha256": preflight_receipt["protocol_sha256"],
        "manifest_sha256": preflight_receipt["manifest_sha256"],
        "registered_source_sha256": SOURCE_SHA256,
        "preflight_sha256": canonical_sha256(preflight_receipt),
        "implementation": preflight_receipt["implementation"],
        "git_commit": preflight_receipt["runtime"]["git_commit"],
        "device": device,
        "permutations_per_family": 20,
    }
    result = dict(run_implementation_smoke(checkpoint_dir, device=device))
    _validate_implementation_smoke_result(result)
    return {
        **result,
        "binding": binding,
        "binding_sha256": canonical_sha256(binding),
    }


def _validate_implementation_smoke_result(value: Mapping[str, object]) -> None:
    nulls = value.get("nulls")
    checkpoint = value.get("checkpoint_receipt")
    candidate_ids = value.get("candidate_ids")
    rank_candidates = value.get("rank_candidates")
    if (
        value.get("schema") != IMPLEMENTATION_SMOKE_SCHEMA
        or value.get("smoke_passed") is not True
        or value.get("checkpoint_replay_identical") is not True
        or value.get("permutations_per_family") != 20
        or value.get("complete_refit_steps") != list(SMOKE_COMPLETE_REFIT_STEPS)
        or value.get("molecular_outcomes_read") is not False
        or value.get("biological_experiment_run") is not False
        or value.get("engineering_decision_available") is not False
        or any(
            value.get(name) is not False
            for name in (
                "authorizes_h_cell",
                "authorizes_h_intrinsic",
                "authorizes_h_ref",
                "authorizes_full_heir",
            )
        )
        or not isinstance(candidate_ids, list)
        or len(candidate_ids) < 2
        or any(not isinstance(candidate_id, str) for candidate_id in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
        or value.get("selected_candidate_id") not in candidate_ids
        or not isinstance(value.get("selected_epoch"), int)
        or rank_candidates != [2, 4]
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != "heir.neural_probe_checkpoint_receipt.v1"
        or any(
            not isinstance(checkpoint.get(name), str) or len(str(checkpoint[name])) != 64
            for name in ("state_sha256", "array_registry_sha256", "metadata_sha256")
        )
        or not isinstance(nulls, Mapping)
        or set(nulls)
        != {
            "within_section_type_derangement",
            "different_spatial_block_reassignment",
        }
    ):
        raise ValueError("v2 nested implementation smoke failed closed")
    for result in nulls.values():
        empirical_p = result.get("empirical_p") if isinstance(result, Mapping) else None
        if (
            not isinstance(result, Mapping)
            or result.get("permutations") != 20
            or result.get("model_refit_for_every_permutation") is not True
            or result.get("completed_steps") != list(SMOKE_COMPLETE_REFIT_STEPS)
            or not isinstance(result.get("mapping_set_sha256"), str)
            or len(str(result["mapping_set_sha256"])) != 64
            or not isinstance(result.get("refit_receipts_sha256"), str)
            or len(str(result["refit_receipts_sha256"])) != 64
            or isinstance(empirical_p, bool)
            or not isinstance(empirical_p, (int, float))
            or not np.isfinite(empirical_p)
            or not 0.0 <= float(empirical_p) <= 1.0
        ):
            raise ValueError("implementation-smoke null receipt is incomplete")


def validate_biological_authorization(
    value: Mapping[str, object],
    *,
    device: Optional[str] = None,
) -> Mapping[str, object]:
    """Validate the fail-closed authorization shared by data and full-run APIs."""

    authorized_device = value.get("device")
    if (
        value.get("schema") != BIOLOGICAL_AUTHORIZATION_SCHEMA
        or value.get("registered_source_sha256") != SOURCE_SHA256
        or value.get("protocol_sha256") != V2_PROTOCOL_SHA256
        or value.get("biological_execution_authorized") is not True
        or value.get("molecular_outcomes_read") is not False
        or authorized_device not in {"cpu", "cuda", "cuda:0"}
        or (device is not None and authorized_device != device)
        or not isinstance(value.get("implementation"), Mapping)
        or not isinstance(value.get("git_commit"), str)
        or any(
            value.get(name) is not False
            for name in (
                "authorizes_h_cell",
                "authorizes_h_intrinsic",
                "authorizes_h_ref",
                "authorizes_full_heir",
            )
        )
    ):
        raise ValueError("HEST biological source requires a valid production authorization")
    return value


def authorize_biological_execution(
    *,
    source: Path,
    preflight_receipt: Mapping[str, object],
    implementation_smoke_receipt: Mapping[str, object],
    calibration_receipt: Mapping[str, object],
    device: Optional[str] = None,
) -> Mapping[str, object]:
    """Verify a production calibration and authorize exactly the bound full runner."""

    if (
        preflight_receipt.get("schema") != PREFLIGHT_SCHEMA
        or preflight_receipt.get("preflight_passed") is not True
        or preflight_receipt.get("molecular_outcomes_read") is not False
        or preflight_receipt.get("runtime", {}).get("git_worktree_dirty") is not False
    ):
        raise ValueError("biological execution requires a clean passing preflight")
    topology, topology_receipt = load_calibration_topology(source)
    del topology
    base_binding = {
        "protocol_sha256": preflight_receipt["protocol_sha256"],
        "manifest_sha256": preflight_receipt["manifest_sha256"],
        "registered_source_sha256": SOURCE_SHA256,
        "registration_topology_sha256": TOPOLOGY_RECEIPT_SHA256,
        "preflight_sha256": canonical_sha256(preflight_receipt),
        "implementation": preflight_receipt["implementation"],
        "git_commit": preflight_receipt["runtime"]["git_commit"],
        "topology_access": topology_receipt,
    }
    smoke_binding = implementation_smoke_receipt.get("binding")
    if not isinstance(smoke_binding, Mapping):
        raise ValueError("biological execution implementation-smoke binding is malformed")
    smoke_device = smoke_binding.get("device")
    if smoke_device not in {"cpu", "cuda", "cuda:0"} or (
        device is not None and smoke_device != device
    ):
        raise ValueError("full execution device differs from the implementation smoke")
    expected_smoke_binding = {
        "protocol_sha256": preflight_receipt["protocol_sha256"],
        "manifest_sha256": preflight_receipt["manifest_sha256"],
        "registered_source_sha256": SOURCE_SHA256,
        "preflight_sha256": canonical_sha256(preflight_receipt),
        "implementation": preflight_receipt["implementation"],
        "git_commit": preflight_receipt["runtime"]["git_commit"],
        "device": smoke_device,
        "permutations_per_family": 20,
    }
    if (
        implementation_smoke_receipt.get("schema") != IMPLEMENTATION_SMOKE_SCHEMA
        or implementation_smoke_receipt.get("binding") != expected_smoke_binding
        or implementation_smoke_receipt.get("binding_sha256")
        != canonical_sha256(expected_smoke_binding)
        or implementation_smoke_receipt.get("smoke_passed") is not True
        or implementation_smoke_receipt.get("checkpoint_replay_identical") is not True
        or implementation_smoke_receipt.get("permutations_per_family") != 20
        or implementation_smoke_receipt.get("molecular_outcomes_read") is not False
        or implementation_smoke_receipt.get("biological_experiment_run") is not False
    ):
        raise ValueError("biological execution implementation-smoke receipt is invalid")
    _validate_implementation_smoke_result(implementation_smoke_receipt)
    expected_binding = {
        **base_binding,
        "trials_per_condition": PRODUCTION_TRIALS,
        "smoke": False,
        "registration_bootstrap_resamples": 20_000,
    }
    binding_sha256 = canonical_sha256(expected_binding)
    if (
        calibration_receipt.get("schema") != CALIBRATION_SCHEMA
        or calibration_receipt.get("binding") != expected_binding
        or calibration_receipt.get("binding_sha256") != binding_sha256
        or calibration_receipt.get("production_calibration") is not True
        or calibration_receipt.get("trials_per_condition") != PRODUCTION_TRIALS
        or calibration_receipt.get("completed_trials")
        != PRODUCTION_TRIALS * len(CALIBRATION_EXPECTATIONS)
        or calibration_receipt.get("calibration_passed") is not True
        or calibration_receipt.get("HEST_molecular_execution_authorized") is not True
        or calibration_receipt.get("molecular_outcomes_read") is not False
        or calibration_receipt.get("biological_experiment_run") is not False
    ):
        raise ValueError("biological execution calibration receipt is not production-valid")
    checkpoint_path = Path(str(calibration_receipt.get("checkpoint_path", ""))).resolve()
    checkpoint = _load_checkpoint(checkpoint_path, binding_sha256)
    if checkpoint.get("sha256") != calibration_receipt.get("checkpoint_sha256"):
        raise ValueError("calibration receipt is bound to a different checkpoint")
    records = checkpoint.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("production calibration checkpoint records are malformed")
    pass_counts: dict[str, dict[str, int]] = {}
    for condition in CALIBRATION_EXPECTATIONS:
        local = records.get(condition)
        if not isinstance(local, list) or len(local) != PRODUCTION_TRIALS:
            raise ValueError("production calibration checkpoint is incomplete")
        if any(not isinstance(record, Mapping) for record in local):
            raise ValueError("production calibration trial record is malformed")
        if [record.get("trial_index") for record in local] != list(range(PRODUCTION_TRIALS)):
            raise ValueError("production calibration trial sequence is not contiguous")
        if any(type(record.get(name)) is not bool for record in local for name in ("D1", "D2")):
            raise ValueError("production calibration decisions are not booleans")
        pass_counts[condition] = {
            name: sum(record[name] for record in local) for name in ("D1", "D2")
        }
    expected_calibration = evaluate_synthetic_calibration(
        pass_counts,
        trials_per_condition=PRODUCTION_TRIALS,
    )
    if (
        calibration_receipt.get("pass_counts") != pass_counts
        or calibration_receipt.get("calibration") != expected_calibration
    ):
        raise ValueError("production calibration summary differs from its checkpoint")
    return {
        "schema": BIOLOGICAL_AUTHORIZATION_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "registered_source_sha256": SOURCE_SHA256,
        "protocol_sha256": V2_PROTOCOL_SHA256,
        "preflight_sha256": canonical_sha256(preflight_receipt),
        "calibration_receipt_sha256": canonical_sha256(calibration_receipt),
        "implementation_smoke_receipt_sha256": canonical_sha256(implementation_smoke_receipt),
        "calibration_checkpoint_sha256": checkpoint["sha256"],
        "implementation": preflight_receipt["implementation"],
        "git_commit": preflight_receipt["runtime"]["git_commit"],
        "device": smoke_device,
        "biological_execution_authorized": True,
        "molecular_outcomes_read": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_h_ref": False,
        "authorizes_full_heir": False,
    }


__all__ = [
    "BIOLOGICAL_AUTHORIZATION_SCHEMA",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_SOURCE_MEMBER_ALLOWLIST",
    "CHECKPOINT_SCHEMA",
    "EVALUATION_ROWS",
    "FORMAL_Q3_ROWS",
    "IMPLEMENTATION_SMOKE_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "PRODUCTION_TRIALS",
    "atomic_json",
    "authorize_biological_execution",
    "calibrate",
    "implementation_hashes",
    "implementation_smoke",
    "load_calibration_topology",
    "preflight",
    "run_calibration_trials",
    "validate_biological_authorization",
]
