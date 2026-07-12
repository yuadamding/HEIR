#!/usr/bin/env python3
"""Run the resumable five-seed redesigned snPATHO refinement benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from heir.data import PrototypeSet
from heir.inference import PredictionBundle, validate_wrong_donor_prototype_filter

SAMPLES = ("4066", "4399", "4411")
MOLECULAR_GENERATIONS = ("r1", "r2")
SAMPLE_SITES = {
    "4066": "primary_breast",
    "4399": "liver_metastasis",
    "4411": "liver_metastasis",
}
SEEDS = (17, 41, 89, 131, 197)
ABLATION_SEEDS = (17, 41, 89)
UNKNOWN_MASS_SENSITIVITY = (0.0, 0.01, 0.05, 0.10, 0.20)
DEFAULT_UOT_UNKNOWN_MASS = 0.05
PREDICTION_CONTROLS = (
    "round0_prototype_only",
    "refined_prototype_only",
    "image_shuffle",
    "graph_shuffle",
    "no_graph",
)
PROTOTYPE_ONLY_CONTROLS = frozenset(
    {"prototype_only", "round0_prototype_only", "refined_prototype_only"}
)
WRONG_PROTOTYPE_BANK_CONTROL = "wrong_prototype_bank"
LEGACY_WRONG_DONOR_CONTROL = "wrong_donor"
REFINEMENT_RUN_MANIFEST_SCHEMA = "heir.snpatho_refinement_run_manifest.v2"
LEGACY_FIVE_SEED_MANIFEST_SCHEMA = "heir.snpatho_five_seed_refinement_manifest.v1"
UNKNOWN_MASS_MANIFEST_SCHEMA = "heir.snpatho_unknown_mass_run_manifest.v1"
UNKNOWN_MASS_STAGE_NAMES = (
    "train_round0",
    "build_views",
    "refine",
    "predict_round0",
    "predict_refined",
)
UNKNOWN_MASS_SOURCE_FILES = (
    "scripts/run_snpatho_refinement_benchmark.py",
    "scripts/build_refinement_views.py",
    "src/heir/__main__.py",
    "src/heir/cli.py",
    "src/heir/inference.py",
    "src/heir/models/heir.py",
    "src/heir/refinement/iterative.py",
    "src/heir/training/trainer.py",
)
REFINEMENT_RUN_SOURCE_FILES = (
    *UNKNOWN_MASS_SOURCE_FILES,
    "scripts/benchmark_snpatho_refinement_matrix.py",
)
ENVIRONMENT_SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "environment.yml",
    "environment.yaml",
    "requirements.txt",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PlannedStage:
    """One independently resumable benchmark stage."""

    sample: str
    seed: int
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    validate: Callable[[], None]
    unknown_mass: Optional[float] = None
    inputs: tuple[tuple[str, Path], ...] = ()
    output_roles: tuple[str, ...] = ()
    control: Optional[str] = None
    prototype_donor_id: Optional[str] = None


def wrong_prototype_bank_pairings(samples: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return every directed target/source prototype-bank pairing without self-pairs."""

    unique = tuple(dict.fromkeys(str(sample) for sample in samples))
    return tuple((target, source) for target in unique for source in unique if source != target)


def wrong_donor_pairings(samples: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Compatibility alias for :func:`wrong_prototype_bank_pairings`."""

    return wrong_prototype_bank_pairings(samples)


def _sha256(path: Path) -> str:
    source = path.expanduser().resolve()
    stat = source.stat()
    return _sha256_cached(str(source), stat.st_size, stat.st_mtime_ns)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(
    repository: Path,
    relative_paths: Sequence[str],
) -> Mapping[str, Any]:
    repository = repository.expanduser().resolve()
    files = []
    for relative in relative_paths:
        path = (repository / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError("validation-recipe source file is absent: %s" % path)
        files.append(
            {
                "relative_path": relative,
                "path": relative,
                "sha256": _sha256(path),
            }
        )
    digest_rows = [
        {"relative_path": row["relative_path"], "sha256": row["sha256"]} for row in files
    ]
    environment = _runtime_environment_identity()
    return {
        "schema": "heir.source_identity.v1",
        "runner": dict(files[0]),
        "files": files,
        "runtime_environment": environment,
        "aggregate_sha256": _canonical_sha256(
            {
                "files": digest_rows,
                "runtime_environment_sha256": environment["aggregate_sha256"],
            }
        ),
    }


@lru_cache(maxsize=1)
def _runtime_environment_identity() -> Mapping[str, Any]:
    """Hash the exact installed Python and accelerator runtime used by subprocesses."""

    packages = sorted(
        (
            {
                "name": str(distribution.metadata.get("Name", "")).strip(),
                "version": str(distribution.version),
            }
            for distribution in importlib.metadata.distributions()
            if str(distribution.metadata.get("Name", "")).strip()
        ),
        key=lambda row: (row["name"].lower(), row["version"]),
    )
    accelerator: dict[str, Any] = {"torch_importable": False}
    try:
        import torch

        accelerator = {
            "torch_importable": True,
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            accelerator.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                accelerator["nvidia_driver_versions"] = sorted(
                    set(line.strip() for line in completed.stdout.splitlines() if line.strip())
                )
            except (OSError, subprocess.SubprocessError):
                accelerator["nvidia_driver_versions"] = []
    except ImportError:
        pass
    payload = {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "package_count": len(packages),
        "accelerator": accelerator,
    }
    return {**payload, "aggregate_sha256": _canonical_sha256(payload)}


def _runtime_source_files(
    repository: Path,
    entrypoints: Sequence[str],
) -> tuple[str, ...]:
    """Return entrypoints plus the complete importable HEIR and environment recipe."""

    repository = repository.expanduser().resolve()
    candidates = list(entrypoints)
    source_root = repository / "src" / "heir"
    if source_root.is_dir():
        candidates.extend(
            str(path.relative_to(repository))
            for path in sorted(source_root.rglob("*.py"))
            if path.is_file()
        )
    candidates.extend(
        relative
        for relative in ENVIRONMENT_SOURCE_FILES
        if (repository / relative).is_file()
    )
    return tuple(dict.fromkeys(candidates))


def unknown_mass_source_identity(repository: Path) -> Mapping[str, Any]:
    """Return the exact source inventory used to validate the sensitivity plan."""

    return _source_identity(
        repository,
        _runtime_source_files(repository, UNKNOWN_MASS_SOURCE_FILES),
    )


def refinement_run_source_identity(repository: Path) -> Mapping[str, Any]:
    """Return the current full-matrix planning, validation, and scoring recipe."""

    return _source_identity(
        repository,
        _runtime_source_files(repository, REFINEMENT_RUN_SOURCE_FILES),
    )


@lru_cache(maxsize=512)
def _sha256_cached(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, *, schema: Optional[str] = None) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON artifact %s: %s" % (path, error)) from error
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must contain an object: %s" % path)
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(
            "JSON artifact %s has schema %r, expected %r" % (path, payload.get("schema"), schema)
        )
    return payload


def _repository_path(repository: Path, *parts: str) -> Path:
    return repository.joinpath(*parts).expanduser().resolve()


def _output_path(repository: Path, value: Path) -> Path:
    expanded = value.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (repository / expanded).resolve()


def _heir_source_binding(
    repository: Path,
    *,
    require_sources: bool = False,
) -> Mapping[str, Any]:
    """Bind CLI subprocesses to this interpreter and repository source tree."""

    repository = repository.expanduser().resolve()
    source_root = _repository_path(repository, "src")
    entrypoint = _repository_path(source_root, "heir", "__main__.py")
    cli_source = _repository_path(source_root, "heir", "cli.py")
    # Preserve the environment entrypoint instead of resolving a venv/conda
    # symlink to its base interpreter; isolated mode must retain this
    # environment's site-packages while ignoring ambient import paths.
    interpreter = Path(os.path.abspath(sys.executable)).expanduser()
    for label, path in (
        ("Python interpreter", interpreter),
        ("HEIR module entrypoint", entrypoint),
        ("HEIR CLI source", cli_source),
    ):
        if require_sources and not path.is_file():
            raise FileNotFoundError("%s is absent: %s" % (label, path))
    bootstrap = (
        "import runpy,sys;"
        "sys.path.insert(0,%s);"
        "runpy.run_module('heir',run_name='__main__')"
        % json.dumps(str(source_root), ensure_ascii=True)
    )
    command_prefix = (str(interpreter), "-I", "-c", bootstrap)
    return {
        "schema": "heir.source_bound_cli.v1",
        "mode": "isolated_python_explicit_repository_src_run_module",
        "python_executable": str(interpreter),
        "python_executable_sha256": _sha256(interpreter),
        "source_root": str(source_root),
        "module_entrypoint": str(entrypoint),
        "module_entrypoint_sha256": _sha256(entrypoint) if entrypoint.is_file() else None,
        "cli_source": str(cli_source),
        "cli_source_sha256": _sha256(cli_source) if cli_source.is_file() else None,
        "bootstrap_sha256": hashlib.sha256(bootstrap.encode("utf-8")).hexdigest(),
        "command_prefix": list(command_prefix),
    }


def _heir_source_command(repository: Path, *arguments: str) -> list[str]:
    binding = _heir_source_binding(repository)
    return [*(str(value) for value in binding["command_prefix"]), *arguments]


def _repository_script_command(
    repository: Path,
    script_relative_path: str,
    *arguments: str,
) -> tuple[str, ...]:
    """Run one repository script in isolated mode against this exact source tree."""

    repository = repository.expanduser().resolve()
    source_root = _repository_path(repository, "src")
    script = _repository_path(repository, *script_relative_path.split("/"))
    interpreter = Path(os.path.abspath(sys.executable)).expanduser()
    bootstrap = (
        "import runpy,sys;"
        "sys.path.insert(0,%s);"
        "sys.argv=[%s]+sys.argv[1:];"
        "runpy.run_path(%s,run_name='__main__')"
        % (
            json.dumps(str(source_root), ensure_ascii=True),
            json.dumps(str(script), ensure_ascii=True),
            json.dumps(str(script), ensure_ascii=True),
        )
    )
    return (str(interpreter), "-I", "-c", bootstrap, *arguments)


def _run(
    command: Sequence[str],
    outputs: Sequence[Path],
    execute: bool,
    *,
    validator: Callable[[], None],
    repository: Path,
) -> str:
    present = [path.is_file() for path in outputs]
    if all(present):
        try:
            validator()
        except Exception as error:
            raise RuntimeError("existing stage outputs are invalid: %s" % error) from error
        return "skipped_valid"
    if any(present):
        raise RuntimeError("partial stage output exists: %s" % ", ".join(map(str, outputs)))
    if not execute:
        print(shlex.join(command))
        return "planned"
    subprocess.run(command, check=True, cwd=repository)
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("stage completed without outputs: %s" % ", ".join(map(str, missing)))
    try:
        validator()
    except Exception as error:
        raise RuntimeError("stage produced invalid outputs: %s" % error) from error
    return "completed"


def _scalar(archive: Any, name: str) -> str:
    if name not in archive:
        raise ValueError("NPZ artifact is missing %s" % name)
    return str(np.asarray(archive[name]).item())


def _string_vector(archive: Any, name: str) -> tuple[str, ...]:
    if name not in archive:
        raise ValueError("NPZ artifact is missing %s" % name)
    values = np.asarray(archive[name])
    if values.ndim != 1:
        raise ValueError("NPZ field %s must be one-dimensional" % name)
    return tuple(str(value) for value in values.tolist())


def _batch_identity(path: Path) -> Mapping[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if _scalar(archive, "__contract__") != "heir.training_batch":
            raise ValueError("artifact is not a HEIR training batch: %s" % path)
        version = int(np.asarray(archive["__version__"]).item())
        if version not in {2, 3, 4, 5}:
            raise ValueError("unsupported training-batch version %d" % version)
        result = {
            name: _scalar(archive, name)
            for name in (
                "sample_id",
                "bag_id",
                "donor_id",
                "block_id",
                "analysis_role",
            )
        }
        for name in ("source_artifacts", "source_sha256", "source_roles"):
            result[name] = _string_vector(archive, name)
    if not result["source_sha256"] or any(
        _SHA256_PATTERN.fullmatch(value) is None for value in result["source_sha256"]
    ):
        raise ValueError("training batch has invalid source SHA-256 provenance")
    return result


def _checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    source = path.expanduser().resolve()
    stat = source.stat()
    return _checkpoint_metadata_cached(str(source), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=128)
def _checkpoint_metadata_cached(path: str, size: int, modified_ns: int) -> Mapping[str, Any]:
    del size, modified_ns
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("invalid HEIR checkpoint %s: %s" % (path, error)) from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping: %s" % path)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint lacks metadata: %s" % path)
    return dict(metadata)


def _assert_batch_metadata(
    rows: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError("checkpoint %s metadata must contain exactly one batch" % label)
    row = rows[0]
    for name in ("sample_id", "bag_id", "donor_id", "block_id"):
        if str(row.get(name, "")) != expected[name]:
            raise ValueError(
                "checkpoint %s batch %s does not match the current batch" % (label, name)
            )
    for name in ("source_sha256", "source_roles"):
        if tuple(str(value) for value in row.get(name, ())) != expected[name]:
            raise ValueError(
                "checkpoint %s batch %s does not match the current batch" % (label, name)
            )
    if "analysis_role" in row and str(row["analysis_role"]) != expected["analysis_role"]:
        raise ValueError("checkpoint %s batch analysis_role does not match" % label)
    if (
        "source_artifacts" in row
        and tuple(str(value) for value in row["source_artifacts"]) != expected["source_artifacts"]
    ):
        raise ValueError("checkpoint %s batch source_artifacts do not match" % label)


def _validate_fixed_unknown_mass(
    metadata: Mapping[str, Any],
    expected_unknown_mass: float,
    *,
    label: str,
) -> None:
    """Reject legacy or mismatched checkpoints before stage adoption."""

    clean_root_guidance = (
        "regenerate this case under a clean output root; do not reuse or adopt the legacy "
        "stage directory"
    )
    if "uot_unknown_mass" not in metadata or "uot_unknown_mass_mode" not in metadata:
        raise ValueError(
            "%s checkpoint lacks complete fixed unknown-mass metadata; %s"
            % (label, clean_root_guidance)
        )
    try:
        observed = float(metadata["uot_unknown_mass"])
        expected = float(expected_unknown_mass)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "%s checkpoint has invalid unknown-mass metadata; %s" % (label, clean_root_guidance)
        ) from error
    if metadata["uot_unknown_mass_mode"] != "fixed" or not math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "%s checkpoint unknown-mass metadata is not fixed at %.12g; %s"
            % (label, expected, clean_root_guidance)
        )


def _validate_trained_pair(
    checkpoint: Path,
    history: Path,
    *,
    sample: str,
    seed: int,
    train_batch: Path,
    validation_batch: Path,
    decoder: Path,
    residual_geometry: Path,
    expected_unknown_mass: float,
) -> None:
    metadata = _checkpoint_metadata(checkpoint)
    if metadata.get("schema") != "heir.trained_model.v1":
        raise ValueError("round-zero checkpoint has an unsupported schema")
    if metadata.get("training_stage") != "personalized":
        raise ValueError("round-zero checkpoint is not a personalized model")
    if metadata.get("seed") != seed:
        raise ValueError("round-zero checkpoint seed does not match the requested seed")
    if set(str(value) for value in metadata.get("training_donors", ())) != {sample}:
        raise ValueError("round-zero checkpoint donor does not match the requested sample")
    _validate_fixed_unknown_mass(
        metadata,
        expected_unknown_mass,
        label="round-zero",
    )
    _assert_batch_metadata(
        metadata.get("training_batches"),
        _batch_identity(train_batch),
        label="training",
    )
    _assert_batch_metadata(
        metadata.get("validation_batches"),
        _batch_identity(validation_batch),
        label="validation",
    )
    if metadata.get("rna_vae_sha256") != _sha256(decoder):
        raise ValueError("round-zero checkpoint does not bind the current RNA decoder")
    if metadata.get("residual_geometry_sha256") != _sha256(residual_geometry):
        raise ValueError("round-zero checkpoint does not bind the current residual geometry")

    payload = _json_object(history)
    rows = payload.get("history")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("training history must contain non-empty epoch records")
    if payload.get("best_epoch") != metadata.get("best_epoch"):
        raise ValueError("history best_epoch does not match its checkpoint")
    try:
        paired_loss = math.isclose(
            float(payload["best_validation_loss"]),
            float(metadata["best_validation_loss"]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("history has invalid best-validation metadata") from error
    if not paired_loss:
        raise ValueError("history best_validation_loss does not match its checkpoint")


def _validate_refinement_views(
    output: Path,
    *,
    checkpoint: Path,
    batch: Path,
    sample: str,
) -> None:
    try:
        with np.load(output, allow_pickle=False) as archive:
            required = {
                "nucleus_ids",
                "view_predictions",
                "view_ids",
                "view_source_sha256",
                "metadata_json",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("refinement views are missing: %s" % ", ".join(missing))
            metadata = json.loads(_scalar(archive, "metadata_json"))
            nucleus_ids = np.asarray(archive["nucleus_ids"])
            predictions = np.asarray(archive["view_predictions"])
            view_ids = _string_vector(archive, "view_ids")
            source_hashes = _string_vector(archive, "view_source_sha256")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid refinement-view artifact %s: %s" % (output, error)) from error
    if not isinstance(metadata, dict) or metadata.get("schema") not in {
        "heir.refinement_views.v1",
        "heir.refinement_views.v2",
    }:
        raise ValueError("refinement views have an unsupported metadata schema")
    if metadata.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("refinement views do not bind the current checkpoint")
    if metadata.get("batch_sha256") != _sha256(batch):
        raise ValueError("refinement views do not bind the current training batch")
    if metadata.get("schema") == "heir.refinement_views.v2" and (
        metadata.get("sample_id") != sample or metadata.get("donor_id") != sample
    ):
        raise ValueError("refinement views do not match the requested sample")
    if (
        predictions.ndim != 3
        or predictions.shape[0] < 2
        or predictions.shape[1] != len(nucleus_ids)
    ):
        raise ValueError("refinement view predictions have an invalid shape")
    if not np.isfinite(predictions).all():
        raise ValueError("refinement view predictions must be finite")
    if len(view_ids) != predictions.shape[0] or len(source_hashes) != predictions.shape[0]:
        raise ValueError("refinement view identities do not align to predictions")
    if len(set(str(value) for value in nucleus_ids.tolist())) != len(nucleus_ids):
        raise ValueError("refinement view nucleus IDs must be unique")
    if any(_SHA256_PATTERN.fullmatch(value) is None for value in source_hashes):
        raise ValueError("refinement view source hashes are invalid")


def _prototype_identity(path: Path) -> Mapping[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if _scalar(archive, "__contract__") != "heir.prototype_set":
            raise ValueError("artifact is not a HEIR PrototypeSet: %s" % path)
        donor_id = _scalar(archive, "donor_id")
        sample_ids = _string_vector(archive, "sample_ids")
        latent_space_id = _scalar(archive, "latent_space_id")
    if not donor_id or not sample_ids or not latent_space_id:
        raise ValueError("prototype artifact lacks required provenance")
    return {"donor_id": donor_id, "sample_ids": sample_ids, "latent_space_id": latent_space_id}


def _validate_refined_pair(
    checkpoint: Path,
    audit: Path,
    prototype: Path,
    *,
    sample: str,
    seed: int,
    parent_checkpoint: Path,
    view_artifact: Path,
    train_batch: Path,
    validation_batch: Path,
    round_checkpoints: Sequence[Path],
    expected_unknown_mass: float,
) -> None:
    metadata = _checkpoint_metadata(checkpoint)
    if metadata.get("schema") != "heir.refined_model.v1":
        raise ValueError("refined checkpoint has an unsupported schema")
    if metadata.get("seed") != seed:
        raise ValueError("refined checkpoint seed does not match the requested seed")
    if metadata.get("parent_checkpoint_sha256") != _sha256(parent_checkpoint):
        raise ValueError("refined checkpoint does not bind the current round-zero checkpoint")
    if set(str(value) for value in metadata.get("refinement_training_donors", ())) != {sample}:
        raise ValueError("refined checkpoint donor does not match the requested sample")
    _validate_fixed_unknown_mass(
        metadata,
        expected_unknown_mass,
        label="refined",
    )
    _assert_batch_metadata(
        metadata.get("refinement_training_batches"),
        _batch_identity(train_batch),
        label="refinement training",
    )
    _assert_batch_metadata(
        metadata.get("refinement_validation_batches"),
        _batch_identity(validation_batch),
        label="refinement validation",
    )
    view_rows = metadata.get("refinement_view_artifacts")
    if not isinstance(view_rows, list) or len(view_rows) != 1:
        raise ValueError("refined checkpoint must bind exactly one view artifact")
    view_row = view_rows[0]
    if not isinstance(view_row, Mapping) or view_row.get("sha256") != _sha256(view_artifact):
        raise ValueError("refined checkpoint does not bind the current view artifact")

    payload = _json_object(audit)
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("refinement audit does not contain a round trajectory")
    expected_round_ids = list(range(1, len(rounds) + 1))
    if [row.get("round_id") for row in rounds] != expected_round_ids:
        raise ValueError("refinement audit round IDs are not consecutive")
    if round_checkpoints and expected_round_ids != [1, 2, 3, 4]:
        raise ValueError(
            "primary refinement audit does not contain the fixed four-round trajectory"
        )
    if len(rounds) > 4:
        raise ValueError("refinement audit exceeds the prespecified four-round maximum")
    if rounds != metadata.get("refinement_rounds"):
        raise ValueError("refinement audit trajectory does not match its checkpoint")
    if payload.get("selected_round") != metadata.get("refinement_round"):
        raise ValueError("refinement audit selected round does not match its checkpoint")
    if payload.get("stopped_reason") != metadata.get("refinement_stopped_reason"):
        raise ValueError("refinement audit stop reason does not match its checkpoint")
    if metadata.get("refinement_rounds_executed") != len(rounds):
        raise ValueError("refined checkpoint has an inconsistent round count")
    try:
        zero_loss_matches = math.isclose(
            float(payload["round_zero_validation_loss"]),
            float(metadata["refinement_round_zero_validation_loss"]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("refinement audit has invalid round-zero metadata") from error
    if not zero_loss_matches:
        raise ValueError("refinement audit round-zero loss does not match its checkpoint")

    prototype_rows = payload.get("prototype_artifacts")
    key = "%s::%s" % (sample, sample)
    if not isinstance(prototype_rows, Mapping) or key not in prototype_rows:
        raise ValueError("refinement audit does not identify its prototype artifact")
    if Path(str(prototype_rows[key])).expanduser().resolve() != prototype.resolve():
        raise ValueError("refinement audit points to a different prototype artifact")
    prototype_metadata = _prototype_identity(prototype)
    if prototype_metadata["donor_id"] != sample or set(prototype_metadata["sample_ids"]) != {
        sample
    }:
        raise ValueError("refined prototype artifact does not match the requested sample")

    round_rows = payload.get("round_checkpoints")
    if not isinstance(round_rows, Mapping):
        raise ValueError("refinement audit round_checkpoints must be an object")
    expected_keys = {str(index) for index in range(1, len(round_checkpoints) + 1)}
    if set(round_rows) != expected_keys:
        raise ValueError("refinement audit has incomplete or unexpected round checkpoints")
    selected_round = int(payload["selected_round"])
    for round_id, round_path in enumerate(round_checkpoints, start=1):
        if Path(str(round_rows[str(round_id)])).expanduser().resolve() != round_path.resolve():
            raise ValueError(
                "refinement audit points to a different round-%d checkpoint" % round_id
            )
        round_metadata = _checkpoint_metadata(round_path)
        if round_metadata.get("schema") != "heir.refined_round_model.v1":
            raise ValueError("round-%d checkpoint has an unsupported schema" % round_id)
        if round_metadata.get("seed") != seed or round_metadata.get("refinement_round") != round_id:
            raise ValueError("round-%d checkpoint identity is stale" % round_id)
        if round_metadata.get("parent_checkpoint_sha256") != _sha256(parent_checkpoint):
            raise ValueError("round-%d checkpoint has a stale parent" % round_id)
        _validate_fixed_unknown_mass(
            round_metadata,
            expected_unknown_mass,
            label="round-%d" % round_id,
        )
        if bool(round_metadata.get("selected_by_parent_run")) != (round_id == selected_round):
            raise ValueError("round-%d checkpoint selection flag is inconsistent" % round_id)
        if bool(round_metadata.get("refinement_round_committed")) != bool(
            rounds[round_id - 1].get("committed")
        ):
            raise ValueError("round-%d checkpoint commit flag is inconsistent" % round_id)


def _control_flags(control: Optional[str]) -> Mapping[str, bool]:
    telemetry_names = {
        "image_shuffle": "image_feature_shuffle",
        "graph_shuffle": "graph_node_shuffle",
        "no_graph": "no_graph",
        WRONG_PROTOTYPE_BANK_CONTROL: "wrong_donor",
        LEGACY_WRONG_DONOR_CONTROL: "wrong_donor",
    }
    if control in PROTOTYPE_ONLY_CONTROLS:
        active_telemetry_name = "prototype_only"
    elif control is None:
        active_telemetry_name = None
    else:
        active_telemetry_name = telemetry_names.get(control)
    if control is not None and active_telemetry_name is None:
        raise ValueError("unknown prediction control %s" % control)
    return {
        telemetry_name: telemetry_name == active_telemetry_name
        for telemetry_name in (
            "prototype_only",
            "image_feature_shuffle",
            "graph_node_shuffle",
            "no_graph",
            "wrong_donor",
        )
    }


def _expected_shuffle_transform(
    control: str,
    seed: int,
    histology: Path,
) -> Mapping[str, Any]:
    algorithms = {
        "image_shuffle": (
            "apply default_rng(seed).permutation(n_nuclei) to histology feature rows"
        ),
        "graph_shuffle": (
            "apply default_rng(seed).permutation(n_nuclei) to graph edge endpoint indices"
        ),
    }
    if control not in algorithms:
        raise ValueError("unsupported shuffle control %s" % control)
    with np.load(histology, allow_pickle=False) as archive:
        nucleus_ids = np.asarray(archive["nucleus_ids"])
    permutation = np.asarray(
        np.random.default_rng(seed).permutation(len(nucleus_ids)),
        dtype="<i8",
    )
    map_sha256 = hashlib.sha256(permutation.tobytes(order="C")).hexdigest()
    recipe = {
        "schema": "heir.inference_control_transform.v1",
        "control": control,
        "seed": int(seed),
        "random_generator": "numpy.random.default_rng",
        "algorithm": algorithms[control],
        "nuclei": int(len(nucleus_ids)),
        "map_encoding": "little-endian-int64-c-order",
        "expected_transform_map_sha256": map_sha256,
    }
    return {
        **recipe,
        "recipe_sha256": _canonical_sha256(recipe),
        "map_sha256": map_sha256,
    }


def _validate_prediction(
    output: Path,
    telemetry: Path,
    *,
    sample: str,
    seed: int,
    checkpoint: Path,
    prototypes: Path,
    histology: Path,
    ood_artifact: Path,
    refinement_round: Optional[int],
    control: Optional[str] = None,
    prototype_donor_id: Optional[str] = None,
) -> None:
    try:
        prediction = PredictionBundle.from_npz(output)
    except Exception as error:
        raise ValueError("invalid PredictionBundle %s: %s" % (output, error)) from error
    expected = {
        "checkpoint_sha256": _sha256(checkpoint),
        "prototype_sha256": _sha256(prototypes),
        "histology_sha256": _sha256(histology),
        "ood_sha256": _sha256(ood_artifact),
    }
    for name, value in expected.items():
        if getattr(prediction, name) != value:
            raise ValueError("prediction %s does not match the current input artifact" % name)
    if prediction.sample_id != sample or prediction.donor_id != sample:
        raise ValueError("prediction sample/donor provenance is stale")
    if prediction.inference_seed != seed:
        raise ValueError("prediction seed does not match the requested seed")
    expected_round = refinement_round
    if expected_round is None:
        expected_round = int(_checkpoint_metadata(checkpoint).get("refinement_round", 0))
    if prediction.refinement_round != expected_round:
        raise ValueError("prediction refinement round does not match its checkpoint")
    if prediction.latent_samples != 20:
        raise ValueError("prediction latent-sample setting is stale")
    if not math.isclose(float(prediction.probability_threshold), 0.35):
        raise ValueError("prediction probability threshold is stale")
    if not math.isclose(float(prediction.artifact_threshold), 0.50):
        raise ValueError("prediction artifact threshold is stale")

    report = _json_object(telemetry, schema="heir.inference_telemetry.v1")
    if report.get("prediction_sha256") != _sha256(output):
        raise ValueError("prediction telemetry hash does not match the PredictionBundle")
    try:
        reported_path = Path(str(report["prediction_path"])).expanduser().resolve()
    except KeyError as error:
        raise ValueError("prediction telemetry lacks prediction_path") from error
    if reported_path != output.resolve():
        raise ValueError("prediction telemetry points to a different output")
    if report.get("nuclei") != len(prediction.nucleus_ids):
        raise ValueError("prediction telemetry nucleus count is stale")
    if report.get("genes") != len(prediction.gene_names):
        raise ValueError("prediction telemetry gene count is stale")
    if report.get("latent_samples") != 20 or report.get("mc_chunk_size") != 8:
        raise ValueError("prediction telemetry Monte Carlo settings are stale")
    negative = report.get("negative_control")
    if not isinstance(negative, Mapping):
        raise ValueError("prediction telemetry lacks negative-control provenance")
    for name, enabled in _control_flags(control).items():
        if negative.get(name) is not enabled:
            raise ValueError("prediction telemetry has stale control flag %s" % name)
    expected_donor = prototype_donor_id or sample
    is_wrong_bank = control in {WRONG_PROTOTYPE_BANK_CONTROL, LEGACY_WRONG_DONOR_CONTROL}
    if is_wrong_bank and expected_donor == sample:
        raise ValueError("wrong-prototype-bank validation requires a non-matched prototype donor")
    if not is_wrong_bank and expected_donor != sample:
        raise ValueError("matched prediction validation cannot use a non-matched prototype donor")
    if negative.get("prototype_donor_id") != expected_donor or negative.get("seed") != seed:
        raise ValueError("prediction telemetry has stale control donor/seed provenance")
    transform = negative.get("transform")
    if control in {"image_shuffle", "graph_shuffle"}:
        expected_transform = _expected_shuffle_transform(control, seed, histology)
        if not isinstance(transform, Mapping):
            raise ValueError("shuffle prediction telemetry lacks deterministic transform hashes")
        for name, expected_value in expected_transform.items():
            if transform.get(name) != expected_value:
                raise ValueError("shuffle prediction telemetry has stale transform %s" % name)
    elif transform is not None:
        raise ValueError("non-shuffle prediction telemetry unexpectedly reports a transform")
    if is_wrong_bank:
        source_prototypes = PrototypeSet.load_npz(prototypes)
        validate_wrong_donor_prototype_filter(
            source_prototypes,
            prediction.type_names.tolist(),
            prediction.prototype_ids.tolist(),
            negative.get("prototype_filter"),
            source_sha256=_sha256(prototypes),
        )
    elif negative.get("prototype_filter") is not None:
        raise ValueError("matched prediction telemetry unexpectedly reports prototype filtering")


def _predict_command(
    repository: Path,
    sample: str,
    seed: int,
    checkpoint: Path,
    prototypes: Path,
    output: Path,
    telemetry: Path,
    control: Optional[str] = None,
) -> list[str]:
    command = _heir_source_command(
        repository,
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--histology",
        str(_repository_path(repository, "artifacts", "snpatho", sample, "histology_full.npz")),
        "--prototypes",
        str(prototypes),
        "--genes",
        str(_repository_path(repository, "manifests", "gene_panel_snpatho_500.tsv")),
        "--output",
        str(output),
        "--telemetry-output",
        str(telemetry),
        "--latent-samples",
        "20",
        "--mc-chunk-size",
        "8",
        "--probability-threshold",
        "0.35",
        "--artifact-threshold",
        "0.50",
        "--sample-id",
        sample,
        "--donor-id",
        sample,
        "--ood-artifact",
        str(
            _repository_path(
                repository,
                "artifacts",
                "snpatho",
                sample,
                "ood_target_calibrated.npz",
            )
        ),
        "--mixed-precision",
        "--seed",
        str(seed),
        "--device",
        "cuda",
    )
    flags = {
        "prototype_only": "--prototype-only",
        "round0_prototype_only": "--prototype-only",
        "refined_prototype_only": "--prototype-only",
        "image_shuffle": "--image-feature-shuffle",
        "graph_shuffle": "--graph-node-shuffle",
        "no_graph": "--no-graph",
        WRONG_PROTOTYPE_BANK_CONTROL: "--wrong-donor-control",
        LEGACY_WRONG_DONOR_CONTROL: "--wrong-donor-control",
    }
    if control is not None:
        command.append(flags[control])
    return command


def _unknown_mass_label(value: float) -> str:
    return ("%.2f" % value).replace(".", "p")


def _model_directories(
    root: Path,
    seed: int,
    unknown_mass: Optional[float],
) -> tuple[Path, Path]:
    suffix = "" if unknown_mass is None else "_unknown_mass_%s" % _unknown_mass_label(unknown_mass)
    prefix = "model_refinement_r1_v1_seed%d%s" % (seed, suffix)
    return root / (prefix + "_round0"), root / (prefix + "_refined")


def _prediction_stage(
    repository: Path,
    *,
    sample: str,
    seed: int,
    name: str,
    checkpoint: Path,
    prototypes: Path,
    directory: Path,
    refinement_round: Optional[int],
    unknown_mass: Optional[float],
    control: Optional[str] = None,
    prototype_donor_id: Optional[str] = None,
) -> PlannedStage:
    output = directory / "predictions.npz"
    telemetry = directory / "prediction.telemetry.json"
    histology = _repository_path(repository, "artifacts", "snpatho", sample, "histology_full.npz")
    ood = _repository_path(
        repository,
        "artifacts",
        "snpatho",
        sample,
        "ood_target_calibrated.npz",
    )
    genes = _repository_path(repository, "manifests", "gene_panel_snpatho_500.tsv")
    return PlannedStage(
        sample=sample,
        seed=seed,
        name=name,
        command=tuple(
            _predict_command(
                repository,
                sample,
                seed,
                checkpoint,
                prototypes,
                output,
                telemetry,
                control,
            )
        ),
        outputs=(output, telemetry),
        validate=partial(
            _validate_prediction,
            output,
            telemetry,
            sample=sample,
            seed=seed,
            checkpoint=checkpoint,
            prototypes=prototypes,
            histology=histology,
            ood_artifact=ood,
            refinement_round=refinement_round,
            control=control,
            prototype_donor_id=prototype_donor_id,
        ),
        unknown_mass=unknown_mass,
        inputs=(
            ("checkpoint", checkpoint),
            ("prototype", prototypes),
            ("histology", histology),
            ("gene_panel", genes),
            ("ood", ood),
        ),
        output_roles=("prediction", "telemetry"),
        control=control,
        prototype_donor_id=prototype_donor_id or sample,
    )


def _case_stages(
    repository: Path,
    *,
    artifact_root: Path,
    sample: str,
    seed: int,
    controls: bool,
    unknown_mass: Optional[float],
    molecular_generation: str,
) -> list[PlannedStage]:
    expected_unknown_mass = (
        DEFAULT_UOT_UNKNOWN_MASS if unknown_mass is None else float(unknown_mass)
    )
    if molecular_generation not in MOLECULAR_GENERATIONS:
        raise ValueError("molecular_generation must be r1 or r2")
    source_root = _repository_path(
        repository,
        "artifacts",
        "snpatho",
        "%s_scanvi" % molecular_generation,
        sample,
    )
    root = artifact_root.expanduser().resolve() / sample
    round0, refined = _model_directories(root, seed, unknown_mass)
    train_batch = source_root / "batch_train_rare_complete.npz"
    validation_batch = source_root / "batch_validation_rare_complete.npz"
    native_prototypes = source_root / "prototypes_rare_complete.npz"
    residual_geometry = source_root / "residual_geometry_rare_complete_v2.npz"
    checkpoint = round0 / "heir.pt"
    history = round0 / "history.json"
    views = round0 / "refinement_views.npz"
    refined_checkpoint = refined / "heir_refined.pt"
    audit = refined / "refinement.json"
    refined_prototypes = refined / "prototypes" / ("%s__%s.npz" % (sample, sample))
    decoder = _repository_path(
        repository.parent,
        "HEIR_assets",
        "pretrained",
        (
            "snpatho_scanvi_r1_v1_decoder.pt"
            if molecular_generation == "r1"
            else "snpatho_scanvi_r2_preserve_biology_v1_decoder.pt"
        ),
    )
    ontology = _repository_path(repository, "configs", "ontologies", "snpatho_%s.tsv" % sample)

    train_command = _heir_source_command(
        repository,
        "train",
        "--train-batch",
        str(train_batch),
        "--validation-batch",
        str(validation_batch),
        "--output",
        str(round0),
        "--stage",
        "personalized",
        "--epochs",
        "100",
        "--learning-rate",
        "0.0001",
        "--adapter-learning-rate",
        "0.00001",
        "--weight-decay",
        "0.0001",
        "--warmup-fraction",
        "0.05",
        "--gradient-clip-norm",
        "1.0",
        "--bag-size",
        "16384",
        "--reference-batch-size",
        "2048",
        "--maximum-sample-cells",
        "16384",
        "--early-stopping-patience",
        "15",
        "--graph-hidden-dim",
        "256",
        "--graph-output-dim",
        "256",
        "--graph-layers",
        "3",
        "--trunk-hidden-dims",
        "512,256",
        "--decoder-hidden-dims",
        "128,256",
        "--dropout",
        "0.05",
        "--abstain-threshold",
        "0.35",
        "--rna-vae-checkpoint",
        str(decoder),
        "--ontology",
        str(ontology),
        "--residual-geometry",
        str(residual_geometry),
        "--allow-split-overlap",
        "--unsafe-allow-molecular-validation-overlap",
        "--mixed-precision",
        "--seed",
        str(seed),
        "--device",
        "cuda",
    )
    train_command.extend(
        [
            "--uot-unknown-mass",
            "%.2f" % expected_unknown_mass,
            "--uot-unknown-mass-mode",
            "fixed",
        ]
    )

    round_checkpoints = (
        tuple(refined / ("round_%d" % round_id) / "heir_refined.pt" for round_id in range(1, 5))
        if unknown_mass is None
        else ()
    )
    refine_command = _heir_source_command(
        repository,
        "refine",
        "--checkpoint",
        str(checkpoint),
        "--train-batch",
        str(train_batch),
        "--validation-batch",
        str(validation_batch),
        "--output",
        str(refined),
    )
    if unknown_mass is None:
        refine_command.append("--save-round-checkpoints")
    refine_command.extend(
        [
            "--maximum-rounds",
            "4",
            "--broad-refinement-rounds",
            "2",
            "--epochs-per-round",
            "30",
            "--min-probability",
            "0.90",
            "--max-normalized-entropy",
            "0.20",
            "--teacher-ema",
            "0.99",
            "--prior-old-weight",
            "1.0",
            "--minimum-segmentation-confidence",
            "0.50",
            "--maximum-validation-loss-degradation",
            "0.01",
            "--objective-relative-stability-tolerance",
            "0.01",
            "--round-selection-mode",
            "fixed",
            "--view-predictions",
            "%s::%s::%s_train=%s" % (sample, sample, sample, views),
            "--bag-size",
            "16384",
            "--reference-batch-size",
            "2048",
            "--maximum-sample-cells",
            "16384",
            "--allow-split-overlap",
            "--mixed-precision",
            "--seed",
            str(seed),
            "--device",
            "cuda",
        ]
    )
    refine_command.extend(
        [
            "--uot-unknown-mass",
            "%.2f" % expected_unknown_mass,
            "--uot-unknown-mass-mode",
            "fixed",
        ]
    )

    stages = [
        PlannedStage(
            sample,
            seed,
            "train_round0",
            tuple(train_command),
            (checkpoint, history),
            partial(
                _validate_trained_pair,
                checkpoint,
                history,
                sample=sample,
                seed=seed,
                train_batch=train_batch,
                validation_batch=validation_batch,
                decoder=decoder,
                residual_geometry=residual_geometry,
                expected_unknown_mass=expected_unknown_mass,
            ),
            unknown_mass,
            inputs=(
                ("train_batch", train_batch),
                ("validation_batch", validation_batch),
                ("rna_decoder", decoder),
                ("residual_geometry", residual_geometry),
                ("ontology", ontology),
            ),
            output_roles=("round0_checkpoint", "training_history"),
        ),
        PlannedStage(
            sample,
            seed,
            "build_views",
            _repository_script_command(
                repository,
                "scripts/build_refinement_views.py",
                "--checkpoint",
                str(checkpoint),
                "--batch",
                str(train_batch),
                "--output",
                str(views),
                "--device",
                "cuda",
            ),
            (views,),
            partial(
                _validate_refinement_views,
                views,
                checkpoint=checkpoint,
                batch=train_batch,
                sample=sample,
            ),
            unknown_mass,
            inputs=(("round0_checkpoint", checkpoint), ("train_batch", train_batch)),
            output_roles=("refinement_views",),
        ),
        PlannedStage(
            sample,
            seed,
            "refine",
            tuple(refine_command),
            (refined_checkpoint, audit, refined_prototypes, *round_checkpoints),
            partial(
                _validate_refined_pair,
                refined_checkpoint,
                audit,
                refined_prototypes,
                sample=sample,
                seed=seed,
                parent_checkpoint=checkpoint,
                view_artifact=views,
                train_batch=train_batch,
                validation_batch=validation_batch,
                round_checkpoints=round_checkpoints,
                expected_unknown_mass=expected_unknown_mass,
            ),
            unknown_mass,
            inputs=(
                ("round0_checkpoint", checkpoint),
                ("train_batch", train_batch),
                ("validation_batch", validation_batch),
                ("refinement_views", views),
            ),
            output_roles=(
                "refined_checkpoint",
                "refinement_audit",
                "refined_prototype",
                *("round_%d_checkpoint" % round_id for round_id in range(1, 5)),
            )
            if unknown_mass is None
            else ("refined_checkpoint", "refinement_audit", "refined_prototype"),
        ),
        _prediction_stage(
            repository,
            sample=sample,
            seed=seed,
            name="predict_round0",
            checkpoint=checkpoint,
            prototypes=native_prototypes,
            directory=round0,
            refinement_round=0,
            unknown_mass=unknown_mass,
        ),
        _prediction_stage(
            repository,
            sample=sample,
            seed=seed,
            name="predict_refined",
            checkpoint=refined_checkpoint,
            prototypes=refined_prototypes,
            directory=refined,
            refinement_round=None if unknown_mass is not None else 4,
            unknown_mass=unknown_mass,
        ),
    ]
    if unknown_mass is not None:
        return stages

    if seed == SEEDS[0]:
        for round_id in (1, 2, 3):
            directory = refined / ("round_%d" % round_id)
            stages.append(
                _prediction_stage(
                    repository,
                    sample=sample,
                    seed=seed,
                    name="predict_round%d" % round_id,
                    checkpoint=directory / "heir_refined.pt",
                    prototypes=native_prototypes,
                    directory=directory,
                    refinement_round=round_id,
                    unknown_mass=None,
                )
            )
    if controls and seed in ABLATION_SEEDS:
        for control in PREDICTION_CONTROLS:
            if control == "round0_prototype_only":
                control_checkpoint = checkpoint
                control_prototypes = native_prototypes
                control_directory = round0 / "control_prototype_only"
                control_round = 0
            else:
                control_checkpoint = refined_checkpoint
                control_prototypes = refined_prototypes
                control_directory = refined / (
                    "control_prototype_only"
                    if control == "refined_prototype_only"
                    else "control_" + control
                )
                control_round = 4
            stages.append(
                _prediction_stage(
                    repository,
                    sample=sample,
                    seed=seed,
                    name=control,
                    checkpoint=control_checkpoint,
                    prototypes=control_prototypes,
                    directory=control_directory,
                    refinement_round=control_round,
                    unknown_mass=None,
                    control=control,
                )
            )
    return stages


def build_plan(
    repository: Path,
    *,
    samples: Sequence[str],
    seeds: Sequence[int],
    controls: bool = False,
    unknown_mass_sensitivity: bool = False,
    artifact_root: Optional[Path] = None,
    molecular_generation: str = "r1",
) -> tuple[PlannedStage, ...]:
    """Build a repository-rooted plan without reading or running artifacts."""

    repository = repository.expanduser().resolve()
    if molecular_generation not in MOLECULAR_GENERATIONS:
        raise ValueError("molecular_generation must be r1 or r2")
    if artifact_root is None:
        artifact_root = _repository_path(
            repository,
            "artifacts",
            "snpatho",
            "%s_scanvi" % molecular_generation,
        )
    else:
        artifact_root = artifact_root.expanduser().resolve()
    if unknown_mass_sensitivity and tuple(seeds) != (SEEDS[0],):
        raise ValueError("unknown-mass sensitivity is prespecified for seed 17 only")
    if unknown_mass_sensitivity and controls:
        raise ValueError("unknown-mass sensitivity does not duplicate negative controls")
    masses: tuple[Optional[float], ...] = (
        tuple(UNKNOWN_MASS_SENSITIVITY) if unknown_mass_sensitivity else (None,)
    )
    stages: list[PlannedStage] = []
    for seed in seeds:
        for sample in samples:
            for mass in masses:
                stages.extend(
                    _case_stages(
                        repository,
                        artifact_root=artifact_root,
                        sample=sample,
                        seed=seed,
                        controls=controls,
                        unknown_mass=mass,
                        molecular_generation=molecular_generation,
                    )
                )

    if controls:
        pairings = wrong_prototype_bank_pairings(samples)
        for seed in seeds:
            if seed not in ABLATION_SEEDS:
                continue
            for target, source in pairings:
                target_root = artifact_root / target
                donor_root = artifact_root / source
                _, target_refined = _model_directories(target_root, seed, None)
                _, donor_refined = _model_directories(donor_root, seed, None)
                stages.append(
                    _prediction_stage(
                        repository,
                        sample=target,
                        seed=seed,
                        name="wrong_prototype_bank_" + source,
                        checkpoint=target_refined / "heir_refined.pt",
                        prototypes=donor_refined / "prototypes" / ("%s__%s.npz" % (source, source)),
                        directory=target_refined / ("control_wrong_donor_" + source),
                        refinement_round=4,
                        unknown_mass=None,
                        control=WRONG_PROTOTYPE_BANK_CONTROL,
                        prototype_donor_id=source,
                    )
                )
    return tuple(stages)


def _stage_id(stage: PlannedStage) -> str:
    return "%s/seed%d/%s" % (stage.sample, stage.seed, stage.name)


def _path_for_manifest(path: Path, manifest_directory: Path) -> str:
    return os.path.relpath(path.expanduser().resolve(), manifest_directory.expanduser().resolve())


def _artifact_rows(
    artifacts: Sequence[tuple[str, Path]],
    *,
    manifest_directory: Path,
) -> list[dict[str, str]]:
    rows = []
    for role, path in artifacts:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise RuntimeError("cannot manifest missing %s artifact: %s" % (role, resolved))
        rows.append(
            {
                "role": role,
                "path": _path_for_manifest(resolved, manifest_directory),
                "sha256": _sha256(resolved),
            }
        )
    return rows


def _control_transform_recipe(stage: PlannedStage) -> Optional[Mapping[str, Any]]:
    if stage.control is None:
        return None
    recipe: dict[str, Any] = {
        "schema": "heir.inference_control_transform.v1",
        "control": stage.control,
        "seed": stage.seed,
        "random_generator": "numpy.random.default_rng"
        if stage.control
        in {
            "image_shuffle",
            "graph_shuffle",
        }
        else None,
    }
    algorithms = {
        "prototype_only": ("zero residual-gate weights and set bias=-100 before inference"),
        "round0_prototype_only": (
            "round-zero checkpoint with residual-gate weights zeroed and bias=-100"
        ),
        "refined_prototype_only": (
            "refined checkpoint with residual-gate weights zeroed and bias=-100"
        ),
        "image_shuffle": (
            "apply default_rng(seed).permutation(n_nuclei) to histology feature rows"
        ),
        "graph_shuffle": (
            "apply default_rng(seed).permutation(n_nuclei) to graph edge endpoint indices"
        ),
        "no_graph": "replace edge_index and edge_weight by empty arrays",
        "wrong_prototype_bank": (
            "use the explicitly supplied non-matched PrototypeSet while retaining the shared "
            "molecular backbone"
        ),
        "wrong_donor": ("legacy CLI spelling for the wrong-prototype-bank control"),
    }
    recipe["algorithm"] = algorithms[stage.control]
    if stage.control in {"image_shuffle", "graph_shuffle"}:
        return _expected_shuffle_transform(
            stage.control,
            stage.seed,
            dict(stage.inputs)["histology"],
        )
    recipe["recipe_sha256"] = _canonical_sha256(recipe)
    return recipe


def full_matrix_plan_payload(
    stages: Sequence[PlannedStage],
    *,
    molecular_generation: str = "r1",
) -> Mapping[str, Any]:
    """Serialize the exact canonical full-matrix plan without execution claims."""

    expected_stage_count = (
        15 * 5
        + 3 * len(SAMPLES)
        + (len(PREDICTION_CONTROLS) * len(ABLATION_SEEDS) * len(SAMPLES))
        + (len(wrong_prototype_bank_pairings(SAMPLES)) * len(ABLATION_SEEDS))
    )
    if len(stages) != expected_stage_count:
        raise ValueError(
            "full refinement plan has %d stages, expected %d" % (len(stages), expected_stage_count)
        )
    rows = []
    identifiers = set()
    for index, stage in enumerate(stages):
        identifier = _stage_id(stage)
        if identifier in identifiers:
            raise ValueError("full refinement plan has duplicate stage %s" % identifier)
        identifiers.add(identifier)
        if stage.unknown_mass is not None:
            raise ValueError("full refinement plan cannot include unknown-mass sensitivity stages")
        if len(stage.outputs) != len(stage.output_roles):
            raise ValueError("stage %s output roles do not align" % identifier)
        rows.append(
            {
                "stage_index": index,
                "stage_id": identifier,
                "sample": stage.sample,
                "seed": stage.seed,
                "stage": stage.name,
                "control": stage.control,
                "prototype_donor_id": stage.prototype_donor_id,
                "command": list(stage.command),
                "command_sha256": _canonical_sha256(list(stage.command)),
                "inputs": [
                    {"role": role, "path": str(path.expanduser().resolve())}
                    for role, path in stage.inputs
                ],
                "outputs": [
                    {"role": role, "path": str(path.expanduser().resolve())}
                    for role, path in zip(stage.output_roles, stage.outputs)
                ],
                "deterministic_transform_recipe": _control_transform_recipe(stage),
            }
        )
    expected = build_plan(
        Path(__file__).resolve().parents[1],
        samples=SAMPLES,
        seeds=SEEDS,
        controls=True,
        molecular_generation=molecular_generation,
    )
    expected_ids = [_stage_id(stage) for stage in expected]
    if [row["stage_id"] for row in rows] != expected_ids:
        raise ValueError("full refinement plan is not the canonical stage ordering")
    return {
        "samples": list(SAMPLES),
        "seeds": list(SEEDS),
        "control_seeds": list(ABLATION_SEEDS),
        "trajectory_seed": SEEDS[0],
        "molecular_generation": molecular_generation,
        "controls": [*PREDICTION_CONTROLS, WRONG_PROTOTYPE_BANK_CONTROL],
        "wrong_prototype_bank_pairings": [
            {
                "target": target,
                "source": source,
                "site_matched": SAMPLE_SITES[target] == SAMPLE_SITES[source],
            }
            for target, source in wrong_prototype_bank_pairings(SAMPLES)
        ],
        "wrong_prototype_bank_pairing_count_per_control_seed": len(
            wrong_prototype_bank_pairings(SAMPLES)
        ),
        # Deprecated compatibility fields. Scientific reports use
        # ``wrong_prototype_bank_*`` because the molecular backbone is shared.
        "wrong_donor_pairings": [
            {"target": target, "source": source}
            for target, source in wrong_prototype_bank_pairings(SAMPLES)
        ],
        "wrong_donor_pairing_count_per_control_seed": len(wrong_prototype_bank_pairings(SAMPLES)),
        "stage_count": len(rows),
        "stages": rows,
    }


def _compatibility_cases(
    repository: Path,
    stages: Sequence[PlannedStage],
    *,
    manifest_directory: Path,
    molecular_generation: str,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    native_path = (
        repository / "reports" / "snpatho_scanvi_r1_manifest.json"
        if molecular_generation == "r1"
        else repository / "artifacts" / "snpatho" / "r2_scanvi" / "native_manifest.json"
    )
    native_schema = (
        "heir.snpatho_scanvi_r1_manifest.v1"
        if molecular_generation == "r1"
        else "heir.snpatho_scanvi_r2_manifest.v1"
    )
    native = _json_object(native_path, schema=native_schema)
    lookup = {(stage.sample, stage.seed, stage.name): stage for stage in stages}
    cases = []
    for seed in SEEDS:
        for sample in SAMPLES:
            stage = lookup[(sample, seed, "predict_refined")]
            prediction = stage.outputs[0]
            cases.append(
                {
                    "section_id": sample,
                    "seed": seed,
                    "predictions": _path_for_manifest(prediction, manifest_directory),
                    "predictions_sha256": _sha256(prediction),
                }
            )
    compatibility = {
        "schema": LEGACY_FIVE_SEED_MANIFEST_SCHEMA,
        "analysis_role": (
            "prespecified_five_seed_native_scanvi_integrated_annotation_sensitivity"
            if molecular_generation == "r1"
            else "specimen_preserving_scanvi_integrated_annotation_sensitivity"
        ),
        "molecular_generation": molecular_generation,
        "negative_control": False,
        "native_scanvi_manifest_sha256": _sha256(native_path),
        "latent_space_id": native.get("latent_space_id"),
        "expression_space_id": native.get("expression_space_id"),
        "seeds": list(SEEDS),
        "samples": list(SAMPLES),
        "controls_available": sorted([*PREDICTION_CONTROLS, WRONG_PROTOTYPE_BANK_CONTROL]),
        "wrong_prototype_bank_pairings": [
            {"target": target, "source": source}
            for target, source in wrong_prototype_bank_pairings(SAMPLES)
        ],
        "wrong_prototype_bank_coverage_complete": True,
        # Deprecated aliases retained for report readers built against v1.
        "wrong_donor_pairings": [
            {"target": target, "source": source}
            for target, source in wrong_prototype_bank_pairings(SAMPLES)
        ],
        "wrong_donor_coverage_complete": True,
        "cases": cases,
    }
    return compatibility, cases


def build_refinement_run_manifest(
    repository: Path,
    stages: Sequence[PlannedStage],
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    molecular_generation: str = "r1",
) -> Mapping[str, Any]:
    """Adopt or record the exact full matrix while preserving execution semantics."""

    repository = repository.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    manifest_directory = manifest_path.parent
    plan = full_matrix_plan_payload(
        stages,
        molecular_generation=molecular_generation,
    )
    if len(records) != len(stages):
        raise ValueError("execution records do not align to the full refinement plan")
    manifested_stages = []
    transform_hashes_verified = []
    status_counts = {"completed": 0, "skipped_valid": 0}
    for planned, stage, record in zip(plan["stages"], stages, records):
        status = str(record.get("status", ""))
        if status not in status_counts:
            raise ValueError("cannot manifest unvalidated stage status %r" % status)
        for field, expected in (
            ("sample", stage.sample),
            ("seed", stage.seed),
            ("stage", stage.name),
        ):
            if record.get(field) != expected:
                raise ValueError("execution record %s differs from its stage" % field)
        status_counts[status] += 1
        inputs = _artifact_rows(stage.inputs, manifest_directory=manifest_directory)
        outputs = _artifact_rows(
            tuple(zip(stage.output_roles, stage.outputs)),
            manifest_directory=manifest_directory,
        )
        recipe = planned["deterministic_transform_recipe"]
        transform_verified = True
        telemetry_transform = None
        if recipe is not None and stage.control in {"image_shuffle", "graph_shuffle"}:
            telemetry = _json_object(stage.outputs[1], schema="heir.inference_telemetry.v1")
            negative = telemetry.get("negative_control")
            telemetry_transform = (
                negative.get("transform") if isinstance(negative, Mapping) else None
            )
            transform_verified = bool(
                isinstance(telemetry_transform, Mapping)
                and all(telemetry_transform.get(key) == value for key, value in recipe.items())
            )
            if not transform_verified:
                raise ValueError(
                    "cannot manifest shuffle stage with missing or stale transform hashes: %s"
                    % planned["stage_id"]
                )
        if recipe is not None:
            transform_hashes_verified.append(transform_verified)
        manifested_stages.append(
            {
                **{
                    key: value for key, value in planned.items() if key not in {"inputs", "outputs"}
                },
                "inputs": inputs,
                "outputs": outputs,
                "runner_status": status,
                "status": (
                    "completed_current_invocation"
                    if status == "completed"
                    else "adopted_existing_output_after_current_validation"
                ),
                "current_recipe_validation": "passed",
                "original_execution_source_verified": status == "completed",
                "execution_transform_hash_verified": transform_verified,
                "telemetry_transform": telemetry_transform,
            }
        )
    all_completed = status_counts["completed"] == len(stages)
    all_adopted = status_counts["skipped_valid"] == len(stages)
    transform_verified = all(transform_hashes_verified) if transform_hashes_verified else True
    execution_verified = all_completed and transform_verified
    if all_completed:
        execution_mode = "all_stages_completed_current_invocation"
        manifest_role = "current_invocation_execution_and_validation"
    elif all_adopted:
        execution_mode = "all_existing_outputs_posthoc_adopted_after_current_validation"
        manifest_role = "posthoc_output_adoption_not_original_execution_proof"
    else:
        execution_mode = "mixed_current_execution_and_posthoc_adoption"
        manifest_role = "mixed_execution_and_adoption_not_complete_original_execution_proof"
    compatibility, cases = _compatibility_cases(
        repository,
        stages,
        manifest_directory=manifest_directory,
        molecular_generation=molecular_generation,
    )
    return {
        "schema": REFINEMENT_RUN_MANIFEST_SCHEMA,
        "manifest_role": manifest_role,
        "analysis_role": compatibility["analysis_role"],
        "negative_control": False,
        "molecular_generation": molecular_generation,
        "native_scanvi_manifest_sha256": compatibility["native_scanvi_manifest_sha256"],
        "latent_space_id": compatibility["latent_space_id"],
        "expression_space_id": compatibility["expression_space_id"],
        "seeds": compatibility["seeds"],
        "samples": compatibility["samples"],
        "controls_available": compatibility["controls_available"],
        "wrong_prototype_bank_pairings": compatibility.get(
            "wrong_prototype_bank_pairings",
            compatibility.get("wrong_donor_pairings", []),
        ),
        "wrong_prototype_bank_coverage_complete": compatibility.get(
            "wrong_prototype_bank_coverage_complete",
            compatibility.get("wrong_donor_coverage_complete", False),
        ),
        "wrong_donor_pairings": compatibility["wrong_donor_pairings"],
        "wrong_donor_coverage_complete": compatibility["wrong_donor_coverage_complete"],
        "cases": cases,
        "legacy_five_seed_compatibility": compatibility,
        "plan_sha256": _canonical_sha256(plan),
        "plan": {key: value for key, value in plan.items() if key != "stages"},
        "cli_source_binding": _heir_source_binding(repository, require_sources=True),
        "validation_recipe_source_identity": refinement_run_source_identity(repository),
        "execution": {
            "execute_requested": True,
            "execution_mode": execution_mode,
            "stage_status_counts": status_counts,
            "current_recipe_validation_complete": True,
            "posthoc_adoption_present": status_counts["skipped_valid"] > 0,
            "original_execution_source_verified": all_completed,
            "execution_transform_hash_verified": transform_verified,
            "execution_provenance_verified": execution_verified,
            "limitation": (
                None
                if execution_verified
                else (
                    "Current validation proves that adopted outputs satisfy the current recipe, "
                    "but does not prove which source revision originally executed every stage."
                )
            ),
        },
        "stage_count": len(manifested_stages),
        "stages": manifested_stages,
    }


def unknown_mass_plan_payload(
    stages: Sequence[PlannedStage],
    *,
    samples: Sequence[str],
    molecular_generation: str = "r1",
) -> Mapping[str, Any]:
    """Serialize the canonical sensitivity plan independently of execution status."""

    samples = tuple(str(sample) for sample in samples)
    if molecular_generation not in MOLECULAR_GENERATIONS:
        raise ValueError("molecular_generation must be r1 or r2")
    expected = len(samples) * len(UNKNOWN_MASS_SENSITIVITY) * len(UNKNOWN_MASS_STAGE_NAMES)
    if len(stages) != expected:
        raise ValueError("unknown-mass plan has %d stages, expected %d" % (len(stages), expected))
    rows = []
    position = 0
    for sample in samples:
        for mass in UNKNOWN_MASS_SENSITIVITY:
            for stage_name in UNKNOWN_MASS_STAGE_NAMES:
                stage = stages[position]
                position += 1
                if (
                    stage.sample != sample
                    or stage.seed != SEEDS[0]
                    or stage.name != stage_name
                    or stage.unknown_mass != mass
                ):
                    raise ValueError("unknown-mass plan is not the exact canonical grid")
                rows.append(
                    {
                        "stage_index": position - 1,
                        "sample": stage.sample,
                        "seed": stage.seed,
                        "unknown_mass": stage.unknown_mass,
                        "stage": stage.name,
                        "command": list(stage.command),
                        "inputs": [
                            {"role": role, "path": str(path.resolve())}
                            for role, path in stage.inputs
                        ],
                        "outputs": [str(path.resolve()) for path in stage.outputs],
                    }
                )
    return {
        "samples": list(samples),
        "seed": SEEDS[0],
        "molecular_generation": molecular_generation,
        "unknown_masses": list(UNKNOWN_MASS_SENSITIVITY),
        "stage_names": list(UNKNOWN_MASS_STAGE_NAMES),
        "stages": rows,
    }


def build_unknown_mass_manifest(
    repository: Path,
    stages: Sequence[PlannedStage],
    records: Sequence[Mapping[str, Any]],
    *,
    samples: Sequence[str],
    molecular_generation: str = "r1",
) -> Mapping[str, Any]:
    """Build a hash-bound manifest after every canonical output validates."""

    repository = repository.expanduser().resolve()
    plan = unknown_mass_plan_payload(
        stages,
        samples=samples,
        molecular_generation=molecular_generation,
    )
    if len(records) != len(stages):
        raise ValueError("execution records do not align to the unknown-mass plan")
    manifested_stages = []
    for planned, stage, record in zip(plan["stages"], stages, records):
        status = str(record.get("status", ""))
        if status not in {"completed", "skipped_valid"}:
            raise ValueError("cannot manifest an unvalidated stage status %r" % status)
        for field in ("sample", "seed", "stage", "unknown_mass"):
            if record.get(field) != planned[field]:
                raise ValueError("execution record %s does not match its planned stage" % field)
        output_rows = []
        input_rows = []
        for role, path in stage.inputs:
            if not path.is_file():
                raise RuntimeError("cannot manifest a missing stage input: %s" % path)
            input_rows.append(
                {
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
        for path in stage.outputs:
            if not path.is_file():
                raise RuntimeError("cannot manifest a missing stage output: %s" % path)
            output_rows.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        manifested_stages.append(
            {
                **planned,
                "inputs": input_rows,
                "outputs": output_rows,
                "status": status,
            }
        )
    statuses = {row["status"] for row in manifested_stages}
    if statuses == {"skipped_valid"}:
        execution_mode = "all_skipped_valid"
    elif statuses == {"completed"}:
        execution_mode = "all_completed"
    else:
        execution_mode = "mixed_completed_and_skipped_valid"
    return {
        "schema": UNKNOWN_MASS_MANIFEST_SCHEMA,
        "samples": list(plan["samples"]),
        "seed": plan["seed"],
        "molecular_generation": plan["molecular_generation"],
        "unknown_masses": list(plan["unknown_masses"]),
        "stage_names": list(plan["stage_names"]),
        "stage_count": len(manifested_stages),
        "plan_sha256": _canonical_sha256(plan),
        "execution_mode": execution_mode,
        "manifest_role": "post_execute_output_adoption_and_validation",
        "lineage_scope": (
            "current source/environment, exact commands, and SHA-256 identities of every "
            "planned stage input and output"
        ),
        "cli_source_binding": _heir_source_binding(repository, require_sources=True),
        "validation_recipe_source_identity": unknown_mass_source_identity(repository),
        "stages": manifested_stages,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", choices=("all", *SAMPLES))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument(
        "--molecular-generation",
        choices=MOLECULAR_GENERATIONS,
        default="r2",
        help="R2 preserves specimen biology; use r1 only for historical reproduction",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Output root for model and prediction stages; immutable inputs remain canonical",
    )
    parser.add_argument(
        "--prohibit-adoption",
        action="store_true",
        help=(
            "Fail before execution if any planned output already exists; use this with a fresh "
            "artifact root for execution-provenance-clean runs"
        ),
    )
    parser.add_argument(
        "--unknown-mass-sensitivity",
        action="store_true",
        help="plan or execute the seed-17 fixed unknown-mass grid 0,.01,.05,.10,.20",
    )
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args(argv)
    requested_samples = (
        SAMPLES if not args.sample or "all" in args.sample else tuple(dict.fromkeys(args.sample))
    )
    requested_seeds = SEEDS if not args.seed else tuple(dict.fromkeys(args.seed))
    if any(seed not in SEEDS for seed in requested_seeds):
        raise ValueError("seed must be one of the five prespecified primary seeds")
    if args.unknown_mass_sensitivity:
        if args.seed and requested_seeds != (SEEDS[0],):
            raise ValueError("unknown-mass sensitivity is prespecified for seed 17 only")
        if args.controls:
            raise ValueError("unknown-mass sensitivity does not duplicate negative controls")
        requested_seeds = (SEEDS[0],)
    if args.manifest_output is not None and not args.execute:
        raise ValueError("--manifest-output requires --execute so every output is validated")
    if args.prohibit_adoption and not args.execute:
        raise ValueError("--prohibit-adoption requires --execute")
    if args.manifest_output is not None and not args.unknown_mass_sensitivity:
        if (
            tuple(requested_samples) != SAMPLES
            or tuple(requested_seeds) != SEEDS
            or not args.controls
        ):
            raise ValueError(
                "the full refinement run manifest requires all three samples, all five seeds, "
                "and --controls"
            )

    stages = build_plan(
        repository,
        samples=requested_samples,
        seeds=requested_seeds,
        controls=args.controls,
        unknown_mass_sensitivity=args.unknown_mass_sensitivity,
        artifact_root=(
            None
            if args.artifact_root is None
            else _output_path(repository, args.artifact_root)
        ),
        molecular_generation=args.molecular_generation,
    )
    if args.prohibit_adoption:
        existing = sorted(
            {path for stage in stages for path in stage.outputs if path.exists()},
            key=str,
        )
        if existing:
            preview = ", ".join(str(path) for path in existing[:5])
            suffix = "" if len(existing) <= 5 else " (and %d more)" % (len(existing) - 5)
            raise RuntimeError(
                "--prohibit-adoption found existing planned outputs: %s%s. Choose a clean "
                "--artifact-root; existing endpoints are never deleted automatically."
                % (preview, suffix)
            )
    records = []
    for stage in stages:
        status = _run(
            stage.command,
            stage.outputs,
            args.execute,
            validator=stage.validate,
            repository=repository,
        )
        record: dict[str, Any] = {
            "sample": stage.sample,
            "seed": stage.seed,
            "stage": stage.name,
            "status": status,
        }
        if stage.unknown_mass is not None:
            record["unknown_mass"] = stage.unknown_mass
        records.append(record)

    if args.manifest_output is not None and args.execute and args.unknown_mass_sensitivity:
        destination = _output_path(repository, args.manifest_output)
        _write_json(
            destination,
            build_unknown_mass_manifest(
                repository,
                stages,
                records,
                samples=requested_samples,
                molecular_generation=args.molecular_generation,
            ),
        )
    elif args.manifest_output is not None and args.execute:
        destination = _output_path(repository, args.manifest_output)
        _write_json(
            destination,
            build_refinement_run_manifest(
                repository,
                stages,
                records,
                manifest_path=destination,
                molecular_generation=args.molecular_generation,
            ),
        )
    print(json.dumps({"execute": args.execute, "records": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
