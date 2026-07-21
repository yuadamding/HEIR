#!/usr/bin/env python3
"""Build or independently validate the receipt-bound HEST blank control."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.data import current_git_commit, require_clean_worktree
from heir.evaluation import blank_patch_receipt as contract
from heir.evaluation.nonlinear_qualification_contract import file_sha256, validate_protocol
from heir.features import load_encoder_manifest
from heir.features.hoptimus1 import HOptimus1Encoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz")
DEFAULT_PROTOCOL = ROOT / "configs/hest_nonlinear_qualification_v1.json"
DEFAULT_ENCODER_MANIFEST = ROOT / "manifests/encoders/hoptimus1.json"
DEFAULT_MODEL_DIR = Path("/mnt/seagate/HnE/pretrained/H-optimus-1")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/blank_patch.npz")
DEFAULT_RECEIPT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/blank_patch_receipt.json")


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _protocol_contract(path: Path) -> Mapping[str, object]:
    resolved = path.expanduser().resolve()
    protocol = validate_protocol(_json(resolved))
    registered = protocol["registered_source"]
    dataset = protocol["dataset"]
    if (
        protocol.get("protocol_id") != "hest_nonlinear_qualification_v1"
        or not isinstance(registered, Mapping)
        or not isinstance(dataset, Mapping)
        or registered.get("sha256") is None
        or dataset.get("registered_cells") != 36121
    ):
        raise ValueError("blank construction requires the permanently blocked HEST v1 identity")
    return {
        "sha256": file_sha256(resolved),
        "registered_source_sha256": str(registered["sha256"]),
        "source_rows": int(dataset["registered_cells"]),
    }


def _implementation_identity() -> Mapping[str, object]:
    paths = {
        "builder_sha256": Path(__file__).resolve(),
        "receipt_validator_sha256": Path(contract.__file__).resolve(),
        "encoder_adapter_sha256": ROOT / "src/heir/features/hoptimus1.py",
        "encoder_base_sha256": ROOT / "src/heir/features/base.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _cuda_device_index(device: str, current_device: int) -> int:
    if device == "cuda":
        return int(current_device)
    prefix, separator, index = device.partition(":")
    if prefix != "cuda" or separator != ":" or not index.isdigit():
        raise ValueError("blank encoding device must be cuda or an indexed CUDA device")
    return int(index)


def _configure_torch(device: str) -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must request deterministic CUDA behavior")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - required production dependency
        raise RuntimeError("install HEIR's torch dependency") from error
    if not device.startswith("cuda"):
        raise ValueError("the registered HEST blank must use the CUDA production encoder path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered HEST blank encoding")
    torch.cuda.set_device(_cuda_device_index(device, torch.cuda.current_device()))
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return torch


def _runtime_identity(torch: object, device: str, git_commit: str) -> Mapping[str, object]:
    return {
        "git_commit": git_commit,
        "git_worktree_dirty_at_start": False,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "timm_version": importlib.metadata.version("timm"),
        "safetensors_version": importlib.metadata.version("safetensors"),
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_version": torch.version.cuda,
        "tf32_enabled": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _encode_blank(encoder: object) -> tuple[np.ndarray, np.ndarray]:
    """Encode only internally constructed white pixels; no feature input is accepted."""

    one = np.asarray(encoder.encode(contract.blank_rgb(1)))
    many = np.asarray(encoder.encode(contract.blank_rgb(32)))
    contract.validate_feature_matrix(one, 1)
    contract.validate_feature_matrix(many, 32)
    return one, many


def _release_encoder(encoder: object) -> None:
    model = getattr(encoder, "_model", None)
    if model is not None and hasattr(model, "to"):
        model.to("cpu")
    del encoder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=resolved.parent, prefix=f".{resolved.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=resolved.parent, prefix=f".{resolved.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build(args: argparse.Namespace) -> Mapping[str, object]:
    """Build the supplement from pixels after every outcome-blind identity check passes."""

    require_clean_worktree(ROOT)
    git_commit = current_git_commit(ROOT)
    frozen = _protocol_contract(args.protocol)
    manifest = load_encoder_manifest(args.encoder_manifest)
    contract.validate_encoder_contract(manifest, args.model_dir)
    source = contract.inspect_registered_source(
        args.source,
        expected_sha256=str(frozen["registered_source_sha256"]),
        expected_rows=int(frozen["source_rows"]),
    )
    torch = _configure_torch(args.device)
    encoder = HOptimus1Encoder(args.model_dir, manifest, device=args.device)
    try:
        batch_one, batch_32 = _encode_blank(encoder)
    finally:
        _release_encoder(encoder)
    parity = contract.batch_parity_receipt(batch_one, batch_32)

    # Recheck immediately before publishing either artifact; source hashing and
    # giant-model inference can be long-running operations.
    require_clean_worktree(ROOT)
    if current_git_commit(ROOT) != git_commit:
        raise ValueError("Git commit changed during blank supplement construction")
    _atomic_npz(args.output, contract.supplement_payload(source, manifest, batch_one))
    supplement = contract.load_supplement(args.output)
    implementation = _implementation_identity()
    receipt = contract.build_receipt(
        source=source,
        supplement=supplement,
        manifest=manifest,
        protocol_sha256=str(frozen["sha256"]),
        parity=parity,
        implementation=implementation,
        runtime=_runtime_identity(torch, args.device, git_commit),
    )
    contract.validate_receipt(
        receipt,
        source=source,
        supplement=supplement,
        manifest=manifest,
        protocol_sha256=str(frozen["sha256"]),
        expected_implementation=implementation,
        recomputed_batch_one=batch_one,
        recomputed_batch_32=batch_32,
    )
    _atomic_json(args.receipt, receipt)
    return receipt


def validate(args: argparse.Namespace) -> Mapping[str, object]:
    """Independently recompute from pixels and validate an existing artifact pair."""

    frozen = _protocol_contract(args.protocol)
    manifest = load_encoder_manifest(args.encoder_manifest)
    contract.validate_encoder_contract(manifest, args.model_dir)
    source = contract.inspect_registered_source(
        args.source,
        expected_sha256=str(frozen["registered_source_sha256"]),
        expected_rows=int(frozen["source_rows"]),
    )
    supplement = contract.load_supplement(args.output)
    receipt = _json(args.receipt)
    torch = _configure_torch(args.device)
    encoder = HOptimus1Encoder(args.model_dir, manifest, device=args.device)
    try:
        batch_one, batch_32 = _encode_blank(encoder)
    finally:
        _release_encoder(encoder)
    contract.validate_receipt(
        receipt,
        source=source,
        supplement=supplement,
        manifest=manifest,
        protocol_sha256=str(frozen["sha256"]),
        expected_implementation=_implementation_identity(),
        recomputed_batch_one=batch_one,
        recomputed_batch_32=batch_32,
    )
    return {
        "schema": "heir.hest_blank_patch_validation.v1",
        "passed": True,
        "registered_source_sha256": source.sha256,
        "supplement_sha256": supplement.sha256,
        "receipt_sha256": file_sha256(args.receipt),
        "recomputed_feature_sha256": contract.array_sha256(batch_one),
        "molecular_outcomes_read": False,
        "predictions_read": False,
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--encoder-manifest", type=Path, default=DEFAULT_ENCODER_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    args.source = args.source.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    args.encoder_manifest = args.encoder_manifest.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.receipt = args.receipt.expanduser().resolve()
    result = validate(args) if args.validate_only else build(args)
    summary = {
        "schema": result["schema"],
        "status": result.get("status", "passed"),
        "passed": result.get("passed", result.get("status") == "passed"),
        "supplement": str(args.output),
        "receipt": str(args.receipt),
        "feature_sha256": result.get("feature_sha256", result.get("recomputed_feature_sha256")),
        "molecular_outcomes_read": False,
        "predictions_read": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
