#!/usr/bin/env python3
"""Build or independently validate the receipt-bound HEST blank-image supplement."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.evaluation.blank_patch_receipt import (
    batch_parity_receipt,
    blank_pixels,
    load_supplement,
    make_receipt,
    source_identity,
    supplement_arrays,
    validate_encoder_contract,
    validate_receipt,
)
from heir.evaluation.nonlinear_qualification_contract import file_sha256, validate_protocol
from heir.features import load_encoder_manifest
from heir.features.base import EncoderManifest, sha256_file
from heir.features.hoptimus1 import HOptimus1Encoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz")
DEFAULT_PROTOCOL = ROOT / "configs/hest_nonlinear_qualification_v1.json"
DEFAULT_MANIFEST = ROOT / "manifests/encoders/hoptimus1.json"
DEFAULT_MODEL_DIR = Path("/mnt/seagate/HnE/pretrained/H-optimus-1")
DEFAULT_SUPPLEMENT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/blank_patch.npz")
DEFAULT_RECEIPT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/blank_patch_receipt.json")
IMPLEMENTATION_FILES = {
    "builder_sha256": Path(__file__).resolve(),
    "receipt_validator_sha256": ROOT / "src/heir/evaluation/blank_patch_receipt.py",
    "encoder_adapter_sha256": ROOT / "src/heir/features/hoptimus1.py",
    "encoder_base_sha256": ROOT / "src/heir/features/base.py",
}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _protocol_contract(path: Path) -> Mapping[str, object]:
    resolved = path.expanduser().resolve()
    protocol = validate_protocol(_json(resolved))
    source = protocol["registered_source"]
    dataset = protocol["dataset"]
    if not isinstance(source, Mapping) or not isinstance(dataset, Mapping):
        raise ValueError("frozen v1 protocol source contract is malformed")
    return {
        "sha256": str(source["sha256"]),
        "rows": int(dataset["registered_cells"]),
        "protocol_sha256": file_sha256(resolved),
    }


def _git_identity() -> Mapping[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "cannot inspect the Git identity for blank-control construction"
        ) from error
    if status:
        raise ValueError("blank-control construction requires a committed, clean Git worktree")
    return {"git_commit": commit, "git_worktree_dirty_at_start": False}


def _configure_torch(device: str) -> object:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must request deterministic CUDA behavior")
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install HEIR's torch dependency") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return torch


def _runtime_identity(
    device: str, torch: object, git: Mapping[str, object]
) -> Mapping[str, object]:
    cuda_name = None
    if device.startswith("cuda"):
        cuda_name = torch.cuda.get_device_name(torch.device(device))
    return {
        **git,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "timm_version": importlib.metadata.version("timm"),
        "safetensors_version": importlib.metadata.version("safetensors"),
        "device": device,
        "cuda_device_name": cuda_name,
        "cuda_autocast_enabled_by_encoder": device.startswith("cuda"),
        "tf32_enabled": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _implementation_identity() -> Mapping[str, object]:
    return {name: sha256_file(path) for name, path in IMPLEMENTATION_FILES.items()}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=resolved.parent, prefix=f".{resolved.name}.")
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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=resolved.parent, prefix=f".{resolved.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _encode_blank_batches(encoder: HOptimus1Encoder) -> tuple[np.ndarray, np.ndarray]:
    single = np.asarray(encoder.encode(blank_pixels(1)), dtype=np.float32)
    repeated = np.asarray(encoder.encode(blank_pixels(32)), dtype=np.float32)
    return single, repeated


def _load_inputs(
    *,
    source_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    model_dir: Path,
) -> tuple[Mapping[str, object], Mapping[str, object], EncoderManifest]:
    contract = _protocol_contract(protocol_path)
    manifest = load_encoder_manifest(manifest_path)
    validate_encoder_contract(manifest, model_dir)
    source = source_identity(
        source_path,
        expected_sha256=str(contract["sha256"]),
        expected_rows=int(contract["rows"]),
    )
    return contract, source, manifest


def build(args: argparse.Namespace) -> Mapping[str, object]:
    git = _git_identity()
    torch = _configure_torch(args.device)
    contract, source, manifest = _load_inputs(
        source_path=args.source,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        model_dir=args.model_dir,
    )
    encoder = HOptimus1Encoder(args.model_dir, manifest, device=args.device)
    single, repeated = _encode_blank_batches(encoder)
    parity = batch_parity_receipt(single, repeated)
    arrays = supplement_arrays(single, source=source, manifest=manifest)
    _atomic_npz(args.output, arrays)
    supplement = load_supplement(args.output)
    receipt = make_receipt(
        source=source,
        supplement=supplement,
        manifest=manifest,
        protocol_sha256=str(contract["protocol_sha256"]),
        parity=parity,
        implementation=_implementation_identity(),
        runtime=_runtime_identity(args.device, torch, git),
    )
    validate_receipt(
        receipt,
        source=source,
        supplement=supplement,
        manifest=manifest,
        protocol_sha256=str(contract["protocol_sha256"]),
        expected_implementation=_implementation_identity(),
    )
    _atomic_json(args.receipt, receipt)
    return receipt


def validate(args: argparse.Namespace) -> Mapping[str, object]:
    torch = _configure_torch(args.device)
    contract, source, manifest = _load_inputs(
        source_path=args.source,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        model_dir=args.model_dir,
    )
    supplement = load_supplement(args.output)
    receipt = _json(args.receipt)
    encoder = HOptimus1Encoder(args.model_dir, manifest, device=args.device)
    single, repeated = _encode_blank_batches(encoder)
    result = validate_receipt(
        receipt,
        source=source,
        supplement=supplement,
        manifest=manifest,
        protocol_sha256=str(contract["protocol_sha256"]),
        expected_implementation=_implementation_identity(),
        recomputed_single=single,
        recomputed_repeated=repeated,
    )
    return {
        **result,
        "validator_runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "device": args.device,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--encoder-manifest", dest="manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="recompute the pixels and validate an existing supplement and receipt",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("source", "protocol", "manifest", "model_dir", "output", "receipt"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    result = validate(args) if args.validate_only else build(args)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "source_sha256": result.get(
                    "registered_source_sha256", result.get("source_sha256")
                ),
                "supplement_sha256": result["supplement_sha256"],
                "feature_sha256": result["feature_sha256"],
                "molecular_outcomes_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
