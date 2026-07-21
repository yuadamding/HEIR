"""Canonical hashes and auditable persistence for neural residual probes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import torch

PathLike = Union[str, Path]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_model_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU bytes in key order."""

    if not state_dict:
        raise ValueError("model state must not be empty")
    digest = hashlib.sha256(b"heir.neural_model_state.v1\0")
    for name in sorted(state_dict):
        if not isinstance(name, str) or not name:
            raise ValueError("model-state names must be non-empty strings")
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("model state must contain tensors only")
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("model state contains non-finite tensors")
        array = tensor.numpy()
        header = {
            "name": name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        encoded = _canonical_json(header)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def canonical_metadata_sha256(metadata: Mapping[str, object]) -> str:
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    return hashlib.sha256(_canonical_json(metadata)).hexdigest()


def canonical_array_registry_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"heir.neural_array_registry.v1\0")
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise ValueError("array-registry names must be non-empty strings")
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        if array.dtype.hasobject:
            raise ValueError("array registry must not contain object arrays")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError("array registry contains non-finite values")
        header = _canonical_json(
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}
        )
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def save_neural_probe_bundle(
    path: PathLike,
    state_dict: Mapping[str, torch.Tensor],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    """Atomically persist model state plus every fitted preprocessing array."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    state_arrays = {
        "state::" + name: tensor.detach().cpu().contiguous().numpy()
        for name, tensor in sorted(state_dict.items())
    }
    fitted_arrays = {
        "array::" + name: np.ascontiguousarray(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    if set(state_arrays) & set(fitted_arrays):
        raise ValueError("checkpoint state and fitted arrays collide")
    state_sha = canonical_model_state_sha256(state_dict)
    array_sha = canonical_array_registry_sha256(arrays)
    metadata_sha = canonical_metadata_sha256(metadata)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name + ".", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **state_arrays, **fitted_arrays)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    receipt = {
        "schema": "heir.neural_probe_checkpoint_receipt.v1",
        "path": str(target),
        "state_sha256": state_sha,
        "array_registry_sha256": array_sha,
        "metadata_sha256": metadata_sha,
        "tensor_names": sorted(state_dict),
        "array_names": sorted(arrays),
        "metadata": dict(metadata),
    }
    receipt_path = target.with_suffix(target.suffix + ".json")
    descriptor, receipt_temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=receipt_path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(receipt_temporary_name, receipt_path)
    except BaseException:
        Path(receipt_temporary_name).unlink(missing_ok=True)
        raise
    return receipt


def load_neural_probe_bundle(
    path: PathLike,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, np.ndarray], Mapping[str, object]]:
    target = Path(path).expanduser().resolve()
    receipt_path = target.with_suffix(target.suffix + ".json")
    if not target.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("neural probe checkpoint or receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "heir.neural_probe_checkpoint_receipt.v1":
        raise ValueError("neural probe checkpoint receipt schema is unsupported")
    metadata = receipt.get("metadata")
    if not isinstance(metadata, Mapping) or canonical_metadata_sha256(metadata) != receipt.get(
        "metadata_sha256"
    ):
        raise ValueError("neural probe metadata hash differs from its receipt")
    with np.load(target, allow_pickle=False) as archive:
        state = {
            name.removeprefix("state::"): torch.as_tensor(np.asarray(archive[name]).copy())
            for name in sorted(archive.files)
            if name.startswith("state::")
        }
        arrays = {
            name.removeprefix("array::"): np.asarray(archive[name]).copy()
            for name in sorted(archive.files)
            if name.startswith("array::")
        }
    if sorted(state) != list(receipt.get("tensor_names", [])) or sorted(arrays) != list(
        receipt.get("array_names", [])
    ):
        raise ValueError("neural probe checkpoint registry differs from its receipt")
    if canonical_model_state_sha256(state) != receipt.get("state_sha256"):
        raise ValueError("neural probe model-state hash differs from its receipt")
    if canonical_array_registry_sha256(arrays) != receipt.get("array_registry_sha256"):
        raise ValueError("neural probe fitted-array hash differs from its receipt")
    return state, arrays, metadata


def save_neural_checkpoint(
    path: PathLike,
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    """Atomically store a safe NumPy checkpoint plus a hash-bound JSON receipt.

    The file is an NPZ container even when callers choose a custom suffix.
    Object arrays and pickle are never used.  Scientific identity is defined by
    ``canonical_model_state_sha256``, not by container bytes or ZIP timestamps.
    """

    target = Path(path).expanduser().resolve()
    bundle_receipt = save_neural_probe_bundle(target, state_dict, {}, metadata)
    receipt = {
        **bundle_receipt,
        "schema": "heir.neural_checkpoint_receipt.v1",
    }
    receipt_path = target.with_suffix(target.suffix + ".json")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=receipt_path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, receipt_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return receipt


def load_neural_checkpoint(path: PathLike) -> Mapping[str, torch.Tensor]:
    target = Path(path).expanduser().resolve()
    receipt_path = target.with_suffix(target.suffix + ".json")
    if not target.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("checkpoint or receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "heir.neural_checkpoint_receipt.v1":
        raise ValueError("neural checkpoint receipt schema is unsupported")
    metadata = receipt.get("metadata")
    if not isinstance(metadata, Mapping) or canonical_metadata_sha256(metadata) != receipt.get(
        "metadata_sha256"
    ):
        raise ValueError("checkpoint metadata hash differs from its receipt")
    with np.load(target, allow_pickle=False) as archive:
        state = {
            name.removeprefix("state::"): torch.as_tensor(np.asarray(archive[name]).copy())
            for name in sorted(archive.files)
            if name.startswith("state::")
        }
        unexpected = [name for name in archive.files if not name.startswith("state::")]
    if unexpected or sorted(state) != list(receipt.get("tensor_names", [])):
        raise ValueError("model-only checkpoint tensor registry differs from its receipt")
    if canonical_model_state_sha256(state) != receipt.get("state_sha256"):
        raise ValueError("checkpoint state hash differs from its receipt")
    if canonical_array_registry_sha256({}) != receipt.get("array_registry_sha256"):
        raise ValueError("model-only checkpoint carries a non-empty fitted-array identity")
    return state


__all__ = [
    "canonical_metadata_sha256",
    "canonical_array_registry_sha256",
    "canonical_model_state_sha256",
    "load_neural_checkpoint",
    "load_neural_probe_bundle",
    "save_neural_checkpoint",
    "save_neural_probe_bundle",
]
