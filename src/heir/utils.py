"""Deterministic device, hashing, and atomic-output helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import torch

PathLike = Union[str, os.PathLike]


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.strip().lower()
    device = torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value)
    if value == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def sha256_file(path: PathLike, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_output_input_collisions(
    output_paths: Sequence[PathLike], input_paths: Sequence[PathLike], *, label: str
) -> None:
    outputs = [Path(value).expanduser().resolve() for value in output_paths]
    inputs = [Path(value).expanduser().resolve() for value in input_paths]
    if len(set(outputs)) != len(outputs):
        raise ValueError("%s output paths collide" % label)
    for output in outputs:
        for source in inputs:
            aliases = output == source
            if output.exists() and source.exists():
                aliases = aliases or output.samefile(source)
            if source.is_dir() and source in output.parents:
                aliases = True
            if aliases:
                raise ValueError("%s output would overwrite a bound input" % label)


def atomic_json_dump(payload: Dict[str, Any], path: PathLike) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = ["atomic_json_dump", "reject_output_input_collisions", "resolve_device", "sha256_file"]
