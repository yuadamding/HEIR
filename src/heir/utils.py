"""Small deterministic and provenance utilities used throughout HEIR."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

PathLike = Union[str, os.PathLike]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch without silently changing the seed."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if deterministic:
        # This must be set before the first cuBLAS workspace is created;
        # otherwise seeded CUDA GEMMs are not reproducible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda`` or an explicit CUDA device."""

    value = requested.strip().lower()
    if value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        # Ampere and newer GPUs execute eligible float32 matrix operations via
        # TF32 tensor cores. This is deterministic on a fixed software/device
        # stack and materially accelerates HEIR's dense trunks and decoders.
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    return device


def sha256_file(path: PathLike, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream a SHA-256 checksum without loading a cohort artifact into RAM."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def reject_output_input_collisions(
    output_paths: Sequence[PathLike],
    input_paths: Sequence[PathLike],
    *,
    label: str,
) -> None:
    """Reject output aliases of each other or of any bound input artifact.

    Resolving paths catches lexical and symbolic-link aliases.  ``samefile``
    additionally catches existing hard links, which must not provide an
    alternate spelling that permits an input artifact to be overwritten.
    Archive-member specifications protect their archive container, and bound
    input directories protect every descendant path.
    """

    outputs = [Path(value).expanduser().resolve() for value in output_paths]
    inputs = []
    for value in input_paths:
        raw = os.fspath(value)
        container = raw.partition("::")[0]
        inputs.append(Path(container).expanduser().resolve())

    def aliases(first: Path, second: Path) -> bool:
        if first == second:
            return True
        try:
            return first.samefile(second)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ValueError(
                "%s could not verify output/input path independence: %s and %s"
                % (label, first, second)
            ) from error

    for index, output in enumerate(outputs):
        if any(aliases(output, other) for other in outputs[:index]):
            raise ValueError("%s output paths collide with each other" % label)

    def aliases_directory_member(output: Path, source: Path) -> bool:
        if not source.is_dir():
            return False
        if source in output.parents:
            return True
        if not output.exists():
            return False
        try:
            return any(item.is_file() and aliases(output, item) for item in source.rglob("*"))
        except OSError as error:
            raise ValueError(
                "%s could not verify output/input directory independence: %s and %s"
                % (label, output, source)
            ) from error

    collisions = []
    for output in outputs:
        if any(
            aliases(output, source) or aliases_directory_member(output, source)
            for source in inputs
        ):
            collisions.append(output)
    if collisions:
        raise ValueError(
            "%s output would overwrite a bound input: %s"
            % (label, ", ".join(str(path) for path in sorted(set(collisions), key=str)))
        )


def atomic_json_dump(payload: Dict[str, Any], path: PathLike) -> None:
    """Write JSON atomically so interrupted jobs do not leave valid-looking files."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
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


def tensor_to_float(value: Union[torch.Tensor, float, int]) -> float:
    """Detach a scalar tensor for structured metric logging."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("only scalar tensors can be converted to a metric")
        return float(value.detach().cpu().item())
    return float(value)


def optional_import_error(package: str, extra: Optional[str] = None) -> ImportError:
    """Create one consistent actionable error for optional dependencies."""

    suffix = ""
    if extra:
        suffix = " Install the project with `pip install -e '.[%s]'`." % extra
    return ImportError("Optional dependency %s is required.%s" % (package, suffix))
