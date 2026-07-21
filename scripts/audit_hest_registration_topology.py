#!/usr/bin/env python3
"""Write the outcome-blind HEST registration-topology receipt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.evaluation.nonlinear_qualification_contract import file_sha256, validate_protocol
from heir.evaluation.registration_topology import audit_topology, build_receipt
from heir.features.base import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz")
DEFAULT_PROTOCOL = ROOT / "configs/hest_nonlinear_qualification_v1.json"
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/hest_nonlinear_v2/registration_topology.json")
IMPLEMENTATION_FILES = {
    "audit_script_sha256": Path(__file__).resolve(),
    "topology_contract_sha256": ROOT / "src/heir/evaluation/registration_topology.py",
    "array_identity_sha256": ROOT / "src/heir/evaluation/blank_patch_receipt.py",
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
        "source_sha256": str(source["sha256"]),
        "source_rows": int(dataset["registered_cells"]),
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
        raise ValueError("cannot inspect Git identity for the topology audit") from error
    if status:
        raise ValueError("registration-topology audit requires a committed, clean worktree")
    return {
        "git_commit": commit,
        "git_worktree_dirty_at_start": False,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }


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


def run(source: Path, protocol: Path, output: Path) -> Mapping[str, object]:
    runtime = _git_identity()
    contract = _protocol_contract(protocol)
    audit = audit_topology(
        source,
        expected_sha256=str(contract["source_sha256"]),
        expected_rows=int(contract["source_rows"]),
    )
    receipt = build_receipt(
        {**audit, "protocol_sha256": contract["protocol_sha256"]},
        implementation={name: sha256_file(path) for name, path in IMPLEMENTATION_FILES.items()},
        runtime=runtime,
    )
    _atomic_json(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run(
        args.source.expanduser().resolve(),
        args.protocol.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "status": receipt["status"],
                "source_sha256": receipt["registered_source_sha256"],
                "evaluation_rows": receipt["evaluation_rows"],
                "selected_band": receipt["selected_band"],
                "selection_status": receipt["selection_status"],
                "molecular_outcomes_read": receipt["molecular_outcomes_read"],
                "output": str(args.output.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
