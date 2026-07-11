#!/usr/bin/env python3
"""Run a prespecified development-only train/refine/predict/evaluate sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_plan(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "heir.refinement_development_plan.v1"
    ):
        raise ValueError("development plan has an unsupported schema")
    role = str(payload.get("analysis_role", "")).lower()
    if role not in {"development", "development_validation"}:
        raise ValueError("refinement orchestration is restricted to development roles")
    rounds = int(payload.get("refinement_rounds", 0))
    if rounds <= 0 or rounds > 5:
        raise ValueError("refinement_rounds must lie in [1, 5]")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not all(
        isinstance(item, Mapping) for item in raw_stages
    ):
        raise ValueError("development plan stages must be a list of mappings")
    expected = ["train", "predict_round_0"]
    for round_id in range(1, rounds + 1):
        expected.extend(("refine_round_%d" % round_id, "predict_round_%d" % round_id))
    expected.append("development_spatial_evaluation")
    observed = [str(item.get("name", "")) for item in raw_stages]
    if observed != expected:
        raise ValueError("development stages must be exactly: %s" % ", ".join(expected))
    for stage in raw_stages:
        command = stage.get("command")
        outputs = stage.get("outputs")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) and value for value in command)
        ):
            raise ValueError("every development stage requires a non-empty argv command")
        if not isinstance(outputs, list) or not all(
            isinstance(value, str) and value for value in outputs
        ):
            raise ValueError("every development stage requires an outputs list")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", type=Path)
    args = parser.parse_args(argv)
    plan_path = args.plan.expanduser().resolve()
    plan = _load_plan(plan_path)
    records = []
    for raw in plan["stages"]:
        stage = dict(raw)
        outputs = [
            (repository / value).resolve()
            if not Path(value).is_absolute()
            else Path(value).resolve()
            for value in stage["outputs"]
        ]
        complete = bool(outputs) and all(path.is_file() for path in outputs)
        if complete:
            state = "skipped_existing"
        elif not args.execute:
            state = "planned"
        else:
            subprocess.run(stage["command"], cwd=repository, check=True)
            missing = [str(path) for path in outputs if not path.is_file()]
            if missing:
                raise RuntimeError(
                    "stage %s completed without outputs: %s" % (stage["name"], ", ".join(missing))
                )
            state = "completed"
        records.append(
            {
                "name": stage["name"],
                "state": state,
                "command": stage["command"],
                "outputs": [str(path) for path in outputs],
            }
        )
    result = {
        "schema": "heir.refinement_development_status.v1",
        "analysis_role": plan["analysis_role"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "execute": bool(args.execute),
        "stages": records,
    }
    if args.status is not None:
        _atomic_json(args.status.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
