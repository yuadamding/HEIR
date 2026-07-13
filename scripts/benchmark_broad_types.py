#!/usr/bin/env python3
"""Run the label-dependent, no-graph broad-cell-type development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from heir.evaluation.broad_type import (
    BroadTypeGateBlocked,
    blocked_broad_type_report,
    inspect_broad_type_gate,
    run_broad_type_gate,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions


def _plan_input_paths(path: Path) -> tuple[Path, ...]:
    """Return the plan and every artifact path it declares, even if validation fails."""

    source = path.expanduser().resolve()
    inputs = [source]
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return tuple(inputs)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "path" and isinstance(child, str) and child.strip():
                    candidate = Path(child).expanduser()
                    inputs.append(
                        candidate.resolve()
                        if candidate.is_absolute()
                        else (source.parent / candidate).resolve()
                    )
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return tuple(dict.fromkeys(inputs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Prespecified JSON plan with hashed labels")
    parser.add_argument("--output", required=True, help="Destination report JSON")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override the plan device for the small frozen-feature classifier",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Check manifests and label readiness without loading features or fitting models",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = Path(args.plan)
    reject_output_input_collisions(
        [args.output],
        _plan_input_paths(plan_path),
        label="broad-type benchmark",
    )
    if args.inspect:
        report = inspect_broad_type_gate(plan_path)
        atomic_json_dump(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    try:
        report = run_broad_type_gate(plan_path, device_name=args.device)
    except (BroadTypeGateBlocked, FileNotFoundError, ValueError) as error:
        report = blocked_broad_type_report(plan_path, error)
        atomic_json_dump(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
