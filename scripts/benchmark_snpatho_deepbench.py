#!/usr/bin/env python3
"""Run the retrospective snPATHO-DeepBench-v1 executable subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heir.evaluation import (
    load_deepbench_plan,
    load_snpatho_plan,
    run_deepbench,
    write_deepbench_report,
)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=repository / "configs" / "experiments" / "snpatho_deepbench_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    plan = load_deepbench_plan(args.plan)
    locked_plan = load_snpatho_plan(plan.frozen_plan)
    outputs = {
        path.expanduser().resolve()
        for path in (args.output, args.tsv, args.markdown)
        if path is not None
    }
    if len(outputs) != sum(value is not None for value in (args.output, args.tsv, args.markdown)):
        raise ValueError("DeepBench output paths must be distinct")
    reserved = {
        plan.source_path,
        plan.frozen_plan,
        plan.historical_report,
        locked_plan.gene_panel,
    }
    for case in locked_plan.cases:
        reserved.update((case.predictions, case.truth, case.matched_reference, case.telemetry))
    reserved.update(path for path in plan.optional_artifacts.values() if path is not None)
    if outputs & reserved:
        raise ValueError("DeepBench output must not overwrite a plan or input artifact")
    report = run_deepbench(plan)
    output, tabular, markdown = write_deepbench_report(
        report,
        json_path=args.output,
        tsv_path=args.tsv,
        markdown_path=args.markdown,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "tsv": None if tabular is None else str(tabular),
                "markdown": None if markdown is None else str(markdown),
                "status": report["benchmark"]["status"],
                "requested_primary_status": report["primary"]["requested_primary_status"],
                "full_plan_complete": report["reporting"]["full_plan_complete"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
