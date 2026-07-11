#!/usr/bin/env python3
"""Score a frozen three-sample snPATHO benchmark plan exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heir.evaluation import (
    load_snpatho_plan,
    run_snpatho_benchmark,
    write_snpatho_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-donors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="development/testing only; the reported benchmark normally requires all three donors",
    )
    args = parser.parse_args()
    plan = load_snpatho_plan(args.plan)
    outputs = {args.output.expanduser().resolve()}
    if args.tsv is not None:
        outputs.add(args.tsv.expanduser().resolve())
    reserved = {plan.source_path, plan.gene_panel}
    for case in plan.cases:
        reserved.update((case.predictions, case.truth, case.matched_reference))
        if case.telemetry is not None:
            reserved.add(case.telemetry)
    if outputs & reserved:
        raise ValueError("benchmark output must not overwrite its plan or input artifacts")
    result = run_snpatho_benchmark(
        plan,
        seed=args.seed,
        iterations=args.iterations,
        confidence=args.confidence,
        minimum_donors=args.minimum_donors,
        require_complete=not args.allow_partial,
    )
    output, tabular = write_snpatho_benchmark(
        result,
        json_path=args.output,
        tsv_path=args.tsv,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "tsv": None if tabular is None else str(tabular),
                "cases": [case.section_id for case in result.cases],
                "donor_metrics": len(result.benchmark.donor_metrics),
                "summaries": len(result.benchmark.summaries),
                "comparisons": len(result.benchmark.comparisons),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
