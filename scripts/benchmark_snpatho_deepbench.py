#!/usr/bin/env python3
"""Run the retrospective snPATHO-DeepBench-v1 executable subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from heir.evaluation import (
    load_deepbench_plan,
    load_snpatho_plan,
    run_deepbench,
    write_deepbench_report,
)
from heir.utils import reject_output_input_collisions

_MANIFEST_PATH_KEYS = {
    "checkpoint",
    "cli_source",
    "decoder",
    "gene_panel",
    "latent_reference",
    "manifest",
    "module_entrypoint",
    "molecular_producer",
    "native_scanvi_manifest",
    "panel_reference",
    "predictions",
    "prototypes",
    "python_executable",
    "rare_complete_prototypes",
    "refined_prototype",
    "refinement_audit",
    "residual_geometry",
    "source_root",
    "telemetry",
    "truth",
}


def _transitive_manifest_inputs(
    manifests: Sequence[Path],
    *,
    repository: Path,
) -> tuple[Path, ...]:
    """Discover files/directories recursively declared by optional JSON manifests."""

    repository = repository.expanduser().resolve()
    inputs: list[Path] = []
    queue = [Path(path).expanduser().resolve() for path in manifests]
    parsed: set[Path] = set()
    while queue:
        manifest = queue.pop()
        if manifest in parsed or not manifest.is_file() or manifest.suffix.lower() != ".json":
            continue
        parsed.add(manifest)
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        bases = tuple(dict.fromkeys((manifest.parent, manifest.parent.parent, repository)))

        def visit(value: object) -> None:
            if isinstance(value, Mapping):
                for raw_key, child in value.items():
                    key = str(raw_key)
                    if (
                        (key == "path" or key.endswith("_path") or key in _MANIFEST_PATH_KEYS)
                        and isinstance(child, str)
                        and child.strip()
                    ):
                        candidate = Path(child).expanduser()
                        candidates = (
                            (candidate.resolve(),)
                            if candidate.is_absolute()
                            else tuple((base / candidate).resolve() for base in bases)
                        )
                        existing = tuple(path for path in candidates if path.exists())
                        for path in existing or candidates:
                            if path not in inputs:
                                inputs.append(path)
                            if path.is_file() and path.suffix.lower() == ".json":
                                queue.append(path)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
    return tuple(inputs)


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
    outputs = [path for path in (args.output, args.tsv, args.markdown) if path is not None]
    reserved = [
        plan.source_path,
        plan.frozen_plan,
        plan.historical_report,
        locked_plan.gene_panel,
    ]
    for case in locked_plan.cases:
        reserved.extend((case.predictions, case.truth, case.matched_reference))
        if case.telemetry is not None:
            reserved.append(case.telemetry)
    optional_manifests = tuple(
        path for path in plan.optional_artifacts.values() if path is not None
    )
    reserved.extend(optional_manifests)
    reserved.extend(_transitive_manifest_inputs(optional_manifests, repository=repository))
    reject_output_input_collisions(
        outputs,
        reserved,
        label="snPATHO DeepBench",
    )
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
