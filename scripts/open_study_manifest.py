#!/usr/bin/env python3
"""Record the one-way opening of a locked cohort manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from heir.data import StudyManifest, current_git_commit, open_manifest_content
from heir.utils import atomic_json_dump, reject_output_input_collisions


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--permitted-claim", action="append", default=[])
    args = parser.parse_args(argv)
    locked_path = args.locked_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    reject_output_input_collisions((output,), (locked_path,), label="study manifest opening")
    locked = StudyManifest.load(
        locked_path,
        require_status="locked",
        verify_runtime=True,
        repository_root=args.repository_root,
    )
    opened = open_manifest_content(
        locked,
        opened_by_commit=current_git_commit(args.repository_root),
        permitted_claims=args.permitted_claim,
    )
    atomic_json_dump(dict(opened), output)
    verified = StudyManifest.load(output, require_status="opened")
    print(json.dumps({"study_id": verified.study_id, "sha256": verified.sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
