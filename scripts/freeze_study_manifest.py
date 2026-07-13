#!/usr/bin/env python3
"""Freeze a draft study manifest against a clean Git commit and container digest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from heir.data import (
    StudyManifest,
    current_git_commit,
    freeze_manifest_content,
    require_clean_worktree,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--container-digest",
        required=True,
        help="immutable sha256:<64 hex> digest of the execution image",
    )
    args = parser.parse_args(argv)
    draft_path = args.draft.expanduser().resolve()
    output = args.output.expanduser().resolve()
    root = args.repository_root.expanduser().resolve()
    if not draft_path.is_file():
        raise ValueError("draft study manifest does not exist")
    reject_output_input_collisions((output,), (draft_path,), label="study manifest freeze")
    draft = StudyManifest.load(draft_path, require_status="draft")
    require_clean_worktree(root)
    locked = freeze_manifest_content(
        draft.content,
        git_commit=current_git_commit(root),
        container_digest=args.container_digest,
    )
    atomic_json_dump(dict(locked), output)
    verified = StudyManifest.load(
        output,
        require_status="locked",
        verify_runtime=True,
        repository_root=root,
    )
    print(json.dumps({"study_id": verified.study_id, "sha256": verified.sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

