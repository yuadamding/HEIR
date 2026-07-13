#!/usr/bin/env python3
"""Validate a draft, locked, or opened HEIR study manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from heir.data import StudyManifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args(argv)
    manifest = StudyManifest.load(
        args.manifest,
        verify_runtime=args.verify_runtime,
        repository_root=args.repository_root,
    )
    print(
        json.dumps(
            {
                "study_id": manifest.study_id,
                "status": manifest.status,
                "hypothesis_ids": manifest.hypothesis_ids,
                "sha256": manifest.sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

