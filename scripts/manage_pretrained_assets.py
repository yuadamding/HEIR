#!/usr/bin/env python3
"""Locate and checksum external pretrained assets required by HEIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") != "heir.pretrained_assets.v1":
        raise ValueError("pretrained asset manifest has an unsupported schema")
    assets = payload.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        raise ValueError("pretrained asset manifest contains no assets")
    return payload


def _asset_root(repository: Path, override: Optional[Path]) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    configured = os.environ.get("HEIR_PRETRAINED_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (repository.parent / "HEIR_assets" / "pretrained").resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "verify"))
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository / "assets" / "pretrained_checkpoints.json",
    )
    args = parser.parse_args(argv)
    payload = _load_manifest(args.manifest.expanduser().resolve())
    root = _asset_root(repository, args.root)
    records = []
    for name, raw in sorted(payload["assets"].items()):
        if not isinstance(raw, Mapping):
            raise ValueError("asset %s metadata is malformed" % name)
        path = (root / str(raw["relative_path"])).resolve()
        record = {
            "name": str(name),
            "path": str(path),
            "exists": path.is_file(),
            "expected_bytes": int(raw["bytes"]),
            "expected_sha256": str(raw["sha256"]),
            "role": str(raw["role"]),
        }
        if args.command == "verify" and path.is_file():
            record["observed_bytes"] = path.stat().st_size
            record["observed_sha256"] = _sha256(path)
            record["valid"] = (
                record["observed_bytes"] == record["expected_bytes"]
                and record["observed_sha256"] == record["expected_sha256"]
            )
        elif args.command == "verify":
            record["valid"] = False
        records.append(record)
    print(json.dumps({"root": str(root), "assets": records}, indent=2, sort_keys=True))
    return 0 if args.command == "list" or all(item.get("valid") for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
