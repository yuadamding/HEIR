"""Portable external-checkpoint manifest tests."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_external_asset_verifier_accepts_exact_file_and_rejects_corruption(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path / "pretrained"
    checkpoint = root / "tiny" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"compact-test-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "heir.pretrained_assets.v1",
                "environment_variable": "HEIR_PRETRAINED_DIR",
                "assets": {
                    "tiny": {
                        "relative_path": "tiny/checkpoint.pt",
                        "sha256": digest,
                        "bytes": checkpoint.stat().st_size,
                        "role": "test",
                        "tracked_by_git": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(repository / "scripts" / "manage_pretrained_assets.py"),
        "verify",
        "--root",
        str(root),
        "--manifest",
        str(manifest),
    ]
    valid = subprocess.run(command, check=False, capture_output=True, text=True)
    assert valid.returncode == 0
    assert json.loads(valid.stdout)["assets"][0]["valid"] is True

    checkpoint.write_bytes(b"corrupted")
    invalid = subprocess.run(command, check=False, capture_output=True, text=True)
    assert invalid.returncode == 1
    assert json.loads(invalid.stdout)["assets"][0]["valid"] is False
