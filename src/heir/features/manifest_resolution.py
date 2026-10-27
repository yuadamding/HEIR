"""Resolve a frozen manifest's location without changing its byte identity."""

from pathlib import Path

from .base import sha256_file


def resolve_encoder_manifest(
    configured_path, expected_sha256, *, repository_root=None, asset_roots=()
):
    """Find the exact manifest locally; never inspect or download its checkpoint.

    An existing configured file is authoritative and must match. A missing file
    can move to ``manifests/encoders/<basename>`` in a checkout or to that layout
    (or a direct basename) in an explicitly supplied asset root. No filename or
    location substitutes for the required SHA-256. Candidate paths are bounded;
    unrelated directories are never searched recursively.
    """
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("encoder manifest identity requires a lowercase SHA-256")
    configured = Path(configured_path).expanduser()
    if not configured.name or configured.name in {".", ".."}:
        raise ValueError("encoder manifest location requires a filename")
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    candidates = [configured, root / "manifests" / "encoders" / configured.name]
    for asset_root in asset_roots:
        asset_root = Path(asset_root).expanduser()
        candidates.extend(
            (asset_root / "manifests" / "encoders" / configured.name, asset_root / configured.name)
        )
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            if sha256_file(candidate) != expected_sha256:
                raise ValueError(f"encoder manifest SHA-256 mismatch: {candidate}")
            return candidate.resolve()
    raise FileNotFoundError(
        f"encoder manifest {configured.name!r} was not found at its configured location, "
        "repository manifest directory, or supplied asset roots"
    )
