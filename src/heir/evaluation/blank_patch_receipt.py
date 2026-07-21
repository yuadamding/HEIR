"""Receipt contract for the outcome-blind HEST H-optimus blank control."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.features.base import EncoderManifest, sha256_file, verified_model_file

RECEIPT_SCHEMA = "heir.hest_blank_patch_supplement.v1"
SUPPLEMENT_SCHEMA = "heir.hest_blank_patch_supplement_npz.v1"
CONTROL_ID = "blank_white_negative_control"
BROADCAST_RULE = "repeat_identical_feature_for_each_observation"
SOURCE_MEMBER_ALLOWLIST = frozenset({"observation_ids"})
SUPPLEMENT_MEMBERS = frozenset(
    {
        "schema_version",
        "blank_patch_feature",
        "registered_source_sha256",
        "ordered_observation_ids_sha256",
        "source_rows",
        "broadcast_rows",
        "broadcast_rule",
        "encoder_manifest_sha256",
        "feature_sha256",
    }
)

ENCODER_REPOSITORY = "bioptimus/H-optimus-1"
ENCODER_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
ENCODER_MANIFEST_SHA256 = "f6852288e1ae146a4865bf19e38ce994c0be9ce1c2bfa09bdf77747043ac8fd9"
ENCODER_CONFIG_SHA256 = "b10c4f37ce804ff58bec7f2ffd35cc29ecbfdb2b96ac81f3c1b3e37e2b27616e"
ENCODER_CHECKPOINT_SHA256 = "c4f1e5b457ddf00679626053b0bf2899be6a19c3a04ad191c87ad1cdfd1abfe1"
INPUT_PIXELS = 224
FEATURE_WIDTH = 1536
PARITY_MINIMUM_COSINE = 0.9999

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SourceIdentity:
    """Outcome-blind identity of the registered source and its ordered rows."""

    path: Path
    sha256: str
    bytes: int
    rows: int
    observation_ids_sha256: str
    members_opened: tuple[str, ...] = ("observation_ids",)


@dataclass(frozen=True)
class SupplementIdentity:
    """Validated compact blank supplement."""

    path: Path
    sha256: str
    feature: np.ndarray
    registered_source_sha256: str
    ordered_observation_ids_sha256: str
    source_rows: int
    broadcast_rows: int
    broadcast_rule: str
    encoder_manifest_sha256: str
    feature_sha256: str


def array_sha256(value: object) -> str:
    """Hash array semantics, including dtype and shape, before its contiguous bytes."""

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("receipt-bound arrays cannot contain Python objects")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def blank_rgb(batch_size: int = 1) -> np.ndarray:
    """Construct the only image input accepted for this control."""

    if batch_size <= 0:
        raise ValueError("blank batch size must be positive")
    return np.full((batch_size, INPUT_PIXELS, INPUT_PIXELS, 3), 255, dtype=np.uint8)


def read_source_member(path: Path, name: str) -> np.ndarray:
    """Read one explicitly allowed NPY member directly from the source ZIP."""

    if name not in SOURCE_MEMBER_ALLOWLIST:
        raise ValueError(f"source member {name!r} is not allowed for the blank receipt")
    member = f"{name}.npy"
    resolved = path.expanduser().resolve()
    try:
        with zipfile.ZipFile(resolved) as archive:
            if archive.namelist().count(member) != 1:
                raise ValueError(f"registered source must contain exactly one {member}")
            with archive.open(member) as handle:
                value = np.lib.format.read_array(handle, allow_pickle=False)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("registered source is not a readable NPZ archive") from error
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"registered source member {name} cannot use object dtype")
    return array


def _ordered_observation_ids(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"S", "U"}:
        raise ValueError("registered observation_ids must be a one-dimensional string array")
    text = array.astype(str)
    if len(text) == 0 or np.any(np.char.str_len(text) == 0):
        raise ValueError("registered observation_ids cannot be empty")
    if len(set(text.tolist())) != len(text):
        raise ValueError("registered observation_ids must be unique")
    return np.asarray(text, dtype="S")


def ordered_observation_ids_sha256(value: object) -> str:
    """Hash validated observation IDs independently of their string storage dtype."""

    return array_sha256(_ordered_observation_ids(value))


def inspect_registered_source(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> SourceIdentity:
    """Verify source bytes, then bind order using only ``observation_ids``."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not _valid_sha256(expected_sha256):
        raise ValueError("expected registered source SHA-256 is invalid")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError("registered source SHA-256 differs from the frozen protocol")
    observation_ids = _ordered_observation_ids(read_source_member(resolved, "observation_ids"))
    if len(observation_ids) != expected_rows or expected_rows <= 0:
        raise ValueError("registered source row count differs from the frozen protocol")
    return SourceIdentity(
        path=resolved,
        sha256=actual_sha256,
        bytes=resolved.stat().st_size,
        rows=len(observation_ids),
        observation_ids_sha256=ordered_observation_ids_sha256(observation_ids),
    )


def validate_encoder_contract(manifest: EncoderManifest, model_dir: Optional[Path] = None) -> None:
    """Require the exact H-optimus identity used by the registered HEST source."""

    expected = {
        "sha256": ENCODER_MANIFEST_SHA256,
        "repository": ENCODER_REPOSITORY,
        "revision": ENCODER_REVISION,
        "config_sha256": ENCODER_CONFIG_SHA256,
        "checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
        "feature_width": FEATURE_WIDTH,
        "input_pixels": INPUT_PIXELS,
        "model_mpp": 0.5,
        "interpolation": "bicubic",
        "pooling_rule": "direct_features",
        "fine_tuning": "prohibited",
    }
    if not manifest.available or {key: getattr(manifest, key) for key in expected} != expected:
        raise ValueError("encoder manifest differs from the frozen HEST H-optimus contract")
    if manifest.config_filename != "config.json" or manifest.checkpoint_filename != (
        "model.safetensors"
    ):
        raise ValueError("encoder model filenames differ from the frozen H-optimus contract")
    if model_dir is not None:
        verified_model_file(model_dir, manifest.config_filename, ENCODER_CONFIG_SHA256)
        verified_model_file(model_dir, manifest.checkpoint_filename, ENCODER_CHECKPOINT_SHA256)


def validate_feature_matrix(value: object, rows: int) -> np.ndarray:
    """Reject malformed, non-production, non-finite, and zero embeddings."""

    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or array.shape != (rows, FEATURE_WIDTH):
        raise ValueError(f"blank feature must be float32 with shape ({rows}, {FEATURE_WIDTH})")
    if not np.isfinite(array).all():
        raise ValueError("blank feature contains non-finite values")
    if np.any(np.linalg.norm(array.astype(np.float64), axis=1) <= 0.0):
        raise ValueError("blank feature cannot contain a zero vector")
    return np.ascontiguousarray(array)


def batch_parity_receipt(batch_one: object, batch_32: object) -> Mapping[str, object]:
    """Compare production inference at batch sizes one and 32."""

    one = validate_feature_matrix(batch_one, 1)
    many = validate_feature_matrix(batch_32, 32)
    repeated = np.repeat(one.astype(np.float64), len(many), axis=0)
    many64 = many.astype(np.float64)
    denominator = np.linalg.norm(repeated, axis=1) * np.linalg.norm(many64, axis=1)
    if np.any(denominator <= 0.0):
        raise ValueError("batch parity cosine is undefined for a zero vector")
    cosine = np.sum(repeated * many64, axis=1) / denominator
    minimum = float(cosine.min())
    passed = minimum >= PARITY_MINIMUM_COSINE
    result: Mapping[str, object] = {
        "batch_sizes": [1, 32],
        "batch_1_feature_sha256": array_sha256(one),
        "batch_32_features_sha256": array_sha256(many),
        "minimum_cosine": minimum,
        "mean_cosine": float(cosine.mean()),
        "maximum_absolute_error": float(np.max(np.abs(repeated - many64))),
        "minimum_required_cosine": PARITY_MINIMUM_COSINE,
        "all_finite": True,
        "all_nonzero": True,
        "passed": passed,
    }
    if not passed:
        raise ValueError("batch-size 1 and 32 blank embeddings fail frozen parity tolerance")
    return result


def supplement_payload(
    source: SourceIdentity,
    manifest: EncoderManifest,
    feature: object,
) -> Mapping[str, np.ndarray]:
    """Create a compact one-vector NPZ payload; no duplicate row matrix is permitted."""

    values = validate_feature_matrix(feature, 1)
    return {
        "schema_version": np.asarray(SUPPLEMENT_SCHEMA),
        "blank_patch_feature": values,
        "registered_source_sha256": np.asarray(source.sha256),
        "ordered_observation_ids_sha256": np.asarray(source.observation_ids_sha256),
        "source_rows": np.asarray(source.rows, dtype=np.int64),
        "broadcast_rows": np.asarray(source.rows, dtype=np.int64),
        "broadcast_rule": np.asarray(BROADCAST_RULE),
        "encoder_manifest_sha256": np.asarray(manifest.sha256),
        "feature_sha256": np.asarray(array_sha256(values)),
    }


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = np.asarray(archive[name])
    if value.ndim != 0 or value.dtype.hasobject:
        raise ValueError(f"blank supplement member {name} must be a non-object scalar")
    return value.item()


def load_supplement(path: Path) -> SupplementIdentity:
    """Load and validate the compact supplement without allowing pickle data."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != SUPPLEMENT_MEMBERS:
                raise ValueError("blank supplement members differ from the frozen schema")
            if str(_scalar(archive, "schema_version")) != SUPPLEMENT_SCHEMA:
                raise ValueError("blank supplement schema is unsupported")
            feature = validate_feature_matrix(archive["blank_patch_feature"], 1).copy()
            identity = SupplementIdentity(
                path=resolved,
                sha256=sha256_file(resolved),
                feature=feature,
                registered_source_sha256=str(_scalar(archive, "registered_source_sha256")),
                ordered_observation_ids_sha256=str(
                    _scalar(archive, "ordered_observation_ids_sha256")
                ),
                source_rows=int(_scalar(archive, "source_rows")),
                broadcast_rows=int(_scalar(archive, "broadcast_rows")),
                broadcast_rule=str(_scalar(archive, "broadcast_rule")),
                encoder_manifest_sha256=str(_scalar(archive, "encoder_manifest_sha256")),
                feature_sha256=str(_scalar(archive, "feature_sha256")),
            )
    except (KeyError, OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("blank supplement"):
            raise
        raise ValueError("blank supplement is malformed") from error
    if identity.feature_sha256 != array_sha256(identity.feature):
        raise ValueError("blank supplement feature hash is invalid")
    if identity.broadcast_rule != BROADCAST_RULE or identity.broadcast_rows != identity.source_rows:
        raise ValueError("blank supplement broadcast contract is invalid")
    return identity


def build_receipt(
    *,
    source: SourceIdentity,
    supplement: SupplementIdentity,
    manifest: EncoderManifest,
    protocol_sha256: str,
    parity: Mapping[str, object],
    implementation: Mapping[str, object],
    runtime: Mapping[str, object],
) -> Mapping[str, object]:
    """Assemble the auditable JSON sidecar for a real production encoding."""

    validate_encoder_contract(manifest)
    _validate_sha256_mapping(implementation, "implementation")
    if not _valid_sha256(protocol_sha256):
        raise ValueError("protocol SHA-256 is invalid")
    if runtime.get("git_worktree_dirty_at_start") is not False:
        raise ValueError("blank supplement construction requires a clean Git worktree")
    if not _valid_git_commit(runtime.get("git_commit")):
        raise ValueError("blank supplement runtime requires a committed Git identity")
    receipt: Mapping[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "scope": "outcome_blind_blank_image_control_only",
        "control_id": CONTROL_ID,
        "protocol_sha256": protocol_sha256,
        "registered_source_sha256": source.sha256,
        "registered_source_bytes": source.bytes,
        "ordered_observation_ids_sha256": source.observation_ids_sha256,
        "source_rows": source.rows,
        "source_members_opened": list(source.members_opened),
        "source_bytes_hashed_without_deserialization": True,
        "encoder_repository": manifest.repository,
        "encoder_revision": manifest.revision,
        "encoder_manifest_sha256": manifest.sha256,
        "config_sha256": manifest.config_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "input_shape": [1, INPUT_PIXELS, INPUT_PIXELS, 3],
        "input_dtype": "uint8",
        "input_layout": "NHWC_RGB",
        "input_rgb_value": [255, 255, 255],
        "input_sha256": array_sha256(blank_rgb()),
        "feature_shape": [1, FEATURE_WIDTH],
        "feature_dtype": "float32",
        "feature_sha256": supplement.feature_sha256,
        "broadcast_rows": source.rows,
        "broadcast_rule": BROADCAST_RULE,
        "supplement_schema": SUPPLEMENT_SCHEMA,
        "supplement_sha256": supplement.sha256,
        "supplement_members": sorted(SUPPLEMENT_MEMBERS),
        "batch_parity": dict(parity),
        "implementation": dict(implementation),
        "runtime": dict(runtime),
        "molecular_outcomes_read": False,
        "predictions_read": False,
        "biological_experiment_run": False,
    }
    validate_receipt(
        receipt,
        source=source,
        supplement=supplement,
        manifest=manifest,
        protocol_sha256=protocol_sha256,
        expected_implementation=implementation,
        recomputed_batch_one=supplement.feature,
        recomputed_batch_32=None,
    )
    return receipt


def validate_receipt(
    receipt: Mapping[str, object],
    *,
    source: SourceIdentity,
    supplement: SupplementIdentity,
    manifest: EncoderManifest,
    protocol_sha256: str,
    expected_implementation: Mapping[str, object],
    recomputed_batch_one: object,
    recomputed_batch_32: Optional[object],
) -> None:
    """Validate all identities and require pixel recomputation of the stored feature."""

    validate_encoder_contract(manifest)
    _validate_sha256_mapping(expected_implementation, "expected implementation")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "scope": "outcome_blind_blank_image_control_only",
        "control_id": CONTROL_ID,
        "protocol_sha256": protocol_sha256,
        "registered_source_sha256": source.sha256,
        "registered_source_bytes": source.bytes,
        "ordered_observation_ids_sha256": source.observation_ids_sha256,
        "source_rows": source.rows,
        "source_members_opened": ["observation_ids"],
        "source_bytes_hashed_without_deserialization": True,
        "encoder_repository": ENCODER_REPOSITORY,
        "encoder_revision": ENCODER_REVISION,
        "encoder_manifest_sha256": ENCODER_MANIFEST_SHA256,
        "config_sha256": ENCODER_CONFIG_SHA256,
        "checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
        "input_shape": [1, INPUT_PIXELS, INPUT_PIXELS, 3],
        "input_dtype": "uint8",
        "input_layout": "NHWC_RGB",
        "input_rgb_value": [255, 255, 255],
        "input_sha256": array_sha256(blank_rgb()),
        "feature_shape": [1, FEATURE_WIDTH],
        "feature_dtype": "float32",
        "feature_sha256": supplement.feature_sha256,
        "broadcast_rows": source.rows,
        "broadcast_rule": BROADCAST_RULE,
        "supplement_schema": SUPPLEMENT_SCHEMA,
        "supplement_sha256": supplement.sha256,
        "supplement_members": sorted(SUPPLEMENT_MEMBERS),
        "implementation": dict(expected_implementation),
        "molecular_outcomes_read": False,
        "predictions_read": False,
        "biological_experiment_run": False,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"blank receipt field {name} differs from its frozen identity")
    if set(receipt) != set(expected) | {"batch_parity", "runtime"}:
        raise ValueError("blank receipt fields differ from the frozen schema")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("blank receipt runtime is malformed")
    if runtime.get("git_worktree_dirty_at_start") is not False or not _valid_git_commit(
        runtime.get("git_commit")
    ):
        raise ValueError("blank receipt was not produced from a clean committed worktree")
    if (
        supplement.registered_source_sha256 != source.sha256
        or supplement.ordered_observation_ids_sha256 != source.observation_ids_sha256
        or supplement.source_rows != source.rows
        or supplement.broadcast_rows != source.rows
        or supplement.encoder_manifest_sha256 != manifest.sha256
    ):
        raise ValueError("blank supplement is not aligned to its source and encoder")
    recomputed_one = validate_feature_matrix(recomputed_batch_one, 1)
    if array_sha256(recomputed_one) != supplement.feature_sha256:
        raise ValueError("white-pixel recomputation did not reproduce the registered feature hash")
    parity = receipt.get("batch_parity")
    if not isinstance(parity, Mapping):
        raise ValueError("blank receipt batch parity is malformed")
    if recomputed_batch_32 is not None:
        fresh_parity = batch_parity_receipt(recomputed_one, recomputed_batch_32)
        if fresh_parity["batch_1_feature_sha256"] != supplement.feature_sha256:
            raise ValueError("recomputed blank batch-one feature differs from the supplement")
    else:
        required = {
            "batch_sizes",
            "batch_1_feature_sha256",
            "batch_32_features_sha256",
            "minimum_cosine",
            "mean_cosine",
            "maximum_absolute_error",
            "minimum_required_cosine",
            "all_finite",
            "all_nonzero",
            "passed",
        }
        if (
            set(parity) != required
            or parity.get("batch_sizes") != [1, 32]
            or parity.get("batch_1_feature_sha256") != supplement.feature_sha256
            or not _valid_sha256(parity.get("batch_32_features_sha256"))
            or not np.isfinite(float(parity.get("minimum_cosine", np.nan)))
            or float(parity.get("minimum_cosine", -np.inf)) < PARITY_MINIMUM_COSINE
            or not np.isfinite(float(parity.get("mean_cosine", np.nan)))
            or not np.isfinite(float(parity.get("maximum_absolute_error", np.nan)))
            or float(parity.get("maximum_absolute_error", -1.0)) < 0.0
            or parity.get("minimum_required_cosine") != PARITY_MINIMUM_COSINE
            or parity.get("all_finite") is not True
            or parity.get("all_nonzero") is not True
            or parity.get("passed") is not True
        ):
            raise ValueError("blank receipt batch parity contract is invalid")


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None and value != "0" * 64


def _valid_git_commit(value: object) -> bool:
    return isinstance(value, str) and _GIT_COMMIT.fullmatch(value) is not None


def _validate_sha256_mapping(value: Mapping[str, object], name: str) -> None:
    if not value or any(not _valid_sha256(item) for item in value.values()):
        raise ValueError(f"{name} must contain nonzero SHA-256 identities")


__all__: Sequence[str] = (
    "BROADCAST_RULE",
    "CONTROL_ID",
    "FEATURE_WIDTH",
    "PARITY_MINIMUM_COSINE",
    "RECEIPT_SCHEMA",
    "SOURCE_MEMBER_ALLOWLIST",
    "SUPPLEMENT_MEMBERS",
    "SUPPLEMENT_SCHEMA",
    "SourceIdentity",
    "SupplementIdentity",
    "array_sha256",
    "batch_parity_receipt",
    "blank_rgb",
    "build_receipt",
    "inspect_registered_source",
    "load_supplement",
    "read_source_member",
    "supplement_payload",
    "validate_encoder_contract",
    "validate_feature_matrix",
    "validate_receipt",
)
