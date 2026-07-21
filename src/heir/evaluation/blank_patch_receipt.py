"""Receipt contract for the outcome-blind HEST blank-image supplement."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Mapping, Optional

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
        "control_id",
        "blank_patch_feature",
        "source_sha256",
        "observation_ids_sha256",
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
BATCH_PARITY_MINIMUM_COSINE = 0.9999
NON_AUTHORIZING_FIELDS = {
    "biological_experiment_run": False,
    "engineering_decision_available": False,
    "authorizes_h_cell": False,
    "authorizes_h_intrinsic": False,
    "authorizes_h_ref": False,
    "authorizes_full_heir": False,
}


def array_sha256(value: object) -> str:
    """Hash an array together with its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("receipt-bound arrays cannot contain Python objects")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def blank_pixels(rows: int = 1) -> np.ndarray:
    """Construct the exact registered white RGB negative control."""

    if rows < 1:
        raise ValueError("blank-patch row count must be positive")
    return np.full((rows, INPUT_PIXELS, INPUT_PIXELS, 3), 255, dtype=np.uint8)


def _direct_npz_member(path: Path, name: str, *, allowlist: frozenset[str]) -> np.ndarray:
    """Read one explicitly allowed NPY member without opening any sibling payload."""

    if name not in allowlist:
        raise ValueError(f"NPZ member {name!r} is outside the outcome-blind allowlist")
    member = f"{name}.npy"
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.namelist().count(member) != 1:
                raise ValueError(f"NPZ must contain exactly one {name} member")
            with archive.open(member) as handle:
                value = np.lib.format.read_array(handle, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot read allowed NPZ member {name}") from error
    result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError(f"allowed NPZ member {name} cannot contain objects")
    return result


def source_identity(
    source: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> Mapping[str, object]:
    """Bind source bytes and observation order without deserializing outcomes."""

    resolved = source.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError("registered HEST source SHA-256 differs from the frozen protocol")
    raw_ids = _direct_npz_member(
        resolved,
        "observation_ids",
        allowlist=SOURCE_MEMBER_ALLOWLIST,
    )
    if raw_ids.ndim != 1 or len(raw_ids) != expected_rows:
        raise ValueError("registered source observation IDs have the wrong shape")
    observation_ids = raw_ids.astype(str)
    if any(not value for value in observation_ids.tolist()):
        raise ValueError("registered source observation IDs cannot be empty")
    if len(set(observation_ids.tolist())) != expected_rows:
        raise ValueError("registered source observation IDs must be unique")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
        "rows": expected_rows,
        "observation_ids_sha256": array_sha256(np.asarray(observation_ids, dtype="S")),
        "members_read": ["observation_ids"],
        "member_allowlist": sorted(SOURCE_MEMBER_ALLOWLIST),
        "molecular_outcomes_read": False,
    }


def validate_encoder_contract(manifest: EncoderManifest, model_dir: Path) -> None:
    """Require the exact H-optimus identity used for the registered source."""

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
    if not manifest.available or {name: getattr(manifest, name) for name in expected} != expected:
        raise ValueError("encoder manifest differs from the frozen HEST H-optimus contract")
    if manifest.config_filename != "config.json" or manifest.checkpoint_filename != (
        "model.safetensors"
    ):
        raise ValueError("encoder filenames differ from the frozen HEST H-optimus contract")
    verified_model_file(model_dir, manifest.config_filename, ENCODER_CONFIG_SHA256)
    verified_model_file(model_dir, manifest.checkpoint_filename, ENCODER_CHECKPOINT_SHA256)


def validate_feature(value: object, *, rows: int) -> np.ndarray:
    """Return one finite, nonzero float32 feature matrix or fail closed."""

    feature = np.asarray(value)
    if feature.dtype != np.dtype(np.float32) or feature.shape != (rows, FEATURE_WIDTH):
        raise ValueError(f"blank feature must be float32 with shape ({rows}, {FEATURE_WIDTH})")
    if not np.isfinite(feature).all():
        raise ValueError("blank feature contains non-finite values")
    squared_norms = np.square(feature.astype(np.float64)).sum(axis=1)
    if np.any(squared_norms <= 0.0):
        raise ValueError("blank feature cannot contain a zero vector")
    return np.ascontiguousarray(feature)


def batch_parity_receipt(single: object, repeated: object) -> Mapping[str, object]:
    """Compare production encoding at exact input batch sizes one and 32."""

    one = validate_feature(single, rows=1)
    thirty_two = validate_feature(repeated, rows=32)
    numerator = np.sum(
        thirty_two.astype(np.float64) * one.astype(np.float64),
        axis=1,
    )
    denominator = np.linalg.norm(thirty_two, axis=1) * np.linalg.norm(one, axis=1)[0]
    if np.any(denominator <= 0.0):
        raise ValueError("batch-parity cosine is undefined for a zero vector")
    cosine = numerator / denominator
    absolute = np.abs(thirty_two.astype(np.float64) - one.astype(np.float64))
    minimum_cosine = float(cosine.min())
    passed = minimum_cosine >= BATCH_PARITY_MINIMUM_COSINE
    if not passed:
        raise ValueError("blank feature differs across production batch sizes 1 and 32")
    return {
        "batch_sizes": [1, 32],
        "single_feature_sha256": array_sha256(one),
        "repeated_features_sha256": array_sha256(thirty_two),
        "minimum_cosine": minimum_cosine,
        "minimum_required_cosine": BATCH_PARITY_MINIMUM_COSINE,
        "maximum_absolute_error": float(absolute.max()),
        "passed": True,
    }


def supplement_arrays(
    feature: object,
    *,
    source: Mapping[str, object],
    manifest: EncoderManifest,
) -> Mapping[str, np.ndarray]:
    """Construct the compact one-vector supplement payload."""

    value = validate_feature(feature, rows=1)
    rows = int(source["rows"])
    return {
        "schema_version": np.asarray(SUPPLEMENT_SCHEMA),
        "control_id": np.asarray(CONTROL_ID),
        "blank_patch_feature": value,
        "source_sha256": np.asarray(str(source["sha256"])),
        "observation_ids_sha256": np.asarray(str(source["observation_ids_sha256"])),
        "source_rows": np.asarray(rows, dtype=np.int64),
        "broadcast_rows": np.asarray(rows, dtype=np.int64),
        "broadcast_rule": np.asarray(BROADCAST_RULE),
        "encoder_manifest_sha256": np.asarray(manifest.sha256),
        "feature_sha256": np.asarray(array_sha256(value)),
    }


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError(f"blank supplement member {name} must be scalar")
    return value.item()


def load_supplement(path: Path) -> Mapping[str, object]:
    """Load and strictly validate the compact supplement container."""

    resolved = path.expanduser().resolve()
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != SUPPLEMENT_MEMBERS:
                raise ValueError("blank supplement members differ from the frozen schema")
            feature = validate_feature(archive["blank_patch_feature"], rows=1)
            values = {
                "schema": str(_scalar(archive, "schema_version")),
                "control_id": str(_scalar(archive, "control_id")),
                "feature": feature,
                "source_sha256": str(_scalar(archive, "source_sha256")),
                "observation_ids_sha256": str(_scalar(archive, "observation_ids_sha256")),
                "source_rows": int(_scalar(archive, "source_rows")),
                "broadcast_rows": int(_scalar(archive, "broadcast_rows")),
                "broadcast_rule": str(_scalar(archive, "broadcast_rule")),
                "encoder_manifest_sha256": str(_scalar(archive, "encoder_manifest_sha256")),
                "feature_sha256": str(_scalar(archive, "feature_sha256")),
            }
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("blank supplement is malformed") from error
    if values["schema"] != SUPPLEMENT_SCHEMA or values["control_id"] != CONTROL_ID:
        raise ValueError("blank supplement identity is unsupported")
    if values["feature_sha256"] != array_sha256(values["feature"]):
        raise ValueError("blank supplement feature hash is invalid")
    return {
        **values,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "members": sorted(SUPPLEMENT_MEMBERS),
    }


def make_receipt(
    *,
    source: Mapping[str, object],
    supplement: Mapping[str, object],
    manifest: EncoderManifest,
    protocol_sha256: str,
    parity: Mapping[str, object],
    implementation: Mapping[str, object],
    runtime: Mapping[str, object],
) -> Mapping[str, object]:
    """Create the auditable JSON receipt from already validated identities."""

    return {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "scope": "blank_control_only_no_molecular_outcomes",
        "control_id": CONTROL_ID,
        "source_protocol_sha256": protocol_sha256,
        "registered_source_sha256": source["sha256"],
        "ordered_observation_ids_sha256": source["observation_ids_sha256"],
        "source_rows": source["rows"],
        "source_member_allowlist": sorted(SOURCE_MEMBER_ALLOWLIST),
        "source_members_read": list(source["members_read"]),
        "source_bytes_hashed_without_deserialization": True,
        "encoder_repository": manifest.repository,
        "encoder_revision": manifest.revision,
        "encoder_manifest_sha256": manifest.sha256,
        "config_sha256": manifest.config_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "encoder_path": "heir.features.hoptimus1.HOptimus1Encoder.encode",
        "input_dtype": "uint8",
        "input_layout": "NHWC_RGB",
        "input_shape": [1, INPUT_PIXELS, INPUT_PIXELS, 3],
        "input_rgb_value": [255, 255, 255],
        "input_sha256": array_sha256(blank_pixels()),
        "feature_dtype": "float32",
        "feature_shape": [1, FEATURE_WIDTH],
        "feature_sha256": supplement["feature_sha256"],
        "broadcast_rows": source["rows"],
        "broadcast_rule": BROADCAST_RULE,
        "supplement_schema": SUPPLEMENT_SCHEMA,
        "supplement_sha256": supplement["sha256"],
        "supplement_members": supplement["members"],
        "batch_parity": dict(parity),
        "implementation": dict(implementation),
        "runtime": dict(runtime),
        "molecular_outcomes_read": False,
        "predictions_read": False,
        "query_outcomes_used": False,
        **NON_AUTHORIZING_FIELDS,
    }


def _sha256_text(value: object, name: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"blank receipt {name} must be a lowercase SHA-256")
    return text


def validate_receipt(
    receipt: Mapping[str, object],
    *,
    source: Mapping[str, object],
    supplement: Mapping[str, object],
    manifest: EncoderManifest,
    protocol_sha256: str,
    expected_implementation: Optional[Mapping[str, object]] = None,
    recomputed_single: Optional[np.ndarray] = None,
    recomputed_repeated: Optional[np.ndarray] = None,
) -> Mapping[str, object]:
    """Validate all receipt bindings, optionally including fresh production encoding."""

    expected = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "scope": "blank_control_only_no_molecular_outcomes",
        "control_id": CONTROL_ID,
        "source_protocol_sha256": protocol_sha256,
        "registered_source_sha256": source["sha256"],
        "ordered_observation_ids_sha256": source["observation_ids_sha256"],
        "source_rows": source["rows"],
        "source_member_allowlist": sorted(SOURCE_MEMBER_ALLOWLIST),
        "source_members_read": ["observation_ids"],
        "source_bytes_hashed_without_deserialization": True,
        "encoder_repository": ENCODER_REPOSITORY,
        "encoder_revision": ENCODER_REVISION,
        "encoder_manifest_sha256": manifest.sha256,
        "config_sha256": manifest.config_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "encoder_path": "heir.features.hoptimus1.HOptimus1Encoder.encode",
        "input_dtype": "uint8",
        "input_layout": "NHWC_RGB",
        "input_shape": [1, INPUT_PIXELS, INPUT_PIXELS, 3],
        "input_rgb_value": [255, 255, 255],
        "input_sha256": array_sha256(blank_pixels()),
        "feature_dtype": "float32",
        "feature_shape": [1, FEATURE_WIDTH],
        "feature_sha256": supplement["feature_sha256"],
        "broadcast_rows": source["rows"],
        "broadcast_rule": BROADCAST_RULE,
        "supplement_schema": SUPPLEMENT_SCHEMA,
        "supplement_sha256": supplement["sha256"],
        "supplement_members": supplement["members"],
        "molecular_outcomes_read": False,
        "predictions_read": False,
        "query_outcomes_used": False,
        **NON_AUTHORIZING_FIELDS,
    }
    if set(receipt) != set(expected) | {"batch_parity", "implementation", "runtime"}:
        raise ValueError("blank receipt fields differ from the frozen contract")
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"blank receipt field {name} differs from the frozen contract")
    if (
        supplement["source_sha256"] != source["sha256"]
        or supplement["observation_ids_sha256"] != source["observation_ids_sha256"]
        or supplement["source_rows"] != source["rows"]
        or supplement["broadcast_rows"] != source["rows"]
        or supplement["broadcast_rule"] != BROADCAST_RULE
        or supplement["encoder_manifest_sha256"] != manifest.sha256
    ):
        raise ValueError("blank supplement does not bind the registered source and encoder")
    implementation = receipt.get("implementation")
    runtime = receipt.get("runtime")
    parity = receipt.get("batch_parity")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "builder_sha256",
        "receipt_validator_sha256",
        "encoder_adapter_sha256",
        "encoder_base_sha256",
    }:
        raise ValueError("blank receipt implementation identity is incomplete")
    for name, value in implementation.items():
        _sha256_text(value, f"implementation.{name}")
    if expected_implementation is not None and dict(implementation) != dict(
        expected_implementation
    ):
        raise ValueError("blank receipt implementation differs from the current validator")
    if not isinstance(runtime, Mapping) or runtime.get("git_worktree_dirty_at_start") is not False:
        raise ValueError("blank receipt must record a clean Git worktree at build start")
    if re.fullmatch(r"[0-9a-f]{40}", str(runtime.get("git_commit", ""))) is None:
        raise ValueError("blank receipt Git commit is invalid")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError("blank receipt lacks passing batch parity")
    if parity.get("batch_sizes") != [1, 32] or parity.get("minimum_required_cosine") != (
        BATCH_PARITY_MINIMUM_COSINE
    ):
        raise ValueError("blank receipt batch-parity contract changed")
    if float(parity.get("minimum_cosine", -np.inf)) < BATCH_PARITY_MINIMUM_COSINE:
        raise ValueError("blank receipt batch parity is below tolerance")
    if parity.get("single_feature_sha256") != supplement["feature_sha256"]:
        raise ValueError("blank receipt batch-one hash differs from the supplement")
    _sha256_text(parity.get("repeated_features_sha256"), "batch_parity repeated hash")
    if (recomputed_single is None) != (recomputed_repeated is None):
        raise ValueError("receipt recomputation requires both batch sizes")
    recomputed = None
    if recomputed_single is not None and recomputed_repeated is not None:
        fresh = batch_parity_receipt(recomputed_single, recomputed_repeated)
        if fresh["single_feature_sha256"] != parity["single_feature_sha256"]:
            raise ValueError("fresh blank encoding differs from the registered batch-one feature")
        recomputed_sha256 = array_sha256(validate_feature(recomputed_single, rows=1))
        if recomputed_sha256 != supplement["feature_sha256"]:
            raise ValueError("fresh blank encoding does not reproduce the registered feature hash")
        recomputed = True
    return {
        "schema": "heir.hest_blank_patch_supplement_validation.v1",
        "status": "passed",
        "source_sha256": source["sha256"],
        "supplement_sha256": supplement["sha256"],
        "feature_sha256": supplement["feature_sha256"],
        "blank_patch_embedding_available": recomputed is True,
        "feature_recomputed_from_registered_pixels": recomputed is True,
        "molecular_outcomes_read": False,
        **NON_AUTHORIZING_FIELDS,
    }


__all__ = [
    "BATCH_PARITY_MINIMUM_COSINE",
    "BROADCAST_RULE",
    "CONTROL_ID",
    "FEATURE_WIDTH",
    "RECEIPT_SCHEMA",
    "SOURCE_MEMBER_ALLOWLIST",
    "SUPPLEMENT_MEMBERS",
    "SUPPLEMENT_SCHEMA",
    "array_sha256",
    "batch_parity_receipt",
    "blank_pixels",
    "load_supplement",
    "make_receipt",
    "source_identity",
    "supplement_arrays",
    "validate_encoder_contract",
    "validate_feature",
    "validate_receipt",
]
