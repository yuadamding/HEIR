"""Locked study manifests that prohibit post-lock scientific overrides."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from heir.utils import sha256_file

PathLike = Union[str, Path]
STUDY_MANIFEST_SCHEMA = "heir.study_manifest.v1"
# Keys excluded from the tamper-evident content digest: the digest field itself, the
# one-way opening receipt, and status (which changes locked->opened while the frozen
# scientific content must stay bit-for-bit identical).
_CONTENT_DIGEST_EXCLUDED_KEYS = ("locked_content_sha256", "opening", "status")
HYPOTHESIS_IDS = {
    "H-MEAS",
    "H-REGIONAL",
    "H-CELL",
    "H-INTRINSIC",
    "H-REF",
    "H-END2END",
    "H-COMP",
    "H-EXT",
}


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("study manifest %s must be a lowercase SHA-256" % name)
    return digest


def _content_digest(content: Mapping[str, object]) -> str:
    """Hash the frozen scientific content, ignoring the digest field, opening receipt, and status.

    This binds every locked scientific field (donors, source and panel hashes, thresholds,
    git commit, container digest, ...) so that editing a locked manifest after freezing is
    detectable even while checked out at the locked commit.
    """

    payload = {
        key: value for key, value in content.items() if key not in _CONTENT_DIGEST_EXCLUDED_KEYS
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not fields.issubset(value):
        raise ValueError("study manifest %s is incomplete" % name)
    return value


def _strings(value: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("study manifest %s must be a list" % name)
    result = tuple(str(item) for item in value)
    if (not result and not allow_empty) or any(not item.strip() for item in result):
        raise ValueError("study manifest %s contains empty values" % name)
    if len(set(result)) != len(result):
        raise ValueError("study manifest %s contains duplicates" % name)
    return result


def current_git_commit(root: PathLike) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root).expanduser().resolve(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot resolve the running Git commit") from error
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("running Git commit is malformed")
    return value


def require_clean_worktree(root: PathLike) -> None:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=Path(root).expanduser().resolve(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot inspect the Git worktree") from error
    if status.strip():
        raise ValueError("a study cannot be locked from a dirty worktree")


@dataclass(frozen=True)
class StudyManifest:
    """A validated draft, locked, or opened study contract."""

    path: Path
    sha256: str
    content: Mapping[str, object]
    study_id: str
    status: str
    hypothesis_ids: tuple[str, ...]
    development_donors: tuple[str, ...]
    locked_test_donors: tuple[str, ...]
    external_test_donors: tuple[str, ...]

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        require_status: Optional[str] = None,
        verify_runtime: bool = False,
        repository_root: Optional[PathLike] = None,
    ) -> "StudyManifest":
        resolved = Path(path).expanduser().resolve()
        try:
            content = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("study manifest is not valid JSON") from error
        if not isinstance(content, Mapping) or content.get("schema") != STUDY_MANIFEST_SCHEMA:
            raise ValueError("study manifest schema is unsupported")
        required = {
            "schema",
            "study_id",
            "status",
            "hypothesis_ids",
            "git_commit",
            "analysis_plan_sha256",
            "container_digest",
            "dataset",
            "partitions",
            "observations",
            "encoder",
            "crop_protocols",
            "target_gene_panel_sha256",
            "type_marker_panel_sha256",
            "technical_covariates",
            "controls",
            "hyperparameter_grid",
            "randomization",
            "primary_endpoint",
            "secondary_endpoints",
            "coverage_requirements",
            "decision_thresholds",
        }
        if not required.issubset(content):
            raise ValueError("study manifest is incomplete")
        study_id = str(content["study_id"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,127}", study_id):
            raise ValueError("study manifest study_id is invalid")
        status = str(content["status"])
        if status not in {"draft", "locked", "opened"}:
            raise ValueError("study manifest status is unsupported")
        if require_status is not None and status != require_status:
            raise ValueError("study manifest must have status %s" % require_status)
        hypotheses = _strings(content["hypothesis_ids"], "hypothesis_ids")
        if any(value not in HYPOTHESIS_IDS for value in hypotheses):
            raise ValueError("study manifest has an unknown hypothesis ID")
        commit = str(content["git_commit"])
        if status in {"locked", "opened"} and not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("locked study manifest Git commit is invalid")
        _sha256(content["analysis_plan_sha256"], "analysis_plan_sha256")
        container = str(content["container_digest"])
        if status in {"locked", "opened"} and not re.fullmatch(r"sha256:[0-9a-f]{64}", container):
            raise ValueError("locked study manifest container digest is invalid")

        dataset = _mapping(
            content["dataset"],
            "dataset",
            {"repository", "revision", "source_study", "source_manifest_sha256"},
        )
        if any(
            not str(dataset[name]).strip()
            for name in ("repository", "revision", "source_study")
        ):
            raise ValueError("study manifest dataset identity is empty")
        _sha256(dataset["source_manifest_sha256"], "dataset.source_manifest_sha256")
        partitions = _mapping(
            content["partitions"],
            "partitions",
            {
                "development_donors",
                "locked_test_donors",
                "external_test_donors",
                "split_manifest_sha256",
            },
        )
        development = _strings(partitions["development_donors"], "development_donors")
        locked = _strings(partitions["locked_test_donors"], "locked_test_donors")
        external = _strings(
            partitions["external_test_donors"], "external_test_donors", allow_empty=True
        )
        if set(development) & set(locked) or set(development) & set(external) or set(locked) & set(
            external
        ):
            raise ValueError("study manifest donor partitions overlap")
        _sha256(partitions["split_manifest_sha256"], "partitions.split_manifest_sha256")
        observations = _mapping(
            content["observations"],
            "observations",
            {
                "level",
                "registration_method",
                "target_variants",
                "broad_type_field",
                "fine_type_field",
            },
        )
        _strings(observations["target_variants"], "observations.target_variants")
        if any(
            not str(observations[name]).strip()
            for name in ("level", "registration_method", "broad_type_field", "fine_type_field")
        ):
            raise ValueError("study manifest observation identity is empty")
        encoder = _mapping(
            content["encoder"],
            "encoder",
            {"manifest_sha256", "feature_space_id", "checkpoint_sha256"},
        )
        _sha256(encoder["manifest_sha256"], "encoder.manifest_sha256")
        _sha256(encoder["checkpoint_sha256"], "encoder.checkpoint_sha256")
        if not str(encoder["feature_space_id"]).strip():
            raise ValueError("study manifest feature space is empty")
        crop_protocols = content["crop_protocols"]
        if not isinstance(crop_protocols, list) or not crop_protocols:
            raise ValueError("study manifest crop protocols are missing")
        for crop in crop_protocols:
            _sha256(crop, "crop_protocols[]")
        _sha256(content["target_gene_panel_sha256"], "target_gene_panel_sha256")
        _sha256(content["type_marker_panel_sha256"], "type_marker_panel_sha256")
        _strings(content["technical_covariates"], "technical_covariates", allow_empty=True)
        _strings(content["controls"], "controls")
        for name in (
            "hyperparameter_grid",
            "randomization",
            "primary_endpoint",
            "coverage_requirements",
            "decision_thresholds",
        ):
            if not isinstance(content[name], Mapping) or not content[name]:
                raise ValueError("study manifest %s is empty" % name)
        if not isinstance(content["secondary_endpoints"], list):
            raise ValueError("study manifest secondary endpoints must be a list")
        if status in {"locked", "opened"}:
            locked_at = str(content.get("locked_at", ""))
            if not locked_at:
                raise ValueError("locked study manifest lacks locked_at")
            recorded_digest = _sha256(
                content.get("locked_content_sha256", ""), "locked_content_sha256"
            )
            if recorded_digest != _content_digest(content):
                raise ValueError("locked study manifest content was modified after locking")
        if status == "opened":
            opening = _mapping(
                content.get("opening"),
                "opening",
                {"locked_manifest_sha256", "opened_by_commit", "opened_at", "permitted_claims"},
            )
            _sha256(opening["locked_manifest_sha256"], "opening.locked_manifest_sha256")
            if not re.fullmatch(r"[0-9a-f]{40}", str(opening["opened_by_commit"])):
                raise ValueError("opened study commit is invalid")
            _strings(opening["permitted_claims"], "opening.permitted_claims", allow_empty=True)
            if opening.get("adoption_for_future_models") is not False:
                raise ValueError("opened locked evidence cannot become future development data")
        if verify_runtime:
            root = Path(repository_root or resolved.parent).expanduser().resolve()
            if current_git_commit(root) != commit:
                raise ValueError("running commit differs from the locked study manifest")
        return cls(
            path=resolved,
            sha256=sha256_file(resolved),
            content=content,
            study_id=study_id,
            status=status,
            hypothesis_ids=hypotheses,
            development_donors=development,
            locked_test_donors=locked,
            external_test_donors=external,
        )

    def reject_cli_overrides(self, overrides: Mapping[str, object]) -> None:
        """Locked scientific parameters may only come from this manifest."""

        if self.status not in {"locked", "opened"}:
            raise ValueError("only a locked study can authorize a benchmark")
        supplied = {name: value for name, value in overrides.items() if value is not None}
        if supplied:
            raise ValueError(
                "locked study prohibits CLI scientific overrides: %s"
                % ", ".join(sorted(supplied))
            )


def freeze_manifest_content(
    draft: Mapping[str, object],
    *,
    git_commit: str,
    container_digest: str,
    locked_at: Optional[str] = None,
) -> Mapping[str, object]:
    """Create locked content without mutating a caller's draft mapping."""

    if draft.get("schema") != STUDY_MANIFEST_SCHEMA or draft.get("status") != "draft":
        raise ValueError("only a v1 draft study manifest can be frozen")
    value = json.loads(json.dumps(draft))
    value.pop("locked_content_sha256", None)
    value.pop("opening", None)
    value["status"] = "locked"
    value["git_commit"] = git_commit
    value["container_digest"] = container_digest
    value["locked_at"] = locked_at or datetime.now(timezone.utc).isoformat()
    value["locked_content_sha256"] = _content_digest(value)
    return value


def open_manifest_content(
    locked: StudyManifest,
    *,
    opened_by_commit: str,
    permitted_claims: Sequence[str],
    opened_at: Optional[str] = None,
) -> Mapping[str, object]:
    """Record the one-way locked-to-opened transition with the locked receipt."""

    if locked.status != "locked":
        raise ValueError("only a locked study may be opened")
    value = json.loads(json.dumps(locked.content))
    value["status"] = "opened"
    value["opening"] = {
        "locked_manifest_sha256": locked.sha256,
        "opened_by_commit": opened_by_commit,
        "opened_at": opened_at or datetime.now(timezone.utc).isoformat(),
        "permitted_claims": list(permitted_claims),
        "adoption_for_future_models": False,
    }
    return value


__all__ = [
    "STUDY_MANIFEST_SCHEMA",
    "StudyManifest",
    "current_git_commit",
    "freeze_manifest_content",
    "open_manifest_content",
    "require_clean_worktree",
]
