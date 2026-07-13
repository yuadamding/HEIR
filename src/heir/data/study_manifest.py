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
STUDY_MANIFEST_SCHEMA = "heir.study_manifest.v2"
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
    study_stage: str
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
            "study_stage",
            "status",
            "hypothesis_ids",
            "git_commit",
            "analysis_plan_sha256",
            "container_digest",
            "dataset",
            "partitions",
            "observations",
            "candidate_target_gene_panel_sha256",
            "type_marker_panel_sha256",
            "randomization",
            "primary_endpoint",
            "secondary_endpoints",
            "coverage_requirements",
            "decision_thresholds",
            "lock_protection",
            "label_target_independence",
        }
        if not required.issubset(content):
            raise ValueError("study manifest is incomplete")
        study_id = str(content["study_id"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,127}", study_id):
            raise ValueError("study manifest study_id is invalid")
        study_stage = str(content["study_stage"])
        if study_stage not in {"measurement_development", "confirmatory_morphology"}:
            raise ValueError("study manifest stage is unsupported")
        status = str(content["status"])
        if status not in {"draft", "locked", "opened"}:
            raise ValueError("study manifest status is unsupported")
        if require_status is not None and status != require_status:
            raise ValueError("study manifest must have status %s" % require_status)
        hypotheses = _strings(content["hypothesis_ids"], "hypothesis_ids")
        if any(value not in HYPOTHESIS_IDS for value in hypotheses):
            raise ValueError("study manifest has an unknown hypothesis ID")
        if study_stage == "measurement_development" and hypotheses != ("H-MEAS",):
            raise ValueError("measurement-development manifest may authorize only H-MEAS")
        if study_stage == "confirmatory_morphology" and (
            "H-CELL" not in hypotheses or "H-MEAS" in hypotheses
        ):
            raise ValueError(
                "confirmatory morphology must be separate from H-MEAS and authorize H-CELL"
            )
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
            not str(dataset[name]).strip() for name in ("repository", "revision", "source_study")
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
        if (
            set(development) & set(locked)
            or set(development) & set(external)
            or set(locked) & set(external)
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
        _sha256(
            content["candidate_target_gene_panel_sha256"],
            "candidate_target_gene_panel_sha256",
        )
        _sha256(content["type_marker_panel_sha256"], "type_marker_panel_sha256")

        protection = _mapping(
            content["lock_protection"],
            "lock_protection",
            {
                "reserved_exclusively_for",
                "reserved_donor_ids",
                "prior_outcome_access_confirmed_false",
                "hescape_analysis_scope",
                "hescape_allowed_donor_ids",
                "forbidden_prior_outcome_uses",
            },
        )
        reserved = _strings(protection["reserved_donor_ids"], "lock_protection.reserved_donors")
        hescape_allowed = _strings(
            protection["hescape_allowed_donor_ids"],
            "lock_protection.hescape_allowed_donor_ids",
        )
        forbidden_prior = _strings(
            protection["forbidden_prior_outcome_uses"],
            "lock_protection.forbidden_prior_outcome_uses",
        )
        if (
            protection["reserved_exclusively_for"] != "H-CELL"
            or protection["prior_outcome_access_confirmed_false"] is not True
            or protection["hescape_analysis_scope"] != "development_donors_only_hest_lock_unopened"
            or set(reserved) != set(locked)
            or set(hescape_allowed) != set(development)
            or "HESCAPE_locked_regional_outcomes" not in forbidden_prior
        ):
            raise ValueError("study manifest does not protect the HEST locked donors")

        independence = _mapping(
            content["label_target_independence"],
            "label_target_independence",
            {
                "strategy",
                "marker_panel_sha256",
                "candidate_marker_target_overlap_count",
                "establishes_full_target_independence",
                "limitation",
            },
        )
        if (
            _sha256(
                independence["marker_panel_sha256"],
                "label_target_independence.marker_panel_sha256",
            )
            != content["type_marker_panel_sha256"]
            or independence["candidate_marker_target_overlap_count"] != 0
            or not str(independence["strategy"]).strip()
            or not str(independence["limitation"]).strip()
            or not isinstance(independence["establishes_full_target_independence"], bool)
        ):
            raise ValueError("study manifest label-target independence contract is malformed")

        if study_stage == "measurement_development":
            morphology_only = {
                "prerequisites",
                "encoder",
                "crop_protocols",
                "target_gene_panel_sha256",
                "technical_covariates",
                "controls",
                "hyperparameter_grid",
                "morphology_gate",
                "reference_splits",
            }
            present = sorted(morphology_only & set(content))
            if present:
                raise ValueError(
                    "measurement-development manifest contains morphology-only fields: %s"
                    % ", ".join(present)
                )
            measurement_randomization = _mapping(
                content["randomization"],
                "randomization",
                {"transcript_split_salt", "donor_cross_fit_seed", "selection_partition"},
            )
            if measurement_randomization["selection_partition"] != "development_only":
                raise ValueError(
                    "measurement-development target selection must use development donors only"
                )
            measurement_coverage = _mapping(
                content["coverage_requirements"],
                "coverage_requirements",
                {
                    "minimum_development_donors_per_fine_type",
                    "minimum_locked_donors_per_fine_type",
                },
            )
            locked_minimum = measurement_coverage["minimum_locked_donors_per_fine_type"]
            if (
                isinstance(locked_minimum, bool)
                or int(locked_minimum) != locked_minimum
                or int(locked_minimum) != 0
            ):
                raise ValueError(
                    "measurement-development locked donor coverage minimum must be zero"
                )
        else:
            morphology_required = {
                "prerequisites",
                "encoder",
                "crop_protocols",
                "target_gene_panel_sha256",
                "technical_covariates",
                "controls",
                "hyperparameter_grid",
                "morphology_gate",
                "reference_splits",
            }
            if not morphology_required.issubset(content):
                raise ValueError("confirmatory morphology manifest is incomplete")
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
            reference_splits = _mapping(
                content["reference_splits"],
                "reference_splits",
                {"primary_split_id", "split_ids"},
            )
            reference_split_ids = _strings(
                reference_splits["split_ids"], "reference_splits.split_ids"
            )
            if (
                len(reference_split_ids) < 3
                or reference_splits["primary_split_id"] != reference_split_ids[0]
            ):
                raise ValueError(
                    "confirmatory morphology requires a primary and two frozen reference splits"
                )
            _strings(content["technical_covariates"], "technical_covariates", allow_empty=True)
            _strings(content["controls"], "controls")
            prerequisites = _mapping(
                content["prerequisites"],
                "prerequisites",
                {
                    "measurement_report_sha256",
                    "measurement_study_manifest_sha256",
                    "measurement_source_sha256",
                },
            )
            prerequisite_values = tuple(
                prerequisites[name]
                for name in (
                    "measurement_report_sha256",
                    "measurement_study_manifest_sha256",
                    "measurement_source_sha256",
                )
            )
            if status == "draft":
                if any(value is not None for value in prerequisite_values) and not all(
                    value is not None for value in prerequisite_values
                ):
                    raise ValueError("draft morphology prerequisites must be all bound or all null")
                for index, value in enumerate(prerequisite_values):
                    if value is not None:
                        _sha256(value, "prerequisites[%d]" % index)
            else:
                for index, value in enumerate(prerequisite_values):
                    _sha256(value, "prerequisites[%d]" % index)

            selected_panel = content["target_gene_panel_sha256"]
            supported_types = observations.get("supported_fine_type_ids")
            supported_types_sha = observations.get("supported_fine_type_ids_sha256")
            if status == "draft" and selected_panel is None:
                if supported_types not in (None, []) or supported_types_sha is not None:
                    raise ValueError(
                        "draft H-CELL target and fine-type bindings must resolve together"
                    )
            else:
                _sha256(selected_panel, "target_gene_panel_sha256")
                selected_types = _strings(supported_types, "observations.supported_fine_type_ids")
                expected_type_hash = hashlib.sha256(
                    json.dumps(
                        list(selected_types), separators=(",", ":"), ensure_ascii=True
                    ).encode("utf-8")
                ).hexdigest()
                if (
                    _sha256(supported_types_sha, "observations.supported_fine_type_ids_sha256")
                    != expected_type_hash
                ):
                    raise ValueError("supported fine-type IDs differ from their frozen hash")

            morphology_gate = _mapping(
                content["morphology_gate"],
                "morphology_gate",
                {
                    "experiment_role",
                    "scientific_scope",
                    "final_inference",
                    "calibration_receipt_sha256",
                    "minimum_final_permutations",
                    "minimum_coordinate_delta",
                    "minimum_stain_delta",
                    "minimum_null_shuffled_fraction",
                    "minimum_strata_coverage",
                    "minimum_expression_error_reduction",
                    "minimum_basis_ceiling_r2",
                    "maximum_direct_contrast_p",
                    "minimum_mask_implementation_pass_fraction",
                    "donor_bootstrap_iterations",
                    "donor_bootstrap_seed",
                    "prespecified_fixed_hyperparameters",
                },
            )
            if not isinstance(morphology_gate["final_inference"], bool):
                raise ValueError("morphology_gate.final_inference must be boolean")
            if not str(morphology_gate["experiment_role"]).strip():
                raise ValueError("morphology_gate.experiment_role is empty")
            if morphology_gate["scientific_scope"] != "registered_cell_local_context_association":
                raise ValueError("confirmatory H-CELL scope must remain a local-context claim")
            for name in (
                "minimum_final_permutations",
                "donor_bootstrap_iterations",
            ):
                value = morphology_gate[name]
                if isinstance(value, bool) or int(value) != value or int(value) < 1:
                    raise ValueError("morphology_gate.%s must be a positive integer" % name)
            seed_value = morphology_gate["donor_bootstrap_seed"]
            if isinstance(seed_value, bool) or int(seed_value) != seed_value:
                raise ValueError("morphology_gate.donor_bootstrap_seed must be an integer")
            for name in (
                "minimum_coordinate_delta",
                "minimum_stain_delta",
                "minimum_null_shuffled_fraction",
                "minimum_strata_coverage",
                "minimum_expression_error_reduction",
                "minimum_basis_ceiling_r2",
                "minimum_mask_implementation_pass_fraction",
            ):
                value = float(morphology_gate[name])
                if not 0 <= value <= 1:
                    raise ValueError("morphology_gate.%s must be in [0, 1]" % name)
            direct_contrast_p = float(morphology_gate["maximum_direct_contrast_p"])
            if not 0 < direct_contrast_p <= 1:
                raise ValueError("morphology_gate.maximum_direct_contrast_p must be in (0, 1]")
            if not isinstance(morphology_gate["prespecified_fixed_hyperparameters"], bool):
                raise ValueError(
                    "morphology_gate.prespecified_fixed_hyperparameters must be boolean"
                )
            coverage_balance = content["coverage_requirements"].get(
                "maximum_reference_evaluation_absolute_smd"
            )
            categorical_balance = content["coverage_requirements"].get(
                "maximum_reference_evaluation_categorical_total_variation"
            )
            if (
                coverage_balance is None
                or not 0 < float(coverage_balance) <= 1
                or categorical_balance is None
                or not 0 < float(categorical_balance) <= 1
            ):
                raise ValueError(
                    "confirmatory morphology requires continuous and categorical balance thresholds"
                )
            calibration_sha = morphology_gate["calibration_receipt_sha256"]
            if morphology_gate["final_inference"]:
                if status == "draft" and calibration_sha is None:
                    pass
                else:
                    _sha256(
                        calibration_sha,
                        "morphology_gate.calibration_receipt_sha256",
                    )
            elif calibration_sha is not None:
                raise ValueError("exploratory morphology cannot bind a calibration receipt")
            if status in {"locked", "opened"} and (
                independence["establishes_full_target_independence"] is not True
            ):
                raise ValueError(
                    "confirmatory H-CELL requires proven gene-disjoint label-target independence"
                )

        for name in (
            "randomization",
            "primary_endpoint",
            "coverage_requirements",
            "decision_thresholds",
        ):
            if not isinstance(content[name], Mapping) or not content[name]:
                raise ValueError("study manifest %s is empty" % name)
        if study_stage == "confirmatory_morphology" and (
            not isinstance(content["hyperparameter_grid"], Mapping)
            or not content["hyperparameter_grid"]
        ):
            raise ValueError("study manifest hyperparameter_grid is empty")
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
            study_stage=study_stage,
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
                "locked study prohibits CLI scientific overrides: %s" % ", ".join(sorted(supplied))
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
        raise ValueError("only a v2 draft study manifest can be frozen")
    if draft.get("study_stage") == "confirmatory_morphology":
        prerequisites = _mapping(
            draft.get("prerequisites"),
            "prerequisites",
            {
                "measurement_report_sha256",
                "measurement_study_manifest_sha256",
                "measurement_source_sha256",
            },
        )
        for name in (
            "measurement_report_sha256",
            "measurement_study_manifest_sha256",
            "measurement_source_sha256",
        ):
            _sha256(prerequisites[name], "prerequisites.%s" % name)
        _sha256(draft.get("target_gene_panel_sha256"), "target_gene_panel_sha256")
        observations = _mapping(
            draft.get("observations"),
            "observations",
            {"supported_fine_type_ids", "supported_fine_type_ids_sha256"},
        )
        selected_types = _strings(
            observations["supported_fine_type_ids"],
            "observations.supported_fine_type_ids",
        )
        expected_type_hash = hashlib.sha256(
            json.dumps(list(selected_types), separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            _sha256(
                observations["supported_fine_type_ids_sha256"],
                "observations.supported_fine_type_ids_sha256",
            )
            != expected_type_hash
        ):
            raise ValueError("supported fine-type IDs differ from their frozen hash")
        independence = _mapping(
            draft.get("label_target_independence"),
            "label_target_independence",
            {"establishes_full_target_independence"},
        )
        if independence["establishes_full_target_independence"] is not True:
            raise ValueError(
                "confirmatory H-CELL cannot lock before label-target independence is proven"
            )
        morphology_gate = _mapping(
            draft.get("morphology_gate"),
            "morphology_gate",
            {"final_inference", "calibration_receipt_sha256"},
        )
        if morphology_gate["final_inference"] is True:
            _sha256(
                morphology_gate["calibration_receipt_sha256"],
                "morphology_gate.calibration_receipt_sha256",
            )
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
