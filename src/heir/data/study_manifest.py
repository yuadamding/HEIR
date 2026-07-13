"""Locked study manifests that prohibit post-lock scientific overrides."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
_OPENING_RECEIPT_FIELDS = {
    "locked_manifest_sha256",
    "locked_content_sha256",
    "opened_by_commit",
    "opened_at",
    "permitted_claims",
    "adoption_for_future_models",
    "opening_receipt_sha256",
}
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

# These GSE250346 outcomes were materialized before the current prospective
# protocol was registered.  A manifest declaration cannot make that exposure
# disappear; the immutable receipt records the evidence supporting this list.
KNOWN_GSE250346_PRE_REGISTRATION_EXPOSED_DONORS = frozenset(
    {"THD0008", "THD0011", "TILD117", "VUILD78", "VUILD96"}
)
GSE250346_LOCK_EXPOSURE_RECEIPT_SHA256 = (
    "7c9dd840968ec45f2bce4b2e9f6ae315688834da9a17acb94daf24060e24804b"
)

LABEL_TARGET_INDEPENDENCE_FIELDS = {
    "strategy",
    "evidence_kind",
    "annotation_receipt_sha256",
    "ordered_annotation_feature_ids",
    "ordered_annotation_feature_ids_sha256",
    "ordered_target_gene_ids",
    "ordered_target_gene_ids_sha256",
    "annotation_target_overlap_count",
    "annotation_training_scope",
    "annotation_training_donor_ids",
    "annotation_training_donor_ids_sha256",
    "training_label_ontology_source",
    "training_label_provenance_receipt_sha256",
    "training_label_target_gene_overlap_count",
    "training_labels_establish_target_independence",
    "annotation_quality_contract",
    "locked_donors_used_for_training",
    "same_cohort_annotation",
    "cross_fitting_method",
    "cross_fitting_receipt_sha256",
    "establishes_full_target_independence",
    "limitation",
}
LABEL_TARGET_INDEPENDENCE_PROTOCOL_FIELDS = LABEL_TARGET_INDEPENDENCE_FIELDS - {
    "ordered_target_gene_ids",
    "ordered_target_gene_ids_sha256",
    "annotation_target_overlap_count",
}
LABEL_TARGET_INDEPENDENCE_EVIDENCE_KINDS = {
    "external_gene_disjoint_annotation",
    "development_donor_cross_fitted_gene_disjoint_annotation",
    "orthogonal_modality_annotation",
}
ANNOTATION_QUALITY_CONTRACT_FIELDS = {
    "minimum_fine_type_macro_f1",
    "calibration_metric",
    "calibration_binning",
    "calibration_bin_count",
    "maximum_calibration_error",
    "minimum_major_class_sensitivity",
    "minimum_prediction_coverage_fraction",
    "abstention_method",
    "confidence_threshold",
    "abstain_label",
    "tie_breaking",
    "probability_sum_tolerance",
    "per_validation_donor_metrics_required",
    "validation_designs_by_evidence_kind",
}
DEFAULT_ANNOTATION_QUALITY_CONTRACT = {
    "minimum_fine_type_macro_f1": 0.70,
    "calibration_metric": "multiclass_expected_calibration_error",
    "calibration_binning": "equal_width_confidence_bins",
    "calibration_bin_count": 10,
    "maximum_calibration_error": 0.10,
    "minimum_major_class_sensitivity": 0.50,
    "minimum_prediction_coverage_fraction": 0.80,
    "abstention_method": "maximum_probability_below_threshold",
    "confidence_threshold": 0.50,
    "abstain_label": "__ABSTAIN__",
    "tie_breaking": "ordered_fine_type_ids",
    "probability_sum_tolerance": 1.0e-6,
    "per_validation_donor_metrics_required": True,
    "validation_designs_by_evidence_kind": {
        "external_gene_disjoint_annotation": "external_donor_held_out",
        "development_donor_cross_fitted_gene_disjoint_annotation": (
            "leave_one_development_donor_out"
        ),
        "orthogonal_modality_annotation": "orthogonal_modality_validation",
    },
}

# H-MEAS and the post-opening H-CELL audit must use the same scientific
# measurement contract.  These are deliberately exact rather than merely
# range-checked: changing any value requires a new study/protocol version.
FROZEN_H_MEAS_SHARED_MEASUREMENT_THRESHOLDS = {
    "maximum_annotation_nucleus_p95_um": 8.0,
    "maximum_annotation_cell_p95_um": 12.0,
    "maximum_cell_nucleus_p95_um": 8.0,
    "maximum_registration_nucleus_diameter_ratio_p95": 0.5,
    "maximum_registration_nearest_neighbor_ratio_p95": 0.5,
    "best_registration_quality_max_fraction_of_limit": 0.25,
    "intermediate_registration_quality_max_fraction_of_limit": 0.6,
    "maximum_registration_outlier_fraction": 0.05,
    "maximum_nucleus_outside_cell_fraction": 0.01,
    "minimum_nucleus_cell_area_ratio": 0.05,
    "maximum_nucleus_cell_area_ratio": 0.95,
    "maximum_segmentation_outlier_fraction": 0.05,
    "maximum_crop_padding_p95": 0.25,
    "mostly_padded_cutoff": 0.5,
    "maximum_mostly_padded_fraction": 0.01,
    "minimum_within_fine_type_reliability": 0.4,
    "minimum_reliability_rows": 40,
}
CONFIRMATORY_PRIMARY_ENDPOINT_NAME = (
    "joint_donor_type_and_donor_section_type_macro_residual_coordinate_r2"
)
CONFIRMATORY_DONOR_TYPE_METRIC = "donor_equal_type_equal_residual_coordinate_r2"
CONFIRMATORY_DONOR_SECTION_TYPE_METRIC = (
    "donor_equal_section_equal_type_equal_residual_coordinate_r2"
)
CONFIRMATORY_PRIMARY_MINIMUM_EFFECT = 0.05
CONFIRMATORY_PRIMARY_DECISION_RULE = "both_endpoints_must_meet_frozen_minimum"


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


def _opening_receipt_digest(opening: Mapping[str, object]) -> str:
    """Hash every opening field while excluding only the digest itself."""

    payload = {key: value for key, value in opening.items() if key != "opening_receipt_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not fields.issubset(value):
        raise ValueError("study manifest %s is incomplete" % name)
    return value


def _validate_frozen_h_meas_shared_thresholds(value: Mapping[str, object], name: str) -> None:
    for field, expected in FROZEN_H_MEAS_SHARED_MEASUREMENT_THRESHOLDS.items():
        observed = value.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or observed != expected
        ):
            raise ValueError(
                "study manifest %s.%s must equal the frozen H-MEAS value %s"
                % (name, field, expected)
            )


def _validate_confirmatory_primary_endpoint(
    value: object, fine_type_field: object
) -> Mapping[str, object]:
    fields = {
        "name",
        "condition_on",
        "decision_rule",
        "donor_type_macro",
        "donor_section_type_macro",
    }
    endpoint = _mapping(value, "primary_endpoint", fields)
    if set(endpoint) != fields:
        raise ValueError("study manifest confirmatory primary_endpoint has extra fields")
    donor_type = _mapping(
        endpoint["donor_type_macro"],
        "primary_endpoint.donor_type_macro",
        {"metric", "minimum_effect"},
    )
    donor_section_type = _mapping(
        endpoint["donor_section_type_macro"],
        "primary_endpoint.donor_section_type_macro",
        {"metric", "minimum_effect"},
    )
    if set(donor_type) != {"metric", "minimum_effect"} or set(donor_section_type) != {
        "metric",
        "minimum_effect",
    }:
        raise ValueError("study manifest confirmatory primary endpoint has extra fields")
    for name, observed in (
        ("donor_type_macro", donor_type["minimum_effect"]),
        ("donor_section_type_macro", donor_section_type["minimum_effect"]),
    ):
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) != CONFIRMATORY_PRIMARY_MINIMUM_EFFECT
        ):
            raise ValueError(
                "study manifest primary_endpoint.%s.minimum_effect must equal %.2f"
                % (name, CONFIRMATORY_PRIMARY_MINIMUM_EFFECT)
            )
    if (
        endpoint["name"] != CONFIRMATORY_PRIMARY_ENDPOINT_NAME
        or endpoint["condition_on"] != fine_type_field
        or endpoint["decision_rule"] != CONFIRMATORY_PRIMARY_DECISION_RULE
        or donor_type["metric"] != CONFIRMATORY_DONOR_TYPE_METRIC
        or donor_section_type["metric"] != CONFIRMATORY_DONOR_SECTION_TYPE_METRIC
    ):
        raise ValueError(
            "study manifest confirmatory primary_endpoint differs from the frozen joint endpoint"
        )
    return endpoint


def _strings(value: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("study manifest %s must be a list" % name)
    result = tuple(str(item) for item in value)
    if (not result and not allow_empty) or any(not item.strip() for item in result):
        raise ValueError("study manifest %s contains empty values" % name)
    if len(set(result)) != len(result):
        raise ValueError("study manifest %s contains duplicates" % name)
    return result


def _ordered_ids_sha256(values: Sequence[object]) -> str:
    encoded = json.dumps(
        [str(value) for value in values],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _optional_sha256(value: object, name: str) -> Optional[str]:
    return None if value is None else _sha256(value, name)


def _validate_annotation_quality_contract(value: object) -> Mapping[str, object]:
    quality = _mapping(
        value,
        "label_target_independence.annotation_quality_contract",
        ANNOTATION_QUALITY_CONTRACT_FIELDS,
    )
    if set(quality) != ANNOTATION_QUALITY_CONTRACT_FIELDS:
        raise ValueError("study manifest annotation quality contract has extra fields")

    def number(name: str) -> float:
        raw = quality[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("study manifest annotation quality %s must be numeric" % name)
        result = float(raw)
        if not math.isfinite(result):
            raise ValueError("study manifest annotation quality %s must be finite" % name)
        return result

    macro_f1 = number("minimum_fine_type_macro_f1")
    calibration_error = number("maximum_calibration_error")
    sensitivity = number("minimum_major_class_sensitivity")
    coverage = number("minimum_prediction_coverage_fraction")
    confidence = number("confidence_threshold")
    tolerance = number("probability_sum_tolerance")
    designs = quality["validation_designs_by_evidence_kind"]
    if (
        not 0.70 <= macro_f1 <= 1.0
        or not 0.0 <= calibration_error <= 0.10
        or not 0.50 <= sensitivity <= 1.0
        or not 0.80 <= coverage <= 1.0
        or not 0.50 <= confidence < 1.0
        or not 0.0 < tolerance <= 1.0e-6
        or quality["calibration_metric"] != "multiclass_expected_calibration_error"
        or quality["calibration_binning"] != "equal_width_confidence_bins"
        or isinstance(quality["calibration_bin_count"], bool)
        or not isinstance(quality["calibration_bin_count"], int)
        or quality["calibration_bin_count"] != 10
        or quality["abstention_method"] != "maximum_probability_below_threshold"
        or quality["abstain_label"] != "__ABSTAIN__"
        or quality["tie_breaking"] != "ordered_fine_type_ids"
        or quality["per_validation_donor_metrics_required"] is not True
        or not isinstance(designs, Mapping)
        or dict(designs)
        != {
            "external_gene_disjoint_annotation": "external_donor_held_out",
            "development_donor_cross_fitted_gene_disjoint_annotation": (
                "leave_one_development_donor_out"
            ),
            "orthogonal_modality_annotation": "orthogonal_modality_validation",
        }
    ):
        raise ValueError("study manifest annotation quality contract weakens frozen guardrails")
    return quality


def _validate_label_target_independence(
    content: Mapping[str, object],
) -> Mapping[str, object]:
    independence = _mapping(
        content.get("label_target_independence"),
        "label_target_independence",
        LABEL_TARGET_INDEPENDENCE_FIELDS,
    )
    if set(independence) != LABEL_TARGET_INDEPENDENCE_FIELDS:
        raise ValueError("study manifest label-target independence contract has extra fields")
    if not str(independence["strategy"]).strip() or not str(independence["limitation"]).strip():
        raise ValueError("study manifest label-target independence strategy is empty")
    _validate_annotation_quality_contract(independence["annotation_quality_contract"])

    annotation_ids = _strings(
        independence["ordered_annotation_feature_ids"],
        "label_target_independence.ordered_annotation_feature_ids",
        allow_empty=True,
    )
    target_ids = _strings(
        independence["ordered_target_gene_ids"],
        "label_target_independence.ordered_target_gene_ids",
        allow_empty=True,
    )
    training_donor_ids = _strings(
        independence["annotation_training_donor_ids"],
        "label_target_independence.annotation_training_donor_ids",
        allow_empty=True,
    )
    annotation_hash = _optional_sha256(
        independence["ordered_annotation_feature_ids_sha256"],
        "label_target_independence.ordered_annotation_feature_ids_sha256",
    )
    target_hash = _optional_sha256(
        independence["ordered_target_gene_ids_sha256"],
        "label_target_independence.ordered_target_gene_ids_sha256",
    )
    training_donor_hash = _optional_sha256(
        independence["annotation_training_donor_ids_sha256"],
        "label_target_independence.annotation_training_donor_ids_sha256",
    )
    annotation_receipt = _optional_sha256(
        independence["annotation_receipt_sha256"],
        "label_target_independence.annotation_receipt_sha256",
    )
    cross_fit_receipt = _optional_sha256(
        independence["cross_fitting_receipt_sha256"],
        "label_target_independence.cross_fitting_receipt_sha256",
    )
    training_label_receipt = _optional_sha256(
        independence["training_label_provenance_receipt_sha256"],
        "label_target_independence.training_label_provenance_receipt_sha256",
    )
    evidence_kind = str(independence["evidence_kind"])
    same_cohort = independence["same_cohort_annotation"]
    establishes = independence["establishes_full_target_independence"]
    training_labels_independent = independence["training_labels_establish_target_independence"]
    if (
        not isinstance(same_cohort, bool)
        or not isinstance(establishes, bool)
        or not isinstance(training_labels_independent, bool)
    ):
        raise ValueError("study manifest label-target independence flags must be boolean")

    if evidence_kind == "pending":
        if (
            annotation_receipt is not None
            or annotation_ids
            or annotation_hash is not None
            or target_ids
            or target_hash is not None
            or independence["annotation_target_overlap_count"] is not None
            or independence["annotation_training_scope"] != "unknown_pending_provenance"
            or training_donor_ids
            or training_donor_hash is not None
            or independence["training_label_ontology_source"] != "pending"
            or training_label_receipt is not None
            or independence["training_label_target_gene_overlap_count"] is not None
            or training_labels_independent is not False
            or independence["locked_donors_used_for_training"] is not None
            or independence["cross_fitting_method"] != "pending"
            or cross_fit_receipt is not None
            or establishes is not False
        ):
            raise ValueError("pending label-target independence must not claim resolved evidence")
        return independence

    if evidence_kind not in LABEL_TARGET_INDEPENDENCE_EVIDENCE_KINDS:
        raise ValueError("study manifest label-target independence evidence kind is unsupported")
    if annotation_receipt is None or not annotation_ids or not target_ids:
        raise ValueError(
            "proven label-target independence requires a receipt and exact annotation/target IDs"
        )
    if annotation_hash != _ordered_ids_sha256(annotation_ids):
        raise ValueError("ordered annotation feature IDs differ from their frozen hash")
    if target_hash != _ordered_ids_sha256(target_ids):
        raise ValueError("ordered target gene IDs differ from their frozen hash")
    target_panel_field = (
        "candidate_target_gene_panel_sha256"
        if content.get("study_stage") == "measurement_development"
        else "target_gene_panel_sha256"
    )
    if target_hash != content.get(target_panel_field):
        raise ValueError(
            "independence target genes differ from the frozen measurement/confirmatory panel"
        )
    overlap = set(annotation_ids) & set(target_ids)
    annotation_overlap_count = independence["annotation_target_overlap_count"]
    if (
        isinstance(annotation_overlap_count, bool)
        or not isinstance(annotation_overlap_count, int)
        or annotation_overlap_count != 0
        or overlap
    ):
        raise ValueError("annotation features overlap the frozen H-CELL target panel")
    if independence["locked_donors_used_for_training"] is not False:
        raise ValueError("locked donors cannot train the annotation procedure")
    if (
        training_label_receipt is None
        or isinstance(independence["training_label_target_gene_overlap_count"], bool)
        or not isinstance(independence["training_label_target_gene_overlap_count"], int)
        or independence["training_label_target_gene_overlap_count"] != 0
        or training_labels_independent is not True
    ):
        raise ValueError("annotation training-label ontology is not proven target-independent")
    partitions = content.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("label-target independence requires frozen donor partitions")
    development_donors = set(str(value) for value in partitions.get("development_donors", ()))
    locked_donors = set(str(value) for value in partitions.get("locked_test_donors", ()))
    if set(training_donor_ids) & locked_donors:
        raise ValueError("annotation training donor IDs include locked donors")
    if establishes is not True:
        raise ValueError("non-pending independence evidence must establish the frozen contract")

    if same_cohort:
        if (
            evidence_kind != "development_donor_cross_fitted_gene_disjoint_annotation"
            or independence["training_label_ontology_source"]
            != "de_novo_gene_disjoint_development_ontology"
            or independence["annotation_training_scope"] != "development_donors_only"
            or not training_donor_ids
            or set(training_donor_ids) != development_donors
            or training_donor_hash != _ordered_ids_sha256(training_donor_ids)
            or independence["cross_fitting_method"] != "leave_one_donor_out"
            or cross_fit_receipt is None
        ):
            raise ValueError("same-cohort annotation requires development-only donor cross-fitting")
    elif evidence_kind == "external_gene_disjoint_annotation":
        if (
            independence["training_label_ontology_source"] != "external_target_disjoint_ontology"
            or independence["annotation_training_scope"] != "external_donors_only"
            or not training_donor_ids
            or set(training_donor_ids) & development_donors
            or training_donor_hash != _ordered_ids_sha256(training_donor_ids)
            or independence["cross_fitting_method"] != "not_applicable"
            or cross_fit_receipt is not None
        ):
            raise ValueError("external annotation has an invalid training contract")
    elif evidence_kind == "orthogonal_modality_annotation":
        if (
            independence["training_label_ontology_source"] != "orthogonal_modality_ontology"
            or independence["annotation_training_scope"] != "orthogonal_no_rna_training"
            or training_donor_ids
            or training_donor_hash is not None
            or independence["cross_fitting_method"] != "not_applicable"
            or cross_fit_receipt is not None
        ):
            raise ValueError("orthogonal annotation has an invalid training contract")
    else:
        raise ValueError(
            "label-target independence evidence kind conflicts with same-cohort annotation scope"
        )
    return independence


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
        raise ValueError("study locking or confirmatory execution requires a clean Git worktree")


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
        require_clean_runtime: bool = False,
        verify_container_digest: bool = False,
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
        if (
            dataset["repository"] == "MahmoodLab/hest"
            and dataset["source_study"] == "GSE250346"
            and (
                observations["broad_type_field"] != "broad_lineage"
                or observations["fine_type_field"] != "fine_type"
            )
        ):
            raise ValueError(
                "HEST must condition on receipt-bound independent broad_lineage/fine_type labels"
            )
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
                "prior_outcome_access_status",
                "prior_outcome_exposure_receipt_sha256",
                "prospective_lock_eligible",
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
        access_confirmed_false = protection["prior_outcome_access_confirmed_false"]
        access_status = protection["prior_outcome_access_status"]
        exposure_receipt = protection["prior_outcome_exposure_receipt_sha256"]
        prospective_lock_eligible = protection["prospective_lock_eligible"]
        if not isinstance(access_confirmed_false, bool) or not isinstance(
            prospective_lock_eligible, bool
        ):
            raise ValueError("study manifest lock-exposure flags must be boolean")
        known_exposed = bool(set(locked) & KNOWN_GSE250346_PRE_REGISTRATION_EXPOSED_DONORS)
        if known_exposed and (
            access_confirmed_false is not False
            or access_status != "molecular_outcomes_materialized_pre_registration"
            or exposure_receipt != GSE250346_LOCK_EXPOSURE_RECEIPT_SHA256
            or prospective_lock_eligible is not False
        ):
            raise ValueError(
                "known GSE250346 donor exposure cannot be reset by a manifest declaration"
            )
        if access_confirmed_false:
            exposure_contract_valid = bool(
                access_status == "unopened"
                and exposure_receipt is None
                and prospective_lock_eligible
                and protection["hescape_analysis_scope"]
                == "development_donors_only_hest_lock_unopened"
            )
        else:
            _sha256(
                exposure_receipt,
                "lock_protection.prior_outcome_exposure_receipt_sha256",
            )
            exposure_contract_valid = bool(
                access_status == "molecular_outcomes_materialized_pre_registration"
                and prospective_lock_eligible is False
                and protection["hescape_analysis_scope"]
                == "development_donors_only_reserved_outcomes_previously_materialized"
            )
        if (
            protection["reserved_exclusively_for"] != "H-CELL"
            or not exposure_contract_valid
            or set(reserved) != set(locked)
            or set(hescape_allowed) != set(development)
            or "HESCAPE_locked_regional_outcomes" not in forbidden_prior
        ):
            raise ValueError("study manifest does not protect the HEST locked donors")
        if (
            study_stage == "confirmatory_morphology"
            and status in {"locked", "opened"}
            and not prospective_lock_eligible
        ):
            raise ValueError(
                "previously materialized donor outcomes cannot become prospective H-CELL evidence"
            )

        independence = _validate_label_target_independence(content)
        if (
            study_stage == "measurement_development"
            and status in {"locked", "opened"}
            and independence["evidence_kind"] == "pending"
        ):
            raise ValueError(
                "H-MEAS cannot lock while fine-type and training-label provenance is pending"
            )

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
                "locked_measurement_audit",
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
                    "same_section_source_overlap_allowed",
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
            if measurement_coverage["same_section_source_overlap_allowed"] is not True:
                raise ValueError(
                    "measurement-development must report shared same-section source identity"
                )
            measurement_decisions = _mapping(
                content["decision_thresholds"],
                "decision_thresholds",
                {
                    "required_opposite_pool_guard_um",
                    "minimum_common_reliable_genes",
                    *FROZEN_H_MEAS_SHARED_MEASUREMENT_THRESHOLDS,
                },
            )
            _validate_frozen_h_meas_shared_thresholds(measurement_decisions, "decision_thresholds")
            required_guard = measurement_decisions["required_opposite_pool_guard_um"]
            if (
                isinstance(required_guard, bool)
                or not isinstance(required_guard, (int, float))
                or not 0 < float(required_guard) < float("inf")
            ):
                raise ValueError(
                    "measurement-development required opposite-pool guard "
                    "must be finite and positive"
                )
            target_strategy = _mapping(
                content.get("target_panel_strategy"),
                "target_panel_strategy",
                {
                    "primary",
                    "minimum_primary_gene_count",
                    "fallback",
                    "fallback_activation",
                    "fallback_requires_new_h_meas_manifest",
                    "fallback_requires_new_calibration",
                    "locked_outcomes_may_inform_fallback",
                },
            )
            minimum_genes = measurement_decisions["minimum_common_reliable_genes"]
            if (
                target_strategy["primary"]
                != "common_reliable_panel_across_all_supported_fine_types"
                or target_strategy["fallback"]
                != "new_study_version_type_specific_panels_or_prespecified_programs"
                or target_strategy["fallback_activation"]
                != "only_after_development_h_meas_failure_before_any_confirmatory_opening"
                or target_strategy["fallback_requires_new_h_meas_manifest"] is not True
                or target_strategy["fallback_requires_new_calibration"] is not True
                or target_strategy["locked_outcomes_may_inform_fallback"] is not False
                or isinstance(minimum_genes, bool)
                or int(minimum_genes) != minimum_genes
                or int(minimum_genes) < 1
                or target_strategy["minimum_primary_gene_count"] != minimum_genes
            ):
                raise ValueError("measurement target-panel fallback is not frozen pre-outcome")
        else:
            if "target_panel_strategy" in content:
                raise ValueError("confirmatory morphology cannot redefine H-MEAS target strategy")
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
                "locked_measurement_audit",
            }
            if not morphology_required.issubset(content):
                raise ValueError("confirmatory morphology manifest is incomplete")
            morphology_coverage = _mapping(
                content["coverage_requirements"],
                "coverage_requirements",
                {
                    "minimum_reference_cells_per_donor_section_type",
                    "minimum_evaluation_cells_per_donor_section_type",
                },
            )
            for name in (
                "minimum_reference_cells_per_donor_section_type",
                "minimum_evaluation_cells_per_donor_section_type",
            ):
                value = morphology_coverage[name]
                if isinstance(value, bool) or int(value) != value or int(value) < 1:
                    raise ValueError("confirmatory morphology %s must be positive" % name)
            _validate_confirmatory_primary_endpoint(
                content["primary_endpoint"], observations["fine_type_field"]
            )
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
            locked_audit = _mapping(
                content["locked_measurement_audit"],
                "locked_measurement_audit",
                {
                    "audit_timing",
                    "selection_changes_forbidden",
                    "coverage_denominator",
                    "maximum_annotation_nucleus_p95_um",
                    "maximum_annotation_cell_p95_um",
                    "maximum_cell_nucleus_p95_um",
                    "maximum_registration_nucleus_diameter_ratio_p95",
                    "maximum_registration_nearest_neighbor_ratio_p95",
                    "best_registration_quality_max_fraction_of_limit",
                    "intermediate_registration_quality_max_fraction_of_limit",
                    "maximum_registration_outlier_fraction",
                    "maximum_nucleus_outside_cell_fraction",
                    "minimum_nucleus_cell_area_ratio",
                    "maximum_nucleus_cell_area_ratio",
                    "maximum_segmentation_outlier_fraction",
                    "maximum_crop_padding_p95",
                    "mostly_padded_cutoff",
                    "maximum_mostly_padded_fraction",
                    "minimum_within_fine_type_reliability",
                    "minimum_reliability_rows",
                    "minimum_locked_donor_type_reliability_fraction",
                },
            )
            _validate_frozen_h_meas_shared_thresholds(locked_audit, "locked_measurement_audit")
            if (
                locked_audit["audit_timing"]
                != "after_confirmatory_lock_before_morphology_inference"
                or locked_audit["selection_changes_forbidden"] is not True
                or locked_audit["coverage_denominator"]
                != "all_h_meas_supported_fine_types_and_locked_donors"
            ):
                raise ValueError("locked measurement audit timing or population is mutable")
            for name in (
                "maximum_registration_outlier_fraction",
                "best_registration_quality_max_fraction_of_limit",
                "intermediate_registration_quality_max_fraction_of_limit",
                "maximum_nucleus_outside_cell_fraction",
                "minimum_nucleus_cell_area_ratio",
                "maximum_nucleus_cell_area_ratio",
                "maximum_segmentation_outlier_fraction",
                "maximum_crop_padding_p95",
                "mostly_padded_cutoff",
                "maximum_mostly_padded_fraction",
                "minimum_within_fine_type_reliability",
                "minimum_locked_donor_type_reliability_fraction",
            ):
                if not 0 <= float(locked_audit[name]) <= 1:
                    raise ValueError("locked_measurement_audit.%s must be in [0, 1]" % name)
            if not (
                float(locked_audit["best_registration_quality_max_fraction_of_limit"])
                < float(locked_audit["intermediate_registration_quality_max_fraction_of_limit"])
                < 1.0
            ):
                raise ValueError(
                    "locked measurement registration-quality strata must be ordered below one"
                )
            if int(locked_audit["minimum_reliability_rows"]) < 2:
                raise ValueError("locked measurement reliability needs at least two rows")
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
                _OPENING_RECEIPT_FIELDS,
            )
            if set(opening) != _OPENING_RECEIPT_FIELDS:
                raise ValueError("study manifest opening receipt has extra fields")
            _sha256(opening["locked_manifest_sha256"], "opening.locked_manifest_sha256")
            opening_locked_content = _sha256(
                opening["locked_content_sha256"], "opening.locked_content_sha256"
            )
            if opening_locked_content != recorded_digest:
                raise ValueError("opening receipt does not bind the locked scientific content")
            if not re.fullmatch(r"[0-9a-f]{40}", str(opening["opened_by_commit"])):
                raise ValueError("opened study commit is invalid")
            permitted_claims = _strings(
                opening["permitted_claims"], "opening.permitted_claims", allow_empty=True
            )
            if not set(permitted_claims).issubset(hypotheses):
                raise ValueError(
                    "opening permitted claims must be a subset of the frozen hypotheses"
                )
            if opening.get("adoption_for_future_models") is not False:
                raise ValueError("opened locked evidence cannot become future development data")
            opening_receipt = _sha256(
                opening["opening_receipt_sha256"], "opening.opening_receipt_sha256"
            )
            if opening_receipt != _opening_receipt_digest(opening):
                raise ValueError("opening receipt was modified after the study was opened")
        if verify_runtime:
            root = Path(repository_root or resolved.parent).expanduser().resolve()
            if current_git_commit(root) != commit:
                raise ValueError("running commit differs from the locked study manifest")
            if require_clean_runtime:
                require_clean_worktree(root)
            if verify_container_digest:
                runtime_container = os.environ.get("HEIR_CONTAINER_DIGEST")
                if runtime_container is None:
                    raise ValueError("HEIR_CONTAINER_DIGEST is required for confirmatory execution")
                if runtime_container != container:
                    raise ValueError(
                        "runtime container digest differs from the locked study manifest"
                    )
        elif require_clean_runtime or verify_container_digest:
            raise ValueError("full runtime checks require verify_runtime=True")
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
    if draft.get("study_stage") == "measurement_development":
        independence = _validate_label_target_independence(draft)
        if independence["evidence_kind"] == "pending":
            raise ValueError(
                "H-MEAS cannot lock while fine-type and training-label provenance is pending"
            )
    if draft.get("study_stage") == "confirmatory_morphology":
        partitions = _mapping(draft.get("partitions"), "partitions", {"locked_test_donors"})
        locked_donors = _strings(partitions["locked_test_donors"], "partitions.locked_test_donors")
        if set(locked_donors) & KNOWN_GSE250346_PRE_REGISTRATION_EXPOSED_DONORS:
            raise ValueError(
                "known GSE250346 donor exposure cannot be reset by a manifest declaration"
            )
        protection = _mapping(
            draft.get("lock_protection"),
            "lock_protection",
            {
                "prior_outcome_access_confirmed_false",
                "prior_outcome_access_status",
                "prospective_lock_eligible",
            },
        )
        if (
            protection["prior_outcome_access_confirmed_false"] is not True
            or protection["prior_outcome_access_status"] != "unopened"
            or protection["prospective_lock_eligible"] is not True
        ):
            raise ValueError(
                "confirmatory H-CELL cannot lock previously materialized donor outcomes"
            )
        independence = _validate_label_target_independence(draft)
        if independence["evidence_kind"] == "pending":
            raise ValueError(
                "confirmatory H-CELL cannot lock before label-target independence is proven"
            )
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
    if not re.fullmatch(r"[0-9a-f]{40}", opened_by_commit):
        raise ValueError("opened study commit is invalid")
    claims = tuple(str(value) for value in permitted_claims)
    if (
        any(not value.strip() for value in claims)
        or len(set(claims)) != len(claims)
        or not set(claims).issubset(locked.hypothesis_ids)
    ):
        raise ValueError("opening permitted claims must be a subset of the frozen hypotheses")
    value = json.loads(json.dumps(locked.content))
    value["status"] = "opened"
    opening = {
        "locked_manifest_sha256": locked.sha256,
        "locked_content_sha256": locked.content["locked_content_sha256"],
        "opened_by_commit": opened_by_commit,
        "opened_at": opened_at or datetime.now(timezone.utc).isoformat(),
        "permitted_claims": list(claims),
        "adoption_for_future_models": False,
    }
    opening["opening_receipt_sha256"] = _opening_receipt_digest(opening)
    value["opening"] = opening
    return value


__all__ = [
    "LABEL_TARGET_INDEPENDENCE_FIELDS",
    "LABEL_TARGET_INDEPENDENCE_PROTOCOL_FIELDS",
    "DEFAULT_ANNOTATION_QUALITY_CONTRACT",
    "STUDY_MANIFEST_SCHEMA",
    "StudyManifest",
    "current_git_commit",
    "freeze_manifest_content",
    "open_manifest_content",
    "require_clean_worktree",
]
