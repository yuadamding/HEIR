#!/usr/bin/env python3
"""Freeze measurement-qualified morphology artifacts from a registered source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact, StudyManifest, ordered_ids_sha256
from heir.evaluation.control_models import (
    HEST_CROP_CONTRACT,
    REQUIRED_HEST_CONTROL_DECLARATIONS,
    REQUIRED_HEST_CROP_IDS,
)
from heir.evaluation.measurement_gate import load_passing_measurement_receipt
from heir.evaluation.reliability import feature_reliability, normalize_split_counts
from heir.utils import reject_output_input_collisions, sha256_file

PLAN_SCHEMA = "heir.morphology_ridge_preparation_plan.v1"


def _mapping(value: object, name: str, required: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("%s is incomplete" % name)
    return value


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("%s must be a lowercase SHA-256" % name)
    return digest


def _load_plan(path: Path, source: Path) -> Mapping[str, object]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("morphology-ridge plan is not valid JSON") from error
    required = {"schema", "source_schema", "source_observations_sha256"}
    if not isinstance(plan, Mapping) or not required.issubset(plan):
        raise ValueError("morphology-ridge source plan is incomplete")
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError("morphology-ridge preparation plan schema is unsupported")
    if plan["source_observations_sha256"] != sha256_file(source):
        raise ValueError("morphology-ridge source observations differ from the source plan")
    return plan


def _write_npz(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz.tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _row_digest(identities: np.ndarray, values: np.ndarray, role: str) -> str:
    digest = hashlib.sha256(role.encode("utf-8"))
    for identity in identities.astype(str):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _identity_digest(identities: np.ndarray, role: str) -> str:
    digest = hashlib.sha256(role.encode("utf-8"))
    for identity in sorted(identities.astype(str).tolist()):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _scalar(archive: Mapping[str, np.ndarray], names: Sequence[str]) -> Optional[object]:
    for name in names:
        if name in archive:
            value = np.asarray(archive[name])
            if value.ndim != 0:
                raise ValueError("source field %s must be scalar" % name)
            return value.item()
    return None


def _array(
    archive: Mapping[str, np.ndarray], names: Sequence[str], *, required: bool = True
) -> Optional[np.ndarray]:
    for name in names:
        if name in archive:
            return np.array(archive[name], copy=True)
    if required:
        raise ValueError("source cell artifact is missing: %s" % names[0])
    return None


def _named_matrix(
    archive: Mapping[str, np.ndarray],
    matrix_names: Sequence[str],
    feature_names: Sequence[str],
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _array(archive, matrix_names, required=False)
    names = _array(archive, feature_names, required=False)
    if matrix is None and names is None:
        return np.empty((rows, 0), dtype=np.float64), np.asarray([], dtype=str)
    if matrix is None or names is None:
        raise ValueError("source %s and its names must be supplied together" % matrix_names[0])
    values = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(names).astype(str)
    if (
        values.ndim != 2
        or values.shape[0] != rows
        or labels.ndim != 1
        or values.shape[1] != len(labels)
        or len(set(labels.tolist())) != len(labels)
        or any(not label.strip() for label in labels.tolist())
        or not np.isfinite(values).all()
    ):
        raise ValueError("source %s differ from their names" % matrix_names[0])
    return values, labels


def _target_selection(
    manifest: StudyManifest,
    measurement_path: Path,
) -> tuple[Mapping[str, object], tuple[str, ...], tuple[str, ...], str]:
    prerequisites = _mapping(
        manifest.content.get("prerequisites"),
        "study manifest prerequisites",
        {
            "measurement_report_sha256",
            "measurement_study_manifest_sha256",
            "measurement_source_sha256",
        },
    )
    report_sha = _sha256(
        prerequisites["measurement_report_sha256"],
        "prerequisites.measurement_report_sha256",
    )
    measurement_study_sha = _sha256(
        prerequisites["measurement_study_manifest_sha256"],
        "prerequisites.measurement_study_manifest_sha256",
    )
    expected_source_sha = _sha256(
        prerequisites["measurement_source_sha256"],
        "prerequisites.measurement_source_sha256",
    )
    report = load_passing_measurement_receipt(
        measurement_path,
        expected_receipt_sha256=report_sha,
        expected_study_manifest_sha256=measurement_study_sha,
        expected_source_sha256=expected_source_sha,
    )
    measurement_thresholds = _mapping(report.get("thresholds"), "H-MEAS frozen thresholds", set())
    locked_audit = _mapping(
        manifest.content.get("locked_measurement_audit"),
        "locked measurement audit",
        {
            "selection_changes_forbidden",
            "coverage_denominator",
            "minimum_locked_donor_type_reliability_fraction",
        },
    )
    shared_thresholds = {
        name: value
        for name, value in locked_audit.items()
        if name
        not in {
            "audit_timing",
            "selection_changes_forbidden",
            "coverage_denominator",
            "minimum_locked_donor_type_reliability_fraction",
        }
    }
    if (
        any(measurement_thresholds.get(name) != value for name, value in shared_thresholds.items())
        or locked_audit["selection_changes_forbidden"] is not True
        or locked_audit["coverage_denominator"]
        != "all_h_meas_supported_fine_types_and_locked_donors"
    ):
        raise ValueError("locked-row measurement audit differs from frozen H-MEAS thresholds")
    selection = _mapping(
        report.get("target_selection_receipt"),
        "H-MEAS target selection receipt",
        {
            "selection_partition",
            "primary_target_variant",
            "ordered_reliable_gene_ids",
            "ordered_reliable_gene_panel_sha256",
            "supported_fine_type_ids",
            "supported_fine_type_panel_sha256",
            "locked_test_molecular_outcomes_used",
        },
    )
    genes = tuple(str(value) for value in selection["ordered_reliable_gene_ids"])
    fine_types = tuple(str(value) for value in selection["supported_fine_type_ids"])
    if (
        selection.get("schema") != "heir.measurement_target_selection.v1"
        or selection.get("pass") is not True
        or selection["selection_partition"] != "development_only"
        or selection["primary_target_variant"] != "nucleus_overlapping_transcripts"
        or selection["locked_test_molecular_outcomes_used"] is not False
        or not genes
        or not fine_types
        or len(set(genes)) != len(genes)
        or len(set(fine_types)) != len(fine_types)
        or selection["ordered_reliable_gene_panel_sha256"] != ordered_ids_sha256(genes)
        or selection["supported_fine_type_panel_sha256"] != ordered_ids_sha256(fine_types)
    ):
        raise ValueError("H-MEAS target selection receipt is not development-only and bound")
    if manifest.content["target_gene_panel_sha256"] != ordered_ids_sha256(genes):
        raise ValueError("locked H-CELL target panel differs from the H-MEAS panel")
    observations = _mapping(
        manifest.content["observations"],
        "study manifest observations",
        {"fine_type_field"},
    )
    declared_types = observations.get("supported_fine_type_ids")
    if declared_types is not None and tuple(str(value) for value in declared_types) != fine_types:
        raise ValueError("locked H-CELL fine types differ from the H-MEAS support set")
    return report, genes, fine_types, report_sha


def _lock_protection(
    manifest: StudyManifest,
    cohort_id: str,
    donor_ids: np.ndarray,
    archive: Mapping[str, np.ndarray],
) -> None:
    if cohort_id == "HESCAPE":
        protection = _mapping(
            manifest.content.get("lock_protection"),
            "study manifest lock_protection",
            {
                "reserved_donor_ids",
                "hescape_analysis_scope",
                "hescape_allowed_donor_ids",
            },
        )
        source_reserved = _array(archive, ("reserved_hest_locked_donors",))
        reserved = tuple(np.asarray(source_reserved).astype(str).tolist())
        source_loaded = _scalar(archive, ("reserved_donor_outcomes_loaded",))
        analysis_scope = str(_scalar(archive, ("analysis_scope",)) or "")
        actual_donors = set(donor_ids.astype(str).tolist())
        protected = set(str(value) for value in protection["reserved_donor_ids"])
        allowed = set(str(value) for value in protection["hescape_allowed_donor_ids"])
        overlap = sorted(actual_donors & protected)
        if (
            set(reserved) != protected
            or protected != set(manifest.locked_test_donors)
            or allowed != set(manifest.development_donors)
            or not actual_donors.issubset(allowed)
            or source_loaded is not False
            or overlap
        ):
            raise ValueError("HESCAPE contains or declares reserved HEST locked outcomes")
        if analysis_scope != protection["hescape_analysis_scope"]:
            raise ValueError("HESCAPE source must declare a development-only analysis scope")
    elif cohort_id == "HEST":
        protection = _mapping(
            manifest.content.get("lock_protection"),
            "study manifest lock_protection",
            {
                "reserved_exclusively_for",
                "reserved_donor_ids",
                "prior_outcome_access_confirmed_false",
            },
        )
        reserved = tuple(str(value) for value in protection["reserved_donor_ids"])
        if (
            protection["reserved_exclusively_for"] != "H-CELL"
            or protection["prior_outcome_access_confirmed_false"] is not True
            or set(reserved) != set(manifest.locked_test_donors)
            or len(set(reserved)) != len(reserved)
        ):
            raise ValueError("HEST locked donors are not exclusively protected for H-CELL")


def _scientific_scope(manifest: StudyManifest, cohort_id: str) -> str:
    if cohort_id == "HEST":
        gate = _mapping(
            manifest.content.get("morphology_gate"),
            "study manifest morphology_gate",
            {"scientific_scope"},
        )
        scope = str(gate["scientific_scope"])
        if scope != "registered_cell_local_context_association":
            raise ValueError("HEST scientific scope is not the locked G2 local-context claim")
        return scope
    protection = _mapping(
        manifest.content.get("lock_protection"),
        "study manifest lock_protection",
        {"hescape_analysis_scope"},
    )
    scope = str(protection["hescape_analysis_scope"])
    if scope != "development_donors_only_hest_lock_unopened":
        raise ValueError("HESCAPE scientific scope is not development-only")
    return scope


def _crop_metadata(
    plan: Mapping[str, object],
    archive: Mapping[str, np.ndarray],
    crop_ids: tuple[str, ...],
    *,
    require_source: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    direct_roles = _array(archive, ("crop_roles",), required=False)
    direct_families = _array(archive, ("crop_comparison_families",), required=False)
    if direct_roles is not None or direct_families is not None:
        if direct_roles is None or direct_families is None:
            raise ValueError("source crop roles and comparison families must be supplied together")
        roles = tuple(np.asarray(direct_roles).astype(str).tolist())
        families = tuple(np.asarray(direct_families).astype(str).tolist())
        if len(roles) != len(crop_ids) or len(families) != len(crop_ids):
            raise ValueError("source crop metadata differs from the full crop ladder")
        return roles, families
    if require_source:
        raise ValueError("confirmatory HEST source must carry its bound crop metadata")
    variants: Optional[object] = None
    direct = plan.get("crop_metadata")
    if isinstance(direct, Mapping):
        variants = direct.get("variants")
    provenance_value = _scalar(archive, ("provenance_json",))
    if variants is None and provenance_value is not None:
        try:
            provenance = json.loads(str(provenance_value))
        except json.JSONDecodeError as error:
            raise ValueError("source provenance_json is malformed") from error
        if isinstance(provenance, Mapping):
            metadata = provenance.get("crop_metadata")
            if isinstance(metadata, Mapping):
                variants = metadata.get("variants")
    if variants is None:
        crop_manifest = plan.get("crop_manifest")
        if isinstance(crop_manifest, Mapping):
            path = Path(str(crop_manifest.get("path", ""))).expanduser().resolve()
            expected_sha = str(crop_manifest.get("sha256", ""))
            if path.is_file() and sha256_file(path) == expected_sha:
                try:
                    content = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    raise ValueError("crop manifest is malformed") from error
                variants = content.get("variants") if isinstance(content, Mapping) else None
    if not isinstance(variants, list):
        raise ValueError("source plan does not bind crop roles and comparison families")
    indexed = {str(value.get("crop_id")): value for value in variants if isinstance(value, Mapping)}
    if set(indexed) != set(crop_ids):
        raise ValueError("crop metadata differs from the full source crop ladder")
    roles = tuple(str(indexed[crop]["role"]) for crop in crop_ids)
    families = tuple(str(indexed[crop]["comparison_family"]) for crop in crop_ids)
    if any(not value for value in roles + families):
        raise ValueError("crop roles and comparison families must be explicit")
    return roles, families


def _validate_hest_source_contract(
    manifest: StudyManifest,
    archive: Mapping[str, np.ndarray],
    *,
    plan: Mapping[str, object],
    source_gene_ids: np.ndarray,
    selected_target_gene_ids: tuple[str, ...],
    marker_genes: tuple[str, ...],
    crop_ids: tuple[str, ...],
    crop_roles: tuple[str, ...],
    crop_families: tuple[str, ...],
    named: Mapping[str, tuple[np.ndarray, np.ndarray]],
    source_identity: Mapping[str, object],
) -> None:
    opening = _mapping(
        manifest.content.get("opening"),
        "study manifest opening",
        {"opening_receipt_sha256", "permitted_claims"},
    )
    opening_receipt_sha256 = _sha256(
        opening["opening_receipt_sha256"], "opening.opening_receipt_sha256"
    )
    if (
        str(_scalar(archive, ("study_stage",)) or "") != "confirmatory_morphology"
        or str(_scalar(archive, ("source_scope",)) or "")
        != "development_and_locked_after_confirmatory_opening"
        or bool(_scalar(archive, ("locked_donor_outcomes_materialized",))) is not True
        or str(_scalar(archive, ("study_manifest_sha256",)) or "") != manifest.sha256
        or str(_scalar(archive, ("opening_receipt_sha256",)) or "") != opening_receipt_sha256
        or plan.get("opening_receipt_sha256") != opening_receipt_sha256
        or "H-CELL" not in opening["permitted_claims"]
    ):
        raise ValueError("HEST morphology source was not built from the authorized opening receipt")
    encoder = _mapping(
        manifest.content.get("encoder"),
        "study manifest encoder",
        {"manifest_sha256", "feature_space_id", "checkpoint_sha256"},
    )
    source_encoder_manifest = str(_scalar(archive, ("encoder_manifest_sha256",)) or "")
    source_crop_manifest = str(_scalar(archive, ("crop_manifest_sha256",)) or "")
    crop_protocols = manifest.content.get("crop_protocols")
    if (
        not isinstance(crop_protocols, list)
        or tuple(str(value) for value in crop_protocols) != (source_crop_manifest,)
        or source_encoder_manifest != encoder["manifest_sha256"]
        or source_identity["feature_space_id"] != encoder["feature_space_id"]
        or source_identity["feature_checkpoint_sha256"] != encoder["checkpoint_sha256"]
    ):
        raise ValueError("HEST encoder or crop ladder differs from the locked manifest")
    source_marker_panel = str(_scalar(archive, ("fine_type_marker_panel_sha256",)) or "")
    if (
        ordered_ids_sha256(source_gene_ids.astype(str).tolist())
        != manifest.content["candidate_target_gene_panel_sha256"]
        or ordered_ids_sha256(marker_genes) != manifest.content["type_marker_panel_sha256"]
        or source_marker_panel != manifest.content["type_marker_panel_sha256"]
    ):
        raise ValueError("HEST candidate target or fine-type marker panel is not locked")
    independence = _mapping(
        manifest.content.get("label_target_independence"),
        "study manifest label_target_independence",
        {
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
            "locked_donors_used_for_training",
            "same_cohort_annotation",
            "cross_fitting_method",
            "cross_fitting_receipt_sha256",
            "establishes_full_target_independence",
        },
    )
    try:
        source_independence = json.loads(str(_scalar(archive, ("label_target_independence_json",))))
        provenance = json.loads(str(_scalar(archive, ("provenance_json",))))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("HEST source label-target independence provenance is malformed") from error
    expected_independence = dict(independence)
    source_independence_sha = hashlib.sha256(
        json.dumps(source_independence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    annotation_ids = tuple(str(value) for value in independence["ordered_annotation_feature_ids"])
    target_ids = tuple(str(value) for value in independence["ordered_target_gene_ids"])
    source_annotation_ids = tuple(
        np.asarray(_array(archive, ("annotation_feature_ids",))).astype(str).tolist()
    )
    source_annotation_receipt = str(_scalar(archive, ("annotation_receipt_sha256",)) or "")
    source_annotation_predictions = str(
        _scalar(archive, ("annotation_prediction_export_sha256",)) or ""
    )
    source_label_sha256 = str(_scalar(archive, ("label_source_sha256",)) or "")
    source_label_kind = str(_scalar(archive, ("label_source_kind",)) or "")
    if (
        source_independence != expected_independence
        or str(_scalar(archive, ("label_target_independence_sha256",))) != source_independence_sha
        or not isinstance(provenance, Mapping)
        or provenance.get("label_target_independence") != expected_independence
        or plan.get("label_target_independence") != expected_independence
        or source_annotation_receipt != independence["annotation_receipt_sha256"]
        or provenance.get("label_receipt_sha256") != source_annotation_receipt
        or plan.get("annotation_receipt_sha256") != source_annotation_receipt
        or not re.fullmatch(r"[0-9a-f]{64}", source_annotation_predictions)
        or provenance.get("label_source_sha256") != source_annotation_predictions
        or plan.get("annotation_prediction_export_sha256") != source_annotation_predictions
        or source_label_sha256 != source_annotation_predictions
        or source_label_kind != "independent_annotation_prediction_export"
        or provenance.get("label_source_kind") != source_label_kind
        or plan.get("label_source_kind") != source_label_kind
        or source_annotation_ids != annotation_ids
        or independence["ordered_annotation_feature_ids_sha256"]
        != ordered_ids_sha256(annotation_ids)
        or independence["ordered_target_gene_ids_sha256"] != ordered_ids_sha256(target_ids)
        or independence["ordered_target_gene_ids_sha256"]
        != manifest.content["target_gene_panel_sha256"]
        or tuple(selected_target_gene_ids) != target_ids
        or set(annotation_ids) & set(target_ids)
        or independence["annotation_target_overlap_count"] != 0
        or independence["locked_donors_used_for_training"] is not False
        or independence["establishes_full_target_independence"] is not True
        or independence["evidence_kind"] == "pending"
    ):
        raise ValueError(
            "HEST source label-target independence contract differs from the locked manifest"
        )
    if independence["same_cohort_annotation"] is True and (
        independence["cross_fitting_method"] != "leave_one_donor_out"
        or independence["cross_fitting_receipt_sha256"] is None
    ):
        raise ValueError("HEST same-cohort labels are not donor-cross-fitted")

    declared_controls = manifest.content.get("controls")
    if not isinstance(declared_controls, list) or set(declared_controls) != set(
        REQUIRED_HEST_CONTROL_DECLARATIONS
    ):
        raise ValueError("HEST strong-control family differs from the locked manifest")
    for family in ("stain", "nuclear", "cell", "cellvit", "density", "boundary", "spatial"):
        if named[family][0].shape[1] == 0:
            raise ValueError("HEST locked control %s is unavailable" % family)
    declared_technical = set(str(value) for value in manifest.content["technical_covariates"])
    effective_technical = set(named["technical"][1].astype(str).tolist()) | {
        "section_id",
        "disease_status",
        "site_id",
        "batch_id",
    }
    if declared_technical != effective_technical:
        raise ValueError("HEST technical and metadata covariates differ from the lock")
    if tuple(named["technical"][1].astype(str).tolist()) != ("log1p_library_size",):
        raise ValueError("HEST depth covariate differs from the locked endpoint")
    if not any(
        name in archive for name in ("spatial_control_features", "spatial_features")
    ) or not any(
        name in archive for name in ("spatial_control_feature_names", "spatial_feature_names")
    ):
        raise ValueError("HEST source omits the explicit smooth spatial control")
    if set(crop_ids) != set(REQUIRED_HEST_CROP_IDS):
        raise ValueError("HEST source does not carry the complete locked crop family")
    actual_crop_contract = {
        crop_id: (role, family)
        for crop_id, role, family in zip(crop_ids, crop_roles, crop_families)
    }
    if actual_crop_contract != HEST_CROP_CONTRACT:
        raise ValueError("HEST source crop roles or comparison families differ from the lock")


def _reference_splits(
    archive: Mapping[str, np.ndarray], rows: int, manifest: StudyManifest
) -> tuple[tuple[str, ...], np.ndarray]:
    split_ids = _array(archive, ("reference_split_ids", "pool_role_split_ids"), required=False)
    role_matrix = _array(archive, ("pool_roles_by_split",), required=False)
    primary_roles = np.asarray(_array(archive, ("pool_roles", "pool_role"))).astype(str)
    if split_ids is None and role_matrix is None:
        declared = manifest.content.get("reference_splits")
        if isinstance(declared, Mapping):
            declared_ids = tuple(str(value) for value in declared.get("split_ids", ()))
            if len(declared_ids) != 1:
                raise ValueError(
                    "source omits the multiple frozen reference splits in the locked manifest"
                )
        primary_id = (
            str(declared.get("primary_split_id"))
            if isinstance(declared, Mapping) and declared.get("primary_split_id")
            else "primary"
        )
        return (primary_id,), primary_roles[:, None]
    if split_ids is None or role_matrix is None:
        raise ValueError("multiple frozen split IDs and pool roles must be supplied together")
    identities = tuple(np.asarray(split_ids).astype(str).tolist())
    roles = np.asarray(role_matrix).astype(str)
    if roles.shape != (rows, len(identities)) or len(set(identities)) != len(identities):
        raise ValueError("multiple frozen reference split roles are malformed")
    declared = _mapping(
        manifest.content.get("reference_splits"),
        "study manifest reference_splits",
        {"primary_split_id", "split_ids"},
    )
    declared_ids = tuple(str(value) for value in declared["split_ids"])
    primary = str(declared["primary_split_id"])
    if set(declared_ids) != set(identities) or primary not in identities:
        raise ValueError("source reference splits differ from the locked manifest")
    order = [identities.index(primary)] + [
        identities.index(value) for value in declared_ids if value != primary
    ]
    return tuple(identities[index] for index in order), roles[:, order]


def _standardized_mean_difference(reference: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    if reference.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    pooled = np.sqrt((reference.var(axis=0) + evaluation.var(axis=0)) / 2.0)
    pooled = np.maximum(pooled, 1.0e-8)
    return (evaluation.mean(axis=0) - reference.mean(axis=0)) / pooled


def _balance_report(
    values: Mapping[str, np.ndarray],
    reference: np.ndarray,
    evaluation: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    type_names: tuple[str, ...],
    feature_matrix: np.ndarray,
    feature_names: tuple[str, ...],
    continuous_threshold: Optional[float],
    categorical_threshold: Optional[float],
) -> Mapping[str, object]:
    strata = []
    maximum = 0.0
    feature_maxima = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    categorical_maxima = {
        name: 0.0 for name in ("section_ids", "disease_states", "site_ids", "batch_ids")
    }
    for donor in sorted(set(donors[evaluation].tolist())):
        donor_evaluation = evaluation & (donors == donor)
        for section_id in sorted(set(values["section_ids"][donor_evaluation].astype(str).tolist())):
            section = values["section_ids"].astype(str) == section_id
            for type_index, type_name in enumerate(type_names):
                local_reference = reference & (donors == donor) & section & (labels == type_index)
                local_evaluation = evaluation & (donors == donor) & section & (labels == type_index)
                if not local_reference.any() or not local_evaluation.any():
                    continue
                differences = _standardized_mean_difference(
                    feature_matrix[local_reference], feature_matrix[local_evaluation]
                )
                local_maximum = float(np.max(np.abs(differences))) if len(differences) else 0.0
                maximum = max(maximum, local_maximum)
                if len(differences):
                    feature_maxima = np.maximum(feature_maxima, np.abs(differences))
                local_categorical = {}
                for name in categorical_maxima:
                    values_by_name = values[name].astype(str)
                    levels = sorted(
                        set(values_by_name[local_reference].tolist())
                        | set(values_by_name[local_evaluation].tolist())
                    )
                    reference_fraction = np.asarray(
                        [np.mean(values_by_name[local_reference] == level) for level in levels]
                    )
                    evaluation_fraction = np.asarray(
                        [np.mean(values_by_name[local_evaluation] == level) for level in levels]
                    )
                    total_variation = float(
                        0.5 * np.abs(reference_fraction - evaluation_fraction).sum()
                    )
                    local_categorical[name] = total_variation
                    categorical_maxima[name] = max(categorical_maxima[name], total_variation)
                strata.append(
                    {
                        "donor_id": donor,
                        "section_id": section_id,
                        "fine_type_id": type_name,
                        "reference_cells": int(local_reference.sum()),
                        "evaluation_cells": int(local_evaluation.sum()),
                        "maximum_absolute_standardized_mean_difference": local_maximum,
                        "categorical_total_variation": local_categorical,
                    }
                )
    categorical = {}
    for name in ("section_ids", "disease_states", "site_ids", "batch_ids"):
        array = values[name].astype(str)
        categorical[name] = {
            "reference_counts": {
                value: int(np.count_nonzero(reference & (array == value)))
                for value in sorted(set(array[reference].tolist()))
            },
            "evaluation_counts": {
                value: int(np.count_nonzero(evaluation & (array == value)))
                for value in sorted(set(array[evaluation].tolist()))
            },
        }
    return {
        "continuous_feature_names": list(feature_names),
        "maximum_absolute_standardized_mean_difference_by_feature": {
            name: float(value) for name, value in zip(feature_names, feature_maxima)
        },
        "maximum_absolute_standardized_mean_difference": maximum,
        "maximum_allowed_absolute_standardized_mean_difference": continuous_threshold,
        "maximum_categorical_total_variation_by_field": categorical_maxima,
        "maximum_categorical_total_variation": max(categorical_maxima.values()),
        "maximum_allowed_categorical_total_variation": categorical_threshold,
        "strata": strata,
        "categorical_distributions": categorical,
        "pass": (
            None
            if continuous_threshold is None or categorical_threshold is None
            else bool(
                maximum <= continuous_threshold
                and max(categorical_maxima.values()) <= categorical_threshold
            )
        ),
    }


def _json(value: object) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _locked_measurement_audit_report(
    *,
    contract: Mapping[str, object],
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    fine_type_ids: np.ndarray,
    locked_donors: Sequence[str],
    supported_types: Sequence[str],
    gene_ids: Sequence[str],
    half_a_counts: np.ndarray,
    half_b_counts: np.ndarray,
    half_a_library_sizes: np.ndarray,
    half_b_library_sizes: np.ndarray,
    source_locked_measurement_qc_pass: np.ndarray,
    target_qc_pass: np.ndarray,
    registration_qc_pass: np.ndarray,
    segmentation_qc_pass: np.ndarray,
    crop_qc_pass: np.ndarray,
    annotation_nucleus_um: np.ndarray,
    annotation_cell_um: np.ndarray,
    cell_nucleus_um: np.ndarray,
    nucleus_area_um2: np.ndarray,
    nearest_neighbor_um: np.ndarray,
    nucleus_inside_cell: np.ndarray,
    cell_area_um2: np.ndarray,
    crop_ids: Sequence[str],
    crop_padding_fractions: np.ndarray,
) -> Mapping[str, object]:
    """Audit locked measurement quality without selecting genes, types, or thresholds."""

    locked = np.isin(donor_ids, np.asarray(tuple(locked_donors)))
    if not locked.any():
        raise ValueError("confirmatory source lacks locked rows for measurement audit")
    nucleus_diameter = 2.0 * np.sqrt(nucleus_area_um2 / np.pi)
    area_ratio = nucleus_area_um2 / np.maximum(cell_area_um2, np.finfo(float).eps)
    maximum_registration_outliers = float(contract["maximum_registration_outlier_fraction"])

    def summarize_threshold(
        values: np.ndarray,
        selected: np.ndarray,
        *,
        maximum: float,
        maximum_outliers: float,
    ) -> tuple[Mapping[str, object], np.ndarray]:
        valid = np.isfinite(values) & (values >= 0.0)
        row_pass = valid & (values <= maximum)
        valid_values = values[selected & valid]
        p95 = float(np.quantile(valid_values, 0.95)) if len(valid_values) else None
        outlier_fraction = float(np.mean(~row_pass[selected])) if selected.any() else 1.0
        report = {
            "rows": int(np.count_nonzero(selected)),
            "p95": p95,
            "maximum_allowed_p95": float(maximum),
            "outlier_fraction": outlier_fraction,
            "maximum_allowed_outlier_fraction": float(maximum_outliers),
            "pass": bool(
                selected.any()
                and np.count_nonzero(selected & valid) == np.count_nonzero(selected)
                and p95 is not None
                and p95 <= maximum
                and outlier_fraction <= maximum_outliers
            ),
        }
        return report, row_pass

    def absolute_metric(
        values: np.ndarray, maximum: float
    ) -> tuple[Mapping[str, object], np.ndarray]:
        overall, row_pass = summarize_threshold(
            values,
            locked,
            maximum=maximum,
            maximum_outliers=maximum_registration_outliers,
        )
        by_section = {}
        for section_id in sorted(set(section_ids[locked].astype(str).tolist())):
            selected = locked & (section_ids.astype(str) == section_id)
            by_section[section_id], _ = summarize_threshold(
                values,
                selected,
                maximum=maximum,
                maximum_outliers=maximum_registration_outliers,
            )
        return {
            **overall,
            "by_section": by_section,
            "pass": bool(overall["pass"] and all(row["pass"] for row in by_section.values())),
        }, row_pass

    def relative_metric(
        errors: np.ndarray, scales: np.ndarray, maximum: float
    ) -> tuple[Mapping[str, object], np.ndarray]:
        valid_scale = np.isfinite(scales) & (scales > 0.0)
        overall_ratios = np.full(len(errors), np.nan, dtype=np.float64)
        if np.any(locked & valid_scale):
            overall_ratios[locked] = errors[locked] / float(np.median(scales[locked & valid_scale]))
        overall, _ = summarize_threshold(
            overall_ratios,
            locked,
            maximum=maximum,
            maximum_outliers=maximum_registration_outliers,
        )
        section_ratios = np.full(len(errors), np.nan, dtype=np.float64)
        by_section = {}
        for section_id in sorted(set(section_ids[locked].astype(str).tolist())):
            selected = locked & (section_ids.astype(str) == section_id)
            selected_scale = selected & valid_scale
            if selected_scale.any():
                section_ratios[selected] = errors[selected] / float(
                    np.median(scales[selected_scale])
                )
            by_section[section_id], _ = summarize_threshold(
                section_ratios,
                selected,
                maximum=maximum,
                maximum_outliers=maximum_registration_outliers,
            )
        row_pass = np.isfinite(section_ratios) & (section_ratios <= maximum)
        return {
            **overall,
            "normalization_denominator": "median_geometry_scale_um",
            "by_section": by_section,
            "pass": bool(overall["pass"] and all(row["pass"] for row in by_section.values())),
        }, row_pass

    annotation_nucleus_report, annotation_nucleus_pass = absolute_metric(
        annotation_nucleus_um, float(contract["maximum_annotation_nucleus_p95_um"])
    )
    annotation_cell_report, annotation_cell_pass = absolute_metric(
        annotation_cell_um, float(contract["maximum_annotation_cell_p95_um"])
    )
    cell_nucleus_report, cell_nucleus_pass = absolute_metric(
        cell_nucleus_um, float(contract["maximum_cell_nucleus_p95_um"])
    )
    diameter_report, diameter_pass = relative_metric(
        annotation_nucleus_um,
        nucleus_diameter,
        float(contract["maximum_registration_nucleus_diameter_ratio_p95"]),
    )
    neighbor_report, neighbor_pass = relative_metric(
        annotation_nucleus_um,
        nearest_neighbor_um,
        float(contract["maximum_registration_nearest_neighbor_ratio_p95"]),
    )
    recomputed_registration_pass = (
        annotation_nucleus_pass
        & annotation_cell_pass
        & cell_nucleus_pass
        & diameter_pass
        & neighbor_pass
    )

    valid_area = (
        np.isfinite(nucleus_area_um2)
        & np.isfinite(cell_area_um2)
        & (nucleus_area_um2 > 0.0)
        & (cell_area_um2 > 0.0)
    )
    area_pass = (
        valid_area
        & (area_ratio >= float(contract["minimum_nucleus_cell_area_ratio"]))
        & (area_ratio <= float(contract["maximum_nucleus_cell_area_ratio"]))
    )
    recomputed_segmentation_pass = nucleus_inside_cell & area_pass

    def segmentation_summary(selected: np.ndarray) -> Mapping[str, object]:
        outside_fraction = float(np.mean(~nucleus_inside_cell[selected]))
        area_outlier_fraction = float(np.mean(~area_pass[selected]))
        return {
            "rows": int(np.count_nonzero(selected)),
            "nucleus_outside_cell_fraction": outside_fraction,
            "maximum_nucleus_outside_cell_fraction": float(
                contract["maximum_nucleus_outside_cell_fraction"]
            ),
            "area_ratio_outlier_fraction": area_outlier_fraction,
            "maximum_area_ratio_outlier_fraction": float(
                contract["maximum_segmentation_outlier_fraction"]
            ),
            "pass": bool(
                outside_fraction <= float(contract["maximum_nucleus_outside_cell_fraction"])
                and area_outlier_fraction
                <= float(contract["maximum_segmentation_outlier_fraction"])
            ),
        }

    segmentation_overall = segmentation_summary(locked)
    segmentation_by_section = {
        section_id: segmentation_summary(locked & (section_ids.astype(str) == section_id))
        for section_id in sorted(set(section_ids[locked].astype(str).tolist()))
    }
    segmentation_report = {
        **segmentation_overall,
        "by_section": segmentation_by_section,
        "pass": bool(
            segmentation_overall["pass"]
            and all(row["pass"] for row in segmentation_by_section.values())
        ),
    }

    padding = np.asarray(crop_padding_fractions, dtype=np.float64)
    if padding.ndim != 2 or padding.shape != (len(donor_ids), len(crop_ids)):
        raise ValueError("locked crop padding audit differs from the frozen crop family")
    valid_padding = np.isfinite(padding) & (padding >= 0.0) & (padding <= 1.0)
    recomputed_crop_pass = np.all(
        valid_padding & (padding <= float(contract["maximum_crop_padding_p95"])), axis=1
    )
    recomputed_nonmolecular_qc_pass = (
        recomputed_registration_pass & recomputed_segmentation_pass & recomputed_crop_pass
    )
    recomputed_qualified_qc_pass = recomputed_nonmolecular_qc_pass & target_qc_pass
    crop_reports = {}
    for column, crop_id in enumerate(crop_ids):
        values = padding[:, column]

        def crop_summary(selected: np.ndarray) -> Mapping[str, object]:
            valid = valid_padding[:, column]
            selected_values = values[selected & valid]
            p95 = float(np.quantile(selected_values, 0.95)) if len(selected_values) else None
            mostly = float(
                np.mean(
                    ~valid[selected] | (values[selected] > float(contract["mostly_padded_cutoff"]))
                )
            )
            return {
                "rows": int(np.count_nonzero(selected)),
                "padding_p95": p95,
                "mostly_padded_fraction": mostly,
                "pass": bool(
                    np.count_nonzero(selected & valid) == np.count_nonzero(selected)
                    and p95 is not None
                    and p95 <= float(contract["maximum_crop_padding_p95"])
                    and mostly <= float(contract["maximum_mostly_padded_fraction"])
                ),
            }

        overall = crop_summary(locked)
        by_section = {
            section_id: crop_summary(locked & (section_ids.astype(str) == section_id))
            for section_id in sorted(set(section_ids[locked].astype(str).tolist()))
        }
        crop_reports[str(crop_id)] = {
            **overall,
            "by_section": by_section,
            "pass": bool(overall["pass"] and all(row["pass"] for row in by_section.values())),
        }

    source_qc_matches = {
        "registration_qc_matches_recomputed": bool(
            np.array_equal(registration_qc_pass[locked], recomputed_registration_pass[locked])
        ),
        "segmentation_qc_matches_recomputed": bool(
            np.array_equal(segmentation_qc_pass[locked], recomputed_segmentation_pass[locked])
        ),
        "crop_qc_matches_recomputed": bool(
            np.array_equal(crop_qc_pass[locked], recomputed_crop_pass[locked])
        ),
        "locked_measurement_qc_matches_recomputed_conjunction": bool(
            np.array_equal(
                source_locked_measurement_qc_pass[locked],
                recomputed_nonmolecular_qc_pass[locked],
            )
        ),
    }
    distribution_checks = {
        "annotation_nucleus": bool(annotation_nucleus_report["pass"]),
        "annotation_cell": bool(annotation_cell_report["pass"]),
        "cell_nucleus": bool(cell_nucleus_report["pass"]),
        "nucleus_diameter_relative": bool(diameter_report["pass"]),
        "nearest_neighbor_relative": bool(neighbor_report["pass"]),
        "segmentation": bool(segmentation_report["pass"]),
        "crop_padding": bool(all(report["pass"] for report in crop_reports.values())),
        **source_qc_matches,
    }
    maximum_crop_padding = np.max(padding, axis=1)
    summaries = {
        "registration": {
            "annotation_to_nucleus_distance_um": annotation_nucleus_report,
            "annotation_to_cell_distance_um": annotation_cell_report,
            "native_cell_to_nucleus_distance_um": cell_nucleus_report,
            "annotation_error_over_median_nucleus_diameter": diameter_report,
            "annotation_error_over_median_nearest_neighbor_distance": neighbor_report,
        },
        "segmentation": segmentation_report,
        "crop_padding": crop_reports,
        "maximum_crop_padding_p95": float(np.quantile(maximum_crop_padding[locked], 0.95)),
        "rows_before_frozen_qc": int(np.count_nonzero(locked)),
        "rows_after_frozen_qc": int(np.count_nonzero(locked & recomputed_qualified_qc_pass)),
        "source_locked_measurement_qc_false_positive_rows": int(
            np.count_nonzero(
                locked & source_locked_measurement_qc_pass & ~recomputed_nonmolecular_qc_pass
            )
        ),
        "reliability_row_policy": (
            "recomputed_registration_and_segmentation_and_crop_and_target_qc"
        ),
    }
    normalized_a = normalize_split_counts(half_a_counts, library_sizes=half_a_library_sizes)
    normalized_b = normalize_split_counts(half_b_counts, library_sizes=half_b_library_sizes)
    donor_type_reports = {}
    reliable = 0
    denominator = 0
    for donor in locked_donors:
        for fine_type in supported_types:
            denominator += 1
            selected = (
                (donor_ids == donor) & (fine_type_ids == fine_type) & recomputed_qualified_qc_pass
            )
            report = feature_reliability(
                normalized_a[selected],
                normalized_b[selected],
                gene_ids,
                minimum_rows=int(contract["minimum_reliability_rows"]),
            )
            median = report["median_spearman_brown_reliability"]
            passes = bool(
                median is not None
                and median >= float(contract["minimum_within_fine_type_reliability"])
            )
            reliable += int(passes)
            donor_type_reports["%s|%s" % (donor, fine_type)] = {
                **report,
                "planned": True,
                "passes_frozen_reliability": passes,
            }
    reliable_fraction = float(reliable / max(denominator, 1))
    audit_pass = bool(
        all(distribution_checks.values())
        and reliable_fraction >= float(contract["minimum_locked_donor_type_reliability_fraction"])
    )
    return {
        "schema": "heir.locked_measurement_audit.v1",
        "selection_changes_forbidden": True,
        "coverage_denominator": "all_h_meas_supported_fine_types_and_locked_donors",
        "thresholds": dict(contract),
        "summaries": summaries,
        "distribution_checks": distribution_checks,
        "donor_type_reliability": donor_type_reports,
        "planned_donor_type_count": denominator,
        "reliable_donor_type_count": reliable,
        "reliable_donor_type_fraction": reliable_fraction,
        "pass": audit_pass,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--measurement-report", type=Path, default=None)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-observations", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--locked-test-output", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_path = args.study_manifest.expanduser().resolve()
    measurement_path = (
        args.measurement_report.expanduser().resolve()
        if args.measurement_report is not None
        else None
    )
    plan_path = args.plan.expanduser().resolve()
    source_path = args.source_observations.expanduser().resolve()
    development_path = args.development_output.expanduser().resolve()
    locked_path = (
        args.locked_test_output.expanduser().resolve()
        if args.locked_test_output is not None
        else None
    )
    input_paths = tuple(
        path
        for path in (manifest_path, measurement_path, plan_path, source_path)
        if path is not None
    )
    if any(not path.is_file() for path in input_paths) or len(set(input_paths)) != len(input_paths):
        raise ValueError("manifest, receipt, plan, and source must be distinct existing files")
    outputs = (development_path,) + ((locked_path,) if locked_path is not None else ())
    reject_output_input_collisions(outputs, input_paths, label="morphology-ridge preparation")
    if len(set(outputs)) != len(outputs):
        raise ValueError("morphology-ridge outputs must be distinct")

    plan = _load_plan(plan_path, source_path)
    source_sha = sha256_file(source_path)
    with np.load(source_path, allow_pickle=False) as source_identity_archive:
        cohort_id = str(
            _scalar(source_identity_archive, ("cohort_id",)) or plan.get("cohort_id", "")
        )
    if cohort_id == "HEST":
        manifest = StudyManifest.load(
            manifest_path,
            require_status="opened",
            verify_runtime=True,
            require_clean_runtime=True,
            verify_container_digest=True,
            repository_root=Path(__file__).resolve().parents[1],
        )
        opening = manifest.content["opening"]
        if "H-CELL" not in opening["permitted_claims"]:
            raise ValueError("opened study manifest does not permit the H-CELL claim")
        opening_receipt_sha256 = _sha256(
            opening["opening_receipt_sha256"], "opening.opening_receipt_sha256"
        )
    elif cohort_id == "HESCAPE":
        manifest = StudyManifest.load(manifest_path, require_status="locked")
        opening_receipt_sha256 = None
    else:
        raise ValueError("morphology preparation supports only HEST or HESCAPE")
    with np.load(source_path, allow_pickle=False) as archive:
        if str(_scalar(archive, ("schema_version",))) != str(plan["source_schema"]):
            raise ValueError("source observation schema differs from the source plan")
        cohort_id = str(_scalar(archive, ("cohort_id",)) or plan.get("cohort_id", ""))
        cohort_release = str(
            _scalar(archive, ("cohort_release",)) or plan.get("cohort_release", "")
        )
        if cohort_id == "HEST" and manifest.study_stage != "confirmatory_morphology":
            raise ValueError("HEST morphology artifacts require a confirmatory morphology manifest")
        observation_ids = np.asarray(_array(archive, ("observation_ids", "observation_id"))).astype(
            str
        )
        rows = len(observation_ids)
        source_type_labels = np.asarray(_array(archive, ("type_labels",)), dtype=np.int64)
        fine_type_values = _array(archive, ("fine_type_ids", "fine_type"), required=False)
        if fine_type_values is None:
            source_type_names = np.asarray(_array(archive, ("type_names",))).astype(str)
            fine_type_values = source_type_names[source_type_labels]
        registration_identity = _array(archive, ("registration_is_one_to_one",), required=False)
        if registration_identity is None:
            cardinality = _array(archive, ("registration_cardinality",))
            registration_identity = np.asarray(cardinality) == 1
        values: dict[str, np.ndarray] = {
            "observation_ids": observation_ids,
            "donor_ids": np.asarray(_array(archive, ("donor_ids", "donor_id"))).astype(str),
            "block_ids": np.asarray(_array(archive, ("block_ids", "block_id"))).astype(str),
            "roi_ids": np.asarray(_array(archive, ("roi_ids", "roi_id"))).astype(str),
            "section_ids": np.asarray(_array(archive, ("section_ids", "section_id"))).astype(str),
            "disease_states": np.asarray(
                _array(
                    archive,
                    ("disease_statuses", "disease_states", "disease_state"),
                )
            ).astype(str),
            "site_ids": np.asarray(_array(archive, ("site_ids", "site_id"))).astype(str),
            "batch_ids": np.asarray(_array(archive, ("batch_ids", "batch_id"))).astype(str),
            "type_labels": source_type_labels,
            "fine_type_ids": np.asarray(fine_type_values).astype(str),
            "molecular_targets": np.asarray(
                _array(
                    archive,
                    (
                        ("nucleus_molecular_targets",)
                        if cohort_id == "HEST"
                        else ("molecular_targets",)
                    ),
                ),
                dtype=np.float64,
            ),
            "coordinate_features": np.asarray(
                _array(archive, ("coordinate_features",)), dtype=np.float64
            ),
            "registration_is_one_to_one": np.asarray(
                registration_identity,
                dtype=np.bool_,
            ),
        }
        _lock_protection(manifest, cohort_id, values["donor_ids"], archive)
        if cohort_id == "HEST":
            if measurement_path is None:
                raise ValueError("HEST morphology preparation requires a passing H-MEAS report")
            measurement, genes, supported_types, measurement_sha = _target_selection(
                manifest, measurement_path
            )
            measurement_source_sha = _sha256(
                manifest.content["prerequisites"]["measurement_source_sha256"],
                "prerequisites.measurement_source_sha256",
            )
        else:
            if cohort_id != "HESCAPE":
                raise ValueError("morphology preparation supports only HEST or HESCAPE")
            source_genes = np.asarray(_array(archive, ("gene_ids", "ordered_gene_ids"))).astype(str)
            genes = tuple(source_genes.tolist())
            supported_types = tuple(
                value
                for value in np.asarray(_array(archive, ("type_names",))).astype(str).tolist()
                if value in set(values["fine_type_ids"].tolist())
            )
            measurement = {"coverage": {"support": {}, "pass": True}}
            measurement_sha = "0" * 64
            measurement_source_sha = source_sha
        source_gene_ids = np.asarray(_array(archive, ("gene_ids", "ordered_gene_ids"))).astype(str)
        gene_index = {gene: index for index, gene in enumerate(source_gene_ids.tolist())}
        if any(gene not in gene_index for gene in genes):
            raise ValueError("H-MEAS selected a gene absent from the registered source")
        selected_gene_indices = np.asarray([gene_index[gene] for gene in genes], dtype=np.int64)
        values["molecular_targets"] = values["molecular_targets"][:, selected_gene_indices]
        locked_audit_inputs = None
        if cohort_id == "HEST":
            source_thresholds = json.loads(
                str(_scalar(archive, ("locked_measurement_audit_thresholds_json",)))
            )
            if (
                source_thresholds != manifest.content["locked_measurement_audit"]
                or str(_scalar(archive, ("locked_measurement_audit_thresholds_sha256",)))
                != hashlib.sha256(
                    json.dumps(source_thresholds, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
            ):
                raise ValueError("confirmatory source changed the frozen locked measurement audit")
            locked_audit_inputs = {
                "half_a_counts": np.asarray(
                    _array(archive, ("nucleus_target_counts_half_a",)), dtype=np.float64
                )[:, selected_gene_indices],
                "half_b_counts": np.asarray(
                    _array(archive, ("nucleus_target_counts_half_b",)), dtype=np.float64
                )[:, selected_gene_indices],
                "half_a_library_sizes": np.asarray(
                    _array(archive, ("nucleus_library_size_half_a",)), dtype=np.float64
                ),
                "half_b_library_sizes": np.asarray(
                    _array(archive, ("nucleus_library_size_half_b",)), dtype=np.float64
                ),
                "source_locked_measurement_qc_pass": np.asarray(
                    _array(archive, ("locked_measurement_qc_pass",)), dtype=np.bool_
                ),
                "target_qc_pass": np.asarray(_array(archive, ("target_qc_pass",)), dtype=np.bool_),
                "registration_qc_pass": np.asarray(
                    _array(archive, ("registration_qc_pass",)), dtype=np.bool_
                ),
                "segmentation_qc_pass": np.asarray(
                    _array(archive, ("segmentation_qc_pass",)), dtype=np.bool_
                ),
                "crop_qc_pass": np.asarray(_array(archive, ("crop_qc_pass",)), dtype=np.bool_),
                "annotation_nucleus_um": np.asarray(
                    _array(archive, ("registration_distance_um",)), dtype=np.float64
                ),
                "annotation_cell_um": np.asarray(
                    _array(archive, ("annotation_cell_distance_um",)), dtype=np.float64
                ),
                "cell_nucleus_um": np.asarray(
                    _array(archive, ("cell_nucleus_centroid_distance_um",)), dtype=np.float64
                ),
                "nucleus_area_um2": np.asarray(
                    _array(archive, ("nucleus_area_um2",)), dtype=np.float64
                ),
                "nearest_neighbor_um": np.asarray(
                    _array(archive, ("local_density_features",)), dtype=np.float64
                )[:, 0],
                "nucleus_inside_cell": np.asarray(
                    _array(archive, ("nucleus_centroid_inside_cell",)), dtype=np.bool_
                ),
                "cell_area_um2": np.asarray(_array(archive, ("cell_area_um2",)), dtype=np.float64),
                "crop_ids": tuple(np.asarray(_array(archive, ("crop_ids",))).astype(str).tolist()),
                "crop_padding_fractions": np.asarray(
                    _array(archive, ("crop_padding_fractions",)), dtype=np.float64
                ),
            }
        type_index = {value: index for index, value in enumerate(supported_types)}
        remapped_labels = np.asarray(
            [type_index.get(value, -1) for value in values["fine_type_ids"]], dtype=np.int64
        )
        values["type_labels"] = remapped_labels
        image_source = _array(
            archive,
            ("image_features_by_crop_and_encoder", "image_features"),
            required=False,
        )
        crop_source = _array(archive, ("crop_ids",), required=False)
        primary_source = _scalar(archive, ("primary_crop_id",))
        if image_source is None:
            frozen = np.asarray(_array(archive, ("frozen_features",)), dtype=np.float64)
            image_source = frozen[:, None, :]
            primary_source = str(_scalar(archive, ("crop_role",)) or "regional_primary")
            crop_source = np.asarray([primary_source])
        image_tensor = np.asarray(image_source)
        crop_ids = tuple(np.asarray(crop_source).astype(str).tolist())
        primary_crop_id = str(primary_source)
        if image_tensor.ndim != 3 or image_tensor.shape[:2] != (rows, len(crop_ids)):
            raise ValueError("source full crop feature tensor is malformed")
        if cohort_id == "HESCAPE":
            crop_roles = tuple(str(_scalar(archive, ("crop_role",))) for _ in crop_ids)
            crop_families = tuple("regional_development_pilot" for _ in crop_ids)
        else:
            crop_roles, crop_families = _crop_metadata(plan, archive, crop_ids, require_source=True)
        primary_index = crop_ids.index(primary_crop_id)
        values["image_feature_tensor"] = image_tensor
        values["frozen_features"] = image_tensor[:, primary_index, :]
        named = {
            "stain": _named_matrix(archive, ("stain_features",), ("stain_feature_names",), rows),
            "composition": _named_matrix(
                archive, ("composition_features",), ("composition_feature_names",), rows
            ),
            "technical": _named_matrix(
                archive, ("technical_covariates",), ("technical_covariate_names",), rows
            ),
            "nuclear": _named_matrix(
                archive,
                ("nuclear_morphometric_features", "nuclear_morphometrics"),
                ("nuclear_morphometric_feature_names", "nuclear_morphometric_names"),
                rows,
            ),
            "cell": _named_matrix(
                archive,
                ("cell_morphometric_features", "cell_morphometrics"),
                ("cell_morphometric_feature_names", "cell_morphometric_names"),
                rows,
            ),
            "cellvit": _named_matrix(
                archive,
                ("cellvit_context_features", "cellvit_sensitivity_features"),
                ("cellvit_context_feature_names", "cellvit_sensitivity_feature_names"),
                rows,
            ),
            "density": _named_matrix(
                archive,
                ("local_density_features",),
                ("local_density_feature_names",),
                rows,
            ),
            "boundary": _named_matrix(
                archive, ("boundary_features",), ("boundary_feature_names",), rows
            ),
            "spatial": _named_matrix(
                archive,
                (
                    "spatial_control_features",
                    "spatial_features",
                    "coordinate_features",
                ),
                (
                    "spatial_control_feature_names",
                    "spatial_feature_names",
                    "coordinate_feature_names",
                ),
                rows,
            ),
        }
        reference_split_ids, pool_roles_by_split = _reference_splits(archive, rows, manifest)
        marker_values = _array(
            archive, ("fine_type_marker_gene_ids", "type_marker_gene_ids"), required=False
        )
        marker_genes = tuple(
            np.asarray(marker_values).astype(str).tolist() if marker_values is not None else ()
        )
        source_qc = np.ones(rows, dtype=np.bool_)
        for field_name in (
            "registration_qc_pass",
            "segmentation_qc_pass",
            "target_qc_pass",
            "crop_qc_pass",
            "locked_measurement_qc_pass",
        ):
            field = _array(archive, (field_name,), required=False)
            if field is not None:
                source_qc &= np.asarray(field, dtype=np.bool_)
        planned_values = _array(archive, ("planned_stratum_ids",), required=False)
        planned_strata = tuple(
            np.asarray(planned_values).astype(str).tolist()
            if planned_values is not None
            else sorted(
                {
                    "%s|%s|%s" % row
                    for row in zip(
                        values["donor_ids"].tolist(),
                        values["section_ids"].tolist(),
                        values["fine_type_ids"].tolist(),
                    )
                }
            )
        )
        planned_sha_value = _scalar(archive, ("planned_stratum_manifest_sha256",))
        planned_sha = (
            str(planned_sha_value)
            if planned_sha_value is not None
            else ordered_ids_sha256(sorted(planned_strata))
        )
        source_identity = {
            name: _scalar(archive, (name,)) or plan.get(name, "")
            for name in (
                "feature_space_id",
                "feature_checkpoint_sha256",
                "molecular_space_id",
                "label_source_sha256",
                "registration_source_sha256",
                "exclusion_policy_sha256",
                "registration_method",
                "encoder_name",
                "crop_scale",
                "assay",
                "observation_level",
                "target_construction",
            )
        }
        if cohort_id == "HEST":
            _validate_hest_source_contract(
                manifest,
                archive,
                plan=plan,
                source_gene_ids=source_gene_ids,
                selected_target_gene_ids=genes,
                marker_genes=marker_genes,
                crop_ids=crop_ids,
                crop_roles=crop_roles,
                crop_families=crop_families,
                named=named,
                source_identity=source_identity,
            )

    if len(set(values["observation_ids"].tolist())) != rows:
        raise ValueError("source observation identities are not unique")
    if any(len(array) != rows for array in values.values() if array.ndim >= 1):
        raise ValueError("source cell rows are misaligned")
    if not values["registration_is_one_to_one"].all():
        raise ValueError("source contains a non-one-to-one registration")
    if set(genes) & set(marker_genes):
        raise ValueError("measurement-qualified targets overlap fine-type marker genes")
    if cohort_id == "HESCAPE":
        if (
            tuple(str(value) for value in plan.get("development_donors", ()))
            != manifest.development_donors
            or tuple(plan.get("locked_test_donors", ()))
            or plan.get("analysis_scope") != "development_donors_only_hest_lock_unopened"
            or plan.get("reserved_donor_outcomes_loaded") is not False
        ):
            raise ValueError("HESCAPE source plan violates the protected development scope")
        if locked_path is not None:
            raise ValueError("HESCAPE is development-only and cannot produce a locked artifact")
        role_specs = (("development", manifest.development_donors, development_path),)
        evidence_scope = "development_pilot"
    else:
        if set(str(value) for value in plan.get("development_donors", ())) != set(
            manifest.development_donors
        ) or set(str(value) for value in plan.get("locked_test_donors", ())) != set(
            manifest.locked_test_donors
        ):
            raise ValueError("HEST source plan donor partitions differ from the locked manifest")
        if locked_path is None:
            raise ValueError("HEST requires a separate locked-test artifact output")
        role_specs = (
            ("development", manifest.development_donors, development_path),
            ("locked_test", manifest.locked_test_donors, locked_path),
        )
        evidence_scope = "internal_locked_hest"
    coverage = _mapping(measurement.get("coverage"), "H-MEAS coverage", {"pass"})
    measurement_coverage_summary = {
        str(name): value for name, value in coverage.items() if name != "support"
    }
    retained_support = coverage.get("support", {})
    retained_strata = {
        str(name)
        for name, report in retained_support.items()
        if isinstance(report, Mapping) and report.get("supported") is True
    }
    if cohort_id == "HESCAPE":
        retained_strata = set(planned_strata)
    row_strata = np.asarray(
        [
            "%s|%s|%s" % value
            for value in zip(
                values["donor_ids"].tolist(),
                values["section_ids"].tolist(),
                values["fine_type_ids"].tolist(),
            )
        ]
    )
    primary_roles = pool_roles_by_split[:, 0]
    if set(primary_roles.tolist()) != {"evaluation", "reference"}:
        raise ValueError("primary split must contain evaluation and reference pools")
    coverage_requirements = _mapping(
        manifest.content["coverage_requirements"], "coverage requirements", set()
    )
    balance_threshold_value = coverage_requirements.get("maximum_reference_evaluation_absolute_smd")
    balance_threshold = None if balance_threshold_value is None else float(balance_threshold_value)
    categorical_balance_threshold_value = coverage_requirements.get(
        "maximum_reference_evaluation_categorical_total_variation"
    )
    categorical_balance_threshold = (
        None
        if categorical_balance_threshold_value is None
        else float(categorical_balance_threshold_value)
    )
    balance_values = np.concatenate(
        (
            values["coordinate_features"],
            named["technical"][0],
            named["stain"][0],
            named["nuclear"][0],
            named["cell"][0],
            named["cellvit"][0],
            named["density"][0],
            named["boundary"][0],
            named["spatial"][0],
        ),
        axis=1,
    )
    balance_names = tuple(
        ["coordinate::%d" % index for index in range(values["coordinate_features"].shape[1])]
        + ["technical::%s" % value for value in named["technical"][1].tolist()]
        + ["stain::%s" % value for value in named["stain"][1].tolist()]
        + ["nuclear::%s" % value for value in named["nuclear"][1].tolist()]
        + ["cell::%s" % value for value in named["cell"][1].tolist()]
        + ["cellvit::%s" % value for value in named["cellvit"][1].tolist()]
        + ["density::%s" % value for value in named["density"][1].tolist()]
        + ["boundary::%s" % value for value in named["boundary"][1].tolist()]
        + ["spatial::%s" % value for value in named["spatial"][1].tolist()]
    )
    locked_measurement_audit = None
    if cohort_id == "HEST":
        if locked_audit_inputs is None:
            raise ValueError("confirmatory HEST source lacks locked measurement audit inputs")
        locked_measurement_audit = _locked_measurement_audit_report(
            contract=manifest.content["locked_measurement_audit"],
            donor_ids=values["donor_ids"],
            section_ids=values["section_ids"],
            fine_type_ids=values["fine_type_ids"],
            locked_donors=manifest.locked_test_donors,
            supported_types=supported_types,
            gene_ids=genes,
            **locked_audit_inputs,
        )
    minimum_reference = (
        int(
            manifest.content["coverage_requirements"][
                "minimum_reference_cells_per_donor_section_type"
            ]
        )
        if cohort_id == "HEST"
        else 1
    )
    minimum_evaluation = (
        int(
            manifest.content["coverage_requirements"][
                "minimum_evaluation_cells_per_donor_section_type"
            ]
        )
        if cohort_id == "HEST"
        else 1
    )
    for role, role_donors, output in role_specs:
        donor_mask = np.isin(values["donor_ids"], np.asarray(role_donors))
        frozen_type_mask = values["type_labels"] >= 0
        locked_support_audit = None
        if role == "locked_test":
            support_mask = np.zeros(rows, dtype=np.bool_)
            locked_support_audit = {}
            minimum_locked_donors = int(
                manifest.content["coverage_requirements"]["minimum_locked_donors_per_fine_type"]
            )
            primary_support = {}
            for donor in role_donors:
                donor_sections = sorted(
                    set(values["section_ids"][values["donor_ids"] == donor].astype(str).tolist())
                )
                for section_id in donor_sections:
                    for fine_type in supported_types:
                        group = (
                            (values["donor_ids"] == donor)
                            & (values["section_ids"] == section_id)
                            & (values["fine_type_ids"] == fine_type)
                            & frozen_type_mask
                            & source_qc
                        )
                        reference_count = int(
                            np.count_nonzero(group & (primary_roles == "reference"))
                        )
                        evaluation_count = int(
                            np.count_nonzero(group & (primary_roles == "evaluation"))
                        )
                        primary_support[(donor, section_id, fine_type)] = bool(
                            reference_count >= minimum_reference
                            and evaluation_count >= minimum_evaluation
                        )
                        locked_support_audit["%s|%s|%s" % (donor, section_id, fine_type)] = {
                            "planned": True,
                            "reference_rows_after_frozen_qc": reference_count,
                            "evaluation_rows_after_frozen_qc": evaluation_count,
                            "minimum_reference_rows": minimum_reference,
                            "minimum_evaluation_rows": minimum_evaluation,
                        }
            supported_donors_by_type = {
                fine_type: sum(
                    any(
                        primary_support[(donor, section_id, fine_type)]
                        for section_id in sorted(
                            set(
                                values["section_ids"][values["donor_ids"] == donor]
                                .astype(str)
                                .tolist()
                            )
                        )
                    )
                    for donor in role_donors
                )
                for fine_type in supported_types
            }
            for donor in role_donors:
                donor_sections = sorted(
                    set(values["section_ids"][values["donor_ids"] == donor].astype(str).tolist())
                )
                for section_id in donor_sections:
                    for fine_type in supported_types:
                        group = (
                            (values["donor_ids"] == donor)
                            & (values["section_ids"] == section_id)
                            & (values["fine_type_ids"] == fine_type)
                            & frozen_type_mask
                            & source_qc
                        )
                        type_support = supported_donors_by_type[fine_type]
                        evaluable = bool(
                            primary_support[(donor, section_id, fine_type)]
                            and type_support >= minimum_locked_donors
                        )
                        if evaluable:
                            support_mask |= group
                        locked_support_audit["%s|%s|%s" % (donor, section_id, fine_type)].update(
                            {
                                "locked_donors_with_primary_support": type_support,
                                "minimum_locked_donors_per_fine_type": minimum_locked_donors,
                                "fine_type_support_pass": type_support >= minimum_locked_donors,
                                "evaluable": evaluable,
                            }
                        )
        else:
            support_mask = frozen_type_mask & np.isin(
                row_strata, np.asarray(sorted(retained_strata))
            )
        qualified = donor_mask & source_qc & support_mask
        evaluation = qualified & (primary_roles == "evaluation")
        if set(values["donor_ids"][evaluation].tolist()) != set(role_donors):
            raise ValueError("every frozen donor needs qualified evaluation cells")
        evaluation_donors = values["donor_ids"][evaluation]
        evaluation_sections = values["section_ids"][evaluation]
        evaluation_labels = values["type_labels"][evaluation]
        references_by_split = np.zeros(
            (int(evaluation.sum()), len(reference_split_ids), len(genes)), dtype=np.float64
        )
        balance_by_split = {}
        reference_digests = []
        reference_membership_digests = {}
        for split_index, split_id in enumerate(reference_split_ids):
            split_roles = pool_roles_by_split[:, split_index]
            if not np.all(split_roles[evaluation] == "evaluation"):
                raise ValueError("frozen split changes primary evaluation rows")
            reference = qualified & (split_roles == "reference")
            for donor in role_donors:
                donor_evaluation = evaluation & (values["donor_ids"] == donor)
                for section_id in sorted(
                    set(values["section_ids"][donor_evaluation].astype(str).tolist())
                ):
                    section_reference = (
                        reference
                        & (values["donor_ids"] == donor)
                        & (values["section_ids"] == section_id)
                    )
                    section_evaluation = donor_evaluation & (values["section_ids"] == section_id)
                    reference_blocks = set(values["block_ids"][section_reference])
                    evaluation_blocks = set(values["block_ids"][section_evaluation])
                    if (
                        not reference_blocks
                        or not evaluation_blocks
                        or reference_blocks & evaluation_blocks
                    ):
                        raise ValueError(
                            "reference and evaluation blocks are not spatially disjoint"
                        )
                    evaluation_section_rows = (evaluation_donors == donor) & (
                        evaluation_sections == section_id
                    )
                    for type_index in sorted(
                        set(evaluation_labels[evaluation_section_rows].tolist())
                    ):
                        source_selected = section_reference & (values["type_labels"] == type_index)
                        target_selected = evaluation_section_rows & (
                            evaluation_labels == type_index
                        )
                        if np.count_nonzero(source_selected) < minimum_reference:
                            raise ValueError(
                                "an evaluated donor/section/type lacks the frozen independent "
                                "reference support"
                            )
                        references_by_split[target_selected, split_index] = values[
                            "molecular_targets"
                        ][source_selected].mean(axis=0)
            balance_by_split[split_id] = _balance_report(
                values,
                reference,
                evaluation,
                values["donor_ids"],
                values["type_labels"],
                supported_types,
                balance_values,
                balance_names,
                balance_threshold,
                categorical_balance_threshold,
            )
            reference_digests.append(
                _row_digest(
                    values["observation_ids"][reference],
                    values["molecular_targets"][reference],
                    role + "_reference_" + split_id,
                )
            )
            reference_membership_digests[split_id] = _identity_digest(
                values["observation_ids"][reference], role + "_reference_membership"
            )
        if len(set(reference_membership_digests.values())) != len(reference_split_ids):
            raise ValueError("frozen reference splits have identical memberships")
        role_planned = tuple(
            value
            for value in planned_strata
            if value.split("|", 1)[0] in set(role_donors)
            and value.rsplit("|", 1)[-1] in set(supported_types)
        )
        retained_role_strata = sorted(set(row_strata[evaluation].tolist()))
        coverage_audit = {
            "measurement_gate_coverage_summary": measurement_coverage_summary,
            "planned_stratum_ids": list(role_planned),
            "retained_evaluation_stratum_ids": retained_role_strata,
            "planned_strata": len(role_planned),
            "retained_evaluation_strata": len(retained_role_strata),
            "retained_fraction": float(len(retained_role_strata) / max(len(role_planned), 1)),
            "source_qc_retained_rows": int(np.count_nonzero(qualified)),
            "evaluation_rows": int(np.count_nonzero(evaluation)),
            "reference_membership_sha256_by_split": reference_membership_digests,
            "locked_measurement_audit": (
                locked_measurement_audit if role == "locked_test" else None
            ),
            "locked_support_audit": locked_support_audit,
        }
        primary_reference = references_by_split[:, 0, :]
        payload = {
            "schema_version": np.asarray(MorphologyRidgeDatasetArtifact.SCHEMA),
            "observation_ids": values["observation_ids"][evaluation],
            "donor_ids": evaluation_donors,
            "block_ids": values["block_ids"][evaluation],
            "roi_ids": values["roi_ids"][evaluation],
            "section_ids": values["section_ids"][evaluation],
            "disease_states": values["disease_states"][evaluation],
            "site_ids": values["site_ids"][evaluation],
            "batch_ids": values["batch_ids"][evaluation],
            "type_labels": evaluation_labels,
            "type_names": np.asarray(supported_types),
            "frozen_features": values["frozen_features"][evaluation],
            "image_feature_tensor": values["image_feature_tensor"][evaluation],
            "crop_ids": np.asarray(crop_ids),
            "crop_roles": np.asarray(crop_roles),
            "crop_comparison_families": np.asarray(crop_families),
            "primary_crop_id": np.asarray(primary_crop_id),
            "molecular_targets": values["molecular_targets"][evaluation],
            "reference_means": primary_reference,
            "reference_split_ids": np.asarray(reference_split_ids),
            "reference_means_by_split": references_by_split,
            "coordinate_features": values["coordinate_features"][evaluation],
            "stain_features": named["stain"][0][evaluation],
            "stain_feature_names": named["stain"][1],
            "composition_features": named["composition"][0][evaluation],
            "composition_feature_names": named["composition"][1],
            "technical_covariates": named["technical"][0][evaluation],
            "technical_covariate_names": named["technical"][1],
            "nuclear_morphometrics": named["nuclear"][0][evaluation],
            "nuclear_morphometric_names": named["nuclear"][1],
            "cell_morphometrics": named["cell"][0][evaluation],
            "cell_morphometric_names": named["cell"][1],
            "cellvit_context_features": named["cellvit"][0][evaluation],
            "cellvit_context_feature_names": named["cellvit"][1],
            "local_density_features": named["density"][0][evaluation],
            "local_density_feature_names": named["density"][1],
            "boundary_features": named["boundary"][0][evaluation],
            "boundary_feature_names": named["boundary"][1],
            "spatial_control_features": named["spatial"][0][evaluation],
            "spatial_control_feature_names": named["spatial"][1],
            "gene_ids": np.asarray(genes),
            "type_marker_gene_ids": np.asarray(marker_genes),
            "planned_stratum_ids": np.asarray(role_planned),
            "planned_stratum_manifest_sha256": np.asarray(planned_sha),
            "coverage_audit_json": _json(coverage_audit),
            "reference_evaluation_balance_json": _json(balance_by_split),
            "study_manifest_sha256": np.asarray(manifest.sha256),
            "opening_receipt_sha256": np.asarray(opening_receipt_sha256 or ""),
            "measurement_receipt_sha256": np.asarray(measurement_sha),
            "measurement_source_sha256": np.asarray(measurement_source_sha),
            "hypothesis_ids": np.asarray(manifest.hypothesis_ids),
            "scientific_scope": np.asarray(_scientific_scope(manifest, cohort_id)),
            "evidence_scope": np.asarray(evidence_scope),
            "authorizes_nucleus_intrinsic_claim": np.asarray(False),
            "feature_space_id": np.asarray(source_identity["feature_space_id"]),
            "feature_checkpoint_sha256": np.asarray(source_identity["feature_checkpoint_sha256"]),
            "molecular_space_id": np.asarray(source_identity["molecular_space_id"]),
            "reference_source_sha256": np.asarray(
                hashlib.sha256("".join(reference_digests).encode("ascii")).hexdigest()
            ),
            "label_source_sha256": np.asarray(source_identity["label_source_sha256"]),
            "target_source_sha256": np.asarray(
                _row_digest(
                    values["observation_ids"][evaluation],
                    values["molecular_targets"][evaluation],
                    role + "_evaluation",
                )
            ),
            "registration_source_sha256": np.asarray(source_identity["registration_source_sha256"]),
            "exclusion_policy_sha256": np.asarray(source_identity["exclusion_policy_sha256"]),
            "registration_method": np.asarray(source_identity["registration_method"]),
            "encoder_name": np.asarray(source_identity["encoder_name"]),
            "crop_scale": np.asarray(source_identity["crop_scale"]),
            "cohort_id": np.asarray(cohort_id),
            "cohort_release": np.asarray(cohort_release),
            "assay": np.asarray(source_identity["assay"]),
            "observation_level": np.asarray(source_identity["observation_level"]),
            "target_construction": np.asarray(source_identity["target_construction"]),
            "reference_pool_independent": np.asarray(True),
            "labels_independent_of_images": np.asarray(True),
            "registration_is_one_to_one": np.asarray(True),
        }
        _write_npz(output, payload)
        MorphologyRidgeDatasetArtifact.load_npz(output, role=role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
