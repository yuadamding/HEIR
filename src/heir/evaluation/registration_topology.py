"""Outcome-blind registration-topology audit for exposed HEST rows."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from heir.evaluation.blank_patch_receipt import array_sha256
from heir.features.base import sha256_file

RECEIPT_SCHEMA = "heir.hest_registration_topology.v1"
SOURCE_MEMBER_ALLOWLIST = (
    "observation_ids",
    "donor_ids",
    "section_ids",
    "type_labels",
    "pool_roles",
    "registration_quality_scores",
    "registration_quality_strata",
)
SOURCE_STRATA = ("best", "intermediate", "near_threshold", "failed")
SUPPORT_THRESHOLDS = (5, 10, 20)
BAND_DEFINITIONS = (
    ("best", ("best",)),
    ("intermediate", ("intermediate",)),
    ("near_threshold", ("near_threshold",)),
    ("registration_failed", ("failed",)),
    ("best_or_intermediate", ("best", "intermediate")),
    (
        "best_or_intermediate_or_near_threshold",
        ("best", "intermediate", "near_threshold"),
    ),
    ("all_rows", SOURCE_STRATA),
)
SELECTION_CANDIDATES = (
    ("Q1", "best"),
    ("Q2", "best_or_intermediate"),
    ("Q3", "best_or_intermediate_or_near_threshold"),
)
SELECTION_RULE = {
    "minimum_donors": 12,
    "minimum_evaluation_rows": 500,
    "minimum_evaluation_rows_per_included_donor": 20,
    "minimum_represented_types_per_included_donor": 2,
    "maximum_donor_fraction": 0.20,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class TopologyArrays:
    """The complete and only source payload allowed for this audit."""

    observation_ids: np.ndarray
    donor_ids: np.ndarray
    section_ids: np.ndarray
    type_labels: np.ndarray
    pool_roles: np.ndarray
    registration_quality_scores: np.ndarray
    registration_quality_strata: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.observation_ids)


class DirectNpzReader:
    """Open only explicitly allowlisted NPY payloads from one NPZ archive."""

    def __init__(
        self,
        path: Path,
        *,
        allowed_members: Sequence[str] = SOURCE_MEMBER_ALLOWLIST,
    ):
        allowed = tuple(str(value) for value in allowed_members)
        if not allowed or len(set(allowed)) != len(allowed) or any(not value for value in allowed):
            raise ValueError("source-member allowlist must contain unique nonempty names")
        self.path = path.expanduser().resolve()
        self.allowed_members = allowed
        self.members_opened: list[str] = []

    def read(self, name: str) -> np.ndarray:
        if name not in self.allowed_members:
            raise ValueError(f"source member {name!r} is outside the topology allowlist")
        member = f"{name}.npy"
        try:
            with zipfile.ZipFile(self.path) as archive:
                if archive.namelist().count(member) != 1:
                    raise ValueError(f"registered source must contain exactly one {member}")
                with archive.open(member) as handle:
                    value = np.lib.format.read_array(handle, allow_pickle=False)
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError("registered source is not a readable NPZ archive") from error
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"topology member {name} cannot use object dtype")
        self.members_opened.append(name)
        return array


def _strings(value: object, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != rows or array.dtype.kind not in {"S", "U"}:
        raise ValueError(f"topology member {name} must be a row-aligned string vector")
    result = array.astype(str)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"topology member {name} cannot contain empty strings")
    return result


def load_topology_arrays(reader: DirectNpzReader, *, expected_rows: int) -> TopologyArrays:
    """Load the exact seven-member topology contract and validate row alignment."""

    raw = {name: reader.read(name) for name in SOURCE_MEMBER_ALLOWLIST}
    observation_ids = _strings(raw["observation_ids"], "observation_ids", expected_rows)
    if len(set(observation_ids.tolist())) != expected_rows:
        raise ValueError("topology observation IDs must be unique")
    donor_ids = _strings(raw["donor_ids"], "donor_ids", expected_rows)
    section_ids = _strings(raw["section_ids"], "section_ids", expected_rows)
    pool_roles = _strings(raw["pool_roles"], "pool_roles", expected_rows)
    strata = _strings(
        raw["registration_quality_strata"],
        "registration_quality_strata",
        expected_rows,
    )
    observed_strata = set(strata.tolist())
    if not observed_strata or not observed_strata <= set(SOURCE_STRATA):
        raise ValueError("registration-quality strata differ from the frozen four categories")
    labels = np.asarray(raw["type_labels"])
    if labels.ndim != 1 or len(labels) != expected_rows or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("topology type labels must be a row-aligned integer vector")
    scores = np.asarray(raw["registration_quality_scores"], dtype=np.float64)
    if scores.ndim != 1 or len(scores) != expected_rows:
        raise ValueError("registration-quality scores must be a row-aligned numeric vector")
    nonfailed = strata != "failed"
    if (
        not np.isfinite(scores[nonfailed]).all()
        or np.any(scores[nonfailed] < 0.0)
        or np.any(scores[nonfailed] > 1.0)
        or np.any(
            np.isfinite(scores[~nonfailed])
            & ((scores[~nonfailed] <= 1.0) | (scores[~nonfailed] < 0.0))
        )
    ):
        raise ValueError("registration scores and frozen strata are inconsistent")
    return TopologyArrays(
        observation_ids=observation_ids,
        donor_ids=donor_ids,
        section_ids=section_ids,
        type_labels=labels.astype(np.int64),
        pool_roles=pool_roles,
        registration_quality_scores=scores,
        registration_quality_strata=strata,
    )


def _counts(values: np.ndarray) -> Mapping[str, int]:
    names, counts = np.unique(values.astype(str), return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def _donor_type_matrix(arrays: TopologyArrays, mask: np.ndarray) -> Mapping[str, object]:
    matrix: dict[str, dict[str, int]] = {}
    for donor, label in zip(arrays.donor_ids[mask], arrays.type_labels[mask]):
        row = matrix.setdefault(str(donor), {})
        key = str(int(label))
        row[key] = row.get(key, 0) + 1
    return {
        donor: {label: row[label] for label in sorted(row, key=int)}
        for donor, row in sorted(matrix.items())
    }


def _donor_section_type_matrix(arrays: TopologyArrays, mask: np.ndarray) -> Mapping[str, object]:
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    rows = zip(arrays.donor_ids[mask], arrays.section_ids[mask], arrays.type_labels[mask])
    for donor, section, label in rows:
        section_rows = matrix.setdefault(str(donor), {}).setdefault(str(section), {})
        key = str(int(label))
        section_rows[key] = section_rows.get(key, 0) + 1
    return {
        donor: {
            section: {label: labels[label] for label in sorted(labels, key=int)}
            for section, labels in sorted(sections.items())
        }
        for donor, sections in sorted(matrix.items())
    }


def _flatten_counts(matrix: Mapping[str, object], *, section_level: bool) -> list[int]:
    counts: list[int] = []
    for value in matrix.values():
        assert isinstance(value, Mapping)
        if section_level:
            for labels in value.values():
                assert isinstance(labels, Mapping)
                counts.extend(int(count) for count in labels.values())
        else:
            counts.extend(int(count) for count in value.values())
    return counts


def _support_summary(matrix: Mapping[str, object], *, section_level: bool) -> Mapping[str, object]:
    counts = _flatten_counts(matrix, section_level=section_level)
    return {
        str(threshold): {
            "supported_strata": int(sum(count >= threshold for count in counts)),
            "total_observed_strata": len(counts),
        }
        for threshold in SUPPORT_THRESHOLDS
    }


def _score_summary(scores: np.ndarray) -> Mapping[str, object]:
    finite = scores[np.isfinite(scores)]
    return {
        "finite_rows": len(finite),
        "nonfinite_rows": int(len(scores) - len(finite)),
        "minimum": float(finite.min()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
    }


def summarize_band(
    arrays: TopologyArrays,
    *,
    band_id: str,
    source_strata: Sequence[str],
    evaluation_mask: np.ndarray,
) -> Mapping[str, object]:
    """Summarize one frozen exclusive or cumulative band on evaluation rows."""

    mask = evaluation_mask & np.isin(arrays.registration_quality_strata, source_strata)
    rows = int(mask.sum())
    donor_counts = _counts(arrays.donor_ids[mask]) if rows else {}
    section_counts = _counts(arrays.section_ids[mask]) if rows else {}
    type_counts = _counts(arrays.type_labels[mask].astype(str)) if rows else {}
    donor_type = _donor_type_matrix(arrays, mask)
    donor_section_type = _donor_section_type_matrix(arrays, mask)
    types_by_donor = {
        donor: len(labels) for donor, labels in donor_type.items() if isinstance(labels, Mapping)
    }
    included_donors = sorted(
        donor
        for donor in donor_counts
        if donor_counts[donor] >= SELECTION_RULE["minimum_evaluation_rows_per_included_donor"]
        and types_by_donor[donor] >= SELECTION_RULE["minimum_represented_types_per_included_donor"]
    )
    excluded_donors = {
        donor: {
            "rows": donor_counts[donor],
            "represented_types": types_by_donor[donor],
            "below_minimum_rows": donor_counts[donor]
            < SELECTION_RULE["minimum_evaluation_rows_per_included_donor"],
            "below_minimum_types": types_by_donor[donor]
            < SELECTION_RULE["minimum_represented_types_per_included_donor"],
        }
        for donor in donor_counts
        if donor not in included_donors
    }
    included_counts = {donor: donor_counts[donor] for donor in included_donors}
    selection_rows = sum(included_counts.values())
    largest_donor = (
        min(included_counts, key=lambda donor: (-included_counts[donor], donor))
        if included_counts
        else None
    )
    worst_donor = (
        min(
            included_counts,
            key=lambda donor: (included_counts[donor], types_by_donor[donor], donor),
        )
        if included_counts
        else None
    )
    largest_fraction = included_counts[largest_donor] / selection_rows if largest_donor else None
    minimum_rows = min(included_counts.values()) if included_counts else 0
    minimum_types = (
        min(types_by_donor[donor] for donor in included_donors) if included_donors else 0
    )
    criteria = {
        "minimum_donors": {
            "observed": len(included_donors),
            "required": SELECTION_RULE["minimum_donors"],
            "passed": len(included_donors) >= SELECTION_RULE["minimum_donors"],
        },
        "minimum_evaluation_rows": {
            "observed": selection_rows,
            "required": SELECTION_RULE["minimum_evaluation_rows"],
            "passed": selection_rows >= SELECTION_RULE["minimum_evaluation_rows"],
        },
        "minimum_rows_per_included_donor": {
            "observed": minimum_rows,
            "required": SELECTION_RULE["minimum_evaluation_rows_per_included_donor"],
            "passed": minimum_rows >= SELECTION_RULE["minimum_evaluation_rows_per_included_donor"],
        },
        "minimum_types_per_included_donor": {
            "observed": minimum_types,
            "required": SELECTION_RULE["minimum_represented_types_per_included_donor"],
            "passed": minimum_types
            >= SELECTION_RULE["minimum_represented_types_per_included_donor"],
        },
        "maximum_donor_fraction": {
            "observed": largest_fraction,
            "required_maximum": SELECTION_RULE["maximum_donor_fraction"],
            "passed": largest_fraction is not None
            and largest_fraction <= SELECTION_RULE["maximum_donor_fraction"],
        },
    }
    return {
        "band_id": band_id,
        "population": "evaluation_pool_roles_only",
        "source_strata": list(source_strata),
        "rows": rows,
        "donors": len(donor_counts),
        "sections": len(section_counts),
        "types": len(type_counts),
        "counts_by_donor": donor_counts,
        "counts_by_section": section_counts,
        "counts_by_type_label": type_counts,
        "types_represented_by_donor": types_by_donor,
        "donor_inclusion_rule": {
            "minimum_rows": SELECTION_RULE["minimum_evaluation_rows_per_included_donor"],
            "minimum_represented_types": SELECTION_RULE[
                "minimum_represented_types_per_included_donor"
            ],
            "applied_before_donor_count_row_count_and_fraction_criteria": True,
        },
        "included_donor_ids": included_donors,
        "excluded_donors": excluded_donors,
        "selection_rows": selection_rows,
        "donor_type_count_matrix": donor_type,
        "donor_section_type_count_matrix": donor_section_type,
        "donor_type_support": _support_summary(donor_type, section_level=False),
        "donor_section_type_support": _support_summary(donor_section_type, section_level=True),
        "largest_contributing_donor": largest_donor,
        "largest_donor_fraction": largest_fraction,
        "worst_support_donor": (
            {
                "donor_id": worst_donor,
                "rows": included_counts[worst_donor],
                "represented_types": types_by_donor[worst_donor],
            }
            if worst_donor
            else None
        ),
        "registration_quality_scores": _score_summary(arrays.registration_quality_scores[mask]),
        "selection_criteria": criteria,
        "selection_eligible": all(bool(value["passed"]) for value in criteria.values()),
    }


def audit_topology(
    source: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> Mapping[str, object]:
    """Build the complete outcome-blind topology payload and frozen band choice."""

    resolved = source.expanduser().resolve()
    if sha256_file(resolved) != expected_sha256:
        raise ValueError("registered source SHA-256 differs from the frozen protocol")
    reader = DirectNpzReader(resolved)
    arrays = load_topology_arrays(reader, expected_rows=expected_rows)
    evaluation = np.char.startswith(arrays.pool_roles.astype(str), "evaluation")
    if not evaluation.any():
        raise ValueError("registered source contains no evaluation pool rows")
    bands = {
        band_id: summarize_band(
            arrays,
            band_id=band_id,
            source_strata=source_strata,
            evaluation_mask=evaluation,
        )
        for band_id, source_strata in BAND_DEFINITIONS
    }
    selected = None
    for quality_id, band_id in SELECTION_CANDIDATES:
        if bands[band_id]["selection_eligible"]:
            selected = {
                "quality_band_id": quality_id,
                "band_id": band_id,
                "source_strata": bands[band_id]["source_strata"],
                "included_donor_ids": bands[band_id]["included_donor_ids"],
                "selection_rows": bands[band_id]["selection_rows"],
            }
            break
    return {
        "registered_source_path": str(resolved),
        "registered_source_sha256": expected_sha256,
        "registered_source_bytes": resolved.stat().st_size,
        "source_rows": arrays.rows,
        "ordered_observation_ids_sha256": array_sha256(
            np.asarray(arrays.observation_ids, dtype="S")
        ),
        "source_member_allowlist": list(SOURCE_MEMBER_ALLOWLIST),
        "source_members_opened": reader.members_opened,
        "pool_role_counts": _counts(arrays.pool_roles),
        "evaluation_rows": int(evaluation.sum()),
        "registration_strata_counts_all_source": _counts(arrays.registration_quality_strata),
        "registration_strata_counts_evaluation": _counts(
            arrays.registration_quality_strata[evaluation]
        ),
        "support_thresholds_audited": list(SUPPORT_THRESHOLDS),
        "bands": bands,
        "selection_candidates_in_order": [
            {"quality_band_id": quality_id, "band_id": band_id}
            for quality_id, band_id in SELECTION_CANDIDATES
        ],
        "selection_rule": dict(SELECTION_RULE),
        "selected_band": selected,
        "selection_status": "passed" if selected else "no_frozen_cumulative_band_eligible",
        "molecular_outcomes_read": False,
        "reference_expression_read": False,
        "image_features_read": False,
        "predictions_read": False,
        "model_reports_read": False,
    }


def build_receipt(
    audit: Mapping[str, object],
    *,
    implementation: Mapping[str, object],
    runtime: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind the topology audit to committed code and a clean runtime."""

    if not implementation or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None or value == "0" * 64
        for value in implementation.values()
    ):
        raise ValueError("topology implementation identities must be nonzero SHA-256 values")
    if (
        runtime.get("git_worktree_dirty_at_start") is not False
        or not isinstance(runtime.get("git_commit"), str)
        or _GIT_COMMIT.fullmatch(str(runtime["git_commit"])) is None
    ):
        raise ValueError("registration-topology audit requires a clean Git worktree")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "scope": "outcome_blind_registration_topology_only",
        **audit,
        "implementation": dict(implementation),
        "runtime": dict(runtime),
        "outcome_blind_band_selection": True,
        "biological_experiment_run": False,
        "engineering_decision_available": False,
    }
    if receipt["source_members_opened"] != list(SOURCE_MEMBER_ALLOWLIST):
        raise ValueError("topology audit did not open exactly the frozen allowlist")
    if any(
        receipt[name] is not False
        for name in (
            "molecular_outcomes_read",
            "reference_expression_read",
            "image_features_read",
            "predictions_read",
            "model_reports_read",
        )
    ):
        raise ValueError("topology receipt is not outcome blind")
    return receipt


__all__ = [
    "BAND_DEFINITIONS",
    "DirectNpzReader",
    "RECEIPT_SCHEMA",
    "SELECTION_CANDIDATES",
    "SELECTION_RULE",
    "SOURCE_MEMBER_ALLOWLIST",
    "SOURCE_STRATA",
    "SUPPORT_THRESHOLDS",
    "TopologyArrays",
    "audit_topology",
    "build_receipt",
    "load_topology_arrays",
    "summarize_band",
]
