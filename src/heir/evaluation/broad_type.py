"""Fail-closed supervised development gate for broad H&E cell identity.

The gate intentionally consumes only frozen nucleus features and independently
supplied reviewer labels.  It does not derive labels from HEIR, RNA prototypes,
spot expression, or graph neighborhoods.  Its purpose is to establish that a
basic morphology-to-broad-type map works before molecular refinement is
interpreted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from heir.utils import resolve_device, set_seed, sha256_file

from .metrics import cell_type_metrics, risk_coverage_curve

BROAD_TYPE_PLAN_SCHEMA = "heir.broad_type_supervised_plan.v1"
BROAD_TYPE_REPORT_SCHEMA = "heir.broad_type_supervised_report.v1"
BROAD_TYPE_INSPECTION_SCHEMA = "heir.broad_type_supervised_inspection.v1"
LABEL_SCHEMA = "heir.independent_broad_type_labels.v1"
SPLIT_POLICIES = ("grouped_roi", "leave_one_donor_out")
MODEL_NAMES = ("balanced_logistic_probe", "balanced_frozen_feature_mlp")
REQUIRED_LABEL_COLUMNS = (
    "section_id",
    "nucleus_id",
    "donor_id",
    "roi_id",
    "compartment",
    "broad_type",
    "reviewer_confidence",
    "reviewer_count",
    "adjudication_status",
    "annotation_source",
    "independent_of_heir_predictions",
)


class BroadTypeGateBlocked(ValueError):
    """Raised when evidence is absent or violates the supervised gate contract."""


@dataclass(frozen=True)
class BroadTypeData:
    """Joined frozen features and independent labels for one evaluation task."""

    features: np.ndarray
    labels: np.ndarray
    section_ids: np.ndarray
    nucleus_ids: np.ndarray
    donor_ids: np.ndarray
    roi_ids: np.ndarray
    compartments: np.ndarray
    reviewer_confidence: np.ndarray
    classes: Tuple[str, ...]
    input_artifacts: Tuple[Mapping[str, object], ...]
    excluded_low_confidence: int
    excluded_reviewer_count: int


class _FrozenFeatureMLP(nn.Module):
    """Small classifier head; the supplied image features remain immutable."""

    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BroadTypeGateBlocked("%s must be an object" % name)
    return value


def _require_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list) or not value:
        raise BroadTypeGateBlocked("%s must be a non-empty list" % name)
    return value


def _resolve_path(base: Path, value: object, name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise BroadTypeGateBlocked("%s is absent" % name)
    path = Path(text).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _verified_file(
    base: Path,
    artifact: object,
    name: str,
) -> Tuple[Path, Dict[str, object]]:
    record = _require_mapping(artifact, name)
    path = _resolve_path(base, record.get("path"), "%s.path" % name)
    expected = str(record.get("sha256") or "").strip().lower()
    if len(expected) != 64:
        raise BroadTypeGateBlocked("%s.sha256 must contain 64 hexadecimal characters" % name)
    try:
        int(expected, 16)
    except ValueError as error:
        raise BroadTypeGateBlocked("%s.sha256 is not hexadecimal" % name) from error
    if not path.is_file():
        raise BroadTypeGateBlocked("%s is absent: %s" % (name, path))
    observed = sha256_file(path)
    if observed != expected:
        raise BroadTypeGateBlocked("%s SHA-256 mismatch" % name)
    return path, {"path": str(path), "sha256": observed}


def load_broad_type_plan(path: Path, *, require_ready: bool = True) -> Dict[str, Any]:
    """Load and structurally validate a prespecified supervised-gate plan."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise BroadTypeGateBlocked("broad-type plan is absent: %s" % source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BroadTypeGateBlocked("broad-type plan is not valid JSON") from error
    plan = dict(_require_mapping(payload, "broad-type plan"))
    if plan.get("schema_version") != BROAD_TYPE_PLAN_SCHEMA:
        raise BroadTypeGateBlocked("broad-type plan schema is invalid")
    if plan.get("status") not in {"ready", "labels_pending"}:
        raise BroadTypeGateBlocked("broad-type plan status must be ready or labels_pending")
    if require_ready and plan.get("status") != "ready":
        raise BroadTypeGateBlocked(
            "broad-type plan status must be ready; independently reviewed labels are pending"
        )
    seeds = [int(value) for value in _require_sequence(plan.get("seeds"), "seeds")]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise BroadTypeGateBlocked("at least three distinct non-negative seeds are required")
    plan["seeds"] = seeds
    graph = _require_mapping(plan.get("graph"), "graph")
    if graph.get("enabled") is not False:
        raise BroadTypeGateBlocked("the supervised broad-type gate requires graph.enabled=false")
    residual = _require_mapping(plan.get("molecular_residual"), "molecular_residual")
    if residual.get("enabled") is not False:
        raise BroadTypeGateBlocked(
            "the supervised broad-type gate requires molecular_residual.enabled=false"
        )
    label_policy = _require_mapping(plan.get("label_policy"), "label_policy")
    required_label_policy = {
        "generated_by_pipeline": False,
        "independent_of_heir_predictions_required": True,
        "roi_compartment_required": True,
        "reviewer_confidence_required": True,
    }
    for field, expected in required_label_policy.items():
        if label_policy.get(field) is not expected:
            raise BroadTypeGateBlocked("label_policy.%s must be %s" % (field, expected))
    tasks = _require_sequence(plan.get("tasks"), "tasks")
    identifiers = []
    for index, raw_task in enumerate(tasks):
        task = _require_mapping(raw_task, "tasks[%d]" % index)
        task_id = str(task.get("task_id") or "").strip()
        if not task_id or task_id in identifiers:
            raise BroadTypeGateBlocked("every task needs a unique non-empty task_id")
        identifiers.append(task_id)
        if task.get("split_policy") not in SPLIT_POLICIES:
            raise BroadTypeGateBlocked("task %s has an invalid split_policy" % task_id)
        _require_sequence(task.get("datasets"), "task %s datasets" % task_id)
    return plan


def inspect_broad_type_gate(plan_path: Path) -> Dict[str, object]:
    """Inspect readiness without reading feature matrices, fitting, or inventing labels."""

    source = Path(plan_path).expanduser().resolve()
    try:
        plan = load_broad_type_plan(source, require_ready=False)
    except BroadTypeGateBlocked as error:
        return {
            "schema_version": BROAD_TYPE_INSPECTION_SCHEMA,
            "status": "invalid_plan",
            "ready_to_run": False,
            "biological_success_claimed": False,
            "plan": {
                "path": str(source),
                "sha256": sha256_file(source) if source.is_file() else None,
            },
            "blockers": [str(error)],
        }
    base = source.parent
    task_reports: List[Dict[str, object]] = []
    all_blockers: List[str] = []
    for raw_task in plan["tasks"]:
        task = _require_mapping(raw_task, "task")
        task_id = str(task["task_id"])
        blockers: List[str] = []
        try:
            ontology_path, _ = _verified_file(
                base, task.get("ontology"), "task %s ontology" % task_id
            )
            load_broad_type_ontology(ontology_path)
        except BroadTypeGateBlocked as error:
            blockers.append(str(error))
        dataset_rows = []
        for raw_dataset in task["datasets"]:
            dataset = _require_mapping(raw_dataset, "task %s dataset" % task_id)
            section_id = str(dataset.get("section_id") or "").strip() or "unknown"
            dataset_blockers: List[str] = []
            # Labels are checked before large frozen feature files so a pending
            # annotation never triggers needless feature I/O.
            try:
                label_path, _ = _verified_file(
                    base,
                    dataset.get("labels"),
                    "task %s section %s independent labels" % (task_id, section_id),
                )
                with label_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    missing = sorted(set(REQUIRED_LABEL_COLUMNS) - set(reader.fieldnames or ()))
                    if missing:
                        raise BroadTypeGateBlocked(
                            "independent labels are missing required columns: %s"
                            % ", ".join(missing)
                        )
            except BroadTypeGateBlocked as error:
                dataset_blockers.append(str(error))
            try:
                _verified_file(
                    base,
                    dataset.get("features"),
                    "task %s section %s features" % (task_id, section_id),
                )
            except BroadTypeGateBlocked as error:
                dataset_blockers.append(str(error))
            blockers.extend(dataset_blockers)
            dataset_rows.append(
                {
                    "section_id": section_id,
                    "ready": not dataset_blockers,
                    "blockers": dataset_blockers,
                }
            )
        task_reports.append(
            {
                "task_id": task_id,
                "ready": not blockers,
                "split_policy": task["split_policy"],
                "datasets": dataset_rows,
                "blockers": blockers,
            }
        )
        all_blockers.extend("%s: %s" % (task_id, value) for value in blockers)
    if plan.get("status") != "ready":
        all_blockers.insert(0, "plan status is labels_pending")
    return {
        "schema_version": BROAD_TYPE_INSPECTION_SCHEMA,
        "status": "ready" if not all_blockers else "blocked_evidence",
        "ready_to_run": not all_blockers,
        "biological_success_claimed": False,
        "plan": {"path": str(source), "sha256": sha256_file(source)},
        "execution_contract": {
            "graph_used": False,
            "molecular_residual_used": False,
            "labels_generated_by_pipeline": False,
            "minimum_seed_count": 3,
        },
        "tasks": task_reports,
        "blockers": all_blockers,
    }


def load_broad_type_ontology(path: Path) -> Tuple[str, ...]:
    """Read a prespecified ordered broad ontology without deriving mappings."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(
            csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        )
    if not rows or "broad_type" not in rows[0]:
        raise BroadTypeGateBlocked("broad ontology needs a broad_type column")
    classes = tuple(str(row.get("broad_type") or "").strip() for row in rows)
    if any(not value for value in classes) or len(set(classes)) != len(classes):
        raise BroadTypeGateBlocked("broad ontology classes must be unique and non-empty")
    if len(classes) < 2:
        raise BroadTypeGateBlocked("broad ontology needs at least two classes")
    return classes


def _read_string_array(artifact: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in artifact:
            values = np.asarray(artifact[name])
            if values.ndim != 1:
                raise BroadTypeGateBlocked("feature artifact %s must be a vector" % name)
            return values.astype(str)
    raise BroadTypeGateBlocked("feature artifact is missing %s" % "/".join(names))


def _parse_true(value: object, name: str) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "1", "yes"}:
        raise BroadTypeGateBlocked("%s must be true for every reviewed label" % name)
    return True


def _load_dataset(
    *,
    task_id: str,
    dataset: Mapping[str, Any],
    classes: Tuple[str, ...],
    base: Path,
    minimum_confidence: float,
    minimum_reviewers: int,
    accepted_adjudication_statuses: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object], int, int]:
    section_id = str(dataset.get("section_id") or "").strip()
    declared_donor = str(dataset.get("donor_id") or "").strip()
    if not section_id or not declared_donor:
        raise BroadTypeGateBlocked("task %s datasets require section_id and donor_id" % task_id)
    features_path, feature_lineage = _verified_file(
        base,
        dataset.get("features"),
        "task %s section %s features" % (task_id, section_id),
    )
    labels_path, label_lineage = _verified_file(
        base,
        dataset.get("labels"),
        "task %s section %s independent labels" % (task_id, section_id),
    )
    with np.load(features_path, allow_pickle=False) as artifact:
        nucleus_ids = _read_string_array(artifact, ("nucleus_ids", "nucleus_id"))
        if "features" not in artifact:
            raise BroadTypeGateBlocked("feature artifact is missing features")
        features = np.asarray(artifact["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(nucleus_ids) or features.shape[1] < 1:
        raise BroadTypeGateBlocked("feature artifact arrays are misaligned")
    if not np.isfinite(features).all():
        raise BroadTypeGateBlocked("frozen features must be finite")
    if len(set(nucleus_ids.tolist())) != len(nucleus_ids):
        raise BroadTypeGateBlocked("feature artifact contains duplicate nucleus IDs")
    feature_lookup = {value: index for index, value in enumerate(nucleus_ids.tolist())}

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise BroadTypeGateBlocked("independent labels TSV has no header")
        missing = sorted(set(REQUIRED_LABEL_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise BroadTypeGateBlocked(
                "independent labels are missing required columns: %s" % ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise BroadTypeGateBlocked("independent labels TSV is empty")

    selected_features: List[np.ndarray] = []
    selected: Dict[str, List[object]] = {
        "labels": [],
        "section_ids": [],
        "nucleus_ids": [],
        "donor_ids": [],
        "roi_ids": [],
        "compartments": [],
        "reviewer_confidence": [],
    }
    observed_ids = set()
    excluded_confidence = 0
    excluded_reviewers = 0
    accepted = {str(value).strip().lower() for value in accepted_adjudication_statuses}
    for row_index, row in enumerate(rows, start=2):
        row_section = str(row.get("section_id") or "").strip()
        nucleus_id = str(row.get("nucleus_id") or "").strip()
        donor_id = str(row.get("donor_id") or "").strip()
        roi_id = str(row.get("roi_id") or "").strip()
        compartment = str(row.get("compartment") or "").strip()
        broad_type = str(row.get("broad_type") or "").strip()
        source = str(row.get("annotation_source") or "").strip()
        adjudication = str(row.get("adjudication_status") or "").strip().lower()
        _parse_true(
            row.get("independent_of_heir_predictions"),
            "independent_of_heir_predictions at label row %d" % row_index,
        )
        if row_section != section_id or donor_id != declared_donor:
            raise BroadTypeGateBlocked(
                "label row %d section/donor conflicts with its dataset manifest" % row_index
            )
        if not nucleus_id or nucleus_id in observed_ids:
            raise BroadTypeGateBlocked("label nucleus IDs must be unique within a dataset")
        observed_ids.add(nucleus_id)
        if nucleus_id not in feature_lookup:
            raise BroadTypeGateBlocked("reviewed nucleus %s has no frozen feature row" % nucleus_id)
        if not roi_id:
            raise BroadTypeGateBlocked("label row %d is missing roi_id" % row_index)
        if not compartment:
            raise BroadTypeGateBlocked("label row %d is missing compartment" % row_index)
        if not source or source.lower().startswith("heir"):
            raise BroadTypeGateBlocked(
                "label row %d lacks an independent annotation source" % row_index
            )
        if broad_type not in classes:
            raise BroadTypeGateBlocked(
                "label row %d broad_type is outside the prespecified ontology" % row_index
            )
        if adjudication not in accepted:
            raise BroadTypeGateBlocked("label row %d is not independently adjudicated" % row_index)
        try:
            confidence = float(str(row.get("reviewer_confidence") or ""))
            reviewers = int(str(row.get("reviewer_count") or ""))
        except ValueError as error:
            raise BroadTypeGateBlocked(
                "reviewer confidence/count must be numeric at label row %d" % row_index
            ) from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise BroadTypeGateBlocked("reviewer confidence must lie in [0,1]")
        if confidence < minimum_confidence:
            excluded_confidence += 1
            continue
        if reviewers < minimum_reviewers:
            excluded_reviewers += 1
            continue
        selected_features.append(features[feature_lookup[nucleus_id]])
        selected["labels"].append(classes.index(broad_type))
        selected["section_ids"].append(section_id)
        selected["nucleus_ids"].append(nucleus_id)
        selected["donor_ids"].append(donor_id)
        selected["roi_ids"].append(roi_id)
        selected["compartments"].append(compartment)
        selected["reviewer_confidence"].append(confidence)
    if not selected_features:
        raise BroadTypeGateBlocked("no labels remain after reviewer confidence/count filtering")
    arrays = {
        "features": np.stack(selected_features).astype(np.float32),
        "labels": np.asarray(selected["labels"], dtype=np.int64),
        "section_ids": np.asarray(selected["section_ids"], dtype=str),
        "nucleus_ids": np.asarray(selected["nucleus_ids"], dtype=str),
        "donor_ids": np.asarray(selected["donor_ids"], dtype=str),
        "roi_ids": np.asarray(selected["roi_ids"], dtype=str),
        "compartments": np.asarray(selected["compartments"], dtype=str),
        "reviewer_confidence": np.asarray(selected["reviewer_confidence"], dtype=np.float32),
    }
    lineage: Dict[str, object] = {
        "section_id": section_id,
        "donor_id": declared_donor,
        "features": feature_lineage,
        "labels": label_lineage,
        "label_schema": LABEL_SCHEMA,
        "reviewed_rows": len(rows),
        "eligible_rows": len(selected_features),
    }
    return arrays, lineage, excluded_confidence, excluded_reviewers


def load_broad_type_data(
    task: Mapping[str, Any],
    *,
    base: Path,
    defaults: Mapping[str, Any],
) -> BroadTypeData:
    """Load one task and enforce its label, ontology, and hash contracts."""

    task_id = str(task.get("task_id") or "").strip()
    ontology_path, ontology_lineage = _verified_file(
        base,
        task.get("ontology"),
        "task %s ontology" % task_id,
    )
    classes = load_broad_type_ontology(ontology_path)
    minimum_confidence = float(
        task.get("minimum_reviewer_confidence", defaults.get("minimum_reviewer_confidence", 0.70))
    )
    minimum_reviewers = int(task.get("minimum_reviewers", defaults.get("minimum_reviewers", 2)))
    if not 0.0 <= minimum_confidence <= 1.0 or minimum_reviewers < 1:
        raise BroadTypeGateBlocked("reviewer thresholds are invalid")
    statuses = task.get(
        "accepted_adjudication_statuses",
        defaults.get(
            "accepted_adjudication_statuses",
            ["adjudicated", "reviewer_agreement", "orthogonal_assay_confirmed"],
        ),
    )
    if not isinstance(statuses, list) or not statuses:
        raise BroadTypeGateBlocked("accepted_adjudication_statuses must be a non-empty list")
    arrays: Dict[str, List[np.ndarray]] = {
        name: []
        for name in (
            "features",
            "labels",
            "section_ids",
            "nucleus_ids",
            "donor_ids",
            "roi_ids",
            "compartments",
            "reviewer_confidence",
        )
    }
    lineage: List[Mapping[str, object]] = []
    excluded_confidence = 0
    excluded_reviewers = 0
    feature_dim: Optional[int] = None
    datasets = _require_sequence(task.get("datasets"), "task %s datasets" % task_id)
    for raw_dataset in datasets:
        dataset = _require_mapping(raw_dataset, "task %s dataset" % task_id)
        loaded, dataset_lineage, low_confidence, low_reviewers = _load_dataset(
            task_id=task_id,
            dataset=dataset,
            classes=classes,
            base=base,
            minimum_confidence=minimum_confidence,
            minimum_reviewers=minimum_reviewers,
            accepted_adjudication_statuses=statuses,
        )
        if feature_dim is None:
            feature_dim = int(loaded["features"].shape[1])
        elif loaded["features"].shape[1] != feature_dim:
            raise BroadTypeGateBlocked(
                "all task feature artifacts must share one feature dimension"
            )
        for name, values in loaded.items():
            arrays[name].append(values)
        lineage.append(dataset_lineage)
        excluded_confidence += low_confidence
        excluded_reviewers += low_reviewers
    joined = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    class_counts = np.bincount(joined["labels"], minlength=len(classes))
    minimum_class_count = int(
        task.get("minimum_class_count", defaults.get("minimum_class_count", 10))
    )
    if minimum_class_count < 1 or np.any(class_counts < minimum_class_count):
        missing = [classes[index] for index in np.flatnonzero(class_counts < minimum_class_count)]
        raise BroadTypeGateBlocked(
            "task %s lacks minimum reviewed support for: %s" % (task_id, ", ".join(missing))
        )
    return BroadTypeData(
        features=joined["features"],
        labels=joined["labels"],
        section_ids=joined["section_ids"],
        nucleus_ids=joined["nucleus_ids"],
        donor_ids=joined["donor_ids"],
        roi_ids=joined["roi_ids"],
        compartments=joined["compartments"],
        reviewer_confidence=joined["reviewer_confidence"],
        classes=classes,
        input_artifacts=tuple(lineage) + ({"ontology": ontology_lineage},),
        excluded_low_confidence=excluded_confidence,
        excluded_reviewer_count=excluded_reviewers,
    )


def _partition_rois(
    indices: np.ndarray,
    donor_ids: np.ndarray,
    roi_ids: np.ndarray,
    *,
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(
        ["%s\x1f%s" % (donor_ids[index], roi_ids[index]) for index in indices],
        dtype=str,
    )
    unique = np.unique(groups)
    if len(unique) < 2:
        raise BroadTypeGateBlocked("ROI-aware partitioning requires at least two ROI groups")
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    selected_count = min(len(shuffled) - 1, max(1, int(round(fraction * len(shuffled)))))
    selected_groups = shuffled[:selected_count]
    selected = indices[np.isin(groups, selected_groups)]
    remaining = indices[~np.isin(groups, selected_groups)]
    return remaining, selected


def make_broad_type_split(
    data: BroadTypeData,
    *,
    policy: str,
    seed: int,
    seed_position: int,
    calibration_fraction: float,
    test_fraction: float,
    test_donor: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Create donor-held-out or ROI-held-out train/calibration/test splits."""

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(data.labels), dtype=np.int64)
    donors = np.unique(data.donor_ids)
    donor_independent_test = False
    if policy == "leave_one_donor_out":
        if len(donors) < 2:
            raise BroadTypeGateBlocked("leave_one_donor_out requires at least two donors")
        if test_donor is None or str(test_donor) not in set(donors.tolist()):
            raise BroadTypeGateBlocked(
                "leave_one_donor_out requires an explicit observed test donor"
            )
        resolved_test_donor = str(test_donor)
        test = all_indices[data.donor_ids == resolved_test_donor]
        remaining = all_indices[data.donor_ids != resolved_test_donor]
        donor_independent_test = True
        remaining_donors = np.unique(data.donor_ids[remaining])
        if len(remaining_donors) >= 2:
            calibration_donor = remaining_donors[seed_position % len(remaining_donors)]
            calibration = remaining[data.donor_ids[remaining] == calibration_donor]
            train = remaining[data.donor_ids[remaining] != calibration_donor]
        else:
            train, calibration = _partition_rois(
                remaining,
                data.donor_ids,
                data.roi_ids,
                fraction=calibration_fraction,
                rng=rng,
            )
    elif policy == "grouped_roi":
        remaining, test = _partition_rois(
            all_indices,
            data.donor_ids,
            data.roi_ids,
            fraction=test_fraction,
            rng=rng,
        )
        train, calibration = _partition_rois(
            remaining,
            data.donor_ids,
            data.roi_ids,
            fraction=calibration_fraction,
            rng=rng,
        )
    else:
        raise BroadTypeGateBlocked("unsupported split policy: %s" % policy)
    partitions = {"train": train, "calibration": calibration, "test": test}
    if any(len(indices) == 0 for indices in partitions.values()):
        raise BroadTypeGateBlocked("a donor/ROI-aware split produced an empty partition")
    train_classes = np.unique(data.labels[train])
    if len(train_classes) != len(data.classes):
        absent = [
            data.classes[index]
            for index in sorted(set(range(len(data.classes))) - set(train_classes))
        ]
        raise BroadTypeGateBlocked("training split lacks ontology classes: %s" % ", ".join(absent))
    if len(np.unique(data.labels[calibration])) < 2 or len(np.unique(data.labels[test])) < 2:
        raise BroadTypeGateBlocked("calibration and test splits each require at least two classes")
    roi_sets = {
        name: {
            "%s::%s" % (data.donor_ids[index], data.roi_ids[index]) for index in indices.tolist()
        }
        for name, indices in partitions.items()
    }
    if any(
        roi_sets[left] & roi_sets[right]
        for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))
    ):
        raise RuntimeError("ROI leakage detected after grouped split construction")
    if donor_independent_test and (
        set(data.donor_ids[test].tolist())
        & set(data.donor_ids[np.concatenate((train, calibration))].tolist())
    ):
        raise RuntimeError("donor leakage detected after leave-one-donor-out construction")
    assignment_rows = sorted(
        (
            str(data.section_ids[index]),
            str(data.nucleus_ids[index]),
            name,
        )
        for name, indices in partitions.items()
        for index in indices.tolist()
    )
    split_report: Dict[str, object] = {
        "policy": policy,
        "donor_independent_test": donor_independent_test,
        "roi_disjoint": True,
        "assignment_sha256": _canonical_sha256(assignment_rows),
        "counts": {name: int(len(indices)) for name, indices in partitions.items()},
        "donors": {
            name: sorted(set(data.donor_ids[indices].tolist()))
            for name, indices in partitions.items()
        },
        "roi_groups": {name: sorted(values) for name, values in roi_sets.items()},
        "class_counts": {
            name: {
                data.classes[index]: int(count)
                for index, count in enumerate(
                    np.bincount(data.labels[indices], minlength=len(data.classes)).tolist()
                )
            }
            for name, indices in partitions.items()
        },
    }
    return train, calibration, test, split_report


def _balanced_sample_weights(
    labels: np.ndarray, confidence: np.ndarray, classes: int
) -> np.ndarray:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    if np.any(counts == 0):
        raise BroadTypeGateBlocked("class-balanced training requires every ontology class")
    class_weight = len(labels) / (classes * counts)
    weights = class_weight[labels] * confidence.astype(np.float64)
    return (weights / weights.mean()).astype(np.float32)


def _logistic_logits(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    logits = np.asarray(model.decision_function(features), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack((-0.5 * logits, 0.5 * logits))
    return logits


def _fit_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    calibration_features: np.ndarray,
    test_features: np.ndarray,
    *,
    seed: int,
    maximum_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        max_iter=maximum_iterations,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(scaler.transform(train_features), train_labels, sample_weight=train_weights)
    if len(model.classes_) != len(np.unique(train_labels)):
        raise RuntimeError("logistic probe did not retain all training classes")
    calibration_logits = _logistic_logits(model, scaler.transform(calibration_features))
    test_logits = _logistic_logits(model, scaler.transform(test_features))
    return (
        calibration_logits,
        test_logits,
        {
            "implementation": "sklearn.linear_model.LogisticRegression",
            "feature_standardization": "training_split_only",
            "class_balancing": "inverse_training_frequency_x_reviewer_confidence",
            "maximum_iterations": maximum_iterations,
            "iterations": np.asarray(model.n_iter_, dtype=int).tolist(),
        },
    )


def _fit_mlp(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    scaler = StandardScaler().fit(train_features)
    train_x = torch.as_tensor(scaler.transform(train_features), dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    weights = torch.as_tensor(train_weights, dtype=torch.float32, device=device)
    calibration_x = torch.as_tensor(
        scaler.transform(calibration_features), dtype=torch.float32, device=device
    )
    calibration_y = torch.as_tensor(calibration_labels, dtype=torch.long, device=device)
    test_x = torch.as_tensor(scaler.transform(test_features), dtype=torch.float32, device=device)
    model = _FrozenFeatureMLP(train_x.shape[1], hidden_dim, len(np.unique(train_labels))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = F.cross_entropy(model(train_x), train_y, reduction="none")
        loss = (losses * weights).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            calibration_loss = float(F.cross_entropy(model(calibration_x), calibration_y).cpu())
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("frozen-feature MLP did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        calibration_logits = model(calibration_x).cpu().numpy().astype(np.float64)
        test_logits = model(test_x).cpu().numpy().astype(np.float64)
    return (
        calibration_logits,
        test_logits,
        {
            "implementation": "torch_one_hidden_layer_classifier",
            "frozen_input_features": True,
            "graph_used": False,
            "feature_standardization": "training_split_only",
            "class_balancing": "inverse_training_frequency_x_reviewer_confidence",
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "best_epoch_by_calibration_nll": best_epoch,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
        },
    )


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)

    def objective(log_temperature: float) -> float:
        scaled = values / math.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(truth)), truth]))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    if not result.success:
        raise RuntimeError("temperature calibration failed")
    return float(math.exp(float(result.x)))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def _risk_summary(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    predicted = probabilities.argmax(axis=1)
    uncertainty = 1.0 - probabilities.max(axis=1)
    curve = risk_coverage_curve(labels, predicted, uncertainty)
    coverage = np.asarray(curve["coverage"], dtype=np.float64)
    risk = np.asarray(curve["risk"], dtype=np.float64)
    fixed: Dict[str, float] = {}
    for target in (0.50, 0.70, 0.80, 0.90, 1.00):
        index = int(np.searchsorted(coverage, target, side="left"))
        index = min(index, len(risk) - 1)
        fixed["%.2f" % target] = float(risk[index])
    return {
        "uncertainty": "one_minus_max_calibrated_probability",
        "aurc": float(np.sum(np.diff(coverage) * (risk[:-1] + risk[1:]) * 0.5)),
        "risk_at_coverage": fixed,
        "curve": curve,
    }


def _occupancy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    compartments: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, object]:
    predicted = probabilities.argmax(axis=1)

    def counts(values: np.ndarray) -> Dict[str, object]:
        observed = np.bincount(labels[values], minlength=len(classes))
        inferred = np.bincount(predicted[values], minlength=len(classes))
        return {
            "n": int(len(values)),
            "observed_counts": {classes[index]: int(value) for index, value in enumerate(observed)},
            "predicted_counts": {
                classes[index]: int(value) for index, value in enumerate(inferred)
            },
            "observed_fraction": {
                classes[index]: float(value / max(len(values), 1))
                for index, value in enumerate(observed)
            },
            "predicted_fraction": {
                classes[index]: float(value / max(len(values), 1))
                for index, value in enumerate(inferred)
            },
        }

    overall = counts(np.arange(len(labels), dtype=np.int64))
    predicted_counts = np.asarray(list(overall["predicted_counts"].values()), dtype=np.int64)  # type: ignore[union-attr]
    return {
        "overall": overall,
        "predicted_class_occupancy_fraction": float(
            np.count_nonzero(predicted_counts) / len(classes)
        ),
        "by_compartment": {
            str(compartment): counts(np.flatnonzero(compartments == compartment))
            for compartment in sorted(np.unique(compartments).tolist())
        },
    }


def _shuffle_features(
    features: np.ndarray,
    donor_ids: np.ndarray,
    partitions: Sequence[np.ndarray],
    *,
    seed: int,
) -> Tuple[np.ndarray, str]:
    """Break nucleus-feature pairing within each split/donor without leakage."""

    permutation = np.arange(len(features), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for partition in partitions:
        for donor in np.unique(donor_ids[partition]):
            group = partition[donor_ids[partition] == donor]
            if len(group) < 2:
                continue
            shuffled = group.copy()
            rng.shuffle(shuffled)
            if np.array_equal(group, shuffled):
                shuffled = np.roll(shuffled, 1)
            permutation[group] = shuffled
    return features[permutation], hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest()


def _evaluate_model(
    *,
    name: str,
    features: np.ndarray,
    data: BroadTypeData,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, object]:
    weights = _balanced_sample_weights(
        data.labels[train], data.reviewer_confidence[train], len(data.classes)
    )
    if name == "balanced_logistic_probe":
        calibration_logits, test_logits, training = _fit_logistic(
            features[train],
            data.labels[train],
            weights,
            features[calibration],
            features[test],
            seed=seed,
            maximum_iterations=int(model_config.get("logistic_maximum_iterations", 500)),
        )
    elif name == "balanced_frozen_feature_mlp":
        calibration_logits, test_logits, training = _fit_mlp(
            features[train],
            data.labels[train],
            weights,
            features[calibration],
            data.labels[calibration],
            features[test],
            seed=seed,
            hidden_dim=int(model_config.get("mlp_hidden_dim", 64)),
            epochs=int(model_config.get("mlp_epochs", 100)),
            learning_rate=float(model_config.get("mlp_learning_rate", 1.0e-3)),
            weight_decay=float(model_config.get("mlp_weight_decay", 1.0e-4)),
            device=device,
        )
    else:
        raise ValueError("unknown broad-type model: %s" % name)
    temperature = _fit_temperature(calibration_logits, data.labels[calibration])
    probabilities = _softmax(test_logits, temperature)
    metrics = cell_type_metrics(data.labels[test], probabilities)
    for key, value in list(metrics.items()):
        if isinstance(value, float) and not math.isfinite(value):
            metrics[key] = None
    return {
        "model": name,
        "training": training,
        "temperature_calibration": {
            "analysis_role": "calibration",
            "temperature": temperature,
            "n": int(len(calibration)),
        },
        "metrics": metrics,
        "occupancy": _occupancy(
            data.labels[test], probabilities, data.compartments[test], data.classes
        ),
        "risk_coverage": _risk_summary(data.labels[test], probabilities),
    }


def _summarize_task(
    runs: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    models: Dict[str, object] = {}
    checks: Dict[str, bool] = {}
    for model_name in MODEL_NAMES:
        model_runs = [run for run in runs if run["model"] == model_name]
        real = [run["real"] for run in model_runs]
        null = [run["image_shuffle_null"] for run in model_runs]
        real_f1 = np.asarray([item["metrics"]["macro_f1"] for item in real], dtype=np.float64)
        null_f1 = np.asarray([item["metrics"]["macro_f1"] for item in null], dtype=np.float64)
        real_ece = np.asarray([item["metrics"]["ece"] for item in real], dtype=np.float64)
        occupancy = np.asarray(
            [item["occupancy"]["predicted_class_occupancy_fraction"] for item in real],
            dtype=np.float64,
        )
        delta = real_f1 - null_f1
        minimum_f1 = float(thresholds.get("minimum_macro_f1", 0.65))
        minimum_delta = float(
            thresholds.get("minimum_image_shuffle_macro_f1_delta", 0.05)
        )
        maximum_ece = float(thresholds.get("maximum_ece", 0.10))
        minimum_occupancy = float(
            thresholds.get("minimum_predicted_class_occupancy_fraction", 0.75)
        )
        minimum_run_pass_fraction = float(
            thresholds.get("minimum_seed_donor_run_pass_fraction", 1.0)
        )
        run_pass = (
            (real_f1 >= minimum_f1)
            & (delta >= minimum_delta)
            & (real_ece < maximum_ece)
            & (occupancy >= minimum_occupancy)
        )
        seeds = np.asarray([int(run["seed"]) for run in model_runs], dtype=np.int64)
        unique_seeds = sorted(set(seeds.tolist()))
        seed_rows = []
        real_beats_shuffle_seed_count = 0
        for seed in unique_seeds:
            selected = seeds == seed
            seed_beats_shuffle = bool(np.all(delta[selected] > 0))
            real_beats_shuffle_seed_count += int(seed_beats_shuffle)
            seed_rows.append(
                {
                    "seed": seed,
                    "run_count": int(np.count_nonzero(selected)),
                    "minimum_macro_f1": float(real_f1[selected].min()),
                    "minimum_image_shuffle_macro_f1_delta": float(delta[selected].min()),
                    "maximum_ece": float(real_ece[selected].max()),
                    "minimum_predicted_class_occupancy_fraction": float(
                        occupancy[selected].min()
                    ),
                    "all_seed_donor_runs_pass": bool(np.all(run_pass[selected])),
                    "real_beats_shuffle_in_every_donor_fold": seed_beats_shuffle,
                }
            )
        models[model_name] = {
            "run_count": len(real),
            "seed_count": len(unique_seeds),
            "held_out_donor_count": len({str(run["held_out_test_donor"]) for run in model_runs}),
            "macro_f1_mean": float(real_f1.mean()),
            "macro_f1_standard_deviation": float(real_f1.std()),
            "minimum_macro_f1": float(real_f1.min()),
            "image_shuffle_macro_f1_mean": float(null_f1.mean()),
            "real_minus_image_shuffle_macro_f1_mean": float(delta.mean()),
            "minimum_real_minus_image_shuffle_macro_f1": float(delta.min()),
            "real_beats_shuffle_run_count": int(np.count_nonzero(delta > 0)),
            "real_beats_shuffle_seed_count": real_beats_shuffle_seed_count,
            "ece_mean": float(real_ece.mean()),
            "maximum_ece": float(real_ece.max()),
            "minimum_predicted_class_occupancy_fraction": float(occupancy.min()),
            "seed_donor_run_pass_fraction": float(run_pass.mean()),
            "minimum_required_seed_donor_run_pass_fraction": minimum_run_pass_fraction,
            "per_seed_stability": seed_rows,
        }
        checks["%s_macro_f1" % model_name] = bool(
            real_f1.mean() >= minimum_f1
        )
        checks["%s_image_shuffle_delta" % model_name] = bool(
            delta.mean() >= minimum_delta
        )
        checks["%s_ece" % model_name] = bool(
            real_ece.mean() < maximum_ece
        )
        checks["%s_occupancy" % model_name] = bool(
            occupancy.min() >= minimum_occupancy
        )
        checks["%s_seed_donor_stability" % model_name] = bool(
            len(unique_seeds) >= 3 and run_pass.mean() >= minimum_run_pass_fraction
        )
    return {"models": models}, {"checks": checks, "pass": bool(checks and all(checks.values()))}


def run_broad_type_gate(plan_path: Path, *, device_name: Optional[str] = None) -> Dict[str, object]:
    """Run all prespecified tasks, models, seeds, calibration, and shuffle nulls."""

    source = Path(plan_path).expanduser().resolve()
    plan = load_broad_type_plan(source)
    base = source.parent
    defaults = _require_mapping(plan.get("defaults", {}), "defaults")
    model_config = _require_mapping(plan.get("models", {}), "models")
    thresholds = _require_mapping(plan.get("gate_thresholds", {}), "gate_thresholds")
    split_config = _require_mapping(plan.get("splits", {}), "splits")
    requested_device = device_name or str(plan.get("device", "auto"))
    device = resolve_device(requested_device)
    task_reports: List[Dict[str, object]] = []
    claimed_nucleus_identities: Dict[Tuple[str, str], str] = {}
    for raw_task in plan["tasks"]:
        task = _require_mapping(raw_task, "task")
        task_id = str(task["task_id"])
        data = load_broad_type_data(task, base=base, defaults=defaults)
        identities = set(zip(data.section_ids.tolist(), data.nucleus_ids.tolist()))
        overlap = sorted(
            identity for identity in identities if identity in claimed_nucleus_identities
        )
        if overlap:
            section_id, nucleus_id = overlap[0]
            raise BroadTypeGateBlocked(
                "task %s reuses section/nucleus identity %s/%s already claimed by task %s"
                % (task_id, section_id, nucleus_id, claimed_nucleus_identities[overlap[0]])
            )
        claimed_nucleus_identities.update({identity: task_id for identity in identities})
        runs: List[Dict[str, object]] = []
        split_reports: List[Dict[str, object]] = []
        policy = str(task["split_policy"])
        held_out_donors: Tuple[Optional[str], ...] = (
            tuple(str(value) for value in np.unique(data.donor_ids).tolist())
            if policy == "leave_one_donor_out"
            else (None,)
        )
        for seed_position, seed in enumerate(plan["seeds"]):
            for held_out_donor in held_out_donors:
                set_seed(seed)
                train, calibration, test, split_report = make_broad_type_split(
                    data=data,
                    policy=policy,
                    seed=seed,
                    seed_position=seed_position,
                    calibration_fraction=float(split_config.get("calibration_fraction", 0.20)),
                    test_fraction=float(split_config.get("test_fraction", 0.20)),
                    test_donor=held_out_donor,
                )
                split_report["seed"] = seed
                split_report["held_out_test_donor"] = held_out_donor
                split_reports.append(split_report)
                shuffled, permutation_sha256 = _shuffle_features(
                    data.features,
                    data.donor_ids,
                    (train, calibration, test),
                    seed=seed + 1_000_003,
                )
                for model_name in MODEL_NAMES:
                    set_seed(seed)
                    real = _evaluate_model(
                        name=model_name,
                        features=data.features,
                        data=data,
                        train=train,
                        calibration=calibration,
                        test=test,
                        seed=seed,
                        model_config=model_config,
                        device=device,
                    )
                    set_seed(seed)
                    null = _evaluate_model(
                        name=model_name,
                        features=shuffled,
                        data=data,
                        train=train,
                        calibration=calibration,
                        test=test,
                        seed=seed,
                        model_config=model_config,
                        device=device,
                    )
                    runs.append(
                        {
                            "seed": seed,
                            "held_out_test_donor": held_out_donor,
                            "model": model_name,
                            "real": real,
                            "image_shuffle_null": null,
                            "image_shuffle": {
                                "policy": "feature_rows_permuted_within_split_and_donor_v1",
                                "permutation_sha256": permutation_sha256,
                            },
                        }
                    )
        summary, gate = _summarize_task(runs, thresholds=thresholds)
        task_reports.append(
            {
                "task_id": task_id,
                "analysis_scope": task.get("analysis_scope", "supervised_development"),
                "ontology_classes": list(data.classes),
                "n_reviewed_eligible": int(len(data.labels)),
                "n_donors": int(len(np.unique(data.donor_ids))),
                "n_roi_groups": int(len(set(zip(data.donor_ids.tolist(), data.roi_ids.tolist())))),
                "n_compartments": int(len(np.unique(data.compartments))),
                "reviewer_filtering": {
                    "excluded_low_confidence": data.excluded_low_confidence,
                    "excluded_low_reviewer_count": data.excluded_reviewer_count,
                },
                "input_artifacts": list(data.input_artifacts),
                "splits": split_reports,
                "runs": runs,
                "summary": summary,
                "gate": gate,
            }
        )
    task_gate_results = {str(task["task_id"]): bool(task["gate"]["pass"]) for task in task_reports}
    return {
        "schema_version": BROAD_TYPE_REPORT_SCHEMA,
        "status": "complete",
        "claim_scope": "supervised_development_only",
        "biological_success_claimed": False,
        "plan": {"path": str(source), "sha256": sha256_file(source)},
        "execution": {
            "device": str(device),
            "graph_used": False,
            "molecular_residual_used": False,
            "models": list(MODEL_NAMES),
            "seeds": list(plan["seeds"]),
            "class_balancing": "inverse_training_frequency_x_reviewer_confidence",
            "calibration_split_used": True,
        },
        "label_contract": {
            "schema": LABEL_SCHEMA,
            "independent_of_heir_predictions_required": True,
            "reviewer_confidence_required": True,
            "reviewer_count_required": True,
            "roi_compartment_required": True,
            "labels_generated_by_pipeline": False,
        },
        "tasks": task_reports,
        "overall_gate": {
            "task_pass": task_gate_results,
            "pass": bool(task_gate_results and all(task_gate_results.values())),
            "interpretation": (
                "Threshold result for the prespecified supervised development gate; "
                "it is not evidence of molecular-refinement success."
            ),
        },
    }


def blocked_broad_type_report(plan_path: Path, error: BaseException) -> Dict[str, object]:
    """Create an explicit non-success report when label evidence is unavailable."""

    source = Path(plan_path).expanduser().resolve()
    plan_hash = sha256_file(source) if source.is_file() else None
    return {
        "schema_version": BROAD_TYPE_REPORT_SCHEMA,
        "status": "blocked_evidence",
        "claim_scope": "supervised_development_only",
        "biological_success_claimed": False,
        "plan": {"path": str(source), "sha256": plan_hash},
        "blockers": [str(error)],
        "label_contract": {
            "schema": LABEL_SCHEMA,
            "independent_of_heir_predictions_required": True,
            "reviewer_confidence_required": True,
            "reviewer_count_required": True,
            "roi_compartment_required": True,
            "labels_generated_by_pipeline": False,
        },
        "overall_gate": {"pass": False},
    }
