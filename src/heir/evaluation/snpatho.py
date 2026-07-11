"""Locked snPATHO Visium benchmark and prespecified negative controls.

This module is an evaluator, never a trainer.  It accepts already-frozen HEIR
predictions, matched snRNA references, and versioned locked truth artifacts.
The spatial target is loaded only after prediction/reference provenance checks,
and target hashes are rejected if they overlap any model input hash exposed by
the prediction bundle.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from heir.baselines import type_mean_prediction
from heir.data import RNAReference, SpatialTruthArtifact
from heir.expression import EXPRESSION_SPACE_ID, EXPRESSION_TARGET_SUM
from heir.inference import PredictionBundle
from heir.utils import atomic_json_dump, sha256_file

from .benchmark import (
    BenchmarkReport,
    BenchmarkStatus,
    DonorMetricRow,
    MethodComparisonRow,
    build_benchmark_report,
    write_benchmark_tsv,
)
from .metrics import expression_metrics

SNPATHO_PLAN_SCHEMA = "heir.snpatho_benchmark_plan.v1"
SNPATHO_REPORT_SCHEMA = "heir.snpatho_benchmark.v1"
INFERENCE_TELEMETRY_SCHEMA = "heir.inference_telemetry.v1"
SNPATHO_SECTION_IDS = frozenset({"4066", "4399", "4411"})

HEIR_METHOD = "heir"
PSEUDOBULK_METHOD = "matched_snrna_pseudobulk"
SHUFFLE_METHOD = "heir_spatial_shuffle"
TYPE_MEAN_METHOD = "matched_type_mean"
BENCHMARK_METHODS = (HEIR_METHOD, PSEUDOBULK_METHOD, SHUFFLE_METHOD, TYPE_MEAN_METHOD)

EXPRESSION_METRICS = (
    "median_gene_pearson",
    "median_gene_spearman",
    "median_gene_mse",
    "mean_location_cosine",
    "fraction_genes_defined",
)
METRIC_DIRECTIONS = {
    "median_gene_pearson": True,
    "median_gene_spearman": True,
    "median_gene_mse": False,
    "mean_location_cosine": True,
    "fraction_genes_defined": True,
}


@dataclass(frozen=True)
class SnPathoCase:
    """Paths for one frozen donor-level evaluation case."""

    section_id: str
    checkpoint_sha256: str
    predictions: Path
    predictions_sha256: str
    truth: Path
    truth_sha256: str
    matched_reference: Path
    matched_reference_sha256: str
    telemetry: Path
    telemetry_sha256: str

    def __post_init__(self) -> None:
        if not self.section_id.strip():
            raise ValueError("section_id cannot be empty")
        for name in (
            "checkpoint_sha256",
            "predictions_sha256",
            "truth_sha256",
            "matched_reference_sha256",
            "telemetry_sha256",
        ):
            if not _is_sha256(str(getattr(self, name))):
                raise ValueError("case %s must be a lowercase SHA-256 digest" % name)
        for name in ("predictions", "truth", "matched_reference", "telemetry"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(str(path))
            object.__setattr__(self, name, path)
        self.validate_artifact_hashes()

    def validate_artifact_hashes(self, *, include_truth: bool = True) -> None:
        """Recheck frozen case inputs, optionally leaving locked truth unopened."""

        pairs = [
            ("predictions", self.predictions, self.predictions_sha256),
            ("matched_reference", self.matched_reference, self.matched_reference_sha256),
            ("telemetry", self.telemetry, self.telemetry_sha256),
        ]
        if include_truth:
            pairs.append(("truth", self.truth, self.truth_sha256))
        for name, path, expected in pairs:
            if sha256_file(path) != expected:
                raise ValueError("%s SHA-256 differs from the frozen snPATHO plan" % name)

    def validate_truth_hash(self) -> None:
        """Open the locked artifact only after non-target prediction checks pass."""

        if sha256_file(self.truth) != self.truth_sha256:
            raise ValueError("truth SHA-256 differs from the frozen snPATHO plan")


@dataclass(frozen=True)
class SnPathoBenchmarkPlan:
    """Frozen checkpoint/panel plus the locked cases to open once."""

    source_path: Path
    source_sha256: str
    checkpoint_sha256: str
    gene_panel: Path
    gene_panel_sha256: str
    frozen_model_version: str
    cases: Tuple[SnPathoCase, ...]

    def __post_init__(self) -> None:
        source = Path(self.source_path).expanduser().resolve()
        panel = Path(self.gene_panel).expanduser().resolve()
        if not source.is_file() or not panel.is_file():
            raise FileNotFoundError(str(source if not source.is_file() else panel))
        if not _is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if sha256_file(source) != self.source_sha256:
            raise ValueError("benchmark plan changed while it was being loaded")
        if self.checkpoint_sha256 and not _is_sha256(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if not _is_sha256(self.gene_panel_sha256):
            raise ValueError("gene_panel_sha256 must be a lowercase SHA-256 digest")
        if sha256_file(panel) != self.gene_panel_sha256:
            raise ValueError("gene panel SHA-256 differs from the frozen snPATHO plan")
        if not self.frozen_model_version.strip():
            raise ValueError("frozen_model_version cannot be empty")
        if not self.cases:
            raise ValueError("snPATHO benchmark plan must contain cases")
        identifiers = [case.section_id for case in self.cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("snPATHO benchmark case section IDs must be unique")
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "gene_panel", panel)


@dataclass(frozen=True)
class InferenceTelemetry:
    """Measured prediction runtime loaded from a hash-bound sidecar."""

    wall_seconds: float
    peak_cuda_memory_bytes: int
    device_type: str
    device_name: str
    mixed_precision: bool
    nuclei: int
    prediction_sha256: str

    @classmethod
    def from_json(cls, path: Path, prediction: Path) -> "InferenceTelemetry":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        required = {
            "schema",
            "wall_seconds",
            "peak_cuda_memory_bytes",
            "device_type",
            "device_name",
            "mixed_precision",
            "nuclei",
            "prediction_sha256",
        }
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError("inference telemetry sidecar is incomplete")
        if payload["schema"] != INFERENCE_TELEMETRY_SCHEMA:
            raise ValueError("unsupported inference telemetry schema")
        if str(payload["prediction_sha256"]) != sha256_file(prediction):
            raise ValueError("inference telemetry is bound to a different prediction")
        wall = float(payload["wall_seconds"])
        peak = int(payload["peak_cuda_memory_bytes"])
        nuclei = int(payload["nuclei"])
        device_type = str(payload["device_type"])
        device_name = str(payload["device_name"])
        mixed_precision = payload["mixed_precision"]
        if not np.isfinite(wall) or wall <= 0:
            raise ValueError("telemetry wall_seconds must be finite and positive")
        if peak < 0 or nuclei <= 0:
            raise ValueError("telemetry memory/nuclei values are invalid")
        if device_type not in {"cpu", "cuda"} or not device_name.strip():
            raise ValueError("telemetry device identity is invalid")
        if not isinstance(mixed_precision, bool):
            raise ValueError("telemetry mixed_precision must be boolean")
        if not _is_sha256(str(payload["prediction_sha256"])):
            raise ValueError("telemetry prediction_sha256 is invalid")
        return cls(
            wall_seconds=wall,
            peak_cuda_memory_bytes=peak,
            device_type=device_type,
            device_name=device_name,
            mixed_precision=mixed_precision,
            nuclei=nuclei,
            prediction_sha256=str(payload["prediction_sha256"]),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "available": True,
            "wall_seconds": self.wall_seconds,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "peak_cuda_memory_gib": self.peak_cuda_memory_bytes / float(1024**3),
            "device_type": self.device_type,
            "device_name": self.device_name,
            "mixed_precision": self.mixed_precision,
            "nuclei": self.nuclei,
            "nuclei_per_second": self.nuclei / self.wall_seconds,
            "prediction_sha256": self.prediction_sha256,
        }


@dataclass(frozen=True)
class SnPathoCaseResult:
    section_id: str
    donor_id: str
    spots_total: int
    spots_evaluated: int
    nuclei_total: int
    methods: Mapping[str, Mapping[str, Optional[float]]]
    per_gene: Mapping[str, Mapping[str, object]]
    coverage: Mapping[str, float]
    telemetry: Mapping[str, object]
    provenance: Mapping[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "section_id": self.section_id,
            "donor_id": self.donor_id,
            "spots_total": self.spots_total,
            "spots_evaluated": self.spots_evaluated,
            "nuclei_total": self.nuclei_total,
            "methods": {name: dict(values) for name, values in self.methods.items()},
            "per_gene": {name: dict(values) for name, values in self.per_gene.items()},
            "coverage": dict(self.coverage),
            "telemetry": dict(self.telemetry),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class SnPathoBenchmarkResult:
    plan_sha256: str
    checkpoint_sha256_by_section: Mapping[str, str]
    gene_panel_sha256: str
    frozen_model_version: str
    cases: Tuple[SnPathoCaseResult, ...]
    benchmark: BenchmarkReport
    seed: int
    schema_version: str = SNPATHO_REPORT_SCHEMA

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "isolation": {
                "target_spatial_truth_role": "locked_validation",
                "target_spatial_expression_used_for_training": False,
                "target_histology_used_for_training": True,
                "target_spatial_metadata_used_for_capture_filtering": True,
                "checkpoint_sha256_by_section": dict(self.checkpoint_sha256_by_section),
                "gene_panel_sha256": self.gene_panel_sha256,
                "frozen_model_version": self.frozen_model_version,
                "plan_sha256": self.plan_sha256,
            },
            "seed": self.seed,
            "cases": [case.to_dict() for case in self.cases],
            "aggregate": self.benchmark.to_dict(),
        }


def load_snpatho_plan(path: Path) -> SnPathoBenchmarkPlan:
    """Load a plan and resolve every artifact relative to the plan file."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {
        "schema_version",
        "gene_panel",
        "gene_panel_sha256",
        "frozen_model_version",
        "cases",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("snPATHO benchmark plan is incomplete")
    if payload["schema_version"] != SNPATHO_PLAN_SCHEMA:
        raise ValueError("unsupported snPATHO benchmark plan schema")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ValueError("snPATHO benchmark plan cases must be a non-empty list")
    global_checkpoint = str(payload.get("checkpoint_sha256", ""))
    root = source.parent

    def resolve(value: object) -> Path:
        candidate = Path(str(value)).expanduser()
        return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    cases: List[SnPathoCase] = []
    for index, case in enumerate(payload["cases"]):
        case_required = {
            "section_id",
            "checkpoint_sha256",
            "predictions",
            "predictions_sha256",
            "truth",
            "truth_sha256",
            "matched_reference",
            "matched_reference_sha256",
            "telemetry",
            "telemetry_sha256",
        }
        if not isinstance(case, dict) or not case_required.issubset(case):
            raise ValueError("snPATHO benchmark plan case %d is incomplete" % index)
        cases.append(
            SnPathoCase(
                section_id=str(case["section_id"]),
                checkpoint_sha256=str(case["checkpoint_sha256"]),
                predictions=resolve(case["predictions"]),
                predictions_sha256=str(case["predictions_sha256"]),
                truth=resolve(case["truth"]),
                truth_sha256=str(case["truth_sha256"]),
                matched_reference=resolve(case["matched_reference"]),
                matched_reference_sha256=str(case["matched_reference_sha256"]),
                telemetry=resolve(case["telemetry"]),
                telemetry_sha256=str(case["telemetry_sha256"]),
            )
        )
    return SnPathoBenchmarkPlan(
        source_path=source,
        source_sha256=sha256_file(source),
        checkpoint_sha256=global_checkpoint,
        gene_panel=resolve(payload["gene_panel"]),
        gene_panel_sha256=str(payload["gene_panel_sha256"]),
        frozen_model_version=str(payload["frozen_model_version"]),
        cases=tuple(cases),
    )


def run_snpatho_benchmark(
    plan: SnPathoBenchmarkPlan,
    *,
    seed: int = 17,
    iterations: int = 10000,
    confidence: float = 0.95,
    minimum_donors: int = 2,
    require_complete: bool = True,
) -> SnPathoBenchmarkResult:
    """Score frozen predictions and three target-free baselines."""

    if sha256_file(plan.source_path) != plan.source_sha256:
        raise ValueError("frozen snPATHO plan changed after loading")
    if sha256_file(plan.gene_panel) != plan.gene_panel_sha256:
        raise ValueError("frozen snPATHO gene panel changed after loading")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    observed_sections = {case.section_id for case in plan.cases}
    if require_complete and observed_sections != SNPATHO_SECTION_IDS:
        raise ValueError("complete snPATHO benchmark requires sections 4066, 4399, and 4411")
    panel_genes = _gene_panel(plan.gene_panel)
    panel_sha256 = sha256_file(plan.gene_panel)
    case_results: List[SnPathoCaseResult] = []
    donor_rows: List[DonorMetricRow] = []
    for case in sorted(plan.cases, key=lambda value: value.section_id):
        result, rows = _score_case(
            case,
            plan=plan,
            panel_genes=panel_genes,
            panel_sha256=panel_sha256,
            seed=_stable_seed(seed, case.section_id),
        )
        case_results.append(result)
        donor_rows.extend(rows)

    base_report = build_benchmark_report(
        donor_rows,
        iterations=iterations,
        confidence=confidence,
        minimum_donors=minimum_donors,
        seed=seed,
    )
    comparison_rows = [row for row in donor_rows if row.metric in EXPRESSION_METRICS]
    comparisons: List[MethodComparisonRow] = []
    for baseline in (PSEUDOBULK_METHOD, SHUFFLE_METHOD, TYPE_MEAN_METHOD):
        compared = build_benchmark_report(
            comparison_rows,
            baseline_method=baseline,
            metric_directions=METRIC_DIRECTIONS,
            iterations=iterations,
            confidence=confidence,
            minimum_donors=minimum_donors,
            seed=seed,
        )
        comparisons.extend(
            row
            for row in compared.comparisons
            if row.method == HEIR_METHOD and row.baseline_method == baseline
        )
    benchmark = BenchmarkReport(
        donor_metrics=base_report.donor_metrics,
        summaries=base_report.summaries,
        comparisons=tuple(comparisons),
        confidence=base_report.confidence,
        iterations=base_report.iterations,
        minimum_donors=base_report.minimum_donors,
        seed=base_report.seed,
    )
    return SnPathoBenchmarkResult(
        plan_sha256=plan.source_sha256,
        checkpoint_sha256_by_section={
            case.section_id: case.checkpoint_sha256 for case in plan.cases
        },
        gene_panel_sha256=panel_sha256,
        frozen_model_version=plan.frozen_model_version,
        cases=tuple(case_results),
        benchmark=benchmark,
        seed=seed,
    )


def write_snpatho_benchmark(
    result: SnPathoBenchmarkResult,
    *,
    json_path: Path,
    tsv_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    output = Path(json_path).expanduser().resolve()
    tabular_destination = None if tsv_path is None else Path(tsv_path).expanduser().resolve()
    if tabular_destination is not None and tabular_destination == output:
        raise ValueError("snPATHO JSON and TSV outputs must be different paths")
    atomic_json_dump(result.to_dict(), output)
    tabular = None
    if tabular_destination is not None:
        tabular = write_benchmark_tsv(result.benchmark, tabular_destination)
    return output, tabular


def _score_case(
    case: SnPathoCase,
    *,
    plan: SnPathoBenchmarkPlan,
    panel_genes: Tuple[str, ...],
    panel_sha256: str,
    seed: int,
) -> Tuple[SnPathoCaseResult, List[DonorMetricRow]]:
    # Validate frozen non-target artifacts first.  The locked target is opened
    # only after these checks pass.
    case.validate_artifact_hashes(include_truth=False)
    prediction = PredictionBundle.from_npz(case.predictions)
    prediction.validate(require_provenance=True)
    with np.load(case.matched_reference, allow_pickle=False) as archive:
        if "__version__" not in archive or int(np.asarray(archive["__version__"]).item()) < 3:
            raise ValueError("snPATHO benchmark requires RNAReference v3 full-library denominators")
    reference = RNAReference.load_npz(case.matched_reference)
    if prediction.checkpoint_sha256 != case.checkpoint_sha256:
        raise ValueError("prediction checkpoint differs from the frozen snPATHO plan")
    if prediction.model_version != plan.frozen_model_version:
        raise ValueError("prediction model version differs from the frozen snPATHO plan")
    if tuple(str(value) for value in prediction.gene_names.tolist()) != panel_genes:
        raise ValueError("prediction genes differ from the frozen snPATHO panel")
    if tuple(str(value) for value in reference.gene_ids.tolist()) != panel_genes:
        raise ValueError("matched reference genes differ from the frozen snPATHO panel")
    if prediction.expression_space_id != EXPRESSION_SPACE_ID:
        raise ValueError("prediction does not use canonical expression space")

    case.validate_truth_hash()
    truth = SpatialTruthArtifact.from_npz(case.truth)
    if truth.analysis_role != "locked_validation" or truth.cohort_id != "snpatho_seq":
        raise ValueError("snPATHO truth must be a snpatho_seq locked_validation artifact")
    if truth.section_id != case.section_id:
        raise ValueError("truth section differs from its benchmark plan case")
    if tuple(str(value) for value in truth.gene_names.tolist()) != panel_genes:
        raise ValueError("truth genes differ from the frozen snPATHO panel")
    if truth.expression_space_id != prediction.expression_space_id:
        raise ValueError("truth and prediction expression spaces differ")
    if not np.array_equal(truth.nucleus_ids, prediction.nucleus_ids.astype(str)):
        raise ValueError("truth and prediction nucleus identities/order differ")
    if prediction.donor_id != truth.donor_id or prediction.sample_id != truth.specimen_id:
        raise ValueError("prediction donor/sample identities differ from locked truth")
    if set(reference.donor_ids.tolist()) != {truth.donor_id}:
        raise ValueError("matched reference donor identity differs from locked truth")
    if set(reference.sample_ids.tolist()) != {truth.specimen_id}:
        raise ValueError("matched reference sample identity differs from locked truth")
    truth_panel_hashes = truth.source_sha256[truth.source_roles == "canonical_gene_panel"].tolist()
    if truth_panel_hashes != [panel_sha256]:
        raise ValueError("truth was created with a different canonical gene panel")
    _validate_target_isolation(prediction, reference, truth)

    spot_index = truth.nucleus_spot_index
    evaluable = np.bincount(spot_index[spot_index >= 0], minlength=len(truth.spot_ids)) > 0
    if not evaluable.any():
        raise ValueError("locked truth has no evaluable spots")
    gene_order = _exact_gene_order(prediction.gene_names, truth.gene_names)
    heir_spots, mass = _aggregate_log_expression(
        prediction.expression_mean[:, gene_order],
        spot_index,
        len(truth.spot_ids),
    )
    if not np.array_equal(mass > 0, evaluable):
        raise RuntimeError("HEIR spot aggregation disagrees with truth assignment")

    reference_expression = _reference_log_expression(reference)
    linear_pseudobulk = np.asarray(np.expm1(reference_expression).mean(axis=0), dtype=np.float64)
    pseudobulk_spots = np.repeat(
        np.log1p(linear_pseudobulk)[None, :], len(truth.spot_ids), axis=0
    ).astype(np.float32)

    rng = np.random.default_rng(seed)
    assigned_cells = np.flatnonzero(spot_index >= 0)
    permuted_expression = np.asarray(prediction.expression_mean[:, gene_order]).copy()
    permuted_expression[assigned_cells] = permuted_expression[rng.permutation(assigned_cells)]
    shuffled_spots, shuffled_mass = _aggregate_log_expression(
        permuted_expression,
        spot_index,
        len(truth.spot_ids),
    )
    if not np.array_equal(shuffled_mass, mass):
        raise RuntimeError("spatial shuffle changed spot mass")

    type_mean_cells = _type_mean_cells(
        reference_expression,
        reference.cell_type_labels,
        prediction.type_probabilities,
        prediction.type_names,
    )
    type_mean_spots = None
    if type_mean_cells is not None:
        type_mean_spots, type_mass = _aggregate_log_expression(
            type_mean_cells,
            spot_index,
            len(truth.spot_ids),
        )
        if not np.array_equal(type_mass, mass):
            raise RuntimeError("type-mean aggregation changed spot mass")

    predictions = {
        HEIR_METHOD: heir_spots,
        PSEUDOBULK_METHOD: pseudobulk_spots,
        SHUFFLE_METHOD: shuffled_spots,
    }
    if type_mean_spots is not None:
        predictions[TYPE_MEAN_METHOD] = type_mean_spots
    methods: Dict[str, Mapping[str, Optional[float]]] = {}
    per_gene: Dict[str, Mapping[str, object]] = {}
    rows: List[DonorMetricRow] = []
    for method in BENCHMARK_METHODS:
        if method not in predictions:
            methods[method] = {metric: None for metric in EXPRESSION_METRICS}
            per_gene[method] = {
                "gene_names": list(panel_genes),
                "pearson": [None] * len(panel_genes),
                "spearman": [None] * len(panel_genes),
                "mse": [None] * len(panel_genes),
            }
            rows.extend(
                DonorMetricRow(
                    cohort_id=truth.cohort_id,
                    donor_id=truth.donor_id,
                    method=method,
                    metric=metric,
                    value=None,
                    status=BenchmarkStatus.MISSING,
                    reason="matched reference labels do not overlap the prediction ontology",
                    n_observations=int(evaluable.sum()),
                )
                for metric in EXPRESSION_METRICS
            )
            continue
        with warnings.catch_warnings():
            # Constant pseudobulk intentionally has undefined spatial
            # correlation; retain it as an explicit missing endpoint without
            # emitting a misleading runtime warning.
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            raw_metrics = expression_metrics(
                predictions[method][evaluable],
                truth.observed_expression[evaluable],
            )
        selected = {metric: _finite_or_none(raw_metrics[metric]) for metric in EXPRESSION_METRICS}
        methods[method] = selected
        per_gene[method] = {
            "gene_names": list(panel_genes),
            "pearson": [_finite_or_none(value) for value in raw_metrics["per_gene_pearson"]],
            "spearman": [_finite_or_none(value) for value in raw_metrics["per_gene_spearman"]],
            "mse": [_finite_or_none(value) for value in raw_metrics["per_gene_mse"]],
        }
        for metric, value in selected.items():
            rows.append(
                DonorMetricRow(
                    cohort_id=truth.cohort_id,
                    donor_id=truth.donor_id,
                    method=method,
                    metric=metric,
                    value=value,
                    status=(BenchmarkStatus.OK if value is not None else BenchmarkStatus.MISSING),
                    reason=("" if value is not None else "metric is undefined for this donor"),
                    n_observations=int(evaluable.sum()),
                )
            )

    assigned = spot_index >= 0
    non_abstained = ~prediction.abstain.astype(bool)
    assigned_non_abstained = assigned & non_abstained
    covered_spots = (
        np.bincount(spot_index[assigned_non_abstained], minlength=len(truth.spot_ids)) > 0
    )
    coverage = {
        "cell_coverage": float(non_abstained.mean()),
        "assigned_cell_coverage": float(assigned_non_abstained.sum() / max(int(assigned.sum()), 1)),
        "spot_coverage": float((covered_spots & evaluable).sum() / evaluable.sum()),
        "abstention_rate": float(prediction.abstain.mean()),
    }
    for metric, value in coverage.items():
        rows.append(
            DonorMetricRow(
                cohort_id=truth.cohort_id,
                donor_id=truth.donor_id,
                method=HEIR_METHOD,
                metric=metric,
                value=value,
                n_observations=len(prediction.nucleus_ids),
            )
        )

    telemetry_payload: Mapping[str, object] = {
        "available": False,
        "reason": "no hash-bound inference telemetry sidecar was supplied",
    }
    telemetry_metrics = {
        "inference_wall_seconds": None,
        "peak_cuda_memory_gib": None,
        "nuclei_per_second": None,
    }
    telemetry = InferenceTelemetry.from_json(case.telemetry, case.predictions)
    if telemetry.nuclei != len(prediction.nucleus_ids):
        raise ValueError("inference telemetry nucleus count differs from prediction")
    telemetry_payload = telemetry.to_dict()
    telemetry_metrics = {
        "inference_wall_seconds": telemetry.wall_seconds,
        "peak_cuda_memory_gib": telemetry.peak_cuda_memory_bytes / float(1024**3),
        "nuclei_per_second": telemetry.nuclei / telemetry.wall_seconds,
    }
    for metric, value in telemetry_metrics.items():
        rows.append(
            DonorMetricRow(
                cohort_id=truth.cohort_id,
                donor_id=truth.donor_id,
                method=HEIR_METHOD,
                metric=metric,
                value=value,
                status=(BenchmarkStatus.OK if value is not None else BenchmarkStatus.MISSING),
                reason=("" if value is not None else "inference telemetry was not supplied"),
                n_observations=len(prediction.nucleus_ids),
            )
        )

    provenance = {
        "predictions": sha256_file(case.predictions),
        "truth": sha256_file(case.truth),
        "matched_reference": sha256_file(case.matched_reference),
    }
    provenance["telemetry"] = sha256_file(case.telemetry)
    return (
        SnPathoCaseResult(
            section_id=case.section_id,
            donor_id=truth.donor_id,
            spots_total=len(truth.spot_ids),
            spots_evaluated=int(evaluable.sum()),
            nuclei_total=len(prediction.nucleus_ids),
            methods=methods,
            per_gene=per_gene,
            coverage=coverage,
            telemetry=telemetry_payload,
            provenance=provenance,
        ),
        rows,
    )


def _validate_target_isolation(
    prediction: PredictionBundle,
    reference: RNAReference,
    truth: SpatialTruthArtifact,
) -> None:
    locked_hashes = {
        digest
        for digest, role in zip(truth.source_sha256.tolist(), truth.source_roles.tolist())
        if role.startswith("locked_spatial") or role == "manifest_spatial_source"
    }
    prediction_hashes = {
        prediction.checkpoint_sha256,
        prediction.prototype_sha256,
        prediction.histology_sha256,
        prediction.program_sha256,
        prediction.ood_sha256,
    } - {""}
    overlap = sorted(locked_hashes & prediction_hashes)
    if overlap:
        raise ValueError("locked target spatial artifact overlaps prediction inputs")
    if reference.source_count_sha256 and reference.source_count_sha256 in locked_hashes:
        raise ValueError("matched reference was derived from locked target spatial counts")


def _reference_log_expression(reference: RNAReference) -> np.ndarray:
    matrix = sparse.csr_matrix(reference.counts, dtype=np.float64)
    library = np.asarray(reference.library_sizes, dtype=np.float64)
    if library.shape != (matrix.shape[0],) or np.any(library <= 0):
        raise ValueError("matched reference needs positive full-transcriptome library sizes")
    normalized = matrix.multiply((EXPRESSION_TARGET_SUM / library)[:, None]).toarray()
    return np.log1p(normalized).astype(np.float32)


def _type_mean_cells(
    reference_expression: np.ndarray,
    reference_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    predicted_type_names: np.ndarray,
) -> Optional[np.ndarray]:
    lookup = {str(value): index for index, value in enumerate(predicted_type_names.tolist())}
    mapped = np.asarray([lookup.get(str(value), -1) for value in reference_labels], dtype=np.int64)
    supported = mapped >= 0
    if not supported.any():
        return None
    return type_mean_prediction(
        reference_expression[supported],
        mapped[supported],
        predicted_probabilities,
    )


def _aggregate_log_expression(
    cell_log_expression: np.ndarray,
    spot_index: np.ndarray,
    num_spots: int,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(cell_log_expression, dtype=np.float64)
    indices = np.asarray(spot_index, dtype=np.int64)
    if values.ndim != 2 or indices.shape != (values.shape[0],):
        raise ValueError("cell expression and spot assignment are misaligned")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("cell log expression must be finite and non-negative")
    if np.any(indices < -1) or np.any(indices >= num_spots):
        raise ValueError("spot assignment contains an unavailable index")
    assigned = indices >= 0
    mass = np.bincount(indices[assigned], minlength=num_spots).astype(np.float64)
    sums = np.zeros((num_spots, values.shape[1]), dtype=np.float64)
    np.add.at(sums, indices[assigned], np.expm1(values[assigned]))
    means = sums / np.maximum(mass[:, None], 1.0)
    return np.log1p(means).astype(np.float32), mass


def _exact_gene_order(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_names = tuple(str(value) for value in left.tolist())
    right_names = tuple(str(value) for value in right.tolist())
    if left_names != right_names:
        raise ValueError("benchmark gene panels must have identical order")
    return np.arange(len(left_names), dtype=np.int64)


def _gene_panel(path: Path) -> Tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        genes = tuple(
            line.strip().split("\t")[0]
            for line in handle
            if line.strip() and not line.startswith("#")
        )
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("frozen gene panel must be non-empty and unique")
    return genes


def _finite_or_none(value: object) -> Optional[float]:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256((str(seed) + "\x1f" + value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "BENCHMARK_METHODS",
    "HEIR_METHOD",
    "INFERENCE_TELEMETRY_SCHEMA",
    "PSEUDOBULK_METHOD",
    "SHUFFLE_METHOD",
    "SNPATHO_PLAN_SCHEMA",
    "SNPATHO_REPORT_SCHEMA",
    "SNPATHO_SECTION_IDS",
    "TYPE_MEAN_METHOD",
    "InferenceTelemetry",
    "SnPathoBenchmarkPlan",
    "SnPathoBenchmarkResult",
    "SnPathoCase",
    "SnPathoCaseResult",
    "load_snpatho_plan",
    "run_snpatho_benchmark",
    "write_snpatho_benchmark",
]
