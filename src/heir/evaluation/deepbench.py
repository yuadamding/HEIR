"""Retrospective snPATHO-DeepBench evaluator.

DeepBench is deliberately separate from the immutable snPATHO locked report.
It reuses only hash-frozen predictions and truth, applies the stricter attached
diagnostic rules where the required fields exist, and records unavailable
tracks instead of silently substituting weaker evidence.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from scipy import sparse  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from scipy.stats import rankdata  # type: ignore[import-untyped]

from heir.data import RNAReference, SpatialTruthArtifact
from heir.expression import EXPRESSION_TARGET_SUM
from heir.inference import PredictionBundle
from heir.utils import atomic_json_dump, sha256_file

from .snpatho import SnPathoBenchmarkResult, load_snpatho_plan, run_snpatho_benchmark
from .spatial import morans_i

DEEPBENCH_PLAN_SCHEMA = "heir.snpatho_deepbench_plan.v1"
DEEPBENCH_REPORT_SCHEMA = "heir.snpatho_deepbench.v1"
DEEPBENCH_STATUS = "retrospective_diagnostic"
DEEPBENCH_SAMPLES = ("4066", "4399", "4411")
PRIMARY_METHOD = "heir_round0_rna_mass"
SELECTIVE_METHOD = "heir_round0_selective_rna_mass"
EQUAL_CELL_METHOD = "heir_round0_equal_cell"
TYPE_MEAN_METHOD = "historical_integrated_type_mean_rna_mass"
PSEUDOBULK_METHOD = "historical_integrated_snrna_pseudobulk"
SHUFFLE_METHOD = "heir_spatial_shuffle_rna_mass"

OPTIONAL_ARTIFACTS = (
    "refined_predictions",
    "five_seed_predictions",
    "alternative_workflow_references",
    "wrong_donor_predictions",
    "generic_atlas_predictions",
    "h_and_e_only_predictions",
    "image_shuffle_predictions",
    "graph_shuffle_predictions",
    "no_geometry_predictions",
    "manual_nucleus_labels",
    "spot_composition_covariates",
    "pathologist_regions",
    "published_program_definitions",
    "author_qc_tissue_fraction",
    "segmentation_sensitivity_predictions",
    "regional_384um_features",
    "native_scanvi_checkpoint",
)


@dataclass(frozen=True)
class DeepBenchPlan:
    """Validated executable subset of the attached DeepBench specification."""

    source_path: Path
    source_sha256: str
    name: str
    status: str
    historical_result_name: str
    frozen_plan: Path
    frozen_plan_sha256: str
    historical_report: Path
    historical_report_sha256: str
    sample_ids: Tuple[str, ...]
    minimum_nuclei: int
    bootstrap_iterations: int
    primary_seeds: Tuple[int, ...]
    optional_artifacts: Mapping[str, Optional[Path]]
    optional_artifact_sha256: Mapping[str, Optional[str]]
    specification: Mapping[str, Any]


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError("DeepBench plan is missing %s" % ".".join(keys))
        current = current[key]
    return current


def _require_equal(mapping: Mapping[str, Any], expected: object, *keys: str) -> None:
    observed = _nested(mapping, *keys)
    if observed != expected:
        raise ValueError(
            "DeepBench plan %s must be %r, observed %r" % (".".join(keys), expected, observed)
        )


def validate_deepbench_specification(payload: Mapping[str, Any]) -> None:
    """Validate the method-critical fields copied from the attached plan."""

    _require_equal(payload, DEEPBENCH_PLAN_SCHEMA, "schema_version")
    _require_equal(payload, "snpatho_deepbench_v1", "benchmark", "name")
    _require_equal(payload, DEEPBENCH_STATUS, "benchmark", "status")
    _require_equal(payload, "snpatho_locked_v0_2", "benchmark", "frozen_historical_result")
    _require_equal(payload, "retrospective_only", "tracks", "capture_aware_a2")
    _require_equal(payload, "prohibited", "inputs", "target_visium_expression_before_freeze")
    _require_equal(payload, "prohibited", "inputs", "target_rctd_before_freeze")
    _require_equal(payload, [32, 128, 384], "image", "patch_diameters_um")
    _require_equal(payload, "frozen_common_segmentation", "image", "segmentation_primary")
    _require_equal(payload, ["CellViT", "StarDist"], "image", "segmentation_sensitivity")
    _require_equal(payload, 12, "graph", "knn")
    _require_equal(payload, 50, "graph", "radius_um")
    _require_equal(payload, 24, "graph", "maximum_degree")
    _require_equal(payload, "scANVI", "rna", "model")
    _require_equal(payload, 32, "rna", "latent_dim")
    _require_equal(payload, 50, "rna", "prototype_minimum_cells")
    _require_equal(payload, 10, "rna", "maximum_prototypes_per_type")
    _require_equal(payload, True, "rna", "decoder_frozen")
    _require_equal(payload, "fixed", "rna", "primary_prior_update")
    _require_equal(payload, 500, "targets", "genes")
    _require_equal(payload, 15, "targets", "programs", "published_robust_nmf_clusters")
    _require_equal(payload, "detached_uot_responsibilities", "refinement", "e_step")
    _require_equal(payload, True, "refinement", "anchors_revocable")
    _require_equal(payload, 3, "refinement", "maximum_rounds")
    _require_equal(payload, 0.90, "refinement", "minimum_probability")
    _require_equal(payload, 0.20, "refinement", "maximum_normalized_entropy")
    _require_equal(payload, 0.50, "refinement", "minimum_segmentation_confidence")
    _require_equal(payload, 2, "refinement", "minimum_view_agreement")
    _require_equal(payload, 2, "refinement", "trusted_after_consecutive_rounds")
    _require_equal(payload, 0.70, "refinement", "revoke_probability_threshold")
    _require_equal(payload, [17, 41, 89, 131, 197], "randomness", "primary_seeds")
    _require_equal(payload, "matched_type_mean", "evaluation", "primary_baseline")
    _require_equal(
        payload,
        "paired_median_gene_spearman_delta",
        "evaluation",
        "primary_endpoint",
    )
    _require_equal(payload, 3, "evaluation", "primary_spot_minimum_nuclei")
    _require_equal(payload, "zero", "evaluation", "constant_prediction_correlation")
    _require_equal(
        payload,
        "matched_type_median_library_size",
        "evaluation",
        "aggregation_primary",
    )
    _require_equal(payload, ["equal_cell"], "evaluation", "aggregation_sensitivity")
    _require_equal(
        payload,
        [
            "gene_pearson",
            "gene_mse",
            "gene_mae",
            "gene_concordance",
            "hotspot_auroc",
            "hotspot_dice",
            "hotspot_jaccard",
            "location_cosine",
            "location_spearman",
            "location_mae",
            "morans_i_agreement",
        ],
        "evaluation",
        "secondary_metrics",
    )
    _require_equal(payload, 10000, "statistics", "bootstrap_iterations")
    _require_equal(payload, 750, "statistics", "spatial_block_um")
    _require_equal(payload, "benjamini_hochberg", "statistics", "per_gene_fdr")
    _require_equal(payload, "equal", "statistics", "specimen_macro_weighting")
    _require_equal(payload, "prohibited", "statistics", "pooled_spot_inference")
    for control in (
        "wrong_donor_rna",
        "generic_atlas",
        "label_permutation",
        "image_shuffle",
        "graph_shuffle",
        "no_refinement",
        "no_unknown",
        "no_graph",
        "state_omission",
    ):
        _require_equal(payload, True, "controls", control)
    if set(str(value) for value in _nested(payload, "controls", "reference_downsampling")) != {
        "1000",
        "2500",
        "5000",
        "all",
    }:
        raise ValueError("DeepBench reference downsampling levels differ from the attached plan")
    samples = _nested(payload, "cohort", "samples")
    if not isinstance(samples, list):
        raise ValueError("DeepBench cohort.samples must be a list")
    identifiers = tuple(str(item.get("id")) for item in samples if isinstance(item, Mapping))
    if identifiers != DEEPBENCH_SAMPLES:
        raise ValueError("DeepBench requires specimens 4066, 4399, and 4411 in order")


def load_deepbench_plan(path: Path) -> DeepBenchPlan:
    """Load the immutable DeepBench plan and validate its frozen dependencies."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("DeepBench plan root must be a mapping")
    validate_deepbench_specification(payload)
    root = source.parent

    def resolve(raw: object) -> Path:
        candidate = Path(str(raw)).expanduser()
        return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    execution = _nested(payload, "execution")
    if not isinstance(execution, Mapping):
        raise ValueError("DeepBench execution block must be a mapping")
    frozen_plan = resolve(execution["frozen_benchmark_plan"])
    historical_report = resolve(execution["frozen_historical_report"])
    frozen_plan_sha256 = str(execution["frozen_benchmark_plan_sha256"])
    historical_report_sha256 = str(execution["frozen_historical_report_sha256"])
    for label, artifact, digest in (
        ("frozen benchmark plan", frozen_plan, frozen_plan_sha256),
        ("historical report", historical_report, historical_report_sha256),
    ):
        if not artifact.is_file():
            raise FileNotFoundError("%s is absent: %s" % (label, artifact))
        if sha256_file(artifact) != digest:
            raise ValueError("%s SHA-256 differs from DeepBench" % label)
    raw_optional = execution.get("optional_artifacts")
    if not isinstance(raw_optional, Mapping) or set(raw_optional) != set(OPTIONAL_ARTIFACTS):
        raise ValueError("DeepBench optional_artifacts must explicitly list every optional track")
    optional: Dict[str, Optional[Path]] = {}
    optional_hashes: Dict[str, Optional[str]] = {}
    for name in OPTIONAL_ARTIFACTS:
        value = raw_optional[name]
        if value is None:
            optional[name] = None
            optional_hashes[name] = None
        elif isinstance(value, Mapping):
            if set(value) != {"path", "sha256"}:
                raise ValueError("optional artifact %s requires path and sha256" % name)
            artifact = resolve(value["path"])
            if not artifact.is_file() or sha256_file(artifact) != str(value["sha256"]):
                raise ValueError("optional artifact %s is absent or hash-mismatched" % name)
            optional[name] = artifact
            optional_hashes[name] = str(value["sha256"])
        else:
            raise ValueError("optional artifact %s must be null or a path/hash mapping" % name)
    return DeepBenchPlan(
        source_path=source,
        source_sha256=sha256_file(source),
        name=str(_nested(payload, "benchmark", "name")),
        status=str(_nested(payload, "benchmark", "status")),
        historical_result_name=str(_nested(payload, "benchmark", "frozen_historical_result")),
        frozen_plan=frozen_plan,
        frozen_plan_sha256=frozen_plan_sha256,
        historical_report=historical_report,
        historical_report_sha256=historical_report_sha256,
        sample_ids=DEEPBENCH_SAMPLES,
        minimum_nuclei=int(_nested(payload, "evaluation", "primary_spot_minimum_nuclei")),
        bootstrap_iterations=int(_nested(payload, "statistics", "bootstrap_iterations")),
        primary_seeds=tuple(
            int(value) for value in _nested(payload, "randomness", "primary_seeds")
        ),
        optional_artifacts=optional,
        optional_artifact_sha256=optional_hashes,
        specification=dict(payload),
    )


def aggregate_cells_to_spots(
    cell_log_expression: np.ndarray,
    spot_index: np.ndarray,
    num_spots: int,
    cell_rna_mass: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate log1p expression with explicit overlap/RNA-mass weights."""

    values = np.asarray(cell_log_expression, dtype=np.float64)
    indices = np.asarray(spot_index, dtype=np.int64)
    if values.ndim != 2 or indices.shape != (values.shape[0],):
        raise ValueError("cell expression and spot assignment are misaligned")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("cell expression must be finite and non-negative")
    if np.any(indices < -1) or np.any(indices >= num_spots):
        raise ValueError("spot assignment contains an invalid index")
    weights = (
        np.ones(values.shape[0], dtype=np.float64)
        if cell_rna_mass is None
        else np.asarray(cell_rna_mass, dtype=np.float64)
    )
    if weights.shape != (values.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("cell_rna_mass must be finite, non-negative, and aligned")
    assigned = (indices >= 0) & (weights > 0)
    mass = np.bincount(indices[assigned], weights=weights[assigned], minlength=num_spots)
    sums = np.zeros((num_spots, values.shape[1]), dtype=np.float64)
    np.add.at(
        sums,
        indices[assigned],
        np.expm1(values[assigned]) * weights[assigned, None],
    )
    return np.log1p(sums / np.maximum(mass[:, None], 1.0e-12)).astype(np.float32), mass


def _reference_linear_profiles(
    reference: RNAReference,
    type_names: Sequence[object],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build type profiles from pooled counts/full-library mass in linear space."""

    counts = sparse.csr_matrix(reference.counts, dtype=np.float64)
    if reference.library_sizes is None:
        raise ValueError("reference is missing full-transcriptome library sizes")
    library = np.asarray(reference.library_sizes, dtype=np.float64)
    if library.shape != (counts.shape[0],) or np.any(library <= 0):
        raise ValueError("reference requires positive full-transcriptome library sizes")
    labels = np.asarray(reference.cell_type_labels).astype(str)
    global_profile = np.asarray(counts.sum(axis=0)).reshape(-1)
    global_profile = global_profile * (EXPRESSION_TARGET_SUM / library.sum())
    global_median = float(np.median(library))
    profiles = []
    median_library_sizes = []
    for raw_name in type_names:
        selected = labels == str(raw_name)
        if selected.any():
            pooled = np.asarray(counts[selected].sum(axis=0)).reshape(-1)
            profiles.append(pooled * (EXPRESSION_TARGET_SUM / library[selected].sum()))
            median_library_sizes.append(float(np.median(library[selected])))
        else:
            profiles.append(global_profile)
            median_library_sizes.append(global_median)
    return (
        np.asarray(profiles, dtype=np.float32),
        np.asarray(median_library_sizes, dtype=np.float64),
    )


def _reference_linear_pseudobulk(reference: RNAReference) -> np.ndarray:
    counts = sparse.csr_matrix(reference.counts, dtype=np.float64)
    if reference.library_sizes is None:
        raise ValueError("reference is missing full-transcriptome library sizes")
    library = np.asarray(reference.library_sizes, dtype=np.float64)
    pooled = np.asarray(counts.sum(axis=0)).reshape(-1)
    return (pooled * (EXPRESSION_TARGET_SUM / library.sum())).astype(np.float32)


def _cell_rna_mass(reference: RNAReference, prediction: PredictionBundle) -> np.ndarray:
    _, medians = _reference_linear_profiles(reference, prediction.type_names.tolist())
    probabilities = np.asarray(prediction.type_probabilities, dtype=np.float64)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1.0e-12)
    mass = probabilities.dot(medians)
    return mass / max(float(np.median(mass)), 1.0e-12)


def _type_mean_cells(
    reference: RNAReference,
    prediction: PredictionBundle,
) -> np.ndarray:
    profiles, _ = _reference_linear_profiles(reference, prediction.type_names.tolist())
    probabilities = np.asarray(prediction.type_probabilities, dtype=np.float64)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1.0e-12)
    hard_types = probabilities.argmax(axis=1)
    return np.log1p(profiles[hard_types]).astype(np.float32)


def _safe_correlation(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    rank: bool,
    eps: float = 1.0e-12,
) -> float:
    if float(np.var(observed)) <= eps:
        return float("nan")
    if float(np.var(predicted)) <= eps:
        return 0.0
    left = rankdata(predicted) if rank else predicted
    right = rankdata(observed) if rank else observed
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _concordance(predicted: np.ndarray, observed: np.ndarray) -> float:
    covariance = float(np.mean((predicted - predicted.mean()) * (observed - observed.mean())))
    denominator = float(
        np.var(predicted) + np.var(observed) + (predicted.mean() - observed.mean()) ** 2
    )
    return 0.0 if denominator <= 1.0e-12 else 2.0 * covariance / denominator


def _top_indices(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    count = max(1, int(math.ceil(len(values) * fraction)))
    return np.argpartition(values, -count)[-count:]


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels.astype(bool)
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float(ranks[positives].sum() - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _hotspot_scores(predicted: np.ndarray, observed: np.ndarray) -> Tuple[float, float, float]:
    observed_top = _top_indices(observed)
    predicted_top = _top_indices(predicted)
    detection_auc = _binary_auc(predicted, observed > 0)
    intersection = len(np.intersect1d(observed_top, predicted_top, assume_unique=False))
    dice = 2.0 * intersection / max(len(observed_top) + len(predicted_top), 1)
    union = len(np.union1d(observed_top, predicted_top))
    return detection_auc, dice, intersection / max(union, 1)


def _spot_graph(coordinates: np.ndarray, neighbors: int = 6) -> np.ndarray:
    count = len(coordinates)
    if count < 2:
        return np.empty((2, 0), dtype=np.int64)
    width = min(neighbors + 1, count)
    _, indices = cKDTree(coordinates).query(coordinates, k=width)
    if indices.ndim == 1:
        indices = indices[:, None]
    source = np.repeat(np.arange(count), width - 1)
    target = indices[:, 1:].reshape(-1)
    return np.stack((source, target)).astype(np.int64)


def deepbench_expression_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    coordinates: np.ndarray,
    gene_names: Optional[Sequence[object]] = None,
) -> Dict[str, Any]:
    """Compute the attached diagnostic metric panel with explicit constants policy."""

    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("predicted and observed expression must have identical 2-D shapes")
    if coordinates.shape != (prediction.shape[0], 2):
        raise ValueError("spot coordinates must align to expression")
    if prediction.shape[0] < 3:
        raise ValueError("DeepBench metrics require at least three spots")
    resolved_gene_names = (
        tuple("gene_%d" % index for index in range(prediction.shape[1]))
        if gene_names is None
        else tuple(str(value) for value in gene_names)
    )
    if len(resolved_gene_names) != prediction.shape[1]:
        raise ValueError("gene_names must align to expression columns")
    per_gene: Dict[str, List[Any]] = {
        name: []
        for name in (
            "gene_names",
            "correlation_status",
            "correlation_reason",
            "spearman",
            "pearson",
            "mse",
            "mae",
            "concordance",
            "hotspot_auroc",
            "hotspot_dice",
            "hotspot_jaccard",
            "observed_mean",
            "observed_variance",
            "predicted_variance",
            "predicted_morans_i",
            "observed_morans_i",
        )
    }
    edges = _spot_graph(coordinates)
    for gene in range(prediction.shape[1]):
        left = prediction[:, gene]
        right = truth[:, gene]
        evaluable = float(np.var(right)) > 1.0e-12
        prediction_constant = float(np.var(left)) <= 1.0e-12
        spearman = _safe_correlation(left, right, rank=True) if evaluable else float("nan")
        pearson = _safe_correlation(left, right, rank=False) if evaluable else float("nan")
        hotspot = _hotspot_scores(left, right) if evaluable else (float("nan"),) * 3
        gene_predicted_i = morans_i(left, edges) if evaluable else float("nan")
        gene_observed_i = morans_i(right, edges) if evaluable else float("nan")
        values = {
            "spearman": spearman,
            "pearson": pearson,
            "mse": float(np.mean((left - right) ** 2)),
            "mae": float(np.mean(np.abs(left - right))),
            "concordance": _concordance(left, right) if evaluable else float("nan"),
            "hotspot_auroc": hotspot[0],
            "hotspot_dice": hotspot[1],
            "hotspot_jaccard": hotspot[2],
            "observed_mean": float(np.mean(right)),
            "observed_variance": float(np.var(right)),
            "predicted_variance": float(np.var(left)),
            "predicted_morans_i": gene_predicted_i,
            "observed_morans_i": gene_observed_i,
        }
        per_gene["gene_names"].append(resolved_gene_names[gene])
        per_gene["correlation_status"].append(
            "excluded_observed_constant"
            if not evaluable
            else ("prediction_constant_scored_zero" if prediction_constant else "ok")
        )
        per_gene["correlation_reason"].append(
            "observed spatial expression is constant"
            if not evaluable
            else (
                "prediction is constant while observed expression varies; correlation scored zero"
                if prediction_constant
                else ""
            )
        )
        for name, value in values.items():
            per_gene[name].append(float(value) if np.isfinite(value) else None)

    def finite(name: str) -> np.ndarray:
        return np.asarray(
            [np.nan if value is None else value for value in per_gene[name]], dtype=np.float64
        )

    left_norm = np.linalg.norm(prediction, axis=1)
    right_norm = np.linalg.norm(truth, axis=1)
    cosine = (prediction * truth).sum(axis=1) / np.maximum(left_norm * right_norm, 1.0e-12)
    location_spearman = np.asarray(
        [
            _safe_correlation(prediction[index], truth[index], rank=True)
            for index in range(len(prediction))
        ]
    )
    predicted_i = finite("predicted_morans_i")
    observed_i = finite("observed_morans_i")
    spatial_valid = np.isfinite(predicted_i) & np.isfinite(observed_i)
    spatial_spearman = (
        _safe_correlation(predicted_i[spatial_valid], observed_i[spatial_valid], rank=True)
        if spatial_valid.sum() >= 3
        else float("nan")
    )

    def median(name: str) -> Optional[float]:
        values = finite(name)
        return float(np.nanmedian(values)) if np.isfinite(values).any() else None

    summary: Dict[str, Optional[float]] = {
        "median_gene_spearman": median("spearman"),
        "median_gene_pearson": median("pearson"),
        "median_gene_mse": median("mse"),
        "median_gene_mae": median("mae"),
        "median_gene_concordance": median("concordance"),
        "median_hotspot_auroc": median("hotspot_auroc"),
        "median_hotspot_dice": median("hotspot_dice"),
        "median_hotspot_jaccard": median("hotspot_jaccard"),
        "mean_location_cosine": float(np.mean(cosine)),
        "median_location_spearman": (
            float(np.nanmedian(location_spearman)) if np.isfinite(location_spearman).any() else None
        ),
        "mean_location_mae": float(np.mean(np.abs(prediction - truth))),
        "morans_i_spearman": spatial_spearman if np.isfinite(spatial_spearman) else None,
        "morans_i_mae": (
            float(np.mean(np.abs(predicted_i[spatial_valid] - observed_i[spatial_valid])))
            if spatial_valid.any()
            else None
        ),
        "fraction_genes_evaluable": float(np.isfinite(finite("spearman")).mean()),
    }
    return {"summary": summary, "per_gene": per_gene}


def _stable_seed(seed: int, sample: str) -> int:
    digest = hashlib.sha256((str(seed) + "\x1f" + sample).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def _readiness(plan: DeepBenchPlan) -> Tuple[Dict[str, Any], ...]:
    ready = [
        ("locked_round0_predictions", "Hash-frozen v0.2 predictions for all three specimens"),
        (
            "historical_integrated_type_mean",
            "Derived from the v0.2 pooled integrated multi-workflow reference",
        ),
        (
            "historical_integrated_pseudobulk",
            "Derived from the v0.2 pooled integrated multi-workflow reference",
        ),
        (
            "historical_final_cell_record_shuffle",
            "Deterministic final cell-record shuffle; this is distinct from and does not "
            "satisfy the required image-feature or coordinate/graph shuffle controls",
        ),
        (
            "rna_mass_aggregation",
            "Historical integrated-reference type median library sizes",
        ),
    ]
    records: List[Dict[str, Any]] = [
        {"component": name, "status": "ready", "reason": reason} for name, reason in ready
    ]
    partial = {
        "primary_ffpe_snpatho_reference": (
            "blocked_noncompliant_input",
            "v0.2 reference500 artifacts pool integrated RNA workflows; no FFPE-snPATHO-only "
            "reference is frozen",
        ),
        "primary_spot_qc": (
            "partial",
            "processed RDS spots are author-QC-whitelisted and positive-library with >=3 nuclei; "
            "the required >=50% per-spot H&E tissue fraction plus explicit exclusion flags and "
            "reasons are absent",
        ),
        "hierarchical_spatial_bootstrap": (
            "partial",
            "paired donor/gene bootstrap is available; the historical run lacks frozen connected "
            "spatial-block definitions",
        ),
        "alternative_rna_raw_inputs": (
            "partial",
            "downloaded integrated objects expose FFPE snPATHO, frozen SNAP snPATHO/Flex, and "
            "frozen 3-prime strata, but workflow-specific frozen references/predictions have "
            "not been prepared and no scFFPE stratum is present in those objects",
        ),
        "externally_frozen_ood_rule": (
            "blocked_noncompliant_historical_input",
            "v0.2 recalibrated the OOD threshold from each target H&E slide's 95th percentile; "
            "the capture-aware historical run therefore does not establish Track A/A2 compliance",
        ),
        "segmentation_confidence_anchor_gate": (
            "blocked_nonfunctional_gate",
            "Space Ranger exports no calibrated segmentation confidence and v0.2 substituted "
            "1.0, so the >=0.50 anchor gate is vacuous and refinement is not benchmark-ready",
        ),
    }
    records.extend(
        {"component": name, "status": status, "reason": reason}
        for name, (status, reason) in partial.items()
    )
    blocked_plan_components = {
        "track_a1_external_personalization": (
            "No externally frozen H&E-plus-snRNA-only predictions exist"
        ),
        "track_b_leave_one_specimen_out": (
            "No nested leave-one-specimen-out configurations or predictions are frozen"
        ),
        "independent_snpatho_reannotation": (
            "Historical labels came from integrated workflow objects; no snPATHO-only frozen "
            "ontology and marker-review artifact exists"
        ),
        "reference_size_and_per_type_caps": (
            "The requested five draws at 1k/2.5k/5k/all and 100/250/500/1k per-type caps were "
            "not generated"
        ),
        "hierarchical_ontology_scoring": (
            "No frozen compartment/major-type/supported-subtype mapping and evaluation output "
            "is available"
        ),
        "manual_segmentation_roi_audit": (
            "The 24 stratified ROIs per specimen and independent detection annotations do not exist"
        ),
        "image_multiscale_and_morphology_ablations": (
            "The 32/128/384-um and explicit-morphology ablation predictions were not generated"
        ),
        "graph_sensitivity_and_rewiring": (
            "The 8-NN, radius, multiscale, no-graph, and degree-preserving rewiring runs are absent"
        ),
        "refinement_trajectory_and_ablations": (
            "No round 1-3 predictions, anchor telemetry, E-step comparison, or prior-update "
            "sensitivity is frozen"
        ),
        "composition_controlled_residuals": (
            "Five spatial-block folds, independent composition covariates, library covariates, "
            "and pathologist regions are absent"
        ),
        "manual_cell_type_benchmark": (
            "No two-reviewer consensus nucleus labels or evaluation-only confidence scores exist"
        ),
        "spot_composition_consensus": (
            "No frozen RCTD/cell2location/DestVI consensus artifact is available"
        ),
        "uncertainty_calibration_and_risk_coverage": (
            "Historical artifacts do not contain full posterior ensembles or fixed-coverage "
            "evaluation outputs"
        ),
        "unknown_state_omission_stress_test": (
            "Per-major-type reference-omission predictions were not generated"
        ),
        "complete_negative_control_matrix": (
            "Label/prototype/state permutations, reference downsampling, image perturbations, "
            "refinement gate ablations, block shuffles, toroidal shifts, and coordinate "
            "perturbations were not generated"
        ),
        "core_model_ablation_matrix": (
            "No-UOT, balanced-OT, query/final-latent UOT, no/low-rank residual, no covariance, "
            "fixed/updated prior, and initializer ablations are absent"
        ),
        "expanded_spatial_structure_metrics": (
            "Geary C, semivariogram, spatial EMD, and boundary-localization scorers are not "
            "implemented in this executable subset"
        ),
        "per_gene_block_permutation_fdr": (
            "Frozen connected blocks and block-permutation nulls needed for BH-FDR are absent"
        ),
        "biological_case_study_endpoints": (
            "Prespecified HER2/DCIS/calcium, tumor-liver, and liver-resident program definitions "
            "and region labels are absent"
        ),
        "seed_ensemble_stability": (
            "Only seed 17 exists, so map, anchor, assignment, and between-model stability cannot "
            "be scored"
        ),
        "complete_computational_benchmark": (
            "Historical inference telemetry exists, but segmentation, feature extraction, "
            "training, refinement, CPU memory, checkpoint/cache size, and energy are incomplete"
        ),
    }
    records.extend(
        {
            "component": name,
            "status": "blocked_not_implemented_or_missing_artifact",
            "reason": reason,
        }
        for name, reason in blocked_plan_components.items()
    )
    reasons = {
        "refined_predictions": "No post-redesign refined predictions are supplied",
        "five_seed_predictions": "Only the historical seed-17 prediction is frozen",
        "alternative_workflow_references": "No scFFPE/Flex/3-prime reference artifacts are frozen",
        "wrong_donor_predictions": "Wrong-donor HEIR predictions have not been generated",
        "generic_atlas_predictions": "Generic-atlas HEIR predictions have not been generated",
        "h_and_e_only_predictions": "No RNA-free H&E prediction artifact is supplied",
        "image_shuffle_predictions": "Required shuffled-image-feature HEIR predictions are absent",
        "graph_shuffle_predictions": "Required coordinate-shuffled graph predictions are absent",
        "no_geometry_predictions": "The historical run used capture-area geometry",
        "manual_nucleus_labels": "No evaluation-only consensus nucleus annotations are available",
        "spot_composition_covariates": "No frozen independent spot-composition covariates exist",
        "pathologist_regions": "No frozen pathologist-region artifact exists",
        "published_program_definitions": (
            "The 15 published program definitions are not frozen locally"
        ),
        "author_qc_tissue_fraction": (
            "The processed RDS materializes the author-QC spot whitelist, but explicit exclusion "
            "flags/reasons and the required >=50% per-spot H&E tissue fraction are absent"
        ),
        "segmentation_sensitivity_predictions": (
            "Only the common Space Ranger nucleus set is frozen"
        ),
        "regional_384um_features": "Historical OmiCLIP features contain only 32 and 128 um scales",
        "native_scanvi_checkpoint": "The historical molecular decoder is the B1 SVD/MLP fallback",
    }
    for name in OPTIONAL_ARTIFACTS:
        artifact = plan.optional_artifacts[name]
        records.append(
            {
                "component": name,
                "status": (
                    "registered_not_implemented"
                    if artifact is not None
                    else "blocked_missing_artifact"
                ),
                "reason": (
                    "Hash-validated artifact is registered, but no scorer consumes this schema yet"
                    if artifact is not None
                    else reasons[name]
                ),
                "path": None if artifact is None else str(artifact),
            }
        )
    return tuple(records)


def _bootstrap_macro_delta(
    deltas: Sequence[np.ndarray],
    observed_means: Sequence[np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    donor_estimates = np.asarray([np.nanmedian(values) for values in deltas], dtype=np.float64)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected_donors = rng.integers(0, len(deltas), size=len(deltas))
        estimates = []
        for donor in selected_donors:
            values = np.asarray(deltas[donor], dtype=np.float64)
            means = np.asarray(observed_means[donor], dtype=np.float64)
            valid = np.isfinite(values) & np.isfinite(means)
            values = values[valid]
            means = means[valid]
            if not len(values):
                continue
            boundaries = np.quantile(means, [0.25, 0.50, 0.75])
            strata = np.digitize(means, boundaries, right=True)
            sampled = []
            for stratum in range(4):
                candidates = values[strata == stratum]
                if len(candidates):
                    sampled.append(rng.choice(candidates, size=len(candidates), replace=True))
            estimates.append(float(np.median(np.concatenate(sampled))))
        bootstrap[iteration] = float(np.mean(estimates)) if estimates else np.nan
    finite = bootstrap[np.isfinite(bootstrap)]
    return {
        "estimate": float(np.mean(donor_estimates)),
        "ci_lower": float(np.quantile(finite, 0.025)) if len(finite) else None,
        "ci_upper": float(np.quantile(finite, 0.975)) if len(finite) else None,
        "probability_positive": float(np.mean(finite > 0)) if len(finite) else None,
        "iterations": int(iterations),
        "method": "paired specimen/gene abundance-stratified bootstrap",
        "limitation": "connected spatial blocks were not frozen in the historical artifacts",
    }


def _primary_diagnostic(cases: Sequence[Dict[str, Any]], plan: DeepBenchPlan) -> Dict[str, Any]:
    deltas = []
    observed_means = []
    rows = []
    for case in cases:
        methods = case["methods"]
        assert isinstance(methods, Mapping)
        primary = methods[PRIMARY_METHOD]
        baseline = methods[TYPE_MEAN_METHOD]
        shuffled = methods[SHUFFLE_METHOD]
        assert isinstance(primary, Mapping) and isinstance(baseline, Mapping)
        primary_gene = primary["per_gene"]
        baseline_gene = baseline["per_gene"]
        shuffled_gene = shuffled["per_gene"]
        assert isinstance(primary_gene, Mapping) and isinstance(baseline_gene, Mapping)
        left = np.asarray(
            [np.nan if value is None else value for value in primary_gene["spearman"]],
            dtype=np.float64,
        )
        right = np.asarray(
            [np.nan if value is None else value for value in baseline_gene["spearman"]],
            dtype=np.float64,
        )
        shuffle_values = np.asarray(
            [np.nan if value is None else value for value in shuffled_gene["spearman"]],
            dtype=np.float64,
        )
        difference = left - right
        deltas.append(difference)
        observed_means.append(
            np.asarray(
                [np.nan if value is None else value for value in primary_gene["observed_mean"]],
                dtype=np.float64,
            )
        )
        primary_summary = primary["summary"]
        baseline_summary = baseline["summary"]
        assert isinstance(primary_summary, Mapping) and isinstance(baseline_summary, Mapping)
        rows.append(
            {
                "section_id": case["section_id"],
                "median_gene_spearman_delta_vs_type_mean": float(np.nanmedian(difference)),
                "median_gene_spearman_delta_vs_spatial_shuffle": float(
                    np.nanmedian(left - shuffle_values)
                ),
                "median_mse_improvement_vs_type_mean": float(
                    baseline_summary["median_gene_mse"] - primary_summary["median_gene_mse"]
                ),
            }
        )
    macro_delta = float(np.mean([row["median_gene_spearman_delta_vs_type_mean"] for row in rows]))
    rules = {
        "macro_delta_positive": macro_delta > 0,
        "positive_in_at_least_two_specimens": sum(
            row["median_gene_spearman_delta_vs_type_mean"] > 0 for row in rows
        )
        >= 2,
        "no_specimen_below_minus_0_01": all(
            row["median_gene_spearman_delta_vs_type_mean"] >= -0.01 for row in rows
        ),
        "mse_improves_in_at_least_two_specimens": sum(
            row["median_mse_improvement_vs_type_mean"] > 0 for row in rows
        )
        >= 2,
        "beats_spatial_shuffle_in_at_least_two_specimens": sum(
            row["median_gene_spearman_delta_vs_spatial_shuffle"] > 0 for row in rows
        )
        >= 2,
        "composition_adjusted_residual_positive": None,
    }
    refined_available = plan.optional_artifacts["refined_predictions"] is not None
    requested_blockers = [
        "required FFPE-snPATHO-only primary reference is absent",
        "refined predictions are absent or their schema is not consumed",
        "composition-adjusted residual inputs are absent",
        "required per-spot H&E tissue fraction is absent",
    ]
    return {
        "requested_primary_contrast": "refined_heir_minus_matched_type_mean",
        "requested_primary_status": (
            "not_testable_registered_refined_schema_not_implemented"
            if refined_available
            else "not_testable_missing_refined_predictions"
        ),
        "requested_primary_blockers": requested_blockers,
        "diagnostic_contrast": ("historical_round0_minus_historical_integrated_type_mean"),
        "diagnostic_status": (
            "fails_available_criteria"
            if any(value is False for value in rules.values())
            else "incomplete_without_composition_adjustment"
        ),
        "specimens": rows,
        "macro_delta": macro_delta,
        "rules": rules,
        "bootstrap": _bootstrap_macro_delta(
            deltas,
            observed_means,
            iterations=plan.bootstrap_iterations,
            seed=plan.primary_seeds[0],
        ),
    }


def _method_macro_summaries(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize methods with specimens, rather than spots, as equal-weight units."""

    if not cases:
        return {}
    first_methods = cases[0]["methods"]
    if not isinstance(first_methods, Mapping):
        raise ValueError("DeepBench case methods must be a mapping")
    result: Dict[str, Any] = {}
    for method in first_methods:
        specimen_payloads = []
        for case in cases:
            methods = case["methods"]
            if not isinstance(methods, Mapping) or method not in methods:
                raise ValueError("DeepBench cases must expose the same methods")
            payload = methods[method]
            if not isinstance(payload, Mapping):
                raise ValueError("DeepBench method payload must be a mapping")
            specimen_payloads.append(payload)
        summary_names = sorted(
            {
                str(name)
                for payload in specimen_payloads
                for name in cast(Mapping[str, Any], payload["summary"])
            }
            | {"spot_coverage"}
        )
        metrics: Dict[str, Any] = {}
        for name in summary_names:
            values = []
            for payload in specimen_payloads:
                summary = cast(Mapping[str, Any], payload["summary"])
                value = payload["spot_coverage"] if name == "spot_coverage" else summary[name]
                if value is not None and np.isfinite(float(value)):
                    values.append(float(value))
            metrics[name] = {
                "macro_mean": float(np.mean(values)) if values else None,
                "minimum": float(np.min(values)) if values else None,
                "maximum": float(np.max(values)) if values else None,
                "specimens_evaluable": len(values),
            }
        result[str(method)] = {
            "aggregation": specimen_payloads[0]["aggregation"],
            "specimens": len(specimen_payloads),
            "metrics": metrics,
        }
    return result


def run_deepbench(plan: DeepBenchPlan) -> Dict[str, Any]:
    """Run every DeepBench component supported by the frozen local artifacts."""

    if sha256_file(plan.source_path) != plan.source_sha256:
        raise ValueError("DeepBench plan changed after loading")
    if sha256_file(plan.frozen_plan) != plan.frozen_plan_sha256:
        raise ValueError("frozen benchmark plan changed after DeepBench loading")
    if sha256_file(plan.historical_report) != plan.historical_report_sha256:
        raise ValueError("historical locked report changed after DeepBench loading")
    for name, artifact in plan.optional_artifacts.items():
        expected = plan.optional_artifact_sha256[name]
        if artifact is not None and (expected is None or sha256_file(artifact) != expected):
            raise ValueError("optional DeepBench artifact %s changed after loading" % name)
    locked_plan = load_snpatho_plan(plan.frozen_plan)
    if tuple(sorted(case.section_id for case in locked_plan.cases)) != plan.sample_ids:
        raise ValueError("frozen benchmark plan does not contain the DeepBench specimens")
    # This re-evaluation validates all frozen hashes and target-isolation
    # contracts; it does not write or mutate the historical report.
    locked: SnPathoBenchmarkResult = run_snpatho_benchmark(
        locked_plan,
        seed=plan.primary_seeds[0],
        iterations=plan.bootstrap_iterations,
        minimum_donors=2,
        require_complete=True,
    )
    cases: List[Dict[str, Any]] = []
    for case in sorted(locked_plan.cases, key=lambda item: item.section_id):
        prediction = PredictionBundle.from_npz(case.predictions)
        reference = RNAReference.load_npz(case.matched_reference)
        truth = SpatialTruthArtifact.from_npz(case.truth)
        rna_mass = _cell_rna_mass(reference, prediction)
        type_cells = _type_mean_cells(reference, prediction)
        spot_index = truth.nucleus_spot_index
        spot_counts = np.bincount(
            spot_index[spot_index >= 0], minlength=len(truth.spot_ids)
        ).astype(np.int64)
        primary_spots = spot_counts >= plan.minimum_nuclei
        if primary_spots.sum() < 3:
            raise ValueError("DeepBench primary spot proxy contains fewer than three spots")
        gene_order = tuple(str(value) for value in prediction.gene_names.tolist())
        if gene_order != tuple(str(value) for value in truth.gene_names.tolist()):
            raise ValueError("DeepBench prediction and truth gene orders differ")
        heir_rna, heir_mass = aggregate_cells_to_spots(
            prediction.expression_mean,
            spot_index,
            len(truth.spot_ids),
            rna_mass,
        )
        selective_rna, selective_mass = aggregate_cells_to_spots(
            prediction.expression_mean,
            spot_index,
            len(truth.spot_ids),
            rna_mass * (~prediction.abstain.astype(bool)),
        )
        heir_equal, equal_mass = aggregate_cells_to_spots(
            prediction.expression_mean,
            spot_index,
            len(truth.spot_ids),
        )
        type_spots, type_mass = aggregate_cells_to_spots(
            type_cells,
            spot_index,
            len(truth.spot_ids),
            rna_mass,
        )
        pseudobulk = np.log1p(_reference_linear_pseudobulk(reference))
        pseudobulk_spots = np.repeat(pseudobulk[None, :], len(truth.spot_ids), axis=0)
        rng = np.random.default_rng(_stable_seed(plan.primary_seeds[0], case.section_id))
        assigned = np.flatnonzero(spot_index >= 0)
        permutation = rng.permutation(assigned)
        shuffled_cells = np.asarray(prediction.expression_mean).copy()
        shuffled_cell_mass = rna_mass.copy()
        shuffled_cells[assigned] = shuffled_cells[permutation]
        shuffled_cell_mass[assigned] = shuffled_cell_mass[permutation]
        shuffled_spots, shuffled_mass = aggregate_cells_to_spots(
            shuffled_cells,
            spot_index,
            len(truth.spot_ids),
            shuffled_cell_mass,
        )
        method_values = {
            PRIMARY_METHOD: (
                heir_rna,
                heir_mass,
                "historical_integrated_type_median_library_size",
            ),
            SELECTIVE_METHOD: (
                selective_rna,
                selective_mass,
                "historical_integrated_type_median_library_size_nonabstained_only",
            ),
            EQUAL_CELL_METHOD: (heir_equal, equal_mass, "equal_cell"),
            TYPE_MEAN_METHOD: (
                type_spots,
                type_mass,
                "historical_integrated_type_median_library_size",
            ),
            PSEUDOBULK_METHOD: (
                pseudobulk_spots,
                np.ones(len(truth.spot_ids)),
                "spatially_constant",
            ),
            SHUFFLE_METHOD: (shuffled_spots, shuffled_mass, "spatially_shuffled_cells"),
        }
        methods: Dict[str, Dict[str, Any]] = {}
        for method, (values, mass, aggregation) in method_values.items():
            evaluable = primary_spots & (mass > 0)
            metrics = deepbench_expression_metrics(
                values[evaluable],
                truth.observed_expression[evaluable],
                truth.spot_coordinates_px[evaluable],
                truth.gene_names.tolist(),
            )
            methods[method] = {
                "aggregation": aggregation,
                "spots_evaluated": int(evaluable.sum()),
                "spot_coverage": float(evaluable.sum() / primary_spots.sum()),
                **metrics,
            }
        cases.append(
            {
                "section_id": case.section_id,
                "donor_id": truth.donor_id,
                "qc": {
                    "reference_nuclei": int(reference.counts.shape[0]),
                    "genes": int(reference.counts.shape[1]),
                    "segmented_nuclei": int(len(prediction.nucleus_ids)),
                    "assigned_nuclei": int((spot_index >= 0).sum()),
                    "spots_total": int(len(truth.spot_ids)),
                    "spots_at_least_1_nucleus": int((spot_counts >= 1).sum()),
                    "spots_at_least_3_nuclei": int((spot_counts >= 3).sum()),
                    "spots_at_least_5_nuclei": int((spot_counts >= 5).sum()),
                    "cell_coverage": float((~prediction.abstain.astype(bool)).mean()),
                },
                "methods": methods,
                "provenance": {
                    "prediction_sha256": case.predictions_sha256,
                    "truth_sha256": case.truth_sha256,
                    "reference_sha256": case.matched_reference_sha256,
                    "checkpoint_sha256": case.checkpoint_sha256,
                },
            }
        )
    readiness = _readiness(plan)
    return {
        "schema_version": DEEPBENCH_REPORT_SCHEMA,
        "benchmark": {
            "name": plan.name,
            "status": plan.status,
            "interpretation": (
                "retrospective capture-aware architecture diagnostic using historical v0.2 "
                "artifacts; not untouched validation and not a compliant Track A/A2 result"
            ),
            "plan_sha256": plan.source_sha256,
        },
        "historical_lock": {
            "name": plan.historical_result_name,
            "report": str(plan.historical_report),
            "report_sha256": plan.historical_report_sha256,
            "overwritten": False,
            "revalidation_plan_sha256": locked.plan_sha256,
        },
        "spot_policy": {
            "implemented_proxy": (
                "author-QC-whitelisted processed RDS spots, positive library size, and at least "
                "3 assigned nuclei"
            ),
            "author_qc_whitelist": "materialized by inclusion in the processed RDS",
            "minimum_nuclei": plan.minimum_nuclei,
            "missing_required_fields": [
                "explicit_author_qc_exclusion_flag",
                "explicit_author_qc_removal_reason",
                "per_spot_he_tissue_fraction_at_least_0_50",
            ],
            "primary_status": "partial_proxy",
        },
        "reference_policy": {
            "required_primary": "matched FFPE snPATHO-seq only",
            "historical_available": (
                "v0.2 pooled integrated multi-workflow reference containing FFPE snPATHO-seq, "
                "frozen Flex, and frozen 3-prime nuclei"
            ),
            "status": "historical_retrospective_only_not_primary_R1",
        },
        "track_policy": {
            "historical_mode": "capture-area-aware with target H&E",
            "noncompliant_step": (
                "v0.2 derived its OOD threshold from each target H&E slide's 95th percentile"
            ),
            "interpretation": (
                "capture-area geometry is compatible with A2, but target-slide OOD calibration "
                "violates the external freeze; this run establishes neither Track A1 nor A2"
            ),
        },
        "shuffle_policy": {
            "available": "complete final cell-record shuffle across assigned nuclei",
            "status": "historical_diagnostic_only",
            "does_not_replace": [
                "shuffled_image_features",
                "coordinate_shuffled_graph",
            ],
        },
        "refinement_policy": {
            "segmentation_source": "Space Ranger",
            "historical_segmentation_confidence": 1.0,
            "confidence_provenance": (
                "substituted constant because Space Ranger exports no calibrated confidence"
            ),
            "consequence": (
                "the >=0.50 segmentation-confidence anchor gate is vacuous; refinement remains "
                "blocked pending a calibrated confidence measurement"
            ),
        },
        "readiness": list(readiness),
        "cases": cases,
        "method_macro": _method_macro_summaries(cases),
        "primary": _primary_diagnostic(cases, plan),
        "reporting": {
            "specimen_is_biological_unit": True,
            "pooled_spot_inference": False,
            "seeds_requested": list(plan.primary_seeds),
            "seeds_available": [17],
            "full_plan_complete": all(item["status"] == "ready" for item in readiness),
        },
    }


def write_deepbench_report(
    report: Dict[str, Any],
    *,
    json_path: Path,
    tsv_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Write separate DeepBench JSON, long-form TSV, and concise Markdown."""

    output = Path(json_path).expanduser().resolve()
    atomic_json_dump(report, output)
    tabular = None
    if tsv_path is not None:
        tabular = Path(tsv_path).expanduser().resolve()
        tabular.parent.mkdir(parents=True, exist_ok=True)
        with tabular.open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "record_type",
                "section_id",
                "method",
                "aggregation",
                "gene_name",
                "metric",
                "value",
                "spots_evaluated",
                "status",
                "reason",
            )
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for method, payload in report["method_macro"].items():
                for metric, values in payload["metrics"].items():
                    macro_value = values["macro_mean"]
                    lower = values["minimum"]
                    upper = values["maximum"]
                    writer.writerow(
                        {
                            "record_type": "macro",
                            "section_id": "macro",
                            "method": method,
                            "aggregation": payload["aggregation"],
                            "gene_name": "",
                            "metric": metric,
                            "value": "" if macro_value is None else "%.12g" % macro_value,
                            "spots_evaluated": "",
                            "status": "missing" if macro_value is None else "ok",
                            "reason": (
                                "no specimen has an evaluable value"
                                if macro_value is None
                                else "specimen range %.12g to %.12g; n=%d"
                                % (lower, upper, values["specimens_evaluable"])
                            ),
                        }
                    )
            for case in report["cases"]:
                methods = case["methods"]
                for method, payload in methods.items():
                    for metric, value in payload["summary"].items():
                        writer.writerow(
                            {
                                "record_type": "summary",
                                "section_id": case["section_id"],
                                "method": method,
                                "aggregation": payload["aggregation"],
                                "gene_name": "",
                                "metric": metric,
                                "value": "" if value is None else "%.12g" % value,
                                "spots_evaluated": payload["spots_evaluated"],
                                "status": "missing" if value is None else "ok",
                                "reason": (
                                    "summary statistic is unavailable" if value is None else ""
                                ),
                            }
                        )
                    per_gene = payload["per_gene"]
                    gene_names = per_gene["gene_names"]
                    correlation_status = per_gene["correlation_status"]
                    correlation_reason = per_gene["correlation_reason"]
                    metric_names = (
                        name
                        for name in per_gene
                        if name not in {"gene_names", "correlation_status", "correlation_reason"}
                    )
                    for metric in metric_names:
                        for gene_index, gene_name in enumerate(gene_names):
                            value = per_gene[metric][gene_index]
                            status = "ok"
                            reason = ""
                            if metric in {"spearman", "pearson"}:
                                status = correlation_status[gene_index]
                                reason = correlation_reason[gene_index]
                            elif value is None:
                                status = (
                                    correlation_status[gene_index]
                                    if correlation_status[gene_index]
                                    == "excluded_observed_constant"
                                    else "not_evaluable"
                                )
                                reason = (
                                    correlation_reason[gene_index]
                                    if correlation_reason[gene_index]
                                    else "metric is undefined for this gene"
                                )
                            writer.writerow(
                                {
                                    "record_type": "gene",
                                    "section_id": case["section_id"],
                                    "method": method,
                                    "aggregation": payload["aggregation"],
                                    "gene_name": gene_name,
                                    "metric": metric,
                                    "value": "" if value is None else "%.12g" % value,
                                    "spots_evaluated": payload["spots_evaluated"],
                                    "status": status,
                                    "reason": reason,
                                }
                            )
    markdown = None
    if markdown_path is not None:
        markdown = Path(markdown_path).expanduser().resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        primary = report["primary"]
        lines = [
            "# snPATHO-DeepBench-v1 retrospective result",
            "",
            "This report does not replace or reinterpret `snPATHO-Locked-v0.2`.",
            "",
            "## Scope and benchmark contracts",
            "",
            "The available v0.2 artifacts form a retrospective capture-aware diagnostic. "
            "Although target H&E is an allowed input, v0.2 derived its OOD threshold from each "
            "target slide's 95th percentile. That target-specific calibration violates the "
            "external freeze, so these results establish neither Track A1 nor Track A2 "
            "compliance.",
            "",
            "The required primary R1 reference is the matched **FFPE snPATHO-seq-only** "
            "reference. The historical v0.2 reference instead pools FFPE snPATHO-seq, frozen "
            "Flex, and frozen 3-prime nuclei; its results and type-mean baseline are therefore "
            "retrospective diagnostics, not the primary R1 comparison.",
            "",
            "The available null is a complete shuffle of final cell records across assigned "
            "nuclei. It does not substitute for the separately required shuffled-image-feature "
            "and coordinate-shuffled-graph controls.",
            "",
            "Space Ranger supplies the common segmentation but no calibrated segmentation "
            "confidence. Historical v0.2 substituted a constant confidence of 1.0, making the "
            ">=0.50 anchor gate vacuous; refinement benchmarking remains blocked until that "
            "measurement is available.",
            "",
            "## Executability",
            "",
            "| Component | Status | Reason |",
            "|---|---|---|",
        ]
        lines.extend(
            "| %s | %s | %s |"
            % (item["component"], item["status"], str(item["reason"]).replace("|", "/"))
            for item in report["readiness"]
        )
        lines.extend(
            [
                "",
                "## Historical round-0 diagnostic",
                "",
                "| Specimen | Spearman delta vs type mean | MSE improvement vs type mean |",
                "|---|---:|---:|",
            ]
        )
        lines.extend(
            "| %s | %.6f | %.6f |"
            % (
                row["section_id"],
                row["median_gene_spearman_delta_vs_type_mean"],
                row["median_mse_improvement_vs_type_mean"],
            )
            for row in primary["specimens"]
        )
        lines.extend(
            [
                "",
                "## Equal-weight specimen macro summaries",
                "",
                "| Method | Median-gene Spearman | Median-gene MSE | Spot coverage |",
                "|---|---:|---:|---:|",
            ]
        )
        for method, payload in report["method_macro"].items():
            metrics = payload["metrics"]
            lines.append(
                "| %s | %.6f | %.6f | %.6f |"
                % (
                    method,
                    metrics["median_gene_spearman"]["macro_mean"],
                    metrics["median_gene_mse"]["macro_mean"],
                    metrics["spot_coverage"]["macro_mean"],
                )
            )
        lines.extend(
            [
                "",
                "Macro paired Spearman delta: **%.6f**." % primary["macro_delta"],
                "",
                "Requested refined-versus-type-mean endpoint: **%s**."
                % primary["requested_primary_status"],
                "",
                "Spot QC is a partial proxy. Inclusion in the processed RDS materializes the "
                "author-QC whitelist, but explicit per-spot exclusion flags/reasons and the "
                "required >=50% H&E tissue-fraction field are not present in the historical "
                "truth contract.",
            ]
        )
        markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, tabular, markdown


__all__ = [
    "DEEPBENCH_PLAN_SCHEMA",
    "DEEPBENCH_REPORT_SCHEMA",
    "DeepBenchPlan",
    "aggregate_cells_to_spots",
    "deepbench_expression_metrics",
    "load_deepbench_plan",
    "run_deepbench",
    "validate_deepbench_specification",
    "write_deepbench_report",
]
