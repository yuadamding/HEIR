"""Retrospective snPATHO-DeepBench evaluator.

DeepBench is deliberately separate from the immutable snPATHO locked report.
It reuses only hash-frozen predictions and truth, applies the stricter attached
diagnostic rules where the required fields exist, and records unavailable
tracks instead of silently substituting weaker evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from scipy import sparse  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from scipy.stats import rankdata  # type: ignore[import-untyped]

from heir.data import PrototypeSet, RNAReference, SpatialTruthArtifact
from heir.expression import EXPRESSION_TARGET_SUM
from heir.inference import PredictionBundle
from heir.prior.residual_geometry import RNAResidualGeometry
from heir.utils import atomic_json_dump, sha256_file

from .snpatho import SnPathoBenchmarkResult, load_snpatho_plan, run_snpatho_benchmark
from .spatial import morans_i

DEEPBENCH_PLAN_SCHEMA = "heir.snpatho_deepbench_plan.v1"
DEEPBENCH_REPORT_SCHEMA = "heir.snpatho_deepbench.v2"
DEEPBENCH_STATUS = "retrospective_diagnostic"
DEEPBENCH_SAMPLES = ("4066", "4399", "4411")
PRIMARY_METHOD = "heir_round0_historical_integrated_reference_library_size_weighted"
SELECTIVE_METHOD = "heir_round0_historical_integrated_reference_library_size_weighted_nonabstained"
EQUAL_CELL_METHOD = "heir_round0_equal_cell"
TYPE_MEAN_METHOD = "historical_integrated_hard_type_mean"
SOFT_TYPE_MEAN_METHOD = "historical_integrated_soft_type_mean"
HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD = "historical_integrated_hard_type_mean_hard_assigned_type_mass"
EQUAL_CELL_HARD_TYPE_MEAN_METHOD = "historical_integrated_hard_type_mean_equal_cell"
EQUAL_CELL_SOFT_TYPE_MEAN_METHOD = "historical_integrated_soft_type_mean_equal_cell"
PSEUDOBULK_METHOD = "historical_integrated_snrna_pseudobulk"
SHUFFLE_METHOD = (
    "heir_final_cell_record_shuffle_historical_integrated_reference_library_size_weighted"
)
R1_HARD_TYPE_MEAN_METHOD = "r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean"
R1_SOFT_TYPE_MEAN_METHOD = "r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean"
R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD = (
    "r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_hard_assigned_type_mass"
)
R1_EQUAL_CELL_HARD_TYPE_MEAN_METHOD = (
    "r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_equal_cell"
)
R1_EQUAL_CELL_SOFT_TYPE_MEAN_METHOD = (
    "r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean_equal_cell"
)
REFINED_R1_METHOD = "refined_heir_matched_ffpe_r1_reference_library_size_weighted"
REQUESTED_PRIMARY_CONTRAST = "refined_heir_minus_joint_matched_ffpe_r1_hard_and_soft_type_mean"
HISTORICAL_LIBRARY_SIZE_AGGREGATION = "historical_integrated_reference_library_size_weighted"
R1_LIBRARY_SIZE_AGGREGATION = (
    "ffpe_snpatho_reference_type_median_library_size_weighted_integrated_annotation_sensitivity"
)
R1_MANIFEST_SCHEMA = "heir.snpatho_r1_reference_manifest.v1"
REFINED_PREDICTION_MANIFEST_SCHEMA = "heir.snpatho_refined_prediction_manifest.v1"
NATIVE_SCANVI_MANIFEST_SCHEMA = "heir.snpatho_scanvi_r1_manifest.v1"
CLEAN_REANNOTATION_MANIFEST_SCHEMA = "heir.snpatho_clean_reannotation_manifest.v1"
FIVE_SEED_PREDICTION_MANIFEST_SCHEMA = "heir.snpatho_five_seed_refinement_manifest.v1"
REFINEMENT_RUN_MANIFEST_SCHEMA = "heir.snpatho_refinement_run_manifest.v2"
REFINEMENT_MATRIX_PUBLIC_SUMMARY_SCHEMA = "heir.snpatho_refinement_matrix_public_summary.v1"
REFINEMENT_MATRIX_REPORT_SCHEMA = "heir.snpatho_refinement_matrix.v1"
PRIMARY_GATE_SUPPORT_SCHEMA = "heir.snpatho_deepbench_primary_gate_support.v1"
REFINEMENT_MATRIX_CONTROLS = (
    "prototype_only",
    "image_shuffle",
    "graph_shuffle",
    "no_graph",
    "wrong_donor",
)

PRIMARY_MATCHED_R1_BASELINES = (
    R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD,
    R1_SOFT_TYPE_MEAN_METHOD,
)

PRIMARY_GATE_SUPPORT_CONTRACTS: Mapping[str, Tuple[str, str, Mapping[str, object]]] = {
    "composition_adjusted_residuals_hash_validated": (
        "spot_composition_covariates",
        "composition_adjusted_residuals",
        {
            "spatial_block_folds": 5,
            "independent_composition_covariates": True,
            "library_size_covariate": True,
            "pathologist_region_covariate": True,
            "primary_endpoint_evaluated": True,
            "primary_endpoint_pass": True,
        },
    ),
    "required_he_tissue_fraction_qc_hash_validated": (
        "author_qc_tissue_fraction",
        "he_tissue_fraction_qc",
        {
            "minimum_he_tissue_fraction": 0.50,
            "primary_spot_filter_applied": True,
            "explicit_author_qc_exclusion_flag": True,
            "explicit_author_qc_removal_reason": True,
        },
    ),
    "calibrated_segmentation_confidence_hash_validated": (
        "segmentation_sensitivity_predictions",
        "calibrated_segmentation_confidence",
        {
            "minimum_segmentation_confidence": 0.50,
            "calibrated_confidence_measurement": True,
            "independent_calibration_labels": True,
            "constant_substitution": False,
            "anchor_gate_applied": True,
        },
    ),
}

OPTIONAL_ARTIFACTS = (
    "primary_ffpe_snpatho_reference_manifest",
    "refined_predictions",
    "five_seed_predictions",
    "refinement_matrix_summary",
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


def _baseline_estimands() -> Dict[str, Dict[str, str]]:
    """Return the explicit profile-by-cell-mass baseline ladder.

    The historical method identifiers are retained for report compatibility.
    Their estimand metadata makes the previously implicit shared-soft-mass
    weighting visible, while the additive methods expose hard-assigned mass
    and equal-cell controls.
    """

    historical = "historical_integrated_multi_workflow_reference"
    r1 = "matched_ffpe_snpatho_count_reference_integrated_annotation_sensitivity"

    def estimand(
        reference: str,
        profile: str,
        mass: str,
        label: str,
        probability_source: str,
    ) -> Dict[str, str]:
        return {
            "reference": reference,
            "cell_expression_profile": profile,
            "cell_rna_mass": mass,
            "label": label,
            "cell_type_probability_source": probability_source,
        }

    historical_probability = "historical_round0_prediction"
    matched_r1_probability = (
        "refined_prediction_under_test_when_registered_else_historical_round0_sensitivity"
    )
    return {
        HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD: estimand(
            historical,
            "hard_argmax_type_profile",
            "hard_assigned_type_median_library_size",
            "hard profile + hard-assigned type mass",
            historical_probability,
        ),
        TYPE_MEAN_METHOD: estimand(
            historical,
            "hard_argmax_type_profile",
            "shared_soft_expected_type_median_library_size",
            "hard profile + shared soft mass",
            historical_probability,
        ),
        SOFT_TYPE_MEAN_METHOD: estimand(
            historical,
            "probability_weighted_soft_type_profile",
            "expected_soft_type_median_library_size",
            "soft profile + expected soft mass",
            historical_probability,
        ),
        EQUAL_CELL_HARD_TYPE_MEAN_METHOD: estimand(
            historical,
            "hard_argmax_type_profile",
            "equal_cell",
            "equal-cell hard type mean",
            historical_probability,
        ),
        EQUAL_CELL_SOFT_TYPE_MEAN_METHOD: estimand(
            historical,
            "probability_weighted_soft_type_profile",
            "equal_cell",
            "equal-cell soft type mean",
            historical_probability,
        ),
        R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD: estimand(
            r1,
            "hard_argmax_type_profile",
            "hard_assigned_type_median_library_size",
            "hard profile + hard-assigned type mass",
            matched_r1_probability,
        ),
        R1_HARD_TYPE_MEAN_METHOD: estimand(
            r1,
            "hard_argmax_type_profile",
            "shared_soft_expected_type_median_library_size",
            "hard profile + shared soft mass",
            matched_r1_probability,
        ),
        R1_SOFT_TYPE_MEAN_METHOD: estimand(
            r1,
            "probability_weighted_soft_type_profile",
            "expected_soft_type_median_library_size",
            "soft profile + expected soft mass",
            matched_r1_probability,
        ),
        R1_EQUAL_CELL_HARD_TYPE_MEAN_METHOD: estimand(
            r1,
            "hard_argmax_type_profile",
            "equal_cell",
            "equal-cell hard type mean",
            matched_r1_probability,
        ),
        R1_EQUAL_CELL_SOFT_TYPE_MEAN_METHOD: estimand(
            r1,
            "probability_weighted_soft_type_profile",
            "equal_cell",
            "equal-cell soft type mean",
            matched_r1_probability,
        ),
    }


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
    final_cell_record_shuffle_permutations: int
    primary_seeds: Tuple[int, ...]
    optional_artifacts: Mapping[str, Optional[Path]]
    optional_artifact_sha256: Mapping[str, Optional[str]]
    specification: Mapping[str, Any]


@dataclass(frozen=True)
class NativeResidualGeometry:
    """Recursively validated RNA residual geometry for one native specimen."""

    path: Path
    sha256: str
    type_names: Tuple[str, ...]
    rank: int
    latent_dim: int
    bounds: Tuple[float, ...]
    source_reference_sha256: str
    latent_transform_sha256: str
    basis: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class NativeScanviManifest:
    """Validated identity of the external native-scANVI molecular teacher."""

    path: Path
    sha256: str
    latent_space_id: str
    expression_space_id: str
    native_model_sha256: str
    decoder_sha256: str
    annotation_status: str
    specimen_prototype_sha256: Mapping[str, str]
    specimen_prototype_type_names: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    specimen_residual_geometry: Mapping[str, NativeResidualGeometry] = field(default_factory=dict)
    decoder_gene_names: Tuple[str, ...] = ()
    expression_normalization_contract: str = ""
    clean_reannotation_manifest_sha256: Optional[str] = None
    validated_clean_reannotation: bool = False

    @property
    def clean_annotation_complete(self) -> bool:
        return self.validated_clean_reannotation


@dataclass(frozen=True)
class RefinedPredictionArtifact:
    """A prediction whose model, refinement, telemetry, and RNA lineage are bound."""

    path: Path
    sha256: str
    prediction: PredictionBundle
    checkpoint: Path
    checkpoint_sha256: str
    refinement_audit: Path
    refinement_audit_sha256: str
    telemetry: Optional[Path]
    telemetry_sha256: Optional[str]
    refined_prototype: Path
    refined_prototype_sha256: str
    native_prototype_sha256: str


@dataclass(frozen=True)
class FiveSeedPredictionManifest:
    """Validated five-seed prediction matrix and its declared control coverage."""

    path: Path
    sha256: str
    predictions: Mapping[Tuple[str, int], PredictionBundle]
    control_names: Tuple[str, ...]
    execution_provenance_verified: bool = False


@dataclass(frozen=True)
class RefinementMatrixSummary:
    """Validated compact score matrix used by the full-primary evidence gate."""

    path: Path
    sha256: str
    matrix_status: str
    strict_ordering_status: str
    samples: Tuple[str, ...]
    seeds: Tuple[int, ...]
    control_seeds: Tuple[int, ...]
    control_names: Tuple[str, ...]
    requested_artifact_count: int
    scored_artifact_count: int
    overall_status: str
    primary_evidence_status: str
    evidence_blocker_count: int
    execution_provenance_blocker_count: int
    execution_provenance_verified: bool = False
    wrong_donor_coverage_complete: bool = False
    wrong_donor_pairing_count: int = 0
    expected_wrong_donor_pairing_count: int = 0
    missing_wrong_donor_case_count: int = 0

    @property
    def matrix_complete(self) -> bool:
        return (
            self.matrix_status == "complete"
            and self.scored_artifact_count == self.requested_artifact_count
            and self.wrong_donor_coverage_complete
        )

    @property
    def strict_ordering_pass(self) -> bool:
        return self.matrix_complete and self.strict_ordering_status == "pass"

    @property
    def required_followup_evidence_complete(self) -> bool:
        return (
            self.primary_evidence_status == "complete"
            and self.evidence_blocker_count == 0
            and self.execution_provenance_blocker_count == 0
        )


def _require_sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("%s must be a lowercase SHA-256 digest" % name)
    return digest


def _directory_sha256(path: Path) -> str:
    """Hash a directory using the native-scANVI training export contract."""

    if not path.is_dir():
        raise ValueError("native scANVI model directory is absent: %s" % path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("native scANVI model directory is empty")
    for source in files:
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _resolve_artifact(base: Path, value: object, name: str) -> Path:
    raw = str(value).strip()
    if not raw:
        raise ValueError("%s path is missing" % name)
    result = Path(raw).expanduser()
    if not result.is_absolute():
        result = (base / result).resolve()
    return result


def _validate_file_hash(path: Path, digest: object, name: str) -> str:
    expected = _require_sha256(digest, name + "_sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError("%s is absent or hash-mismatched" % name)
    return expected


def _gene_panel_order(path: Path) -> Tuple[str, ...]:
    """Read the canonical ordered gene column from a frozen panel TSV."""

    genes: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            gene = stripped.split("\t", 1)[0].strip()
            if gene:
                genes.append(gene)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("native scANVI gene panel must contain unique ordered genes")
    return tuple(genes)


def _validate_decoder_contract(
    decoder_path: Path,
    payload: Mapping[str, Any],
    native: Mapping[str, Any],
    repository_root: Path,
    sample_ids: Sequence[str],
    *,
    latent_space_id: str,
    expression_space_id: str,
) -> Tuple[Tuple[str, ...], str, int]:
    """Bind decoder gene order and normalization semantics to the frozen panel."""

    panel_sha256 = _require_sha256(payload.get("gene_panel_sha256"), "gene_panel_sha256")
    panel_path = _resolve_artifact(
        repository_root,
        payload.get("gene_panel", "manifests/gene_panel_snpatho_500.tsv"),
        "native_scanvi_gene_panel",
    )
    _validate_file_hash(panel_path, panel_sha256, "native_scanvi_gene_panel")
    panel_genes = _gene_panel_order(panel_path)
    expression_contract = str(payload.get("expression_transform", "")).strip()
    if not expression_contract:
        raise ValueError("native scANVI manifest lacks expression_transform")
    latent_dim = native.get("latent_dim")
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError("native scANVI latent_dim must be a positive integer")

    try:
        checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("distilled decoder checkpoint cannot be parsed") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("distilled decoder checkpoint must be a mapping")
    config = checkpoint.get("config")
    metadata = checkpoint.get("metadata")
    if not isinstance(config, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("distilled decoder lacks config or metadata")
    if metadata.get("schema") not in {
        "heir.scvi_distilled_decoder.v2",
        "heir.scvi_distilled_decoder.v3",
    }:
        raise ValueError("distilled decoder has an unsupported metadata schema")
    raw_gene_names = metadata.get("gene_names")
    if not isinstance(raw_gene_names, list) or any(
        not isinstance(value, str) or not value for value in raw_gene_names
    ):
        raise ValueError("distilled decoder gene_names must be a non-empty string list")
    gene_names = tuple(raw_gene_names)
    if gene_names != panel_genes:
        raise ValueError("distilled decoder gene order differs from the frozen gene panel")
    expected_metadata = {
        "latent_space_id": latent_space_id,
        "expression_space_id": expression_space_id,
        "expression_normalization_contract": expression_contract,
        "decoder_only": True,
    }
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise ValueError("distilled decoder %s differs from its manifest" % name)
    if tuple(metadata.get("training_donors", ())) != tuple(sample_ids):
        raise ValueError("distilled decoder training donors differ from DeepBench specimens")
    normalization = metadata.get("expression_normalization")
    expected_normalization = {
        "method": "scvi.get_normalized_expression",
        "library_size": float(EXPRESSION_TARGET_SUM),
        "library_basis": "full-transcriptome",
        "gene_selection": "after-library-normalization",
        "transform": "log1p",
        "version": 2,
    }
    if normalization != expected_normalization:
        raise ValueError("distilled decoder expression normalization contract is incomplete")
    expected_config = {
        "input_dim": len(gene_names),
        "latent_dim": latent_dim,
        "nonnegative_output": True,
    }
    for name, value in expected_config.items():
        if config.get(name) != value:
            raise ValueError("distilled decoder config %s differs from its manifest" % name)
    return gene_names, expression_contract, latent_dim


def _validate_clean_reannotation_contract(
    value: object,
    repository_root: Path,
    sample_ids: Sequence[str],
) -> str:
    """Validate the separately materialized independent-reannotation evidence."""

    if not isinstance(value, Mapping):
        raise ValueError(
            "independent clean reannotation status requires a hash-bound reannotation manifest"
        )
    manifest_path = _resolve_artifact(
        repository_root,
        value.get("manifest"),
        "clean_reannotation_manifest",
    )
    manifest_sha256 = _validate_file_hash(
        manifest_path,
        value.get("manifest_sha256"),
        "clean_reannotation_manifest",
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if not isinstance(contract, Mapping) or contract.get("schema") != (
        CLEAN_REANNOTATION_MANIFEST_SCHEMA
    ):
        raise ValueError("clean reannotation manifest has an unsupported schema")
    if contract.get("status") != "complete":
        raise ValueError("clean reannotation manifest is not complete")
    if contract.get("workflow_filter") != "processing_method == FFPE_snPATHO":
        raise ValueError("clean reannotation manifest does not isolate FFPE-snPATHO")
    if tuple(contract.get("sample_ids", ())) != tuple(sample_ids):
        raise ValueError("clean reannotation manifest does not cover DeepBench specimens")
    if contract.get("label_source") != "independent_clean_reannotation":
        raise ValueError("clean reannotation label source is not independent")
    if contract.get("qc_complete") is not True or contract.get("adjudication_complete") is not True:
        raise ValueError("clean reannotation QC and adjudication must both be complete")
    for role in ("annotation_table", "ontology", "qc_report", "adjudication_record"):
        artifact = contract.get(role)
        if not isinstance(artifact, Mapping):
            raise ValueError("clean reannotation manifest lacks %s" % role)
        artifact_path = _resolve_artifact(manifest_path.parent, artifact.get("path"), role)
        _validate_file_hash(artifact_path, artifact.get("sha256"), role)
    return manifest_sha256


def _validate_r1_reference_identity(
    reference: RNAReference,
    section_id: str,
    manifest_entry: Mapping[str, Any],
) -> None:
    """Bind an R1 count artifact to its specimen and upstream H5AD lineage."""

    expected_source = str(manifest_entry.get("h5ad_sha256", ""))
    expected_block = "%s_FFPE" % section_id
    if reference.sample_id != section_id:
        raise ValueError("R1 reference sample_id differs from its specimen")
    if set(np.asarray(reference.sample_ids).astype(str).tolist()) != {section_id}:
        raise ValueError("R1 per-cell sample IDs differ from their specimen")
    if set(np.asarray(reference.donor_ids).astype(str).tolist()) != {section_id}:
        raise ValueError("R1 per-cell donor IDs differ from their specimen")
    if reference.block_id != expected_block:
        raise ValueError("R1 reference block identity differs from its FFPE specimen")
    if not expected_source or reference.source_count_sha256 != expected_source:
        raise ValueError("R1 reference source-count lineage differs from its H5AD manifest")


def _load_native_scanvi_manifest(
    path: Optional[Path],
    sample_ids: Sequence[str],
    *,
    manifest_sha256: Optional[str] = None,
) -> Optional[NativeScanviManifest]:
    """Parse and recursively hash-check the native-scANVI identity manifest."""

    if path is None:
        return None
    if not path.is_file():
        raise ValueError("native scANVI manifest is absent")
    observed_manifest_sha256 = sha256_file(path)
    if manifest_sha256 is not None and observed_manifest_sha256 != _require_sha256(
        manifest_sha256, "native_scanvi_manifest_sha256"
    ):
        raise ValueError("native scANVI manifest is hash-mismatched")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") != NATIVE_SCANVI_MANIFEST_SCHEMA:
        raise ValueError("native scANVI manifest has an unsupported schema")
    native_status = str(payload.get("status", ""))
    supported_statuses = {
        "native_scanvi_with_published_integrated_annotation_sensitivity": (
            "published_integrated_annotation_sensitivity_not_clean_reannotation"
        ),
        "native_scanvi_with_independent_clean_reannotation": (
            "independent_clean_reannotation_complete"
        ),
    }
    if native_status not in supported_statuses:
        raise ValueError("native scANVI manifest status is not the declared sensitivity")
    if payload.get("workflow_filter") != "processing_method == FFPE_snPATHO":
        raise ValueError("native scANVI manifest does not isolate FFPE-snPATHO")
    annotation_status = str(
        payload.get(
            "annotation_status",
            supported_statuses[native_status],
        )
    )
    if annotation_status not in {
        "published_integrated_annotation_sensitivity_not_clean_reannotation",
        "independent_clean_reannotation_complete",
    }:
        raise ValueError("native scANVI annotation status is unsupported")
    if annotation_status != supported_statuses[native_status]:
        raise ValueError("native scANVI model and annotation statuses are inconsistent")
    if (
        annotation_status != "independent_clean_reannotation_complete"
        and not str(payload.get("annotation_provenance", "")).strip()
    ):
        raise ValueError("native scANVI integrated-label sensitivity lacks annotation provenance")

    native = payload.get("native_model")
    decoder = payload.get("distilled_decoder")
    if not isinstance(native, Mapping) or not isinstance(decoder, Mapping):
        raise ValueError("native scANVI manifest lacks model or decoder identity")
    repository_root = path.parent.parent
    native_path = _resolve_artifact(repository_root, native.get("external_path"), "native_model")
    native_sha256 = _require_sha256(native.get("sha256"), "native_model_sha256")
    if _directory_sha256(native_path) != native_sha256:
        raise ValueError("native scANVI model directory is hash-mismatched")
    decoder_path = _resolve_artifact(
        repository_root,
        decoder.get("external_path"),
        "distilled_decoder",
    )
    decoder_sha256 = _validate_file_hash(
        decoder_path,
        decoder.get("sha256"),
        "distilled_decoder",
    )
    latent_space_id = str(payload.get("latent_space_id", ""))
    if latent_space_id != "sha256:" + native_sha256:
        raise ValueError("native scANVI latent-space identity differs from its model hash")
    expression_space_id = str(payload.get("expression_space_id", "")).strip()
    if not expression_space_id:
        raise ValueError("native scANVI manifest lacks expression_space_id")
    decoder_gene_names, expression_contract, latent_dim = _validate_decoder_contract(
        decoder_path,
        payload,
        native,
        repository_root,
        sample_ids,
        latent_space_id=latent_space_id,
        expression_space_id=expression_space_id,
    )
    clean_reannotation_sha256: Optional[str] = None
    if annotation_status == "independent_clean_reannotation_complete":
        clean_reannotation_sha256 = _validate_clean_reannotation_contract(
            payload.get("clean_reannotation"),
            repository_root,
            sample_ids,
        )
    elif payload.get("clean_reannotation") is not None:
        raise ValueError(
            "clean reannotation evidence is inconsistent with integrated-label sensitivity"
        )

    specimens = payload.get("specimens")
    if not isinstance(specimens, Mapping) or set(specimens) != set(sample_ids):
        raise ValueError("native scANVI manifest must contain every DeepBench specimen")
    prototype_hashes: Dict[str, str] = {}
    prototype_type_names: Dict[str, Tuple[str, ...]] = {}
    residual_geometries: Dict[str, NativeResidualGeometry] = {}
    for section_id in sample_ids:
        raw = specimens[section_id]
        if not isinstance(raw, Mapping):
            raise ValueError("native scANVI specimen entry must be a mapping")
        latent_path = _resolve_artifact(
            repository_root,
            raw.get("latent_reference"),
            "native_latent_reference_%s" % section_id,
        )
        latent_reference_sha256 = _validate_file_hash(
            latent_path,
            raw.get("latent_reference_sha256"),
            "native_latent_reference_%s" % section_id,
        )
        reference = RNAReference.load_npz(latent_path)
        if (
            reference.sample_id != section_id
            or set(reference.sample_ids.astype(str).tolist()) != {section_id}
            or set(reference.donor_ids.astype(str).tolist()) != {section_id}
            or reference.latent_space_id != latent_space_id
            or reference.latent.shape != (reference.counts.shape[0], latent_dim)
        ):
            raise ValueError("native scANVI latent reference identity differs for %s" % section_id)
        if tuple(str(value) for value in reference.gene_ids.tolist()) != decoder_gene_names:
            raise ValueError("native scANVI reference gene order differs for %s" % section_id)
        if raw.get("cells") is not None and raw.get("cells") != reference.counts.shape[0]:
            raise ValueError("native scANVI reference cell count differs for %s" % section_id)
        prototype_path = _resolve_artifact(
            repository_root,
            raw.get("rare_complete_prototypes"),
            "native_prototypes_%s" % section_id,
        )
        prototype_sha256 = _validate_file_hash(
            prototype_path,
            raw.get("rare_complete_prototypes_sha256"),
            "native_prototypes_%s" % section_id,
        )
        prototypes = PrototypeSet.load_npz(prototype_path)
        if (
            prototypes.donor_id != section_id
            or set(prototypes.sample_ids.astype(str).tolist()) != {section_id}
            or prototypes.latent_space_id != latent_space_id
            or prototypes.means.shape[1] != latent_dim
            or prototypes.source_reference_sha256 != latent_reference_sha256
        ):
            raise ValueError("native scANVI prototype identity differs for %s" % section_id)
        ordered_prototype_types = tuple(
            dict.fromkeys(str(value) for value in prototypes.cell_type_labels.tolist())
        )
        reference_types = {str(value) for value in reference.cell_type_labels.tolist()}
        if set(ordered_prototype_types) != reference_types:
            raise ValueError(
                "native scANVI rare-complete prototypes omit reference types for %s" % section_id
            )

        geometry_path = _resolve_artifact(
            repository_root,
            raw.get("residual_geometry"),
            "native_residual_geometry_%s" % section_id,
        )
        geometry_sha256 = _validate_file_hash(
            geometry_path,
            raw.get("residual_geometry_sha256"),
            "native_residual_geometry_%s" % section_id,
        )
        geometry = RNAResidualGeometry.from_npz(geometry_path)
        geometry_types = tuple(str(value) for value in geometry.type_names.tolist())
        expected_cell_counts = np.asarray(
            [
                np.count_nonzero(reference.cell_type_labels.astype(str) == type_name)
                for type_name in geometry_types
            ],
            dtype=np.int64,
        )
        expected_prototype_counts = np.asarray(
            [
                np.count_nonzero(prototypes.cell_type_labels.astype(str) == type_name)
                for type_name in geometry_types
            ],
            dtype=np.int64,
        )
        if (
            geometry.latent_space_id != latent_space_id
            or geometry.latent_dim != latent_dim
            or geometry.source_reference_sha256 != latent_reference_sha256
            or geometry.latent_transform_sha256 != prototypes.latent_transform_sha256
            or geometry.training_donors != (section_id,)
            or geometry_types != ordered_prototype_types
            or not np.array_equal(geometry.n_cells, expected_cell_counts)
            or not np.array_equal(geometry.n_prototypes, expected_prototype_counts)
        ):
            raise ValueError("native residual geometry lineage differs for %s" % section_id)
        prototype_hashes[section_id] = prototype_sha256
        prototype_type_names[section_id] = ordered_prototype_types
        residual_geometries[section_id] = NativeResidualGeometry(
            path=geometry_path,
            sha256=geometry_sha256,
            type_names=geometry_types,
            rank=geometry.rank,
            latent_dim=geometry.latent_dim,
            bounds=tuple(float(value) for value in geometry.residual_type_max_norm.tolist()),
            source_reference_sha256=geometry.source_reference_sha256,
            latent_transform_sha256=geometry.latent_transform_sha256,
            basis=geometry.residual_type_basis,
        )
    if len({geometry.rank for geometry in residual_geometries.values()}) != 1:
        raise ValueError("native residual geometry rank differs across specimens")
    return NativeScanviManifest(
        path=path,
        sha256=observed_manifest_sha256,
        latent_space_id=latent_space_id,
        expression_space_id=expression_space_id,
        native_model_sha256=native_sha256,
        decoder_sha256=decoder_sha256,
        annotation_status=annotation_status,
        specimen_prototype_sha256=prototype_hashes,
        specimen_prototype_type_names=prototype_type_names,
        specimen_residual_geometry=residual_geometries,
        decoder_gene_names=decoder_gene_names,
        expression_normalization_contract=expression_contract,
        clean_reannotation_manifest_sha256=clean_reannotation_sha256,
        validated_clean_reannotation=clean_reannotation_sha256 is not None,
    )


def _negative_control_metadata_is_clean(value: object, section_id: str, seed: int) -> bool:
    if value is False:
        return True
    if not isinstance(value, Mapping):
        return False
    expected_false = (
        "graph_node_shuffle",
        "image_feature_shuffle",
        "no_graph",
        "prototype_only",
        "wrong_donor",
    )
    return (
        all(value.get(name) is False for name in expected_false)
        and str(value.get("prototype_donor_id", "")) == section_id
        and value.get("seed") == seed
    )


def _validate_refined_checkpoint_metadata(
    checkpoint: Path,
    *,
    section_id: str,
    seed: int,
    selected_round: int,
    native_scanvi: NativeScanviManifest,
    native_prototype_sha256: str,
) -> None:
    """Bind the refined model metadata to the native decoder and source prototype."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != "heir.model.v3":
        raise ValueError("refined checkpoint has an unsupported schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != "heir.refined_model.v1":
        raise ValueError("refined checkpoint lacks refined-model metadata")
    expected = {
        "seed": seed,
        "refinement_round": selected_round,
        "latent_space_id": native_scanvi.latent_space_id,
        "expression_space_id": native_scanvi.expression_space_id,
        "rna_vae_sha256": native_scanvi.decoder_sha256,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError("refined checkpoint %s lineage differs for %s" % (name, section_id))
    if metadata.get("training_donors") != [section_id] or metadata.get(
        "refinement_training_donors"
    ) != [section_id]:
        raise ValueError("refined checkpoint donor identity differs for %s" % section_id)
    geometry = native_scanvi.specimen_residual_geometry.get(section_id)
    if geometry is None:
        raise ValueError("native residual geometry is unavailable for %s" % section_id)
    metadata_geometry = _resolve_artifact(
        checkpoint.parent,
        metadata.get("residual_geometry"),
        "refined_checkpoint_residual_geometry",
    )
    if (
        metadata_geometry != geometry.path
        or metadata.get("residual_geometry_sha256") != geometry.sha256
    ):
        raise ValueError("refined checkpoint residual geometry lineage differs for %s" % section_id)
    if metadata.get("residual_basis_trainable") is not False:
        raise ValueError("refined checkpoint residual basis must remain frozen")
    if tuple(metadata.get("type_names", ())) != geometry.type_names:
        raise ValueError("refined checkpoint residual type order differs for %s" % section_id)
    if native_scanvi.decoder_gene_names and tuple(metadata.get("gene_names", ())) != (
        native_scanvi.decoder_gene_names
    ):
        raise ValueError("refined checkpoint decoder gene order differs for %s" % section_id)
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    checkpoint_geometry = payload.get("residual_geometry")
    if (
        not isinstance(config, Mapping)
        or not isinstance(state_dict, Mapping)
        or not isinstance(checkpoint_geometry, Mapping)
    ):
        raise ValueError("refined checkpoint lacks model or residual geometry state")
    expected_config = {
        "num_cell_types": len(geometry.type_names),
        "latent_dim": geometry.latent_dim,
        "residual_rank": geometry.rank,
    }
    if native_scanvi.decoder_gene_names:
        expected_config["expression_dim"] = len(native_scanvi.decoder_gene_names)
    for name, value in expected_config.items():
        if config.get(name) != value:
            raise ValueError("refined checkpoint residual %s differs for %s" % (name, section_id))
    if checkpoint_geometry.get("basis_trainable") is not False:
        raise ValueError("refined checkpoint serialized residual basis must remain frozen")
    if (
        checkpoint_geometry.get("type_max_norms") is None
        or state_dict.get("residual_type_basis") is None
    ):
        raise ValueError("refined checkpoint lacks residual basis or bounds")
    checkpoint_bounds = torch.as_tensor(checkpoint_geometry["type_max_norms"])
    expected_bounds = torch.as_tensor(geometry.bounds, dtype=checkpoint_bounds.dtype)
    if checkpoint_bounds.shape != expected_bounds.shape or not torch.allclose(
        checkpoint_bounds.cpu(), expected_bounds, rtol=1.0e-6, atol=1.0e-7
    ):
        raise ValueError("refined checkpoint residual bounds differ for %s" % section_id)
    checkpoint_basis = torch.as_tensor(state_dict["residual_type_basis"])
    expected_basis = torch.as_tensor(
        np.array(geometry.basis, copy=True), dtype=checkpoint_basis.dtype
    )
    if checkpoint_basis.shape != expected_basis.shape or not torch.allclose(
        checkpoint_basis.cpu(), expected_basis, rtol=1.0e-6, atol=1.0e-7
    ):
        raise ValueError("refined checkpoint residual basis differs for %s" % section_id)
    source_hashes = {
        str(digest)
        for collection_name in (
            "training_batches",
            "validation_batches",
            "refinement_training_batches",
            "refinement_validation_batches",
        )
        for row in (
            metadata.get(collection_name, [])
            if isinstance(metadata.get(collection_name, []), list)
            else []
        )
        if isinstance(row, Mapping)
        for digest in (
            row.get("source_sha256", []) if isinstance(row.get("source_sha256", []), list) else []
        )
    }
    if native_prototype_sha256 not in source_hashes:
        raise ValueError("refined checkpoint is not bound to its native prototype source")


def _load_refined_prediction_manifest(
    path: Optional[Path],
    sample_ids: Sequence[str],
    native_scanvi: Optional[NativeScanviManifest] = None,
) -> Dict[str, RefinedPredictionArtifact]:
    """Load a fully provenance-bound seed-17 refined sensitivity run."""

    if path is None:
        return {}
    if native_scanvi is None:
        raise ValueError("refined predictions require a validated native scANVI manifest")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        REFINED_PREDICTION_MANIFEST_SCHEMA
    ):
        raise ValueError("refined prediction manifest has an unsupported schema")
    if payload.get("analysis_role") != (
        "development_native_scanvi_published_annotation_sensitivity"
    ):
        raise ValueError("refined predictions are not declared as an integrated-label sensitivity")
    seed = payload.get("seed")
    selected_round = payload.get("selected_round")
    if seed != 17:
        raise ValueError("refined prediction manifest must freeze development seed 17")
    if (
        isinstance(selected_round, bool)
        or not isinstance(selected_round, int)
        or selected_round <= 0
    ):
        raise ValueError("refined prediction manifest must select a post-refinement round")
    if payload.get("round_selection_mode") != "fixed":
        raise ValueError("refined prediction manifest must use frozen fixed-round selection")
    if payload.get("native_scanvi_manifest_sha256") != native_scanvi.sha256:
        raise ValueError("refined predictions are not bound to the native scANVI manifest")
    native_manifest_path = _resolve_artifact(
        path.parent,
        payload.get("native_scanvi_manifest"),
        "native_scanvi_manifest",
    )
    if native_manifest_path != native_scanvi.path.resolve():
        raise ValueError("refined predictions name a different native scANVI manifest")
    if payload.get("latent_space_id") != native_scanvi.latent_space_id:
        raise ValueError("refined prediction latent space differs from native scANVI")
    if payload.get("expression_space_id") != native_scanvi.expression_space_id:
        raise ValueError("refined prediction expression space differs from native scANVI")
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError("refined prediction manifest cases must be a list")
    result: Dict[str, RefinedPredictionArtifact] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("refined prediction manifest case must be a mapping")
        section_id = str(raw.get("section_id", ""))
        if not section_id or section_id in result:
            raise ValueError("refined prediction manifest section IDs must be unique")
        prediction_path = _resolve_artifact(path.parent, raw.get("predictions"), "prediction")
        prediction_sha256 = _validate_file_hash(
            prediction_path,
            raw.get("predictions_sha256"),
            "refined_prediction_%s" % section_id,
        )
        prediction = PredictionBundle.from_npz(prediction_path)
        if prediction.sample_id != section_id or prediction.donor_id != section_id:
            raise ValueError("refined prediction sample/donor identity differs for %s" % section_id)
        if prediction.inference_seed != seed:
            raise ValueError("refined prediction inference seed differs for %s" % section_id)
        if prediction.refinement_round != selected_round or prediction.refinement_round <= 0:
            raise ValueError("refined prediction round differs for %s" % section_id)
        if prediction.latent_space_id != native_scanvi.latent_space_id:
            raise ValueError("refined prediction latent space differs for %s" % section_id)
        if prediction.expression_space_id != native_scanvi.expression_space_id:
            raise ValueError("refined prediction expression space differs for %s" % section_id)

        checkpoint = _resolve_artifact(path.parent, raw.get("checkpoint"), "checkpoint")
        checkpoint_sha256 = _validate_file_hash(
            checkpoint,
            raw.get("checkpoint_sha256"),
            "refined_checkpoint_%s" % section_id,
        )
        if prediction.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("refined prediction checkpoint lineage differs for %s" % section_id)
        refinement_audit = _resolve_artifact(
            path.parent,
            raw.get("refinement_audit"),
            "refinement_audit",
        )
        refinement_audit_sha256 = _validate_file_hash(
            refinement_audit,
            raw.get("refinement_audit_sha256"),
            "refinement_audit_%s" % section_id,
        )
        with refinement_audit.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        if not isinstance(audit, Mapping) or audit.get("selected_round") != selected_round:
            raise ValueError("refinement audit selected round differs for %s" % section_id)
        rounds = audit.get("rounds")
        if not isinstance(rounds, list) or not any(
            isinstance(round_row, Mapping)
            and round_row.get("round_id") == selected_round
            and round_row.get("committed") is True
            for round_row in rounds
        ):
            raise ValueError(
                "refinement audit lacks the committed selected round for %s" % section_id
            )

        refined_prototype = _resolve_artifact(
            path.parent,
            raw.get("refined_prototype"),
            "refined_prototype",
        )
        refined_prototype_sha256 = _validate_file_hash(
            refined_prototype,
            raw.get("refined_prototype_sha256"),
            "refined_prototype_%s" % section_id,
        )
        if prediction.prototype_sha256 != refined_prototype_sha256:
            raise ValueError("refined prediction prototype lineage differs for %s" % section_id)
        prototypes = PrototypeSet.load_npz(refined_prototype)
        if (
            prototypes.donor_id != section_id
            or set(prototypes.sample_ids.astype(str).tolist()) != {section_id}
            or prototypes.latent_space_id != native_scanvi.latent_space_id
        ):
            raise ValueError("refined prototype identity differs for %s" % section_id)
        prototype_artifacts = audit.get("prototype_artifacts")
        expected_key = "%s::%s" % (section_id, section_id)
        if not isinstance(prototype_artifacts, Mapping) or expected_key not in prototype_artifacts:
            raise ValueError("refinement audit lacks its refined prototype for %s" % section_id)
        audit_prototype = _resolve_artifact(
            refinement_audit.parent,
            prototype_artifacts[expected_key],
            "audited_refined_prototype",
        )
        if audit_prototype != refined_prototype:
            raise ValueError("refinement audit prototype path differs for %s" % section_id)
        native_prototype_sha256 = _require_sha256(
            raw.get("native_prototype_sha256"),
            "native_prototype_sha256",
        )
        if native_prototype_sha256 != native_scanvi.specimen_prototype_sha256.get(section_id):
            raise ValueError(
                "refined prediction native prototype lineage differs for %s" % section_id
            )
        _validate_refined_checkpoint_metadata(
            checkpoint,
            section_id=section_id,
            seed=seed,
            selected_round=selected_round,
            native_scanvi=native_scanvi,
            native_prototype_sha256=native_prototype_sha256,
        )

        telemetry_path: Optional[Path] = None
        telemetry_sha256: Optional[str] = None
        control_metadata: object = raw.get("negative_control", payload.get("negative_control"))
        if raw.get("telemetry") is not None or raw.get("telemetry_sha256") is not None:
            telemetry_path = _resolve_artifact(path.parent, raw.get("telemetry"), "telemetry")
            telemetry_sha256 = _validate_file_hash(
                telemetry_path,
                raw.get("telemetry_sha256"),
                "refined_telemetry_%s" % section_id,
            )
            with telemetry_path.open("r", encoding="utf-8") as handle:
                telemetry_payload = json.load(handle)
            if (
                not isinstance(telemetry_payload, Mapping)
                or telemetry_payload.get("schema") != "heir.inference_telemetry.v1"
                or telemetry_payload.get("prediction_sha256") != prediction_sha256
            ):
                raise ValueError(
                    "refined prediction telemetry identity differs for %s" % section_id
                )
            control_metadata = telemetry_payload.get("negative_control")
        if not _negative_control_metadata_is_clean(control_metadata, section_id, seed):
            raise ValueError("refined prediction is a negative control or lacks control provenance")
        result[section_id] = RefinedPredictionArtifact(
            path=prediction_path,
            sha256=prediction_sha256,
            prediction=prediction,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            refinement_audit=refinement_audit,
            refinement_audit_sha256=refinement_audit_sha256,
            telemetry=telemetry_path,
            telemetry_sha256=telemetry_sha256,
            refined_prototype=refined_prototype,
            refined_prototype_sha256=refined_prototype_sha256,
            native_prototype_sha256=native_prototype_sha256,
        )
    if set(result) != set(sample_ids):
        raise ValueError("refined prediction manifest must cover every DeepBench specimen")
    return result


def _load_five_seed_prediction_manifest(
    path: Optional[Path],
    sample_ids: Sequence[str],
    seeds: Sequence[int],
    native_scanvi: Optional[NativeScanviManifest],
) -> Optional[FiveSeedPredictionManifest]:
    """Validate the full specimen-by-seed matrix used to unlock primary claims."""

    if path is None:
        return None
    if native_scanvi is None:
        raise ValueError("five-seed predictions require a validated native scANVI manifest")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") not in {
        FIVE_SEED_PREDICTION_MANIFEST_SCHEMA,
        REFINEMENT_RUN_MANIFEST_SCHEMA,
    }:
        raise ValueError("five-seed prediction manifest has an unsupported schema")
    analysis_role = payload.get("analysis_role")
    expected_analysis_role = (
        "prespecified_five_seed_native_scanvi_clean_annotation_primary"
        if native_scanvi.clean_annotation_complete
        else "prespecified_five_seed_native_scanvi_integrated_annotation_sensitivity"
    )
    if analysis_role != expected_analysis_role:
        raise ValueError("five-seed predictions lack an explicit evidence role")
    expected_seeds = tuple(int(value) for value in seeds)
    if tuple(payload.get("seeds", ())) != expected_seeds:
        raise ValueError("five-seed prediction manifest does not use the prespecified seeds")
    if tuple(payload.get("samples", ())) != tuple(sample_ids):
        raise ValueError("five-seed prediction manifest does not use every specimen")
    if payload.get("negative_control") is not False:
        raise ValueError("five-seed primary predictions lack a no-negative-control declaration")
    if payload.get("native_scanvi_manifest_sha256") != native_scanvi.sha256:
        raise ValueError("five-seed predictions are not bound to native scANVI")
    if payload.get("latent_space_id") != native_scanvi.latent_space_id:
        raise ValueError("five-seed prediction latent space differs from native scANVI")
    if payload.get("expression_space_id") != native_scanvi.expression_space_id:
        raise ValueError("five-seed prediction expression space differs from native scANVI")
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError("five-seed prediction manifest cases must be a list")
    predictions: Dict[Tuple[str, int], PredictionBundle] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("five-seed prediction case must be a mapping")
        section_id = str(raw.get("section_id", ""))
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("five-seed prediction case seed must be an integer")
        key = (section_id, seed)
        if section_id not in sample_ids or seed not in expected_seeds or key in predictions:
            raise ValueError("five-seed prediction case identity is invalid or duplicated")
        prediction_path = _resolve_artifact(
            path.parent,
            raw.get("predictions"),
            "five_seed_prediction",
        )
        _validate_file_hash(
            prediction_path,
            raw.get("predictions_sha256"),
            "five_seed_prediction_%s_%d" % key,
        )
        prediction = PredictionBundle.from_npz(prediction_path)
        if (
            prediction.sample_id != section_id
            or prediction.donor_id != section_id
            or prediction.inference_seed != seed
            or prediction.refinement_round <= 0
            or prediction.latent_space_id != native_scanvi.latent_space_id
            or prediction.expression_space_id != native_scanvi.expression_space_id
        ):
            raise ValueError("five-seed PredictionBundle provenance differs for %s/%d" % key)
        predictions[key] = prediction
    expected = {(section_id, seed) for section_id in sample_ids for seed in expected_seeds}
    if set(predictions) != expected:
        raise ValueError("five-seed prediction manifest does not cover the full matrix")
    controls = payload.get("controls_available", [])
    if not isinstance(controls, list) or any(not isinstance(value, str) for value in controls):
        raise ValueError("five-seed controls_available must be a string list")
    return FiveSeedPredictionManifest(
        path=path,
        sha256=sha256_file(path),
        predictions=predictions,
        control_names=tuple(sorted(set(controls))),
        execution_provenance_verified=bool(
            payload.get("schema") == REFINEMENT_RUN_MANIFEST_SCHEMA
            and isinstance(payload.get("execution"), Mapping)
            and payload["execution"].get("execution_provenance_verified") is True
        ),
    )


def _load_refinement_matrix_summary(
    path: Optional[Path],
    sample_ids: Sequence[str],
    seeds: Sequence[int],
    control_seeds: Sequence[int],
    *,
    minimum_nuclei: int,
    frozen_plan_sha256: str,
    native_scanvi: Optional[NativeScanviManifest],
    summary_sha256: Optional[str],
) -> Optional[RefinementMatrixSummary]:
    """Validate the compact matrix result consumed by full-primary gating.

    This parser deliberately validates the score coverage and outcome rather
    than treating the existence of the five-seed PredictionBundle manifest as
    evidence that the prespecified comparisons passed.
    """

    if path is None:
        if summary_sha256 is not None:
            raise ValueError("refinement matrix summary hash is set without a path")
        return None
    if native_scanvi is None:
        raise ValueError("refinement matrix summary requires a validated native scANVI manifest")
    if summary_sha256 is None:
        raise ValueError("refinement matrix summary must be hash-bound by the DeepBench plan")
    validated_sha256 = _validate_file_hash(
        path,
        summary_sha256,
        "refinement_matrix_summary",
    )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        REFINEMENT_MATRIX_PUBLIC_SUMMARY_SCHEMA
    ):
        raise ValueError("refinement matrix summary has an unsupported schema")
    if payload.get("report_schema") != REFINEMENT_MATRIX_REPORT_SCHEMA:
        raise ValueError("refinement matrix summary report schema is invalid")
    expected_role = (
        "native_scanvi_clean_independent_reannotation_primary"
        if native_scanvi.clean_annotation_complete
        else "native_scanvi_published_integrated_annotation_sensitivity"
    )
    if payload.get("analysis_role") != expected_role:
        raise ValueError("refinement matrix summary analysis role differs from native scANVI")

    request = payload.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("refinement matrix summary request must be a mapping")

    def integer_tuple(value: object, name: str) -> Tuple[int, ...]:
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise ValueError("refinement matrix %s must be an integer list" % name)
        return tuple(value)

    requested_samples = request.get("samples")
    if not isinstance(requested_samples, list) or any(
        not isinstance(value, str) for value in requested_samples
    ):
        raise ValueError("refinement matrix request samples must be a string list")
    expected_samples = tuple(str(value) for value in sample_ids)
    if tuple(requested_samples) != expected_samples:
        raise ValueError("refinement matrix summary does not cover the requested specimens")
    expected_seeds = tuple(int(value) for value in seeds)
    requested_seeds = integer_tuple(request.get("seeds"), "request seeds")
    if requested_seeds != expected_seeds:
        raise ValueError("refinement matrix summary does not cover the prespecified seeds")
    expected_control_seeds = tuple(int(value) for value in control_seeds)
    requested_control_seeds = integer_tuple(
        request.get("control_seeds"),
        "request control seeds",
    )
    if requested_control_seeds != expected_control_seeds:
        raise ValueError("refinement matrix summary does not cover the ablation seeds")
    requested_controls = request.get("controls")
    if not isinstance(requested_controls, list) or tuple(requested_controls) != (
        REFINEMENT_MATRIX_CONTROLS
    ):
        raise ValueError("refinement matrix summary does not cover the required controls")
    trajectory_seed = request.get("trajectory_seed")
    if (
        isinstance(trajectory_seed, bool)
        or not isinstance(trajectory_seed, int)
        or not expected_seeds
        or trajectory_seed != expected_seeds[0]
    ):
        raise ValueError("refinement matrix trajectory seed is not the prespecified seed")
    if request.get("minimum_nuclei") != minimum_nuclei:
        raise ValueError("refinement matrix minimum-nuclei policy differs from DeepBench")
    raw_pairings = request.get("wrong_donor_pairings")
    if raw_pairings is None:
        legacy_target = request.get("wrong_donor_target")
        legacy_source = request.get("wrong_donor_source")
        raw_pairings = [{"target": legacy_target, "source": legacy_source}]
    if not isinstance(raw_pairings, list):
        raise ValueError("refinement matrix wrong-donor pairings must be a list")
    wrong_donor_pairings = []
    for row in raw_pairings:
        if not isinstance(row, Mapping):
            raise ValueError("refinement matrix wrong-donor pairing is malformed")
        target = row.get("target")
        source = row.get("source")
        if target not in expected_samples or source not in expected_samples or target == source:
            raise ValueError("refinement matrix wrong-donor request is invalid")
        wrong_donor_pairings.append((str(target), str(source)))
    if len(set(wrong_donor_pairings)) != len(wrong_donor_pairings):
        raise ValueError("refinement matrix wrong-donor pairings are duplicated")
    expected_wrong_donor_pairings = {
        (target, source)
        for target in expected_samples
        for source in expected_samples
        if source != target
    }
    wrong_donor_coverage_complete = set(wrong_donor_pairings) == expected_wrong_donor_pairings
    missing_wrong_donor_pairings = expected_wrong_donor_pairings - set(wrong_donor_pairings)

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("refinement matrix artifact counts must be a mapping")
    requested_count = artifacts.get("requested")
    scored_count = artifacts.get("scored")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (requested_count, scored_count)
    ):
        raise ValueError("refinement matrix artifact counts must be non-negative integers")
    expected_count = (
        2 * len(expected_samples) * len(expected_seeds)
        + 3 * len(expected_samples)
        + 4 * len(expected_samples) * len(expected_control_seeds)
        + len(wrong_donor_pairings) * len(expected_control_seeds)
    )
    if requested_count != expected_count or scored_count > requested_count:
        raise ValueError("refinement matrix requested artifact coverage is inconsistent")

    matrix_status = payload.get("matrix_status")
    strict_status = payload.get("strict_ordering_status")
    if matrix_status not in {"complete", "blocked"}:
        raise ValueError("refinement matrix completeness status is invalid")
    if strict_status not in {"pass", "fail", "blocked"}:
        raise ValueError("refinement matrix strict-ordering status is invalid")
    matrix_complete = requested_count == scored_count == expected_count
    if (matrix_status == "complete") != matrix_complete:
        raise ValueError("refinement matrix completeness status contradicts artifact coverage")

    strict = payload.get("strict_ordering")
    if not isinstance(strict, Mapping) or strict.get("status") != strict_status:
        raise ValueError("refinement matrix strict-ordering statuses are inconsistent")
    by_check = strict.get("by_check")
    check_counts = strict.get("check_counts")
    if not isinstance(by_check, Mapping) or not isinstance(check_counts, Mapping):
        raise ValueError("refinement matrix strict-ordering coverage is malformed")
    sample_seed_count = len(expected_samples) * len(expected_seeds)
    control_case_count = len(expected_samples) * len(expected_control_seeds)
    expected_check_totals = {
        "refined_gt_round0": sample_seed_count,
        "refined_gt_hard_baseline": sample_seed_count,
        "refined_gt_soft_baseline": sample_seed_count,
        "refined_gt_prototype_only": control_case_count,
        "round0_gt_prototype_only": control_case_count,
        "refined_gt_image_shuffle": control_case_count,
        "refined_gt_graph_shuffle": control_case_count,
        "refined_gt_no_graph": control_case_count,
        "refined_gt_wrong_donor": len(wrong_donor_pairings) * len(expected_control_seeds),
    }
    if set(by_check) != set(expected_check_totals):
        raise ValueError("refinement matrix strict-ordering checks are incomplete")
    observed_counts = {"total": 0, "pass": 0, "fail": 0, "blocked": 0}
    for name, expected_total in expected_check_totals.items():
        row = by_check[name]
        if not isinstance(row, Mapping):
            raise ValueError("refinement matrix strict-ordering check is malformed")
        values = {key: row.get(key) for key in observed_counts}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError("refinement matrix strict-ordering counts are malformed")
        if values["total"] != expected_total or values["total"] != (
            values["pass"] + values["fail"] + values["blocked"]
        ):
            raise ValueError("refinement matrix strict-ordering check coverage is inconsistent")
        for key, value in values.items():
            observed_counts[key] += value
    for key, value in observed_counts.items():
        if check_counts.get(key) != value:
            raise ValueError("refinement matrix strict-ordering aggregate counts are inconsistent")
    if matrix_complete:
        if observed_counts["blocked"]:
            raise ValueError("complete refinement matrix contains blocked ordering checks")
        expected_strict_status = "fail" if observed_counts["fail"] else "pass"
    else:
        expected_strict_status = "blocked"
    if strict_status != expected_strict_status:
        raise ValueError("refinement matrix strict-ordering outcome contradicts check counts")

    blockers = payload.get("blockers")
    if not isinstance(blockers, Mapping):
        raise ValueError("refinement matrix blockers must be a mapping")
    matrix_blocker_count = blockers.get("matrix_count")
    if (
        isinstance(matrix_blocker_count, bool)
        or not isinstance(matrix_blocker_count, int)
        or matrix_blocker_count < 0
        or (matrix_complete and matrix_blocker_count != 0)
        or (not matrix_complete and matrix_blocker_count == 0)
    ):
        raise ValueError("refinement matrix blocker count contradicts completeness")

    count_names = (
        "total_count",
        "matrix_count",
        "evidence_count",
        "execution_provenance_count",
    )
    blocker_counts = {name: blockers.get(name) for name in count_names}
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in blocker_counts.values()
    ):
        raise ValueError("refinement matrix blocker counts must be non-negative integers")
    evidence_blocker_count = blocker_counts["evidence_count"]
    execution_blocker_count = blocker_counts["execution_provenance_count"]
    if blocker_counts["total_count"] != (
        blocker_counts["matrix_count"] + evidence_blocker_count + execution_blocker_count
    ):
        raise ValueError("refinement matrix blocker category counts are inconsistent")
    by_code = blockers.get("by_code")
    by_requirement = blockers.get("by_requirement")
    groups = blockers.get("groups")
    if (
        not isinstance(by_code, Mapping)
        or not isinstance(by_requirement, Mapping)
        or not isinstance(groups, list)
    ):
        raise ValueError("refinement matrix compact blocker detail is malformed")

    def _validated_count_sum(values: Mapping[str, Any], name: str) -> int:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError("refinement matrix %s counts are malformed" % name)
        return sum(int(value) for value in values.values())

    if _validated_count_sum(by_code, "blocker-code") != blocker_counts["total_count"]:
        raise ValueError("refinement matrix blocker-code counts are inconsistent")
    if _validated_count_sum(by_requirement, "blocker-requirement") > blocker_counts["total_count"]:
        raise ValueError("refinement matrix blocker-requirement counts are inconsistent")
    grouped_count = 0
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("refinement matrix blocker group is malformed")
        count = group.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("refinement matrix blocker group count is malformed")
        grouped_count += count
    if grouped_count != blocker_counts["total_count"]:
        raise ValueError("refinement matrix blocker group counts are inconsistent")

    primary_evidence_status = payload.get("primary_evidence_status")
    if primary_evidence_status not in {"complete", "blocked"}:
        raise ValueError("refinement matrix primary-evidence status is invalid")
    expected_primary_evidence_status = (
        "complete" if evidence_blocker_count == 0 and execution_blocker_count == 0 else "blocked"
    )
    if primary_evidence_status != expected_primary_evidence_status:
        raise ValueError("refinement matrix primary-evidence status contradicts blocker counts")
    execution_provenance_verified = payload.get("execution_provenance_verified")
    if not isinstance(execution_provenance_verified, bool) or (
        execution_provenance_verified != (execution_blocker_count == 0)
    ):
        raise ValueError("refinement matrix execution-provenance status contradicts blocker counts")
    overall_status = payload.get("status")
    expected_overall_status = (
        "blocked_matrix"
        if not matrix_complete
        else (
            "blocked_evidence"
            if primary_evidence_status == "blocked"
            else ("complete_ordering_failed" if strict_status == "fail" else "complete")
        )
    )
    if overall_status != expected_overall_status:
        raise ValueError("refinement matrix overall status contradicts component statuses")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("refinement matrix provenance must be a mapping")
    manifests = provenance.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("refinement matrix manifest provenance must be a mapping")
    native_provenance = manifests.get("native_r1")
    truth_provenance = manifests.get("frozen_truth")
    if not isinstance(native_provenance, Mapping) or not isinstance(truth_provenance, Mapping):
        raise ValueError("refinement matrix manifest provenance is incomplete")
    if native_provenance.get("sha256") != native_scanvi.sha256:
        raise ValueError("refinement matrix is not bound to the native scANVI manifest")
    if truth_provenance.get("sha256") != frozen_plan_sha256:
        raise ValueError("refinement matrix is not bound to the frozen truth plan")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_samples):
        raise ValueError("refinement matrix input provenance does not cover every specimen")
    for sample in expected_samples:
        row = inputs[sample]
        if not isinstance(row, Mapping):
            raise ValueError("refinement matrix specimen provenance is malformed")
        truth = row.get("truth")
        reference = row.get("native_r1_reference")
        if not isinstance(truth, Mapping) or not isinstance(reference, Mapping):
            raise ValueError("refinement matrix specimen provenance is incomplete")
        if truth.get("hash_validation") != "matched_frozen_truth_manifest":
            raise ValueError("refinement matrix truth provenance is invalid")
        if reference.get("hash_validation") != "matched_native_scanvi_manifest":
            raise ValueError("refinement matrix native-reference provenance is invalid")
        _require_sha256(truth.get("sha256"), "refinement_matrix_truth_%s_sha256" % sample)
        _require_sha256(
            reference.get("sha256"),
            "refinement_matrix_reference_%s_sha256" % sample,
        )

    if not wrong_donor_coverage_complete:
        matrix_status = "blocked"
        strict_status = "blocked"
        overall_status = "blocked_matrix"

    return RefinementMatrixSummary(
        path=path,
        sha256=validated_sha256,
        matrix_status=str(matrix_status),
        strict_ordering_status=str(strict_status),
        samples=expected_samples,
        seeds=expected_seeds,
        control_seeds=expected_control_seeds,
        control_names=REFINEMENT_MATRIX_CONTROLS,
        requested_artifact_count=requested_count,
        scored_artifact_count=scored_count,
        overall_status=str(overall_status),
        primary_evidence_status=str(primary_evidence_status),
        evidence_blocker_count=evidence_blocker_count,
        execution_provenance_blocker_count=execution_blocker_count,
        execution_provenance_verified=execution_provenance_verified,
        wrong_donor_coverage_complete=wrong_donor_coverage_complete,
        wrong_donor_pairing_count=len(wrong_donor_pairings),
        expected_wrong_donor_pairing_count=len(expected_wrong_donor_pairings),
        missing_wrong_donor_case_count=(
            len(missing_wrong_donor_pairings) * len(expected_control_seeds)
        ),
    )


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
    _require_equal(
        payload,
        "spaceranger_4_1_frozen_common_segmentation",
        "image",
        "segmentation_primary",
    )
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
    _require_equal(payload, "within_type_rna_pca_frozen", "rna", "residual_basis")
    _require_equal(payload, "type_specific_molecular_geometry", "rna", "residual_bound")
    _require_equal(payload, "strongly_shrunken_single_prototype", "rna", "rare_type_policy")
    _require_equal(payload, True, "rna", "retain_rare_types")
    _require_equal(payload, 500, "targets", "genes")
    _require_equal(payload, 15, "targets", "programs", "published_robust_nmf_clusters")
    _require_equal(payload, "detached_uot_responsibilities", "refinement", "e_step")
    _require_equal(payload, True, "refinement", "anchors_revocable")
    _require_equal(payload, 4, "refinement", "maximum_rounds")
    _require_equal(payload, 2, "refinement", "broad_refinement_rounds")
    _require_equal(payload, 0.90, "refinement", "minimum_probability")
    _require_equal(payload, 0.20, "refinement", "maximum_normalized_entropy")
    _require_equal(payload, 0.50, "refinement", "minimum_segmentation_confidence")
    _require_equal(payload, 2, "refinement", "minimum_view_agreement")
    _require_equal(payload, 2, "refinement", "trusted_after_consecutive_rounds")
    _require_equal(payload, 0.70, "refinement", "revoke_probability_threshold")
    _require_equal(payload, 0.05, "refinement", "uot_unknown_mass")
    _require_equal(payload, "fixed", "refinement", "uot_unknown_mass_mode")
    _require_equal(
        payload,
        [0.0, 0.01, 0.05, 0.1, 0.2],
        "refinement",
        "unknown_mass_sensitivity",
    )
    _require_equal(payload, [17, 41, 89, 131, 197], "randomness", "primary_seeds")
    _require_equal(payload, [17, 41, 89], "randomness", "ablation_seeds")
    _require_equal(
        payload,
        "joint_matched_ffpe_r1_hard_and_soft_type_means",
        "evaluation",
        "primary_baseline",
    )
    _require_equal(
        payload,
        [
            "matched_ffpe_r1_hard_type_mean_hard_assigned_mass",
            "matched_ffpe_r1_soft_type_mean_expected_soft_mass",
        ],
        "evaluation",
        "primary_baselines",
    )
    _require_equal(payload, True, "evaluation", "success_requires_all_primary_baselines")
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
            "expression_detection_auroc",
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
    shuffle_permutations = _nested(
        payload,
        "statistics",
        "final_cell_record_shuffle_permutations",
    )
    if (
        isinstance(shuffle_permutations, bool)
        or not isinstance(shuffle_permutations, int)
        or shuffle_permutations < 100
    ):
        raise ValueError(
            "DeepBench final_cell_record_shuffle_permutations must be an integer >= 100"
        )
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
        "prototype_only_no_residual",
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
        final_cell_record_shuffle_permutations=int(
            _nested(payload, "statistics", "final_cell_record_shuffle_permutations")
        ),
        primary_seeds=tuple(
            int(value) for value in _nested(payload, "randomness", "primary_seeds")
        ),
        optional_artifacts=optional,
        optional_artifact_sha256=optional_hashes,
        specification=dict(payload),
    )


def _load_r1_references(
    plan: DeepBenchPlan,
    *,
    gene_panel_sha256: str,
) -> Dict[str, Tuple[RNAReference, PrototypeSet, Dict[str, Any]]]:
    """Load hash-bound FFPE-only references as annotation-sensitivity controls."""

    manifest = plan.optional_artifacts["primary_ffpe_snpatho_reference_manifest"]
    if manifest is None:
        return {}
    with manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema") != R1_MANIFEST_SCHEMA:
        raise ValueError("R1 reference manifest has an unsupported schema")
    if payload.get("filter") != {
        "column": "processing_method",
        "accepted_values": ["FFPE_snPATHO"],
        "matching": "exact",
    }:
        raise ValueError("R1 reference manifest does not freeze the exact FFPE filter")
    panel = payload.get("gene_panel")
    if not isinstance(panel, Mapping) or panel.get("sha256") != gene_panel_sha256:
        raise ValueError("R1 reference manifest uses a different frozen gene panel")
    annotation = payload.get("cell_type_annotation")
    if (
        not isinstance(annotation, Mapping)
        or annotation.get("primary_clean_reannotation_status") != "not_complete"
    ):
        raise ValueError("R1 annotation status must remain explicit and fail closed")
    prototype_adapter = payload.get("development_prototype_adapter")
    if not isinstance(prototype_adapter, Mapping) or prototype_adapter.get("status") != (
        "materialized_svd_fallback_not_primary_scANVI"
    ):
        raise ValueError("R1 development prototype status is not explicit")
    specimens = payload.get("specimens")
    if not isinstance(specimens, Mapping) or set(specimens) != set(plan.sample_ids):
        raise ValueError("R1 reference manifest must contain exactly the DeepBench specimens")
    repository_root = manifest.parent.parent
    references: Dict[str, Tuple[RNAReference, PrototypeSet, Dict[str, Any]]] = {}
    for section_id in plan.sample_ids:
        raw = specimens[section_id]
        if not isinstance(raw, Mapping):
            raise ValueError("R1 specimen manifest entry must be a mapping")
        reference_path = (repository_root / str(raw["panel_reference"])).resolve()
        expected_sha256 = str(raw["panel_reference_sha256"])
        if not reference_path.is_file() or sha256_file(reference_path) != expected_sha256:
            raise ValueError("R1 panel reference is absent or hash-mismatched: %s" % section_id)
        reference = RNAReference.load_npz(reference_path)
        _validate_r1_reference_identity(reference, section_id, raw)
        selected = int(raw["selected_observations"])
        if reference.counts.shape[0] != selected:
            raise ValueError("R1 reference row count differs from its manifest")
        expected_counts = raw.get("cell_type_counts")
        if not isinstance(expected_counts, Mapping):
            raise ValueError("R1 reference manifest lacks cell-type counts")
        labels, counts = np.unique(
            np.asarray(reference.cell_type_labels).astype(str),
            return_counts=True,
        )
        observed_counts = {str(name): int(count) for name, count in zip(labels, counts)}
        if observed_counts != {str(name): int(count) for name, count in expected_counts.items()}:
            raise ValueError("R1 reference cell-type counts differ from its manifest")
        prototype_path = (repository_root / str(raw["prototypes"])).resolve()
        prototype_sha256 = str(raw["prototypes_sha256"])
        if not prototype_path.is_file() or sha256_file(prototype_path) != prototype_sha256:
            raise ValueError("R1 prototype bank is absent or hash-mismatched: %s" % section_id)
        prototypes = PrototypeSet.load_npz(prototype_path)
        if set(np.asarray(prototypes.sample_ids).astype(str).tolist()) != {section_id}:
            raise ValueError("R1 prototype sample identity differs from its specimen")
        if prototypes.donor_id != section_id or prototypes.block_id != "%s_FFPE" % section_id:
            raise ValueError("R1 prototype donor/block identity differs from its specimen")
        if prototypes.source_reference_sha256 != str(raw["latent_reference_sha256"]):
            raise ValueError("R1 prototype source-reference lineage differs from its manifest")
        count_types = set(observed_counts)
        prototype_types = set(np.asarray(prototypes.cell_type_labels).astype(str).tolist())
        expected_supported = {str(value) for value in raw["prototype_supported_types"]}
        expected_omitted = {str(value) for value in raw["prototype_omitted_rare_types"]}
        if prototype_types != expected_supported:
            raise ValueError("R1 prototype-supported types differ from its manifest")
        if prototype_types - count_types or count_types - prototype_types != expected_omitted:
            raise ValueError("R1 prototype-omitted types differ from its count reference")
        references[section_id] = (
            reference,
            prototypes,
            {
                "manifest_sha256": plan.optional_artifact_sha256[
                    "primary_ffpe_snpatho_reference_manifest"
                ],
                "reference_path": str(reference_path),
                "reference_sha256": expected_sha256,
                "prototype_path": str(prototype_path),
                "prototype_sha256": prototype_sha256,
                "selected_observations": selected,
                "filter": dict(cast(Mapping[str, Any], payload["filter"])),
                "annotation_provenance": annotation.get("provenance"),
                "development_prototype_status": prototype_adapter.get("status"),
                "status": ("retrospective_integrated_annotation_sensitivity_not_primary_clean_R1"),
            },
        )
    return references


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
    """Build type profiles in linear space, failing closed on unsupported types."""

    counts = sparse.csr_matrix(reference.counts, dtype=np.float64)
    if reference.library_sizes is None:
        raise ValueError("reference is missing full-transcriptome library sizes")
    library = np.asarray(reference.library_sizes, dtype=np.float64)
    if library.shape != (counts.shape[0],) or np.any(library <= 0):
        raise ValueError("reference requires positive full-transcriptome library sizes")
    labels = np.asarray(reference.cell_type_labels).astype(str)
    requested = tuple(str(value) for value in type_names)
    supported = set(labels.tolist())
    missing = tuple(name for name in requested if name not in supported)
    if missing:
        raise ValueError(
            "historical integrated reference is missing prediction cell types; "
            "global-profile fallback is prohibited: %s" % ", ".join(missing)
        )
    profiles = []
    median_library_sizes = []
    for name in requested:
        selected = labels == name
        pooled = np.asarray(counts[selected].sum(axis=0)).reshape(-1)
        profiles.append(pooled * (EXPRESSION_TARGET_SUM / library[selected].sum()))
        median_library_sizes.append(float(np.median(library[selected])))
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


def _normalized_type_probabilities(prediction: PredictionBundle) -> np.ndarray:
    probabilities = np.asarray(prediction.type_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(prediction.type_names):
        raise ValueError("prediction type probabilities are misaligned")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("prediction type probabilities must be finite and non-negative")
    total = probabilities.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("prediction type probabilities require positive row mass")
    return probabilities / total


def _type_map_diagnostics(
    prediction: PredictionBundle,
    spot_index: np.ndarray,
    evaluated_spots: np.ndarray,
) -> Dict[str, Any]:
    """Audit hard occupancy, soft uncertainty, and spot-mixture variability."""

    probabilities = _normalized_type_probabilities(prediction)
    indices = np.asarray(spot_index, dtype=np.int64)
    selected_spots = np.asarray(evaluated_spots, dtype=bool)
    if indices.shape != (len(probabilities),):
        raise ValueError("type-map spot assignments must align to prediction cells")
    if selected_spots.ndim != 1 or np.any(indices >= len(selected_spots)):
        raise ValueError("type-map evaluated spots are misaligned")
    assigned = np.flatnonzero(indices >= 0)
    selected_cells = assigned[selected_spots[indices[assigned]]]
    selected_spot_indices = np.flatnonzero(selected_spots)
    if not len(selected_cells) or not len(selected_spot_indices):
        raise ValueError("type-map diagnostics require assigned cells in evaluated spots")
    names = tuple(str(value) for value in prediction.type_names.tolist())
    hard = probabilities.argmax(axis=1)
    hard_counts = np.bincount(hard[selected_cells], minlength=len(names)).astype(np.int64)
    cell_count_by_spot = np.bincount(
        indices[selected_cells],
        minlength=len(selected_spots),
    ).astype(np.float64)
    if np.any(cell_count_by_spot[selected_spots] <= 0):
        raise ValueError("type-map evaluated spots require at least one assigned cell")
    spot_mixture: Dict[str, Any] = {}
    for type_index, name in enumerate(names):
        hard_spot_count = np.bincount(
            indices[selected_cells],
            weights=(hard[selected_cells] == type_index).astype(np.float64),
            minlength=len(selected_spots),
        )
        soft_spot_mass = np.bincount(
            indices[selected_cells],
            weights=probabilities[selected_cells, type_index],
            minlength=len(selected_spots),
        )
        hard_fraction = hard_spot_count[selected_spots] / cell_count_by_spot[selected_spots]
        soft_fraction = soft_spot_mass[selected_spots] / cell_count_by_spot[selected_spots]

        def variation(values: np.ndarray) -> Dict[str, Any]:
            standard_deviation = float(np.std(values))
            return {
                "spatial_standard_deviation": standard_deviation,
                "spatially_constant": bool(np.ptp(values) <= 1.0e-12),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }

        spot_mixture[name] = {
            "hard_assignment_fraction": variation(hard_fraction),
            "soft_expected_fraction": variation(soft_fraction),
        }
    if len(names) == 1:
        normalized_entropy = np.zeros(len(selected_cells), dtype=np.float64)
    else:
        selected_probabilities = probabilities[selected_cells]
        entropy = -np.sum(
            np.where(
                selected_probabilities > 0,
                selected_probabilities * np.log(np.maximum(selected_probabilities, 1.0e-300)),
                0.0,
            ),
            axis=1,
        )
        normalized_entropy = entropy / math.log(len(names))
    return {
        "scope": "assigned nuclei in primary evaluated spots",
        "cells_evaluated": int(len(selected_cells)),
        "spots_evaluated": int(len(selected_spot_indices)),
        "hard_assignment_counts": {
            name: int(hard_counts[index]) for index, name in enumerate(names)
        },
        "hard_assignment_fractions": {
            name: float(hard_counts[index] / len(selected_cells))
            for index, name in enumerate(names)
        },
        "occupied_hard_type_count": int(np.sum(hard_counts > 0)),
        "occupied_hard_types": [name for index, name in enumerate(names) if hard_counts[index] > 0],
        "unoccupied_hard_types": [
            name for index, name in enumerate(names) if hard_counts[index] == 0
        ],
        "normalized_probability_entropy": {
            "mean": float(np.mean(normalized_entropy)),
            "median": float(np.median(normalized_entropy)),
            "p95": float(np.quantile(normalized_entropy, 0.95)),
            "minimum": float(np.min(normalized_entropy)),
            "maximum": float(np.max(normalized_entropy)),
        },
        "per_type_spot_mixture": spot_mixture,
        "hard_spot_mixture_spatially_constant_types": [
            name
            for name in names
            if spot_mixture[name]["hard_assignment_fraction"]["spatially_constant"]
        ],
        "soft_spot_mixture_spatially_constant_types": [
            name
            for name in names
            if spot_mixture[name]["soft_expected_fraction"]["spatially_constant"]
        ],
    }


def _reference_type_support(
    reference: RNAReference,
    prediction: PredictionBundle,
) -> Dict[str, Any]:
    """Audit prediction/reference type overlap before constructing a baseline."""

    reference_types = tuple(sorted(set(np.asarray(reference.cell_type_labels).astype(str))))
    prediction_types = tuple(str(value) for value in prediction.type_names.tolist())
    reference_set = set(reference_types)
    supported = tuple(name for name in prediction_types if name in reference_set)
    missing = tuple(name for name in prediction_types if name not in reference_set)
    probabilities = _normalized_type_probabilities(prediction)
    missing_mask = np.asarray([name in missing for name in prediction_types], dtype=bool)
    hard_names = np.asarray(prediction_types, dtype=object)[probabilities.argmax(axis=1)]
    hard_fallback = np.isin(hard_names, np.asarray(missing, dtype=object))
    soft_fallback_mass = (
        probabilities[:, missing_mask].sum(axis=1)
        if missing_mask.any()
        else np.zeros(len(probabilities), dtype=np.float64)
    )
    return {
        "prediction_cell_types": list(prediction_types),
        "reference_cell_types": list(reference_types),
        "reference_supported_prediction_cell_types": list(supported),
        "missing_prediction_cell_types": list(missing),
        "missing_type_policy": "fail_closed_no_global_profile_fallback",
        "hard_assignment_global_fallback_cells": int(hard_fallback.sum()),
        "hard_assignment_global_fallback_cell_fraction": float(hard_fallback.mean()),
        "soft_assignment_global_fallback_probability_mass_mean": float(soft_fallback_mass.mean()),
    }


def _reference_prototype_type_support(
    reference: RNAReference,
    prediction: PredictionBundle,
    prototypes: PrototypeSet,
) -> Dict[str, Any]:
    """Describe the legacy SVD sensitivity prototype bank."""

    prototype_types = tuple(
        dict.fromkeys(str(value) for value in np.asarray(prototypes.cell_type_labels).tolist())
    )
    return _prototype_type_support(
        reference,
        prediction,
        prototype_types,
        source="legacy_svd_sensitivity_prototype_bank",
        policy="types below the frozen minimum cell count are omitted and reserved for unknown",
    )


def _prototype_type_support(
    reference: RNAReference,
    prediction: PredictionBundle,
    prototype_types: Sequence[str],
    *,
    source: str,
    policy: str,
) -> Dict[str, Any]:
    """Separate count-reference support from one explicitly named prototype bank."""

    count_types = tuple(
        sorted({str(value) for value in np.asarray(reference.cell_type_labels).tolist()})
    )
    ordered_prototype_types = tuple(dict.fromkeys(str(value) for value in prototype_types))
    count_set = set(count_types)
    prototype_set = set(ordered_prototype_types)
    unexpected = tuple(sorted(prototype_set - count_set))
    if unexpected:
        raise ValueError(
            "prototype bank contains types absent from its count reference: %s"
            % ", ".join(unexpected)
        )
    prediction_types = tuple(str(value) for value in prediction.type_names.tolist())
    return {
        "count_reference_supported_types": list(count_types),
        "prototype_supported_types": list(ordered_prototype_types),
        "prototype_omitted_types": list(sorted(count_set - prototype_set)),
        "count_reference_supported_prediction_types": [
            name for name in prediction_types if name in count_set
        ],
        "prototype_supported_prediction_types": [
            name for name in prediction_types if name in prototype_set
        ],
        "prototype_omitted_prediction_types": [
            name for name in prediction_types if name in count_set and name not in prototype_set
        ],
        "prototype_support_source": source,
        "prototype_support_policy": policy,
    }


def _cell_rna_mass(reference: RNAReference, prediction: PredictionBundle) -> np.ndarray:
    """Return the shared soft expected type-median library-size weight."""

    _, medians = _reference_linear_profiles(reference, prediction.type_names.tolist())
    probabilities = _normalized_type_probabilities(prediction)
    mass = probabilities.dot(medians)
    return mass / max(float(np.median(mass)), 1.0e-12)


def _hard_assigned_cell_rna_mass(
    reference: RNAReference,
    prediction: PredictionBundle,
) -> np.ndarray:
    """Return the median library size of each cell's hard argmax type."""

    _, medians = _reference_linear_profiles(reference, prediction.type_names.tolist())
    hard_types = _normalized_type_probabilities(prediction).argmax(axis=1)
    mass = medians[hard_types]
    return mass / max(float(np.median(mass)), 1.0e-12)


def _type_mean_cells(
    reference: RNAReference,
    prediction: PredictionBundle,
) -> np.ndarray:
    profiles, _ = _reference_linear_profiles(reference, prediction.type_names.tolist())
    probabilities = _normalized_type_probabilities(prediction)
    hard_types = probabilities.argmax(axis=1)
    return np.log1p(profiles[hard_types]).astype(np.float32)


def _soft_type_mean_cells(
    reference: RNAReference,
    prediction: PredictionBundle,
) -> np.ndarray:
    """Return the probability-weighted historical integrated type-mean baseline."""

    profiles, _ = _reference_linear_profiles(reference, prediction.type_names.tolist())
    probabilities = _normalized_type_probabilities(prediction)
    return np.log1p(probabilities.dot(profiles)).astype(np.float32)


def _matched_r1_baseline_cell_values(
    reference: RNAReference,
    refined_prediction: PredictionBundle,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build both primary R1 baselines from the refined method's own type map."""

    return (
        _cell_rna_mass(reference, refined_prediction),
        _hard_assigned_cell_rna_mass(reference, refined_prediction),
        _type_mean_cells(reference, refined_prediction),
        _soft_type_mean_cells(reference, refined_prediction),
    )


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
    """Select an exact top fraction with a frozen lower-index-wins tie policy."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("top-fraction values must be a non-empty finite vector")
    if not 0 < fraction <= 1:
        raise ValueError("top-fraction must be in (0, 1]")
    count = max(1, int(math.ceil(len(values) * fraction)))
    # ``lexsort`` uses the last key first: larger values rank first, then the
    # original spot index resolves cutoff ties deterministically.
    order = np.lexsort((np.arange(len(values), dtype=np.int64), -values))
    return order[:count]


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


def _detection_and_hotspot_scores(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> Tuple[float, float, float]:
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
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("predicted and observed expression must be finite")
    if coordinates.shape != (prediction.shape[0], 2):
        raise ValueError("spot coordinates must align to expression")
    if not np.isfinite(coordinates).all():
        raise ValueError("spot coordinates must be finite")
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
            "expression_detection_auroc",
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
        hotspot = _detection_and_hotspot_scores(left, right) if evaluable else (float("nan"),) * 3
        gene_predicted_i = morans_i(left, edges) if evaluable else float("nan")
        gene_observed_i = morans_i(right, edges) if evaluable else float("nan")
        values = {
            "spearman": spearman,
            "pearson": pearson,
            "mse": float(np.mean((left - right) ** 2)),
            "mae": float(np.mean(np.abs(left - right))),
            "concordance": _concordance(left, right) if evaluable else float("nan"),
            "expression_detection_auroc": hotspot[0],
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

    correlation_status = per_gene["correlation_status"]
    summary: Dict[str, Any] = {
        "median_gene_spearman": median("spearman"),
        "median_gene_pearson": median("pearson"),
        "median_gene_mse": median("mse"),
        "median_gene_mae": median("mae"),
        "median_gene_concordance": median("concordance"),
        "median_expression_detection_auroc": median("expression_detection_auroc"),
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
        "prediction_constant_scored_zero_count": int(
            sum(value == "prediction_constant_scored_zero" for value in correlation_status)
        ),
        "observed_constant_excluded_count": int(
            sum(value == "excluded_observed_constant" for value in correlation_status)
        ),
    }
    return {"summary": summary, "per_gene": per_gene}


def _stable_seed(seed: int, sample: str) -> int:
    digest = hashlib.sha256((str(seed) + "\x1f" + sample).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def _record_shuffle_seed(seed: int, sample: str, draw_index: int) -> int:
    """Derive independent deterministic seeds while preserving the historical first draw."""

    if draw_index < 0:
        raise ValueError("record-shuffle draw_index must be non-negative")
    if draw_index == 0:
        return _stable_seed(seed, sample)
    return _stable_seed(
        seed,
        "%s\x1ffinal_cell_record_shuffle\x1f%d" % (sample, draw_index),
    )


def _prepare_spearman_truth(
    observed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(observed, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[0] < 3 or not np.isfinite(truth).all():
        raise ValueError("shuffle-null truth must be a finite spots-by-genes matrix")
    evaluable = np.var(truth, axis=0) > 1.0e-12
    if not evaluable.any():
        raise ValueError("shuffle-null truth has no spatially variable genes")
    ranks = np.asarray(rankdata(truth[:, evaluable], axis=0), dtype=np.float64)
    centered = ranks - ranks.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.square(centered).sum(axis=0))
    return evaluable, centered, norm


def _median_gene_spearman_prepared(
    predicted: np.ndarray,
    evaluable_genes: np.ndarray,
    centered_truth_ranks: np.ndarray,
    truth_rank_norm: np.ndarray,
) -> float:
    prediction = np.asarray(predicted, dtype=np.float64)
    if (
        prediction.ndim != 2
        or prediction.shape[1] != len(evaluable_genes)
        or prediction.shape[0] != centered_truth_ranks.shape[0]
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("shuffle-null prediction is invalid or misaligned")
    selected = prediction[:, evaluable_genes]
    variable = np.var(selected, axis=0) > 1.0e-12
    correlations = np.zeros(selected.shape[1], dtype=np.float64)
    if variable.any():
        ranks = np.asarray(rankdata(selected[:, variable], axis=0), dtype=np.float64)
        centered = ranks - ranks.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.square(centered).sum(axis=0)) * truth_rank_norm[variable]
        correlations[variable] = (centered * centered_truth_ranks[:, variable]).sum(
            axis=0
        ) / denominator
    return float(np.median(correlations))


def _compact_shuffle_distribution(values: np.ndarray) -> Dict[str, Any]:
    statistics = np.asarray(values, dtype=np.float64)
    if statistics.ndim != 1 or not len(statistics) or not np.isfinite(statistics).all():
        raise ValueError("shuffle-null statistics must be a finite vector")
    return {
        "statistic": "median_gene_spearman",
        "permutations": int(len(statistics)),
        "mean": float(np.mean(statistics)),
        "median": float(np.median(statistics)),
        "sample_standard_deviation": (
            float(np.std(statistics, ddof=1)) if len(statistics) > 1 else 0.0
        ),
        "minimum": float(np.min(statistics)),
        "maximum": float(np.max(statistics)),
        "empirical_percentile_interval_95": {
            "lower": float(np.quantile(statistics, 0.025)),
            "upper": float(np.quantile(statistics, 0.975)),
        },
    }


def _repeated_final_record_shuffle_null(
    cell_log_expression: np.ndarray,
    cell_weights: np.ndarray,
    spot_index: np.ndarray,
    primary_spots: np.ndarray,
    observed_expression: np.ndarray,
    *,
    sample: str,
    seed: int,
    permutations: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Score a compact repeated final-record shuffle null.

    Expression and its corresponding library-size weight move together. Raw
    per-permutation matrices and per-gene metrics are deliberately not returned
    in the report summary.
    """

    expression = np.asarray(cell_log_expression, dtype=np.float64)
    weights = np.asarray(cell_weights, dtype=np.float64)
    indices = np.asarray(spot_index, dtype=np.int64)
    selected_spots = np.asarray(primary_spots, dtype=bool)
    truth = np.asarray(observed_expression, dtype=np.float64)
    if permutations < 100:
        raise ValueError("repeated final-record shuffle requires at least 100 permutations")
    if (
        expression.ndim != 2
        or weights.shape != (len(expression),)
        or indices.shape != (len(expression),)
        or truth.ndim != 2
        or expression.shape[1] != truth.shape[1]
        or selected_spots.shape != (truth.shape[0],)
    ):
        raise ValueError("record-shuffle inputs are misaligned")
    assigned = np.flatnonzero(indices >= 0)
    if not len(assigned) or np.any(indices[assigned] >= len(truth)):
        raise ValueError("record-shuffle requires valid assigned cells")
    if (
        not np.isfinite(expression[assigned]).all()
        or not np.isfinite(weights[assigned]).all()
        or np.any(weights[assigned] <= 0)
    ):
        raise ValueError("assigned record-shuffle expression and weights must be finite/positive")
    truth_state = _prepare_spearman_truth(truth[selected_spots])
    statistics = np.empty(permutations, dtype=np.float64)
    first_spots: Optional[np.ndarray] = None
    first_mass: Optional[np.ndarray] = None
    assigned_spots = indices[assigned]
    for draw_index in range(permutations):
        rng = np.random.default_rng(_record_shuffle_seed(seed, sample, draw_index))
        permutation = rng.permutation(assigned)
        shuffled_spots, shuffled_mass = aggregate_cells_to_spots(
            expression[permutation],
            assigned_spots,
            len(truth),
            weights[permutation],
        )
        if np.any(shuffled_mass[selected_spots] <= 0):
            raise RuntimeError("record shuffle removed all positive mass from a primary spot")
        statistics[draw_index] = _median_gene_spearman_prepared(
            shuffled_spots[selected_spots],
            *truth_state,
        )
        if draw_index == 0:
            first_spots = shuffled_spots
            first_mass = shuffled_mass
    assert first_spots is not None and first_mass is not None
    return _compact_shuffle_distribution(statistics), first_spots, first_mass, statistics


def _readiness(
    plan: DeepBenchPlan,
    *,
    native_scanvi: Optional[NativeScanviManifest] = None,
    refined_predictions_validated: bool = False,
    five_seed_predictions: Optional[FiveSeedPredictionManifest] = None,
    refinement_matrix_summary: Optional[RefinementMatrixSummary] = None,
) -> Tuple[Dict[str, Any], ...]:
    matrix_complete = bool(
        refinement_matrix_summary is not None and refinement_matrix_summary.matrix_complete
    )
    matrix_controls = set(refinement_matrix_summary.control_names) if matrix_complete else set()
    matrix_strict_status = (
        None
        if refinement_matrix_summary is None
        else refinement_matrix_summary.strict_ordering_status
    )
    ready = [
        ("locked_round0_predictions", "Hash-frozen v0.2 predictions for all three specimens"),
        (
            "historical_integrated_hard_type_mean",
            "Hard argmax profile with shared soft expected RNA-mass weights derived from the "
            "v0.2 pooled multi-workflow reference",
        ),
        (
            "historical_integrated_soft_type_mean",
            "Probability-weighted baseline derived from the v0.2 pooled multi-workflow reference",
        ),
        (
            "historical_integrated_hard_assigned_mass_type_mean",
            "Hard argmax profile with hard-assigned type-median RNA-mass weights",
        ),
        (
            "historical_integrated_equal_cell_type_means",
            "Hard and soft type-profile baselines with equal-cell aggregation",
        ),
        (
            "historical_integrated_pseudobulk",
            "Derived from the v0.2 pooled integrated multi-workflow reference",
        ),
        (
            "historical_final_cell_record_shuffle",
            "%d deterministic independently seeded final-cell-record permutations are "
            "summarized compactly; draw 0 remains the single-method backward comparison. "
            "%s"
            % (
                plan.final_cell_record_shuffle_permutations,
                (
                    "These historical permutations do not substitute for image-feature or "
                    "graph shuffle controls; those controls are consumed separately by the "
                    "native-scANVI refinement matrix"
                    if matrix_complete
                    else ("Neither satisfies image-feature or coordinate/graph shuffle controls")
                ),
            ),
        ),
        (
            "historical_integrated_reference_library_size_weighting",
            "Type-median library sizes from the historical pooled multi-workflow reference",
        ),
    ]
    records: List[Dict[str, Any]] = [
        {"component": name, "status": "ready", "reason": reason} for name, reason in ready
    ]
    primary_reference_reason = (
        "FFPE-snPATHO-only native scANVI references and rare-complete prototype banks are "
        "hash-validated, and the scored refinement matrix consumes the native prototype-only "
        "control. The labels still come from the published integrated-workflow annotation; "
        "an independent clean reannotation is absent, so this remains a sensitivity analysis."
        if native_scanvi is not None and matrix_complete and "prototype_only" in matrix_controls
        else (
            "FFPE-snPATHO-only native scANVI references and rare-complete prototype banks are "
            "hash-validated, but the labels come from the published integrated-workflow "
            "annotation and no scored prototype-only matrix is available; an independent "
            "clean reannotation is absent."
            if native_scanvi is not None
            else (
                "FFPE-snPATHO-only count artifacts are hash-manifested, but use the published "
                "integrated-workflow annotation and have no independent clean reannotation, "
                "primary scANVI encoding, or prototype-only prediction/scorer; materialized "
                "SVD fallback prototypes remain development-only"
            )
        )
    )
    partial = {
        "primary_ffpe_snpatho_reference": (
            "partial_materialized_not_benchmark_ready",
            primary_reference_reason,
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
            (
                "Space Ranger exports no calibrated segmentation confidence and the refinement "
                "runs substitute 1.0, so the scored development matrix does not satisfy the "
                "primary benchmark's calibrated anchor-confidence requirement"
                if matrix_complete
                else (
                    "Space Ranger exports no calibrated segmentation confidence and v0.2 "
                    "substituted 1.0, so the >=0.50 anchor gate is vacuous and refinement is "
                    "not benchmark-ready"
                )
            ),
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
            "No round 1-4 predictions, anchor telemetry, E-step comparison, or prior-update "
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
    if matrix_complete:
        for component, reason in {
            "graph_sensitivity_and_rewiring": (
                "The provenance-validated matrix consumes scored graph-shuffle and no-graph "
                "controls. The requested 8-NN, radius, multiscale, and degree-preserving "
                "rewiring sensitivities remain absent."
            ),
            "refinement_trajectory_and_ablations": (
                "The provenance-validated matrix consumes round 0/final predictions across "
                "five seeds, the complete round 1-4 score trajectory at the prespecified "
                "trajectory seed, and prototype-only/image-shuffle/graph-shuffle/no-graph/"
                "wrong-donor controls. E-step, prior-update, refinement-gate, and anchor/map "
                "stability analyses remain unscored."
            ),
            "complete_negative_control_matrix": (
                "Prototype-only, image-feature-shuffle, graph-shuffle, no-graph, and "
                "wrong-donor controls are consumed by the provenance-validated matrix. Label "
                "and prototype permutations, generic-atlas RNA, state omission, reference "
                "downsampling, block shuffles, toroidal shifts, and coordinate perturbations "
                "remain absent."
            ),
        }.items():
            blocked_plan_components.pop(component)
            records.append(
                {
                    "component": component,
                    "status": "partial_consumed_via_refinement_matrix",
                    "reason": reason,
                }
            )
    if five_seed_predictions is not None or matrix_complete:
        blocked_plan_components.pop("seed_ensemble_stability")
        records.append(
            {
                "component": "seed_ensemble_stability",
                "status": "partial_consumed_performance_matrix_only",
                "reason": (
                    "Five-seed prediction-level performance is consumed, but map, anchor, "
                    "assignment-overlap, and between-model stability have not been scored; "
                    "the matrix must not be interpreted as ensemble-stability evidence."
                ),
            }
        )
    records.extend(
        {
            "component": name,
            "status": "blocked_not_implemented_or_missing_artifact",
            "reason": reason,
        }
        for name, reason in blocked_plan_components.items()
    )
    reasons = {
        "primary_ffpe_snpatho_reference_manifest": (
            "No hash-bound FFPE-snPATHO-only R1 reference manifest is registered"
        ),
        "refined_predictions": "No post-redesign refined predictions are supplied",
        "five_seed_predictions": "Only the historical seed-17 prediction is frozen",
        "refinement_matrix_summary": (
            "No hash-bound compact refinement score matrix is registered"
        ),
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
    matrix_control_artifacts = {
        "wrong_donor_predictions": (
            "wrong_donor",
            "wrong-donor prototype control",
        ),
        "image_shuffle_predictions": (
            "image_shuffle",
            "shuffled-image-feature control",
        ),
        "graph_shuffle_predictions": (
            "graph_shuffle",
            "shuffled-graph control",
        ),
    }
    for name in OPTIONAL_ARTIFACTS:
        artifact = plan.optional_artifacts[name]
        if name == "primary_ffpe_snpatho_reference_manifest" and artifact is not None:
            status = "partial_consumed_retrospective_sensitivity"
            reason = (
                "Hash-validated FFPE-only counts are consumed for the matched type-mean "
                "estimand ladder. Native scANVI references and rare-complete prototype banks "
                "are separately hash-validated, but the published integrated annotations are "
                "not an independent clean R1 reannotation."
                if native_scanvi is not None
                else (
                    "Hash-validated FFPE-only counts and SVD fallback prototype banks are "
                    "consumed for the type-mean ladder, but no primary native-scANVI "
                    "reference or independent clean R1 reannotation is available."
                )
            )
        elif name == "refined_predictions" and matrix_complete:
            status = "consumed_via_provenance_validated_refinement_matrix"
            reason = (
                "Round-0 and final refined predictions for every prespecified specimen and "
                "five-seed case are scored in the hash-bound matrix; strict ordering is %s"
                % matrix_strict_status
            )
        elif name == "refined_predictions" and refined_predictions_validated:
            status = "consumed_provenance_validated_development_refined_predictions"
            reason = (
                "PredictionBundle sample/donor/seed/round, checkpoint, refinement audit, "
                "telemetry, refined/native prototypes, latent space, and expression space are "
                "hash-bound; the result remains a one-seed integrated-label sensitivity"
            )
        elif name == "five_seed_predictions" and five_seed_predictions is not None:
            status = "ready_provenance_validated_five_seed_matrix"
            reason = (
                "Every prespecified specimen/seed PredictionBundle is hash-validated and bound "
                "to the native scANVI latent/expression identities. This establishes scored "
                "performance coverage, not map or anchor ensemble stability"
            )
        elif name == "five_seed_predictions" and matrix_complete:
            status = "consumed_via_provenance_validated_refinement_matrix"
            reason = (
                "The hash-bound score matrix covers every prespecified specimen and seed; "
                "this establishes performance coverage, not map or anchor ensemble stability"
            )
        elif name == "refinement_matrix_summary" and refinement_matrix_summary is not None:
            status = (
                "ready_provenance_validated_matrix_strict_ordering_passed"
                if refinement_matrix_summary.strict_ordering_pass
                else "consumed_provenance_validated_matrix_strict_ordering_failed"
            )
            reason = (
                "The compact summary is plan-hash-bound, covers every requested specimen, "
                "seed, round 0-4 trajectory artifact, prototype-only/image-shuffle/graph-"
                "shuffle/no-graph/wrong-donor control, and strict comparison, and reports "
                "strict ordering %s" % refinement_matrix_summary.strict_ordering_status
            )
        elif name == "native_scanvi_checkpoint" and native_scanvi is not None:
            status = "ready_recursively_hash_validated_native_scanvi"
            reason = (
                "The external native model directory, decoder gene order and expression-"
                "normalization contract, per-specimen latent references, rare-complete "
                "prototype banks, and RNA residual geometries (source reference, latent "
                "identity, type order, rank, and bounds) were parsed and recursively hash-"
                "validated; published integrated annotations remain a declared sensitivity"
            )
        elif (
            matrix_complete
            and name in matrix_control_artifacts
            and matrix_control_artifacts[name][0] in matrix_controls
        ):
            status = "consumed_via_provenance_validated_refinement_matrix"
            reason = (
                "The %s is scored for every prespecified control case in the hash-bound "
                "refinement matrix; strict ordering is %s"
                % (matrix_control_artifacts[name][1], matrix_strict_status)
            )
        elif name == "no_geometry_predictions" and matrix_complete:
            status = "partial_no_graph_consumed_via_refinement_matrix"
            reason = (
                "The hash-bound matrix consumes the prespecified no-graph control, but a "
                "dedicated no-geometry prediction that removes every spatial input remains "
                "absent; strict ordering is %s" % matrix_strict_status
            )
        elif (
            name
            in {
                "refined_predictions",
                "five_seed_predictions",
                "refinement_matrix_summary",
                "native_scanvi_checkpoint",
            }
            and artifact is not None
        ):
            status = "registered_but_not_provenance_validated"
            reason = "The artifact was registered but no parsed provenance object was supplied"
        else:
            status = (
                "registered_not_implemented" if artifact is not None else "blocked_missing_artifact"
            )
            reason = (
                "Hash-validated artifact is registered, but no scorer consumes this schema yet"
                if artifact is not None
                else reasons[name]
            )
        records.append(
            {
                "component": name,
                "status": status,
                "reason": reason,
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
        "bootstrap_fraction_delta_positive": (float(np.mean(finite > 0)) if len(finite) else None),
        "iterations": int(iterations),
        "method": "paired specimen/gene abundance-stratified bootstrap",
        "limitation": "connected spatial blocks were not frozen in the historical artifacts",
    }


def _requested_refined_primary_contrasts(
    cases: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate the joint hard-and-soft matched-R1 success requirement when possible."""

    contrast_methods = {
        "matched_ffpe_r1_hard": R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD,
        "matched_ffpe_r1_soft": R1_SOFT_TYPE_MEAN_METHOD,
    }
    contract: Dict[str, Any] = {
        "joint_contrast": REQUESTED_PRIMARY_CONTRAST,
        "refined_method": REFINED_R1_METHOD,
        "evidence_scope": "developmental_one_seed_integrated_label_sensitivity",
        "full_primary_claim": False,
        "required_baseline_methods": list(PRIMARY_MATCHED_R1_BASELINES),
        "required_contrasts": [
            "refined_heir_minus_matched_ffpe_r1_hard_type_mean",
            "refined_heir_minus_matched_ffpe_r1_soft_type_mean",
        ],
        "endpoint": "paired_median_gene_spearman_delta",
        "success_formula": (
            "macro_delta_vs_matched_ffpe_r1_hard > 0 and macro_delta_vs_matched_ffpe_r1_soft > 0"
        ),
        "requires_both_contrasts": True,
    }
    missing: List[Dict[str, Any]] = []
    for case in cases:
        methods = case.get("methods")
        if not isinstance(methods, Mapping):
            missing.append(
                {"section_id": str(case.get("section_id", "")), "missing_methods": ["methods"]}
            )
            continue
        absent = [
            method
            for method in (REFINED_R1_METHOD, *PRIMARY_MATCHED_R1_BASELINES)
            if method not in methods
        ]
        if absent:
            missing.append(
                {"section_id": str(case.get("section_id", "")), "missing_methods": absent}
            )
    if not cases or missing:
        return {
            **contract,
            "status": "not_testable_missing_report_methods",
            "missing": missing,
            "contrasts": None,
            "refined_beats_both_matched_ffpe_r1_baselines": None,
        }

    contrasts: Dict[str, Any] = {}
    for label, baseline_method in contrast_methods.items():
        rows: List[Dict[str, Any]] = []
        for case in cases:
            methods = cast(Mapping[str, Any], case["methods"])
            refined = cast(Mapping[str, Any], methods[REFINED_R1_METHOD])
            baseline = cast(Mapping[str, Any], methods[baseline_method])
            refined_per_gene = cast(Mapping[str, Any], refined["per_gene"])
            baseline_per_gene = cast(Mapping[str, Any], baseline["per_gene"])
            left = np.asarray(
                [np.nan if value is None else value for value in refined_per_gene["spearman"]],
                dtype=np.float64,
            )
            right = np.asarray(
                [np.nan if value is None else value for value in baseline_per_gene["spearman"]],
                dtype=np.float64,
            )
            if left.shape != right.shape:
                raise ValueError("refined and matched-R1 per-gene metrics are misaligned")
            difference = left - right
            finite = difference[np.isfinite(difference)]
            if not len(finite):
                raise ValueError("refined matched-R1 contrast has no evaluable genes")
            rows.append(
                {
                    "section_id": str(case["section_id"]),
                    "median_paired_per_gene_spearman_delta": float(np.median(finite)),
                }
            )
        macro_delta = float(np.mean([row["median_paired_per_gene_spearman_delta"] for row in rows]))
        contrasts[label] = {
            "baseline_method": baseline_method,
            "baseline_estimand": _baseline_estimands()[baseline_method],
            "specimens": rows,
            "macro_delta": macro_delta,
            "macro_delta_positive": macro_delta > 0,
        }
    joint = all(value["macro_delta_positive"] for value in contrasts.values())
    return {
        **contract,
        "status": "passes_developmental_joint_contrast"
        if joint
        else "fails_developmental_joint_contrast",
        "missing": [],
        "contrasts": contrasts,
        "refined_beats_both_matched_ffpe_r1_baselines": joint,
    }


def _validate_primary_gate_support_artifact(
    plan: DeepBenchPlan,
    artifact_name: str,
    evidence_kind: str,
    required_contract: Mapping[str, object],
) -> bool:
    """Validate one recursively hash-bound full-primary support manifest.

    The optional-artifact registration hash alone only proves the identity of
    a manifest.  Full-primary gating additionally requires the manifest to
    declare the prespecified evidence semantics, cover every specimen, and
    recursively hash-bind each specimen-level result.
    """

    path = plan.optional_artifacts.get(artifact_name)
    expected_sha256 = plan.optional_artifact_sha256.get(artifact_name)
    if path is None or expected_sha256 is None:
        return False
    try:
        _validate_file_hash(path, expected_sha256, artifact_name)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            return False
        if (
            payload.get("schema") != PRIMARY_GATE_SUPPORT_SCHEMA
            or payload.get("evidence_kind") != evidence_kind
            or payload.get("frozen_benchmark_plan_sha256") != plan.frozen_plan_sha256
        ):
            return False
        sample_ids = payload.get("sample_ids")
        if not isinstance(sample_ids, list) or tuple(sample_ids) != plan.sample_ids:
            return False
        requirements = payload.get("requirements")
        if not isinstance(requirements, Mapping) or any(
            requirements.get(name) != value for name, value in required_contract.items()
        ):
            return False
        per_sample = payload.get("per_sample_artifacts")
        if not isinstance(per_sample, Mapping) or set(per_sample) != set(plan.sample_ids):
            return False
        for sample_id in plan.sample_ids:
            record = per_sample[sample_id]
            if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
                return False
            result_path = _resolve_artifact(
                path.parent,
                record["path"],
                "%s_%s" % (artifact_name, sample_id),
            )
            _validate_file_hash(
                result_path,
                record["sha256"],
                "%s_%s" % (artifact_name, sample_id),
            )
    except (OSError, TypeError, ValueError):
        return False
    return True


def _primary_gate_support_status(plan: DeepBenchPlan) -> Dict[str, bool]:
    """Return fail-closed support status for every non-matrix primary gate."""

    return {
        gate: _validate_primary_gate_support_artifact(
            plan,
            artifact_name,
            evidence_kind,
            requirements,
        )
        for gate, (
            artifact_name,
            evidence_kind,
            requirements,
        ) in PRIMARY_GATE_SUPPORT_CONTRACTS.items()
    }


def _full_primary_evidence_gates(
    plan: DeepBenchPlan,
    native_scanvi: Optional[NativeScanviManifest],
    five_seed_predictions: Optional[FiveSeedPredictionManifest],
    refinement_matrix_summary: Optional[RefinementMatrixSummary],
) -> Dict[str, Any]:
    """Fail closed unless the prespecified matrix was scored and passed."""

    clean_annotation = bool(native_scanvi is not None and native_scanvi.clean_annotation_complete)
    five_seed_complete = five_seed_predictions is not None
    matrix_complete = bool(
        refinement_matrix_summary is not None and refinement_matrix_summary.matrix_complete
    )
    strict_ordering_pass = bool(
        refinement_matrix_summary is not None and refinement_matrix_summary.strict_ordering_pass
    )
    scored_controls = (
        set() if refinement_matrix_summary is None else set(refinement_matrix_summary.control_names)
    )
    control_availability = {
        name: bool(
            name in scored_controls
            and (
                name != "wrong_donor"
                or (
                    refinement_matrix_summary is not None
                    and refinement_matrix_summary.wrong_donor_coverage_complete
                )
            )
        )
        for name in REFINEMENT_MATRIX_CONTROLS
    }
    controls_complete = matrix_complete and all(control_availability.values())
    execution_provenance_verified = bool(
        five_seed_predictions is not None
        and five_seed_predictions.execution_provenance_verified
        and refinement_matrix_summary is not None
        and refinement_matrix_summary.execution_provenance_verified
    )
    required_followup_evidence_complete = bool(
        refinement_matrix_summary is not None
        and refinement_matrix_summary.required_followup_evidence_complete
    )
    primary_support = _primary_gate_support_status(plan)
    gates = {
        "independent_clean_reannotation": clean_annotation,
        "prespecified_five_seed_matrix": five_seed_complete,
        "scored_refinement_matrix_complete": matrix_complete,
        "refinement_matrix_strict_ordering_pass": strict_ordering_pass,
        "required_negative_controls": controls_complete,
        "execution_provenance_verified": execution_provenance_verified,
        "required_followup_evidence_complete": required_followup_evidence_complete,
        **primary_support,
    }
    return {
        "eligible_for_full_primary_claim": all(gates.values()),
        "gates": gates,
        "control_availability": control_availability,
        "refinement_matrix": {
            "registered": refinement_matrix_summary is not None,
            "matrix_status": (
                None
                if refinement_matrix_summary is None
                else refinement_matrix_summary.matrix_status
            ),
            "strict_ordering_status": (
                None
                if refinement_matrix_summary is None
                else refinement_matrix_summary.strict_ordering_status
            ),
            "summary_sha256": (
                None if refinement_matrix_summary is None else refinement_matrix_summary.sha256
            ),
            "execution_provenance_verified": (
                False
                if refinement_matrix_summary is None
                else refinement_matrix_summary.execution_provenance_verified
            ),
            "primary_evidence_status": (
                None
                if refinement_matrix_summary is None
                else refinement_matrix_summary.primary_evidence_status
            ),
            "evidence_blocker_count": (
                None
                if refinement_matrix_summary is None
                else refinement_matrix_summary.evidence_blocker_count
            ),
            "execution_provenance_blocker_count": (
                None
                if refinement_matrix_summary is None
                else refinement_matrix_summary.execution_provenance_blocker_count
            ),
            "wrong_donor_coverage_complete": (
                False
                if refinement_matrix_summary is None
                else refinement_matrix_summary.wrong_donor_coverage_complete
            ),
            "wrong_donor_pairing_count": (
                0
                if refinement_matrix_summary is None
                else refinement_matrix_summary.wrong_donor_pairing_count
            ),
            "expected_wrong_donor_pairing_count": (
                len(DEEPBENCH_SAMPLES) * (len(DEEPBENCH_SAMPLES) - 1)
                if refinement_matrix_summary is None
                else refinement_matrix_summary.expected_wrong_donor_pairing_count
            ),
            "missing_wrong_donor_case_count": (
                None
                if refinement_matrix_summary is None
                else refinement_matrix_summary.missing_wrong_donor_case_count
            ),
        },
        "blockers": [name for name, available in gates.items() if not available],
    }


def _primary_diagnostic(
    cases: Sequence[Dict[str, Any]],
    plan: DeepBenchPlan,
    repeated_shuffle_statistics: Optional[Mapping[str, np.ndarray]] = None,
    native_scanvi: Optional[NativeScanviManifest] = None,
    five_seed_predictions: Optional[FiveSeedPredictionManifest] = None,
    refinement_matrix_summary: Optional[RefinementMatrixSummary] = None,
) -> Dict[str, Any]:
    requested_joint = _requested_refined_primary_contrasts(cases)
    full_primary_evidence = _full_primary_evidence_gates(
        plan,
        native_scanvi,
        five_seed_predictions,
        refinement_matrix_summary,
    )
    deltas: List[np.ndarray] = []
    observed_means: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
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
        shuffle_draw_zero_values = np.asarray(
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
        section_id = str(case["section_id"])
        repeated_comparison: Dict[str, Any] = {
            "status": "unavailable",
            "reason": "repeated final-record shuffle statistics were not supplied",
        }
        if repeated_shuffle_statistics is not None and section_id in repeated_shuffle_statistics:
            null_values = np.asarray(
                repeated_shuffle_statistics[section_id],
                dtype=np.float64,
            )
            if null_values.ndim != 1 or not len(null_values) or not np.isfinite(null_values).all():
                raise ValueError("repeated final-record shuffle statistics are invalid")
            observed_spearman = float(primary_summary["median_gene_spearman"])
            null_lower = float(np.quantile(null_values, 0.025))
            null_upper = float(np.quantile(null_values, 0.975))
            repeated_comparison = {
                "status": "available_retrospective_record_shuffle",
                "observed_heir_median_gene_spearman": observed_spearman,
                "null_permutations": int(len(null_values)),
                "null_median": float(np.median(null_values)),
                "null_empirical_percentile_interval_95": {
                    "lower": null_lower,
                    "upper": null_upper,
                },
                "observed_heir_empirical_percentile_in_null": float(
                    np.mean(null_values <= observed_spearman)
                ),
                "empirical_percentile_definition": (
                    "fraction of repeated-null statistics <= observed HEIR statistic"
                ),
                "observed_heir_above_null_95_upper": observed_spearman > null_upper,
                "observed_heir_minus_null_median": (
                    observed_spearman - float(np.median(null_values))
                ),
            }
        rows.append(
            {
                "section_id": section_id,
                "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean": (
                    float(np.nanmedian(difference))
                ),
                "median_paired_per_gene_spearman_delta_vs_final_record_shuffle_draw_0": (
                    float(np.nanmedian(left - shuffle_draw_zero_values))
                ),
                "repeated_final_record_shuffle_null_comparison": repeated_comparison,
                "median_mse_improvement_vs_type_mean": float(
                    baseline_summary["median_gene_mse"] - primary_summary["median_gene_mse"]
                ),
            }
        )
    paired_delta_key = (
        "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
    )
    macro_delta = float(np.mean([row[paired_delta_key] for row in rows]))
    rules = {
        "macro_delta_positive": macro_delta > 0,
        "positive_in_at_least_two_specimens": sum(row[paired_delta_key] > 0 for row in rows) >= 2,
        "no_specimen_below_minus_0_01": all(row[paired_delta_key] >= -0.01 for row in rows),
        "mse_improves_in_at_least_two_specimens": sum(
            row["median_mse_improvement_vs_type_mean"] > 0 for row in rows
        )
        >= 2,
        "beats_final_record_shuffle_draw_0_in_at_least_two_specimens": sum(
            row["median_paired_per_gene_spearman_delta_vs_final_record_shuffle_draw_0"] > 0
            for row in rows
        )
        >= 2,
        "above_repeated_final_record_shuffle_null_95_upper_in_at_least_two_specimens": (
            sum(
                row["repeated_final_record_shuffle_null_comparison"].get(
                    "observed_heir_above_null_95_upper",
                    False,
                )
                for row in rows
            )
            >= 2
            if repeated_shuffle_statistics is not None
            else None
        ),
        "composition_adjusted_residual_positive": full_primary_evidence["gates"][
            "composition_adjusted_residuals_hash_validated"
        ],
    }
    refined_available = plan.optional_artifacts["refined_predictions"] is not None
    molecular_reference_blocker = None
    if native_scanvi is None:
        molecular_reference_blocker = (
            "materialized FFPE-snPATHO-only counts lack independent clean reannotation and "
            "native scANVI encoding; the SVD fallback adapter is development-only"
        )
    elif not native_scanvi.clean_annotation_complete:
        molecular_reference_blocker = (
            "native scANVI uses published integrated annotations rather than an independent clean "
            "FFPE-snPATHO-only reannotation"
        )
    requested_blockers = []
    if molecular_reference_blocker is not None:
        requested_blockers.append(molecular_reference_blocker)
    if not refined_available:
        requested_blockers.append("refined predictions are absent")
    primary_support_blockers = {
        "composition_adjusted_residuals_hash_validated": (
            "composition-adjusted residual evidence is absent or fails its recursive hash contract"
        ),
        "required_he_tissue_fraction_qc_hash_validated": (
            "required per-spot H&E tissue-fraction QC is absent or fails its recursive hash "
            "contract"
        ),
        "calibrated_segmentation_confidence_hash_validated": (
            "calibrated segmentation-confidence evidence is absent or fails its recursive hash "
            "contract"
        ),
    }
    requested_blockers.extend(
        message
        for gate, message in primary_support_blockers.items()
        if not full_primary_evidence["gates"][gate]
    )
    requested_blockers.extend(
        "full-primary evidence gate is unavailable: %s" % blocker
        for blocker in full_primary_evidence["blockers"]
        if blocker not in primary_support_blockers
    )
    joint_status = str(requested_joint["status"])
    if not refined_available:
        requested_primary_status = "not_testable_missing_refined_predictions"
    elif joint_status == "not_testable_missing_report_methods":
        requested_primary_status = joint_status
    elif not full_primary_evidence["eligible_for_full_primary_claim"]:
        requested_primary_status = "developmental_joint_contrast_only_not_primary"
    elif joint_status == "passes_developmental_joint_contrast":
        requested_primary_status = "passes_full_primary_requirement"
    else:
        requested_primary_status = "fails_full_primary_requirement"
    return {
        "requested_primary_contrast": REQUESTED_PRIMARY_CONTRAST,
        "requested_primary_contrast_requirement": requested_joint,
        "developmental_seed17_joint_contrast": {
            **requested_joint,
            "seed": plan.primary_seeds[0],
            "analysis_role": "developmental_only_not_full_primary_gate",
        },
        "requested_primary_status": requested_primary_status,
        "full_primary_evidence": full_primary_evidence,
        "requested_primary_blockers": requested_blockers,
        "diagnostic_contrast": ("historical_round0_minus_historical_integrated_hard_type_mean"),
        "diagnostic_statistic": {
            "label": "median paired per-gene Spearman delta",
            "specimen_formula": (
                "median_g(rho_HEIR,g - rho_historical_integrated_hard_type_mean,g)"
            ),
            "macro_formula": "mean_d(specimen_median_paired_per_gene_delta_d)",
            "not_equal_to": (
                "median_g(rho_HEIR,g) - median_g(rho_historical_integrated_hard_type_mean,g)"
            ),
        },
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
        estimands = [payload.get("estimand") for payload in specimen_payloads]
        if any(value is not None for value in estimands):
            if any(value != estimands[0] for value in estimands[1:]):
                raise ValueError("DeepBench method estimands must agree across specimens")
            result[str(method)]["estimand"] = estimands[0]
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
    r1_references = _load_r1_references(
        plan,
        gene_panel_sha256=locked_plan.gene_panel_sha256,
    )
    baseline_estimands = _baseline_estimands()
    native_scanvi = _load_native_scanvi_manifest(
        plan.optional_artifacts["native_scanvi_checkpoint"],
        plan.sample_ids,
        manifest_sha256=plan.optional_artifact_sha256["native_scanvi_checkpoint"],
    )
    refined_predictions = _load_refined_prediction_manifest(
        plan.optional_artifacts["refined_predictions"],
        plan.sample_ids,
        native_scanvi,
    )
    five_seed_predictions = _load_five_seed_prediction_manifest(
        plan.optional_artifacts["five_seed_predictions"],
        plan.sample_ids,
        plan.primary_seeds,
        native_scanvi,
    )
    refinement_matrix_summary = _load_refinement_matrix_summary(
        plan.optional_artifacts["refinement_matrix_summary"],
        plan.sample_ids,
        plan.primary_seeds,
        tuple(int(value) for value in _nested(plan.specification, "randomness", "ablation_seeds")),
        minimum_nuclei=plan.minimum_nuclei,
        frozen_plan_sha256=plan.frozen_plan_sha256,
        native_scanvi=native_scanvi,
        summary_sha256=plan.optional_artifact_sha256["refinement_matrix_summary"],
    )
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
    shuffle_null_specimens: Dict[str, Dict[str, Any]] = {}
    shuffle_null_draw_statistics: Dict[str, np.ndarray] = {}
    for case in sorted(locked_plan.cases, key=lambda item: item.section_id):
        prediction = PredictionBundle.from_npz(case.predictions)
        reference = RNAReference.load_npz(case.matched_reference)
        truth = SpatialTruthArtifact.from_npz(case.truth)
        type_support = _reference_type_support(reference, prediction)
        rna_mass = _cell_rna_mass(reference, prediction)
        hard_assigned_rna_mass = _hard_assigned_cell_rna_mass(reference, prediction)
        type_cells = _type_mean_cells(reference, prediction)
        soft_type_cells = _soft_type_mean_cells(reference, prediction)
        spot_index = truth.nucleus_spot_index
        spot_counts = np.bincount(
            spot_index[spot_index >= 0], minlength=len(truth.spot_ids)
        ).astype(np.int64)
        primary_spots = spot_counts >= plan.minimum_nuclei
        if primary_spots.sum() < 3:
            raise ValueError("DeepBench primary spot proxy contains fewer than three spots")
        type_map_diagnostics = _type_map_diagnostics(
            prediction,
            spot_index,
            primary_spots,
        )
        gene_order = tuple(str(value) for value in prediction.gene_names.tolist())
        if gene_order != tuple(str(value) for value in truth.gene_names.tolist()):
            raise ValueError("DeepBench prediction and truth gene orders differ")
        r1_payload = r1_references.get(case.section_id)
        r1_type_support: Optional[Dict[str, Any]] = None
        r1_provenance: Optional[Dict[str, Any]] = None
        r1_values: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
        if r1_payload is not None:
            r1_reference, r1_prototypes, r1_provenance = r1_payload
            if gene_order != tuple(str(value) for value in r1_reference.gene_ids.tolist()):
                raise ValueError("R1 reference and DeepBench gene orders differ")
            refined_artifact = refined_predictions.get(case.section_id)
            r1_prediction = prediction
            if refined_artifact is not None:
                refined_prediction = refined_artifact.prediction
                if not np.array_equal(refined_prediction.nucleus_ids, prediction.nucleus_ids):
                    raise ValueError("refined and historical prediction nuclei differ")
                if tuple(str(value) for value in refined_prediction.gene_names.tolist()) != (
                    gene_order
                ):
                    raise ValueError("refined and historical prediction genes differ")
                if tuple(str(value) for value in refined_prediction.type_names.tolist()) != tuple(
                    str(value) for value in prediction.type_names.tolist()
                ):
                    raise ValueError("refined and historical prediction type ontologies differ")
                r1_prediction = refined_prediction
            legacy_prototype_support = _reference_prototype_type_support(
                r1_reference,
                r1_prediction,
                r1_prototypes,
            )
            native_prototype_support: Optional[Dict[str, Any]] = None
            if native_scanvi is not None:
                native_types = native_scanvi.specimen_prototype_type_names.get(case.section_id)
                if native_types is None:
                    raise ValueError(
                        "native scANVI prototype type support is absent for %s" % case.section_id
                    )
                native_prototype_support = _prototype_type_support(
                    r1_reference,
                    r1_prediction,
                    native_types,
                    source="native_scanvi_rare_complete_prototype_bank",
                    policy=(
                        "rare-complete native scANVI bank; every matched FFPE-R1 type is "
                        "required for refined-run fairness"
                    ),
                )
                if (
                    refined_artifact is not None
                    and native_prototype_support["prototype_omitted_types"]
                ):
                    raise ValueError(
                        "native rare-complete prototype support is incomplete for refined-run "
                        "fairness: %s" % case.section_id
                    )
            fairness_support = (
                native_prototype_support
                if refined_artifact is not None and native_prototype_support is not None
                else legacy_prototype_support
            )
            r1_type_support = {
                **_reference_type_support(r1_reference, r1_prediction),
                **fairness_support,
                "refined_run_fairness_prototype_source": fairness_support[
                    "prototype_support_source"
                ],
                "native_scanvi_rare_complete_prototype_support": native_prototype_support,
                "legacy_svd_sensitivity_prototype_support": legacy_prototype_support,
            }
            # The matched-R1 comparison must use exactly the same refined cell-state
            # probabilities as the method under test.  Falling back to the historical
            # round-0 map is allowed only when no refined run is registered, in which
            # case these methods remain an annotation-sensitivity diagnostic.
            (
                r1_cell_mass,
                r1_hard_assigned_mass,
                r1_hard_cells,
                r1_soft_cells,
            ) = _matched_r1_baseline_cell_values(
                r1_reference,
                r1_prediction,
            )
            r1_hard_assigned_spots, r1_hard_assigned_spot_mass = aggregate_cells_to_spots(
                r1_hard_cells,
                spot_index,
                len(truth.spot_ids),
                r1_hard_assigned_mass,
            )
            r1_hard_spots, r1_hard_mass = aggregate_cells_to_spots(
                r1_hard_cells,
                spot_index,
                len(truth.spot_ids),
                r1_cell_mass,
            )
            r1_soft_spots, r1_soft_mass = aggregate_cells_to_spots(
                r1_soft_cells,
                spot_index,
                len(truth.spot_ids),
                r1_cell_mass,
            )
            r1_equal_hard_spots, r1_equal_hard_mass = aggregate_cells_to_spots(
                r1_hard_cells,
                spot_index,
                len(truth.spot_ids),
            )
            r1_equal_soft_spots, r1_equal_soft_mass = aggregate_cells_to_spots(
                r1_soft_cells,
                spot_index,
                len(truth.spot_ids),
            )
            r1_values = {
                R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD: (
                    r1_hard_assigned_spots,
                    r1_hard_assigned_spot_mass,
                    R1_LIBRARY_SIZE_AGGREGATION + "_hard_assigned_type_mass",
                ),
                R1_HARD_TYPE_MEAN_METHOD: (
                    r1_hard_spots,
                    r1_hard_mass,
                    R1_LIBRARY_SIZE_AGGREGATION,
                ),
                R1_SOFT_TYPE_MEAN_METHOD: (
                    r1_soft_spots,
                    r1_soft_mass,
                    R1_LIBRARY_SIZE_AGGREGATION,
                ),
                R1_EQUAL_CELL_HARD_TYPE_MEAN_METHOD: (
                    r1_equal_hard_spots,
                    r1_equal_hard_mass,
                    "equal_cell",
                ),
                R1_EQUAL_CELL_SOFT_TYPE_MEAN_METHOD: (
                    r1_equal_soft_spots,
                    r1_equal_soft_mass,
                    "equal_cell",
                ),
            }
            if refined_artifact is not None:
                refined_prediction = refined_artifact.prediction
                refined_rna_mass = _cell_rna_mass(r1_reference, refined_prediction)
                refined_spots, refined_mass = aggregate_cells_to_spots(
                    refined_prediction.internal_aggregate_expression_mean,
                    spot_index,
                    len(truth.spot_ids),
                    refined_rna_mass,
                )
                r1_values[REFINED_R1_METHOD] = (
                    refined_spots,
                    refined_mass,
                    R1_LIBRARY_SIZE_AGGREGATION + "_refined_heir",
                )
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
        hard_assigned_type_spots, hard_assigned_type_mass = aggregate_cells_to_spots(
            type_cells,
            spot_index,
            len(truth.spot_ids),
            hard_assigned_rna_mass,
        )
        soft_type_spots, soft_type_mass = aggregate_cells_to_spots(
            soft_type_cells,
            spot_index,
            len(truth.spot_ids),
            rna_mass,
        )
        equal_hard_type_spots, equal_hard_type_mass = aggregate_cells_to_spots(
            type_cells,
            spot_index,
            len(truth.spot_ids),
        )
        equal_soft_type_spots, equal_soft_type_mass = aggregate_cells_to_spots(
            soft_type_cells,
            spot_index,
            len(truth.spot_ids),
        )
        pseudobulk = np.log1p(_reference_linear_pseudobulk(reference))
        pseudobulk_spots = np.repeat(pseudobulk[None, :], len(truth.spot_ids), axis=0)
        (
            shuffle_null_summary,
            shuffled_spots,
            shuffled_mass,
            shuffle_draw_statistics,
        ) = _repeated_final_record_shuffle_null(
            prediction.expression_mean,
            rna_mass,
            spot_index,
            primary_spots,
            truth.observed_expression,
            sample=case.section_id,
            seed=plan.primary_seeds[0],
            permutations=plan.final_cell_record_shuffle_permutations,
        )
        shuffle_null_specimens[case.section_id] = shuffle_null_summary
        shuffle_null_draw_statistics[case.section_id] = shuffle_draw_statistics
        method_values = {
            PRIMARY_METHOD: (
                heir_rna,
                heir_mass,
                HISTORICAL_LIBRARY_SIZE_AGGREGATION,
            ),
            SELECTIVE_METHOD: (
                selective_rna,
                selective_mass,
                HISTORICAL_LIBRARY_SIZE_AGGREGATION + "_nonabstained_only",
            ),
            EQUAL_CELL_METHOD: (heir_equal, equal_mass, "equal_cell"),
            HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD: (
                hard_assigned_type_spots,
                hard_assigned_type_mass,
                HISTORICAL_LIBRARY_SIZE_AGGREGATION + "_hard_assigned_type_mass",
            ),
            TYPE_MEAN_METHOD: (
                type_spots,
                type_mass,
                HISTORICAL_LIBRARY_SIZE_AGGREGATION,
            ),
            SOFT_TYPE_MEAN_METHOD: (
                soft_type_spots,
                soft_type_mass,
                HISTORICAL_LIBRARY_SIZE_AGGREGATION,
            ),
            EQUAL_CELL_HARD_TYPE_MEAN_METHOD: (
                equal_hard_type_spots,
                equal_hard_type_mass,
                "equal_cell",
            ),
            EQUAL_CELL_SOFT_TYPE_MEAN_METHOD: (
                equal_soft_type_spots,
                equal_soft_type_mass,
                "equal_cell",
            ),
            PSEUDOBULK_METHOD: (
                pseudobulk_spots,
                np.ones(len(truth.spot_ids)),
                "spatially_constant",
            ),
            SHUFFLE_METHOD: (
                shuffled_spots,
                shuffled_mass,
                "single_draw_0_complete_final_cell_record_shuffle_with_"
                + HISTORICAL_LIBRARY_SIZE_AGGREGATION,
            ),
        }
        method_values.update(r1_values)
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
            if method in baseline_estimands:
                methods[method]["estimand"] = dict(baseline_estimands[method])
            if method == SHUFFLE_METHOD:
                methods[method]["shuffle_role"] = "single_draw_0_preserved_for_backward_comparison"
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
                "reference_type_support": type_support,
                "type_map_diagnostics": type_map_diagnostics,
                "r1_reference_type_support": (
                    r1_type_support
                    if r1_type_support is not None
                    else {
                        "status": "unavailable",
                        "reason": "no hash-bound R1 manifest was registered",
                    }
                ),
                "methods": methods,
                "provenance": {
                    "prediction_sha256": case.predictions_sha256,
                    "truth_sha256": case.truth_sha256,
                    "reference_sha256": case.matched_reference_sha256,
                    "checkpoint_sha256": case.checkpoint_sha256,
                    "r1_reference": r1_provenance,
                    "refined_prediction": (
                        None
                        if case.section_id not in refined_predictions
                        else {
                            "path": str(refined_predictions[case.section_id].path),
                            "sha256": refined_predictions[case.section_id].sha256,
                            "checkpoint_sha256": refined_predictions[
                                case.section_id
                            ].checkpoint_sha256,
                            "refinement_audit_sha256": refined_predictions[
                                case.section_id
                            ].refinement_audit_sha256,
                            "telemetry_sha256": refined_predictions[
                                case.section_id
                            ].telemetry_sha256,
                            "refined_prototype_sha256": refined_predictions[
                                case.section_id
                            ].refined_prototype_sha256,
                            "native_prototype_sha256": refined_predictions[
                                case.section_id
                            ].native_prototype_sha256,
                            "native_scanvi_manifest_sha256": (
                                None if native_scanvi is None else native_scanvi.sha256
                            ),
                            "native_residual_geometry_sha256": (
                                None
                                if native_scanvi is None
                                else native_scanvi.specimen_residual_geometry[
                                    case.section_id
                                ].sha256
                            ),
                        }
                    ),
                },
            }
        )
    if len(shuffle_null_draw_statistics) != len(cases):
        raise RuntimeError("record-shuffle null is incomplete")
    shuffle_macro_statistics = np.mean(
        np.stack(
            [shuffle_null_draw_statistics[case["section_id"]] for case in cases],
            axis=0,
        ),
        axis=0,
    )
    shuffle_null_report = {
        "status": "retrospective_final_cell_record_shuffle_only",
        "permutations_per_specimen": plan.final_cell_record_shuffle_permutations,
        "seed": plan.primary_seeds[0],
        "seed_derivation": (
            "SHA-256(base seed, specimen, draw index); draw 0 preserves the historical "
            "single-shuffle seed"
        ),
        "record_unit": (
            "complete assigned-cell prediction record: expression and library-size weight "
            "move together"
        ),
        "single_draw_method": SHUFFLE_METHOD,
        "single_draw_index": 0,
        "specimens": shuffle_null_specimens,
        "equal_weight_specimen_macro": _compact_shuffle_distribution(shuffle_macro_statistics),
        "raw_permutation_values_reported": False,
        "does_not_replace": [
            "shuffled_image_features_followed_by_model_rerun",
            "coordinate_or_graph_shuffle_followed_by_model_rerun",
        ],
    }
    readiness = _readiness(
        plan,
        native_scanvi=native_scanvi,
        refined_predictions_validated=bool(refined_predictions),
        five_seed_predictions=five_seed_predictions,
        refinement_matrix_summary=refinement_matrix_summary,
    )
    method_macro = _method_macro_summaries(cases)
    locked_macro_spearman = {
        summary.method: summary.estimate
        for summary in locked.benchmark.summaries
        if summary.cohort_id == "snpatho_seq" and summary.metric == "median_gene_spearman"
    }
    reconciliation = {
        "interpretation": (
            "Locked-v0.2 and DeepBench-v1 evaluate different estimands; their numerical "
            "differences are expected and do not alter either negative conclusion"
        ),
        "macro_median_gene_spearman": {
            "locked_v0_2": {
                "heir": locked_macro_spearman.get("heir"),
                "type_mean": locked_macro_spearman.get("matched_type_mean"),
                "shuffle": locked_macro_spearman.get("heir_spatial_shuffle"),
            },
            "deepbench_v1": {
                "heir": method_macro[PRIMARY_METHOD]["metrics"]["median_gene_spearman"][
                    "macro_mean"
                ],
                "hard_type_mean": method_macro[TYPE_MEAN_METHOD]["metrics"]["median_gene_spearman"][
                    "macro_mean"
                ],
                "final_record_shuffle_draw_0": method_macro[SHUFFLE_METHOD]["metrics"][
                    "median_gene_spearman"
                ]["macro_mean"],
            },
        },
        "estimand_differences": [
            {
                "feature": "minimum_nuclei_per_spot",
                "locked_v0_2": ">=1",
                "deepbench_v1": ">=3",
            },
            {
                "feature": "cell_aggregation",
                "locked_v0_2": "equal-cell",
                "deepbench_v1": HISTORICAL_LIBRARY_SIZE_AGGREGATION,
            },
            {
                "feature": "type_profile",
                "locked_v0_2": "historical locked implementation",
                "deepbench_v1": "pooled raw counts divided by full-library mass",
            },
            {
                "feature": "constant_prediction_policy",
                "locked_v0_2": "earlier metric implementation",
                "deepbench_v1": "correlation fixed at zero when observed expression varies",
            },
            {
                "feature": "shuffle",
                "locked_v0_2": "historical spatial shuffle",
                "deepbench_v1": (
                    "complete final-cell-record shuffle draw 0; repeated null reported separately"
                ),
            },
        ],
    }
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
            "materialized_local_manifest": "reports/snpatho_r1_reference_manifest.json",
            "materialized_local_status": (
                "counts isolated and hash-bound; integrated published annotations are "
                "sensitivity-only; independent reannotation, primary scANVI and prototype-only "
                "predictions remain unavailable for the requested primary endpoint"
            ),
            "retrospective_r1_sensitivity_methods": [
                R1_HARD_ASSIGNED_MASS_TYPE_MEAN_METHOD,
                R1_HARD_TYPE_MEAN_METHOD,
                R1_SOFT_TYPE_MEAN_METHOD,
                R1_EQUAL_CELL_HARD_TYPE_MEAN_METHOD,
                R1_EQUAL_CELL_SOFT_TYPE_MEAN_METHOD,
            ],
            "primary_contrast_requirement": {
                "required_baselines": list(PRIMARY_MATCHED_R1_BASELINES),
                "success_rule": (
                    "refined HEIR must have positive paired median-gene Spearman macro delta "
                    "against both matched FFPE-R1 hard and soft type-mean baselines"
                ),
            },
            "historical_available": (
                "v0.2 pooled integrated multi-workflow reference containing FFPE snPATHO-seq, "
                "frozen Flex, and frozen 3-prime nuclei"
            ),
            "status": "historical_retrospective_only_not_primary_R1",
            "machine_readable_workflow_audit": ("reports/snpatho_reference_workflow_audit.json"),
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
            "available": (
                "repeated complete final cell-record shuffle null plus preserved draw-0 method"
            ),
            "status": "historical_diagnostic_only",
            "permutations": plan.final_cell_record_shuffle_permutations,
            "single_draw_method": SHUFFLE_METHOD,
            "does_not_replace": [
                "shuffled_image_features",
                "coordinate_shuffled_graph",
            ],
        },
        "final_cell_record_shuffle_null": shuffle_null_report,
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
        "metric_policy": {
            "expression_detection_auroc": (
                "AUROC(predicted expression, observed expression > 0); this is not a hotspot AUROC"
            ),
            "top_10_percent_hotspot": {
                "selection_size": "ceil(0.10 * spots), at least one",
                "tie_policy": (
                    "exact-k; descending expression, then ascending frozen spot-row index"
                ),
            },
            "morans_i_spatial_weights": {
                "graph": "directed unweighted 6-nearest-neighbor graph",
                "symmetrized": False,
                "row_standardized": False,
                "normalization": "n divided by total directed edge weight",
                "status": "historical DeepBench sensitivity graph, not Visium hex adjacency",
            },
        },
        "baseline_estimands": baseline_estimands,
        "locked_v0_2_deepbench_v1_reconciliation": reconciliation,
        "readiness": list(readiness),
        "cases": cases,
        "method_macro": method_macro,
        "primary": _primary_diagnostic(
            cases,
            plan,
            repeated_shuffle_statistics=shuffle_null_draw_statistics,
            native_scanvi=native_scanvi,
            five_seed_predictions=five_seed_predictions,
            refinement_matrix_summary=refinement_matrix_summary,
        ),
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
            estimands = report.get("baseline_estimands")
            if isinstance(estimands, Mapping):
                for method, estimand in estimands.items():
                    writer.writerow(
                        {
                            "record_type": "baseline_estimand",
                            "section_id": "all",
                            "method": method,
                            "aggregation": cast(Mapping[str, Any], estimand)["cell_rna_mass"],
                            "gene_name": "",
                            "metric": "profile_mass_estimand",
                            "value": "",
                            "spots_evaluated": "",
                            "status": "ok",
                            "reason": json.dumps(estimand, sort_keys=True),
                        }
                    )
            for case in report.get("cases", []):
                support = case.get("r1_reference_type_support")
                if isinstance(support, Mapping):
                    for metric in (
                        "count_reference_supported_types",
                        "prototype_supported_types",
                        "prototype_omitted_types",
                    ):
                        values = support.get(metric)
                        if isinstance(values, list):
                            for type_name in values:
                                writer.writerow(
                                    {
                                        "record_type": "type_support",
                                        "section_id": case["section_id"],
                                        "method": "matched_ffpe_r1_support_ladder",
                                        "aggregation": "",
                                        "gene_name": str(type_name),
                                        "metric": metric,
                                        "value": "1",
                                        "spots_evaluated": "",
                                        "status": "ok",
                                        "reason": "hash-validated count and prototype artifacts",
                                    }
                                )
                    for support_name in (
                        "native_scanvi_rare_complete_prototype_support",
                        "legacy_svd_sensitivity_prototype_support",
                    ):
                        bank_support = support.get(support_name)
                        if not isinstance(bank_support, Mapping):
                            continue
                        for metric in ("prototype_supported_types", "prototype_omitted_types"):
                            values = bank_support.get(metric)
                            if not isinstance(values, list):
                                continue
                            for type_name in values:
                                writer.writerow(
                                    {
                                        "record_type": "prototype_bank_type_support",
                                        "section_id": case["section_id"],
                                        "method": support_name,
                                        "aggregation": "",
                                        "gene_name": str(type_name),
                                        "metric": metric,
                                        "value": "1",
                                        "spots_evaluated": "",
                                        "status": "ok",
                                        "reason": bank_support["prototype_support_policy"],
                                    }
                                )
                diagnostics = case.get("type_map_diagnostics")
                if isinstance(diagnostics, Mapping):
                    entropy = cast(Mapping[str, Any], diagnostics["normalized_probability_entropy"])
                    for metric, value in entropy.items():
                        writer.writerow(
                            {
                                "record_type": "type_map",
                                "section_id": case["section_id"],
                                "method": "historical_type_probability_map",
                                "aggregation": "assigned_nuclei_in_primary_spots",
                                "gene_name": "",
                                "metric": "normalized_probability_entropy_%s" % metric,
                                "value": "%.12g" % value,
                                "spots_evaluated": diagnostics["spots_evaluated"],
                                "status": "ok",
                                "reason": "",
                            }
                        )
            shuffle_null = report.get("final_cell_record_shuffle_null")
            if isinstance(shuffle_null, Mapping):
                null_payloads = {
                    **dict(cast(Mapping[str, Any], shuffle_null["specimens"])),
                    "macro": shuffle_null["equal_weight_specimen_macro"],
                }
                for section_id, payload in null_payloads.items():
                    interval = payload["empirical_percentile_interval_95"]
                    metrics = {
                        "null_mean": payload["mean"],
                        "null_median": payload["median"],
                        "null_sample_standard_deviation": payload["sample_standard_deviation"],
                        "null_minimum": payload["minimum"],
                        "null_maximum": payload["maximum"],
                        "null_empirical_95_lower": interval["lower"],
                        "null_empirical_95_upper": interval["upper"],
                    }
                    for metric, value in metrics.items():
                        writer.writerow(
                            {
                                "record_type": "shuffle_null",
                                "section_id": section_id,
                                "method": SHUFFLE_METHOD,
                                "aggregation": "repeated_complete_final_cell_record_shuffle",
                                "gene_name": "",
                                "metric": metric,
                                "value": "%.12g" % value,
                                "spots_evaluated": "",
                                "status": "ok",
                                "reason": "%d deterministic permutations" % payload["permutations"],
                            }
                        )
                primary_payload = report.get("primary")
                if isinstance(primary_payload, Mapping):
                    for row in primary_payload["specimens"]:
                        comparison = row["repeated_final_record_shuffle_null_comparison"]
                        comparison_metrics = {
                            "observed_heir_median_gene_spearman": comparison[
                                "observed_heir_median_gene_spearman"
                            ],
                            "observed_heir_empirical_percentile_in_null": comparison[
                                "observed_heir_empirical_percentile_in_null"
                            ],
                            "observed_heir_minus_null_median": comparison[
                                "observed_heir_minus_null_median"
                            ],
                            "observed_heir_above_null_95_upper": float(
                                comparison["observed_heir_above_null_95_upper"]
                            ),
                        }
                        for metric, value in comparison_metrics.items():
                            writer.writerow(
                                {
                                    "record_type": "shuffle_null_comparison",
                                    "section_id": row["section_id"],
                                    "method": PRIMARY_METHOD,
                                    "aggregation": HISTORICAL_LIBRARY_SIZE_AGGREGATION,
                                    "gene_name": "",
                                    "metric": metric,
                                    "value": "%.12g" % value,
                                    "spots_evaluated": "",
                                    "status": "ok",
                                    "reason": "%d deterministic permutations"
                                    % comparison["null_permutations"],
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
            "Hash-bound FFPE-only count references and prototype banks are now consumed. The "
            "type-mean ladder separates hard-assigned mass, shared soft mass, expected soft "
            "mass, and equal-cell hard/soft estimands. It retains the published "
            "integrated-workflow annotation. A native-scANVI prototype-only control is scored "
            "in the development matrix, but the annotation and execution-provenance gates "
            "remain incomplete, so this is not the requested clean primary R1 comparison.",
            "",
            "Cell-to-spot weights in this diagnostic are **historical integrated-reference "
            "library-size weights**, not assay-corrected biological RNA-mass estimates. Both "
            "hard-argmax and probability-weighted soft historical type-mean baselines are "
            "reported under each explicit mass estimand. Missing prediction types fail closed; "
            "no global profile is substituted.",
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
            "Expression-detection AUROC uses observed expression > 0 as its label; top-10% "
            "Dice/Jaccard are the hotspot metrics. Exact top-decile sets break cutoff ties by "
            "ascending frozen spot-row index. Moran's I uses a directed, unweighted 6-NN graph "
            "that is not symmetrized or row-standardized.",
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
                "## Locked-v0.2 versus DeepBench-v1 reconciliation",
                "",
                "The two reports use different estimands, so their values need not match.",
                "",
                "| Feature | Locked-v0.2 | DeepBench-v1 |",
                "|---|---|---|",
            ]
        )
        lines.extend(
            "| %s | %s | %s |" % (row["feature"], row["locked_v0_2"], row["deepbench_v1"])
            for row in report["locked_v0_2_deepbench_v1_reconciliation"]["estimand_differences"]
        )
        estimands = report.get("baseline_estimands")
        if isinstance(estimands, Mapping):
            lines.extend(
                [
                    "",
                    "## Type-mean baseline estimands",
                    "",
                    "The legacy hard method IDs remain available, but their shared-soft-mass "
                    "estimand is now explicit.",
                    "",
                    "| Method | Reference | Cell profile | Cell RNA mass |",
                    "|---|---|---|---|",
                ]
            )
            lines.extend(
                "| %s | %s | %s | %s |"
                % (
                    method,
                    payload["reference"],
                    payload["cell_expression_profile"],
                    payload["cell_rna_mass"],
                )
                for method, payload in estimands.items()
            )
        lines.extend(
            [
                "",
                "## Reference type support",
                "",
                "| Specimen | Prediction types | Supported | Missing | Hard fallback cells |",
                "|---|---:|---:|---|---:|",
            ]
        )
        lines.extend(
            "| %s | %d | %d | %s | %d (%.4f) |"
            % (
                case["section_id"],
                len(case["reference_type_support"]["prediction_cell_types"]),
                len(case["reference_type_support"]["reference_supported_prediction_cell_types"]),
                ", ".join(case["reference_type_support"]["missing_prediction_cell_types"])
                or "none",
                case["reference_type_support"]["hard_assignment_global_fallback_cells"],
                case["reference_type_support"]["hard_assignment_global_fallback_cell_fraction"],
            )
            for case in report["cases"]
        )
        r1_cases = [
            case
            for case in report["cases"]
            if "prediction_cell_types" in case["r1_reference_type_support"]
        ]
        if r1_cases:
            lines.extend(
                [
                    "",
                    "FFPE-only R1 count-reference support (integrated-annotation sensitivity):",
                    "",
                    "| Specimen | Prediction types | Supported | Missing | Hard fallback cells |",
                    "|---|---:|---:|---|---:|",
                ]
            )
            lines.extend(
                "| %s | %d | %d | %s | %d (%.4f) |"
                % (
                    case["section_id"],
                    len(case["r1_reference_type_support"]["prediction_cell_types"]),
                    len(
                        case["r1_reference_type_support"][
                            "reference_supported_prediction_cell_types"
                        ]
                    ),
                    ", ".join(case["r1_reference_type_support"]["missing_prediction_cell_types"])
                    or "none",
                    case["r1_reference_type_support"]["hard_assignment_global_fallback_cells"],
                    case["r1_reference_type_support"][
                        "hard_assignment_global_fallback_cell_fraction"
                    ],
                )
                for case in r1_cases
            )
            lines.extend(
                [
                    "",
                    "Count-reference support and prototype-bank support are distinct:",
                    "",
                    "| Specimen | Count-reference-supported types | Prototype-supported types | "
                    "Prototype-omitted types |",
                    "|---|---|---|---|",
                ]
            )
            lines.extend(
                "| %s | %s | %s | %s |"
                % (
                    case["section_id"],
                    ", ".join(case["r1_reference_type_support"]["count_reference_supported_types"])
                    or "none",
                    ", ".join(case["r1_reference_type_support"]["prototype_supported_types"])
                    or "none",
                    ", ".join(case["r1_reference_type_support"]["prototype_omitted_types"])
                    or "none",
                )
                for case in r1_cases
            )
            if any(
                isinstance(
                    case["r1_reference_type_support"].get(
                        "native_scanvi_rare_complete_prototype_support"
                    ),
                    Mapping,
                )
                for case in r1_cases
            ):
                lines.extend(
                    [
                        "",
                        "Native rare-complete support is reported separately from the legacy "
                        "SVD sensitivity bank. Refined-run fairness uses the native bank:",
                        "",
                        "| Specimen | Fairness source | Native supported / omitted | Legacy "
                        "supported / omitted |",
                        "|---|---|---|---|",
                    ]
                )
                for case in r1_cases:
                    support = case["r1_reference_type_support"]
                    native_support = support.get("native_scanvi_rare_complete_prototype_support")
                    legacy_support = support["legacy_svd_sensitivity_prototype_support"]
                    if not isinstance(native_support, Mapping):
                        continue
                    lines.append(
                        "| %s | %s | %s / %s | %s / %s |"
                        % (
                            case["section_id"],
                            support["refined_run_fairness_prototype_source"],
                            ", ".join(native_support["prototype_supported_types"]) or "none",
                            ", ".join(native_support["prototype_omitted_types"]) or "none",
                            ", ".join(legacy_support["prototype_supported_types"]) or "none",
                            ", ".join(legacy_support["prototype_omitted_types"]) or "none",
                        )
                    )
        diagnostic_cases = [
            case
            for case in report["cases"]
            if isinstance(case.get("type_map_diagnostics"), Mapping)
        ]
        if diagnostic_cases:
            lines.extend(
                [
                    "",
                    "## Type-probability map audit",
                    "",
                    "Hard occupancy and hard/soft spot-mixture variation are computed over "
                    "assigned nuclei in the primary evaluated spots.",
                    "",
                    "| Specimen | Occupied hard types | Hard assignments | Mean normalized "
                    "entropy | Hard-mixture constant types | Soft-mixture constant types |",
                    "|---|---:|---|---:|---|---|",
                ]
            )
            lines.extend(
                "| %s | %d | %s | %.6f | %s | %s |"
                % (
                    case["section_id"],
                    case["type_map_diagnostics"]["occupied_hard_type_count"],
                    ", ".join(
                        "%s=%d" % (name, count)
                        for name, count in case["type_map_diagnostics"][
                            "hard_assignment_counts"
                        ].items()
                    ),
                    case["type_map_diagnostics"]["normalized_probability_entropy"]["mean"],
                    ", ".join(
                        case["type_map_diagnostics"]["hard_spot_mixture_spatially_constant_types"]
                    )
                    or "none",
                    ", ".join(
                        case["type_map_diagnostics"]["soft_spot_mixture_spatially_constant_types"]
                    )
                    or "none",
                )
                for case in diagnostic_cases
            )
        lines.extend(
            [
                "",
                "## Historical round-0 diagnostic",
                "",
                "The paired statistic is "
                "`median_g(rho_HEIR,g - rho_historical-integrated-hard-type-mean,g)`; it is "
                "not the difference between the two marginal medians.",
                "",
                "| Specimen | Median paired per-gene delta | MSE improvement vs hard type mean |",
                "|---|---:|---:|",
            ]
        )
        lines.extend(
            "| %s | %.6f | %.6f |"
            % (
                row["section_id"],
                row[
                    "median_paired_per_gene_spearman_delta_vs_historical_integrated_hard_type_mean"
                ],
                row["median_mse_improvement_vs_type_mean"],
            )
            for row in primary["specimens"]
        )
        shuffle_null = report["final_cell_record_shuffle_null"]
        macro_null = shuffle_null["equal_weight_specimen_macro"]
        lines.extend(
            [
                "",
                "## Repeated final-cell-record shuffle null",
                "",
                "The preserved draw-0 method is one member of a %d-per-specimen null. "
                "Expression and its library-size weight move together. This retrospective "
                "record shuffle does not replace image-feature or coordinate/graph reruns."
                % shuffle_null["permutations_per_specimen"],
                "",
                "| Specimen | HEIR median-gene Spearman | Null median | "
                "Null empirical 95% interval | HEIR percentile in null | Above null upper? |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in primary["specimens"]:
            comparison = row["repeated_final_record_shuffle_null_comparison"]
            interval = comparison["null_empirical_percentile_interval_95"]
            lines.append(
                "| %s | %.6f | %.6f | [%.6f, %.6f] | %.3f | %s |"
                % (
                    row["section_id"],
                    comparison["observed_heir_median_gene_spearman"],
                    comparison["null_median"],
                    interval["lower"],
                    interval["upper"],
                    comparison["observed_heir_empirical_percentile_in_null"],
                    "yes" if comparison["observed_heir_above_null_95_upper"] else "no",
                )
            )
        lines.extend(
            [
                "",
                "The equal-weight specimen-macro null median was **%.6f**, with empirical "
                "95%% interval **[%.6f, %.6f]**. HEIR exceeded the specimen null upper "
                "bound in one of three cases, so the prespecified at-least-two rule failed."
                % (
                    macro_null["median"],
                    macro_null["empirical_percentile_interval_95"]["lower"],
                    macro_null["empirical_percentile_interval_95"]["upper"],
                ),
            ]
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
        constant_rows = []
        for case in report["cases"]:
            for method, payload in case["methods"].items():
                summary = payload["summary"]
                predicted_constant = summary["prediction_constant_scored_zero_count"]
                observed_constant = summary["observed_constant_excluded_count"]
                if predicted_constant or observed_constant:
                    constant_rows.append(
                        (case["section_id"], method, predicted_constant, observed_constant)
                    )
        if constant_rows:
            lines.extend(
                [
                    "",
                    "## Constant-prediction audit",
                    "",
                    "Only nonzero counts are listed. A constant prediction receives correlation "
                    "zero when observed expression varies; an observed-constant gene is excluded.",
                    "",
                    "| Specimen | Method | Prediction-constant scored zero | "
                    "Observed-constant excluded |",
                    "|---|---|---:|---:|",
                ]
            )
            lines.extend("| %s | %s | %d | %d |" % row for row in constant_rows)
        requested_joint = primary["developmental_seed17_joint_contrast"]
        matrix_gate = primary["full_primary_evidence"]["refinement_matrix"]
        lines.extend(
            [
                "",
                "Macro mean of specimen median paired per-gene Spearman deltas: **%.6f**."
                % primary["macro_delta"],
                "",
                "Bootstrap fraction with delta > 0: **%.4f** (this is neither a p-value nor "
                "a posterior probability)."
                % primary["bootstrap"]["bootstrap_fraction_delta_positive"],
                "",
                "Requested refined-versus-type-mean endpoint: **%s**."
                % primary["requested_primary_status"],
                "",
                "Developmental seed-17 joint matched-R1 contrast: **%s**. This one-seed "
                "contrast is reported separately and cannot unlock a full-primary claim."
                % requested_joint["status"],
                "",
                "Full-primary refinement matrix: completeness **%s**; strict ordering **%s**."
                % (matrix_gate["matrix_status"], matrix_gate["strict_ordering_status"]),
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
