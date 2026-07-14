#!/usr/bin/env python3
"""Run the preregistered NatCommun regional-fusion v2 validation.

The completed HEST centered-nucleus qualification is an immutable architecture
diagnostic, not a prerequisite here.  This compact runner verifies and reuses
the frozen v1 regional computation engine, then substitutes only the registered
v2 molecular-state prototypes, constrained calibration, full alpha interval,
crop sensitivity, and indication-balanced decision rules.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Sequence

import numpy as np

from heir.evaluation import reference_fusion_v2

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs/natcommun_matched_regional_protocol_v2.json"
V1_PROTOCOL_PATH = REPO_ROOT / "configs/natcommun_matched_regional_protocol.json"
V1_BUILDER_PATH = REPO_ROOT / "scripts/build_natcommun_regional_source.py"
V1_RUNNER_PATH = REPO_ROOT / "scripts/benchmark_natcommun_reference_fusion.py"
REFERENCE_V2_PATH = REPO_ROOT / "src/heir/evaluation/reference_fusion_v2.py"
CROP_BUILDER_PATH = REPO_ROOT / "scripts/build_natcommun_crop_sensitivity.py"
UNI2_BUILDER_PATH = REPO_ROOT / "scripts/build_natcommun_uni2_sensitivity.py"
HOPTIMUS_MANIFEST_PATH = REPO_ROOT / "manifests/encoders/hoptimus1.json"
UNI2_MANIFEST_PATH = REPO_ROOT / "manifests/encoders/uni2h.json"
UNI2_ADAPTER_PATH = REPO_ROOT / "src/heir/features/uni2h.py"
ENCODER_BASE_PATH = REPO_ROOT / "src/heir/features/base.py"
ENCODER_FACTORY_PATH = REPO_ROOT / "src/heir/features/__init__.py"
V1_REFERENCE_PATH = REPO_ROOT / "src/heir/evaluation/reference_fusion.py"
NESTED_RIDGE_PATH = REPO_ROOT / "src/heir/evaluation/hest_nested_ridge.py"
SCORING_PATH = REPO_ROOT / "src/heir/evaluation/hest_scoring.py"

PROTOCOL_SCHEMA = "heir.natcommun_matched_regional_protocol.v2"
PREFLIGHT_SCHEMA = "heir.natcommun_regional_encoder_preflight.v2"
REGISTRATION_REVIEW_SCHEMA = "heir.natcommun_registration_review.v1"
REPORT_SCHEMA = "heir.natcommun_reference_fusion_encoder_report.v2"
CROP_SUPPLEMENT_SCHEMA = "heir.natcommun_crop_sensitivity.v1"
UNI2_SUPPLEMENT_SCHEMA = "heir.natcommun_uni2h_sensitivity.v1"
CROP_SUPPLEMENT_RECEIPT_SCHEMA = "heir.natcommun_crop_sensitivity_receipt.v1"
UNI2_SUPPLEMENT_RECEIPT_SCHEMA = "heir.natcommun_uni2h_sensitivity_receipt.v1"
CROP_SECTION_CACHE_SCHEMA = "heir.natcommun_crop_sensitivity_section_cache.v1"
UNI2_SECTION_CACHE_SCHEMA = "heir.natcommun_uni2h_section_cache.v1"

FROZEN_V1_PROTOCOL_SHA256 = "1a002e63dfe5480cd3272e68c0d5ae0358471a9dc5dba546fba4b7e14201dd5b"
FROZEN_V1_BUILDER_SHA256 = "3b6006f61c72cb46029366e30f1510195109bd17d6471d6b2b4c4d0f55c5fdbb"
FROZEN_V1_RUNNER_SHA256 = "2bdd2b539bb73d20e11ae7b395ad7ebdf308d9146d7e911ff56ac79dc8c958a3"
FROZEN_V1_REFERENCE_SHA256 = "005cbe840fa6fd6ff2a8259e030f15ee09803e6c3b07b7005947f3901a822e75"
FROZEN_NESTED_RIDGE_SHA256 = "6a3830ffc857185d38f9b4bade3cc831d5060ea4ec08eb2b42738679fd57d429"
FROZEN_SCORING_SHA256 = "3ea3965fa31ccb3cfd611933f2dcfb32492326c68c5d1b259e010670a7c51f02"
FROZEN_CROP_BUILDER_SHA256 = "b76c2f982fa3110a913326e7a45a01c65729da9087804c9d02ac5083b48b06f9"
FROZEN_UNI2_BUILDER_SHA256 = "569d7a812b54d1adcfac8f1b555a45cf9d07deb124ceb1e1de10dc2e06d474b7"
FROZEN_UNI2_ADAPTER_SHA256 = "7b8ebdfd496ef37652e1273067590e8f547a0c76bd40a8753680f1b5dd854f67"
FROZEN_ENCODER_BASE_SHA256 = "5b6d26bb4cb69fcd6454a8868f65699a3a287db1985fddd002ae854108014a86"
FROZEN_ENCODER_FACTORY_SHA256 = "d363e99327d1b77abd996dd02e943a4089d2584c93f93cded7f25e86f9b66d24"
FROZEN_HOPTIMUS_MANIFEST_SHA256 = "f6852288e1ae146a4865bf19e38ce994c0be9ce1c2bfa09bdf77747043ac8fd9"
FROZEN_UNI2_MANIFEST_SHA256 = "4ce7aad048abe8be99e6b1542d7eff88dc46e00fdf75057ca01728b21bc2f369"
FROZEN_HEST_REPORT_SHA256 = "2685efc9574a1b6c9b2ff8f5a08cf372b038a1eaadd271f91ff24228b6060f1f"
FROZEN_HEST_SOURCE_SHA256 = "f7e7d4e97727cc17e71a81a252ab35fd2ca1c0e70054cba3ed38c2f7b7f65636"
PRE_AMENDMENT_GIT_HEAD = "a3bfa7f58dbfbbfdd6a13510f3146d5dd68a00b4"
PRE_AMENDMENT_PROTOCOL_SHA256 = "e1e14456e4a8a3cff5a33360592bb7784ee4c4809070b634f06ccc0c5d51bec8"
PRE_AMENDMENT_RUNNER_SHA256 = "ccc26bbbc0475ac734d4abddc508c0bffa8fcb43fe93e5a0481ba4b0b765ad6f"
PRE_AMENDMENT_REFERENCE_V2_SHA256 = (
    "38d1aca82489f24a1240c62d147cdd6e611579dd9a31e4902f4c78e14453a2e5"
)
PRE_AMENDMENT_PREFLIGHT_SHA256 = "eb3781cfcfaef39d7c2610e8a5c629c8bdecbc6f3c0925643f0c0f468528807d"
PRE_AMENDMENT_FAILURE_LOG_SHA256 = (
    "e9e53c9de2a7faaee2fa36b4641b5ca4ce56da4ef2f7e6d1bdb2c83a6f09a18e"
)

EXPECTED_SECTIONS = (
    "B1_2",
    "B1_4",
    "B2_2",
    "B3_2",
    "B4_2",
    "L1_2",
    "L1_4",
    "L2_2",
    "L3_2",
    "L4_2",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
)
PRIMARY_DONORS_BY_INDICATION: Mapping[str, tuple[str, ...]] = {
    "breast": ("B1", "B3", "B4"),
    "lung": ("L1", "L2", "L3", "L4"),
    "dlbcl": ("D1", "D2", "D3", "D4", "D5", "D6"),
}
PRIMARY_DONORS = tuple(
    donor for indication in PRIMARY_DONORS_BY_INDICATION.values() for donor in indication
)
DONOR_INDICATION = {
    donor: indication
    for indication, donors in PRIMARY_DONORS_BY_INDICATION.items()
    for donor in donors
}
PRIMARY_COMPARISONS = (
    "M3_vs_M0_incremental_reference",
    "M3_vs_M1_image_beyond_reference",
    "M3_vs_M2_continuous_state_beyond_type_routing",
    "M3_vs_M4_exact_pairing",
    "M3_vs_M6_matched_specificity",
    "M3_vs_M7_generic_specificity",
)
IMAGE_CONTENT_COMPARISONS = (
    "M3_vs_M5_blank_image_content",
    "M3_vs_M5_coordinates_image_content",
)
ALL_REGISTERED_COMPARISONS = PRIMARY_COMPARISONS + IMAGE_CONTENT_COMPARISONS
DEFAULT_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_FUSION_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0)
RARE_STATE_BASELINE_RECALL_MINIMUM = 0.2
THRESHOLD_COMPARISON_TOLERANCE = 1.0e-12
CROP_IDS = ("target_55um", "context_112um")
HOPTIMUS_ENCODER_ID = "hoptimus1_primary"
UNI2_ENCODER_ID = "uni2h_secondary"
PROGRAM_QUALITY_SCHEMA = "heir.natcommun_reliability_qualified_program_quality.v1"
PROGRAM_SCORE_CALL_MODELS = (
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5_blank",
    "M5_coordinates",
    "M7",
    "M8_raw_cross_half",
    "M6",
)
PROGRAM_RARE_STATE_CALL_MODELS = (
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5_blank",
    "M5_coordinates",
    "M7",
)


@dataclasses.dataclass(frozen=True)
class _EncoderInputs:
    encoder_id: str
    role: str
    repository: str
    crop_sources: Mapping[str, object]
    supplement_path: Path
    supplement_sha256: str
    supplement_receipt: Mapping[str, object]
    implementation_files: Mapping[Path, str]

    def __post_init__(self) -> None:
        if set(self.crop_sources) != set(CROP_IDS):
            raise ValueError("encoder inputs require the exact registered crop arms")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if (
        str(value) != normalized
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return normalized


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not a readable JSON object") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _json_scalar(archive: np.lib.npyio.NpzFile, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(str(np.asarray(archive[name]).reshape(-1)[0]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is missing or malformed") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: object) -> str:
    """Hash an array with the exact frozen NatCommun builder convention."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _strict_feature_stats(value: object, label: str) -> Mapping[str, object]:
    """Recompute the complete feature receipt without trusting its headlines."""

    values = np.asarray(value)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"{label} needs at least two rows")
    width = values.shape[1]
    totals = np.zeros(width, dtype=np.float64)
    sum_squares = np.zeros(width, dtype=np.float64)
    minimum = np.full(width, np.inf, dtype=np.float32)
    maximum = np.full(width, -np.inf, dtype=np.float32)
    minimum_row_norm = np.inf
    maximum_row_norm = -np.inf
    for start in range(0, len(values), 1024):
        block = np.asarray(values[start : start + 1024], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"{label} contains non-finite values")
        totals += block.sum(axis=0, dtype=np.float64)
        sum_squares += np.square(block, dtype=np.float64).sum(axis=0, dtype=np.float64)
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
        row_norms = np.square(block, dtype=np.float64).sum(axis=1)
        minimum_row_norm = min(minimum_row_norm, float(row_norms.min()))
        maximum_row_norm = max(maximum_row_norm, float(row_norms.max()))
    centered_energy = float(
        np.maximum(0.0, sum_squares - np.square(totals) / float(len(values))).sum()
    )
    varying = int(np.count_nonzero(maximum > minimum))
    if minimum_row_norm <= 0 or varying == 0 or centered_energy <= 0:
        raise ValueError(f"{label} is degenerate")
    return {
        "rows": int(values.shape[0]),
        "width": int(width),
        "dtype": str(values.dtype),
        "finite": True,
        "minimum_squared_row_norm": float(minimum_row_norm),
        "maximum_squared_row_norm": float(maximum_row_norm),
        "variable_feature_dimensions": varying,
        "total_centered_feature_energy": centered_energy,
        "array_sha256": _array_sha256(values),
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_protocol(expected_sha256: str) -> Mapping[str, object]:
    if _sha256(PROTOCOL_PATH) != expected_sha256:
        raise ValueError("v2 protocol does not match --expected-protocol-sha256")
    protocol = _load_json(PROTOCOL_PATH, "v2 protocol")
    hest = protocol.get("hest_architecture_diagnostic")
    decision = protocol.get("decision")
    superseded = protocol.get("supersedes_protocol")
    amendment = protocol.get("pre_result_technical_amendment")
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("cohort_id") != "NatCommun_2025_s41467_025_59005_9_E-MTAB-14560"
        or protocol.get("doi") != "10.1038/s41467-025-59005-9"
        or protocol.get("analysis_scope")
        != "retrospective_regional_spot_level_cell_non_authorizing"
        or protocol.get("observation_level") != "Visium_v2_spot_regional_not_cellular"
        or protocol.get("reference_modality") != "matched_Chromium_FLEX_snRNA_from_same_FFPE_block"
        or protocol.get("registered_before_outcome_exposure") is not True
        or not isinstance(superseded, Mapping)
        or superseded.get("path") != "configs/natcommun_matched_regional_protocol.json"
        or superseded.get("sha256") != FROZEN_V1_PROTOCOL_SHA256
        or not isinstance(hest, Mapping)
        or hest.get("role") != "immutable_non_gating_diagnostic_only"
        or hest.get("report_sha256") != FROZEN_HEST_REPORT_SHA256
        or hest.get("source_sha256") != FROZEN_HEST_SOURCE_SHA256
        or hest.get("frozen_result")
        != "failed_natural_unmasked_112um_centered_nucleus_geometry_gate"
        or hest.get("regional_benchmark_blocking") is not False
        or not isinstance(decision, Mapping)
        or decision.get("M8_may_block") is not False
        or decision.get("cell_level_HEIR_authorized") is not False
        or decision.get("independent_replication_still_required") is not True
    ):
        raise ValueError("v2 protocol has an unexpected scientific scope")
    expected_amendment = {
        "id": "2026-07-14_pre_result_molecular_kmeans_completion",
        "status": "registered_before_any_endpoint_summary_or_scientific_decision",
        "previous_identity": {
            "git_head": PRE_AMENDMENT_GIT_HEAD,
            "protocol_sha256": PRE_AMENDMENT_PROTOCOL_SHA256,
            "runner_sha256": PRE_AMENDMENT_RUNNER_SHA256,
            "reference_v2_sha256": PRE_AMENDMENT_REFERENCE_V2_SHA256,
            "hoptimus_preflight_path": (
                "/mnt/seagate/HEIR_runs/natcommun_regional_v2/hoptimus_preflight.json"
            ),
            "hoptimus_preflight_sha256": PRE_AMENDMENT_PREFLIGHT_SHA256,
        },
        "failed_attempt": {
            "encoder": HOPTIMUS_ENCODER_ID,
            "output_dir": ("/mnt/seagate/HEIR_runs/natcommun_regional_v2/hoptimus_primary"),
            "experiment": "target_55um::state_kmeans_8::program_total::natural",
            "failure_stage": (
                "inner_matched_primary_reference_bank_before_fusion_parameter_selection_"
                "and_heldout_endpoint_scoring_or_output"
            ),
            "log_path": (
                "/mnt/seagate/HEIR_runs/natcommun_regional_v2/failed_pre_result_attempts/"
                "a3bfa7f_hoptimus_kmeans_cap100/benchmark.log"
            ),
            "log_sha256": PRE_AMENDMENT_FAILURE_LOG_SHA256,
            "exit_code": 1,
            "experiment_checkpoint_files_written": 0,
            "report_files_written": 0,
            "scientific_outputs_exposed": [],
            "analyst_visible_endpoint_result_exposure": False,
        },
        "runtime_diagnosis": {
            "donor": "L3",
            "cell_type": "Myeloid",
            "rows": 5569,
            "latent_width": 8,
            "seed": "11297572760357870275",
            "previous_maximum_iterations": 100,
            "exact_label_stability_iteration": 105,
            "objective_monotonically_nonincreasing": True,
            "empty_cluster_repair_used": False,
            "assignment_cycle_detected": False,
            "upstream_training_only_ST_computation_occurred": True,
            "scientific_loss_effect_p_value_or_decision_inspected": False,
        },
        "completion_scan": {
            "scope": (
                "program_and_PCA_all_registered_natural_and_equalized_inner_outer_"
                "matched_wrong_and_pooled_bank_seeds_without_image_scoring"
            ),
            "exact_bank_constructions": 920,
            "donor_type_fits": 3882,
            "fits_requiring_more_than_100_iterations": 23,
            "maximum_exact_label_stability_iteration": 180,
            "failures_by_1000_iterations": 0,
            "assignment_cycles": 0,
            "empty_cluster_repairs": 0,
            "image_scores_effects_p_values_or_decisions_inspected": False,
        },
        "amended_completion_rule": {
            "maximum_iterations": 1000,
            "convergence": "exact_consecutive_repaired_assignment_equality",
            "assignment_cycle_detection": (
                "SHA256_canonical_little_endian_int64_label_assignment_repeat_fails_closed"
            ),
            "applies_uniformly_to": ("every_donor_type_fold_crop_endpoint_and_bank_condition"),
            "unchanged": [
                "k_8_primary_and_k_1_centroid_diagnostic",
                "deterministic_farthest_point_initialization",
                "squared_euclidean_objective",
                "deterministic_tie_order",
                "donor_type_partitioning",
                "cluster_count_weights",
                "seeds",
                "model_arms",
                "endpoints",
                "multiplicity",
                "decision_rules",
            ],
        },
        "encoder_execution_scope": {
            "hoptimus1": "primary_only",
            "uni2_h": "excluded_not_run_by_user_instruction",
        },
    }
    if amendment != expected_amendment:
        raise ValueError("v2 pre-result technical amendment changed")
    frozen = protocol.get("immutable_computation_dependencies")
    if not isinstance(frozen, Mapping):
        raise ValueError("v2 protocol lacks immutable dependencies")
    dependency_contract = {
        "v1_source_protocol": (V1_PROTOCOL_PATH, FROZEN_V1_PROTOCOL_SHA256),
        "source_builder": (V1_BUILDER_PATH, FROZEN_V1_BUILDER_SHA256),
        "v1_computation_engine": (V1_RUNNER_PATH, FROZEN_V1_RUNNER_SHA256),
        "v1_reference_fusion_primitives": (
            V1_REFERENCE_PATH,
            FROZEN_V1_REFERENCE_SHA256,
        ),
        "nested_ridge_primitives": (NESTED_RIDGE_PATH, FROZEN_NESTED_RIDGE_SHA256),
        "scoring_primitives": (SCORING_PATH, FROZEN_SCORING_SHA256),
        "hoptimus_crop_sensitivity_builder": (
            CROP_BUILDER_PATH,
            FROZEN_CROP_BUILDER_SHA256,
        ),
        "uni2_sensitivity_builder": (UNI2_BUILDER_PATH, FROZEN_UNI2_BUILDER_SHA256),
        "uni2_adapter": (UNI2_ADAPTER_PATH, FROZEN_UNI2_ADAPTER_SHA256),
        "encoder_base": (ENCODER_BASE_PATH, FROZEN_ENCODER_BASE_SHA256),
        "encoder_factory": (ENCODER_FACTORY_PATH, FROZEN_ENCODER_FACTORY_SHA256),
    }
    protocol_pinned_v2_dependencies = {
        "v2_computation_engine": Path(__file__).resolve(),
        "v2_reference_fusion_primitives": REFERENCE_V2_PATH,
    }
    expected_dependency_names = {
        *dependency_contract,
        *protocol_pinned_v2_dependencies,
        "source_sha256",
    }
    if set(frozen) != expected_dependency_names or frozen.get("source_sha256") != (
        "required_at_execution"
    ):
        raise ValueError("v2 protocol immutable dependency family changed")
    for name, (path, sha256) in dependency_contract.items():
        expected_row = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256,
        }
        if frozen.get(name) != expected_row or _sha256(path) != sha256:
            raise ValueError(f"frozen regional dependency changed: {name}")
    for name, path in protocol_pinned_v2_dependencies.items():
        row = frozen.get(name)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row.get("path") != str(path.relative_to(REPO_ROOT))
        ):
            raise ValueError(f"protocol-pinned v2 dependency identity changed: {name}")
        sha256 = _validate_sha256(str(row.get("sha256", "")), f"{name} SHA-256")
        if _sha256(path) != sha256:
            raise ValueError(f"protocol-pinned v2 dependency bytes changed: {name}")
    expected_semantics = {
        "primary_donors_by_indication": {
            key: list(value) for key, value in PRIMARY_DONORS_BY_INDICATION.items()
        },
        "decisive_comparisons": list(PRIMARY_COMPARISONS),
        "image_content_controls": list(IMAGE_CONTENT_COMPARISONS),
        "failed_reference_sensitivity_donors": ["B2"],
    }
    if any(protocol.get(key) != value for key, value in expected_semantics.items()):
        raise ValueError("v2 protocol donor map or comparison family changed")
    encoders = protocol.get("encoders")
    primary_encoder = encoders.get("primary") if isinstance(encoders, Mapping) else None
    secondary_encoder = encoders.get("secondary") if isinstance(encoders, Mapping) else None
    if (
        not isinstance(primary_encoder, Mapping)
        or primary_encoder.get("repository") != "bioptimus/H-optimus-1"
        or primary_encoder.get("revision") != "3592cb220dec7a150c5d7813fb56e68bd57473b9"
        or primary_encoder.get("manifest") != "manifests/encoders/hoptimus1.json"
        or primary_encoder.get("manifest_sha256") != FROZEN_HOPTIMUS_MANIFEST_SHA256
        or _sha256(HOPTIMUS_MANIFEST_PATH) != FROZEN_HOPTIMUS_MANIFEST_SHA256
        or primary_encoder.get("official_local_parity") != "required_exact_manifest_pass"
        or primary_encoder.get("mode") != "frozen_CUDA_inference_only"
        or primary_encoder.get("fine_tuning") != "prohibited"
        or not isinstance(secondary_encoder, Mapping)
        or secondary_encoder.get("repository") != "MahmoodLab/UNI2-h"
        or secondary_encoder.get("revision") != "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"
        or secondary_encoder.get("manifest") != "manifests/encoders/uni2h.json"
        or secondary_encoder.get("manifest_sha256") != FROZEN_UNI2_MANIFEST_SHA256
        or _sha256(UNI2_MANIFEST_PATH) != FROZEN_UNI2_MANIFEST_SHA256
        or secondary_encoder.get("qualification")
        != (
            "exact_manifest_checkpoint_config_and_implementation_hashes_without_official_"
            "local_parity_claim"
        )
        or secondary_encoder.get("execution")
        != "separate_source_separate_report_same_frozen_design"
        or secondary_encoder.get("may_rescue_primary_failure") is not False
        or secondary_encoder.get("strengthens_encoder_robustness_if_concordant") is not True
    ):
        raise ValueError("v2 protocol encoder identities or roles changed")

    regional_preflight = protocol.get("regional_preflight")
    family_requirements = (
        regional_preflight.get("encoder_family_requirements")
        if isinstance(regional_preflight, Mapping)
        else None
    )
    hoptimus_preflight = (
        family_requirements.get("hoptimus_primary")
        if isinstance(family_requirements, Mapping)
        else None
    )
    uni2_preflight = (
        family_requirements.get("uni2_secondary")
        if isinstance(family_requirements, Mapping)
        else None
    )
    visible_control = (
        regional_preflight.get("visible_control")
        if isinstance(regional_preflight, Mapping)
        else None
    )
    expected_preflight_components = [
        "exact_source_model_protocol_and_implementation_hashes",
        "encoder_family_specific_identity_and_qualification",
        "finite_nondegenerate_image_features_global_and_per_section_for_each_encoder_and_crop_arm",
        "registered_spot_centers_and_exact_alignment_artifact_hashes",
        "blinded_visual_registration_review_for_all_16_sections",
        "donor_held_out_indication_prediction_above_training_only_baseline",
        "blank_and_within_section_deranged_image_controls_constructible",
        "matched_wrong_and_same_indication_generic_reference_banks_constructible",
        "outer_training_only_ST_reliability_screen_available",
    ]
    if (
        not isinstance(regional_preflight, Mapping)
        or regional_preflight.get("independent_of_hest_geometry_gate") is not True
        or regional_preflight.get("required_components") != expected_preflight_components
        or regional_preflight.get("failure_action")
        != "stop_before_any_ST_endpoint_or_reference_fusion_fit"
        or not isinstance(hoptimus_preflight, Mapping)
        or hoptimus_preflight.get("crop_arms") != list(CROP_IDS)
        or hoptimus_preflight.get("official_local_parity") != "required_exact_manifest_pass"
        or hoptimus_preflight.get("visible_control") != "required_separately_for_each_crop_arm"
        or not isinstance(uni2_preflight, Mapping)
        or uni2_preflight.get("crop_arms") != list(CROP_IDS)
        or uni2_preflight.get("official_local_parity") != "not_claimed"
        or uni2_preflight.get("visible_control") != "required_separately_for_each_crop_arm"
        or uni2_preflight.get("decision_is_separate_from_primary") is not True
        or not isinstance(visible_control, Mapping)
        or visible_control.get("endpoint") != "indication"
        or visible_control.get("metric") != "indication_balanced_donor_macro_accuracy"
        or visible_control.get("baseline") != "outer_training_indication_prior_argmax"
        or visible_control.get("pass_rule") != "strictly_greater_than_baseline"
        or visible_control.get("maximum_spots_per_section") != 128
        or visible_control.get("ridge_alphas") != list(DEFAULT_RIDGE_ALPHAS)
    ):
        raise ValueError("v2 protocol encoder-specific preflight changed")

    expected_crops = [
        {
            "id": "target_55um",
            "tissue_signal_width_um": 55.0,
            "model_canvas_width_um": 112.0,
            "construction": (
                "centered_55um_signal_with_surrounding_pixels_whitened_on_native_"
                "112um_Hoptimus_canvas"
            ),
            "role": "primary_target_matched_sensitivity",
        },
        {
            "id": "context_112um",
            "tissue_signal_width_um": 112.0,
            "model_canvas_width_um": 112.0,
            "construction": "natural_unmasked_registered_context",
            "role": "primary_regional_context_sensitivity",
        },
    ]
    if (
        protocol.get("crop_arms") != expected_crops
        or protocol.get("model_arms")
        != [
            "M0_H_only",
            "M1_matched_reference_only",
            "M2_H_type_routing_plus_matched_reference",
            "M3_full_H_query_plus_matched_reference",
            "M4_within_section_deranged_H_plus_matched_reference",
            "M5_blank_and_coordinates_plus_matched_reference",
            "M6_H_plus_hard_wrong_donor_reference",
            "M7_H_plus_same_indication_generic_reference",
            "M8_ST_cross_fitted_split_half_secondary_floor",
        ]
        or protocol.get("endpoints")
        != [
            "outer_training_reliability_qualified_fixed_programs",
            "outer_training_only_PCA_total_expression",
        ]
        or protocol.get("bank_conditions") != ["natural", "composition_equalized"]
    ):
        raise ValueError("v2 protocol crop, model, or endpoint family changed")

    representation = protocol.get("reference_representation")
    calibration = protocol.get("cross_assay_calibration")
    if (
        not isinstance(representation, Mapping)
        or representation.get("latent_input") != "fold_training_donor_ST_calibrated_snRNA_latent"
        or representation.get("uses_fold_training_donor_ST_via_cross_assay_calibration") is not True
        or representation.get("uses_heldout_or_inner_validation_donor_ST_outcomes") is not False
        or representation.get("primary")
        != {
            "method": "deterministic_molecular_kmeans_within_donor_and_type",
            "prototypes_per_donor_type": 8,
            "maximum_iterations": 1000,
            "convergence_rule": "exact_consecutive_repaired_assignment_equality",
            "nonconsecutive_assignment_cycle_action": "fail_closed",
        }
        or representation.get("type_mean_baseline")
        != {
            "method": "one_centroid_per_donor_and_type",
            "prototypes_per_donor_type": 1,
            "role": "secondary_diagnostic",
        }
        or representation.get("random_hash_averages") != "prohibited"
        or reference_fusion_v2.MOLECULAR_KMEANS_MAXIMUM_ITERATIONS != 1000
        or not isinstance(calibration, Mapping)
        or calibration.get("primary") != "indication_aware_diagonal_identity_regularized"
        or calibration.get("fit_rows") != "paired_training_donor_means_within_indication"
        or calibration.get("held_out_donor_ST_used") is not False
        or calibration.get("full_affine_from_donor_means") != "prohibited_as_underdetermined"
        or calibration.get("minimum_training_donors_for_indication_specific_map") != 2
        or calibration.get("sparse_indication_behavior") != "global_diagonal_fallback"
        or calibration.get("ridge_selection")
        != "leave_one_training_donor_out_on_the_same_hierarchical_mapping"
        or calibration.get("ridge_selection_weighting")
        != "indication_equal_then_donor_equal_within_indication"
        or calibration.get("global_fallback_fit_weighting")
        != "indication_equal_then_donor_equal_within_indication"
        or set(calibration)
        != {
            "primary",
            "fit_rows",
            "held_out_donor_ST_used",
            "full_affine_from_donor_means",
            "minimum_training_donors_for_indication_specific_map",
            "sparse_indication_behavior",
            "ridge_selection",
            "ridge_selection_weighting",
            "global_fallback_fit_weighting",
        }
    ):
        raise ValueError("v2 protocol reference representation or calibration changed")
    execution = protocol.get("execution_parameters")
    fusion = protocol.get("one_step_fusion")
    thresholds = protocol.get("effect_thresholds")
    if (
        not isinstance(execution, Mapping)
        or execution.get("seed") != 17
        or execution.get("ridge_alpha_grid") != list(DEFAULT_RIDGE_ALPHAS)
        or execution.get("pca_components") != 20
        or execution.get("pca_genes_selected_within_outer_training_fold") != 256
        or execution.get("bootstrap_iterations") != 2000
        or execution.get("ridge_and_fusion_selection_weighting")
        != "indication_equal_then_donor_equal_within_indication"
        or execution.get("atomic_per_experiment_checkpoints") is not True
        or execution.get("primary_full_experiments") != "both_crops_x_both_endpoints_x_both_banks"
        or execution.get("centroid_diagnostic_experiments")
        != "both_crops_x_program_total_x_natural_bank_only"
        or execution.get("hoptimus_primary_report")
        != "separate_8_primary_plus_2_centroid_diagnostic_experiments"
        or execution.get("uni2_secondary_report")
        != "separate_8_primary_plus_2_centroid_diagnostic_experiments"
        or execution.get("encoder_results_may_not_be_pooled") is not True
        or not isinstance(fusion, Mapping)
        or fusion.get("fusion_alpha_grid") != list(DEFAULT_FUSION_ALPHAS)
        or fusion.get("temperature_grid") != list(DEFAULT_TEMPERATURES)
        or fusion.get("selection") != "grouped_training_donor_only_indication_equal_objective"
        or fusion.get("support_adaptation") != "support_times_coverage_times_one_minus_uncertainty"
        or fusion.get("iterative_refinement") != "prohibited"
        or not isinstance(thresholds, Mapping)
        or thresholds.get("minimum_relative_MSE_gain_M3_vs_M0") != 0.05
        or thresholds.get("minimum_positive_donor_fraction_M3_vs_M0") != 0.70
        or thresholds.get("minimum_positive_indications") != 2
        or thresholds.get("severe_indication_reversal_relative_MSE") != -0.05
        or thresholds.get("require_no_severe_indication_reversal") is not True
        or thresholds.get("minimum_M3_median_within_section_variance_ratio") != 0.5
        or thresholds.get("maximum_M3_abstention_fraction") != 0.5
        or thresholds.get("minimum_M3_median_type_coverage") != 0.5
        or thresholds.get("rare_state_maximum_median_recall_drop_from_M0") != 0.2
        or thresholds.get("rare_state_maximum_single_target_recall_drop_when_M0_at_least_0_2")
        != 0.3
    ):
        raise ValueError("v2 protocol execution parameters or quality thresholds changed")
    if (
        protocol.get("multiplicity")
        != "Holm_across_all_primary_crop_endpoint_bank_and_registered_M3_control_comparisons"
        or protocol.get("familywise_alpha") != 0.05
        or decision.get("crop_arm_pass")
        != (
            "all_decisive_controls_effect_size_donor_consistency_indication_heterogeneity_"
            "variance_coverage_abstention_and_rare_state_guardrails_pass"
        )
        or decision.get("overall_regional_support")
        != "at_least_one_crop_arm_passes_with_global_multiplicity_control"
        or decision.get("regional_research_software_authorized_if_supported") is not True
    ):
        raise ValueError("v2 protocol multiplicity or decision rule changed")
    resource_limits = protocol.get("resource_limits")
    if (
        not isinstance(resource_limits, Mapping)
        or resource_limits.get("cuda_required_for_ridge_and_encoder") is not True
        or resource_limits.get("default_cpu_threads") != 4
        or resource_limits.get("maximum_cpu_threads") != 8
        or resource_limits.get("default_embedding_batch_size") != 4
        or resource_limits.get("maximum_dense_spots_per_preflight_section") != 128
        or resource_limits.get("parallel_outer_experiments") != 1
    ):
        raise ValueError("v2 protocol resource limits changed")
    return protocol


def _load_v1_runner(expected_sha256: str) -> ModuleType:
    if expected_sha256 != FROZEN_V1_RUNNER_SHA256 or _sha256(V1_RUNNER_PATH) != expected_sha256:
        raise ValueError("frozen v1 computation engine SHA-256 mismatch")
    name = "heir_frozen_natcommun_v1_engine"
    spec = importlib.util.spec_from_file_location(name, V1_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen v1 computation engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _implementation_receipt(
    args: argparse.Namespace,
    protocol: Mapping[str, object],
    *,
    extra_expected_files: Mapping[Path, str] | None = None,
) -> Mapping[str, object]:
    expected = {
        Path(__file__).resolve(): args.expected_runner_sha256,
        PROTOCOL_PATH: args.expected_protocol_sha256,
        V1_RUNNER_PATH: args.expected_v1_runner_sha256,
        V1_BUILDER_PATH: args.expected_v1_builder_sha256,
        V1_PROTOCOL_PATH: args.expected_v1_protocol_sha256,
        REFERENCE_V2_PATH: args.expected_reference_v2_sha256,
        V1_REFERENCE_PATH: FROZEN_V1_REFERENCE_SHA256,
        NESTED_RIDGE_PATH: FROZEN_NESTED_RIDGE_SHA256,
        SCORING_PATH: FROZEN_SCORING_SHA256,
    }
    if extra_expected_files is not None:
        for path, sha256 in extra_expected_files.items():
            resolved = path.expanduser().resolve()
            if resolved in expected and expected[resolved] != sha256:
                raise ValueError(f"conflicting implementation identity for {resolved}")
            expected[resolved] = sha256
    observed = {str(path): _sha256(path) for path in expected}
    mismatches = {
        str(path): {"expected": sha, "observed": observed[str(path)]}
        for path, sha in expected.items()
        if observed[str(path)] != sha
    }
    if mismatches:
        raise ValueError(f"implementation SHA-256 mismatch: {mismatches}")
    try:
        git_head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_dirty = None
    if not git_head or git_dirty is not False:
        raise ValueError("scientific execution requires a committed, clean git worktree")
    return {
        "schema": "heir.natcommun_regional_implementation_receipt.v2",
        "files": observed,
        "protocol_schema": protocol["schema"],
        "command": args.command,
        "git_head": git_head,
        "git_worktree_dirty_at_start": git_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _feature_preflight(
    source: object,
    *,
    encoder_id: str = HOPTIMUS_ENCODER_ID,
    crop_id: str = "context_112um",
) -> Mapping[str, object]:
    features = np.asarray(source.image_features, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError(f"{encoder_id}/{crop_id} features must be a non-empty finite matrix")
    centered_norm = float(np.linalg.norm(features - features.mean(axis=0)))
    varying = int(np.sum(np.ptp(features, axis=0) > 1.0e-8))
    per_section: dict[str, Mapping[str, object]] = {}
    for section in sorted(set(source.section_ids.tolist())):
        local = features[source.section_ids == section]
        local_norm = float(np.linalg.norm(local - local.mean(axis=0)))
        local_varying = int(np.sum(np.ptp(local, axis=0) > 1.0e-8))
        if len(local) < 2 or local_norm <= 1.0e-8 or local_varying == 0:
            raise ValueError(f"{encoder_id}/{crop_id} features are degenerate in {section}")
        per_section[section] = {
            "rows": len(local),
            "centered_frobenius_norm": local_norm,
            "varying_dimensions": local_varying,
        }
    if centered_norm <= 1.0e-8 or varying == 0:
        raise ValueError(f"{encoder_id}/{crop_id} features are globally degenerate")
    return {
        "passed": True,
        "encoder_id": encoder_id,
        "crop_id": crop_id,
        "rows": len(features),
        "width": features.shape[1],
        "finite": True,
        "centered_frobenius_norm": centered_norm,
        "varying_dimensions": varying,
        "per_section": per_section,
    }


def _validate_source_v2_contract(source: object) -> Mapping[str, object]:
    receipt = source.source_receipt
    encoder = receipt.get("encoder") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != "heir.natcommun_regional_source_receipt.v2"
        or receipt.get("protocol_sha256") != FROZEN_V1_PROTOCOL_SHA256
        or receipt.get("builder_implementation_sha256") != FROZEN_V1_BUILDER_SHA256
        or receipt.get("observation_level") != "Visium_v2_spot_regional_not_cellular"
        or not isinstance(encoder, Mapping)
        or encoder.get("repository") != "bioptimus/H-optimus-1"
        or encoder.get("revision") != "3592cb220dec7a150c5d7813fb56e68bd57473b9"
        or encoder.get("device") != "cuda"
        or source.image_features.shape[1] != 1536
        or source.blank_image_feature.shape != (1536,)
    ):
        raise ValueError("source is not bound to the frozen regional H-optimus contract")
    if tuple(sorted(set(source.donor_ids.tolist()))) != tuple(sorted(PRIMARY_DONORS)):
        raise ValueError("source does not contain the exact 13 primary donors")
    spot_map = {
        donor: sorted(set(source.indication_ids[source.donor_ids == donor].tolist()))
        for donor in PRIMARY_DONORS
    }
    reference_map = {
        donor: sorted(set(source.sc_indication_ids[source.sc_donor_ids == donor].tolist()))
        for donor in PRIMARY_DONORS
    }
    expected = {donor: [DONOR_INDICATION[donor]] for donor in PRIMARY_DONORS}
    if spot_map != expected or reference_map != expected:
        raise ValueError("source donor-to-indication mapping differs from the frozen map")
    return {
        "passed": True,
        "source_protocol_sha256": FROZEN_V1_PROTOCOL_SHA256,
        "source_builder_sha256": FROZEN_V1_BUILDER_SHA256,
        "encoder_repository": encoder["repository"],
        "encoder_revision": encoder["revision"],
        "feature_width": 1536,
        "spot_donor_indication_map": spot_map,
        "reference_donor_indication_map": reference_map,
    }


def _registration_preflight(
    source: object,
    review_path: Path,
    expected_review_sha256: str,
) -> Mapping[str, object]:
    review_path = review_path.expanduser().resolve()
    if _sha256(review_path) != expected_review_sha256:
        raise ValueError("registration review SHA-256 mismatch")
    review = _load_json(review_path, "registration review")
    review_sections = review.get("sections")
    source_identity_matches = review.get("source_sha256") in {
        source.source_receipt.get("source_sha256"),
        _sha256(source.path),
    }
    if (
        review.get("schema") != REGISTRATION_REVIEW_SCHEMA
        or not source_identity_matches
        or review.get("review_blinded_to_ST_and_reference_outcomes") is not True
        or not str(review.get("reviewer", "")).strip()
        or not isinstance(review_sections, Mapping)
        or set(map(str, review_sections)) != set(EXPECTED_SECTIONS)
    ):
        raise ValueError("registration review scope or blinding is invalid")

    source_sections = source.source_receipt.get("sections")
    if not isinstance(source_sections, list):
        raise ValueError("source receipt lacks per-section registration provenance")
    by_section = {
        str(row.get("section")): row for row in source_sections if isinstance(row, Mapping)
    }
    if set(by_section) != set(EXPECTED_SECTIONS):
        raise ValueError("source receipt does not cover the exact 16 registered sections")

    checked: dict[str, Mapping[str, object]] = {}
    for section in EXPECTED_SECTIONS:
        row = by_section[section]
        embedding = row.get("embedding")
        provenance = row.get("spaceranger_provenance")
        manual = review_sections[section]
        if not all(isinstance(value, Mapping) for value in (embedding, provenance, manual)):
            raise ValueError(f"registration provenance is incomplete for {section}")
        registration = embedding.get("registration_qc")
        if (
            not isinstance(registration, Mapping)
            or registration.get("all_spot_centers_inside_image") is not True
            or float(registration.get("maximum_padding_fraction", 1.0)) > 0.75
            or provenance.get("exact_invocation_fields_verified") is not True
            or provenance.get("alignment_visual_review_required_before_exact_image_claims")
            is not True
            or manual.get("status") != "passed"
        ):
            raise ValueError(f"registration checks did not pass for {section}")
        identity_fields = {
            "h_and_e": ("h_and_e_path", "h_and_e_sha256"),
            "final_alignment": ("final_alignment_path", "final_alignment_sha256"),
            "alignment_qc_image": (
                "alignment_qc_image_path",
                "alignment_qc_image_sha256",
            ),
        }
        current: dict[str, str] = {}
        for label, (path_key, sha_key) in identity_fields.items():
            artifact = Path(str(provenance.get(path_key, ""))).expanduser().resolve()
            declared = str(provenance.get(sha_key, ""))
            if not artifact.is_file() or _sha256(artifact) != declared:
                raise ValueError(f"current {label} identity differs for {section}")
            if manual.get(f"{label}_sha256") != declared:
                raise ValueError(f"manual review {label} identity differs for {section}")
            current[label] = declared
        checked[section] = {
            **current,
            "maximum_padding_fraction": float(registration["maximum_padding_fraction"]),
            "manual_status": "passed",
        }
    return {
        "passed": True,
        "review_path": str(review_path),
        "review_sha256": expected_review_sha256,
        "reviewer": str(review["reviewer"]),
        "blinded": True,
        "sections": checked,
    }


def _stable_section_sample(
    spot_ids: np.ndarray,
    section_ids: np.ndarray,
    maximum_per_section: int,
) -> np.ndarray:
    selected: list[int] = []
    for section in sorted(set(section_ids.tolist())):
        indices = np.flatnonzero(section_ids == section).tolist()
        indices.sort(
            key=lambda index: (
                hashlib.sha256(str(spot_ids[index]).encode("utf-8")).digest(),
                str(spot_ids[index]),
            )
        )
        selected.extend(indices[:maximum_per_section])
    return np.asarray(sorted(selected), dtype=np.int64)


def _indication_equal_prediction_loss(
    legacy: ModuleType,
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
) -> float:
    unknown = sorted(set(donors.tolist()) - set(DONOR_INDICATION))
    if unknown:
        raise ValueError(f"selection rows contain unregistered donors: {unknown}")
    donor_loss = legacy.donor_section_macro_loss(
        truth,
        prediction,
        donors,
        sections,
    )["donor_mse"]
    if set(donor_loss) != set(donors.tolist()):
        raise ValueError("selection loss does not cover every training donor")
    per_indication = {
        indication: float(
            np.mean(
                [
                    float(loss)
                    for donor, loss in donor_loss.items()
                    if DONOR_INDICATION[donor] == indication
                ]
            )
        )
        for indication in sorted({DONOR_INDICATION[donor] for donor in donor_loss})
    }
    return float(np.mean(tuple(per_indication.values())))


def _select_ridge_alpha_indication_equal(
    legacy: ModuleType,
    features: np.ndarray,
    targets: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    alphas: Sequence[float],
    *,
    seed: int,
    device: str,
) -> tuple[float, Mapping[str, float]]:
    unique = sorted(set(donors.tolist()))
    folds = legacy.grouped_donor_folds(
        donors,
        n_splits=min(5, len(unique)),
        seed=seed,
    )
    predictions = np.empty((len(alphas), len(targets), targets.shape[1]), dtype=np.float64)
    for train, validation in folds:
        fit = legacy.fit_weighted_ridge_grid(
            features[train],
            targets[train],
            alphas,
            legacy._donor_section_weights(donors[train], sections[train]),
            device=device,
        )
        if device == "cuda" and not str(fit.fit_device).startswith("cuda"):
            raise RuntimeError("CUDA ridge silently fell back to CPU; scientific run aborted")
        predictions[:, validation] = fit.predict(features[validation])
    losses = {
        f"{float(alpha):g}": _indication_equal_prediction_loss(
            legacy,
            targets,
            predictions[index],
            donors,
            sections,
        )
        for index, alpha in enumerate(alphas)
    }
    selected = min((loss, float(alpha)) for alpha, loss in losses.items())[1]
    return selected, losses


def _select_fusion_parameters_indication_equal(
    legacy: ModuleType,
    image: np.ndarray,
    type_probabilities: np.ndarray,
    target: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    banks: Mapping[str, object],
    type_names: np.ndarray,
    temperatures: Sequence[float],
    alphas: Sequence[float],
) -> tuple[float, float, float, Mapping[str, float]]:
    losses: dict[str, float] = {}
    candidates: list[tuple[float, float, float, float]] = []
    for temperature in temperatures:
        reference = np.empty_like(target)
        distance = np.empty(len(target), dtype=np.float64)
        coverage = np.empty(len(target), dtype=np.float64)
        uncertainty = np.empty(len(target), dtype=np.float64)
        for donor in sorted(set(donors.tolist())):
            selected = donors == donor
            state, diagnostics = legacy._retrieve(
                image[selected],
                type_probabilities[selected],
                banks[donor],
                type_names,
                float(temperature),
            )
            reference[selected] = state
            distance[selected] = diagnostics["support_distance"]
            coverage[selected] = diagnostics["type_coverage"]
            uncertainty[selected] = diagnostics["reference_uncertainty"]
        finite = distance[np.isfinite(distance)]
        threshold = float(np.quantile(finite, 0.95)) if len(finite) else 1.0
        diagnostics = {
            "support_distance": distance,
            "type_coverage": coverage,
            "reference_uncertainty": uncertainty,
        }
        for alpha in alphas:
            prediction, _receipt = legacy._adaptive_fusion(
                image,
                reference,
                diagnostics,
                float(alpha),
                threshold,
            )
            loss = _indication_equal_prediction_loss(
                legacy,
                target,
                prediction,
                donors,
                sections,
            )
            key = f"temperature={float(temperature):g}|alpha={float(alpha):g}"
            losses[key] = loss
            candidates.append((loss, float(alpha), float(temperature), threshold))
    if not candidates:
        raise ValueError("fusion parameter selection has no candidates")
    _loss, alpha, temperature, threshold = min(candidates)
    return alpha, temperature, threshold, losses


def _recompute_visible_control(rows: object) -> Mapping[str, object]:
    if not isinstance(rows, Mapping) or set(map(str, rows)) != set(PRIMARY_DONORS):
        raise ValueError("visible-control rows must cover the exact 13 primary donors")
    normalized: dict[str, Mapping[str, object]] = {}
    for donor in PRIMARY_DONORS:
        row = rows[donor]
        if not isinstance(row, Mapping):
            raise ValueError(f"visible-control row is malformed for {donor}")
        truth = str(row.get("truth", ""))
        prediction = str(row.get("prediction", ""))
        baseline = str(row.get("baseline_prediction", ""))
        if (
            truth != DONOR_INDICATION[donor]
            or prediction not in PRIMARY_DONORS_BY_INDICATION
            or baseline not in PRIMARY_DONORS_BY_INDICATION
        ):
            raise ValueError(f"visible-control identities are malformed for {donor}")
        normalized[donor] = {
            "truth": truth,
            "prediction": prediction,
            "baseline_prediction": baseline,
            "model_correct": prediction == truth,
            "baseline_correct": baseline == truth,
        }
    by_indication: dict[str, Mapping[str, float]] = {}
    for indication, donors in PRIMARY_DONORS_BY_INDICATION.items():
        model = float(np.mean([normalized[donor]["model_correct"] for donor in donors]))
        baseline = float(np.mean([normalized[donor]["baseline_correct"] for donor in donors]))
        by_indication[indication] = {
            "model_accuracy": model,
            "baseline_accuracy": baseline,
        }
    model_balanced = float(np.mean([row["model_accuracy"] for row in by_indication.values()]))
    baseline_balanced = float(np.mean([row["baseline_accuracy"] for row in by_indication.values()]))
    return {
        "passed": model_balanced > baseline_balanced,
        "metric": "indication_balanced_donor_macro_accuracy",
        "model": model_balanced,
        "outer_training_majority_baseline": baseline_balanced,
        "increment": model_balanced - baseline_balanced,
        "per_indication": by_indication,
        "per_donor": normalized,
        "uses_ST_or_reference_outcomes": False,
    }


def _visible_control(
    source: object,
    legacy: ModuleType,
    *,
    ridge_alphas: Sequence[float],
    maximum_per_section: int,
    seed: int,
    device: str,
) -> Mapping[str, object]:
    selected = _stable_section_sample(
        source.spot_ids,
        source.section_ids,
        maximum_per_section,
    )
    features = source.image_features[selected]
    donors = source.donor_ids[selected]
    sections = source.section_ids[selected]
    indications = source.indication_ids[selected]
    indication_names = np.asarray(sorted(PRIMARY_DONORS_BY_INDICATION))
    one_hot = np.eye(len(indication_names))[
        np.asarray([int(np.flatnonzero(indication_names == value)[0]) for value in indications])
    ]
    rows: dict[str, Mapping[str, object]] = {}
    fold_receipts: dict[str, Mapping[str, object]] = {}
    for fold_index, held_out in enumerate(PRIMARY_DONORS):
        train = donors != held_out
        test = donors == held_out
        if not train.any() or not test.any():
            raise ValueError(f"visible-control sample lacks donor {held_out}")
        alpha, cv = _select_ridge_alpha_indication_equal(
            legacy,
            features[train],
            one_hot[train],
            donors[train],
            sections[train],
            ridge_alphas,
            seed=seed + fold_index,
            device=device,
        )
        prediction = legacy._fit_predict_ridge(
            features[train],
            one_hot[train],
            features[test],
            donors[train],
            sections[train],
            alpha,
            device,
        )
        predicted = str(indication_names[int(np.argmax(prediction.mean(axis=0)))])
        training_donor_labels = [
            DONOR_INDICATION[donor] for donor in PRIMARY_DONORS if donor != held_out
        ]
        baseline = min(
            sorted(set(training_donor_labels)),
            key=lambda value: (-training_donor_labels.count(value), value),
        )
        rows[held_out] = {
            "truth": DONOR_INDICATION[held_out],
            "prediction": predicted,
            "baseline_prediction": baseline,
        }
        fold_receipts[held_out] = {
            "training_donors": sorted(set(donors[train].tolist())),
            "heldout_rows": int(test.sum()),
            "selected_ridge_alpha": alpha,
            "inner_grouped_donor_cv": cv,
            "selection_weighting": "indication_equal_then_donor_equal_within_indication",
        }
    result = _recompute_visible_control(rows)
    return {
        **result,
        "sampled_rows": len(selected),
        "sampled_spot_ids_sha256": _array_sha256(np.asarray(source.spot_ids[selected], dtype="S")),
        "maximum_rows_per_section": maximum_per_section,
        "sampling": "SHA256_spot_identity_without_ST_or_reference_outcomes",
        "fold_receipts": fold_receipts,
    }


def _control_constructibility(source: object, legacy: ModuleType) -> Mapping[str, object]:
    if (
        source.blank_image_feature.shape != (source.image_features.shape[1],)
        or not np.isfinite(source.blank_image_feature).all()
    ):
        raise ValueError("blank-image control feature is not constructible")
    shuffle_rows: dict[str, Mapping[str, object]] = {}
    for donor_index, donor in enumerate(PRIMARY_DONORS):
        local = source.donor_ids == donor
        permutation = legacy.deterministic_group_derangement(
            source.section_ids[local],
            source.spot_ids[local],
            seed=17 + donor_index,
        )
        fixed = int(np.sum(permutation == np.arange(len(permutation))))
        if fixed:
            raise ValueError(f"within-section image derangement has fixed points for {donor}")
        shuffle_rows[donor] = {"rows": int(local.sum()), "fixed_points": fixed}

    banks: dict[str, Mapping[str, object]] = {}
    for donor in PRIMARY_DONORS:
        indication = DONOR_INDICATION[donor]
        others = [value for value in PRIMARY_DONORS_BY_INDICATION[indication] if value != donor]
        donor_banks: dict[str, object] = {}
        for mode in ("natural", "composition_equalized"):
            matched, _ = legacy._bank_indices(source, donor, [donor], mode, pooled=False, seed=17)
            matched_qualified, matched_receipt = legacy._qualify_reference_indices(
                source.sc_type_ids, matched
            )
            wrong_receipts = {}
            for wrong in others:
                indices, _ = legacy._bank_indices(
                    source, donor, [wrong], mode, pooled=False, seed=17
                )
                qualified, receipt = legacy._qualify_reference_indices(source.sc_type_ids, indices)
                if not len(qualified):
                    raise ValueError(f"{mode} wrong bank {wrong} is unsupported for {donor}")
                wrong_receipts[wrong] = receipt
            generic, _ = legacy._bank_indices(source, donor, others, mode, pooled=True, seed=17)
            generic_qualified, generic_receipt = legacy._qualify_reference_indices(
                source.sc_type_ids, generic
            )
            if not len(matched_qualified) or not len(generic_qualified):
                raise ValueError(f"{mode} matched/generic bank is unsupported for {donor}")
            donor_banks[mode] = {
                "matched": matched_receipt,
                "wrong": wrong_receipts,
                "generic": generic_receipt,
            }
        banks[donor] = donor_banks
    return {
        "passed": True,
        "blank": dict(source.blank_receipt),
        "shuffle": shuffle_rows,
        "banks": banks,
        "matched_wrong_and_same_indication_generic_available": True,
    }


def _reliability_preflight(source: object, legacy: ModuleType) -> Mapping[str, object]:
    full = legacy._log_normalize(source.st_full, source.st_total_full)
    half_a = legacy._log_normalize(source.st_half_a * 2.0, source.st_total_half_a * 2.0)
    half_b = legacy._log_normalize(source.st_half_b * 2.0, source.st_total_half_b * 2.0)
    reference = legacy._log_normalize(source.sc_counts, source.sc_total_counts)
    folds: dict[str, Mapping[str, object]] = {}
    for held_out in PRIMARY_DONORS:
        training = [donor for donor in PRIMARY_DONORS if donor != held_out]
        endpoint = legacy._endpoint_fold(
            source,
            full,
            half_a,
            half_b,
            reference,
            training,
        )
        gate = legacy._program_reliability_gate(
            endpoint.half_a,
            endpoint.half_b,
            source.donor_ids,
            training,
            endpoint.target_names,
        )
        folds[held_out] = gate
    passed = all(row.get("status") == "feasible" for row in folds.values())
    return {
        "passed": passed,
        "outer_training_only": True,
        "heldout_ST_used": False,
        "folds": folds,
    }


def _component_gate(components: object) -> Mapping[str, object]:
    if not isinstance(components, Mapping) or set(components) != {"common", "crop_arms"}:
        raise ValueError("preflight components are incomplete")
    common = components["common"]
    crop_arms = components["crop_arms"]
    if (
        not isinstance(common, Mapping)
        or set(common) != {"registration", "controls", "reliability"}
        or not isinstance(crop_arms, Mapping)
        or set(crop_arms) != set(CROP_IDS)
    ):
        raise ValueError("preflight common or crop-arm components are incomplete")

    expected_primary_sections = set(EXPECTED_SECTIONS) - {"B2_2"}
    crop_passes: dict[str, bool] = {}
    crop_component_passes: dict[str, Mapping[str, bool]] = {}
    for crop_id in CROP_IDS:
        crop = crop_arms[crop_id]
        if not isinstance(crop, Mapping) or set(crop) != {"features", "visible_control"}:
            raise ValueError(f"preflight crop arm is incomplete: {crop_id}")
        features = crop["features"]
        feature_sections = features.get("per_section") if isinstance(features, Mapping) else None
        feature_passed = bool(
            isinstance(features, Mapping)
            and features.get("crop_id") == crop_id
            and str(features.get("encoder_id", "")) != ""
            and features.get("finite") is True
            and float(features.get("centered_frobenius_norm", 0.0)) > 1.0e-8
            and int(features.get("varying_dimensions", 0)) > 0
            and isinstance(feature_sections, Mapping)
            and set(feature_sections) == expected_primary_sections
            and all(
                isinstance(row, Mapping)
                and int(row.get("rows", 0)) >= 2
                and float(row.get("centered_frobenius_norm", 0.0)) > 1.0e-8
                and int(row.get("varying_dimensions", 0)) > 0
                for row in feature_sections.values()
            )
        )
        if not isinstance(features, Mapping) or features.get("passed") is not feature_passed:
            raise ValueError(f"preflight features headline is inconsistent: {crop_id}")

        declared_visible = crop["visible_control"]
        recomputed_visible = _recompute_visible_control(
            declared_visible.get("per_donor") if isinstance(declared_visible, Mapping) else None
        )
        for field in (
            "passed",
            "metric",
            "model",
            "outer_training_majority_baseline",
            "increment",
            "per_indication",
            "per_donor",
            "uses_ST_or_reference_outcomes",
        ):
            if (
                not isinstance(declared_visible, Mapping)
                or declared_visible.get(field) != recomputed_visible[field]
            ):
                raise ValueError(f"visible-control {field} is inconsistent: {crop_id}")
        visible_passed = bool(recomputed_visible["passed"])
        crop_component_passes[crop_id] = {
            "features": feature_passed,
            "visible_control": visible_passed,
        }
        crop_passes[crop_id] = bool(feature_passed and visible_passed)

    registration = common["registration"]
    registration_sections = (
        registration.get("sections") if isinstance(registration, Mapping) else None
    )
    registration_passed = bool(
        isinstance(registration, Mapping)
        and registration.get("blinded") is True
        and isinstance(registration_sections, Mapping)
        and set(registration_sections) == set(EXPECTED_SECTIONS)
        and all(
            isinstance(row, Mapping)
            and row.get("manual_status") == "passed"
            and float(row.get("maximum_padding_fraction", 1.0)) <= 0.75
            for row in registration_sections.values()
        )
    )

    controls = common["controls"]
    shuffle = controls.get("shuffle") if isinstance(controls, Mapping) else None
    banks = controls.get("banks") if isinstance(controls, Mapping) else None
    controls_passed = bool(
        isinstance(controls, Mapping)
        and controls.get("matched_wrong_and_same_indication_generic_available") is True
        and isinstance(controls.get("blank"), Mapping)
        and isinstance(shuffle, Mapping)
        and set(shuffle) == set(PRIMARY_DONORS)
        and all(
            isinstance(row, Mapping) and int(row.get("fixed_points", -1)) == 0
            for row in shuffle.values()
        )
        and isinstance(banks, Mapping)
        and set(banks) == set(PRIMARY_DONORS)
    )

    reliability = common["reliability"]
    reliability_folds = reliability.get("folds") if isinstance(reliability, Mapping) else None
    reliability_passed = bool(
        isinstance(reliability, Mapping)
        and reliability.get("outer_training_only") is True
        and reliability.get("heldout_ST_used") is False
        and isinstance(reliability_folds, Mapping)
        and set(reliability_folds) == set(PRIMARY_DONORS)
        and all(
            isinstance(row, Mapping) and row.get("status") == "feasible"
            for row in reliability_folds.values()
        )
    )
    common_passes = {
        "registration": registration_passed,
        "controls": controls_passed,
        "reliability": reliability_passed,
    }
    for name, recomputed in common_passes.items():
        declared = common[name]
        if not isinstance(declared, Mapping) or declared.get("passed") is not recomputed:
            raise ValueError(f"preflight {name} headline is inconsistent")
    return {
        "passed": bool(all(common_passes.values()) and all(crop_passes.values())),
        "all_crop_arms_required": True,
        "common_component_passes": common_passes,
        "crop_arm_component_passes": crop_component_passes,
        "crop_arm_passes": crop_passes,
        "failure_action": "stop_before_any_ST_endpoint_or_reference_fusion_fit",
        "hest_geometry_gate_required": False,
    }


def _build_encoder_preflight_components(
    *,
    base_source: object,
    inputs: _EncoderInputs,
    legacy: ModuleType,
    registration_review: Path,
    expected_registration_review_sha256: str,
    ridge_alphas: Sequence[float],
    maximum_per_section: int,
    seed: int,
    device: str,
) -> Mapping[str, object]:
    return {
        "common": {
            "registration": _registration_preflight(
                base_source,
                registration_review,
                expected_registration_review_sha256,
            ),
            "controls": _control_constructibility(inputs.crop_sources["context_112um"], legacy),
            "reliability": _reliability_preflight(base_source, legacy),
        },
        "crop_arms": {
            crop_id: {
                "features": _feature_preflight(
                    crop_source,
                    encoder_id=inputs.encoder_id,
                    crop_id=crop_id,
                ),
                "visible_control": _visible_control(
                    crop_source,
                    legacy,
                    ridge_alphas=ridge_alphas,
                    maximum_per_section=maximum_per_section,
                    seed=seed,
                    device=device,
                ),
            }
            for crop_id, crop_source in inputs.crop_sources.items()
        },
    }


def _run_encoder_preflight(
    args: argparse.Namespace,
    inputs_loader: object,
) -> int:
    protocol = _load_protocol(args.expected_protocol_sha256)
    legacy = _load_v1_runner(args.expected_v1_runner_sha256)
    numeric_backend = legacy._configure_numeric_backend(args.cpu_threads, args.device)
    source = legacy.load_source(
        args.source,
        expected_primary_donors=len(PRIMARY_DONORS),
        expected_source_sha256=args.expected_source_sha256,
    )
    source_contract = _validate_source_v2_contract(source)
    inputs = inputs_loader(args, source)
    implementation = _implementation_receipt(
        args,
        protocol,
        extra_expected_files=inputs.implementation_files,
    )
    maximum = int(protocol["regional_preflight"]["visible_control"]["maximum_spots_per_section"])
    components = _build_encoder_preflight_components(
        base_source=source,
        inputs=inputs,
        legacy=legacy,
        registration_review=args.registration_review,
        expected_registration_review_sha256=args.expected_registration_review_sha256,
        ridge_alphas=args.ridge_alphas,
        maximum_per_section=maximum,
        seed=args.seed,
        device=args.device,
    )
    gate = _component_gate(components)
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed" if gate["passed"] else "failed",
        "source": str(source.path),
        "source_sha256": args.expected_source_sha256,
        "source_contract": source_contract,
        "encoder": {
            "id": inputs.encoder_id,
            "role": inputs.role,
            "repository": inputs.repository,
            "primary_decision_authority": inputs.role == "primary",
            "may_rescue_primary": False,
        },
        "supplement": {
            "path": str(inputs.supplement_path),
            "sha256": inputs.supplement_sha256,
            "receipt": inputs.supplement_receipt,
        },
        "protocol_sha256": args.expected_protocol_sha256,
        "registration_review_sha256": args.expected_registration_review_sha256,
        "implementation_receipt": implementation,
        "numeric_backend": numeric_backend,
        "components": components,
        "component_digest": _canonical_sha256(components),
        "gate": gate,
        "non_gating_diagnostics": {
            "hest_hoptimus1": {
                "result": "failed_frozen_centered_nucleus_geometry_gate",
                "report_sha256": FROZEN_HEST_REPORT_SHA256,
                "source_sha256": FROZEN_HEST_SOURCE_SHA256,
                "regional_benchmark_blocking": False,
            }
        },
    }
    _atomic_json(args.output, report)
    print(args.output, flush=True)
    return 0 if gate["passed"] else 2


def _run_hoptimus_preflight(args: argparse.Namespace) -> int:
    return _run_encoder_preflight(args, _load_hoptimus_inputs)


def _run_uni2_preflight(args: argparse.Namespace) -> int:
    return _run_encoder_preflight(args, _load_uni2_inputs)


def _scalar_archive_text(archive: np.lib.npyio.NpzFile, name: str) -> str:
    try:
        value = np.asarray(archive[name])
    except KeyError as error:
        raise ValueError(f"{name} is missing") from error
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    return str(value.item())


def _source_row_contract(source: object) -> Mapping[str, object]:
    """Read the complete 16-section row contract, including sensitivity-only B2."""

    path = Path(source.path).expanduser().resolve()
    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = _scalar_archive_text(archive, "schema_version")
            spot_ids = np.asarray(archive["spot_ids"]).astype(str)
            barcodes = np.asarray(archive["barcode_ids"]).astype(str)
            sections = np.asarray(archive["section_ids"]).astype(str)
            pixel_xy = np.asarray(archive["pixel_xy"], dtype=np.float64)
            native_features = np.asarray(archive["image_features"])
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("frozen source lacks its complete row contract") from error
    receipt = source.source_receipt
    rows = len(spot_ids)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != "heir.natcommun_regional_source_receipt.v2"
        or receipt.get("builder_implementation_sha256") != FROZEN_V1_BUILDER_SHA256
        or schema != "heir.natcommun_regional_source.v2"
        or spot_ids.ndim != 1
        or not rows
        or len(set(spot_ids.tolist())) != rows
        or barcodes.shape != (rows,)
        or sections.shape != (rows,)
        or pixel_xy.shape != (rows, 2)
        or not np.isfinite(pixel_xy).all()
        or native_features.dtype != np.float16
        or native_features.shape != (rows, 1536)
        or not np.array_equal(spot_ids, np.char.add(np.char.add(sections, ":"), barcodes))
        or list(dict.fromkeys(sections.tolist())) != list(EXPECTED_SECTIONS)
    ):
        raise ValueError("frozen source row identities are invalid")
    section_receipts = receipt.get("sections")
    if (
        not isinstance(section_receipts, list)
        or len(section_receipts) != len(EXPECTED_SECTIONS)
        or [item.get("section") if isinstance(item, Mapping) else None for item in section_receipts]
        != list(EXPECTED_SECTIONS)
    ):
        raise ValueError("frozen source does not contain the exact 16 section receipts")
    return {
        "path": path,
        "sha256": _sha256(path),
        "schema": schema,
        "spot_ids": spot_ids,
        "barcodes": barcodes,
        "sections": sections,
        "pixel_xy": pixel_xy,
        "native_features": native_features,
        "section_receipts": section_receipts,
    }


def _manifest_contract(path: Path, repository: str) -> tuple[Mapping[str, object], str]:
    manifest = _load_json(path, f"{repository} manifest")
    if (
        manifest.get("schema") != "heir.encoder_manifest.v1"
        or manifest.get("repository") != repository
        or manifest.get("feature_width") != 1536
        or manifest.get("input_pixels") != 224
        or manifest.get("fine_tuning") != "prohibited"
    ):
        raise ValueError(f"current {repository} manifest is not the pinned encoder contract")
    return manifest, _sha256(path)


def _section_rows(contract: Mapping[str, object], section: str) -> np.ndarray:
    sections = np.asarray(contract["sections"])
    rows = np.flatnonzero(sections == section).astype(np.int64)
    if len(rows) < 2:
        raise ValueError(f"section {section} has fewer than two source spots")
    return rows


def _validate_common_section_hashes(
    item: Mapping[str, object],
    *,
    section: str,
    rows: np.ndarray,
    source_contract: Mapping[str, object],
    expected_source_sha256: str,
) -> None:
    expected = {
        "source_sha256": expected_source_sha256,
        "section": section,
        "row_indices_sha256": _array_sha256(rows),
        "spot_ids_sha256": _array_sha256(
            np.asarray(np.asarray(source_contract["spot_ids"])[rows], dtype="S")
        ),
        "barcodes_sha256": _array_sha256(
            np.asarray(np.asarray(source_contract["barcodes"])[rows], dtype="S")
        ),
        "pixel_xy_sha256": _array_sha256(
            np.asarray(np.asarray(source_contract["pixel_xy"])[rows], dtype=np.float64)
        ),
    }
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError(f"supplement row hash differs for section {section}")


def _validate_hoptimus_encoder(encoder: object, source: object) -> Mapping[str, object]:
    manifest, manifest_sha256 = _manifest_contract(HOPTIMUS_MANIFEST_PATH, "bioptimus/H-optimus-1")
    source_encoder = source.source_receipt.get("encoder")
    if not isinstance(encoder, Mapping) or not isinstance(source_encoder, Mapping):
        raise ValueError("H-optimus encoder receipts are absent")
    parity = source_encoder.get("official_local_parity")
    expected = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "manifest_sha256": manifest_sha256,
        "architecture": manifest["architecture"],
        "checkpoint_filename": manifest["checkpoint_filename"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_filename": manifest["config_filename"],
        "config_sha256": manifest["config_sha256"],
        "feature_width": 1536,
        "input_pixels": 224,
        "model_mpp": manifest["model_mpp"],
        "device": "cuda",
        "fine_tuning": "none_frozen_eval_inference",
    }
    if (
        not isinstance(parity, Mapping)
        or source_encoder.get("repository") != manifest["repository"]
        or source_encoder.get("revision") != manifest["revision"]
        or source_encoder.get("manifest_sha256") != manifest_sha256
        or source_encoder.get("device") != "cuda"
        or source_encoder.get("stored_feature_dtype") != "float16"
        or any(encoder.get(key) != value for key, value in expected.items())
        or encoder.get("official_local_parity") != parity
    ):
        raise ValueError("H-optimus manifest, checkpoint, config, device, or parity changed")
    return encoder


def _validate_hoptimus_sections(
    receipt: Mapping[str, object],
    *,
    features: np.ndarray,
    source_contract: Mapping[str, object],
    source_sha256: str,
    expected_builder_sha256: str,
    encoder: Mapping[str, object],
) -> None:
    sections = receipt.get("sections")
    if (
        not isinstance(sections, list)
        or len(sections) != len(EXPECTED_SECTIONS)
        or [item.get("section") if isinstance(item, Mapping) else None for item in sections]
        != list(EXPECTED_SECTIONS)
    ):
        raise ValueError("55-um supplement must contain the exact 16 section receipts")
    source_sections = source_contract["section_receipts"]
    assert isinstance(source_sections, list)
    for section, item, source_item in zip(EXPECTED_SECTIONS, sections, source_sections):
        if not isinstance(item, Mapping) or not isinstance(source_item, Mapping):
            raise ValueError(f"55-um section receipt is malformed for {section}")
        rows = _section_rows(source_contract, section)
        _validate_common_section_hashes(
            item,
            section=section,
            rows=rows,
            source_contract=source_contract,
            expected_source_sha256=source_sha256,
        )
        frozen = item.get("frozen_v1_builder")
        source_embedding = source_item.get("embedding")
        if (
            item.get("schema") != CROP_SECTION_CACHE_SCHEMA
            or item.get("builder_implementation_sha256") != expected_builder_sha256
            or not isinstance(frozen, Mapping)
            or frozen.get("sha256") != FROZEN_V1_BUILDER_SHA256
            or item.get("stored_feature_dtype") != "float16"
            or item.get("encoder") != encoder
            or not isinstance(source_embedding, Mapping)
            or item.get("registration_qc") != source_embedding.get("registration_qc")
            or item.get("source_crop") != source_embedding.get("crop")
            or item.get("feature_stats")
            != _strict_feature_stats(features[rows], f"{section} 55-um features")
        ):
            raise ValueError(f"55-um section identity or feature stats differ for {section}")
        target_crop = item.get("target_crop")
        resampling = item.get("resampling")
        if (
            not isinstance(target_crop, Mapping)
            or target_crop.get("construction")
            != "white_outside_registered_center_square_on_native_112um_canvas"
            or target_crop.get("source_canvas_physical_width_um") != 112.0
            or target_crop.get("retained_center_physical_width_um") != 55.0
            or target_crop.get("centering_rule")
            != "independent_floor(center_minus_width_over_two)_registered_bounds"
            or target_crop.get("outside_value") != "white_RGB_uint8_255"
            or target_crop.get("separate_55um_resize") is not False
            or not isinstance(resampling, Mapping)
            or resampling.get("target_canvas_pixels") != [224, 224]
            or resampling.get("implementation") != "frozen_v1_Pillow.Image.Resampling.BICUBIC"
            or resampling.get("qualified_against_official_loader") is not True
        ):
            raise ValueError(f"55-um crop construction differs for section {section}")


def _load_crop_supplement(
    path: Path,
    expected_sha256: str,
    source: object,
    expected_builder_sha256: str,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Load the H-optimus crop arm only after complete receipt recomputation."""

    resolved = path.expanduser().resolve()
    expected_builder_sha256 = _validate_sha256(
        expected_builder_sha256, "55-um crop builder SHA-256"
    )
    if _sha256(CROP_BUILDER_PATH) != expected_builder_sha256:
        raise ValueError("current 55-um crop builder SHA-256 mismatch")
    if _sha256(resolved) != expected_sha256:
        raise ValueError("55-um crop supplement SHA-256 mismatch")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != {
                "schema_version",
                "spot_ids",
                "image_features_55um",
                "source_sha256",
                "receipt_json",
            }:
                raise ValueError("55-um crop supplement fields differ from its schema")
            schema = _scalar_archive_text(archive, "schema_version")
            spot_ids = np.asarray(archive["spot_ids"]).astype(str)
            raw_features = np.asarray(archive["image_features_55um"])
            source_sha = _scalar_archive_text(archive, "source_sha256")
            receipt = _json_scalar(archive, "receipt_json")
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("55-um crop supplement is malformed") from error
    source_contract = _source_row_contract(source)
    all_source_spots = np.asarray(source_contract["spot_ids"])
    receipt_source = receipt.get("source")
    construction = receipt.get("crop_construction")
    row_alignment = receipt.get("row_alignment")
    encoder = _validate_hoptimus_encoder(receipt.get("encoder"), source)
    expected_source_receipt = {
        "path": str(source_contract["path"]),
        "sha256": source_contract["sha256"],
        "schema": source_contract["schema"],
        "builder_implementation_sha256": FROZEN_V1_BUILDER_SHA256,
        "spot_count": len(all_source_spots),
        "spot_ids_sha256": _array_sha256(np.asarray(all_source_spots, dtype="S")),
        "native_112um_feature_stats": {
            "global": _strict_feature_stats(
                source_contract["native_features"], "source 112-um features"
            ),
            "per_section": {
                section: _strict_feature_stats(
                    np.asarray(source_contract["native_features"])[
                        _section_rows(source_contract, section)
                    ],
                    f"source {section} 112-um features",
                )
                for section in EXPECTED_SECTIONS
            },
        },
    }
    if (
        schema != CROP_SUPPLEMENT_SCHEMA
        or source_sha != source_contract["sha256"]
        or raw_features.dtype != np.float16
        or raw_features.shape != (len(spot_ids), 1536)
        or not np.array_equal(spot_ids, all_source_spots)
        or len(set(spot_ids.tolist())) != len(spot_ids)
        or receipt.get("schema") != CROP_SUPPLEMENT_RECEIPT_SCHEMA
        or receipt.get("builder_implementation_sha256") != expected_builder_sha256
        or receipt_source != expected_source_receipt
        or construction
        != {
            "source_canvas_physical_width_um": 112.0,
            "retained_center_physical_width_um": 55.0,
            "operation": (
                "extract_the_registered_112um_canvas_then_whiten_everything_outside_the_"
                "independently_registered_centered_55um_square"
            ),
            "white_value": "RGB_uint8_255",
            "resize_after_masking": (
                "same_single_frozen_v1_Pillow_bicubic_112um_canvas_to_224_pixels"
            ),
            "separate_55um_crop_resize_prohibited": True,
            "model_magnification_unchanged": True,
        }
        or row_alignment
        != {
            "output_spot_ids_exactly_equal_source": True,
            "all_source_rows_written_exactly_once": True,
        }
        or receipt.get("feature_stats")
        != _strict_feature_stats(raw_features, "assembled 55-um features")
    ):
        raise ValueError("55-um crop supplement identity or row alignment is invalid")
    _validate_hoptimus_sections(
        receipt,
        features=raw_features,
        source_contract=source_contract,
        source_sha256=source_sha,
        expected_builder_sha256=expected_builder_sha256,
        encoder=encoder,
    )
    lookup = {spot_id: index for index, spot_id in enumerate(spot_ids.tolist())}
    try:
        primary = raw_features[[lookup[spot_id] for spot_id in source.spot_ids.tolist()]].astype(
            np.float64
        )
    except KeyError as error:
        raise ValueError("55-um crop supplement omits a primary source spot") from error
    candidate = dataclasses.replace(source, image_features=primary)
    _feature_preflight(
        candidate,
        encoder_id=HOPTIMUS_ENCODER_ID,
        crop_id="target_55um",
    )
    return primary, receipt


def _validate_uni2_encoder(encoder: object) -> Mapping[str, object]:
    manifest, manifest_sha256 = _manifest_contract(UNI2_MANIFEST_PATH, "MahmoodLab/UNI2-h")
    if not isinstance(encoder, Mapping):
        raise ValueError("UNI2-h encoder receipt is absent")
    expected = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "architecture": manifest["architecture"],
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "feature_width": 1536,
        "input_pixels": 224,
        "model_mpp": manifest["model_mpp"],
        "normalization": manifest["normalization"],
        "interpolation": manifest["interpolation"],
        "pooling_rule": manifest["pooling_rule"],
        "license": manifest["license"],
        "known_training_datasets": manifest["known_training_datasets"],
        "evaluation_overlap": manifest["evaluation_overlap"],
        "device": "cuda",
        "fine_tuning": "none_frozen_eval_inference",
        "official_local_parity_claim": "none_not_assessed",
        "qualification_role": "manifest_hash_bound_secondary_sensitivity",
    }
    checkpoint_path = Path(str(encoder.get("checkpoint_path", "")))
    config_path = Path(str(encoder.get("config_path", "")))
    if (
        not checkpoint_path.is_file()
        or not config_path.is_file()
        or checkpoint_path.name != manifest["checkpoint_filename"]
        or config_path.name != manifest["config_filename"]
        or _sha256(checkpoint_path) != manifest["checkpoint_sha256"]
        or _sha256(config_path) != manifest["config_sha256"]
        or any(encoder.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("UNI2-h manifest, revision, checkpoint, config, or device changed")
    return encoder


def _validate_uni2_sections(
    receipt: Mapping[str, object],
    *,
    natural: np.ndarray,
    centered: np.ndarray,
    source_contract: Mapping[str, object],
    source_sha256: str,
    implementation: Mapping[str, str],
    encoder: Mapping[str, object],
) -> None:
    sections = receipt.get("sections")
    if (
        not isinstance(sections, list)
        or len(sections) != len(EXPECTED_SECTIONS)
        or [item.get("section") if isinstance(item, Mapping) else None for item in sections]
        != list(EXPECTED_SECTIONS)
    ):
        raise ValueError("UNI2-h supplement must contain the exact 16 section receipts")
    source_sections = source_contract["section_receipts"]
    assert isinstance(source_sections, list)
    expected_preprocessing = {
        "input_to_encoder": "native_registered_uint8_RGB_canvas",
        "explicit_pre_encoder_resize": False,
        "encoder_internal_resize": "torch_bilinear_align_corners_false_antialias_true",
        "natural_112um": "unmodified_registered_canvas",
        "centered_55um": "same_canvas_white_outside_registered_55um_square",
    }
    for section, item, source_item in zip(EXPECTED_SECTIONS, sections, source_sections):
        if not isinstance(item, Mapping) or not isinstance(source_item, Mapping):
            raise ValueError(f"UNI2-h section receipt is malformed for {section}")
        rows = _section_rows(source_contract, section)
        _validate_common_section_hashes(
            item,
            section=section,
            rows=rows,
            source_contract=source_contract,
            expected_source_sha256=source_sha256,
        )
        frozen = item.get("frozen_v1_builder")
        source_embedding = source_item.get("embedding")
        expected_stats = {
            "natural_112um": _strict_feature_stats(
                natural[rows], f"{section} UNI2-h 112-um features"
            ),
            "centered_55um": _strict_feature_stats(
                centered[rows], f"{section} UNI2-h 55-um features"
            ),
        }
        if (
            item.get("schema") != UNI2_SECTION_CACHE_SCHEMA
            or item.get("implementation") != implementation
            or not isinstance(frozen, Mapping)
            or frozen.get("sha256") != FROZEN_V1_BUILDER_SHA256
            or item.get("stored_feature_dtype") != "float16"
            or item.get("encoder") != encoder
            or item.get("preprocessing") != expected_preprocessing
            or item.get("feature_stats") != expected_stats
            or not isinstance(source_embedding, Mapping)
            or item.get("registration_qc") != source_embedding.get("registration_qc")
            or item.get("source_crop") != source_embedding.get("crop")
        ):
            raise ValueError(f"UNI2-h section identity or feature stats differ for {section}")
        target_crop = item.get("target_crop")
        if (
            not isinstance(target_crop, Mapping)
            or target_crop.get("construction")
            != "white_outside_registered_center_square_on_native_112um_canvas"
            or target_crop.get("source_canvas_physical_width_um") != 112.0
            or target_crop.get("retained_center_physical_width_um") != 55.0
            or target_crop.get("centering_rule") != "independent_floor_registered_bounds"
            or target_crop.get("outside_value") != "white_RGB_uint8_255"
        ):
            raise ValueError(f"UNI2-h crop construction differs for section {section}")


def _load_uni2_supplement(
    path: Path,
    expected_sha256: str,
    source: object,
    *,
    expected_builder_sha256: str,
    expected_adapter_sha256: str,
    expected_encoder_base_sha256: str,
    expected_encoder_factory_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Load both UNI2-h crop arms as a separately scored encoder sensitivity."""

    expected_implementation = {
        "builder_sha256": _validate_sha256(expected_builder_sha256, "UNI2-h builder SHA-256"),
        "uni2h_adapter_sha256": _validate_sha256(expected_adapter_sha256, "UNI2-h adapter SHA-256"),
        "encoder_base_sha256": _validate_sha256(
            expected_encoder_base_sha256, "encoder base SHA-256"
        ),
        "encoder_factory_sha256": _validate_sha256(
            expected_encoder_factory_sha256, "encoder factory SHA-256"
        ),
    }
    current = {
        "builder_sha256": _sha256(UNI2_BUILDER_PATH),
        "uni2h_adapter_sha256": _sha256(UNI2_ADAPTER_PATH),
        "encoder_base_sha256": _sha256(ENCODER_BASE_PATH),
        "encoder_factory_sha256": _sha256(ENCODER_FACTORY_PATH),
    }
    if current != expected_implementation:
        raise ValueError("current UNI2-h builder or encoder adapter implementation changed")
    resolved = path.expanduser().resolve()
    if _sha256(resolved) != expected_sha256:
        raise ValueError("UNI2-h supplement SHA-256 mismatch")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != {
                "schema_version",
                "spot_ids",
                "image_features_112um",
                "image_features_55um",
                "blank_image_feature_vector",
                "source_sha256",
                "receipt_json",
            }:
                raise ValueError("UNI2-h supplement fields differ from its schema")
            schema = _scalar_archive_text(archive, "schema_version")
            spot_ids = np.asarray(archive["spot_ids"]).astype(str)
            natural = np.asarray(archive["image_features_112um"])
            centered = np.asarray(archive["image_features_55um"])
            blank = np.asarray(archive["blank_image_feature_vector"])
            source_sha = _scalar_archive_text(archive, "source_sha256")
            receipt = _json_scalar(archive, "receipt_json")
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("UNI2-h supplement is malformed") from error
    source_contract = _source_row_contract(source)
    all_spots = np.asarray(source_contract["spot_ids"])
    encoder = _validate_uni2_encoder(receipt.get("encoder"))
    source_receipt = receipt.get("source")
    preprocessing = receipt.get("preprocessing")
    blank_receipt = receipt.get("blank_image_control")
    if blank.dtype != np.float16 or blank.shape != (1536,) or not np.isfinite(blank).all():
        raise ValueError("UNI2-h blank feature is not a finite float16 1536-vector")
    blank_squared_norm = float(np.square(blank.astype(np.float64)).sum())
    expected_blank_receipt = {
        "construction": "all_white_RGB_uint8_255_at_manifest_input_pixels",
        "applies_to": ["natural_112um", "centered_55um_whitened"],
        "semantic_reason": "an_all_white_canvas_is_identical_under_both_crop_constructions",
        "input_shape": [1, 224, 224, 3],
        "stored_dtype": "float16",
        "finite": True,
        "squared_norm": blank_squared_norm,
        "array_sha256": _array_sha256(blank),
    }
    expected_source_receipt = {
        "path": str(source_contract["path"]),
        "sha256": source_contract["sha256"],
        "schema": source_contract["schema"],
        "builder_implementation_sha256": FROZEN_V1_BUILDER_SHA256,
        "spot_count": len(all_spots),
        "spot_ids_sha256": _array_sha256(np.asarray(all_spots, dtype="S")),
    }
    expected_global_stats = {
        "natural_112um": _strict_feature_stats(natural, "assembled UNI2-h 112-um features"),
        "centered_55um": _strict_feature_stats(centered, "assembled UNI2-h 55-um features"),
    }
    if (
        schema != UNI2_SUPPLEMENT_SCHEMA
        or source_sha != source_contract["sha256"]
        or not np.array_equal(spot_ids, all_spots)
        or len(set(spot_ids.tolist())) != len(spot_ids)
        or natural.dtype != np.float16
        or centered.dtype != np.float16
        or natural.shape != (len(spot_ids), 1536)
        or centered.shape != natural.shape
        or receipt.get("schema") != UNI2_SUPPLEMENT_RECEIPT_SCHEMA
        or receipt.get("implementation") != expected_implementation
        or source_receipt != expected_source_receipt
        or preprocessing
        != {
            "natural_112um": "registered_native_canvas_passed_directly_to_encoder.encode",
            "centered_55um": (
                "same_native_canvas_white_outside_registered_center_55um_then_encoder.encode"
            ),
            "explicit_pre_encoder_resize": False,
            "only_resize": "UNI2HEncoder_manifest_bound_bilinear_interpolation",
            "official_local_parity_claim": "none_not_assessed",
        }
        or blank_receipt != expected_blank_receipt
        or receipt.get("feature_stats") != expected_global_stats
        or receipt.get("row_alignment")
        != {
            "output_spot_ids_exactly_equal_source": True,
            "all_source_rows_written_exactly_once": True,
        }
    ):
        raise ValueError("UNI2-h supplement identity or feature receipts are invalid")
    role = receipt.get("scientific_role")
    if (
        not isinstance(role, Mapping)
        or role.get("encoder") != "secondary_sensitivity_scored_separately_from_H_optimus_1"
        or "pooling_UNI2_h_with_H_optimus_1_primary_results" not in role.get("not_authorized", [])
        or "official_local_UNI2_h_parity_claim" not in role.get("not_authorized", [])
    ):
        raise ValueError("UNI2-h supplement does not preserve separate no-parity scoring")
    _validate_uni2_sections(
        receipt,
        natural=natural,
        centered=centered,
        source_contract=source_contract,
        source_sha256=source_sha,
        implementation=expected_implementation,
        encoder=encoder,
    )
    lookup = {spot_id: index for index, spot_id in enumerate(spot_ids.tolist())}
    try:
        primary_rows = [lookup[spot_id] for spot_id in source.spot_ids.tolist()]
    except KeyError as error:
        raise ValueError("UNI2-h supplement omits a primary source spot") from error
    primary_blank = blank.astype(np.float64)
    crop_sources = {
        "context_112um": dataclasses.replace(
            source,
            image_features=natural[primary_rows].astype(np.float64),
            blank_image_feature=primary_blank,
            blank_receipt=dict(expected_blank_receipt),
        ),
        "target_55um": dataclasses.replace(
            source,
            image_features=centered[primary_rows].astype(np.float64),
            blank_image_feature=primary_blank,
            blank_receipt=dict(expected_blank_receipt),
        ),
    }
    for crop_id, candidate in crop_sources.items():
        _feature_preflight(candidate, encoder_id=UNI2_ENCODER_ID, crop_id=crop_id)
    return crop_sources, receipt


def _load_hoptimus_inputs(args: argparse.Namespace, source: object) -> _EncoderInputs:
    target_features, receipt = _load_crop_supplement(
        args.crop_55_supplement,
        args.expected_crop_55_supplement_sha256,
        source,
        args.expected_crop_builder_sha256,
    )
    return _EncoderInputs(
        encoder_id=HOPTIMUS_ENCODER_ID,
        role="primary",
        repository="bioptimus/H-optimus-1",
        crop_sources={
            "target_55um": dataclasses.replace(source, image_features=target_features),
            "context_112um": source,
        },
        supplement_path=args.crop_55_supplement.expanduser().resolve(),
        supplement_sha256=args.expected_crop_55_supplement_sha256,
        supplement_receipt=receipt,
        implementation_files={
            CROP_BUILDER_PATH: args.expected_crop_builder_sha256,
        },
    )


def _load_uni2_inputs(args: argparse.Namespace, source: object) -> _EncoderInputs:
    crop_sources, receipt = _load_uni2_supplement(
        args.uni2_supplement,
        args.expected_uni2_supplement_sha256,
        source,
        expected_builder_sha256=args.expected_uni2_builder_sha256,
        expected_adapter_sha256=args.expected_uni2_adapter_sha256,
        expected_encoder_base_sha256=args.expected_encoder_base_sha256,
        expected_encoder_factory_sha256=args.expected_encoder_factory_sha256,
    )
    return _EncoderInputs(
        encoder_id=UNI2_ENCODER_ID,
        role="secondary_non_authorizing",
        repository="MahmoodLab/UNI2-h",
        crop_sources=crop_sources,
        supplement_path=args.uni2_supplement.expanduser().resolve(),
        supplement_sha256=args.expected_uni2_supplement_sha256,
        supplement_receipt=receipt,
        implementation_files={
            UNI2_BUILDER_PATH: args.expected_uni2_builder_sha256,
            UNI2_ADAPTER_PATH: args.expected_uni2_adapter_sha256,
            ENCODER_BASE_PATH: args.expected_encoder_base_sha256,
            ENCODER_FACTORY_PATH: args.expected_encoder_factory_sha256,
        },
    )


def _visible_reproducibility_view(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("visible control is not a mapping")
    folds = value.get("fold_receipts")
    if not isinstance(folds, Mapping) or set(folds) != set(PRIMARY_DONORS):
        raise ValueError("visible control lacks the exact donor folds")
    if any(not isinstance(folds[donor], Mapping) for donor in PRIMARY_DONORS):
        raise ValueError("visible control donor fold receipt is malformed")
    return {
        "per_donor": value.get("per_donor"),
        "sampled_rows": value.get("sampled_rows"),
        "sampled_spot_ids_sha256": value.get("sampled_spot_ids_sha256"),
        "maximum_rows_per_section": value.get("maximum_rows_per_section"),
        "sampling": value.get("sampling"),
        "folds": {
            donor: {
                "training_donors": folds[donor].get("training_donors"),
                "heldout_rows": folds[donor].get("heldout_rows"),
                "selected_ridge_alpha": folds[donor].get("selected_ridge_alpha"),
                "selection_weighting": folds[donor].get("selection_weighting"),
            }
            for donor in PRIMARY_DONORS
        },
    }


def _verify_preflight_report(
    args: argparse.Namespace,
    *,
    source: object,
    inputs: _EncoderInputs,
    legacy: ModuleType,
    protocol: Mapping[str, object],
    current_implementation: Mapping[str, object],
) -> Mapping[str, object]:
    report_path = args.preflight_report.expanduser().resolve()
    if _sha256(report_path) != args.expected_preflight_report_sha256:
        raise ValueError("NatCommun preflight report SHA-256 mismatch")
    report = _load_json(report_path, "NatCommun preflight report")
    if (
        report.get("schema") != PREFLIGHT_SCHEMA
        or report.get("source_sha256") != args.expected_source_sha256
        or report.get("protocol_sha256") != args.expected_protocol_sha256
        or report.get("registration_review_sha256") != args.expected_registration_review_sha256
        or report.get("encoder")
        != {
            "id": inputs.encoder_id,
            "role": inputs.role,
            "repository": inputs.repository,
            "primary_decision_authority": inputs.role == "primary",
            "may_rescue_primary": False,
        }
        or report.get("supplement")
        != {
            "path": str(inputs.supplement_path),
            "sha256": inputs.supplement_sha256,
            "receipt": inputs.supplement_receipt,
        }
    ):
        raise ValueError("NatCommun preflight report identities are inconsistent")
    implementation = report.get("implementation_receipt")
    implementation_files = (
        implementation.get("files") if isinstance(implementation, Mapping) else None
    )
    required_files = {
        str(Path(__file__).resolve()): args.expected_runner_sha256,
        str(PROTOCOL_PATH): args.expected_protocol_sha256,
        str(V1_RUNNER_PATH): args.expected_v1_runner_sha256,
        str(V1_BUILDER_PATH): args.expected_v1_builder_sha256,
        str(V1_PROTOCOL_PATH): args.expected_v1_protocol_sha256,
        str(REFERENCE_V2_PATH): args.expected_reference_v2_sha256,
        **{str(path.resolve()): sha for path, sha in inputs.implementation_files.items()},
    }
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("git_head") != current_implementation.get("git_head")
        or not isinstance(implementation_files, Mapping)
        or any(implementation_files.get(path) != sha for path, sha in required_files.items())
    ):
        raise ValueError("NatCommun preflight implementation identities are stale")
    stored_components = report.get("components")
    stored_gate = _component_gate(stored_components)
    if report.get("gate") != stored_gate or report.get("status") != "passed":
        raise ValueError("NatCommun preflight top-level result is inconsistent")
    if report.get("component_digest") != _canonical_sha256(stored_components):
        raise ValueError("NatCommun preflight component digest is inconsistent")

    maximum = int(protocol["regional_preflight"]["visible_control"]["maximum_spots_per_section"])
    current = _build_encoder_preflight_components(
        base_source=source,
        inputs=inputs,
        legacy=legacy,
        registration_review=args.registration_review,
        expected_registration_review_sha256=args.expected_registration_review_sha256,
        ridge_alphas=args.ridge_alphas,
        maximum_per_section=maximum,
        seed=args.seed,
        device=args.device,
    )
    current_gate = _component_gate(current)
    if not current_gate["passed"]:
        raise ValueError("NatCommun regional preflight did not pass")
    for name in ("registration", "controls", "reliability"):
        if _canonical_sha256(current["common"][name]) != _canonical_sha256(
            stored_components["common"][name]
        ):
            raise ValueError(f"current preflight {name} differs from the frozen report")
    benchmark_refit: dict[str, Mapping[str, object]] = {}
    for crop_id in CROP_IDS:
        stored_crop = stored_components["crop_arms"][crop_id]
        current_crop = current["crop_arms"][crop_id]
        if _canonical_sha256(current_crop["features"]) != _canonical_sha256(
            stored_crop["features"]
        ):
            raise ValueError(f"current preflight features differ for {crop_id}")
        stored_view = _visible_reproducibility_view(stored_crop["visible_control"])
        current_view = _visible_reproducibility_view(current_crop["visible_control"])
        if stored_view != current_view:
            raise ValueError(f"fresh visible-control refit differs for {crop_id}")
        benchmark_refit[crop_id] = current_crop["visible_control"]
    return {
        "report": str(report_path),
        "report_sha256": args.expected_preflight_report_sha256,
        "registration_review": str(args.registration_review.expanduser().resolve()),
        "registration_review_sha256": args.expected_registration_review_sha256,
        "encoder_id": inputs.encoder_id,
        "gate_recomputed": True,
        "benchmark_refit": benchmark_refit,
        "both_crop_arms_refit_and_passed": all(current_gate["crop_arm_passes"].values()),
        "passed": True,
    }


def _quality_matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a non-empty finite matrix")
    return matrix


def _fold_retained_variance_preservation(
    truth: object,
    prediction: object,
    donor_ids: Sequence[object],
    section_ids: Sequence[object],
    target_names: Sequence[object],
    retained_by_donor: Mapping[str, Sequence[str]],
) -> Mapping[str, object]:
    """Score variance only on each held-out donor's train-qualified programs."""

    target = _quality_matrix(truth, "program quality truth")
    predicted = _quality_matrix(prediction, "program quality prediction")
    if predicted.shape != target.shape:
        raise ValueError("program quality truth and prediction must align")
    donors = np.asarray(donor_ids).astype(str)
    sections = np.asarray(section_ids).astype(str)
    names = np.asarray(target_names).astype(str)
    if (
        donors.shape != (len(target),)
        or sections.shape != donors.shape
        or names.shape != (target.shape[1],)
        or len(set(names.tolist())) != len(names)
        or set(retained_by_donor) != set(PRIMARY_DONORS)
    ):
        raise ValueError("fold-retained variance inputs do not match the registered design")
    lookup = {name: index for index, name in enumerate(names.tolist())}
    ratios: list[float] = []
    per_donor: dict[str, Mapping[str, object]] = {}
    for donor in PRIMARY_DONORS:
        retained = tuple(str(value) for value in retained_by_donor[donor])
        if not retained or len(set(retained)) != len(retained) or not set(retained) <= set(lookup):
            raise ValueError(f"invalid retained-program set for {donor}")
        columns = np.asarray([lookup[name] for name in retained], dtype=np.int64)
        donor_ratios: list[float] = []
        donor_rows = donors == donor
        if not donor_rows.any():
            raise ValueError(f"program quality rows omit donor {donor}")
        for section in sorted(set(sections[donor_rows].tolist())):
            rows = donor_rows & (sections == section)
            if int(rows.sum()) < 3:
                continue
            local_truth = target[rows][:, columns]
            local_prediction = predicted[rows][:, columns]
            truth_variance = np.var(local_truth, axis=0)
            prediction_variance = np.var(local_prediction, axis=0)
            valid = truth_variance > 1.0e-10
            local_ratios = (prediction_variance[valid] / truth_variance[valid]).tolist()
            donor_ratios.extend(float(value) for value in local_ratios)
            ratios.extend(float(value) for value in local_ratios)
        per_donor[donor] = {
            "retained_programs": list(retained),
            "evaluated_section_program_pairs": len(donor_ratios),
            "median_within_section_variance_ratio": (
                float(np.median(donor_ratios)) if donor_ratios else None
            ),
        }
    return {
        "status": "evaluable" if ratios else "blocked_no_evaluable_section_program_pairs",
        "median_within_section_variance_ratio": float(np.median(ratios)) if ratios else None,
        "evaluated_section_target_pairs": len(ratios),
        "per_donor": per_donor,
        "selection_scope": "outer_training_reliability_qualified_programs_per_heldout_donor",
    }


def _fold_retained_rare_state_metrics(
    truth: object,
    prediction: object,
    thresholds: object,
    target_names: Sequence[object],
    donor_ids: Sequence[object],
    retained_by_donor: Mapping[str, Sequence[str]],
) -> Mapping[str, object]:
    """Compute rare-state recall on rows where the target passed the outer-train gate."""

    target = _quality_matrix(truth, "rare-state truth")
    predicted = _quality_matrix(prediction, "rare-state prediction")
    threshold = _quality_matrix(thresholds, "rare-state thresholds")
    donors = np.asarray(donor_ids).astype(str)
    names = np.asarray(target_names).astype(str)
    if (
        predicted.shape != target.shape
        or threshold.shape != target.shape
        or donors.shape != (len(target),)
        or names.shape != (target.shape[1],)
        or len(set(names.tolist())) != len(names)
        or set(retained_by_donor) != set(PRIMARY_DONORS)
    ):
        raise ValueError("fold-retained rare-state inputs do not match the registered design")
    retained_sets = {
        donor: {str(value) for value in retained_by_donor[donor]} for donor in PRIMARY_DONORS
    }
    known = set(names.tolist())
    if any(not values or not values <= known for values in retained_sets.values()):
        raise ValueError("fold-retained rare-state programs are empty or unknown")
    output: dict[str, Mapping[str, object]] = {}
    for index, name in enumerate(names.tolist()):
        eligible_donors = [donor for donor in PRIMARY_DONORS if name in retained_sets[donor]]
        if not eligible_donors:
            continue
        selected = np.isin(donors, eligible_donors)
        positive = target[selected, index] >= threshold[selected, index]
        called = predicted[selected, index] >= threshold[selected, index]
        true_positive = int(np.sum(positive & called))
        output[name] = {
            "truth_positive": int(positive.sum()),
            "predicted_positive": int(called.sum()),
            "recall": float(true_positive / positive.sum()) if positive.any() else None,
            "coverage_ratio": float(called.sum() / positive.sum()) if positive.any() else None,
            "eligible_donors": eligible_donors,
            "eligible_rows": int(selected.sum()),
        }
    if not output:
        raise ValueError("no program was retained in any outer fold")
    return output


class _V2ProgramQualityBindings:
    """Instrument the immutable v1 runner without retaining large prediction arrays."""

    def __init__(
        self,
        score_model: Callable[..., Mapping[str, object]],
        rare_state_metrics: Callable[..., Mapping[str, object]],
        program_reliability_gate: Callable[..., Mapping[str, object]],
    ) -> None:
        self._score_model = score_model
        self._rare_state_metrics = rare_state_metrics
        self._program_reliability_gate = program_reliability_gate
        self._active_endpoint: str | None = None
        self._folds: dict[str, Mapping[str, object]] = {}
        self._program_names: np.ndarray | None = None
        self._primary_donors: np.ndarray | None = None
        self._score_calls = 0
        self._rare_calls = 0
        self._variance: dict[str, Mapping[str, object]] = {}
        self._rare: dict[str, Mapping[str, object]] = {}

    def start(self, endpoint: str) -> None:
        if self._active_endpoint is not None:
            raise RuntimeError("program quality instrumentation was not finalized")
        self._active_endpoint = str(endpoint)
        self._folds = {}
        self._program_names = None
        self._primary_donors = None
        self._score_calls = 0
        self._rare_calls = 0
        self._variance = {}
        self._rare = {}

    def program_reliability_gate(
        self,
        half_a: np.ndarray,
        half_b: np.ndarray,
        donors: np.ndarray,
        training_donors: Sequence[str],
        names: np.ndarray,
    ) -> Mapping[str, object]:
        result = self._program_reliability_gate(
            half_a,
            half_b,
            donors,
            training_donors,
            names,
        )
        if self._active_endpoint != "program_total":
            return result
        all_donors = set(np.asarray(donors).astype(str).tolist())
        fit_donors = {str(value) for value in training_donors}
        heldout = sorted(all_donors - fit_donors)
        if all_donors != set(PRIMARY_DONORS) or len(heldout) != 1 or heldout[0] in self._folds:
            raise RuntimeError("program reliability hook observed an unexpected outer fold")
        local_names = np.asarray(names).astype(str)
        if local_names.ndim != 1 or len(set(local_names.tolist())) != len(local_names):
            raise RuntimeError("program reliability hook observed invalid target names")
        if self._program_names is None:
            self._program_names = local_names.copy()
        elif not np.array_equal(self._program_names, local_names):
            raise RuntimeError("program target names changed across outer folds")
        retained = result.get("retained_programs") if isinstance(result, Mapping) else None
        if (
            not isinstance(retained, Sequence)
            or isinstance(retained, (str, bytes))
            or not {str(value) for value in retained} <= set(local_names.tolist())
        ):
            raise RuntimeError("program reliability hook lacks valid retained programs")
        self._folds[heldout[0]] = {
            "status": result.get("status"),
            "fit_donors": sorted(fit_donors),
            "retained_programs": [str(value) for value in retained],
            "retained_program_count": len(retained),
        }
        return result

    def score_model(
        self,
        truth: np.ndarray,
        prediction: np.ndarray,
        donors: np.ndarray,
        sections: np.ndarray,
        types: np.ndarray,
    ) -> Mapping[str, object]:
        result = self._score_model(truth, prediction, donors, sections, types)
        if self._active_endpoint != "program_total":
            return result
        if self._score_calls >= len(PROGRAM_SCORE_CALL_MODELS):
            raise RuntimeError("frozen program score call order changed")
        model = PROGRAM_SCORE_CALL_MODELS[self._score_calls]
        self._score_calls += 1
        if model not in {"M0", "M3"}:
            return result
        if self._program_names is None or set(self._folds) != set(PRIMARY_DONORS):
            raise RuntimeError("program score hook ran before all outer reliability gates")
        if any(self._folds[donor].get("status") != "feasible" for donor in PRIMARY_DONORS):
            return result
        aligned_donors = np.asarray(donors).astype(str)
        if model == "M0":
            self._primary_donors = aligned_donors.copy()
        elif self._primary_donors is None or not np.array_equal(
            self._primary_donors, aligned_donors
        ):
            raise RuntimeError("M0 and M3 program score rows differ")
        retained = {donor: self._folds[donor]["retained_programs"] for donor in PRIMARY_DONORS}
        self._variance[model] = _fold_retained_variance_preservation(
            truth,
            prediction,
            aligned_donors,
            sections,
            self._program_names,
            retained,
        )
        return result

    def rare_state_metrics(
        self,
        truth: np.ndarray,
        prediction: np.ndarray,
        thresholds: np.ndarray,
        names: np.ndarray,
    ) -> Mapping[str, object]:
        result = self._rare_state_metrics(truth, prediction, thresholds, names)
        if self._active_endpoint != "program_total":
            return result
        if self._rare_calls >= len(PROGRAM_RARE_STATE_CALL_MODELS):
            raise RuntimeError("frozen rare-state call order changed")
        model = PROGRAM_RARE_STATE_CALL_MODELS[self._rare_calls]
        self._rare_calls += 1
        if model not in {"M0", "M3"}:
            return result
        if any(self._folds[donor].get("status") != "feasible" for donor in PRIMARY_DONORS):
            return result
        local_names = np.asarray(names).astype(str)
        if (
            self._program_names is None
            or self._primary_donors is None
            or not np.array_equal(local_names, self._program_names)
            or len(self._primary_donors) != len(np.asarray(truth))
        ):
            raise RuntimeError("rare-state hook is not aligned to primary program rows")
        retained = {donor: self._folds[donor]["retained_programs"] for donor in PRIMARY_DONORS}
        self._rare[model] = _fold_retained_rare_state_metrics(
            truth,
            prediction,
            thresholds,
            local_names,
            self._primary_donors,
            retained,
        )
        return result

    def finish(self, endpoint: str) -> Mapping[str, object] | None:
        if self._active_endpoint != endpoint:
            raise RuntimeError("program quality instrumentation endpoint changed")
        self._active_endpoint = None
        if endpoint != "program_total":
            return None
        if (
            set(self._folds) != set(PRIMARY_DONORS)
            or self._program_names is None
            or self._score_calls != len(PROGRAM_SCORE_CALL_MODELS)
            or self._rare_calls != len(PROGRAM_RARE_STATE_CALL_MODELS)
        ):
            raise RuntimeError("program quality instrumentation is incomplete")
        retained = {
            donor: list(self._folds[donor]["retained_programs"]) for donor in PRIMARY_DONORS
        }
        retained_union = [
            name
            for name in self._program_names.tolist()
            if any(name in retained[donor] for donor in PRIMARY_DONORS)
        ]
        instrumentation = {
            "frozen_score_call_order": list(PROGRAM_SCORE_CALL_MODELS),
            "frozen_rare_state_call_order": list(PROGRAM_RARE_STATE_CALL_MODELS),
            "large_prediction_matrices_retained_after_call": False,
        }
        if any(self._folds[donor].get("status") != "feasible" for donor in PRIMARY_DONORS):
            return {
                "schema": PROGRAM_QUALITY_SCHEMA,
                "status": "blocked_program_reliability_infeasible",
                "endpoint": "program_total",
                "selection_scope": (
                    "outer_training_reliability_qualified_programs_per_heldout_donor"
                ),
                "heldout_ST_used_for_program_selection": False,
                "target_names": retained_union,
                "folds": {donor: self._folds[donor] for donor in PRIMARY_DONORS},
                "variance_preservation": {},
                "rare_state_recall_coverage": {},
                "instrumentation": instrumentation,
            }
        if set(self._variance) != {"M0", "M3"} or set(self._rare) != {"M0", "M3"}:
            raise RuntimeError("program quality instrumentation omitted M0 or M3")
        if set(self._rare["M0"]) != set(retained_union) or set(self._rare["M3"]) != set(
            retained_union
        ):
            raise RuntimeError("rare-state metrics do not cover the retained-program union")
        return {
            "schema": PROGRAM_QUALITY_SCHEMA,
            "status": "complete",
            "endpoint": "program_total",
            "selection_scope": ("outer_training_reliability_qualified_programs_per_heldout_donor"),
            "heldout_ST_used_for_program_selection": False,
            "target_names": retained_union,
            "folds": {donor: self._folds[donor] for donor in PRIMARY_DONORS},
            "variance_preservation": self._variance,
            "rare_state_recall_coverage": self._rare,
            "instrumentation": instrumentation,
        }


@dataclasses.dataclass
class _CalibrationAdapter:
    calibrator: reference_fusion_v2.ReferenceCalibrator
    reference_donor_ids: np.ndarray
    indication_fallback_to_global: frozenset[str]
    selection_receipt: Mapping[str, object]

    @property
    def fit_donors(self) -> tuple[str, ...]:
        return self.calibrator.fit_donors

    def transform(self, values: object) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape[0] != len(self.reference_donor_ids):
            raise ValueError("calibration adapter requires the aligned full reference matrix")
        global_result = (
            matrix - self.calibrator.source_mean
        ) @ self.calibrator.coefficients + self.calibrator.target_mean
        if not self.calibrator.indication_labels:
            return global_result
        result = global_result.copy()
        row_indications = np.asarray(
            [DONOR_INDICATION[donor] for donor in self.reference_donor_ids]
        )
        assert self.calibrator.indication_slopes is not None
        assert self.calibrator.indication_source_means is not None
        assert self.calibrator.indication_target_means is not None
        for index, indication in enumerate(self.calibrator.indication_labels):
            if indication in self.indication_fallback_to_global:
                continue
            selected = row_indications == indication
            result[selected] = (
                matrix[selected] - self.calibrator.indication_source_means[index]
            ) * self.calibrator.indication_slopes[index] + self.calibrator.indication_target_means[
                index
            ]
        return result


class _V2MethodBindings:
    def __init__(self, source: object) -> None:
        self.source = source
        self.calibration_receipts: list[Mapping[str, object]] = []
        self._calibration_cache: dict[
            str,
            tuple[
                reference_fusion_v2.ReferenceCalibrator,
                Mapping[str, object],
            ],
        ] = {}

    def reset(self) -> None:
        self.calibration_receipts = []

    def fit_calibrator(
        self,
        reference_values: object,
        reference_donor_ids: Sequence[object],
        target_values: object,
        target_donor_ids: Sequence[object],
        fit_donor_ids: Sequence[object],
        *,
        ridge_alpha: float = 1.0,
    ) -> _CalibrationAdapter:
        del ridge_alpha
        reference_donors = np.asarray(reference_donor_ids).astype(str)
        target_donors = np.asarray(target_donor_ids).astype(str)
        fit_donors = tuple(sorted(set(str(value) for value in fit_donor_ids)))
        reference_matrix = np.asarray(reference_values, dtype=np.float64)
        target_matrix = np.asarray(target_values, dtype=np.float64)
        paired_fit_donors = tuple(
            donor
            for donor in fit_donors
            if np.any(reference_donors == donor) and np.any(target_donors == donor)
        )
        if paired_fit_donors != fit_donors:
            raise ValueError("calibration fit donors lack paired reference/ST summaries")
        summary_donors = np.asarray(paired_fit_donors)
        reference_summary = np.vstack(
            [
                reference_matrix[reference_donors == donor].mean(axis=0)
                for donor in paired_fit_donors
            ]
        )
        target_summary = np.vstack(
            [target_matrix[target_donors == donor].mean(axis=0) for donor in paired_fit_donors]
        )
        cache_key = _canonical_sha256(
            {
                "schema": "heir.natcommun_calibration_cache_key.v1",
                "fit_donors": list(fit_donors),
                "paired_fit_donors": list(paired_fit_donors),
                "reference_summary_sha256": _array_sha256(reference_summary),
                "target_summary_sha256": _array_sha256(target_summary),
                "candidate_alphas": list(DEFAULT_RIDGE_ALPHAS),
                "donor_indications": DONOR_INDICATION,
                "calibration_fit_weighting": (
                    "indication_equal_then_donor_equal_within_indication"
                ),
            }
        )
        cached = self._calibration_cache.get(cache_key)
        cache_hit = cached is not None
        if cached is None:
            selected_alpha, selection = reference_fusion_v2.select_reference_calibration_alpha(
                reference_summary,
                summary_donors,
                target_summary,
                summary_donors,
                fit_donors,
                candidate_alphas=DEFAULT_RIDGE_ALPHAS,
                donor_indications=DONOR_INDICATION,
            )
            calibrator = reference_fusion_v2.fit_reference_calibrator(
                reference_summary,
                summary_donors,
                target_summary,
                summary_donors,
                fit_donors,
                ridge_alpha=selected_alpha,
                donor_indications=DONOR_INDICATION,
            )
            self._calibration_cache[cache_key] = (calibrator, selection)
        else:
            calibrator, selection = cached
            selected_alpha = calibrator.ridge_alpha
        indication_counts = {
            indication: sum(DONOR_INDICATION[donor] == indication for donor in fit_donors)
            for indication in PRIMARY_DONORS_BY_INDICATION
        }
        fallback = frozenset(calibrator.fallback_indications)
        receipt = {
            "schema": "heir.natcommun_calibration_fit.v2",
            "fit_donors": list(fit_donors),
            "mode": "indication_diagonal_with_global_fallback_below_two_donors",
            "selected_ridge_alpha": selected_alpha,
            "selection": selection,
            "selection_mapping": "actual_hierarchical_indication_map_with_global_fallback",
            "calibration_fit_weighting": ("indication_equal_then_donor_equal_within_indication"),
            "paired_fit_donor_mean_summaries": list(paired_fit_donors),
            "non_fit_donor_outcomes_used": False,
            "calibration_cache_key": cache_key,
            "calibration_cache_hit": cache_hit,
            "indication_training_donor_counts": indication_counts,
            "global_fallback_indications": sorted(fallback),
            "qualified_indications": list(calibrator.qualified_indications),
            "full_affine_axis_mixing": False,
            "heldout_donor_outcomes_used": False,
        }
        self.calibration_receipts.append(receipt)
        return _CalibrationAdapter(
            calibrator,
            reference_donors,
            fallback,
            selection,
        )


def _adaptive_fusion_v2(
    image: np.ndarray,
    reference: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
    base_alpha: float,
    support_threshold: float,
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    distance = np.asarray(diagnostics["support_distance"], dtype=np.float64)
    coverage = np.asarray(diagnostics["type_coverage"], dtype=np.float64)
    uncertainty = np.asarray(diagnostics["reference_uncertainty"], dtype=np.float64)
    support = np.clip(
        1.0 - distance / max(float(support_threshold), 1.0e-12),
        0.0,
        1.0,
    )
    unsupported = (coverage <= 0.0) | ~np.isfinite(distance)
    support_weight = support * coverage * (1.0 - uncertainty)
    support_weight[unsupported] = 0.0
    prediction, adaptive = reference_fusion_v2.adaptive_residual_fusion(
        image,
        reference,
        support_weight,
        base_alpha,
    )
    abstained = (adaptive <= 1.0e-12) | unsupported
    return prediction, {
        **diagnostics,
        "support_score": support,
        "adaptive_alpha": adaptive,
        "abstained_fallback_to_H": abstained,
    }


def _indication_summary(
    experiment: Mapping[str, object],
    effect_thresholds: Mapping[str, object],
) -> Mapping[str, object]:
    minimum_positive_indications = int(effect_thresholds["minimum_positive_indications"])
    severe_reversal_threshold = float(effect_thresholds["severe_indication_reversal_relative_MSE"])
    headline = experiment.get("headline")
    per_donor = headline.get("per_donor") if isinstance(headline, Mapping) else None
    if not isinstance(per_donor, Mapping) or set(per_donor) != set(PRIMARY_DONORS):
        raise ValueError("experiment headline lacks exact per-donor losses")
    groups: dict[str, Mapping[str, object]] = {}
    donor_effects: dict[str, float] = {}
    for indication, donors in PRIMARY_DONORS_BY_INDICATION.items():
        m0 = np.asarray([float(per_donor[donor]["M0_loss"]) for donor in donors])
        m3 = np.asarray([float(per_donor[donor]["M3_loss"]) for donor in donors])
        effect = m0 - m3
        donor_effects.update({donor: float(value) for donor, value in zip(donors, effect)})
        mean_m0 = float(np.mean(m0))
        mean_m3 = float(np.mean(m3))
        relative = (mean_m0 - mean_m3) / mean_m0 if mean_m0 > 0 else None
        standard_error = (
            float(np.std(effect, ddof=1) / np.sqrt(len(effect))) if len(effect) > 1 else None
        )
        groups[indication] = {
            "donors": list(donors),
            "M0_donor_equal_loss": mean_m0,
            "M3_donor_equal_loss": mean_m3,
            "mean_effect_M0_minus_M3": float(np.mean(effect)),
            "relative_MSE_gain_M3_vs_M0": relative,
            "standard_error_of_mean_effect": standard_error,
            "positive_direction": bool(np.mean(effect) > 0.0),
            "severe_reversal": bool(relative is not None and relative <= severe_reversal_threshold),
        }
    indication_equal_m0 = float(np.mean([row["M0_donor_equal_loss"] for row in groups.values()]))
    indication_equal_m3 = float(np.mean([row["M3_donor_equal_loss"] for row in groups.values()]))
    weights = {}
    for indication, row in groups.items():
        standard_error = row["standard_error_of_mean_effect"]
        weights[indication] = (
            1.0 / max(float(standard_error) ** 2, 1.0e-12) if standard_error is not None else 0.0
        )
    total_weight = sum(weights.values())
    meta_effect = (
        float(
            sum(weights[name] * groups[name]["mean_effect_M0_minus_M3"] for name in groups)
            / total_weight
        )
        if total_weight > 0
        else None
    )
    positive_indications = sum(row["positive_direction"] for row in groups.values())
    no_severe_reversal = not any(row["severe_reversal"] for row in groups.values())
    return {
        "per_indication": groups,
        "donor_effect_M0_minus_M3": donor_effects,
        "donor_equal_M0_loss": float(
            np.mean([float(per_donor[donor]["M0_loss"]) for donor in PRIMARY_DONORS])
        ),
        "donor_equal_M3_loss": float(
            np.mean([float(per_donor[donor]["M3_loss"]) for donor in PRIMARY_DONORS])
        ),
        "indication_equal_M0_loss": indication_equal_m0,
        "indication_equal_M3_loss": indication_equal_m3,
        "positive_indication_count": positive_indications,
        "minimum_positive_indications": minimum_positive_indications,
        "severe_reversal_relative_MSE_threshold": severe_reversal_threshold,
        "no_severe_reversal": no_severe_reversal,
        "heterogeneity_passed": bool(
            positive_indications >= minimum_positive_indications and no_severe_reversal
        ),
        "fixed_effect_inverse_variance_meta_effect_M0_minus_M3": meta_effect,
        "meta_method_limit": "three_indication_descriptive_fixed_effect_summary",
    }


def _primary_experiment_names(crop_id: str) -> tuple[str, ...]:
    return tuple(
        f"{crop_id}::state_kmeans_8::{endpoint}::{bank}"
        for endpoint in ("program_total", "pca_total")
        for bank in ("natural", "composition_equalized")
    )


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _threshold_check(
    *,
    name: str,
    observed: object,
    threshold: float,
    operator: str,
) -> Mapping[str, object]:
    value = _finite_number(observed)
    if value is None:
        return {
            "name": name,
            "status": "blocked_missing_or_nonfinite_metric",
            "observed": None,
            "threshold": threshold,
            "operator": operator,
            "evaluable": False,
            "passed": False,
            "reason": f"{name}:missing_or_nonfinite",
        }
    passed = (
        value + THRESHOLD_COMPARISON_TOLERANCE >= threshold
        if operator == ">="
        else value <= threshold + THRESHOLD_COMPARISON_TOLERANCE
    )
    failure = "below_minimum" if operator == ">=" else "above_maximum"
    return {
        "name": name,
        "status": "passed" if passed else "failed_threshold",
        "observed": value,
        "threshold": threshold,
        "operator": operator,
        "evaluable": True,
        "passed": bool(passed),
        "reason": f"{name}:passed" if passed else f"{name}:{failure}",
    }


def _qualified_program_quality_payload(
    experiment: Mapping[str, object],
) -> Mapping[str, object] | None:
    payload = experiment.get("v2_reliability_qualified_program_quality")
    if not isinstance(payload, Mapping):
        return None
    target_names = payload.get("target_names")
    folds = payload.get("folds")
    variance = payload.get("variance_preservation")
    rare = payload.get("rare_state_recall_coverage")
    if (
        payload.get("schema") != PROGRAM_QUALITY_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("endpoint") != "program_total"
        or payload.get("selection_scope")
        != "outer_training_reliability_qualified_programs_per_heldout_donor"
        or payload.get("heldout_ST_used_for_program_selection") is not False
        or not isinstance(target_names, Sequence)
        or isinstance(target_names, (str, bytes))
        or not target_names
        or len({str(value) for value in target_names}) != len(target_names)
        or not isinstance(folds, Mapping)
        or set(folds) != set(PRIMARY_DONORS)
        or not isinstance(variance, Mapping)
        or set(variance) != {"M0", "M3"}
        or not all(isinstance(variance[model], Mapping) for model in ("M0", "M3"))
        or not isinstance(rare, Mapping)
        or set(rare) != {"M0", "M3"}
        or not all(isinstance(rare[model], Mapping) for model in ("M0", "M3"))
    ):
        return None
    declared = {str(value) for value in target_names}
    if set(rare["M0"]) != declared or set(rare["M3"]) != declared:
        return None
    for donor in PRIMARY_DONORS:
        fold = folds[donor]
        if (
            not isinstance(fold, Mapping)
            or fold.get("status") != "feasible"
            or fold.get("fit_donors") != sorted(value for value in PRIMARY_DONORS if value != donor)
            or not isinstance(fold.get("retained_programs"), list)
            or not fold["retained_programs"]
            or not set(str(value) for value in fold["retained_programs"]) <= declared
        ):
            return None
    return payload


def _rare_state_guardrail(
    experiment: Mapping[str, object],
    *,
    median_drop_maximum: float,
    single_target_drop_maximum: float,
) -> Mapping[str, object]:
    endpoint = experiment.get("endpoint")
    if endpoint == "pca_total":
        return {
            "status": "not_applicable_to_pca_endpoint",
            "evaluable": True,
            "passed": True,
            "reason": "rare_state_recall_is_defined_only_for_program_total",
            "M8_used_to_block": False,
        }
    if endpoint != "program_total":
        return {
            "status": "blocked_unknown_endpoint",
            "evaluable": False,
            "passed": False,
            "reason": "rare_state_collapse:missing_or_unknown_endpoint",
            "M8_used_to_block": False,
        }

    program_quality = _qualified_program_quality_payload(experiment)
    metrics = (
        program_quality.get("rare_state_recall_coverage")
        if isinstance(program_quality, Mapping)
        else None
    )
    m0 = metrics.get("M0") if isinstance(metrics, Mapping) else None
    m3 = metrics.get("M3") if isinstance(metrics, Mapping) else None
    if not isinstance(m0, Mapping) or not isinstance(m3, Mapping):
        return {
            "status": "blocked_missing_reliability_qualified_rare_state_metrics",
            "evaluable": False,
            "passed": False,
            "reason": "rare_state_collapse:qualified_M0_or_M3_metrics_missing",
            "M8_used_to_block": False,
        }

    m0_targets = {str(value) for value in m0}
    m3_targets = {str(value) for value in m3}
    target_names = program_quality.get("target_names")
    if (
        not isinstance(target_names, Sequence)
        or isinstance(target_names, (str, bytes))
        or len(target_names) == 0
    ):
        return {
            "status": "blocked_missing_target_universe",
            "evaluable": False,
            "passed": False,
            "reason": "rare_state_collapse:target_names_missing",
            "M8_used_to_block": False,
        }
    declared_targets = [str(value) for value in target_names]
    declared_set = set(declared_targets)
    if (
        len(declared_set) != len(declared_targets)
        or not declared_set
        or m0_targets != declared_set
        or m3_targets != declared_set
    ):
        return {
            "status": "blocked_target_universe_mismatch",
            "evaluable": False,
            "passed": False,
            "reason": "rare_state_collapse:M0_M3_or_declared_targets_mismatch",
            "declared_targets": sorted(declared_set),
            "M0_targets": sorted(m0_targets),
            "M3_targets": sorted(m3_targets),
            "M8_used_to_block": False,
        }

    drops: dict[str, float] = {}
    baseline_recalls: dict[str, float] = {}
    no_truth_positive: list[str] = []
    invalid_targets: list[str] = []
    for target in sorted(declared_set):
        row_m0 = m0.get(target)
        row_m3 = m3.get(target)
        if not isinstance(row_m0, Mapping) or not isinstance(row_m3, Mapping):
            invalid_targets.append(target)
            continue
        truth_positive_m0 = row_m0.get("truth_positive")
        truth_positive_m3 = row_m3.get("truth_positive")
        if (
            isinstance(truth_positive_m0, (bool, np.bool_))
            or isinstance(truth_positive_m3, (bool, np.bool_))
            or not isinstance(truth_positive_m0, (int, np.integer))
            or not isinstance(truth_positive_m3, (int, np.integer))
            or int(truth_positive_m0) < 0
            or int(truth_positive_m0) != int(truth_positive_m3)
        ):
            invalid_targets.append(target)
            continue
        recall_m0 = _finite_number(row_m0.get("recall"))
        recall_m3 = _finite_number(row_m3.get("recall"))
        if int(truth_positive_m0) == 0:
            if row_m0.get("recall") is not None or row_m3.get("recall") is not None:
                invalid_targets.append(target)
            else:
                no_truth_positive.append(target)
            continue
        if (
            recall_m0 is None
            or recall_m3 is None
            or not 0.0 <= recall_m0 <= 1.0
            or not 0.0 <= recall_m3 <= 1.0
        ):
            invalid_targets.append(target)
            continue
        baseline_recalls[target] = recall_m0
        drops[target] = recall_m0 - recall_m3

    if invalid_targets or not drops:
        return {
            "status": "blocked_invalid_or_unevaluable_rare_state_metrics",
            "evaluable": False,
            "passed": False,
            "reason": (
                "rare_state_collapse:invalid_target_metrics"
                if invalid_targets
                else "rare_state_collapse:no_targets_with_truth_positives"
            ),
            "invalid_targets": invalid_targets,
            "targets_without_truth_positives": no_truth_positive,
            "evaluated_target_count": len(drops),
            "M8_used_to_block": False,
        }

    median_drop = float(np.median(np.asarray(list(drops.values()), dtype=np.float64)))
    eligible = {
        target: drops[target]
        for target in sorted(drops)
        if baseline_recalls[target] >= RARE_STATE_BASELINE_RECALL_MINIMUM
    }
    violating_targets = {
        target: drop
        for target, drop in eligible.items()
        if drop > single_target_drop_maximum + THRESHOLD_COMPARISON_TOLERANCE
    }
    median_passed = median_drop <= median_drop_maximum + THRESHOLD_COMPARISON_TOLERANCE
    individual_passed = not violating_targets
    passed = bool(median_passed and individual_passed)
    reasons = []
    if not median_passed:
        reasons.append("median_recall_drop_above_maximum")
    if not individual_passed:
        reasons.append("eligible_target_recall_drop_above_maximum")
    return {
        "status": "passed" if passed else "failed_threshold",
        "evaluable": True,
        "passed": passed,
        "reason": (
            "rare_state_collapse:passed" if passed else "rare_state_collapse:" + ";".join(reasons)
        ),
        "median_M0_minus_M3_recall_drop": median_drop,
        "maximum_median_recall_drop": median_drop_maximum,
        "median_drop_passed": bool(median_passed),
        "baseline_recall_minimum_for_single_target_check": (RARE_STATE_BASELINE_RECALL_MINIMUM),
        "maximum_single_target_recall_drop": single_target_drop_maximum,
        "single_target_drop_passed": bool(individual_passed),
        "target_M0_recall": baseline_recalls,
        "target_M0_minus_M3_recall_drop": drops,
        "eligible_target_drops": eligible,
        "violating_target_drops": violating_targets,
        "targets_without_truth_positives": no_truth_positive,
        "evaluated_target_count": len(drops),
        "M8_used_to_block": False,
    }


def _scientific_quality_guardrails(
    experiment: Mapping[str, object],
    effect_thresholds: Mapping[str, object],
) -> Mapping[str, object]:
    threshold_keys = {
        "minimum_M3_median_within_section_variance_ratio": ">=",
        "minimum_M3_median_type_coverage": ">=",
        "maximum_M3_abstention_fraction": "<=",
        "rare_state_maximum_median_recall_drop_from_M0": "<=",
        "rare_state_maximum_single_target_recall_drop_when_M0_at_least_0_2": "<=",
    }
    thresholds = {name: _finite_number(effect_thresholds.get(name)) for name in threshold_keys}
    if any(value is None for value in thresholds.values()):
        raise ValueError("scientific quality guardrail thresholds are incomplete")

    endpoint = experiment.get("endpoint")
    program_quality = (
        _qualified_program_quality_payload(experiment) if endpoint == "program_total" else None
    )
    if endpoint == "program_total":
        qualified_variance = (
            program_quality.get("variance_preservation")
            if isinstance(program_quality, Mapping)
            else None
        )
        variance = qualified_variance.get("M3") if isinstance(qualified_variance, Mapping) else None
        variance_source = "outer_training_reliability_qualified_programs_per_heldout_donor"
    else:
        scores = experiment.get("scores")
        m3_score = scores.get("M3") if isinstance(scores, Mapping) else None
        variance = m3_score.get("variance_preservation") if isinstance(m3_score, Mapping) else None
        variance_source = "all_outer_training_PCA_axes"
    coverage = experiment.get("coverage_uncertainty_abstention")
    m3_coverage = coverage.get("M3") if isinstance(coverage, Mapping) else None
    checks = {
        "M3_within_section_variance_preservation": _threshold_check(
            name="M3_median_within_section_variance_ratio",
            observed=(
                variance.get("median_within_section_variance_ratio")
                if isinstance(variance, Mapping)
                else None
            ),
            threshold=float(thresholds["minimum_M3_median_within_section_variance_ratio"]),
            operator=">=",
        ),
        "M3_median_type_coverage": _threshold_check(
            name="M3_median_type_coverage",
            observed=(
                m3_coverage.get("median_type_coverage")
                if isinstance(m3_coverage, Mapping)
                else None
            ),
            threshold=float(thresholds["minimum_M3_median_type_coverage"]),
            operator=">=",
        ),
        "M3_abstention_fraction": _threshold_check(
            name="M3_abstention_fraction",
            observed=(
                m3_coverage.get("abstention_fraction") if isinstance(m3_coverage, Mapping) else None
            ),
            threshold=float(thresholds["maximum_M3_abstention_fraction"]),
            operator="<=",
        ),
    }
    rare_state = _rare_state_guardrail(
        experiment,
        median_drop_maximum=float(thresholds["rare_state_maximum_median_recall_drop_from_M0"]),
        single_target_drop_maximum=float(
            thresholds["rare_state_maximum_single_target_recall_drop_when_M0_at_least_0_2"]
        ),
    )
    all_rows = [*checks.values(), rare_state]
    evaluable = all(bool(row["evaluable"]) for row in all_rows)
    passed = bool(evaluable and all(bool(row["passed"]) for row in all_rows))
    failure_reasons = [str(row["reason"]) for row in all_rows if not row["passed"]]
    return {
        "status": (
            "passed"
            if passed
            else ("failed_threshold" if evaluable else "blocked_missing_or_invalid_metrics")
        ),
        "evaluable": evaluable,
        "passed": passed,
        "checks": checks,
        "rare_state_collapse": rare_state,
        "variance_metric_source": variance_source,
        "program_quality_payload_valid": (
            program_quality is not None if endpoint == "program_total" else None
        ),
        "failure_reasons": failure_reasons,
        "M8_used_to_block": False,
    }


def _crop_decision(
    crop_id: str,
    experiments: Mapping[str, Mapping[str, object]],
    adjusted: Mapping[str, float],
    effect_thresholds: Mapping[str, object],
    familywise_alpha: float,
) -> Mapping[str, object]:
    minimum_relative_gain = float(effect_thresholds["minimum_relative_MSE_gain_M3_vs_M0"])
    minimum_positive_donors = float(effect_thresholds["minimum_positive_donor_fraction_M3_vs_M0"])
    rows: dict[str, Mapping[str, object]] = {}
    blocked: list[str] = []
    effect_rows: dict[str, Mapping[str, object]] = {}
    quality_rows: dict[str, Mapping[str, object]] = {}
    for experiment_name in _primary_experiment_names(crop_id):
        experiment = experiments.get(experiment_name)
        if (
            not isinstance(experiment, Mapping)
            or experiment.get("headline", {}).get("status") != "evaluable"
        ):
            blocked.append(experiment_name)
            continue
        paired = experiment.get("paired_inference")
        if not isinstance(paired, Mapping):
            blocked.append(experiment_name)
            continue
        for comparison in ALL_REGISTERED_COMPARISONS:
            key = f"{experiment_name}::{comparison}"
            summary = paired.get(comparison)
            if not isinstance(summary, Mapping) or key not in adjusted:
                blocked.append(key)
                continue
            effect = float(summary["mean_effect"])
            p_value = float(adjusted[key])
            rows[key] = {
                "mean_effect_control_minus_M3": effect,
                "global_holm_adjusted_exact_sign_flip_p": p_value,
                "familywise_alpha": familywise_alpha,
                "passed": bool(effect > 0.0 and p_value <= familywise_alpha),
            }
        headline = experiment["headline"]
        indication = experiment["indication_balance"]
        effect_rows[experiment_name] = {
            "relative_MSE_gain": headline["relative_MSE_gain_M3_vs_M0"],
            "positive_donor_fraction": headline["positive_donor_fraction_M3_vs_M0"],
            "indication_heterogeneity": indication,
            "passed": bool(
                headline["relative_MSE_gain_M3_vs_M0"] is not None
                and float(headline["relative_MSE_gain_M3_vs_M0"]) >= minimum_relative_gain
                and float(headline["positive_donor_fraction_M3_vs_M0"]) >= minimum_positive_donors
                and indication["heterogeneity_passed"] is True
            ),
            "minimum_relative_MSE_gain": minimum_relative_gain,
            "minimum_positive_donor_fraction": minimum_positive_donors,
        }
        quality = _scientific_quality_guardrails(experiment, effect_thresholds)
        quality_rows[experiment_name] = quality
        if not quality["evaluable"]:
            blocked.append(f"{experiment_name}::scientific_quality_guardrails")
    evaluable = not blocked and len(effect_rows) == 4 and len(quality_rows) == 4 and len(rows) == 32
    supported = bool(
        evaluable
        and all(row["passed"] for row in rows.values())
        and all(row["passed"] for row in effect_rows.values())
        and all(row["passed"] for row in quality_rows.values())
    )
    return {
        "crop_id": crop_id,
        "status": "evaluable" if evaluable else "blocked_fail_closed",
        "decision": (
            "supported"
            if supported
            else ("not_supported" if evaluable else "blocked_indeterminate")
        ),
        "supported": supported,
        "registered_control_comparisons": rows,
        "effect_size_donor_and_indication_checks": effect_rows,
        "scientific_quality_guardrails": quality_rows,
        "blocked_or_missing": blocked,
        "M8_used_to_block": False,
    }


def _secondary_floor_by_crop(
    legacy: ModuleType,
    crop_id: str,
    experiments: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    subset = {
        "::".join(name.split("::")[-2:]): experiments[name]
        for name in _primary_experiment_names(crop_id)
        if name in experiments
    }
    result = legacy._measurement_floor_decision(subset)
    return {
        **result,
        "role": "secondary_nonblocking_gap_closed_denominator_diagnostic",
        "used_to_block_regional_support": False,
    }


def _registered_experiment_specs() -> tuple[tuple[str, str, str, int, str], ...]:
    rows: list[tuple[str, str, str, int, str]] = []
    for crop_id in CROP_IDS:
        for endpoint in ("program_total", "pca_total"):
            for bank in ("natural", "composition_equalized"):
                rows.append((crop_id, endpoint, bank, 8, "primary"))
        rows.append((crop_id, "program_total", "natural", 1, "secondary_diagnostic"))
    return tuple(rows)


def _registered_p_value_family(
    experiments: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, float], tuple[str, ...]]:
    """Materialize the exact 64-test family, conservatively filling absent tests."""

    raw: dict[str, float] = {}
    missing: list[str] = []
    for crop_id in CROP_IDS:
        for experiment_name in _primary_experiment_names(crop_id):
            experiment = experiments.get(experiment_name)
            paired = experiment.get("paired_inference") if isinstance(experiment, Mapping) else None
            if paired is not None and not isinstance(paired, Mapping):
                raise ValueError(f"paired inference is malformed: {experiment_name}")
            if isinstance(paired, Mapping):
                extra = sorted(set(map(str, paired)) - set(ALL_REGISTERED_COMPARISONS))
                if extra:
                    raise ValueError(
                        f"unregistered paired comparisons in {experiment_name}: {extra}"
                    )
            for comparison in ALL_REGISTERED_COMPARISONS:
                key = f"{experiment_name}::{comparison}"
                summary = paired.get(comparison) if isinstance(paired, Mapping) else None
                if summary is None:
                    raw[key] = 1.0
                    missing.append(key)
                    continue
                if not isinstance(summary, Mapping):
                    raise ValueError(f"paired comparison is malformed: {key}")
                p_value = _finite_number(summary.get("exact_sign_flip_p"))
                if p_value is None or not 0.0 <= p_value <= 1.0:
                    raise ValueError(f"paired comparison p-value is invalid: {key}")
                raw[key] = p_value
    expected_size = len(CROP_IDS) * 2 * 2 * len(ALL_REGISTERED_COMPARISONS)
    if len(raw) != expected_size or len(raw) != 64:
        raise RuntimeError("registered global Holm family is not exactly 64 tests")
    return raw, tuple(missing)


def _experiment_checkpoint_path(output_dir: Path, key: str) -> Path:
    safe = key.replace("::", "__")
    if not safe or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in safe
    ):
        raise ValueError("experiment key is unsafe for a checkpoint filename")
    return output_dir / "checkpoints" / f"{safe}.json"


def _load_experiment_checkpoint(
    path: Path,
    identity: Mapping[str, object],
) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    value = _load_json(path, "experiment checkpoint")
    result = value.get("result")
    if (
        value.get("schema") != "heir.natcommun_experiment_checkpoint.v1"
        or value.get("identity") != identity
        or value.get("identity_sha256") != _canonical_sha256(identity)
        or not isinstance(result, Mapping)
        or value.get("result_sha256") != _canonical_sha256(result)
    ):
        raise ValueError(f"experiment checkpoint is stale or corrupted: {path}")
    return dict(result)


def _write_experiment_checkpoint(
    path: Path,
    identity: Mapping[str, object],
    result: Mapping[str, object],
) -> None:
    _atomic_json(
        path,
        {
            "schema": "heir.natcommun_experiment_checkpoint.v1",
            "identity": identity,
            "identity_sha256": _canonical_sha256(identity),
            "result": result,
            "result_sha256": _canonical_sha256(result),
        },
    )


def _markdown(report: Mapping[str, object]) -> str:
    encoder = report["encoder"]
    lines = [
        f"# NatCommun regional matched-reference validation v2: {encoder['id']}",
        "",
        (
            "Scope: retrospective Visium-spot regional evidence. This report cannot "
            "authorize cell-level HEIR claims."
        ),
        "",
        "## Decision",
        "",
        f"- Encoder-specific decision: `{report['encoder_decision']['decision']}`.",
        f"- Encoder role: `{encoder['role']}`; results are not pooled across encoders.",
        "- The frozen HEST centered-nucleus geometry result is non-gating and unchanged.",
        "- M8 is a secondary approximate measurement-floor diagnostic and is nonblocking.",
        "",
        "## Crop-arm decisions",
        "",
        "| Crop | Decision | Supported |",
        "|---|---|---:|",
    ]
    for crop_id, decision in report["crop_decisions"].items():
        lines.append(
            f"| {crop_id} | {decision['decision']} | {str(decision['supported']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## State-aware primary experiments",
            "",
            "| Experiment | M0 loss | M3 loss | Relative gain | Positive donors |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, experiment in report["experiments"].items():
        if "::state_kmeans_8::" not in name:
            continue
        headline = experiment["headline"]
        gain = headline["relative_MSE_gain_M3_vs_M0"]
        lines.append(
            "| {name} | {m0:.4g} | {m3:.4g} | {gain} | {positive:.1%} |".format(
                name=name,
                m0=headline["M0_loss"],
                m3=headline["M3_loss"],
                gain="NA" if gain is None else f"{gain:.1%}",
                positive=headline["positive_donor_fraction_M3_vs_M0"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "- A passing primary H-optimus result can support only a scalable regional "
                "research implementation."
            ),
            (
                "- A UNI2-h result is secondary and cannot authorize, rescue, or override "
                "the H-optimus primary decision."
            ),
            "- Independent replication remains required.",
            (
                "- Cell-level molecular-state and annotation hypotheses remain blocked "
                "until a matched cell-resolved cohort is tested."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _run_encoder_benchmark(
    args: argparse.Namespace,
    inputs_loader: object,
) -> int:
    protocol = _load_protocol(args.expected_protocol_sha256)
    legacy = _load_v1_runner(args.expected_v1_runner_sha256)
    numeric_backend = legacy._configure_numeric_backend(args.cpu_threads, args.device)
    source = legacy.load_source(
        args.source,
        expected_primary_donors=len(PRIMARY_DONORS),
        expected_source_sha256=args.expected_source_sha256,
    )
    source_contract = _validate_source_v2_contract(source)
    inputs = inputs_loader(args, source)
    implementation = _implementation_receipt(
        args,
        protocol,
        extra_expected_files=inputs.implementation_files,
    )
    prerequisite = _verify_preflight_report(
        args,
        source=source,
        inputs=inputs,
        legacy=legacy,
        protocol=protocol,
        current_implementation=implementation,
    )
    crop_sources = inputs.crop_sources
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bindings = _V2MethodBindings(source)
    quality_bindings = _V2ProgramQualityBindings(
        legacy._score_model,
        legacy._rare_state_metrics,
        legacy._program_reliability_gate,
    )
    legacy.build_reference_prototypes = reference_fusion_v2.build_reference_prototypes
    legacy.fit_reference_calibrator = bindings.fit_calibrator
    legacy._adaptive_fusion = _adaptive_fusion_v2
    legacy._select_ridge_alpha = partial(_select_ridge_alpha_indication_equal, legacy)
    legacy._select_fusion_parameters = partial(
        _select_fusion_parameters_indication_equal,
        legacy,
    )
    legacy._score_model = quality_bindings.score_model
    legacy._rare_state_metrics = quality_bindings.rare_state_metrics
    legacy._program_reliability_gate = quality_bindings.program_reliability_gate

    experiments: dict[str, Mapping[str, object]] = {}
    checkpoint_receipts: dict[str, Mapping[str, object]] = {}
    for crop_id, endpoint, bank, prototypes, role in _registered_experiment_specs():
        crop_source = crop_sources[crop_id]
        bindings.source = crop_source
        representation = "state_kmeans_8" if prototypes == 8 else "type_centroid_1"
        key = f"{crop_id}::{representation}::{endpoint}::{bank}"
        checkpoint_path = _experiment_checkpoint_path(args.output_dir, key)
        identity = {
            "schema": "heir.natcommun_experiment_identity.v1",
            "encoder_id": inputs.encoder_id,
            "supplement_sha256": inputs.supplement_sha256,
            "source_sha256": args.expected_source_sha256,
            "protocol_sha256": args.expected_protocol_sha256,
            "runner_sha256": args.expected_runner_sha256,
            "reference_v2_sha256": args.expected_reference_v2_sha256,
            "preflight_report_sha256": args.expected_preflight_report_sha256,
            "registration_review_sha256": args.expected_registration_review_sha256,
            "git_head": implementation["git_head"],
            "numeric_backend_sha256": _canonical_sha256(numeric_backend),
            "cpu_threads": args.cpu_threads,
            "experiment": key,
            "ridge_alphas": list(args.ridge_alphas),
            "fusion_alphas": list(args.fusion_alphas),
            "temperatures": list(args.temperatures),
            "pca_components": args.pca_components,
            "pca_genes": args.pca_genes,
            "prototypes_per_type": prototypes,
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "device": args.device,
        }
        result = _load_experiment_checkpoint(checkpoint_path, identity)
        checkpoint_status = "reused"
        if result is None:
            print(f"NatCommun v2: {inputs.encoder_id}: {key}", flush=True)
            bindings.reset()
            quality_bindings.start(endpoint)
            result = dict(
                legacy.run_endpoint(
                    crop_source,
                    endpoint,
                    bank,
                    ridge_alphas=args.ridge_alphas,
                    fusion_alphas=args.fusion_alphas,
                    temperatures=args.temperatures,
                    pca_components=args.pca_components,
                    pca_genes=args.pca_genes,
                    prototypes_per_type=prototypes,
                    bootstrap_iterations=args.bootstrap_iterations,
                    seed=args.seed,
                    device=args.device,
                )
            )
            program_quality = quality_bindings.finish(endpoint)
            if program_quality is not None:
                result["v2_reliability_qualified_program_quality"] = program_quality
            result["crop_id"] = crop_id
            result["reference_representation"] = representation
            result["inference_role"] = role
            result["v2_calibration_receipts"] = bindings.calibration_receipts
            result["indication_balance"] = _indication_summary(
                result, protocol["effect_thresholds"]
            )
            _write_experiment_checkpoint(checkpoint_path, identity, result)
            checkpoint_status = "created"
        experiments[key] = result
        checkpoint_receipts[key] = {
            "status": checkpoint_status,
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "identity_sha256": _canonical_sha256(identity),
        }

    raw_p, missing_registered_tests = _registered_p_value_family(experiments)
    adjusted = legacy.holm_adjust(raw_p)
    crop_decisions = {
        crop_id: _crop_decision(
            crop_id,
            experiments,
            adjusted,
            protocol["effect_thresholds"],
            float(protocol["familywise_alpha"]),
        )
        for crop_id in CROP_IDS
    }
    supported_crops = [
        crop_id for crop_id, decision in crop_decisions.items() if decision["supported"]
    ]
    regional_supported = bool(supported_crops)
    regional_blocked = bool(
        not regional_supported
        and any(decision["status"] != "evaluable" for decision in crop_decisions.values())
    )
    encoder_decision = {
        "decision": (
            "supported"
            if regional_supported
            else ("blocked_indeterminate" if regional_blocked else "not_supported")
        ),
        "supported": regional_supported,
        "supported_crop_arms": supported_crops,
        "interpretation": (
            "regional_target_and_context_signal"
            if len(supported_crops) == 2
            else (
                "regional_context_only"
                if supported_crops == ["context_112um"]
                else (
                    "target_matched_signal_only"
                    if supported_crops == ["target_55um"]
                    else "no_registered_crop_arm_supported"
                )
            )
        ),
        "M8_used_to_block": False,
    }
    regional_decision = (
        encoder_decision
        if inputs.role == "primary"
        else {
            "decision": "not_assessed_by_secondary_encoder",
            "supported": False,
            "supported_crop_arms": [],
            "interpretation": "UNI2_h_is_a_secondary_non_authorizing_signal_only",
            "M8_used_to_block": False,
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete_retrospective_regional_research_only",
        "analysis_scope": protocol["analysis_scope"],
        "source": str(source.path),
        "source_sha256": args.expected_source_sha256,
        "source_contract": source_contract,
        "encoder": {
            "id": inputs.encoder_id,
            "role": inputs.role,
            "repository": inputs.repository,
            "results_pooled_across_encoders": False,
            "primary_decision_authority": inputs.role == "primary",
            "may_rescue_primary": False,
        },
        "protocol_sha256": args.expected_protocol_sha256,
        "implementation_receipt": implementation,
        "numeric_backend": numeric_backend,
        "prerequisites": {"encoder_preflight": prerequisite},
        "non_gating_diagnostics": {"hest_hoptimus1": protocol["hest_architecture_diagnostic"]},
        "design": {
            "encoder_repository": inputs.repository,
            "encoder_role": inputs.role,
            "supplement": str(inputs.supplement_path),
            "supplement_sha256": inputs.supplement_sha256,
            "supplement_receipt": inputs.supplement_receipt,
            "reference_primary": "deterministic_molecular_kmeans_8_per_donor_type",
            "reference_secondary": "one_centroid_per_donor_type",
            "reference_latent_input": "fold_training_donor_ST_calibrated_snRNA_latent",
            "reference_uses_fold_training_donor_ST_via_calibration": True,
            "reference_uses_heldout_or_inner_validation_donor_ST_outcomes": False,
            "reference_clustering_maximum_iterations": (
                reference_fusion_v2.MOLECULAR_KMEANS_MAXIMUM_ITERATIONS
            ),
            "reference_clustering_convergence": ("exact_consecutive_repaired_assignment_equality"),
            "reference_assignment_cycle_behavior": "nonconsecutive_repeat_fails_closed",
            "registered_primary_experiments": 8,
            "registered_centroid_diagnostics": 2,
            "calibration": "indication_aware_diagonal_training_only",
            "fusion_alpha_grid": list(args.fusion_alphas),
            "iteration": "prohibited",
            "M8_role": "secondary_nonblocking",
        },
        "experiments": experiments,
        "experiment_checkpoints": checkpoint_receipts,
        "multiplicity": {
            "scope": "within_encoder_only_across_8_primary_experiments_and_8_comparisons",
            "encoders_pooled": False,
            "method": "Holm",
            "familywise_alpha": protocol["familywise_alpha"],
            "registered_family_size": 64,
            "observed_test_count": 64 - len(missing_registered_tests),
            "missing_tests_conservatively_entered_as_p_1": list(missing_registered_tests),
        },
        "global_holm_adjusted_exact_sign_flip_p": adjusted,
        "crop_decisions": crop_decisions,
        "secondary_measurement_floor": {
            crop_id: _secondary_floor_by_crop(legacy, crop_id, experiments) for crop_id in CROP_IDS
        },
        "encoder_decision": encoder_decision,
        "secondary_encoder_signal": (encoder_decision if inputs.role != "primary" else None),
        "regional_decision": regional_decision,
        "authorization": {
            "regional_research_implementation_justified": bool(
                inputs.role == "primary" and regional_supported
            ),
            "secondary_signal_supported": (
                regional_supported if inputs.role != "primary" else None
            ),
            "primary_decision_authority": inputs.role == "primary",
            "may_rescue_primary": False,
            "primary_result_overridden": False,
            "independent_replication_required": True,
            "cell_level_HEIR_development_authorized": False,
            "production_or_clinical_use_authorized": False,
        },
    }
    _atomic_json(args.output_dir / "report.json", report)
    _atomic_text(args.output_dir / "report.md", _markdown(report))
    print(args.output_dir / "report.json", flush=True)
    return 0


def _run_hoptimus_benchmark(args: argparse.Namespace) -> int:
    return _run_encoder_benchmark(args, _load_hoptimus_inputs)


def _run_uni2_benchmark(args: argparse.Namespace) -> int:
    return _run_encoder_benchmark(args, _load_uni2_inputs)


def _float_grid(
    value: str,
    *,
    positive: bool,
    upper: float | None = None,
) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid must contain comma-separated numbers") from error
    if not values or any(not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("grid must contain finite numbers")
    if positive and any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("grid values must be positive")
    if not positive and any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("grid values cannot be negative")
    if upper is not None and any(item > upper for item in values):
        raise argparse.ArgumentTypeError(f"grid values cannot exceed {upper:g}")
    return values


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--registration-review", type=Path, required=True)
    parser.add_argument("--expected-registration-review-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument(
        "--expected-v1-protocol-sha256",
        default=FROZEN_V1_PROTOCOL_SHA256,
    )
    parser.add_argument(
        "--expected-v1-builder-sha256",
        default=FROZEN_V1_BUILDER_SHA256,
    )
    parser.add_argument(
        "--expected-v1-runner-sha256",
        default=FROZEN_V1_RUNNER_SHA256,
    )
    parser.add_argument("--expected-reference-v2-sha256", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument(
        "--ridge-alphas",
        type=lambda value: _float_grid(value, positive=True),
        default=DEFAULT_RIDGE_ALPHAS,
    )


def _add_hoptimus_supplement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--crop-55-supplement", type=Path, required=True)
    parser.add_argument("--expected-crop-55-supplement-sha256", required=True)
    parser.add_argument("--expected-crop-builder-sha256", required=True)


def _add_uni2_supplement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uni2-supplement", type=Path, required=True)
    parser.add_argument("--expected-uni2-supplement-sha256", required=True)
    parser.add_argument("--expected-uni2-builder-sha256", required=True)
    parser.add_argument("--expected-uni2-adapter-sha256", required=True)
    parser.add_argument("--expected-encoder-base-sha256", required=True)
    parser.add_argument("--expected-encoder-factory-sha256", required=True)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--expected-preflight-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fusion-alphas",
        type=lambda value: _float_grid(value, positive=False, upper=1.0),
        default=DEFAULT_FUSION_ALPHAS,
    )
    parser.add_argument(
        "--temperatures",
        type=lambda value: _float_grid(value, positive=True),
        default=DEFAULT_TEMPERATURES,
    )
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--pca-genes", type=int, default=256)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    hoptimus_preflight = subparsers.add_parser(
        "preflight-hoptimus",
        help="qualify both registered H-optimus crop arms",
    )
    _add_identity_arguments(hoptimus_preflight)
    _add_hoptimus_supplement_arguments(hoptimus_preflight)
    hoptimus_preflight.add_argument("--output", type=Path, required=True)
    hoptimus_preflight.set_defaults(handler=_run_hoptimus_preflight)

    uni2_preflight = subparsers.add_parser(
        "preflight-uni2",
        help="qualify both registered UNI2-h crop arms as a secondary sensitivity",
    )
    _add_identity_arguments(uni2_preflight)
    _add_uni2_supplement_arguments(uni2_preflight)
    uni2_preflight.add_argument("--output", type=Path, required=True)
    uni2_preflight.set_defaults(handler=_run_uni2_preflight)

    hoptimus_benchmark = subparsers.add_parser(
        "benchmark-hoptimus",
        help="run the registered H-optimus regional M0-M8 validation",
    )
    _add_identity_arguments(hoptimus_benchmark)
    _add_hoptimus_supplement_arguments(hoptimus_benchmark)
    _add_benchmark_arguments(hoptimus_benchmark)
    hoptimus_benchmark.set_defaults(handler=_run_hoptimus_benchmark)

    uni2_benchmark = subparsers.add_parser(
        "benchmark-uni2",
        help="run the separately scored UNI2-h regional M0-M8 sensitivity",
    )
    _add_identity_arguments(uni2_benchmark)
    _add_uni2_supplement_arguments(uni2_benchmark)
    _add_benchmark_arguments(uni2_benchmark)
    uni2_benchmark.set_defaults(handler=_run_uni2_benchmark)

    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    args.command = [sys.executable, str(Path(__file__).resolve()), *raw]
    for name in (
        "expected_source_sha256",
        "expected_registration_review_sha256",
        "expected_protocol_sha256",
        "expected_runner_sha256",
        "expected_v1_protocol_sha256",
        "expected_v1_builder_sha256",
        "expected_v1_runner_sha256",
        "expected_reference_v2_sha256",
    ):
        try:
            _validate_sha256(getattr(args, name), name.replace("_", " "))
        except ValueError as error:
            parser.error(str(error))
    family_names = (
        (
            "expected_crop_55_supplement_sha256",
            "expected_crop_builder_sha256",
        )
        if args.command_name.endswith("hoptimus")
        else (
            "expected_uni2_supplement_sha256",
            "expected_uni2_builder_sha256",
            "expected_uni2_adapter_sha256",
            "expected_encoder_base_sha256",
            "expected_encoder_factory_sha256",
        )
    )
    for name in family_names:
        try:
            _validate_sha256(getattr(args, name), name.replace("_", " "))
        except ValueError as error:
            parser.error(str(error))
    if args.command_name.startswith("benchmark-"):
        for name in ("expected_preflight_report_sha256",):
            try:
                _validate_sha256(getattr(args, name), name.replace("_", " "))
            except ValueError as error:
                parser.error(str(error))
        if tuple(args.fusion_alphas) != DEFAULT_FUSION_ALPHAS:
            parser.error("confirmatory benchmark requires fusion alphas 0,0.1,0.25,0.5,0.75,1")
        if tuple(args.temperatures) != DEFAULT_TEMPERATURES:
            parser.error("confirmatory benchmark requires temperatures 0.25,0.5,1,2,4")
        if args.pca_components != 20 or args.pca_genes != 256:
            parser.error("confirmatory execution requires 20 PCA axes and 256 genes")
        if args.bootstrap_iterations != 2000:
            parser.error("confirmatory execution requires 2000 bootstrap iterations")
    if not 1 <= args.cpu_threads <= 8:
        parser.error("--cpu-threads must lie in [1, 8]")
    if tuple(args.ridge_alphas) != DEFAULT_RIDGE_ALPHAS:
        parser.error("confirmatory execution requires ridge alphas 0.01,0.1,1,10,100")
    if args.seed != 17:
        parser.error("confirmatory execution requires seed 17")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
