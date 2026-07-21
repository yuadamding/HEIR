#!/usr/bin/env python3
"""Leakage-separated matched-ST diagnostic for the frozen NatCommun model.

This is a validation runner, not a model-development runner.  It keeps the
frozen H-optimus-1 representation, gene panel, latent dimension, count model,
training schedule, and one-step PoE fusion rule.  Only the prediction-time
reference modality changes from suspension cells to an adjacent Visium
section.

The real experiment intentionally has four separate stages:

``prepare``
    Trusted partitioning of each donor's two ST sections into a public
    reference/query-H&E bundle and a score-only query-ST target.
``fit-model``
    Fit one donor-excluded frozen model per donor without opening either
    held-out donor ST section.
``predict --direction ...``
    In a fresh process, open exactly one reference section and query H&E.  A
    process is prohibited from predicting more than one reciprocal direction.
``score``
    Validate every prediction artifact in the selected score family before
    opening its score-only target manifest or any query ST target.

The B1 and L1 sections are same-block serial sections on distinct Visium
slides.  They are not spot-paired replicates, independent confirmation, or a
measurement floor; this runner can only produce a distributional adjacent-
section mechanistic diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "4"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# isort: off
import numpy as np  # noqa: E402
import torch  # noqa: E402
# isort: on


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/benchmark_natcommun_generative_development.py"
CORE_PATH = ROOT / "src/heir/evaluation/generative_fusion.py"
DEFAULT_PROTOCOL = ROOT / "configs/natcommun_matched_st_validation.json"
DEFAULT_BASELINE = Path("/mnt/seagate/HEIR_runs/natcommun_generative_development")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_matched_st_validation")

PROTOCOL_SCHEMA = "heir.natcommun_matched_st_validation_protocol.v1"
PREPARED_SCHEMA = "heir.natcommun_matched_st_prepared.v1"
SCORE_TARGET_MANIFEST_SCHEMA = "heir.natcommun_matched_st_score_targets.v1"
MODEL_SCHEMA = "heir.natcommun_matched_st_model.v1"
PREDICTION_SCHEMA = "heir.natcommun_matched_st_predictions.v1"
REPORT_SCHEMA = "heir.natcommun_matched_st_report.v1"
ARMS = ("S0", "S1", "S3", "S4", "S6", "S7", "M1", "M3")
BASELINE_ARMS = ("M0", "M1", "M3")
EXPECTED_DONORS = (
    "B1",
    "B3",
    "B4",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "L1",
    "L2",
    "L3",
    "L4",
)
EXPECTED_PRIMARY_SECTIONS = (
    "B1_2",
    "B1_4",
    "B3_2",
    "B4_2",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "L1_2",
    "L1_4",
    "L2_2",
    "L3_2",
    "L4_2",
)
EXPECTED_ADJACENT_DIRECTIONS = (
    "B1_2__to__B1_4",
    "B1_4__to__B1_2",
    "L1_2__to__L1_4",
    "L1_4__to__L1_2",
)
EXPECTED_DONOR_INPUT_IDENTITIES = {
    "B1": (
        2611344574,
        "0d9fba94a62052470b3841b0debd9b632096725ea08441ae663d998ffda937e5",
        "2e884a9cca7896182166ee31db8e7321ab0547bf35dd77c5acd5f55e40e3879a",
    ),
    "B3": (
        3823175560,
        "40ddf78b42d33bf7f9845038558f98d1aba841dbc7f72c3d855687f32e812fe6",
        "b13f91723f4fd262e6132d488a56331005034854a3b2e026238c09aebf58ef55",
    ),
    "B4": (
        4221163290,
        "05d23fae8059615a5c07b07438aa714d86cdcda630ab63d67435f5d45f1ddd38",
        "5c0f67050953f61e5e534f1389082a4a59e5343595d55fde1c5900254ad984fc",
    ),
    "D1": (
        2339048254,
        "ac81181e99b1666c643af5fbbf792addb174e69f10678625763d7d91784b54e2",
        "f0406787c57364c1f2ae2f99060ddefdad8eec168c0137293fa6f38d602ffd49",
    ),
    "D2": (
        3362693652,
        "aa44556c7cf78b3e72f9d3dd0550b77eb7a7292d56eb0f59822c2b480e68833b",
        "e061a0f5114c7b099914c7a1956a9e92d9142d7865948e9aa4696328911a0028",
    ),
    "D3": (
        1021591616,
        "3b4c003f0418323ba35e33809b76b1b99a1aabe4b00a8409a428491939094920",
        "812be831ae381c9db537b9a3adffe9503a8253896400c9473e0fb76716d9e4d5",
    ),
    "D4": (
        3851824045,
        "ff0e3337c39d46b975178058461cf7b0c1a18617432121892f6e5cfa659ad48b",
        "1320ad33f3802b57cbb336cb9668edea339857d9b0bd3965bdb33b22978e480e",
    ),
    "D5": (
        2037645439,
        "24192645d160b4c6ff0425bcab94d2d6b7fce5cc00cf22a0cd7ae6231e133118",
        "74f6e893d0f2467d2f5f81bb0fd7a0f3365876edd2b69a4643f81a98c3f1ddfb",
    ),
    "D6": (
        1708564394,
        "30844f98f7e9217ef765621a6df62fbe6cacc3b375beec697539296adaee2821",
        "fef4c0740c3782afd98bc99e734944aa78a0d351455b9553b8b41da28532d718",
    ),
    "L1": (
        3043417614,
        "a35d77766a9d90156500f52f8b4ffcafc36e90ae42707995c212e970e95fcd97",
        "8bfd329cd5fb3962fb6b3d9893ea33b3cef4cc9d0afbbe9b22e33035b2f91e22",
    ),
    "L2": (
        4155144987,
        "a83e7cea84aeb7198402ecb885a019b50d9e951124cb029e245214afe52b40b9",
        "75a5fc698e4bbf4774e2c0e73315a6f83419da9a8fec8d06ca1537161159b3d6",
    ),
    "L3": (
        1747313781,
        "c180827543eea262bfe79e850ddc431220ce1b2850adc6a2584f084cbfdbfa57",
        "ad8532faf3d180b98ccd8102c893747649051ee946da8eca5fe8e8c128149a64",
    ),
    "L4": (
        1504764697,
        "cfb15538c5d0b9da9f36a42e1dd236b03f3fd928865a3d89be5b3eda2d9fe8f4",
        "25fa8c1204ca72a4d239ac156ea2dbc10330e8447500f5e2bead7adaa7821020",
    ),
}
FORBIDDEN_PUBLIC_PREFIXES = (
    "query_st",
    "heldout_st",
    "score_target",
    "target_counts",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def _require_canonical_path(value: object, expected: Path, label: str) -> Path:
    text = str(value)
    observed = Path(text)
    canonical = _canonical_path(observed)
    expected_canonical = _canonical_path(expected)
    if observed != canonical or canonical != expected_canonical:
        raise ValueError(f"{label} is not the canonical expected path")
    return canonical


def _manifest_semantic_identity(payload: Mapping[str, object]) -> str:
    semantic = {key: value for key, value in payload.items() if key != "prepared_identity"}
    return hashlib.sha256(_json_bytes(semantic)).hexdigest()


def _score_target_manifest_semantic_identity(payload: Mapping[str, object]) -> str:
    semantic = {
        key: value
        for key, value in payload.items()
        if key != "score_target_manifest_identity"
    }
    return hashlib.sha256(_json_bytes(semantic)).hexdigest()


def _mapping_key_contains(value: object, fragment: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            fragment.casefold() in str(key).casefold()
            or _mapping_key_contains(item, fragment)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_mapping_key_contains(item, fragment) for item in value)
    return False


def _upper_direction_id(section: str, guard_mm: float) -> str:
    guard_label = f"{float(guard_mm):g}".replace(".", "p")
    return f"upper__{section}__guard_{guard_label}mm"


def _expected_upper_directions() -> tuple[str, ...]:
    return tuple(
        _upper_direction_id(section, guard)
        for section in EXPECTED_PRIMARY_SECTIONS
        for guard in (1.0, 2.0)
    )


def _section_donor_indication(section: str) -> tuple[str, str]:
    if section.startswith("B"):
        return section.split("_", 1)[0], "breast"
    if section.startswith("D"):
        return section, "dlbcl"
    if section.startswith("L"):
        return section.split("_", 1)[0], "lung"
    raise ValueError(f"unknown frozen primary section: {section}")


def _assert_zero_process_swap(status_path: Path = Path("/proc/self/status")) -> int:
    text = status_path.read_text(encoding="utf-8")
    value = next(
        (line.split()[1] for line in text.splitlines() if line.startswith("VmSwap:")),
        None,
    )
    if value is None or int(value) != 0:
        raise RuntimeError("matched-ST stage requires zero process swap")
    return 0


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(value.dtype.str.encode())
        digest.update(_json_bytes(list(value.shape)))
        if value.dtype.kind in {"U", "O"}:
            if value.dtype.kind == "O":
                raise TypeError("object arrays are prohibited")
            digest.update(_json_bytes(value.tolist()))
        else:
            digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_torch(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar string")
    item = array.reshape(-1)[0]
    return item.decode() if isinstance(item, bytes) else str(item)


def _safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "heir_frozen_natcommun_v2_matched_st", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen NatCommun runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_protocol(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("matched-ST validation protocol is missing or malformed")
    freeze = payload.get("model_freeze", {})
    adapter = payload.get("spatial_reference_adapter", {})
    boundary = payload.get("target_boundary", {})
    evidence = payload.get("evidence_boundary", {})
    resources = payload.get("resource_limits", {})
    if not all(
        isinstance(value, Mapping)
        for value in (freeze, adapter, boundary, evidence, resources)
    ):
        raise ValueError("matched-ST protocol contracts are incomplete")
    expected_freeze = {
        "image_encoder": "bioptimus/H-optimus-1",
        "image_encoder_revision": "3592cb220dec7a150c5d7813fb56e68bd57473b9",
        "gene_count": 256,
        "latent_dimension": 20,
        "epochs": 80,
        "batch_size": 256,
        "base_seed": 1729,
        "iterative_refinement": "prohibited",
        "UNI2_h": "prohibited_not_run",
    }
    mismatch = [name for name, value in expected_freeze.items() if freeze.get(name) != value]
    if mismatch:
        raise ValueError(f"matched-ST model freeze changed: {mismatch}")
    expected_adapter = {
        "type_proxy_iterations": 80,
        "components_per_supported_type": 3,
        "component_iterations": 25,
        "component_temperature": 1.0,
        "component_variance_floor": 0.0001,
        "minimum_type_proxy_mass": 3.0,
        "minimum_type_proxy_effective_sample_size": 3.0,
        "adapter_tuning_on_query_ST": False,
    }
    mismatch = [name for name, value in expected_adapter.items() if adapter.get(name) != value]
    if mismatch:
        raise ValueError(f"matched-ST reference adapter changed: {mismatch}")
    if (
        boundary.get("prepare_is_outcome_exposed_trusted_partitioning") is not True
        or boundary.get("fit_model_receives_no_heldout_donor_ST") is not True
        or boundary.get("fit_predict_manifest_contains_score_target_paths_or_hashes")
        is not False
        or boundary.get("score_only_target_manifest") is not True
        or boundary.get("score_target_manifest_open_stage")
        != "after_selected_score_family_global_prediction_preflight"
        or boundary.get("one_process_may_predict_multiple_reciprocal_directions") is not False
        or boundary.get("query_ST_open_stage") != "score_only"
        or boundary.get(
            "all_selected_score_family_prediction_artifacts_validated_before_any_"
            "selected_query_ST_is_scored"
        )
        is not True
    ):
        raise ValueError("matched-ST target separation is not fail-closed")
    if any(
        evidence.get(name) is not False
        for name in (
            "spot_registration_available",
            "paired_technical_replicate_claim",
            "measurement_floor_claim",
            "independent_confirmation_claim",
            "cell_level_claim",
        )
    ):
        raise ValueError("matched-ST protocol overstates the evidence boundary")
    if (
        int(resources.get("maximum_CPU_threads", -1)) > 4
        or float(resources.get("maximum_GPU_memory_fraction", math.inf)) > 0.60
        or resources.get("swap_permitted") is not False
    ):
        raise ValueError("matched-ST resource limits exceed the frozen bounds")
    directions = payload.get("directional_pairs", ())
    if not isinstance(directions, Sequence) or len(directions) != 4:
        raise ValueError("matched-ST protocol must contain four reciprocal directions")
    expected = {
        ("B1", "B1_2", "B1_4"),
        ("B1", "B1_4", "B1_2"),
        ("L1", "L1_2", "L1_4"),
        ("L1", "L1_4", "L1_2"),
    }
    observed = {
        (str(item["donor"]), str(item["reference_section"]), str(item["query_section"]))
        for item in directions
        if isinstance(item, Mapping)
    }
    if observed != expected:
        raise ValueError("matched-ST directions differ from the registered repeated sections")
    upper = payload.get("same_section_upper_bound", {})
    if (
        not isinstance(upper, Mapping)
        or upper.get("status") != "secondary_optimistic_sensitivity"
        or upper.get("split_input") != "registered_Visium_array_grid_coordinates_only"
        or upper.get("empty_guard_total_width_mm") != [1.0, 2.0]
        or int(upper.get("minimum_reference_spots", -1)) != 100
        or int(upper.get("minimum_query_spots", -1)) != 100
        or upper.get("reference_and_query_spots_disjoint") is not True
        or upper.get("measurement_floor_claim") is not False
        or upper.get("independent_confirmation_claim") is not False
    ):
        raise ValueError("same-section upper-bound sensitivity contract changed")
    return payload


def _direction_map(protocol: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result = {}
    for raw in protocol["directional_pairs"]:
        if not isinstance(raw, Mapping):
            raise ValueError("direction contract is malformed")
        direction_id = str(raw["direction_id"])
        if not direction_id or direction_id in result:
            raise ValueError("direction identifiers must be nonempty and unique")
        result[direction_id] = raw
    return result


def physical_array_coordinates_mm(array_row_col: np.ndarray) -> np.ndarray:
    """Map the registered Visium hex grid to 0.1-mm spot-center geometry."""

    grid = np.asarray(array_row_col, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 2 or np.any(~np.isfinite(grid)):
        raise ValueError("Visium array coordinates must be finite row/column pairs")
    return np.column_stack((grid[:, 1] * 0.05, grid[:, 0] * np.sqrt(3.0) * 0.05))


def coordinate_pc1_tail_split(
    coordinates_mm: np.ndarray,
    *,
    guard_mm: float,
    minimum_spots: int = 100,
) -> Mapping[str, object]:
    """Return deterministic disjoint PC1 tails around a total-width empty guard."""

    coordinates = np.asarray(coordinates_mm, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 2 * int(minimum_spots)
        or np.any(~np.isfinite(coordinates))
        or float(guard_mm) <= 0
        or int(minimum_spots) < 1
    ):
        raise ValueError("same-section spatial split inputs are malformed")
    centered = coordinates - coordinates.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    largest_loading = int(np.argmax(np.abs(vector)))
    if vector[largest_loading] < 0:
        vector = -vector
    projection = centered @ vector
    split_center = float(np.median(projection))
    half_guard = float(guard_mm) / 2.0
    reference = projection <= split_center - half_guard
    query = projection >= split_center + half_guard
    if np.any(reference & query):
        raise RuntimeError("same-section reference and query tails overlap")
    if int(reference.sum()) < int(minimum_spots) or int(query.sum()) < int(minimum_spots):
        raise ValueError(
            "same-section guard leaves fewer than the prespecified spots in a tail"
        )
    return {
        "reference_mask": reference,
        "query_mask": query,
        "guard_mask": ~(reference | query),
        "projection": projection.astype(np.float32),
        "pc1_vector": vector.astype(np.float64),
        "split_center_mm": split_center,
        "guard_total_width_mm": float(guard_mm),
        "minimum_observed_tail_separation_mm": float(
            np.min(projection[query]) - np.max(projection[reference])
        ),
    }


def _baseline_paths(baseline: Path) -> Mapping[str, Path]:
    return {
        "projected_source_sha256": baseline / "panel_256_projected_counts.npz",
        "prepared_manifest_sha256": baseline / "prepared_manifest.json",
        "fit_predict_manifest_sha256": baseline / "fit_predict_manifest.json",
        "development_report_sha256": baseline / "report.json",
        "gene_panel_sha256": ROOT / "configs/natcommun_generative_gene_panel.json",
        "development_protocol_sha256": ROOT
        / "configs/natcommun_generative_development_protocol.json",
        "development_runner_sha256": RUNNER_PATH,
        "generative_core_sha256": CORE_PATH,
    }


def _verify_frozen_artifacts(
    protocol: Mapping[str, object],
    baseline: Path,
    *,
    verify_source_binding: bool = False,
) -> None:
    frozen = protocol.get("frozen_artifacts", {})
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen artifact identities are missing")
    for identity, path in _baseline_paths(baseline).items():
        if not path.is_file() or _sha256(path) != frozen.get(identity):
            raise ValueError(f"frozen artifact identity changed: {identity}")
    if verify_source_binding:
        prepared = json.loads(
            (baseline / "prepared_manifest.json").read_text(encoding="utf-8")
        )
        if prepared.get("source_sha256") != frozen.get("source_sha256"):
            raise ValueError("registered source identity differs from the matched-ST protocol")


def validate_direction_public(public: Mapping[str, object]) -> None:
    leaking = sorted(
        name
        for name in public
        if any(name.casefold().startswith(prefix) for prefix in FORBIDDEN_PUBLIC_PREFIXES)
    )
    if leaking:
        raise ValueError(f"query ST leaked into prediction input: {leaking}")
    if _scalar_text(public["schema"]) != PREPARED_SCHEMA:
        raise ValueError("direction public schema is malformed")
    donor = _scalar_text(public["donor"])
    reference_section = _scalar_text(public["reference_section"])
    query_section = _scalar_text(public["query_section"])
    design_family = _scalar_text(public["design_family"])
    indication = _scalar_text(public["indication"])
    guard_mm = float(public["guard_total_width_mm"])
    if design_family not in {"adjacent_section_primary", "same_section_upper_bound"}:
        raise ValueError("matched-ST design family is unknown")
    if design_family == "adjacent_section_primary" and reference_section == query_section:
        raise ValueError("adjacent reference and query sections must differ")
    if design_family == "adjacent_section_primary" and guard_mm != -1.0:
        raise ValueError("adjacent-section prediction input has an unexpected guard")
    if design_family == "same_section_upper_bound" and reference_section != query_section:
        raise ValueError("same-section upper-bound blocks must share one section")
    if design_family == "same_section_upper_bound" and guard_mm not in {1.0, 2.0}:
        raise ValueError("same-section prediction input has an unexpected guard")
    reference_ids = np.asarray(public["reference_spot_ids"]).astype(str)
    query_ids = np.asarray(public["query_spot_ids"]).astype(str)
    if not len(reference_ids) or not len(query_ids) or set(reference_ids) & set(query_ids):
        raise ValueError("reference and query spot identities must be nonempty and disjoint")
    reference_counts = np.asarray(public["reference_st_counts"])
    reference_library = np.asarray(public["reference_st_library"], dtype=float)
    if (
        reference_counts.ndim != 2
        or reference_counts.shape[0] != len(reference_ids)
        or reference_library.shape != (len(reference_ids),)
        or np.any(~np.isfinite(reference_counts))
        or np.any(reference_counts < 0)
        or np.any(reference_counts != np.floor(reference_counts))
        or np.any(~np.isfinite(reference_library))
        or np.any(reference_library <= 0)
        or np.any(reference_counts.sum(axis=1) > reference_library + 1.0e-6)
    ):
        raise ValueError("reference ST payload is malformed")
    image = np.asarray(public["query_image"])
    gene_ids = np.asarray(public["gene_ids"]).astype(str)
    genes = len(gene_ids)
    query_sections = np.asarray(public["query_section_ids"]).astype(str)
    query_indications = np.asarray(public["query_indication_ids"]).astype(str)
    if (
        image.ndim != 2
        or len(image) != len(query_ids)
        or np.any(~np.isfinite(image))
        or reference_counts.shape[1] != genes
        or not genes
        or len(set(gene_ids.tolist())) != genes
        or query_sections.shape != (len(query_ids),)
        or query_indications.shape != (len(query_ids),)
        or set(query_sections.tolist()) != {query_section}
        or set(query_indications.tolist()) != {indication}
    ):
        raise ValueError("direction image/count/gene payloads are misaligned")
    for arm in BASELINE_ARMS:
        rate = np.asarray(public[f"baseline_rate_{arm}"])
        if (
            rate.shape != (len(query_ids), genes)
            or np.any(~np.isfinite(rate))
            or np.any(rate <= 0)
        ):
            raise ValueError(f"baseline {arm} comparator is malformed")
    if not donor:
        raise ValueError("direction donor identity is empty")


def type_signatures(
    counts: np.ndarray,
    type_ids: np.ndarray,
    type_order: Sequence[str],
) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    labels = np.asarray(type_ids).astype(str)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("signature counts and type labels are misaligned")
    signatures = []
    for type_name in type_order:
        local = values[labels == str(type_name)]
        if not len(local):
            raise ValueError(f"training reference lacks type {type_name}")
        signatures.append(local.sum(axis=0) + 0.5)
    result = np.asarray(signatures, dtype=np.float64)
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def composition_proxy_from_signatures(
    counts: np.ndarray,
    signatures: np.ndarray,
    *,
    iterations: int = 80,
) -> np.ndarray:
    """Project selected-gene ST proportions onto training-only type signatures."""

    values = np.asarray(counts, dtype=np.float64)
    signature = np.asarray(signatures, dtype=np.float64)
    if (
        values.ndim != 2
        or signature.ndim != 2
        or values.shape[1] != signature.shape[1]
        or len(signature) < 2
        or int(iterations) < 1
    ):
        raise ValueError("composition proxy inputs are malformed")
    target = values + 0.5
    target /= target.sum(axis=1, keepdims=True)
    weights = np.full((len(values), len(signature)), 1.0 / len(signature))
    spectral_bound = float(np.linalg.norm(signature @ signature.T, ord=2))
    step = 0.5 / max(spectral_bound, 1.0e-8)
    for _ in range(int(iterations)):
        gradient = (weights @ signature - target) @ signature.T
        weights = np.maximum(weights - step * gradient, 1.0e-8)
        weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def _stable_seed(identifier: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{identifier}".encode()).digest()[:8], "little")


def weighted_soft_components(
    latent: np.ndarray,
    observation_ids: np.ndarray,
    observation_weights: np.ndarray,
    *,
    components: int,
    seed: int,
    iterations: int = 25,
    temperature: float = 1.0,
    variance_floor: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic weighted counterpart of the frozen soft-state mixture."""

    values = np.asarray(latent, dtype=np.float64)
    identifiers = np.asarray(observation_ids).astype(str)
    weights = np.asarray(observation_weights, dtype=np.float64)
    if (
        values.ndim != 2
        or identifiers.shape != (len(values),)
        or weights.shape != (len(values),)
        or len(set(identifiers.tolist())) != len(identifiers)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0)
        or float(weights.sum()) <= 0
        or not 1 <= int(components) <= len(values)
        or int(iterations) < 1
        or float(temperature) <= 0
    ):
        raise ValueError("weighted soft-component inputs are malformed")
    order = np.argsort(identifiers, kind="stable")
    values, identifiers, weights = values[order], identifiers[order], weights[order]
    weights /= weights.sum()
    maximum = float(weights.max())
    first = min(np.flatnonzero(weights == maximum).tolist(), key=lambda i: identifiers[i])
    chosen = [first]
    centers = [values[first].copy()]
    while len(centers) < int(components):
        distance = np.min(
            np.sum((values[:, None] - np.vstack(centers)[None]) ** 2, axis=2), axis=1
        )
        score = weights * distance
        score[np.asarray(chosen, dtype=np.int64)] = -1
        candidates = np.flatnonzero(score == np.max(score)).tolist()
        selected = min(
            candidates,
            key=lambda i: (_stable_seed(identifiers[i], seed), identifiers[i]),
        )
        chosen.append(selected)
        centers.append(values[selected].copy())
    means = np.vstack(centers)
    center = np.sum(weights[:, None] * values, axis=0)
    scale = max(float(np.sum(weights * np.sum((values - center) ** 2, axis=1))), 1.0e-8)
    for _ in range(int(iterations)):
        logits = -np.sum((values[:, None] - means[None]) ** 2, axis=2) / (
            2 * float(temperature) * scale
        )
        logits -= logits.max(axis=1, keepdims=True)
        responsibility = np.exp(logits)
        responsibility /= responsibility.sum(axis=1, keepdims=True)
        joint = weights[:, None] * responsibility
        mass = joint.sum(axis=0)
        if np.any(mass <= np.finfo(np.float64).tiny):
            raise RuntimeError("weighted ST reference produced an empty component")
        means = (joint.T @ values) / mass[:, None]
    residual = values[:, None] - means[None]
    variances = np.sum(joint[:, :, None] * residual**2, axis=0) / mass[:, None]
    return means, np.maximum(variances, float(variance_floor)), mass / mass.sum()


def build_spatial_reference_mixture(
    core: Any,
    latent: np.ndarray,
    composition: np.ndarray,
    type_names: Sequence[str],
    observation_ids: np.ndarray,
    donor_id: str,
    *,
    seed: int,
    components: int = 3,
    iterations: int = 25,
    temperature: float = 1.0,
    variance_floor: float = 1.0e-4,
    minimum_mass: float = 3.0,
    minimum_effective_sample_size: float = 3.0,
) -> tuple[Any, np.ndarray, Mapping[str, object]]:
    """Compress a mixed-spot ST section into type-weighted latent components."""

    values = np.asarray(latent, dtype=np.float64)
    proxy = np.asarray(composition, dtype=np.float64)
    identifiers = np.asarray(observation_ids).astype(str)
    names = tuple(str(value) for value in type_names)
    if (
        values.ndim != 2
        or proxy.shape != (len(values), len(names))
        or identifiers.shape != (len(values),)
        or np.any(proxy < 0)
        or not np.allclose(proxy.sum(axis=1), 1.0, atol=1.0e-5)
    ):
        raise ValueError("spatial reference inputs are malformed")
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    component_weights: list[float] = []
    out_types: list[str] = []
    component_ids: list[str] = []
    type_mass = proxy.sum(axis=0)
    type_ess = np.divide(
        type_mass**2,
        np.sum(proxy**2, axis=0),
        out=np.zeros_like(type_mass),
        where=np.sum(proxy**2, axis=0) > 0,
    )
    supported = (type_mass >= float(minimum_mass)) & (
        type_ess >= float(minimum_effective_sample_size)
    )
    for type_index, type_name in enumerate(names):
        if not supported[type_index]:
            continue
        local = weighted_soft_components(
            values,
            identifiers,
            proxy[:, type_index],
            components=int(components),
            seed=_stable_seed(f"{donor_id}:{type_name}", seed),
            iterations=int(iterations),
            temperature=float(temperature),
            variance_floor=float(variance_floor),
        )
        for component in range(int(components)):
            means.append(local[0][component])
            variances.append(local[1][component])
            component_weights.append(float(type_mass[type_index] * local[2][component]))
            out_types.append(type_name)
            component_ids.append(f"{donor_id}::{type_name}::weighted_ST::{component}")
    if len(set(out_types)) < 2:
        raise RuntimeError("spatial reference supports fewer than two coarse types")
    total = float(np.sum(component_weights))
    hard_types = np.asarray(names)[np.argmax(proxy, axis=1)]
    mixture = core.ReferenceMixture(
        means=np.asarray(means, dtype=np.float32),
        variances=np.asarray(variances, dtype=np.float32),
        weights=np.asarray(component_weights, dtype=np.float64) / total,
        donor_ids=np.repeat(str(donor_id), len(means)),
        type_labels=np.asarray(out_types),
        component_ids=np.asarray(component_ids),
        source_observation_ids=tuple(identifiers.tolist()),
        source_donor_ids=tuple(np.repeat(str(donor_id), len(identifiers)).tolist()),
        source_type_labels=tuple(hard_types.tolist()),
        source_modality="spatial_st_adjacent_section",
        source_sha256=_array_digest(values, proxy, identifiers),
    )
    reference_type_weight = np.where(supported, type_mass, 0.0)
    reference_type_weight /= reference_type_weight.sum()
    diagnostics = {
        "type_names": list(names),
        "proxy_mass": type_mass.tolist(),
        "proxy_effective_sample_size": type_ess.tolist(),
        "supported": supported.tolist(),
        "supported_type_count": int(supported.sum()),
        "components_per_supported_type": int(components),
        "source_spots": len(values),
        "source_modality": "spatial_st_adjacent_section",
        "type_truth": "training_signature_composition_proxy_not_observed_spot_type_truth",
    }
    return mixture, reference_type_weight.astype(np.float32), diagnostics


def _global_baseline_preflight(
    runner: Any,
    core: Any,
    baseline: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Validate every frozen baseline prediction before trusted partitioning."""

    prepared = runner._read_prepared_manifest(baseline)
    prediction_manifest_path = baseline / "fit_predict_manifest.json"
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    arguments = argparse.Namespace(
        output=baseline,
        seed=1729,
        epochs=80,
        batch_size=256,
        latent_dim=20,
        device="cuda:0",
    )
    runner._validate_prediction_manifest_binding(
        arguments,
        core,
        prepared,
        prediction_manifest,
    )
    return prepared, prediction_manifest


def _read_prepared_manifest(
    output: Path,
    protocol_path: Path,
    baseline_path: Path,
) -> Mapping[str, object]:
    output = _canonical_path(output)
    protocol_path = _canonical_path(protocol_path)
    baseline_path = _canonical_path(baseline_path)
    path = output / "prepared_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != PREPARED_SCHEMA:
        raise ValueError("matched-ST prepared manifest is missing or malformed")
    if payload.get("prepared_identity") != _manifest_semantic_identity(payload):
        raise ValueError("matched-ST prepared manifest self-identity changed")
    if _mapping_key_contains(payload, "score_target"):
        raise ValueError("target-free prepared manifest contains a score-target field")
    expected = {
        "analysis_scope": "outcome_exposed_mechanistic_diagnostic_only",
        "scientific_authorization": "none",
        "validation_runner_sha256": _sha256(Path(__file__).resolve()),
        "validation_protocol_sha256": _sha256(protocol_path),
        "frozen_runner_sha256": _sha256(RUNNER_PATH),
        "frozen_core_sha256": _sha256(CORE_PATH),
        "base_seed": 1729,
        "epochs": 80,
        "batch_size": 256,
        "latent_dim": 20,
        "device": "cuda:0",
        "cpu_threads": 4,
        "gpu_memory_fraction": 0.60,
        "adjacent_section_direction_count": 4,
        "same_section_upper_bound_direction_count": 30,
        "prediction_process_rule": "exactly_one_direction_per_process",
        "query_ST_open_stage": "score_only_after_selected_family_global_prediction_preflight",
        "section_relationship": (
            "same_block_serial_adjacent_Visium_sections_distinct_slides_"
            "distributional_not_spot_paired"
        ),
        "same_section_sensitivity_label": (
            "optimistic_same_section_same_assay_same_batch_ST_upper_bound_not_floor"
        ),
    }
    mismatch = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatch:
        raise ValueError(f"matched-ST prepared identities changed: {mismatch}")
    _require_canonical_path(payload.get("output", ""), output, "prepared output")
    _require_canonical_path(payload.get("baseline", ""), baseline_path, "prepared baseline")
    _require_canonical_path(
        payload.get("validation_protocol", ""), protocol_path, "prepared protocol"
    )
    baseline_predictions = json.loads(
        (baseline_path / "fit_predict_manifest.json").read_text(encoding="utf-8")
    )
    donors = tuple(str(value) for value in payload.get("donors", ()))
    donor_inputs = payload.get("donor_model_inputs")
    directions = payload.get("directions")
    if (
        donors != EXPECTED_DONORS
        or not isinstance(donor_inputs, Mapping)
        or set(donor_inputs) != set(EXPECTED_DONORS)
        or not isinstance(directions, Mapping)
    ):
        raise ValueError("prepared donor contracts differ from the 13 frozen donors")
    expected_directions = set(EXPECTED_ADJACENT_DIRECTIONS) | set(
        _expected_upper_directions()
    )
    if set(directions) != expected_directions or len(directions) != 34:
        raise ValueError("prepared direction set is not exactly 4 adjacent plus 30 upper")
    for donor in EXPECTED_DONORS:
        item = donor_inputs[donor]
        baseline_prediction = baseline_predictions["folds"][donor]
        if (
            not isinstance(item, Mapping)
            or item.get("heldout_ST_available_to_fit_model") is not False
        ):
            raise ValueError(f"prepared model input is malformed for {donor}")
        _require_canonical_path(
            item.get("baseline_public_path", ""),
            baseline_path / "folds" / donor / "fit_predict_input.npz",
            f"{donor} baseline public",
        )
        _require_canonical_path(
            item.get("baseline_prediction_path", ""),
            baseline_path / "folds" / donor / "predictions.npz",
            f"{donor} baseline predictions",
        )
        seed, public_sha256, prediction_sha256 = EXPECTED_DONOR_INPUT_IDENTITIES[donor]
        if prediction_sha256 != baseline_prediction["prediction_semantic_sha256"]:
            raise ValueError(f"frozen prediction identity changed for {donor}")
        expected_item = {
            "baseline_public_semantic_sha256": public_sha256,
            "baseline_prediction_semantic_sha256": prediction_sha256,
            "fold_seed": seed,
            "heldout_ST_available_to_fit_model": False,
        }
        if any(item.get(key) != value for key, value in expected_item.items()):
            raise ValueError(f"prepared model input semantics changed for {donor}")
    for direction_id, item in directions.items():
        if not isinstance(item, Mapping) or item.get("direction_id") != direction_id:
            raise ValueError(f"prepared direction receipt is malformed for {direction_id}")
        _require_canonical_path(
            item.get("predict_input_path", ""),
            output / "directions" / direction_id / "predict_input.npz",
            f"{direction_id} prediction input",
        )
        prepare_receipt_path = _require_canonical_path(
            item.get("prepare_receipt_path", ""),
            output / "directions" / direction_id / "prepare_receipt.json",
            f"{direction_id} prepare receipt",
        )
        if not str(item.get("predict_input_semantic_sha256", "")):
            raise ValueError(f"prepared direction semantic identity is empty for {direction_id}")
        if item.get("query_ST_available_to_predict") is not False:
            raise ValueError(f"prepared direction exposes query ST for {direction_id}")
        if direction_id in EXPECTED_ADJACENT_DIRECTIONS:
            reference_section, query_section = direction_id.split("__to__")
            donor, indication = _section_donor_indication(reference_section)
            if (
                item.get("design_family") != "adjacent_section_primary"
                or item.get("donor") != donor
                or item.get("indication") != indication
                or item.get("reference_section") != reference_section
                or item.get("query_section") != query_section
                or item.get("evidence_label")
                != "distributional_adjacent_section_mechanistic_diagnostic_only"
                or float(item.get("guard_total_width_mm", math.inf)) != -1.0
                or int(item.get("reference_spots", -1)) < 1
                or int(item.get("query_spots", -1)) < 1
                or int(item.get("query_score_eligible_spots", -1)) < 1
            ):
                raise ValueError(f"adjacent design identity changed for {direction_id}")
        else:
            _, section, guard_label = direction_id.split("__")
            expected_guard = 1.0 if guard_label == "guard_1mm" else 2.0
            donor, indication = _section_donor_indication(section)
            if (
                item.get("design_family") != "same_section_upper_bound"
                or item.get("donor") != donor
                or item.get("indication") != indication
                or item.get("reference_section") != section
                or item.get("query_section") != section
                or item.get("evidence_label")
                != "optimistic_same_section_same_assay_same_batch_ST_upper_bound"
                or float(item.get("guard_total_width_mm", -1)) != expected_guard
                or int(item.get("reference_spots", -1)) < 100
                or int(item.get("query_spots", -1)) < 100
                or int(item.get("query_score_eligible_spots", -1)) < 100
            ):
                raise ValueError(f"same-section upper contract changed for {direction_id}")
        if (
            not prepare_receipt_path.is_file()
            or json.loads(prepare_receipt_path.read_text(encoding="utf-8")) != item
        ):
            raise ValueError(f"prepared direction receipt changed for {direction_id}")
    return payload


def _read_score_target_manifest(
    output: Path,
    prepared: Mapping[str, object],
) -> Mapping[str, object]:
    output = _canonical_path(output)
    path = output / "score_target_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != SCORE_TARGET_MANIFEST_SCHEMA
        or payload.get("prepared_identity") != prepared.get("prepared_identity")
        or payload.get("score_access_boundary")
        != "read_only_after_selected_score_family_global_prediction_preflight"
        or payload.get("score_target_manifest_identity")
        != _score_target_manifest_semantic_identity(payload)
    ):
        raise ValueError("score-only target manifest is missing, stale, or malformed")
    _require_canonical_path(payload.get("output", ""), output, "score-target output")
    directions = payload.get("directions")
    if not isinstance(directions, Mapping) or set(directions) != set(prepared["directions"]):
        raise ValueError("score-only target directions differ from target-free preparation")
    for direction_id, item in directions.items():
        public = prepared["directions"][direction_id]
        if (
            not isinstance(item, Mapping)
            or item.get("direction_id") != direction_id
            or item.get("donor") != public.get("donor")
            or item.get("indication") != public.get("indication")
            or item.get("reference_section") != public.get("reference_section")
            or item.get("query_section") != public.get("query_section")
            or item.get("design_family") != public.get("design_family")
            or item.get("evidence_label") != public.get("evidence_label")
            or not np.isclose(
                float(item.get("guard_total_width_mm", math.inf)),
                float(public.get("guard_total_width_mm", -math.inf)),
            )
            or item.get("predict_input_semantic_sha256")
            != public.get("predict_input_semantic_sha256")
        ):
            raise ValueError(f"score-only target binding changed for {direction_id}")
        _require_canonical_path(
            item.get("score_target_path", ""),
            output / "directions" / direction_id / "score_target.npz",
            f"{direction_id} score target",
        )
        if not str(item.get("score_target_semantic_sha256", "")):
            raise ValueError(f"score-target semantic identity is empty for {direction_id}")
    return payload


def _validate_runtime_binding(
    args: argparse.Namespace,
    manifest: Mapping[str, object],
) -> None:
    observed = {
        "device": str(args.device),
        "cpu_threads": int(args.cpu_threads),
        "gpu_memory_fraction": float(args.gpu_memory_fraction),
    }
    mismatch = [name for name, value in observed.items() if manifest.get(name) != value]
    if mismatch:
        raise ValueError(f"runtime resources differ from matched-ST preparation: {mismatch}")


def prepare(args: argparse.Namespace, runner: Any, core: Any) -> Mapping[str, object]:
    protocol = _load_protocol(args.protocol)
    _verify_frozen_artifacts(protocol, args.baseline, verify_source_binding=True)
    baseline_prepared, baseline_predictions = _global_baseline_preflight(
        runner,
        core,
        args.baseline,
    )
    directions = _direction_map(protocol)
    output = _canonical_path(args.output)
    direction_receipts: dict[str, object] = {}
    score_target_receipts: dict[str, object] = {}
    donor_inputs: dict[str, object] = {}
    source_path = Path(str(baseline_prepared["source"]))
    if _sha256(source_path) != protocol["frozen_artifacts"]["source_sha256"]:
        raise ValueError("registered source changed before same-section spatial partitioning")
    with np.load(source_path, allow_pickle=False) as source_archive:
        source_spot_ids = np.asarray(source_archive["spot_ids"]).astype(str)
        source_primary = np.asarray(source_archive["spot_primary_eligible"], dtype=bool)
        source_sections = np.asarray(source_archive["section_ids"]).astype(str)
        source_array_grid = np.asarray(source_archive["array_row_col"], dtype=np.float64)
    if source_array_grid.shape != (len(source_spot_ids), 2):
        raise ValueError("registered source lacks aligned Visium array coordinates")
    primary_sections = tuple(sorted(set(source_sections[source_primary].tolist())))
    if primary_sections != tuple(sorted(EXPECTED_PRIMARY_SECTIONS)):
        raise ValueError("registered primary section set is not exactly the frozen 15 sections")
    primary_grid_by_spot = {
        spot_id: source_array_grid[index]
        for index, spot_id in enumerate(source_spot_ids)
        if source_primary[index]
    }
    for donor in [str(value) for value in baseline_prepared["donors"]]:
        fold = baseline_prepared["folds"][donor]
        prediction_receipt = baseline_predictions["folds"][donor]
        public_path = Path(str(fold["public_path"]))
        public = runner._verify_semantic_file(
            public_path,
            str(fold["public_semantic_sha256"]),
        )
        runner.validate_public_fold(public)
        predictions = runner._verify_semantic_file(
            Path(str(prediction_receipt["prediction_path"])),
            str(prediction_receipt["prediction_semantic_sha256"]),
        )
        runner._validate_prediction_artifact(
            predictions,
            public,
            donor=donor,
            epochs=80,
        )
        # This is the only trusted target-partitioning open.  The model-fit and
        # prediction stages receive paths that cannot reach this complete file.
        secret = runner._verify_semantic_file(
            Path(str(fold["score_target_path"])),
            str(fold["score_target_semantic_sha256"]),
        )
        if not np.array_equal(public["query_spot_ids"], secret["heldout_spot_ids"]):
            raise ValueError(f"baseline public/secret spot order differs for {donor}")
        donor_inputs[donor] = {
            "baseline_public_path": str(public_path.resolve()),
            "baseline_public_semantic_sha256": str(fold["public_semantic_sha256"]),
            "baseline_prediction_path": str(
                Path(str(prediction_receipt["prediction_path"])).resolve()
            ),
            "baseline_prediction_semantic_sha256": str(
                prediction_receipt["prediction_semantic_sha256"]
            ),
            "fold_seed": int(fold["seed"]),
            "heldout_ST_available_to_fit_model": False,
        }
        sections = np.asarray(secret["heldout_section_ids"]).astype(str)
        indication_ids = np.asarray(secret["heldout_indication_ids"]).astype(str)
        for direction_id, contract in directions.items():
            if str(contract["donor"]) != donor:
                continue
            reference_section = str(contract["reference_section"])
            query_section = str(contract["query_section"])
            reference_keep_all = sections == reference_section
            query_keep = sections == query_section
            if not reference_keep_all.any() or not query_keep.any():
                raise ValueError(f"registered section is missing for {direction_id}")
            if set(indication_ids[reference_keep_all].tolist()) != {
                str(contract["indication"])
            } or set(indication_ids[query_keep].tolist()) != {str(contract["indication"])}:
                raise ValueError(f"direction indication differs from the protocol: {direction_id}")
            reference_positive = np.asarray(secret["heldout_st_library"], dtype=float) > 0
            reference_keep = reference_keep_all & reference_positive
            direction_dir = output / "directions" / direction_id
            public_direction: dict[str, object] = {
                "schema": np.asarray(PREPARED_SCHEMA),
                "direction_id": np.asarray(direction_id),
                "donor": np.asarray(donor),
                "indication": np.asarray(str(contract["indication"])),
                "design_family": np.asarray("adjacent_section_primary"),
                "evidence_label": np.asarray(
                    "distributional_adjacent_section_mechanistic_diagnostic_only"
                ),
                "guard_total_width_mm": np.asarray(-1.0, dtype=np.float32),
                "reference_section": np.asarray(reference_section),
                "query_section": np.asarray(query_section),
                "gene_ids": np.asarray(public["gene_ids"]),
                "reference_spot_ids": np.asarray(secret["heldout_spot_ids"])[
                    reference_keep
                ],
                "reference_st_counts": np.asarray(secret["heldout_st_counts"])[
                    reference_keep
                ],
                "reference_st_library": np.asarray(secret["heldout_st_library"])[
                    reference_keep
                ],
                "query_spot_ids": np.asarray(public["query_spot_ids"])[query_keep],
                "query_section_ids": np.asarray(public["query_section_ids"])[query_keep],
                "query_indication_ids": np.asarray(public["query_indication_ids"])[
                    query_keep
                ],
                "query_image": np.asarray(public["query_image"])[query_keep],
                "baseline_training_dispersion": np.asarray(
                    predictions["training_only_dispersion"]
                ),
                **{
                    f"baseline_rate_{arm}": np.asarray(predictions[f"rate_{arm}"])[
                        query_keep
                    ]
                    for arm in BASELINE_ARMS
                },
            }
            validate_direction_public(public_direction)
            score_target = {
                "schema": np.asarray(PREPARED_SCHEMA),
                "direction_id": np.asarray(direction_id),
                "donor": np.asarray(donor),
                "indication": np.asarray(str(contract["indication"])),
                "design_family": np.asarray("adjacent_section_primary"),
                "evidence_label": np.asarray(
                    "distributional_adjacent_section_mechanistic_diagnostic_only"
                ),
                "guard_total_width_mm": np.asarray(-1.0, dtype=np.float32),
                "query_section": np.asarray(query_section),
                "gene_ids": np.asarray(public["gene_ids"]),
                "query_spot_ids": np.asarray(secret["heldout_spot_ids"])[query_keep],
                "query_section_ids": np.asarray(secret["heldout_section_ids"])[query_keep],
                "query_indication_ids": np.asarray(secret["heldout_indication_ids"])[
                    query_keep
                ],
                "query_st_counts": np.asarray(secret["heldout_st_counts"])[query_keep],
                "query_st_library": np.asarray(secret["heldout_st_library"])[query_keep],
                "primary_score_eligible": np.asarray(secret["primary_score_eligible"])[
                    query_keep
                ],
            }
            if not np.array_equal(
                public_direction["query_spot_ids"], score_target["query_spot_ids"]
            ):
                raise RuntimeError("trusted query partition changed spot order")
            public_direction_path = direction_dir / "predict_input.npz"
            score_target_path = direction_dir / "score_target.npz"
            _atomic_npz(public_direction_path, public_direction)
            _atomic_npz(score_target_path, score_target)
            prepare_receipt_path = direction_dir / "prepare_receipt.json"
            receipt = {
                "schema": PREPARED_SCHEMA,
                "direction_id": direction_id,
                "donor": donor,
                "indication": str(contract["indication"]),
                "reference_section": reference_section,
                "query_section": query_section,
                "predict_input_path": str(public_direction_path.resolve()),
                "predict_input_semantic_sha256": runner._semantic_array_hash(
                    public_direction
                ),
                "prepare_receipt_path": str(prepare_receipt_path.resolve()),
                "reference_spots": int(reference_keep.sum()),
                "reference_zero_depth_excluded": int(
                    reference_keep_all.sum() - reference_keep.sum()
                ),
                "query_spots": int(query_keep.sum()),
                "query_score_eligible_spots": int(
                    np.asarray(score_target["primary_score_eligible"], dtype=bool).sum()
                ),
                "query_ST_available_to_predict": False,
                "spot_registration_claim": False,
                "design_family": "adjacent_section_primary",
                "evidence_label": (
                    "distributional_adjacent_section_mechanistic_diagnostic_only"
                ),
                "guard_total_width_mm": -1.0,
            }
            target_receipt = {
                "schema": SCORE_TARGET_MANIFEST_SCHEMA,
                "direction_id": direction_id,
                "donor": donor,
                "indication": str(contract["indication"]),
                "reference_section": reference_section,
                "query_section": query_section,
                "design_family": "adjacent_section_primary",
                "evidence_label": (
                    "distributional_adjacent_section_mechanistic_diagnostic_only"
                ),
                "guard_total_width_mm": -1.0,
                "predict_input_semantic_sha256": receipt[
                    "predict_input_semantic_sha256"
                ],
                "score_target_path": str(score_target_path.resolve()),
                "score_target_semantic_sha256": runner._semantic_array_hash(score_target),
            }
            _atomic_json(prepare_receipt_path, receipt)
            direction_receipts[direction_id] = receipt
            score_target_receipts[direction_id] = target_receipt
        upper_contract = protocol["same_section_upper_bound"]
        minimum_tail = int(upper_contract["minimum_reference_spots"])
        all_spot_ids = np.asarray(secret["heldout_spot_ids"]).astype(str)
        for section in sorted(set(sections.tolist())):
            section_keep = sections == section
            local_ids = all_spot_ids[section_keep]
            try:
                local_grid = np.vstack([primary_grid_by_spot[value] for value in local_ids])
            except KeyError as error:
                raise ValueError(
                    f"registered array coordinate is missing for {section}/{error.args[0]}"
                ) from error
            physical_coordinates = physical_array_coordinates_mm(local_grid)
            for guard_mm in upper_contract["empty_guard_total_width_mm"]:
                split = coordinate_pc1_tail_split(
                    physical_coordinates,
                    guard_mm=float(guard_mm),
                    minimum_spots=minimum_tail,
                )
                local_rows = np.flatnonzero(section_keep)
                reference_keep_all = np.zeros(len(sections), dtype=bool)
                query_keep = np.zeros(len(sections), dtype=bool)
                reference_keep_all[local_rows[np.asarray(split["reference_mask"], dtype=bool)]] = (
                    True
                )
                query_keep[local_rows[np.asarray(split["query_mask"], dtype=bool)]] = True
                reference_positive = np.asarray(secret["heldout_st_library"], dtype=float) > 0
                reference_keep = reference_keep_all & reference_positive
                if int(reference_keep.sum()) < minimum_tail:
                    raise ValueError(
                        f"positive-depth reference tail is too small for {section}/{guard_mm}mm"
                    )
                direction_id = _upper_direction_id(section, float(guard_mm))
                direction_dir = output / "directions" / direction_id
                public_direction = {
                    "schema": np.asarray(PREPARED_SCHEMA),
                    "direction_id": np.asarray(direction_id),
                    "donor": np.asarray(donor),
                    "indication": np.asarray(str(indication_ids[section_keep][0])),
                    "design_family": np.asarray("same_section_upper_bound"),
                    "evidence_label": np.asarray(
                        "optimistic_same_section_same_assay_same_batch_ST_upper_bound"
                    ),
                    "guard_total_width_mm": np.asarray(guard_mm, dtype=np.float32),
                    "reference_section": np.asarray(section),
                    "query_section": np.asarray(section),
                    "gene_ids": np.asarray(public["gene_ids"]),
                    "reference_spot_ids": np.asarray(secret["heldout_spot_ids"])[
                        reference_keep
                    ],
                    "reference_st_counts": np.asarray(secret["heldout_st_counts"])[
                        reference_keep
                    ],
                    "reference_st_library": np.asarray(secret["heldout_st_library"])[
                        reference_keep
                    ],
                    "query_spot_ids": np.asarray(public["query_spot_ids"])[query_keep],
                    "query_section_ids": np.asarray(public["query_section_ids"])[query_keep],
                    "query_indication_ids": np.asarray(public["query_indication_ids"])[
                        query_keep
                    ],
                    "query_image": np.asarray(public["query_image"])[query_keep],
                    "baseline_training_dispersion": np.asarray(
                        predictions["training_only_dispersion"]
                    ),
                    "split_PC1_vector": np.asarray(split["pc1_vector"], dtype=np.float64),
                    "split_center_mm": np.asarray(
                        split["split_center_mm"], dtype=np.float64
                    ),
                    "minimum_tail_separation_mm": np.asarray(
                        split["minimum_observed_tail_separation_mm"], dtype=np.float64
                    ),
                    **{
                        f"baseline_rate_{arm}": np.asarray(predictions[f"rate_{arm}"])[
                            query_keep
                        ]
                        for arm in BASELINE_ARMS
                    },
                }
                validate_direction_public(public_direction)
                score_target = {
                    "schema": np.asarray(PREPARED_SCHEMA),
                    "direction_id": np.asarray(direction_id),
                    "donor": np.asarray(donor),
                    "indication": np.asarray(str(indication_ids[section_keep][0])),
                    "design_family": np.asarray("same_section_upper_bound"),
                    "evidence_label": np.asarray(
                        "optimistic_same_section_same_assay_same_batch_ST_upper_bound"
                    ),
                    "guard_total_width_mm": np.asarray(guard_mm, dtype=np.float32),
                    "query_section": np.asarray(section),
                    "gene_ids": np.asarray(public["gene_ids"]),
                    "query_spot_ids": np.asarray(secret["heldout_spot_ids"])[query_keep],
                    "query_section_ids": np.asarray(secret["heldout_section_ids"])[
                        query_keep
                    ],
                    "query_indication_ids": np.asarray(secret["heldout_indication_ids"])[
                        query_keep
                    ],
                    "query_st_counts": np.asarray(secret["heldout_st_counts"])[query_keep],
                    "query_st_library": np.asarray(secret["heldout_st_library"])[query_keep],
                    "primary_score_eligible": np.asarray(secret["primary_score_eligible"])[
                        query_keep
                    ],
                }
                query_score_eligible = int(
                    np.asarray(score_target["primary_score_eligible"], dtype=bool).sum()
                )
                if query_score_eligible < minimum_tail:
                    raise ValueError(
                        f"score-eligible query tail is too small for {section}/{guard_mm}mm"
                    )
                if not np.array_equal(
                    public_direction["query_spot_ids"], score_target["query_spot_ids"]
                ):
                    raise RuntimeError("same-section target partition changed spot order")
                public_direction_path = direction_dir / "predict_input.npz"
                score_target_path = direction_dir / "score_target.npz"
                _atomic_npz(public_direction_path, public_direction)
                _atomic_npz(score_target_path, score_target)
                prepare_receipt_path = direction_dir / "prepare_receipt.json"
                receipt = {
                    "schema": PREPARED_SCHEMA,
                    "direction_id": direction_id,
                    "donor": donor,
                    "indication": str(indication_ids[section_keep][0]),
                    "reference_section": section,
                    "query_section": section,
                    "predict_input_path": str(public_direction_path.resolve()),
                    "predict_input_semantic_sha256": runner._semantic_array_hash(
                        public_direction
                    ),
                    "prepare_receipt_path": str(prepare_receipt_path.resolve()),
                    "reference_spots": int(reference_keep.sum()),
                    "reference_zero_depth_excluded": int(
                        reference_keep_all.sum() - reference_keep.sum()
                    ),
                    "query_spots": int(query_keep.sum()),
                    "query_score_eligible_spots": query_score_eligible,
                    "guard_spots": int(np.asarray(split["guard_mask"], dtype=bool).sum()),
                    "guard_total_width_mm": float(guard_mm),
                    "minimum_observed_tail_separation_mm": float(
                        split["minimum_observed_tail_separation_mm"]
                    ),
                    "query_ST_available_to_predict": False,
                    "spot_registration_claim": False,
                    "design_family": "same_section_upper_bound",
                    "evidence_label": (
                        "optimistic_same_section_same_assay_same_batch_ST_upper_bound"
                    ),
                    "measurement_floor_claim": False,
                    "independent_confirmation_claim": False,
                }
                target_receipt = {
                    "schema": SCORE_TARGET_MANIFEST_SCHEMA,
                    "direction_id": direction_id,
                    "donor": donor,
                    "indication": str(indication_ids[section_keep][0]),
                    "reference_section": section,
                    "query_section": section,
                    "design_family": "same_section_upper_bound",
                    "evidence_label": (
                        "optimistic_same_section_same_assay_same_batch_ST_upper_bound"
                    ),
                    "guard_total_width_mm": float(guard_mm),
                    "predict_input_semantic_sha256": receipt[
                        "predict_input_semantic_sha256"
                    ],
                    "score_target_path": str(score_target_path.resolve()),
                    "score_target_semantic_sha256": runner._semantic_array_hash(
                        score_target
                    ),
                }
                _atomic_json(prepare_receipt_path, receipt)
                direction_receipts[direction_id] = receipt
                score_target_receipts[direction_id] = target_receipt
    expected_direction_ids = set(EXPECTED_ADJACENT_DIRECTIONS) | set(
        _expected_upper_directions()
    )
    if (
        tuple(sorted(donor_inputs)) != EXPECTED_DONORS
        or set(direction_receipts) != expected_direction_ids
        or set(score_target_receipts) != expected_direction_ids
    ):
        raise RuntimeError("trusted preparation did not produce the exact frozen cohort design")
    manifest: dict[str, object] = {
        "schema": PREPARED_SCHEMA,
        "analysis_scope": "outcome_exposed_mechanistic_diagnostic_only",
        "scientific_authorization": "none",
        "output": str(output),
        "validation_runner_sha256": _sha256(Path(__file__).resolve()),
        "validation_protocol": str(_canonical_path(args.protocol)),
        "validation_protocol_sha256": _sha256(_canonical_path(args.protocol)),
        "frozen_runner_sha256": _sha256(RUNNER_PATH),
        "frozen_core_sha256": _sha256(CORE_PATH),
        "baseline": str(_canonical_path(args.baseline)),
        "base_seed": 1729,
        "epochs": 80,
        "batch_size": 256,
        "latent_dim": 20,
        "device": args.device,
        "cpu_threads": int(args.cpu_threads),
        "gpu_memory_fraction": float(args.gpu_memory_fraction),
        "donors": sorted(donor_inputs),
        "donor_model_inputs": donor_inputs,
        "directions": direction_receipts,
        "adjacent_section_direction_count": int(
            sum(
                receipt["design_family"] == "adjacent_section_primary"
                for receipt in direction_receipts.values()
            )
        ),
        "same_section_upper_bound_direction_count": int(
            sum(
                receipt["design_family"] == "same_section_upper_bound"
                for receipt in direction_receipts.values()
            )
        ),
        "prediction_process_rule": "exactly_one_direction_per_process",
        "query_ST_open_stage": (
            "score_only_after_selected_family_global_prediction_preflight"
        ),
        "section_relationship": (
            "same_block_serial_adjacent_Visium_sections_distinct_slides_"
            "distributional_not_spot_paired"
        ),
        "same_section_sensitivity_label": (
            "optimistic_same_section_same_assay_same_batch_ST_upper_bound_not_floor"
        ),
    }
    manifest["prepared_identity"] = _manifest_semantic_identity(manifest)
    _atomic_json(output / "prepared_manifest.json", manifest)
    target_manifest: dict[str, object] = {
        "schema": SCORE_TARGET_MANIFEST_SCHEMA,
        "output": str(output),
        "prepared_identity": manifest["prepared_identity"],
        "directions": score_target_receipts,
        "score_access_boundary": (
            "read_only_after_selected_score_family_global_prediction_preflight"
        ),
    }
    target_manifest["score_target_manifest_identity"] = (
        _score_target_manifest_semantic_identity(target_manifest)
    )
    _atomic_json(output / "score_target_manifest.json", target_manifest)
    validated = _read_prepared_manifest(output, args.protocol, args.baseline)
    _read_score_target_manifest(output, validated)
    return validated


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _model_identity(
    manifest: Mapping[str, object],
    donor: str,
    donor_input: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "schema": MODEL_SCHEMA,
                "donor": donor,
                "prepared_identity": manifest["prepared_identity"],
                "baseline_public_semantic_sha256": donor_input[
                    "baseline_public_semantic_sha256"
                ],
                "fold_seed": donor_input["fold_seed"],
                "validation_runner_sha256": manifest["validation_runner_sha256"],
                "validation_protocol_sha256": manifest["validation_protocol_sha256"],
                "frozen_runner_sha256": manifest["frozen_runner_sha256"],
                "frozen_core_sha256": manifest["frozen_core_sha256"],
                "epochs": manifest["epochs"],
                "batch_size": manifest["batch_size"],
                "latent_dim": manifest["latent_dim"],
                "device": manifest["device"],
            }
        )
    ).hexdigest()


def _validate_model_receipt(
    receipt: Mapping[str, object],
    *,
    output: Path,
    donor: str,
    expected_identity: str,
    prepared_identity: str,
) -> Path:
    output = _canonical_path(output)
    checkpoint_path = _require_canonical_path(
        receipt.get("checkpoint_path", ""),
        output / "models" / donor / "frozen_model.pt",
        f"{donor} model checkpoint",
    )
    receipt_path = _require_canonical_path(
        receipt.get("fit_receipt_path", ""),
        output / "models" / donor / "fit_receipt.json",
        f"{donor} model receipt",
    )
    if (
        receipt.get("schema") != MODEL_SCHEMA
        or receipt.get("donor") != donor
        or receipt.get("model_identity") != expected_identity
        or receipt.get("prepared_identity") != prepared_identity
        or receipt.get("heldout_ST_opened") is not False
        or receipt.get("query_ST_opened") is not False
        or int(receipt.get("process_swap_kib_at_completion", -1)) != 0
        or not receipt_path.is_file()
        or json.loads(receipt_path.read_text(encoding="utf-8")) != receipt
        or not checkpoint_path.is_file()
        or _sha256(checkpoint_path) != receipt.get("checkpoint_sha256")
    ):
        raise ValueError(f"matched-ST model receipt is stale for {donor}")
    return checkpoint_path


def _validate_model_checkpoint_payload(
    payload: Mapping[str, object],
    *,
    donor: str,
    expected_identity: str,
    prepared_identity: str,
    manifest: Mapping[str, object],
) -> None:
    donor_input = manifest["donor_model_inputs"][donor]
    if (
        payload.get("schema") != MODEL_SCHEMA
        or payload.get("donor") != donor
        or payload.get("model_identity") != expected_identity
        or payload.get("prepared_identity") != prepared_identity
        or payload.get("heldout_ST_opened") is not False
        or payload.get("validation_runner_sha256")
        != manifest["validation_runner_sha256"]
        or payload.get("validation_protocol_sha256")
        != manifest["validation_protocol_sha256"]
        or payload.get("frozen_runner_sha256") != manifest["frozen_runner_sha256"]
        or payload.get("frozen_core_sha256") != manifest["frozen_core_sha256"]
        or payload.get("baseline_public_semantic_sha256")
        != donor_input["baseline_public_semantic_sha256"]
        or int(payload.get("seed", -1)) != int(donor_input["fold_seed"])
        or int(payload.get("genes", -1)) != 256
        or int(payload.get("latent_dim", -1)) != 20
        or payload.get("matched_suspension_used_for_model_fit") is not False
        or payload.get(
            "matched_suspension_encoded_after_model_fit_for_frozen_comparator"
        )
        is not True
        or payload.get("UNI2_h_run") is not False
    ):
        raise ValueError(f"matched-ST model checkpoint payload is stale for {donor}")
    train_donors = np.asarray(payload.get("train_donor_ids", ())).astype(str)
    train_indications = np.asarray(payload.get("train_indication_ids", ())).astype(str)
    matched_donors = np.asarray(payload.get("matched_sc_donor_ids", ())).astype(str)
    if (
        not len(train_donors)
        or train_donors.shape != train_indications.shape
        or donor in set(train_donors.tolist())
        or set(matched_donors.tolist()) != {donor}
    ):
        raise ValueError(f"matched-ST checkpoint donor exclusion changed for {donor}")


def _fit_donor_model(
    runner: Any,
    core: Any,
    public: Mapping[str, np.ndarray],
    *,
    donor: str,
    seed: int,
    device: str,
) -> Mapping[str, object]:
    """Fit only the frozen donor-excluded molecular and H&E models."""

    runner.validate_public_fold(public)
    if donor in set(np.asarray(public["train_donor_ids"]).astype(str).tolist()):
        raise ValueError("held-out donor leaked into spatial model training")
    runner.seed_everything(seed)
    train_counts = np.asarray(public["train_st_counts"], dtype=np.float32)
    train_library = np.asarray(public["train_st_library"], dtype=np.float32)
    sc_counts = np.asarray(public["train_sc_counts"], dtype=np.float32)
    sc_library = np.asarray(public["train_sc_library"], dtype=np.float32)
    genes = train_counts.shape[1]
    train_augmented = runner._append_other_count_bin(train_counts, train_library)
    sc_augmented = runner._append_other_count_bin(sc_counts, sc_library)
    dispersion = np.asarray(
        runner._call_with_supported_kwargs(
            core.fit_nb2_dispersion,
            train_counts,
            training_observation_ids=np.asarray(public["train_spot_ids"]).astype(str),
            training_donor_ids=np.asarray(public["train_donor_ids"]).astype(str),
            library_size=train_library,
        ),
        dtype=np.float32,
    )
    molecular_counts = np.concatenate((train_augmented, sc_augmented), axis=0)
    modality = np.concatenate(
        (
            np.ones(len(train_counts), dtype=np.int64),
            np.zeros(len(sc_counts), dtype=np.int64),
        )
    )
    molecular_donors = np.concatenate(
        (
            np.asarray(public["train_donor_ids"]).astype(str),
            np.asarray(public["train_sc_donor_ids"]).astype(str),
        )
    )
    observation_ids = np.concatenate(
        (
            np.char.add("st::", np.asarray(public["train_spot_ids"]).astype(str)),
            np.char.add("reference::", np.asarray(public["train_sc_cell_ids"]).astype(str)),
        )
    )
    heldout_ids = np.concatenate(
        (
            np.char.add("st::", np.asarray(public["query_spot_ids"]).astype(str)),
            np.char.add("reference::", np.asarray(public["matched_sc_cell_ids"]).astype(str)),
        )
    )
    vae_hidden = min(256, max(32, genes))
    vae = runner._instantiate(
        core.CountVAE,
        n_genes=genes + 1,
        latent_dim=20,
        hidden_dim=vae_hidden,
    )
    vae.to(torch.device(device))
    fit = getattr(vae, "fit_model", None) or getattr(vae, "fit", None)
    runner._call_with_supported_kwargs(
        fit,
        molecular_counts,
        modality=modality,
        training_donor_ids=molecular_donors,
        alignment_weight=1.0,
        observation_ids=observation_ids,
        heldout_observation_ids=heldout_ids,
        library_size=np.concatenate((train_library, sc_library)),
        epochs=80,
        batch_size=256,
        device=device,
        seed=seed,
    )
    alignment = getattr(vae, "alignment_diagnostics", None)
    if alignment is None or not bool(alignment.support_criterion_met):
        raise RuntimeError("frozen training-only cross-assay alignment failed")
    train_st_latent = runner._encode(vae, train_augmented, modality="st", device=device)
    train_sc_latent = runner._encode(vae, sc_augmented, modality="scrna", device=device)
    train_sc_types = np.asarray(public["train_sc_type_ids"]).astype(str)
    all_types = tuple(sorted(set(train_sc_types.tolist())))
    composition, proxy_types = runner.training_only_composition_proxy(
        train_counts,
        sc_counts,
        train_sc_types,
    )
    if proxy_types != all_types:
        raise RuntimeError("training composition type order changed")
    anchors = np.vstack(
        [train_sc_latent[train_sc_types == type_name].mean(axis=0) for type_name in all_types]
    ).astype(np.float32)
    image_dim = int(np.asarray(public["train_image"]).shape[1])
    state_hidden = min(256, max(32, 20 * 4))
    state = runner._instantiate(
        core.CompositionStateModel,
        image_dim=image_dim,
        type_labels=all_types,
        n_genes=genes + 1,
        latent_dim=20,
        hidden_dim=state_hidden,
    )
    state.to(torch.device(device))
    state_fit = getattr(state, "fit_model", None) or getattr(state, "fit", None)
    runner._call_with_supported_kwargs(
        state_fit,
        np.asarray(public["train_image"], dtype=np.float32),
        train_st_latent,
        composition_targets=composition,
        type_ids=all_types,
        type_anchor_means=anchors,
        epochs=80,
        batch_size=256,
        device=device,
        seed=seed,
    )
    signatures = type_signatures(sc_counts, train_sc_types, all_types)
    matched_sc_counts = np.asarray(public["matched_sc_counts"], dtype=np.float32)
    matched_sc_augmented = runner._append_other_count_bin(
        matched_sc_counts,
        np.asarray(public["matched_sc_library"], dtype=np.float32),
    )
    matched_sc_latent = runner._encode(
        vae,
        matched_sc_augmented,
        modality="scrna",
        device=device,
    )
    matched_sc_donors = np.asarray(public["matched_sc_donor_ids"]).astype(str)
    if set(matched_sc_donors.tolist()) != {donor}:
        raise ValueError("suspension comparator does not contain only the matched donor")
    return {
        "schema": MODEL_SCHEMA,
        "donor": donor,
        "seed": int(seed),
        "genes": int(genes),
        "latent_dim": 20,
        "image_dim": image_dim,
        "vae_hidden_dim": vae_hidden,
        "state_hidden_dim": state_hidden,
        "type_names": list(all_types),
        "dispersion": torch.as_tensor(dispersion),
        "type_anchor_means": torch.as_tensor(anchors),
        "type_signatures": torch.as_tensor(signatures),
        "train_st_latent": torch.as_tensor(train_st_latent),
        "train_st_composition_proxy": torch.as_tensor(composition),
        "train_spot_ids": np.asarray(public["train_spot_ids"]).astype(str).tolist(),
        "train_donor_ids": np.asarray(public["train_donor_ids"]).astype(str).tolist(),
        "train_indication_ids": np.asarray(public["train_indication_ids"]).astype(str).tolist(),
        "matched_sc_latent": torch.as_tensor(matched_sc_latent),
        "matched_sc_cell_ids": np.asarray(public["matched_sc_cell_ids"]).astype(str).tolist(),
        "matched_sc_donor_ids": matched_sc_donors.tolist(),
        "matched_sc_type_ids": np.asarray(public["matched_sc_type_ids"]).astype(str).tolist(),
        "vae_state_dict": _cpu_state_dict(vae),
        "state_state_dict": _cpu_state_dict(state),
        "alignment": {
            "donor_ids": list(alignment.donor_ids),
            "pre_matched_to_mismatched_ratio": float(
                alignment.pre_matched_to_mismatched_ratio
            ),
            "post_matched_to_mismatched_ratio": float(
                alignment.post_matched_to_mismatched_ratio
            ),
            "support_criterion_met": bool(alignment.support_criterion_met),
        },
        "heldout_ST_opened": False,
        "matched_suspension_used_for_model_fit": False,
        "matched_suspension_encoded_after_model_fit_for_frozen_comparator": True,
        "UNI2_h_run": False,
    }


def fit_models(args: argparse.Namespace, runner: Any, core: Any) -> Mapping[str, object]:
    protocol = _load_protocol(args.protocol)
    _verify_frozen_artifacts(protocol, args.baseline)
    manifest = _read_prepared_manifest(args.output, args.protocol, args.baseline)
    _validate_runtime_binding(args, manifest)
    requested = [args.donor] if args.donor else [str(value) for value in manifest["donors"]]
    unknown = sorted(set(requested) - set(str(value) for value in manifest["donors"]))
    if unknown:
        raise ValueError(f"unknown matched-ST donor models: {unknown}")
    output = _canonical_path(args.output)
    receipts: dict[str, object] = {}
    receipt_path = output / "model_manifest.json"
    if args.resume and receipt_path.is_file():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(old, Mapping)
            or old.get("schema") != MODEL_SCHEMA
            or old.get("prepared_identity") != manifest["prepared_identity"]
            or not isinstance(old.get("models"), Mapping)
        ):
            raise ValueError("existing model manifest is stale or malformed")
        for old_donor, old_receipt in old["models"].items():
            if old_donor not in manifest["donor_model_inputs"] or not isinstance(
                old_receipt, Mapping
            ):
                raise ValueError("existing model manifest contains an unknown donor")
            old_identity = _model_identity(
                manifest,
                old_donor,
                manifest["donor_model_inputs"][old_donor],
            )
            checkpoint_path = _validate_model_receipt(
                old_receipt,
                output=output,
                donor=old_donor,
                expected_identity=old_identity,
                prepared_identity=str(manifest["prepared_identity"]),
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"model checkpoint is malformed for {old_donor}")
            _validate_model_checkpoint_payload(
                checkpoint,
                donor=old_donor,
                expected_identity=old_identity,
                prepared_identity=str(manifest["prepared_identity"]),
                manifest=manifest,
            )
            receipts[old_donor] = old_receipt
    for donor in requested:
        donor_input = manifest["donor_model_inputs"][donor]
        identity = _model_identity(manifest, donor, donor_input)
        model_dir = output / "models" / donor
        model_path = model_dir / "frozen_model.pt"
        model_receipt_path = model_dir / "fit_receipt.json"
        if args.resume and donor in receipts:
            continue
        public = runner._verify_semantic_file(
            Path(str(donor_input["baseline_public_path"])),
            str(donor_input["baseline_public_semantic_sha256"]),
        )
        payload = dict(
            _fit_donor_model(
                runner,
                core,
                public,
                donor=donor,
                seed=int(donor_input["fold_seed"]),
                device=args.device,
            )
        )
        payload.update(
            {
                "model_identity": identity,
                "prepared_identity": manifest["prepared_identity"],
                "validation_runner_sha256": manifest["validation_runner_sha256"],
                "validation_protocol_sha256": manifest["validation_protocol_sha256"],
                "frozen_runner_sha256": manifest["frozen_runner_sha256"],
                "frozen_core_sha256": manifest["frozen_core_sha256"],
                "baseline_public_semantic_sha256": donor_input[
                    "baseline_public_semantic_sha256"
                ],
            }
        )
        _atomic_torch(model_path, payload)
        receipt = {
            "schema": MODEL_SCHEMA,
            "donor": donor,
            "model_identity": identity,
            "prepared_identity": manifest["prepared_identity"],
            "checkpoint_path": str(model_path.resolve()),
            "fit_receipt_path": str(model_receipt_path.resolve()),
            "checkpoint_sha256": _sha256(model_path),
            "baseline_public_semantic_sha256": donor_input[
                "baseline_public_semantic_sha256"
            ],
            "heldout_ST_opened": False,
            "query_ST_opened": False,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "epochs": 80,
            "batch_size": 256,
            "latent_dim": 20,
            "process_swap_kib_at_completion": _assert_zero_process_swap(),
        }
        _atomic_json(model_receipt_path, receipt)
        receipts[donor] = receipt
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    aggregate = {
        "schema": MODEL_SCHEMA,
        "prepared_identity": manifest["prepared_identity"],
        "validation_runner_sha256": manifest["validation_runner_sha256"],
        "validation_protocol_sha256": manifest["validation_protocol_sha256"],
        "models": receipts,
        "all_models_complete": set(receipts) == set(manifest["donors"]),
        "heldout_ST_opened": False,
    }
    _atomic_json(receipt_path, aggregate)
    return aggregate


def _load_model_checkpoint(
    args: argparse.Namespace,
    manifest: Mapping[str, object],
    donor: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    output = _canonical_path(args.output)
    model_manifest_path = output / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(model_manifest, Mapping)
        or model_manifest.get("schema") != MODEL_SCHEMA
        or model_manifest.get("prepared_identity") != manifest["prepared_identity"]
        or model_manifest.get("validation_runner_sha256")
        != manifest["validation_runner_sha256"]
        or model_manifest.get("validation_protocol_sha256")
        != manifest["validation_protocol_sha256"]
        or not isinstance(model_manifest.get("models"), Mapping)
        or not set(model_manifest.get("models", ())).issubset(set(manifest["donors"]))
        or donor not in model_manifest["models"]
    ):
        raise ValueError(f"matched-ST model checkpoint is unavailable for {donor}")
    receipt = model_manifest["models"][donor]
    if not isinstance(receipt, Mapping):
        raise ValueError(f"matched-ST model receipt is malformed for {donor}")
    donor_input = manifest["donor_model_inputs"][donor]
    expected_identity = _model_identity(manifest, donor, donor_input)
    checkpoint_path = _validate_model_receipt(
        receipt,
        output=output,
        donor=donor,
        expected_identity=expected_identity,
        prepared_identity=str(manifest["prepared_identity"]),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"matched-ST model checkpoint payload is malformed for {donor}")
    _validate_model_checkpoint_payload(
        payload,
        donor=donor,
        expected_identity=expected_identity,
        prepared_identity=str(manifest["prepared_identity"]),
        manifest=manifest,
    )
    return payload, receipt


def _restore_models(
    runner: Any,
    core: Any,
    checkpoint: Mapping[str, object],
    device: str,
) -> tuple[Any, Any]:
    vae = runner._instantiate(
        core.CountVAE,
        n_genes=int(checkpoint["genes"]) + 1,
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint["vae_hidden_dim"]),
    )
    vae.load_state_dict(checkpoint["vae_state_dict"])
    vae.to(torch.device(device)).eval()
    state = runner._instantiate(
        core.CompositionStateModel,
        image_dim=int(checkpoint["image_dim"]),
        type_labels=tuple(str(value) for value in checkpoint["type_names"]),
        n_genes=int(checkpoint["genes"]) + 1,
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint["state_hidden_dim"]),
    )
    state.load_state_dict(checkpoint["state_state_dict"])
    state.to(torch.device(device)).eval()
    return vae, state


def _state_details(
    runner: Any,
    state: Any,
    image: np.ndarray,
    reference: object | None,
    mode: str,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    method = getattr(state, "predict_details_numpy", None)
    if method is None:
        raise RuntimeError("CompositionStateModel exposes no detailed predictor")
    value = runner._call_with_supported_kwargs(
        method,
        np.asarray(image, dtype=np.float32),
        reference=reference,
        reference_mixture=reference,
        mode=mode,
        device=device,
    )
    if not isinstance(value, Mapping):
        raise RuntimeError("CompositionStateModel details are malformed")
    return {
        name: (
            item.detach().cpu().numpy().astype(np.float32)
            if torch.is_tensor(item)
            else np.asarray(item, dtype=np.float32)
        )
        for name, item in value.items()
    }


def _reference_only_details(
    rows: int,
    reference: Any,
    type_names: Sequence[str],
    type_anchor_means: np.ndarray,
    type_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    names = tuple(str(value) for value in type_names)
    weights = np.asarray(type_weights, dtype=np.float64)
    anchors = np.asarray(type_anchor_means, dtype=np.float64)
    if weights.shape != (len(names),) or anchors.ndim != 2 or anchors.shape[0] != len(names):
        raise ValueError("reference-only type parameters are malformed")
    means = anchors.copy()
    variances = np.zeros_like(means)
    entropy = np.zeros(len(names), dtype=np.float64)
    for type_index, type_name in enumerate(names):
        component_index = np.flatnonzero(np.asarray(reference.type_labels) == type_name)
        if not len(component_index):
            if weights[type_index] > 0:
                raise ValueError("reference type weights include an unsupported type")
            continue
        local_weight = np.asarray(reference.weights[component_index], dtype=np.float64)
        local_weight /= local_weight.sum()
        local_mean = np.asarray(reference.means[component_index], dtype=np.float64)
        local_variance = np.asarray(reference.variances[component_index], dtype=np.float64)
        means[type_index] = np.sum(local_weight[:, None] * local_mean, axis=0)
        variances[type_index] = np.sum(
            local_weight[:, None]
            * (local_variance + (local_mean - means[type_index]) ** 2),
            axis=0,
        )
        entropy[type_index] = -np.sum(local_weight * np.log(np.maximum(local_weight, 1.0e-12)))
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("reference-only composition lacks positive mass")
    weights /= weights.sum()
    composition = np.repeat(weights[None], int(rows), axis=0)
    type_mean = np.repeat(means[None], int(rows), axis=0).astype(np.float32)
    type_variance = np.repeat(variances[None], int(rows), axis=0).astype(np.float32)
    latent = np.sum(composition[:, :, None] * type_mean, axis=1)
    return {
        "composition": composition.astype(np.float32),
        "type_mean": type_mean,
        "type_variance": type_variance,
        "latent": latent.astype(np.float32),
        "reference_entropy": np.repeat(entropy[None], int(rows), axis=0).astype(np.float32),
    }


def _component_posterior(
    image_details: Mapping[str, np.ndarray],
    output_details: Mapping[str, np.ndarray],
    reference: object | None,
    mode: str,
    type_names: Sequence[str],
    latent_dim: int,
) -> dict[str, np.ndarray]:
    names = tuple(str(value) for value in type_names)
    rows, types = np.asarray(output_details["composition"]).shape
    if mode in {"image_only", "composition_reference_mean"} or reference is None:
        maximum = 1
    else:
        maximum = max(
            1,
            max(
                np.count_nonzero(np.asarray(reference.type_labels) == name)
                for name in names
            ),
        )
    component_mean = np.zeros((rows, types, maximum, latent_dim), dtype=np.float32)
    component_variance = np.zeros_like(component_mean)
    component_weight = np.zeros((rows, types, maximum), dtype=np.float32)
    image_mean = np.asarray(image_details["type_mean"], dtype=np.float64)
    image_variance = np.asarray(image_details["type_variance"], dtype=np.float64)
    for type_index, type_name in enumerate(names):
        component_index = (
            np.asarray([], dtype=np.int64)
            if reference is None
            else np.flatnonzero(np.asarray(reference.type_labels) == type_name)
        )
        if mode == "full_poe" and len(component_index) >= 2:
            reference_mean = np.asarray(reference.means[component_index], dtype=np.float64)
            reference_variance = np.asarray(
                reference.variances[component_index], dtype=np.float64
            )
            natural_weight = np.asarray(reference.weights[component_index], dtype=np.float64)
            natural_weight /= natural_weight.sum()
            precision = 1.0 / image_variance[:, type_index, None] + 1.0 / reference_variance[None]
            local_variance = 1.0 / precision
            local_mean = local_variance * (
                image_mean[:, type_index, None] / image_variance[:, type_index, None]
                + reference_mean[None] / reference_variance[None]
            )
            overlap_variance = image_variance[:, type_index, None] + reference_variance[None]
            logits = np.log(natural_weight)[None] - 0.5 * np.sum(
                np.log(2 * np.pi * overlap_variance)
                + (image_mean[:, type_index, None] - reference_mean[None]) ** 2
                / overlap_variance,
                axis=2,
            )
            logits -= logits.max(axis=1, keepdims=True)
            local_weight = np.exp(logits)
            local_weight /= local_weight.sum(axis=1, keepdims=True)
            count = len(component_index)
            component_mean[:, type_index, :count] = local_mean.astype(np.float32)
            component_variance[:, type_index, :count] = local_variance.astype(np.float32)
            component_weight[:, type_index, :count] = local_weight.astype(np.float32)
        elif mode == "reference_only" and len(component_index):
            count = len(component_index)
            natural_weight = np.asarray(reference.weights[component_index], dtype=np.float64)
            natural_weight /= natural_weight.sum()
            component_mean[:, type_index, :count] = np.asarray(
                reference.means[component_index], dtype=np.float32
            )[None]
            component_variance[:, type_index, :count] = np.asarray(
                reference.variances[component_index], dtype=np.float32
            )[None]
            component_weight[:, type_index, :count] = natural_weight[None]
        else:
            component_mean[:, type_index, 0] = np.asarray(
                output_details["type_mean"], dtype=np.float32
            )[:, type_index]
            component_variance[:, type_index, 0] = np.asarray(
                output_details["type_variance"], dtype=np.float32
            )[:, type_index]
            component_weight[:, type_index, 0] = 1.0
    return {
        "composition": np.asarray(output_details["composition"], dtype=np.float32),
        "component_mean": component_mean,
        "component_variance": component_variance,
        "component_weight": component_weight,
    }


def _decode_type_mixture(
    vae: Any,
    details: Mapping[str, np.ndarray],
    dispersion: np.ndarray,
    *,
    latent_dim: int,
    genes: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    composition = np.asarray(details["composition"], dtype=np.float32)
    component_mean = np.asarray(details["component_mean"], dtype=np.float32)
    component_variance = np.asarray(details["component_variance"], dtype=np.float32)
    component_weight = np.asarray(details["component_weight"], dtype=np.float32)
    rows, types, components, _ = component_mean.shape
    method = getattr(vae, "decode_diagonal_gaussian_numpy", None)
    if method is None:
        raise RuntimeError("CountVAE lacks frozen posterior decoder integration")
    moments = method(
        component_mean.reshape(-1, latent_dim),
        component_variance.reshape(-1, latent_dim),
        library_size=np.ones(rows * types * components, dtype=np.float32),
        modality="st",
        endpoint_gene_indices=np.arange(genes),
        dispersion=dispersion,
        samples=32,
        batch_size=batch_size,
        seed=seed,
    )
    component_rate = np.asarray(moments["mean_counts"], dtype=np.float32).reshape(
        rows, types, components, genes
    )
    type_rate = np.sum(component_weight[..., None] * component_rate, axis=2)
    return np.sum(composition[:, :, None] * type_rate, axis=1).astype(np.float32)


def _baseline_reconstruction_check(
    generated: np.ndarray,
    frozen: np.ndarray,
    arm: str,
) -> Mapping[str, object]:
    left = np.asarray(generated, dtype=np.float64)
    right = np.asarray(frozen, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError(f"generated and frozen {arm} shapes differ")
    difference = np.abs(left - right)
    relative = difference / np.maximum(np.abs(right), 1.0e-8)
    diagnostic = {
        "passed": True,
        "rtol": 1.0e-5,
        "atol": 1.0e-6,
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "maximum_relative_error": float(relative.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()),
    }
    if not np.allclose(left, right, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(
            f"reconstructed {arm} differs from the frozen baseline: {diagnostic}"
        )
    return diagnostic


def _prediction_identity(
    manifest: Mapping[str, object],
    direction: Mapping[str, object],
    model_receipt: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "schema": PREDICTION_SCHEMA,
                "prepared_identity": manifest["prepared_identity"],
                "direction_id": direction["direction_id"],
                "predict_input_semantic_sha256": direction[
                    "predict_input_semantic_sha256"
                ],
                "model_identity": model_receipt["model_identity"],
                "checkpoint_sha256": model_receipt["checkpoint_sha256"],
                "validation_runner_sha256": manifest["validation_runner_sha256"],
                "validation_protocol_sha256": manifest["validation_protocol_sha256"],
                "arms": ARMS,
            }
        )
    ).hexdigest()


def _expected_s6_wrong_donors(
    checkpoint: Mapping[str, object],
    *,
    donor: str,
    indication: str,
) -> tuple[str, ...]:
    train_donors = np.asarray(checkpoint["train_donor_ids"]).astype(str)
    train_indications = np.asarray(checkpoint["train_indication_ids"]).astype(str)
    if train_donors.shape != train_indications.shape:
        raise ValueError("checkpoint training donor/indication identities are misaligned")
    return tuple(
        sorted(set(train_donors[train_indications == indication].tolist()) - {donor})
    )


def _validate_prediction_receipt(
    receipt: Mapping[str, object],
    *,
    output: Path,
    direction: Mapping[str, object],
    expected_identity: str,
    prepared_identity: str,
    model_receipt: Mapping[str, object],
) -> Path:
    output = _canonical_path(output)
    direction_id = str(direction["direction_id"])
    prediction_path = _require_canonical_path(
        receipt.get("prediction_path", ""),
        output / "directions" / direction_id / "predictions.npz",
        f"{direction_id} prediction",
    )
    receipt_path = _require_canonical_path(
        receipt.get("predict_receipt_path", ""),
        output / "directions" / direction_id / "predict_receipt.json",
        f"{direction_id} prediction receipt",
    )
    expected_values = {
        "schema": PREDICTION_SCHEMA,
        "direction_id": direction_id,
        "donor": direction["donor"],
        "indication": direction["indication"],
        "reference_section": direction["reference_section"],
        "query_section": direction["query_section"],
        "design_family": direction["design_family"],
        "evidence_label": direction["evidence_label"],
        "prediction_identity": expected_identity,
        "prepared_identity": prepared_identity,
        "predict_input_semantic_sha256": direction["predict_input_semantic_sha256"],
        "model_identity": model_receipt["model_identity"],
        "checkpoint_sha256": model_receipt["checkpoint_sha256"],
        "query_ST_opened": False,
        "reference_ST_opened": True,
        "directions_predicted_in_process": [direction_id],
        "process_isolation_rule_satisfied": True,
        "process_swap_kib_at_completion": 0,
    }
    if (
        any(receipt.get(key) != value for key, value in expected_values.items())
        or not np.isclose(
            float(receipt.get("guard_total_width_mm", math.inf)),
            float(direction["guard_total_width_mm"]),
        )
        or not receipt_path.is_file()
        or json.loads(receipt_path.read_text(encoding="utf-8")) != receipt
        or not prediction_path.is_file()
        or not str(receipt.get("prediction_semantic_sha256", ""))
    ):
        raise ValueError(f"prediction receipt is malformed for {direction_id}")
    return prediction_path


def predict_direction(
    args: argparse.Namespace,
    runner: Any,
    core: Any,
) -> Mapping[str, object]:
    protocol = _load_protocol(args.protocol)
    _verify_frozen_artifacts(protocol, args.baseline)
    if not args.direction:
        raise ValueError("--direction is required; one process may predict exactly one direction")
    manifest = _read_prepared_manifest(args.output, args.protocol, args.baseline)
    _validate_runtime_binding(args, manifest)
    if args.direction not in manifest["directions"]:
        raise ValueError(f"unknown matched-ST direction: {args.direction}")
    direction = manifest["directions"][args.direction]
    donor = str(direction["donor"])
    public = runner._verify_semantic_file(
        Path(str(direction["predict_input_path"])),
        str(direction["predict_input_semantic_sha256"]),
    )
    validate_direction_public(public)
    if _scalar_text(public["direction_id"]) != args.direction:
        raise ValueError("direction identity differs between manifest and prediction input")
    checkpoint, model_receipt = _load_model_checkpoint(args, manifest, donor)
    identity = _prediction_identity(manifest, direction, model_receipt)
    direction_dir = _canonical_path(args.output) / "directions" / args.direction
    prediction_path = direction_dir / "predictions.npz"
    receipt_path = direction_dir / "predict_receipt.json"
    if args.resume and prediction_path.is_file() and receipt_path.is_file():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(old, Mapping):
            raise ValueError(f"prediction receipt is malformed for {args.direction}")
        resumed_path = _validate_prediction_receipt(
            old,
            output=args.output,
            direction=direction,
            expected_identity=identity,
            prepared_identity=str(manifest["prepared_identity"]),
            model_receipt=model_receipt,
        )
        arrays = runner._verify_semantic_file(
            resumed_path,
            str(old["prediction_semantic_sha256"]),
        )
        _validate_prediction_artifact(
            arrays,
            public,
            old,
            direction,
            model_receipt,
            checkpoint,
            manifest,
        )
        return old

    vae, state = _restore_models(runner, core, checkpoint, args.device)
    genes = int(checkpoint["genes"])
    latent_dim = int(checkpoint["latent_dim"])
    seed = int(checkpoint["seed"])
    type_names = tuple(str(value) for value in checkpoint["type_names"])
    dispersion = np.asarray(checkpoint["dispersion"], dtype=np.float32)
    baseline_dispersion = np.asarray(public["baseline_training_dispersion"], dtype=np.float32)
    if not np.array_equal(dispersion, baseline_dispersion):
        raise RuntimeError("refitted training-only dispersion differs from the frozen baseline")
    anchors = np.asarray(checkpoint["type_anchor_means"], dtype=np.float32)
    signatures = np.asarray(checkpoint["type_signatures"], dtype=np.float32)
    adapter = protocol["spatial_reference_adapter"]

    reference_augmented = runner._append_other_count_bin(
        np.asarray(public["reference_st_counts"], dtype=np.float32),
        np.asarray(public["reference_st_library"], dtype=np.float32),
    )
    reference_latent = runner._encode(
        vae,
        reference_augmented,
        modality="st",
        device=args.device,
    )
    reference_composition = composition_proxy_from_signatures(
        np.asarray(public["reference_st_counts"]),
        signatures,
        iterations=int(adapter["type_proxy_iterations"]),
    )
    matched_st, matched_st_type_weights, matched_diagnostics = build_spatial_reference_mixture(
        core,
        reference_latent,
        reference_composition,
        type_names,
        np.asarray(public["reference_spot_ids"]),
        donor,
        seed=seed,
        components=int(adapter["components_per_supported_type"]),
        iterations=int(adapter["component_iterations"]),
        temperature=float(adapter["component_temperature"]),
        variance_floor=float(adapter["component_variance_floor"]),
        minimum_mass=float(adapter["minimum_type_proxy_mass"]),
        minimum_effective_sample_size=float(
            adapter["minimum_type_proxy_effective_sample_size"]
        ),
    )
    matched_st.assert_no_outcome_overlap(np.asarray(public["query_spot_ids"]).astype(str))

    train_latent = np.asarray(checkpoint["train_st_latent"], dtype=np.float32)
    train_proxy = np.asarray(checkpoint["train_st_composition_proxy"], dtype=np.float32)
    train_spot_ids = np.asarray(checkpoint["train_spot_ids"]).astype(str)
    train_donors = np.asarray(checkpoint["train_donor_ids"]).astype(str)
    train_indications = np.asarray(checkpoint["train_indication_ids"]).astype(str)
    indication = _scalar_text(public["indication"])
    wrong_donors = tuple(
        sorted(
            set(train_donors[train_indications == indication].tolist()) - {donor}
        )
    )
    if not wrong_donors:
        raise RuntimeError("matched-ST S6 lacks a same-indication wrong donor")
    wrong_mixtures: dict[str, Any] = {}
    wrong_diagnostics: dict[str, object] = {}
    for wrong_donor in wrong_donors:
        keep = train_donors == wrong_donor
        mixture, _, diagnostics = build_spatial_reference_mixture(
            core,
            train_latent[keep],
            train_proxy[keep],
            type_names,
            train_spot_ids[keep],
            wrong_donor,
            seed=seed,
            components=int(adapter["components_per_supported_type"]),
            iterations=int(adapter["component_iterations"]),
            temperature=float(adapter["component_temperature"]),
            variance_floor=float(adapter["component_variance_floor"]),
            minimum_mass=float(adapter["minimum_type_proxy_mass"]),
            minimum_effective_sample_size=float(
                adapter["minimum_type_proxy_effective_sample_size"]
            ),
        )
        wrong_mixtures[wrong_donor] = mixture
        wrong_diagnostics[wrong_donor] = diagnostics
    generic_mixture, _, generic_diagnostics = build_spatial_reference_mixture(
        core,
        train_latent,
        train_proxy,
        type_names,
        train_spot_ids,
        "query_excluded_pooled_training_ST",
        seed=seed,
        components=int(adapter["components_per_supported_type"]),
        iterations=int(adapter["component_iterations"]),
        temperature=float(adapter["component_temperature"]),
        variance_floor=float(adapter["component_variance_floor"]),
        minimum_mass=float(adapter["minimum_type_proxy_mass"]),
        minimum_effective_sample_size=float(
            adapter["minimum_type_proxy_effective_sample_size"]
        ),
    )

    matched_sc_latent = np.asarray(checkpoint["matched_sc_latent"], dtype=np.float32)
    matched_sc_types_all = np.asarray(checkpoint["matched_sc_type_ids"]).astype(str)
    matched_sc_keep = np.isin(matched_sc_types_all, type_names)
    matched_sc = runner._call_with_supported_kwargs(
        core.build_reference_mixture,
        matched_sc_latent[matched_sc_keep],
        type_ids=matched_sc_types_all[matched_sc_keep],
        donor_ids=np.asarray(checkpoint["matched_sc_donor_ids"]).astype(str)[matched_sc_keep],
        observation_ids=np.asarray(checkpoint["matched_sc_cell_ids"]).astype(str)[
            matched_sc_keep
        ],
        source_modality="single_cell",
        n_components=3,
        seed=seed,
    )
    matched_sc_type_names = tuple(matched_sc.type_names)
    natural_sc = dict(zip(matched_sc_type_names, matched_sc.type_weights()))
    matched_sc_type_weights = np.asarray(
        [natural_sc.get(name, 0.0) for name in type_names], dtype=np.float32
    )

    query_image = np.asarray(public["query_image"], dtype=np.float32)
    image_m0 = _state_details(
        runner,
        state,
        query_image,
        None,
        "image_only",
        device=args.device,
    )
    query_composition = np.asarray(image_m0["composition"], dtype=np.float32)
    direction_seed = _stable_seed(args.direction, seed) & 0xFFFFFFFF
    shuffle_index, composition_strata = runner._composition_stratified_derangement(
        np.asarray(public["query_section_ids"]),
        query_composition,
        int(direction_seed),
    )
    shuffled_image = _state_details(
        runner,
        state,
        query_image[shuffle_index],
        None,
        "image_only",
        device=args.device,
    )

    output_s1 = _reference_only_details(
        len(query_image),
        matched_st,
        type_names,
        anchors,
        matched_st_type_weights,
    )
    output_s3 = _state_details(
        runner,
        state,
        query_image,
        matched_st,
        "full_poe",
        device=args.device,
    )
    output_s4 = _state_details(
        runner,
        state,
        query_image[shuffle_index],
        matched_st,
        "full_poe",
        device=args.device,
    )
    output_s7 = _state_details(
        runner,
        state,
        query_image,
        generic_mixture,
        "full_poe",
        device=args.device,
    )
    output_s6 = {
        wrong_donor: _state_details(
            runner,
            state,
            query_image,
            mixture,
            "full_poe",
            device=args.device,
        )
        for wrong_donor, mixture in wrong_mixtures.items()
    }
    output_m1 = _reference_only_details(
        len(query_image),
        matched_sc,
        type_names,
        anchors,
        matched_sc_type_weights,
    )
    output_m3 = _state_details(
        runner,
        state,
        query_image,
        matched_sc,
        "full_poe",
        device=args.device,
    )

    posterior = {
        "M0": _component_posterior(
            image_m0, image_m0, None, "image_only", type_names, latent_dim
        ),
        "M1": _component_posterior(
            image_m0, output_m1, matched_sc, "reference_only", type_names, latent_dim
        ),
        "M3": _component_posterior(
            image_m0, output_m3, matched_sc, "full_poe", type_names, latent_dim
        ),
        "S1": _component_posterior(
            image_m0, output_s1, matched_st, "reference_only", type_names, latent_dim
        ),
        "S3": _component_posterior(
            image_m0, output_s3, matched_st, "full_poe", type_names, latent_dim
        ),
        "S4": _component_posterior(
            shuffled_image, output_s4, matched_st, "full_poe", type_names, latent_dim
        ),
        "S7": _component_posterior(
            image_m0, output_s7, generic_mixture, "full_poe", type_names, latent_dim
        ),
    }
    posterior_s6 = {
        wrong_donor: _component_posterior(
            image_m0,
            output_s6[wrong_donor],
            wrong_mixtures[wrong_donor],
            "full_poe",
            type_names,
            latent_dim,
        )
        for wrong_donor in wrong_donors
    }
    decode_seed = seed + 60_000
    generated = {
        arm: _decode_type_mixture(
            vae,
            details,
            dispersion,
            latent_dim=latent_dim,
            genes=genes,
            seed=decode_seed,
            batch_size=256,
        )
        for arm, details in posterior.items()
    }
    generated_s6 = np.stack(
        [
            _decode_type_mixture(
                vae,
                posterior_s6[wrong_donor],
                dispersion,
                latent_dim=latent_dim,
                genes=genes,
                seed=decode_seed,
                batch_size=256,
            )
            for wrong_donor in wrong_donors
        ],
        axis=0,
    )
    reconstruction = {
        arm: _baseline_reconstruction_check(
            generated[arm],
            np.asarray(public[f"baseline_rate_{arm}"]),
            arm,
        )
        for arm in BASELINE_ARMS
    }
    predictions: dict[str, object] = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "direction_id": np.asarray(args.direction),
        "donor": np.asarray(donor),
        "indication": np.asarray(_scalar_text(public["indication"])),
        "design_family": np.asarray(_scalar_text(public["design_family"])),
        "evidence_label": np.asarray(_scalar_text(public["evidence_label"])),
        "guard_total_width_mm": np.asarray(public["guard_total_width_mm"]),
        "reference_section": np.asarray(_scalar_text(public["reference_section"])),
        "query_section": np.asarray(_scalar_text(public["query_section"])),
        "query_spot_ids": np.asarray(public["query_spot_ids"]),
        "query_section_ids": np.asarray(public["query_section_ids"]),
        "query_indication_ids": np.asarray(public["query_indication_ids"]),
        "gene_ids": np.asarray(public["gene_ids"]),
        "training_only_dispersion": dispersion,
        "prediction_scale": np.asarray("per_unit_actual_ST_library_rate"),
        "rate_S0": np.asarray(public["baseline_rate_M0"], dtype=np.float32),
        "rate_S1": generated["S1"],
        "rate_S3": generated["S3"],
        "rate_S4": generated["S4"],
        "rate_S6_candidates": generated_s6,
        "rate_S7": generated["S7"],
        "rate_M1": np.asarray(public["baseline_rate_M1"], dtype=np.float32),
        "rate_M3": np.asarray(public["baseline_rate_M3"], dtype=np.float32),
        "S6_wrong_donor_ids": np.asarray(wrong_donors),
        "query_shuffle_index": np.asarray(shuffle_index, dtype=np.int64),
        "query_composition_stratum": np.asarray(composition_strata, dtype=np.int64),
        "matched_ST_type_proxy": reference_composition,
        "matched_ST_type_names": np.asarray(type_names),
        "matched_ST_type_weights": matched_st_type_weights,
        "matched_ST_adapter_diagnostics_json": np.asarray(
            json.dumps(_safe(matched_diagnostics), sort_keys=True)
        ),
        "S6_adapter_diagnostics_json": np.asarray(
            json.dumps(_safe(wrong_diagnostics), sort_keys=True)
        ),
        "S7_adapter_diagnostics_json": np.asarray(
            json.dumps(_safe(generic_diagnostics), sort_keys=True)
        ),
        "baseline_reconstruction_json": np.asarray(
            json.dumps(_safe(reconstruction), sort_keys=True)
        ),
        "model_identity": np.asarray(str(model_receipt["model_identity"])),
        "model_checkpoint_sha256": np.asarray(str(model_receipt["checkpoint_sha256"])),
        "prediction_identity": np.asarray(identity),
        "prepared_identity": np.asarray(str(manifest["prepared_identity"])),
        "query_ST_opened": np.asarray(False),
        "UNI2_h_run": np.asarray(False),
    }
    for arm in ("S0", "S1", "S3", "S4", "S7", "M1", "M3"):
        rate = np.asarray(predictions[f"rate_{arm}"])
        if rate.shape != (len(query_image), genes) or np.any(~np.isfinite(rate)) or np.any(
            rate <= 0
        ):
            raise RuntimeError(f"matched-ST prediction arm {arm} is malformed")
    _atomic_npz(prediction_path, predictions)
    receipt = {
        "schema": PREDICTION_SCHEMA,
        "direction_id": args.direction,
        "donor": donor,
        "indication": _scalar_text(public["indication"]),
        "prediction_identity": identity,
        "prepared_identity": manifest["prepared_identity"],
        "prediction_path": str(prediction_path.resolve()),
        "predict_receipt_path": str(receipt_path.resolve()),
        "prediction_semantic_sha256": runner._semantic_array_hash(predictions),
        "predict_input_semantic_sha256": direction["predict_input_semantic_sha256"],
        "model_identity": model_receipt["model_identity"],
        "checkpoint_sha256": model_receipt["checkpoint_sha256"],
        "query_ST_opened": False,
        "reference_ST_opened": True,
        "directions_predicted_in_process": [args.direction],
        "process_isolation_rule_satisfied": True,
        "process_swap_kib_at_completion": _assert_zero_process_swap(),
        "S6_S7_fixed_support": False,
        "design_family": _scalar_text(public["design_family"]),
        "evidence_label": _scalar_text(public["evidence_label"]),
        "guard_total_width_mm": float(public["guard_total_width_mm"]),
        "reference_section": _scalar_text(public["reference_section"]),
        "query_section": _scalar_text(public["query_section"]),
        "measurement_floor_claim": False,
        "independent_confirmation_claim": False,
    }
    _atomic_json(receipt_path, receipt)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return receipt


def _validate_prediction_artifact(
    predictions: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    receipt: Mapping[str, object],
    direction: Mapping[str, object],
    model_receipt: Mapping[str, object],
    checkpoint: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    direction_id = str(direction["direction_id"])
    expected_identity = _prediction_identity(manifest, direction, model_receipt)
    if (
        receipt.get("schema") != PREDICTION_SCHEMA
        or receipt.get("direction_id") != direction_id
        or receipt.get("prepared_identity") != manifest["prepared_identity"]
        or receipt.get("prediction_identity") != expected_identity
        or receipt.get("query_ST_opened") is not False
        or receipt.get("directions_predicted_in_process") != [direction_id]
        or receipt.get("process_isolation_rule_satisfied") is not True
    ):
        raise ValueError(f"prediction receipt is malformed for {direction_id}")
    if (
        _scalar_text(predictions["schema"]) != PREDICTION_SCHEMA
        or _scalar_text(predictions["direction_id"]) != direction_id
        or _scalar_text(predictions["prepared_identity"])
        != str(manifest["prepared_identity"])
        or _scalar_text(predictions["prediction_identity"]) != expected_identity
        or _scalar_text(predictions["model_identity"])
        != str(model_receipt["model_identity"])
        or _scalar_text(predictions["model_checkpoint_sha256"])
        != str(model_receipt["checkpoint_sha256"])
        or bool(np.asarray(predictions["query_ST_opened"]))
        or bool(np.asarray(predictions["UNI2_h_run"]))
    ):
        raise ValueError(f"prediction identity is malformed for {direction_id}")
    scalar_bindings = {
        "donor": direction["donor"],
        "indication": direction["indication"],
        "reference_section": direction["reference_section"],
        "query_section": direction["query_section"],
        "design_family": direction["design_family"],
        "evidence_label": direction["evidence_label"],
    }
    if any(
        _scalar_text(predictions[key]) != str(value)
        or _scalar_text(public[key]) != str(value)
        for key, value in scalar_bindings.items()
    ) or not np.isclose(
        float(predictions["guard_total_width_mm"]),
        float(direction["guard_total_width_mm"]),
    ) or not np.isclose(
        float(public["guard_total_width_mm"]),
        float(direction["guard_total_width_mm"]),
    ):
        raise ValueError(f"prediction semantic binding differs for {direction_id}")
    for key in (
        "query_spot_ids",
        "query_section_ids",
        "query_indication_ids",
        "gene_ids",
    ):
        if not np.array_equal(predictions[key], public[key]):
            raise ValueError(f"prediction/public {key} order differs for {direction_id}")
    rows = len(public["query_spot_ids"])
    genes = len(public["gene_ids"])
    for arm in ("S0", "S1", "S3", "S4", "S7", "M1", "M3"):
        rate = np.asarray(predictions[f"rate_{arm}"])
        if rate.shape != (rows, genes) or np.any(~np.isfinite(rate)) or np.any(rate <= 0):
            raise ValueError(f"prediction rate {arm} is malformed for {direction_id}")
    wrong = np.asarray(predictions["S6_wrong_donor_ids"]).astype(str)
    expected_wrong = _expected_s6_wrong_donors(
        checkpoint,
        donor=str(direction["donor"]),
        indication=_scalar_text(public["indication"]),
    )
    rate_s6 = np.asarray(predictions["rate_S6_candidates"])
    if (
        not expected_wrong
        or tuple(wrong.tolist()) != expected_wrong
        or rate_s6.shape != (len(wrong), rows, genes)
        or np.any(~np.isfinite(rate_s6))
        or np.any(rate_s6 <= 0)
    ):
        raise ValueError(f"prediction S6 candidates are malformed for {direction_id}")
    shuffle = np.asarray(predictions["query_shuffle_index"], dtype=np.int64)
    if (
        shuffle.shape != (rows,)
        or np.any(shuffle < 0)
        or np.any(shuffle >= rows)
        or np.any(shuffle == np.arange(rows))
        or not np.array_equal(np.sort(shuffle), np.arange(rows))
    ):
        raise ValueError(f"prediction S4 derangement is malformed for {direction_id}")
    dispersion = np.asarray(predictions["training_only_dispersion"])
    if dispersion.shape != (genes,) or np.any(~np.isfinite(dispersion)) or np.any(
        dispersion <= 0
    ):
        raise ValueError(f"prediction dispersion is malformed for {direction_id}")
    reconstruction = json.loads(_scalar_text(predictions["baseline_reconstruction_json"]))
    if not isinstance(reconstruction, Mapping) or set(reconstruction) != set(BASELINE_ARMS):
        raise ValueError(f"baseline reconstruction receipt is missing for {direction_id}")
    for arm in BASELINE_ARMS:
        if not np.array_equal(
            np.asarray(predictions[f"rate_{'S0' if arm == 'M0' else arm}"]),
            np.asarray(public[f"baseline_rate_{arm}"]),
        ):
            raise ValueError(f"saved {arm} comparator differs from frozen public baseline")
        diagnostic = reconstruction[arm]
        if (
            not isinstance(diagnostic, Mapping)
            or diagnostic.get("passed") is not True
            or diagnostic.get("rtol") != 1.0e-5
            or diagnostic.get("atol") != 1.0e-6
            or any(
                not np.isfinite(float(diagnostic.get(name, math.nan)))
                or float(diagnostic.get(name, -1.0)) < 0
                for name in (
                    "maximum_absolute_error",
                    "maximum_relative_error",
                    "mean_absolute_error",
                )
            )
        ):
            raise ValueError(f"baseline reconstruction receipt is invalid for {arm}")


def _validate_score_target(
    secret: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    direction: Mapping[str, object],
) -> None:
    direction_id = str(direction["direction_id"])
    scalar_bindings = {
        "direction_id": direction_id,
        "donor": direction["donor"],
        "indication": direction["indication"],
        "query_section": direction["query_section"],
        "design_family": direction["design_family"],
        "evidence_label": direction["evidence_label"],
    }
    if _scalar_text(secret["schema"]) != PREPARED_SCHEMA or any(
        _scalar_text(secret[key]) != str(value) for key, value in scalar_bindings.items()
    ):
        raise ValueError(f"score target identity is malformed for {direction_id}")
    if not np.isclose(
        float(secret["guard_total_width_mm"]),
        float(direction["guard_total_width_mm"]),
    ):
        raise ValueError(f"score target guard identity differs for {direction_id}")
    for key in (
        "query_spot_ids",
        "query_section_ids",
        "query_indication_ids",
        "gene_ids",
    ):
        if not np.array_equal(secret[key], public[key]) or not np.array_equal(
            secret[key], predictions[key]
        ):
            raise ValueError(f"score target {key} order differs for {direction_id}")
    rows = len(np.asarray(public["query_spot_ids"]))
    genes = len(np.asarray(public["gene_ids"]))
    section_ids = np.asarray(secret["query_section_ids"]).astype(str)
    indication_ids = np.asarray(secret["query_indication_ids"]).astype(str)
    if (
        section_ids.shape != (rows,)
        or indication_ids.shape != (rows,)
        or set(section_ids.tolist()) != {str(direction["query_section"])}
        or set(indication_ids.tolist()) != {_scalar_text(public["indication"])}
    ):
        raise ValueError(f"score target section/indication rows differ for {direction_id}")
    counts = np.asarray(secret["query_st_counts"])
    library = np.asarray(secret["query_st_library"])
    eligibility_raw = np.asarray(secret["primary_score_eligible"])
    if (
        counts.shape != (rows, genes)
        or library.shape != (rows,)
        or eligibility_raw.shape != (rows,)
        or eligibility_raw.dtype.kind != "b"
        or not np.issubdtype(counts.dtype, np.number)
        or not np.issubdtype(library.dtype, np.number)
    ):
        raise ValueError(f"score target count/library shapes are malformed for {direction_id}")
    counts_float = counts.astype(np.float64)
    library_float = library.astype(np.float64)
    eligibility = eligibility_raw.astype(bool)
    if (
        np.any(~np.isfinite(counts_float))
        or np.any(counts_float < 0)
        or np.any(counts_float != np.floor(counts_float))
        or np.any(~np.isfinite(library_float))
        or np.any(library_float < 0)
        or np.any(counts_float.sum(axis=1) > library_float + 1.0e-6)
        or not np.array_equal(eligibility, library_float > 0)
    ):
        raise ValueError(f"score target count/library values are malformed for {direction_id}")
    minimum = 100 if str(direction["design_family"]) == "same_section_upper_bound" else 1
    if int(eligibility.sum()) < minimum:
        raise ValueError(f"score target has too few eligible rows for {direction_id}")


def _score_direction(
    runner: Any,
    core: Any,
    secret: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
) -> Mapping[str, object]:
    if _scalar_text(secret["direction_id"]) != _scalar_text(predictions["direction_id"]):
        raise ValueError("prediction and score target directions differ")
    if not np.array_equal(secret["query_spot_ids"], predictions["query_spot_ids"]):
        raise ValueError("prediction and score target spots differ")
    keep = np.asarray(secret["primary_score_eligible"], dtype=bool)
    counts = np.asarray(secret["query_st_counts"], dtype=np.float32)[keep]
    library = np.asarray(secret["query_st_library"], dtype=np.float32)[keep]
    if not len(counts) or np.any(library <= 0):
        raise ValueError("matched-ST score target has no positive-depth rows")
    theta = np.asarray(predictions["training_only_dispersion"], dtype=np.float32)
    losses: dict[str, float] = {}
    row_losses: dict[str, list[float]] = {}
    for arm in ("S0", "S1", "S3", "S4", "S7", "M1", "M3"):
        rate = np.asarray(predictions[f"rate_{arm}"], dtype=np.float32)[keep]
        rows = runner._nb_deviance_rows(core, counts, rate * library[:, None], theta)
        losses[arm] = float(np.mean(rows))
        row_losses[arm] = rows.tolist()
    candidate_rates = np.asarray(predictions["rate_S6_candidates"], dtype=np.float32)[:, keep]
    candidate_rows = [
        runner._nb_deviance_rows(core, counts, rate * library[:, None], theta)
        for rate in candidate_rates
    ]
    candidate_losses = np.asarray([np.mean(value) for value in candidate_rows])
    donor_equal_rows = np.mean(np.stack(candidate_rows, axis=0), axis=0)
    losses["S6"] = float(np.mean(candidate_losses))
    row_losses["S6"] = donor_equal_rows.tolist()
    return {
        "direction_id": _scalar_text(secret["direction_id"]),
        "donor": _scalar_text(secret["donor"]),
        "query_section": _scalar_text(secret["query_section"]),
        "design_family": _scalar_text(predictions["design_family"]),
        "evidence_label": _scalar_text(predictions["evidence_label"]),
        "guard_total_width_mm": float(predictions["guard_total_width_mm"]),
        "scored_spots": int(keep.sum()),
        "zero_depth_excluded": int((~keep).sum()),
        "mean_nb_deviance": losses,
        "row_nb_deviance": row_losses,
        "S6_control": {
            "aggregation": "equal_mean_over_same_indication_wrong_donor_losses",
            "wrong_donor_ids": np.asarray(predictions["S6_wrong_donor_ids"])
            .astype(str)
            .tolist(),
            "candidate_mean_nb_deviance": candidate_losses.tolist(),
            "fixed_effective_sample_size_or_type_support": False,
            "specificity_claim_allowed": False,
        },
        "matched_ST_adapter": json.loads(
            _scalar_text(predictions["matched_ST_adapter_diagnostics_json"])
        ),
        "baseline_reconstruction": json.loads(
            _scalar_text(predictions["baseline_reconstruction_json"])
        ),
    }


def _sign_flip_payload(core: Any, effects: Sequence[float]) -> Mapping[str, object]:
    values = np.asarray(effects, dtype=np.float64)
    result = core.exact_sign_flip_test(values, alternative="greater")
    return {
        "observed_mean": float(result.statistic),
        "p_value": float(result.p_value),
        "confidence_interval": [float(value) for value in result.confidence_interval],
        "positive_fraction": float(result.positive_fraction),
        "inference_units": int(result.donors),
    }


def _aggregate_comparisons(
    core: Any,
    donor_losses: Mapping[str, Mapping[str, float]],
) -> Mapping[str, object]:
    definitions = {
        "S3_vs_S1_H_and_E_conditional_value": ("S1", "S3"),
        "S3_vs_S4_exact_image_pairing_value": ("S4", "S3"),
        "S3_vs_S6_natural_wrong_donor_control": ("S6", "S3"),
        "S3_vs_S7_natural_pooled_control": ("S7", "S3"),
        "M3_minus_S3_cross_assay_penalty": ("M3", "S3"),
        "M1_minus_S1_reference_only_cross_assay_penalty": ("M1", "S1"),
        "S3_vs_S0_incremental_reference_value": ("S0", "S3"),
    }
    result: dict[str, object] = {}
    donors = sorted(donor_losses)
    for name, (left, right) in definitions.items():
        effects = np.asarray(
            [donor_losses[donor][left] - donor_losses[donor][right] for donor in donors],
            dtype=np.float64,
        )
        result[name] = {
            "positive_favors_second_named_candidate": True,
            "left_arm": left,
            "right_arm": right,
            "donor_order": donors,
            "donor_effect": effects.tolist(),
            "mean_effect": float(effects.mean()),
            "median_effect": float(np.median(effects)),
            "exact_sign_flip": _sign_flip_payload(core, effects),
        }
    return result


def _aggregate_family(
    core: Any,
    fold_reports: Mapping[str, Mapping[str, object]],
    *,
    design_family: str,
    guard_mm: float | None,
) -> Mapping[str, object]:
    selected = {
        direction_id: report
        for direction_id, report in fold_reports.items()
        if report["design_family"] == design_family
        and (
            guard_mm is None
            or np.isclose(float(report["guard_total_width_mm"]), float(guard_mm))
        )
    }
    if not selected:
        raise ValueError(f"no scored directions are available for {design_family}/{guard_mm}")
    donor_directions: dict[str, list[Mapping[str, object]]] = {}
    for report in selected.values():
        donor_directions.setdefault(str(report["donor"]), []).append(report)
    donor_losses = {
        donor: {
            arm: float(np.mean([item["mean_nb_deviance"][arm] for item in reports]))
            for arm in ARMS
        }
        for donor, reports in sorted(donor_directions.items())
    }
    direction_counts = {donor: len(reports) for donor, reports in donor_directions.items()}
    return {
        "design_family": design_family,
        "guard_total_width_mm": guard_mm,
        "donor_count": len(donor_losses),
        "direction_count": len(selected),
        "directions_per_donor": direction_counts,
        "donor_mean_nb_deviance": donor_losses,
        "donor_balanced_mean_nb_deviance": {
            arm: float(np.mean([values[arm] for values in donor_losses.values()]))
            for arm in ARMS
        },
        "comparisons": _aggregate_comparisons(core, donor_losses),
    }


def score(args: argparse.Namespace, runner: Any, core: Any) -> Mapping[str, object]:
    protocol = _load_protocol(args.protocol)
    _verify_frozen_artifacts(protocol, args.baseline)
    manifest = _read_prepared_manifest(args.output, args.protocol, args.baseline)
    _validate_runtime_binding(args, manifest)
    family_filter = {
        "adjacent": {"adjacent_section_primary"},
        "upper": {"same_section_upper_bound"},
        "all": {"adjacent_section_primary", "same_section_upper_bound"},
    }[args.score_family]
    selected_directions = {
        direction_id: direction
        for direction_id, direction in manifest["directions"].items()
        if str(direction["design_family"]) in family_filter
    }
    if not selected_directions:
        raise ValueError("score family selects no prepared matched-ST directions")

    # Complete target-free preflight for the selected family before any score
    # target is opened.
    preflight_receipts: dict[str, Mapping[str, object]] = {}
    for direction_id, direction in sorted(selected_directions.items()):
        donor = str(direction["donor"])
        checkpoint, model_receipt = _load_model_checkpoint(args, manifest, donor)
        public = runner._verify_semantic_file(
            Path(str(direction["predict_input_path"])),
            str(direction["predict_input_semantic_sha256"]),
        )
        validate_direction_public(public)
        receipt_path = args.output / "directions" / direction_id / "predict_receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"prediction receipt is missing for {direction_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise ValueError(f"prediction receipt is malformed for {direction_id}")
        prediction_path = _validate_prediction_receipt(
            receipt,
            output=args.output,
            direction=direction,
            expected_identity=_prediction_identity(manifest, direction, model_receipt),
            prepared_identity=str(manifest["prepared_identity"]),
            model_receipt=model_receipt,
        )
        predictions = runner._verify_semantic_file(
            prediction_path,
            str(receipt.get("prediction_semantic_sha256", "")),
        )
        _validate_prediction_artifact(
            predictions,
            public,
            receipt,
            direction,
            model_receipt,
            checkpoint,
            manifest,
        )
        preflight_receipts[direction_id] = receipt
        del public, predictions, checkpoint

    # The score-only manifest itself is not reachable until every prediction
    # in the selected score family has passed the complete target-free preflight.
    target_manifest = _read_score_target_manifest(args.output, manifest)

    fold_reports: dict[str, object] = {}
    for direction_id, direction in sorted(selected_directions.items()):
        public = runner._verify_semantic_file(
            Path(str(direction["predict_input_path"])),
            str(direction["predict_input_semantic_sha256"]),
        )
        # Revalidate immediately before opening this direction's score target.
        validate_direction_public(public)
        checkpoint, model_receipt = _load_model_checkpoint(
            args,
            manifest,
            str(direction["donor"]),
        )
        receipt = preflight_receipts[direction_id]
        prediction_path = _validate_prediction_receipt(
            receipt,
            output=args.output,
            direction=direction,
            expected_identity=_prediction_identity(manifest, direction, model_receipt),
            prepared_identity=str(manifest["prepared_identity"]),
            model_receipt=model_receipt,
        )
        predictions = runner._verify_semantic_file(
            prediction_path,
            str(receipt["prediction_semantic_sha256"]),
        )
        _validate_prediction_artifact(
            predictions,
            public,
            receipt,
            direction,
            model_receipt,
            checkpoint,
            manifest,
        )
        target_receipt = target_manifest["directions"][direction_id]
        secret = runner._verify_semantic_file(
            Path(str(target_receipt["score_target_path"])),
            str(target_receipt["score_target_semantic_sha256"]),
        )
        _validate_score_target(secret, public, predictions, direction)
        report = _score_direction(runner, core, secret, predictions)
        fold_reports[direction_id] = report
        _atomic_json(
            args.output / "directions" / direction_id / "score_report.json",
            _safe(report),
        )

    aggregate: dict[str, object] = {}
    if "adjacent_section_primary" in family_filter:
        adjacent = _aggregate_family(
            core,
            fold_reports,
            design_family="adjacent_section_primary",
            guard_mm=None,
        )
        if adjacent["donor_count"] != 2 or adjacent["direction_count"] != 4:
            raise RuntimeError("adjacent-section primary must contain four directions/two donors")
        aggregate["adjacent_section_primary"] = adjacent
    if "same_section_upper_bound" in family_filter:
        aggregate["same_section_upper_bound"] = {
            f"guard_{guard:g}mm": _aggregate_family(
                core,
                fold_reports,
                design_family="same_section_upper_bound",
                guard_mm=float(guard),
            )
            for guard in (1.0, 2.0)
        }
    report = {
        "schema": REPORT_SCHEMA,
        "analysis_scope": "outcome_exposed_mechanistic_diagnostic_only",
        "score_family": args.score_family,
        "image_encoder": "bioptimus/H-optimus-1",
        "image_encoder_revision": "3592cb220dec7a150c5d7813fb56e68bd57473b9",
        "UNI2_h_run": False,
        "frozen_gene_count": 256,
        "frozen_latent_dimension": 20,
        "folds": fold_reports,
        "aggregate": aggregate,
        "evidence_boundary": {
            "adjacent_section_primary": (
                "same_block_serial_sections_on_distinct_Visium_slides;_"
                "distributional_not_spot_paired;_two_donor_mechanistic_diagnostic"
            ),
            "same_section_upper_bound": (
                "optimistic_same_section_same_assay_same_batch_coordinate_PC1_tail_"
                "split_with_empty_guard"
            ),
            "independent_confirmation": False,
            "paired_technical_replicate": False,
            "measurement_floor": False,
            "cell_level": False,
        },
        "natural_control_boundary": {
            "S6_S7_training_ST_rows_available_to_model_fit": True,
            "fixed_effective_sample_size": False,
            "fixed_type_support": False,
            "personalization_specificity_claim": False,
        },
        "artifact_identities": {
            "prepared_identity": manifest["prepared_identity"],
            "score_target_manifest_identity": target_manifest[
                "score_target_manifest_identity"
            ],
            "validation_runner_sha256": manifest["validation_runner_sha256"],
            "validation_protocol_sha256": manifest["validation_protocol_sha256"],
            "frozen_runner_sha256": manifest["frozen_runner_sha256"],
            "frozen_core_sha256": manifest["frozen_core_sha256"],
            "prepared_manifest_sha256": _sha256(args.output / "prepared_manifest.json"),
            "score_target_manifest_sha256": _sha256(
                args.output / "score_target_manifest.json"
            ),
            "model_manifest_sha256": _sha256(args.output / "model_manifest.json"),
        },
        "scientific_authorization": "none",
        "iterative_refinement_run": False,
        "preflight_scope": (
            "all_predictions_in_selected_score_family_before_any_selected_"
            "target_manifest_or_target_open"
        ),
    }
    report_name = {
        "adjacent": "report.json",
        "upper": "same_section_upper_bound_report.json",
        "all": "combined_report.json",
    }[args.score_family]
    _atomic_json(args.output / report_name, _safe(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "fit-model", "predict", "score"),
        required=True,
        help="run stages in separate processes; there is intentionally no all-in-one mode",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.60)
    parser.add_argument(
        "--donor",
        help="fit one donor model; default fits prepared donors serially",
    )
    parser.add_argument(
        "--direction",
        help="predict exactly one prepared direction in this process",
    )
    parser.add_argument(
        "--score-family",
        choices=("adjacent", "upper", "all"),
        default="adjacent",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.cpu_threads) != 4:
        raise ValueError("the frozen matched-ST validation uses exactly four CPU threads")
    if float(args.gpu_memory_fraction) != 0.60:
        raise ValueError("the frozen matched-ST validation uses GPU memory fraction 0.60")
    if str(args.device) != "cuda:0":
        raise ValueError("the real matched-ST validation is bound to one visible GPU at cuda:0")
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible) != 1 or visible[0] == "-1":
        raise ValueError("matched-ST validation requires exactly one visible CUDA GPU")
    if args.stage == "predict" and not args.direction:
        raise ValueError("--direction is required for prediction process isolation")
    if args.stage != "predict" and args.direction:
        raise ValueError("--direction is accepted only by the isolated predict stage")
    if args.stage != "fit-model" and args.donor:
        raise ValueError("--donor is accepted only by fit-model")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    _assert_zero_process_swap()
    runner = _load_runner()
    core = runner._import_core()
    runner.configure_resources(
        cpu_threads=args.cpu_threads,
        gpu_memory_fraction=args.gpu_memory_fraction,
        device=args.device,
    )
    runner.seed_everything(1729)
    if args.stage == "prepare":
        prepare(args, runner, core)
    elif args.stage == "fit-model":
        fit_models(args, runner, core)
    elif args.stage == "predict":
        predict_direction(args, runner, core)
    else:
        score(args, runner, core)
    _assert_zero_process_swap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
