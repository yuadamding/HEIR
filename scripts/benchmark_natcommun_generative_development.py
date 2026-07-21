#!/usr/bin/env python3
"""Resource-bounded NatCommun generative *development* benchmark.

This runner is intentionally staged.  ``prepare`` is the only stage allowed to
open the registered source archive.  It writes, for every leave-one-donor-out
fold, a public fit/predict bundle and a separate score-only target.  The
``fit-predict`` stage is therefore unable to inspect held-out Visium counts by
construction.  ``score`` opens the target only after predictions have been
sealed.

The experiment is exposed model development, never independent confirmation.
Its primary image input is the frozen 112-um H-optimus-1 representation already
stored in the source.  UNI2-h is rejected rather than silently substituted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCHEMA = "heir.natcommun_generative_development_report.v2"
PREPARED_SCHEMA = "heir.natcommun_generative_prepared.v2"
PREDICTION_SCHEMA = "heir.natcommun_generative_predictions.v2"
SOURCE_SCHEMA = "heir.natcommun_regional_source.v2"
HOPTIMUS_REPOSITORY = "bioptimus/H-optimus-1"
HOPTIMUS_REVISION = "3592cb220dec7a150c5d7813fb56e68bd57473b9"
FROZEN_BASE_SEED = 1729
FROZEN_EPOCHS = 80
FROZEN_BATCH_SIZE = 256
FROZEN_LATENT_DIM = 20
FORBIDDEN_ENCODERS = frozenset({"MahmoodLab/UNI2-h", "UNI2-h", "uni2_h", "uni2"})
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
MODEL_ARMS = (
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5_blank",
    "M5_coordinates",
    "M6",
    "M7",
    "BLEEP",
)
PRIMARY_COMPARISONS = (
    "M3_vs_M0",
    "M3_vs_M1",
    "M3_vs_M4",
    "M3_supported_vs_M2_supported",
    "M3_vs_M6",
    "M3_vs_M7",
    "M3_vs_BLEEP",
    "M8_vs_M3",
)
SECRET_PREFIXES = ("heldout_st", "score_target", "target_counts", "st_test")
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/natcommun_generative_development")
DEFAULT_PROJECTED_SOURCE = DEFAULT_OUTPUT / "panel_256_projected_counts.npz"
DEFAULT_PROJECTED_SOURCE_SHA256 = "71479f891b5945762e20ec5b91d85bac097230b12ed9192aeacd965be119607f"
DEFAULT_PANEL = Path(__file__).resolve().parents[1] / "configs/natcommun_generative_gene_panel.json"
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[1] / "configs/natcommun_generative_development_protocol.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def _semantic_array_hash(arrays: Mapping[str, object]) -> str:
    """Hash array meaning, independent of NPZ ZIP timestamps/compression."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.dtype.str.encode())
        digest.update(_json_bytes(list(value.shape)))
        if value.dtype.kind in {"U", "O"}:
            if value.dtype.kind == "O":
                raise TypeError("object arrays are prohibited")
            digest.update(_json_bytes(value.tolist()))
        else:
            digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected a scalar string")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _json_scalar(archive: np.lib.npyio.NpzFile, key: str) -> Mapping[str, object]:
    if key not in archive.files:
        return {}
    try:
        value = json.loads(_scalar_text(archive[key]))
    except json.JSONDecodeError as error:
        raise ValueError(f"{key} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must contain a JSON object")
    return value


def configure_resources(*, cpu_threads: int, gpu_memory_fraction: float, device: str) -> None:
    if not 1 <= cpu_threads <= 4:
        raise ValueError("CPU threads must be between 1 and 4")
    if not 0.0 < gpu_memory_fraction <= 0.60:
        raise ValueError("GPU memory fraction must be in (0, 0.60]")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work begins.
        pass
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real NatCommun benchmark")
        index = torch.device(device).index or 0
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, index)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def reject_uni2(value: object) -> None:
    text = str(value)
    normalized = text.casefold().replace("-", "").replace("_", "")
    if text in FORBIDDEN_ENCODERS or "uni2" in normalized:
        raise ValueError("UNI2-h is explicitly prohibited in this experiment")


def _validate_encoder_receipt(receipt: Mapping[str, object], *, synthetic: bool) -> None:
    encoder = receipt.get("encoder", {})
    if not isinstance(encoder, Mapping):
        raise ValueError("source encoder receipt is missing")
    repository = str(encoder.get("repository", ""))
    reject_uni2(repository)
    if repository != HOPTIMUS_REPOSITORY:
        raise ValueError("primary source must use frozen bioptimus/H-optimus-1")
    if not synthetic and str(encoder.get("revision", "")) != HOPTIMUS_REVISION:
        raise ValueError("H-optimus-1 revision does not match the frozen primary")
    if str(encoder.get("fine_tuning", "none")) not in {"none", "prohibited"}:
        raise ValueError("fine-tuned image representations are prohibited")
    roles = receipt.get("encoder_roles", {})
    if isinstance(roles, Mapping):
        primary = roles.get("primary", {})
        if isinstance(primary, Mapping):
            reject_uni2(primary.get("repository", ""))
            if primary.get("repository") not in {None, HOPTIMUS_REPOSITORY}:
                raise ValueError("source primary encoder role is not H-optimus-1")
        secondary = roles.get("secondary_comparator", {})
        if (
            isinstance(secondary, Mapping)
            and "uni2" in str(secondary.get("repository", "")).casefold()
        ):
            status = str(secondary.get("status", ""))
            if status not in {"prespecified_not_run_in_primary_source", "not_run", "excluded"}:
                raise ValueError("UNI2-h features must not be present or run")


def _panel_gene_ids(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw: object = ()
    for key in ("selected_gene_ids", "gene_ids", "genes", "selected_genes"):
        if key in payload:
            raw = payload[key]
            break
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("panel must contain an ordered gene list")
    genes: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            item = item.get("gene_id", item.get("gene", item.get("symbol", "")))
        gene = str(item)
        if not gene:
            raise ValueError("panel contains an empty gene")
        genes.append(gene)
    if len(set(genes)) != len(genes):
        raise ValueError("panel genes must be unique")
    return tuple(genes)


def load_panel(path: Path, *, smoke: bool) -> tuple[tuple[str, ...], Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("panel JSON must be an object")
    genes = _panel_gene_ids(payload)
    if (not smoke and len(genes) != 256) or (smoke and len(genes) < 2):
        raise ValueError("the primary panel must contain exactly 256 genes")
    scope = str(payload.get("scope", payload.get("analysis_scope", ""))).casefold()
    if not smoke and "development" not in scope:
        raise ValueError("panel must be explicitly marked as development-selected")
    return genes, payload


def load_protocol(
    path: Path,
    *,
    source_path: Path,
    source_sha256: str,
    panel_path: Path,
    panel_payload: Mapping[str, object],
    gene_count: int,
    smoke: bool,
) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("protocol JSON must be an object")
    expected_schema = "heir.natcommun_generative_development_protocol.v2"
    if payload.get("schema") != expected_schema and not (
        smoke and payload.get("schema") == "synthetic"
    ):
        raise ValueError("unexpected generative development protocol schema")
    if payload.get("analysis_status") != "exposed_development_only_non_confirmatory":
        raise ValueError("protocol must be explicitly non-confirmatory development")
    if int(payload.get("gene_panel_size", -1)) != gene_count:
        raise ValueError("protocol and panel gene counts differ")
    if not smoke and int(payload.get("latent_dimensions", -1)) != 20:
        raise ValueError("protocol must freeze the secondary latent at 20D")

    immutable = payload.get("immutable_inputs", {})
    encoders = payload.get("encoders", {})
    images = payload.get("image_inputs", {})
    outer = payload.get("outer_validation", {})
    model = payload.get("generative_model", {})
    resources = payload.get("resource_limits", {})
    claims = payload.get("claim_boundaries", {})
    split_policy = payload.get("M8_split_policy", {})
    if not all(
        isinstance(value, Mapping)
        for value in (immutable, encoders, images, outer, model, resources, claims, split_policy)
    ):
        raise ValueError("protocol scientific contracts are incomplete")
    if not smoke:
        if immutable.get("source_schema") != SOURCE_SCHEMA:
            raise ValueError("protocol source schema is not the registered NatCommun source")
        if immutable.get("primary_encoder_repository") != HOPTIMUS_REPOSITORY:
            raise ValueError("protocol primary encoder is not H-optimus-1")
        if immutable.get("primary_encoder_revision") != HOPTIMUS_REVISION:
            raise ValueError("protocol H-optimus-1 revision changed")
        declared_source = str(immutable.get("source_sha256", ""))
        if declared_source != source_sha256:
            raise ValueError("source does not match the protocol SHA-256")
        panel_contract = payload.get("gene_panel", {})
        if not isinstance(panel_contract, Mapping):
            raise ValueError("protocol gene-panel contract is malformed")
        configured_panel = Path(str(panel_contract["external_frozen_artifact"]))
        resolved_panel = (
            Path(__file__).resolve().parents[1] / configured_panel
            if not configured_panel.is_absolute()
            else configured_panel
        ).resolve()
        if resolved_panel != panel_path.resolve():
            raise ValueError("panel path does not match the frozen protocol")
        if _sha256(panel_path) != panel_contract.get("external_frozen_artifact_sha256"):
            raise ValueError("panel file does not match the frozen protocol SHA-256")
        if panel_payload.get("artifact_sha256") != panel_contract.get(
            "external_frozen_identity_sha256"
        ):
            raise ValueError("panel semantic identity does not match the frozen protocol")
        frozen_hyperparameters = {
            "base_seed": FROZEN_BASE_SEED,
            "epochs": FROZEN_EPOCHS,
            "maximum_epochs": FROZEN_EPOCHS,
            "batch_size": FROZEN_BATCH_SIZE,
        }
        mismatched_hyperparameters = [
            name
            for name, expected in frozen_hyperparameters.items()
            if int(outer.get(name, -1)) != expected
        ]
        if mismatched_hyperparameters:
            raise ValueError(
                "protocol training hyperparameters are not frozen: "
                f"{mismatched_hyperparameters}"
            )
    if encoders.get("UNI2_h") != "forbidden_not_run":
        raise ValueError("protocol must explicitly prohibit UNI2-h")
    if images.get("primary") != "natural_registered_112um_H_optimus_1":
        raise ValueError("112-um H-optimus-1 must remain the primary image input")
    if images.get("held_out_ST_may_route_image_query") is not False:
        raise ValueError("held-out ST routing must be prohibited")
    if "scoring_only" not in str(outer.get("held_out_donor_ST_use", "")):
        raise ValueError("protocol must reserve held-out ST for scoring")
    if model.get("M0_M3_capacity_rule") != "same_H_and_E_query_encoder_and_same_ST_decoder":
        raise ValueError("protocol must freeze exact M0/M3 capacity identity")
    if int(model.get("iterative_updates", -1)) != 0:
        raise ValueError("iterative refinement is prohibited")
    if int(resources.get("maximum_CPU_threads", -1)) > 4:
        raise ValueError("protocol CPU limit exceeds four threads")
    if float(resources.get("maximum_projected_counts_GiB", math.inf)) > 1.0:
        raise ValueError("protocol projected-count memory limit exceeds 1 GiB")
    if claims.get("cell_level_claims") != "prohibited":
        raise ValueError("protocol must not authorize cell-level claims")
    if split_policy.get("Poisson_split_assumption") != "prohibited_under_overdispersion":
        raise ValueError("protocol must prohibit Poisson splitting under NB overdispersion")
    return payload


def _csr_dense_columns(
    archive: np.lib.npyio.NpzFile,
    prefix: str,
    columns: np.ndarray,
    *,
    row_mask: np.ndarray | None = None,
    chunk_rows: int = 2048,
) -> np.ndarray:
    required = [f"{prefix}_{name}" for name in ("data", "indices", "indptr", "shape")]
    missing = [name for name in required if name not in archive.files]
    if missing:
        raise ValueError(f"{prefix} CSR payload is incomplete: {missing}")
    data = np.asarray(archive[f"{prefix}_data"])
    indices = np.asarray(archive[f"{prefix}_indices"], dtype=np.int64)
    indptr = np.asarray(archive[f"{prefix}_indptr"], dtype=np.int64)
    shape_value = tuple(int(value) for value in np.asarray(archive[f"{prefix}_shape"]).tolist())
    if len(shape_value) != 2 or len(indptr) != shape_value[0] + 1:
        raise ValueError(f"{prefix} CSR shape is invalid")
    if len(data) != len(indices) or indptr[0] != 0 or indptr[-1] != len(data):
        raise ValueError(f"{prefix} CSR arrays are inconsistent")
    selected = np.asarray(columns, dtype=np.int64)
    if selected.ndim != 1 or not len(selected) or np.any(selected < 0):
        raise ValueError("selected columns are invalid")
    lookup = np.full(shape_value[1], -1, dtype=np.int64)
    lookup[selected] = np.arange(len(selected), dtype=np.int64)
    keep = np.ones(shape_value[0], dtype=bool) if row_mask is None else np.asarray(row_mask, bool)
    if keep.shape != (shape_value[0],):
        raise ValueError(f"{prefix} row mask is misaligned")
    output = np.zeros((int(keep.sum()), len(selected)), dtype=np.float32)
    selected_rows = np.flatnonzero(keep)
    for out_start in range(0, len(selected_rows), chunk_rows):
        batch_rows = selected_rows[out_start : out_start + chunk_rows]
        for offset, row in enumerate(batch_rows):
            left, right = int(indptr[row]), int(indptr[row + 1])
            local = lookup[indices[left:right]]
            retained = local >= 0
            if retained.any():
                np.add.at(output[out_start + offset], local[retained], data[left:right][retained])
    return output


@dataclass(frozen=True)
class SourceArrays:
    spot_ids: np.ndarray
    donor_ids: np.ndarray
    section_ids: np.ndarray
    indication_ids: np.ndarray
    image: np.ndarray
    coordinates: np.ndarray
    blank_image: np.ndarray
    st_counts: np.ndarray
    st_library: np.ndarray
    st_half_a: np.ndarray
    st_half_b: np.ndarray
    st_library_half_a: np.ndarray
    st_library_half_b: np.ndarray
    sc_counts: np.ndarray
    sc_library: np.ndarray
    sc_cell_ids: np.ndarray
    sc_donor_ids: np.ndarray
    sc_indication_ids: np.ndarray
    sc_type_ids: np.ndarray
    gene_ids: np.ndarray
    program_names: np.ndarray
    program_gene_membership: np.ndarray
    source_receipt: Mapping[str, object]


def load_projected_source(
    path: Path,
    genes: Sequence[str],
    *,
    source_sha256: str,
    panel: Mapping[str, object],
    expected_sha256: str | None,
) -> SourceArrays:
    observed_sha256 = _sha256(path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise ValueError("projected source does not match its expected SHA-256")
    with np.load(path, allow_pickle=False) as archive:
        metadata = _json_scalar(archive, "metadata_json")
        if metadata.get("schema") != "heir.natcommun_generative_projected_counts.v1":
            raise ValueError("projected source schema is unexpected")
        if metadata.get("analysis_status") != "exposed_development_only_non_confirmatory":
            raise ValueError("projected source does not disclose development exposure")
        if metadata.get("source_sha256") != source_sha256:
            raise ValueError("projected source is not bound to the registered source")
        selection = panel.get("selection", {})
        if not isinstance(selection, Mapping):
            raise ValueError("panel selection identity is missing")
        if metadata.get("panel_identity_sha256") != selection.get("identity_sha256"):
            raise ValueError("projected source is not bound to the panel identity")
        projected_genes = np.asarray(archive["gene_ids"]).astype(str)
        if projected_genes.tolist() != list(genes):
            raise ValueError("projected gene order differs from the frozen panel")
        projected_columns = np.asarray(archive["broad_column_indices"], dtype=np.int64)
        if projected_columns.tolist() != list(panel.get("broad_column_indices", ())):
            raise ValueError("projected broad columns differ from the frozen panel")
        full = np.asarray(archive["st_counts_full"], dtype=np.float32)
        half_a = np.asarray(archive["st_counts_half_a"], dtype=np.float32)
        half_b = np.asarray(archive["st_counts_half_b"], dtype=np.float32)
        if full.shape != half_a.shape or full.shape != half_b.shape:
            raise ValueError("projected ST full/half matrices are misaligned")
        if not np.array_equal(full, half_a + half_b):
            raise ValueError("projected ST halves do not reconstruct full counts")
        library = np.asarray(archive["st_total_umi_counts_full"], dtype=np.float64)
        library_a = np.asarray(archive["st_total_umi_counts_half_a"], dtype=np.float64)
        library_b = np.asarray(archive["st_total_umi_counts_half_b"], dtype=np.float64)
        if not np.array_equal(library, library_a + library_b):
            raise ValueError("projected ST half exposures do not reconstruct full exposure")
        spot_ids = np.asarray(archive["spot_ids"]).astype(str)
        donor_ids = np.asarray(archive["donor_ids"]).astype(str)
        section_ids = np.asarray(archive["section_ids"]).astype(str)
        indication_ids = np.asarray(archive["indication_ids"]).astype(str)
        image = np.asarray(archive["image_features"])
        coordinates = np.asarray(archive["coordinate_features"], dtype=np.float32)
        blank = np.asarray(archive["blank_image_feature_vector"])
        sc_counts = np.asarray(archive["sc_counts"], dtype=np.float32)
        sc_library = np.asarray(archive["sc_total_umi_counts"], dtype=np.float64)
        sc_cell_ids = np.asarray(archive["sc_cell_ids"]).astype(str)
        sc_donor_ids = np.asarray(archive["sc_donor_ids"]).astype(str)
        sc_indication_ids = np.asarray(archive["sc_indication_ids"]).astype(str)
        sc_type_ids = np.asarray(archive["sc_level1_type_ids"]).astype(str)
        program_names = np.asarray(archive["program_names"]).astype(str)
        program_membership = np.asarray(archive["program_gene_membership"], dtype=bool)
    rows = len(spot_ids)
    cells = len(sc_cell_ids)
    if any(value.shape != (rows,) for value in (donor_ids, section_ids, indication_ids, library)):
        raise ValueError("projected spot metadata is misaligned")
    if full.shape != (rows, len(genes)) or image.shape[0] != rows or coordinates.shape[0] != rows:
        raise ValueError("projected spot matrices are misaligned")
    if blank.shape != (image.shape[1],):
        raise ValueError("projected blank feature is misaligned")
    if any(
        value.shape != (cells,)
        for value in (sc_library, sc_donor_ids, sc_indication_ids, sc_type_ids)
    ) or sc_counts.shape != (cells, len(genes)):
        raise ValueError("projected single-nucleus arrays are misaligned")
    if program_membership.shape != (len(program_names), len(genes)):
        raise ValueError("projected biological programs are misaligned")
    donors = tuple(sorted(set(donor_ids.tolist())))
    if donors != EXPECTED_DONORS or set(sc_donor_ids.tolist()) != set(EXPECTED_DONORS):
        raise ValueError("projected source does not contain the 13 frozen donors")
    return SourceArrays(
        spot_ids=spot_ids,
        donor_ids=donor_ids,
        section_ids=section_ids,
        indication_ids=indication_ids,
        image=image,
        coordinates=coordinates,
        blank_image=blank,
        st_counts=full,
        st_library=library,
        st_half_a=half_a,
        st_half_b=half_b,
        st_library_half_a=library_a,
        st_library_half_b=library_b,
        sc_counts=sc_counts,
        sc_library=sc_library,
        sc_cell_ids=sc_cell_ids,
        sc_donor_ids=sc_donor_ids,
        sc_indication_ids=sc_indication_ids,
        sc_type_ids=sc_type_ids,
        gene_ids=projected_genes,
        program_names=program_names,
        program_gene_membership=program_membership,
        source_receipt={
            "projected_source_sha256": observed_sha256,
            "metadata": dict(metadata),
        },
    )


def load_selected_source(
    path: Path,
    genes: Sequence[str],
    *,
    expected_sha256: str | None,
    smoke: bool,
) -> SourceArrays:
    if expected_sha256 and _sha256(path) != expected_sha256:
        raise ValueError("source does not match --expected-source-sha256")
    with np.load(path, allow_pickle=False) as archive:
        schema = _scalar_text(archive["schema_version"])
        if schema != SOURCE_SCHEMA and not (smoke and schema == "synthetic"):
            raise ValueError(f"unexpected source schema: {schema}")
        receipt = _json_scalar(archive, "source_receipt_json")
        _validate_encoder_receipt(receipt, synthetic=smoke)

        broad_genes = np.asarray(archive["broad_gene_ids"]).astype(str)
        index = {gene: position for position, gene in enumerate(broad_genes.tolist())}
        missing = [gene for gene in genes if gene not in index]
        if missing:
            raise ValueError(f"panel genes absent from source: {missing[:5]}")
        columns = np.asarray([index[gene] for gene in genes], dtype=np.int64)

        spot_ids_all = np.asarray(archive["spot_ids"]).astype(str)
        primary = np.asarray(archive["spot_primary_eligible"], dtype=bool)
        if primary.shape != (len(spot_ids_all),) or not primary.any():
            raise ValueError("spot eligibility is malformed")
        donors_all = np.asarray(archive["donor_ids"]).astype(str)
        sections_all = np.asarray(archive["section_ids"]).astype(str)
        indications_all = np.asarray(archive["indication_ids"]).astype(str)
        image_all = np.asarray(archive["image_features"])
        coordinate_all = np.asarray(archive["coordinate_features"], dtype=np.float32)
        blank = np.asarray(archive["blank_image_feature_vector"], dtype=np.float32)
        for name, value in {
            "donor_ids": donors_all,
            "section_ids": sections_all,
            "indication_ids": indications_all,
        }.items():
            if value.shape != (len(spot_ids_all),):
                raise ValueError(f"{name} is not spot aligned")
        if image_all.ndim != 2 or image_all.shape[0] != len(primary):
            raise ValueError("image features are not spot aligned")
        if blank.shape != (image_all.shape[1],):
            raise ValueError("blank image vector is misaligned")
        if coordinate_all.ndim != 2 or coordinate_all.shape[0] != len(primary):
            raise ValueError("coordinate features are not spot aligned")
        st_counts = _csr_dense_columns(archive, "st_broad_counts_full", columns, row_mask=primary)
        if "st_broad_counts_half_a_data" in archive.files:
            st_half_a = _csr_dense_columns(
                archive, "st_broad_counts_half_a", columns, row_mask=primary
            )
            st_half_b = _csr_dense_columns(
                archive, "st_broad_counts_half_b", columns, row_mask=primary
            )
        elif smoke:
            st_half_a = np.floor_divide(st_counts.astype(np.int64), 2).astype(np.float32)
            st_half_b = st_counts - st_half_a
        else:
            raise ValueError("registered source lacks ST count halves")
        st_library_all = np.asarray(archive["st_total_umi_counts_full"], dtype=np.float64)
        if st_library_all.shape != (len(primary),):
            raise ValueError("ST library totals are not spot aligned")
        if "st_total_umi_counts_half_a" in archive.files:
            st_library_a_all = np.asarray(archive["st_total_umi_counts_half_a"], dtype=np.float64)
            st_library_b_all = np.asarray(archive["st_total_umi_counts_half_b"], dtype=np.float64)
        elif smoke:
            st_library_a_all = np.floor_divide(st_library_all.astype(np.int64), 2)
            st_library_b_all = st_library_all - st_library_a_all
        else:
            raise ValueError("registered source lacks ST half-library exposures")
        if not np.array_equal(st_counts, st_half_a + st_half_b) or not np.array_equal(
            st_library_all, st_library_a_all + st_library_b_all
        ):
            raise ValueError("ST count/library halves do not reconstruct full values")

        sc_primary = np.asarray(archive["sc_primary_eligible"], dtype=bool)
        sc_donor_all = np.asarray(archive["sc_donor_ids"]).astype(str)
        selected_donors = np.unique(donors_all[primary])
        sc_keep = sc_primary & np.isin(sc_donor_all, selected_donors)
        sc_counts = _csr_dense_columns(archive, "sc_broad_counts", columns, row_mask=sc_keep)
        sc_library_all = np.asarray(archive["sc_total_umi_counts"], dtype=np.float64)
        sc_cell_all = np.asarray(archive["sc_cell_ids"]).astype(str)
        sc_indication_all = np.asarray(archive["sc_indication_ids"]).astype(str)
        sc_type_all = np.asarray(archive["sc_level1_type_ids"]).astype(str)
        if "program_names" in archive.files and "program_gene_membership" in archive.files:
            program_names = np.asarray(archive["program_names"]).astype(str)
            source_program_genes = np.asarray(archive["gene_ids"]).astype(str)
            source_program_membership = np.asarray(
                archive["program_gene_membership"], dtype=bool
            )
            if source_program_membership.shape != (
                len(program_names),
                len(source_program_genes),
            ):
                raise ValueError("source biological programs are malformed")
            source_program_index = {
                gene: index for index, gene in enumerate(source_program_genes.tolist())
            }
            program_membership = np.zeros((len(program_names), len(genes)), dtype=bool)
            for selected_index, gene in enumerate(genes):
                source_index = source_program_index.get(gene)
                if source_index is not None:
                    program_membership[:, selected_index] = source_program_membership[
                        :, source_index
                    ]
        elif smoke:
            program_names = np.asarray(["synthetic_program"])
            program_membership = np.ones((1, len(genes)), dtype=bool)
        else:
            raise ValueError("source lacks biological program membership")
        if any(
            value.shape != (len(sc_primary),)
            for value in (
                sc_donor_all,
                sc_library_all,
                sc_cell_all,
                sc_indication_all,
                sc_type_all,
            )
        ):
            raise ValueError("single-nucleus metadata is not cell aligned")

    primary_donors = tuple(sorted(set(donors_all[primary].tolist())))
    if not smoke and primary_donors != EXPECTED_DONORS:
        raise ValueError(f"expected the 13 frozen donors, observed {primary_donors}")
    if len(primary_donors) < 3:
        raise ValueError("at least three donors are required for wrong/generic controls")
    for donor in primary_donors:
        spot_indication = set(indications_all[primary & (donors_all == donor)].tolist())
        if len(spot_indication) != 1:
            raise ValueError(f"donor {donor} spans indications")
        if not np.any(sc_keep & (sc_donor_all == donor)):
            raise ValueError(f"donor {donor} has no matched single-nucleus reference")
        indication = next(iter(spot_indication))
        wrong = [
            other
            for other in primary_donors
            if other != donor
            and np.any(primary & (donors_all == other) & (indications_all == indication))
        ]
        if not wrong:
            raise ValueError(f"donor {donor} lacks a same-indication wrong reference")

    return SourceArrays(
        spot_ids=spot_ids_all[primary],
        donor_ids=donors_all[primary],
        section_ids=sections_all[primary],
        indication_ids=indications_all[primary],
        image=image_all[primary],
        coordinates=coordinate_all[primary],
        blank_image=blank,
        st_counts=st_counts,
        st_library=st_library_all[primary],
        st_half_a=st_half_a,
        st_half_b=st_half_b,
        st_library_half_a=st_library_a_all[primary],
        st_library_half_b=st_library_b_all[primary],
        sc_counts=sc_counts,
        sc_library=sc_library_all[sc_keep],
        sc_cell_ids=sc_cell_all[sc_keep],
        sc_donor_ids=sc_donor_all[sc_keep],
        sc_indication_ids=sc_indication_all[sc_keep],
        sc_type_ids=sc_type_all[sc_keep],
        gene_ids=np.asarray(genes),
        program_names=program_names,
        program_gene_membership=program_membership,
        source_receipt=dict(receipt),
    )


def _fold_seed(base_seed: int, donor: str) -> int:
    value = hashlib.sha256(f"{base_seed}:{donor}".encode()).digest()
    return int.from_bytes(value[:4], "little")


def _derangement_indices(section_ids: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic within-section derangement used by M4."""

    sections = np.asarray(section_ids).astype(str)
    result = np.arange(len(sections), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for section in sorted(set(sections.tolist())):
        rows = np.flatnonzero(sections == section)
        if len(rows) < 2:
            # A singleton cannot be deranged; mark it unavailable.
            result[rows] = -1
            continue
        shift = int(rng.integers(1, len(rows)))
        result[rows] = np.roll(rows, shift)
    return result


def _composition_stratified_derangement(
    section_ids: np.ndarray,
    composition: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derange H&E within section and H&E-derived coarse composition strata."""

    sections = np.asarray(section_ids).astype(str)
    values = np.asarray(composition, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(sections):
        raise ValueError("H&E composition predictions are not query aligned")
    if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0, atol=1.0e-4):
        raise ValueError("H&E composition predictions must lie on the simplex")
    result = np.arange(len(sections), dtype=np.int64)
    strata = np.zeros(len(sections), dtype=np.int64)
    rng = np.random.default_rng(seed)
    # Deterministic scalar projection avoids using any held-out molecular value.
    projection = values @ np.linspace(-1.0, 1.0, values.shape[1])
    for section in sorted(set(sections.tolist())):
        rows = np.flatnonzero(sections == section)
        # At most tertiles, but never create a singleton stratum.
        n_strata = min(3, max(1, len(rows) // 20))
        ordered = rows[np.argsort(projection[rows], kind="mergesort")]
        groups = np.array_split(ordered, n_strata)
        for stratum, group in enumerate(groups):
            strata[group] = stratum
            if len(group) < 2:
                raise ValueError("M4 composition stratum cannot be deranged without fixed points")
            shift = int(rng.integers(1, len(group)))
            result[group] = np.roll(group, shift)
    if np.any(result == np.arange(len(result))):
        raise RuntimeError("M4 stratified derangement contains fixed points")
    return result, strata


def _fold_public_arrays(source: SourceArrays, heldout: str, seed: int) -> dict[str, np.ndarray]:
    train_spot = source.donor_ids != heldout
    test_spot = source.donor_ids == heldout
    train_sc = source.sc_donor_ids != heldout
    matched_sc = source.sc_donor_ids == heldout
    indication = str(source.indication_ids[np.flatnonzero(test_spot)[0]])
    wrong_sc = (source.sc_donor_ids != heldout) & (source.sc_indication_ids == indication)
    wrong_donors = sorted(set(source.sc_donor_ids[wrong_sc].tolist()))
    if not wrong_donors:
        raise ValueError(f"{heldout} has no same-indication wrong reference")
    shuffle = _derangement_indices(source.section_ids[test_spot], seed)
    public = {
        "schema": np.asarray(PREPARED_SCHEMA),
        "heldout_donor": np.asarray(heldout),
        "gene_ids": source.gene_ids,
        "train_spot_ids": source.spot_ids[train_spot],
        "train_donor_ids": source.donor_ids[train_spot],
        "train_section_ids": source.section_ids[train_spot],
        "train_indication_ids": source.indication_ids[train_spot],
        "train_image": source.image[train_spot],
        "train_coordinates": source.coordinates[train_spot],
        "train_st_counts": source.st_counts[train_spot],
        "train_st_library": source.st_library[train_spot],
        "train_st_half_a": source.st_half_a[train_spot],
        "train_st_half_b": source.st_half_b[train_spot],
        "train_st_library_half_a": source.st_library_half_a[train_spot],
        "train_st_library_half_b": source.st_library_half_b[train_spot],
        "train_sc_counts": source.sc_counts[train_sc],
        "train_sc_library": source.sc_library[train_sc],
        "train_sc_cell_ids": source.sc_cell_ids[train_sc],
        "train_sc_donor_ids": source.sc_donor_ids[train_sc],
        "train_sc_indication_ids": source.sc_indication_ids[train_sc],
        "train_sc_type_ids": source.sc_type_ids[train_sc],
        "query_spot_ids": source.spot_ids[test_spot],
        "query_section_ids": source.section_ids[test_spot],
        "query_indication_ids": source.indication_ids[test_spot],
        "query_image": source.image[test_spot],
        "query_coordinates": source.coordinates[test_spot],
        "query_blank_image": np.repeat(source.blank_image[None], int(test_spot.sum()), axis=0),
        "query_shuffle_index": shuffle,
        "matched_sc_counts": source.sc_counts[matched_sc],
        "matched_sc_library": source.sc_library[matched_sc],
        "matched_sc_cell_ids": source.sc_cell_ids[matched_sc],
        "matched_sc_donor_ids": source.sc_donor_ids[matched_sc],
        "matched_sc_indication_ids": source.sc_indication_ids[matched_sc],
        "matched_sc_type_ids": source.sc_type_ids[matched_sc],
        # Wrong/generic banks are views into train_sc, avoiding duplicated count
        # payloads in every prepared fold.
        "wrong_train_sc_index": np.flatnonzero(
            source.sc_indication_ids[train_sc] == indication
        ).astype(np.int64),
        "wrong_donor_ids": np.asarray(wrong_donors),
        "program_names": source.program_names,
        "program_gene_membership": source.program_gene_membership,
    }
    return public


def _fold_secret_arrays(source: SourceArrays, heldout: str) -> dict[str, np.ndarray]:
    keep = source.donor_ids == heldout
    usable = source.st_library[keep] > 0
    return {
        "schema": np.asarray(PREPARED_SCHEMA),
        "heldout_donor": np.asarray(heldout),
        "heldout_spot_ids": source.spot_ids[keep],
        "heldout_section_ids": source.section_ids[keep],
        "heldout_indication_ids": source.indication_ids[keep],
        "heldout_st_counts": source.st_counts[keep],
        "heldout_st_library": source.st_library[keep],
        "heldout_st_half_a": source.st_half_a[keep],
        "heldout_st_half_b": source.st_half_b[keep],
        "heldout_st_library_half_a": source.st_library_half_a[keep],
        "heldout_st_library_half_b": source.st_library_half_b[keep],
        "primary_score_eligible": usable,
        "zero_depth_excluded_count": np.asarray(int((~usable).sum()), dtype=np.int64),
    }


def validate_public_fold(public: Mapping[str, object]) -> None:
    names = set(public)
    leaking = sorted(
        name
        for name in names
        if any(name.casefold().startswith(prefix) for prefix in SECRET_PREFIXES)
    )
    if leaking:
        raise ValueError(f"held-out ST leaked into public fold: {leaking}")
    heldout = _scalar_text(public["heldout_donor"])
    train_donors = np.asarray(public["train_donor_ids"]).astype(str)
    train_sc_donors = np.asarray(public["train_sc_donor_ids"]).astype(str)
    matched = np.asarray(public["matched_sc_donor_ids"]).astype(str)
    if heldout in set(train_donors.tolist()) or heldout in set(train_sc_donors.tolist()):
        raise ValueError("held-out donor leaked into fitted ST/snRNA encoders")
    if len(matched) == 0 or set(matched.tolist()) != {heldout}:
        raise ValueError("matched reference is not isolated to prediction-time data")
    query_ids = np.asarray(public["query_spot_ids"]).astype(str)
    if set(query_ids.tolist()) & set(np.asarray(public["train_spot_ids"]).astype(str).tolist()):
        raise ValueError("query spot identities overlap training spots")
    if np.any(np.asarray(public["train_st_library"], dtype=float) <= 0):
        raise ValueError("zero-depth training spots must be removed before fitting")
    if not np.array_equal(
        np.asarray(public["train_st_counts"]),
        np.asarray(public["train_st_half_a"]) + np.asarray(public["train_st_half_b"]),
    ) or not np.array_equal(
        np.asarray(public["train_st_library"]),
        np.asarray(public["train_st_library_half_a"])
        + np.asarray(public["train_st_library_half_b"]),
    ):
        raise ValueError("training ST full/half counts or exposures are inconsistent")
    if not set(np.asarray(public["wrong_donor_ids"]).astype(str).tolist()).isdisjoint({heldout}):
        raise ValueError("wrong-donor control contains the held-out donor")
    wrong_index = np.asarray(public["wrong_train_sc_index"], dtype=np.int64)
    if (
        wrong_index.ndim != 1
        or not len(wrong_index)
        or np.any(wrong_index < 0)
        or np.any(wrong_index >= len(train_sc_donors))
    ):
        raise ValueError("wrong/generic reference indices are invalid")
    if set(train_sc_donors[wrong_index].tolist()) != set(
        np.asarray(public["wrong_donor_ids"]).astype(str).tolist()
    ):
        raise ValueError("wrong/generic reference indices do not match declared donors")


def prepare(args: argparse.Namespace) -> Mapping[str, object]:
    if not args.smoke:
        _validate_args(args)
    genes, panel = load_panel(args.panel, smoke=args.smoke)
    source_sha256 = _sha256(args.source)
    if args.expected_source_sha256 and source_sha256 != args.expected_source_sha256:
        raise ValueError("source does not match --expected-source-sha256")
    protocol = load_protocol(
        args.protocol,
        source_path=args.source,
        source_sha256=source_sha256,
        panel_path=args.panel,
        panel_payload=panel,
        gene_count=len(genes),
        smoke=args.smoke,
    )
    panel_source = panel.get("source", {})
    if not args.smoke and (
        not isinstance(panel_source, Mapping)
        or str(panel_source.get("sha256", "")) != source_sha256
    ):
        raise ValueError("panel is not bound to the registered source SHA-256")
    projected_path = getattr(args, "projected_source", None)
    if projected_path is not None and Path(projected_path).is_file() and not args.smoke:
        source = load_projected_source(
            Path(projected_path),
            genes,
            source_sha256=source_sha256,
            panel=panel,
            expected_sha256=getattr(args, "expected_projected_source_sha256", None),
        )
        projected_receipt: Mapping[str, object] = {
            "path": str(Path(projected_path).resolve()),
            "sha256": _sha256(Path(projected_path)),
            "used": True,
        }
    else:
        source = load_selected_source(
            args.source,
            genes,
            expected_sha256=None,
            smoke=args.smoke,
        )
        projected_receipt = {"used": False}
    donors = tuple(sorted(set(source.donor_ids.tolist())))
    output = args.output.resolve()
    fold_receipts: dict[str, object] = {}
    for donor in donors:
        fold_dir = output / "folds" / donor
        seed = _fold_seed(args.seed, donor)
        public = _fold_public_arrays(source, donor, seed)
        # Zero-depth training spots cannot contribute to likelihood or dispersion.
        positive = np.asarray(public["train_st_library"], dtype=float) > 0
        for name in (
            "train_spot_ids",
            "train_donor_ids",
            "train_section_ids",
            "train_indication_ids",
            "train_image",
            "train_coordinates",
            "train_st_counts",
            "train_st_library",
            "train_st_half_a",
            "train_st_half_b",
            "train_st_library_half_a",
            "train_st_library_half_b",
        ):
            public[name] = np.asarray(public[name])[positive]
        validate_public_fold(public)
        secret = _fold_secret_arrays(source, donor)
        public_path = fold_dir / "fit_predict_input.npz"
        secret_path = fold_dir / "score_target.npz"
        _atomic_npz(public_path, public)
        _atomic_npz(secret_path, secret)
        public_identity = _semantic_array_hash(public)
        secret_identity = _semantic_array_hash(secret)
        receipt = {
            "schema": PREPARED_SCHEMA,
            "heldout_donor": donor,
            "seed": seed,
            "public_path": str(public_path),
            "public_semantic_sha256": public_identity,
            "score_target_path": str(secret_path),
            "score_target_semantic_sha256": secret_identity,
            "fit_predict_has_heldout_ST": False,
            "heldout_ST_open_stage": "score_only",
            "train_spots": int(len(public["train_spot_ids"])),
            "query_spots": int(len(public["query_spot_ids"])),
            "matched_reference_cells": int(len(public["matched_sc_cell_ids"])),
            "zero_depth_score_rows_excluded": int(secret["zero_depth_excluded_count"]),
        }
        _atomic_json(fold_dir / "prepare_receipt.json", receipt)
        fold_receipts[donor] = receipt
    manifest = {
        "schema": PREPARED_SCHEMA,
        "analysis_scope": "exposed_development_only_non_confirmatory",
        "scientific_authorization": "none",
        "source": str(args.source.resolve()),
        "source_sha256": source_sha256,
        "panel": str(args.panel.resolve()),
        "panel_sha256": _sha256(args.panel),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "protocol_schema": protocol["schema"],
        "projected_source": projected_receipt,
        "panel_gene_count": len(genes),
        "panel_selection_scope": str(panel.get("scope", panel.get("analysis_scope", ""))),
        "image_encoder": HOPTIMUS_REPOSITORY,
        "image_scale_um": 112,
        "base_seed": int(args.seed),
        "training_configuration": {
            "base_seed": int(args.seed),
            "epochs": int(getattr(args, "epochs", FROZEN_EPOCHS)),
            "batch_size": int(getattr(args, "batch_size", FROZEN_BATCH_SIZE)),
            "latent_dim": int(getattr(args, "latent_dim", FROZEN_LATENT_DIM)),
            "device": str(getattr(args, "device", "not_bound_during_prepare")),
        },
        "uni2_h_run": False,
        "donors": list(donors),
        "leave_one_donor_out": True,
        "folds_serial": True,
        "folds": fold_receipts,
    }
    _atomic_json(output / "prepared_manifest.json", manifest)
    return manifest


def _read_prepared_manifest(output: Path) -> Mapping[str, object]:
    path = output / "prepared_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != PREPARED_SCHEMA:
        raise ValueError("prepared manifest is missing or malformed")
    if value.get("image_encoder") != HOPTIMUS_REPOSITORY or value.get("uni2_h_run") is not False:
        raise ValueError("prepared manifest violates the encoder contract")
    reject_uni2(value.get("image_encoder"))
    for field, hash_field in (("panel", "panel_sha256"), ("protocol", "protocol_sha256")):
        artifact = Path(str(value.get(field, "")))
        if not artifact.is_file() or _sha256(artifact) != value.get(hash_field):
            raise ValueError(f"prepared {field} identity changed after preparation")
    projected = value.get("projected_source", {})
    if isinstance(projected, Mapping) and projected.get("used") is True:
        artifact = Path(str(projected.get("path", "")))
        if not artifact.is_file() or _sha256(artifact) != projected.get("sha256"):
            raise ValueError("projected source identity changed after preparation")
    return value


def _validate_bound_training_configuration(
    args: argparse.Namespace, prepared: Mapping[str, object]
) -> None:
    """Require every later stage to use the configuration sealed at prepare."""

    configured = prepared.get("training_configuration")
    if not isinstance(configured, Mapping):
        raise ValueError("prepared training configuration is missing")
    observed = {
        "base_seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "latent_dim": int(args.latent_dim),
        "device": str(args.device),
    }
    mismatched = [name for name, value in observed.items() if configured.get(name) != value]
    if mismatched:
        raise ValueError(
            f"stage arguments differ from prepared training configuration: {mismatched}"
        )


def _verify_semantic_file(path: Path, expected: str) -> dict[str, np.ndarray]:
    arrays = _load_arrays(path)
    observed = _semantic_array_hash(arrays)
    if observed != expected:
        raise ValueError(f"semantic identity mismatch for {path}")
    return arrays


def _checkpoint_identity(
    public_identity: str,
    *,
    donor: str,
    seed: int,
    epochs: int,
    latent_dim: int,
    batch_size: int,
    device: str,
    runner_sha256: str,
    core_sha256: str,
    protocol_sha256: str,
) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "schema": PREDICTION_SCHEMA,
                "public_identity": public_identity,
                "donor": donor,
                "seed": seed,
                "epochs": epochs,
                "latent_dim": latent_dim,
                "batch_size": batch_size,
                "device": device,
                "runner_sha256": runner_sha256,
                "core_sha256": core_sha256,
                "protocol_sha256": protocol_sha256,
                "model_arms": MODEL_ARMS,
                "image_encoder": HOPTIMUS_REPOSITORY,
            }
        )
    ).hexdigest()


def _call_with_supported_kwargs(function: Any, /, *args: object, **kwargs: object) -> Any:
    """Call a core primitive while failing on missing required semantic inputs."""

    signature = inspect.signature(function)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return function(*args, **supported)


def _import_core() -> Any:
    try:
        from heir.evaluation import generative_fusion
    except ImportError as error:
        raise RuntimeError("generative_fusion core is required for fit-predict/score") from error
    required = (
        "fit_nb2_dispersion",
        "nb2_log_prob",
        "nb2_deviance",
        "CountVAE",
        "ReferenceMixture",
        "build_reference_mixture",
        "CompositionStateModel",
        "ContrastiveRetrieval",
        "exact_sign_flip_test",
        "holm_adjust",
        "evaluate_ordered_gates",
        "calibration_slope",
        "reliability_adjusted_variance",
        "interval_coverage",
    )
    missing = [name for name in required if not hasattr(generative_fusion, name)]
    if missing:
        raise RuntimeError(f"generative_fusion API is incomplete: {missing}")
    return generative_fusion


def _instantiate(function: Any, **kwargs: object) -> Any:
    signature = inspect.signature(function)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return function(**supported)


def _donor_equal_generic_reference(core: Any, mixture: object, query_donor: str) -> object:
    """Relabel a query-excluded bank after donor-equal/natural-within-donor weighting."""

    required = ("weights", "donor_ids", "type_labels")
    if not all(hasattr(mixture, name) for name in required) or not hasattr(mixture, "__dict__"):
        raise RuntimeError("ReferenceMixture cannot express the donor-equal generic bank")
    donor_ids = np.asarray(mixture.donor_ids).astype(str)
    type_ids = np.asarray(mixture.type_labels).astype(str)
    weights = np.asarray(mixture.weights, dtype=np.float64).copy()
    for type_name in sorted(set(type_ids.tolist())):
        type_keep = type_ids == type_name
        donors = sorted(set(donor_ids[type_keep].tolist()))
        for donor in donors:
            keep = type_keep & (donor_ids == donor)
            weights[keep] /= max(float(weights[keep].sum()), 1.0e-12)
            weights[keep] /= len(donors)
    payload = dict(mixture.__dict__)
    payload["weights"] = weights
    payload["donor_ids"] = np.repeat(query_donor, len(weights))
    if "component_ids" in payload:
        payload["component_ids"] = np.asarray(
            [f"generic::{query_donor}::{value}" for value in payload["component_ids"]]
        )
    return core.ReferenceMixture(**payload)


def _fit_module(
    module: Any,
    *,
    counts: np.ndarray,
    covariates: np.ndarray | None,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> Any:
    """Use the core object's bounded fit contract (``fit`` or ``fit_model``)."""

    method = getattr(module, "fit_model", None) or getattr(module, "fit", None)
    if method is None:
        raise RuntimeError(f"{type(module).__name__} exposes no bounded fit method")
    return _call_with_supported_kwargs(
        method,
        counts,
        covariates=covariates,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )


def _encode(module: Any, counts: np.ndarray, *, modality: str, device: str) -> np.ndarray:
    method = getattr(module, "encode_numpy", None) or getattr(module, "encode", None)
    if method is None:
        raise RuntimeError(f"{type(module).__name__} exposes no encoder")
    value = _call_with_supported_kwargs(method, counts, modality=modality, device=device)
    if isinstance(value, tuple):
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or len(result) != len(counts):
        raise RuntimeError("molecular encoder returned malformed latents")
    return result


def _decode_rate(
    module: Any, latent: np.ndarray, *, endpoint_genes: int, device: str
) -> np.ndarray:
    """Decode per-unit-library rates; score applies the sealed held-out exposure."""

    method = getattr(module, "decode_numpy", None) or getattr(module, "decode", None)
    if method is None:
        raise RuntimeError(f"{type(module).__name__} exposes no decoder")
    value = _call_with_supported_kwargs(
        method,
        latent,
        library_size=np.ones(len(latent), dtype=np.float32),
        library=np.ones(len(latent), dtype=np.float32),
        modality="st",
        device=device,
    )
    if isinstance(value, tuple):
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (len(latent), getattr(module, "n_genes", result.shape[1])):
        if result.ndim != 2 or len(result) != len(latent):
            raise RuntimeError("molecular decoder returned malformed rates")
    if result.shape[1] <= endpoint_genes:
        raise RuntimeError("decoder lacks the required unselected-transcript bin")
    return np.maximum(result[:, :endpoint_genes], 1.0e-8)


def _normalized_coordinates(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(train, dtype=float).mean(axis=0)
    scale = np.asarray(train, dtype=float).std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    return ((train - mean) / scale).astype(np.float32), ((query - mean) / scale).astype(np.float32)


def training_only_composition_proxy(
    train_st_counts: np.ndarray,
    train_sc_counts: np.ndarray,
    train_sc_type_ids: np.ndarray,
    *,
    iterations: int = 80,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Derive a disclosed coarse routing proxy using training outcomes only.

    This deterministic simplex-constrained least-squares deconvolution matches
    selected-gene proportions to training-donor snRNA type signatures.  It is
    not observed spot composition truth.  The public fold makes it impossible
    for this helper to receive held-out ST through the benchmark call path.
    """

    st = np.asarray(train_st_counts, dtype=np.float64)
    sc = np.asarray(train_sc_counts, dtype=np.float64)
    type_ids = np.asarray(train_sc_type_ids).astype(str)
    if st.ndim != 2 or sc.ndim != 2 or st.shape[1] != sc.shape[1]:
        raise ValueError("training count matrices must share genes")
    types = tuple(sorted(set(type_ids.tolist())))
    if len(types) < 2:
        raise ValueError("composition proxy requires at least two training reference types")
    signatures = []
    for type_name in types:
        cells = sc[type_ids == type_name]
        if not len(cells):
            raise ValueError(f"reference type {type_name} has no cells")
        signatures.append(cells.sum(axis=0) + 0.5)
    signature = np.asarray(signatures, dtype=np.float64)
    signature /= signature.sum(axis=1, keepdims=True)
    target = st + 0.5
    target /= target.sum(axis=1, keepdims=True)
    weights = np.full((len(st), len(types)), 1.0 / len(types), dtype=np.float64)
    spectral_bound = float(np.linalg.norm(signature @ signature.T, ord=2))
    step = 0.5 / max(spectral_bound, 1.0e-8)
    for _ in range(iterations):
        gradient = (weights @ signature - target) @ signature.T
        weights = np.maximum(weights - step * gradient, 1.0e-8)
        weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32), types


def donor_type_proxy_alignment_statistics(
    st_latent: np.ndarray,
    st_donor_ids: np.ndarray,
    st_type_composition: np.ndarray,
    reference_latent: np.ndarray,
    reference_donor_ids: np.ndarray,
    reference_type_ids: np.ndarray,
    type_order: Sequence[str],
) -> Mapping[str, object]:
    """Training-only donor×type proxy alignment diagnostic.

    ST has no observed spot-level type truth, so its donor×type latent mean is
    composition-weighted using the disclosed training-only deconvolution proxy.
    Each evaluable type contributes equally to matched and off-diagonal donor
    distances.  This is a coarse adequacy diagnostic, not cell-level evidence.
    """

    st = np.asarray(st_latent, dtype=np.float64)
    st_donors = np.asarray(st_donor_ids).astype(str)
    composition = np.asarray(st_type_composition, dtype=np.float64)
    reference = np.asarray(reference_latent, dtype=np.float64)
    reference_donors = np.asarray(reference_donor_ids).astype(str)
    reference_types = np.asarray(reference_type_ids).astype(str)
    if (
        st.ndim != 2
        or reference.ndim != 2
        or st.shape[1] != reference.shape[1]
        or st_donors.shape != (len(st),)
        or composition.shape != (len(st), len(type_order))
        or reference_donors.shape != (len(reference),)
        or reference_types.shape != (len(reference),)
        or not np.isfinite(st).all()
        or not np.isfinite(reference).all()
        or not np.isfinite(composition).all()
    ):
        raise ValueError("donor-type proxy alignment inputs are malformed")
    matched_by_type: list[float] = []
    mismatched_by_type: list[float] = []
    evaluable_types: list[str] = []
    donor_pairs = 0
    for type_index, type_name in enumerate(type_order):
        donors = sorted(
            set(st_donors.tolist())
            & set(reference_donors[reference_types == type_name].tolist())
        )
        if len(donors) < 2:
            continue
        st_means = []
        reference_means = []
        for donor in donors:
            local_st = st_donors == donor
            weights = composition[local_st, type_index]
            if not len(weights) or float(weights.sum()) <= 0:
                break
            st_means.append(np.average(st[local_st], axis=0, weights=weights))
            local_reference = (reference_donors == donor) & (
                reference_types == type_name
            )
            reference_means.append(reference[local_reference].mean(axis=0))
        if len(st_means) != len(donors):
            continue
        pairwise = np.mean(
            (np.asarray(st_means)[:, None] - np.asarray(reference_means)[None]) ** 2,
            axis=2,
        )
        diagonal = np.eye(len(donors), dtype=bool)
        matched_by_type.append(float(np.mean(pairwise[diagonal])))
        mismatched_by_type.append(float(np.mean(pairwise[~diagonal])))
        evaluable_types.append(str(type_name))
        donor_pairs += len(donors)
    if not evaluable_types:
        return {
            "evaluable": False,
            "evaluable_types": [],
            "donor_type_pairs": 0,
            "reason": "no_type_has_two_training_donors_with_reference_cells",
        }
    matched = float(np.mean(matched_by_type))
    mismatched = float(np.mean(mismatched_by_type))
    return {
        "evaluable": True,
        "evaluable_types": evaluable_types,
        "donor_type_pairs": donor_pairs,
        "matched_MSE": matched,
        "mismatched_MSE": mismatched,
        "matched_to_mismatched_ratio": matched / max(mismatched, 1.0e-12),
        "ST_type_source": "training_only_composition_weighted_proxy_not_observed_truth",
    }


def biologically_hard_negative_indices(
    donor_ids: np.ndarray,
    indication_ids: np.ndarray,
    composition: np.ndarray,
    molecular_latent: np.ndarray,
) -> np.ndarray:
    """Choose different-donor negatives using a disclosed latent projection.

    Candidates are prioritized within the same indication and dominant coarse
    composition.  The bounded implementation then chooses the largest distance
    along one fixed latent projection; it does not claim an exact farthest
    neighbor under the full latent-space metric.
    """

    donors = np.asarray(donor_ids).astype(str)
    indications = np.asarray(indication_ids).astype(str)
    mixture = np.asarray(composition, dtype=np.float64)
    latent = np.asarray(molecular_latent, dtype=np.float64)
    if (
        donors.shape != indications.shape
        or mixture.shape[0] != len(donors)
        or latent.shape[0] != len(donors)
    ):
        raise ValueError("hard-negative inputs must be row aligned")
    dominant_type = np.argmax(mixture, axis=1)
    projection = latent @ np.linspace(-1.0, 1.0, latent.shape[1])
    def extrema(keys: list[tuple[object, ...]]) -> dict[tuple[object, ...], dict[str, list[int]]]:
        groups: dict[tuple[object, ...], dict[str, list[int]]] = {}
        for row, key in enumerate(keys):
            by_donor = groups.setdefault(key, {})
            if donors[row] not in by_donor:
                by_donor[donors[row]] = [row, row]
            else:
                low, high = by_donor[donors[row]]
                if projection[row] < projection[low]:
                    low = row
                if projection[row] > projection[high]:
                    high = row
                by_donor[donors[row]] = [low, high]
        return groups

    primary = extrema(
        [(indications[row], int(dominant_type[row])) for row in range(len(donors))]
    )
    indication_only = extrema([(indications[row],) for row in range(len(donors))])
    global_group = extrema([("all",) for _ in range(len(donors))])
    result = np.empty(len(donors), dtype=np.int64)
    for index in range(len(donors)):
        candidate_rows: list[int] = []
        for groups, key in (
            (primary, (indications[index], int(dominant_type[index]))),
            (indication_only, (indications[index],)),
            (global_group, ("all",)),
        ):
            candidate_rows = [
                row
                for donor, pair in groups[key].items()
                if donor != donors[index]
                for row in pair
            ]
            if candidate_rows:
                break
        if not candidate_rows:
            raise ValueError("hard negatives require at least two training donors")
        candidates = np.asarray(sorted(set(candidate_rows)), dtype=np.int64)
        distance = np.abs(projection[candidates] - projection[index])
        result[index] = int(candidates[np.argmax(distance)])
    return result


def _append_other_count_bin(counts: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Add the unselected-transcript bin required by the softmax count decoder."""

    values = np.asarray(counts, dtype=np.float32)
    exposure = np.asarray(library, dtype=np.float32)
    if values.ndim != 2 or exposure.shape != (len(values),):
        raise ValueError("counts and full-library exposure are misaligned")
    other = exposure - values.sum(axis=1)
    if np.any(other < -1.0e-4):
        raise ValueError("selected panel counts exceed the full-library exposure")
    return np.concatenate((values, np.maximum(other, 0.0)[:, None]), axis=1)


def _nb_compatible_split(
    core: Any,
    counts: np.ndarray,
    dispersion: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    split_function = next(
        (
            getattr(core, name)
            for name in (
                "nb2_beta_binomial_split",
                "negative_binomial_split",
                "nb2_split",
                "split_nb2_counts",
            )
            if hasattr(core, name)
        ),
        None,
    )
    if split_function is None:
        raise RuntimeError("generative core exposes no NB-compatible split routine")
    split = _call_with_supported_kwargs(
        split_function,
        counts,
        dispersion=dispersion,
        theta=dispersion,
        fraction=0.5,
        seed=seed,
    )
    if isinstance(split, Mapping):
        first = split.get("a", split.get("first", split.get("train")))
        second = split.get("b", split.get("second", split.get("test")))
    elif hasattr(split, "first") and hasattr(split, "second"):
        first, second = split.first, split.second
    else:
        first, second = split[:2]
    if torch.is_tensor(first):
        first = first.detach().cpu().numpy()
    if torch.is_tensor(second):
        second = second.detach().cpu().numpy()
    half_a = np.asarray(first, dtype=np.float32)
    half_b = np.asarray(second, dtype=np.float32)
    if half_a.shape != counts.shape or half_b.shape != counts.shape:
        raise RuntimeError("NB split returned malformed halves")
    if not np.array_equal(half_a + half_b, counts):
        raise RuntimeError("NB-compatible halves must reconstruct the full count target")
    return half_a, half_b


def fit_m8_cross_half_predictor(
    core: Any,
    train_counts: np.ndarray,
    train_library: np.ndarray,
    dispersion: np.ndarray,
    augmented_dispersion: np.ndarray,
    *,
    seed: int,
    ridge: float = 10.0,
) -> Mapping[str, np.ndarray]:
    """Fit the molecular ceiling on training-donor NB-compatible halves only."""

    augmented = _append_other_count_bin(train_counts, train_library)
    if np.asarray(augmented_dispersion).shape != (augmented.shape[1],):
        raise ValueError("M8 augmented dispersion differs from the other-inclusive panel")
    if not np.allclose(
        np.asarray(augmented_dispersion)[: train_counts.shape[1]], dispersion
    ):
        raise ValueError("M8 endpoint and augmented dispersion fits disagree")
    half_a_full, half_b_full = _nb_compatible_split(
        core, augmented, augmented_dispersion, seed=seed
    )
    exposure_a = half_a_full.sum(axis=1, dtype=np.float64)
    exposure_b = half_b_full.sum(axis=1, dtype=np.float64)
    keep = (exposure_a > 0) & (exposure_b > 0)
    if not keep.any():
        raise RuntimeError("training M8 split has no positive-depth paired halves")
    half_a = half_a_full[keep, : train_counts.shape[1]]
    half_b = half_b_full[keep, : train_counts.shape[1]]
    exposure_a = exposure_a[keep]
    exposure_b = exposure_b[keep]
    scale = 10_000.0
    x = np.log1p(half_a * (scale / exposure_a[:, None]))
    y = np.log1p(half_b * (scale / exposure_b[:, None]))
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1.0e-6] = 1.0
    x_standard = (x - x_mean) / x_scale
    y_mean = y.mean(axis=0)
    y_center = y - y_mean
    gram = x_standard.T @ x_standard
    gram.flat[:: len(gram) + 1] += ridge
    coefficient = np.linalg.solve(gram, x_standard.T @ y_center)
    return {
        "coefficient": coefficient.astype(np.float32),
        "intercept": y_mean.astype(np.float32),
        "input_mean": x_mean.astype(np.float32),
        "input_scale": x_scale.astype(np.float32),
        "normalization_scale": np.asarray(scale, dtype=np.float32),
        "ridge": np.asarray(ridge, dtype=np.float32),
        "training_split_seed": np.asarray(seed, dtype=np.int64),
        "training_target": np.asarray("NB_compatible_half_B"),
        "training_zero_split_depth_excluded": np.asarray((~keep).sum(), dtype=np.int64),
        "split_includes_unselected_transcript_bin": np.asarray(True),
    }


def apply_m8_cross_half_predictor(
    half_a: np.ndarray,
    half_exposure: np.ndarray,
    parameters: Mapping[str, np.ndarray],
) -> np.ndarray:
    scale = float(np.asarray(parameters["normalization_scale"]))
    exposure = np.maximum(np.asarray(half_exposure, dtype=np.float64), 1.0)
    x = np.log1p(np.asarray(half_a, dtype=np.float64) * (scale / exposure[:, None]))
    standardized = (x - parameters["input_mean"]) / parameters["input_scale"]
    log_rate_scaled = standardized @ parameters["coefficient"] + parameters["intercept"]
    return np.maximum(np.expm1(log_rate_scaled) / scale, 1.0e-8).astype(np.float32)


def fit_training_diagnostics(
    train_counts: np.ndarray,
    train_library: np.ndarray,
    train_half_a: np.ndarray,
    train_half_b: np.ndarray,
    train_library_half_a: np.ndarray,
    train_library_half_b: np.ndarray,
    program_names: np.ndarray,
    program_membership: np.ndarray,
    *,
    latent_dim: int,
) -> Mapping[str, np.ndarray]:
    """Freeze the secondary molecular basis and quality thresholds on training rows."""

    counts = np.asarray(train_counts, dtype=np.float64)
    library = np.asarray(train_library, dtype=np.float64)
    scale = 10_000.0
    normalized = np.log1p(counts * (scale / library[:, None]))
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:latent_dim]
    components = eigenvectors[:, order].T
    scores = centered @ components.T
    score_scale = scores.std(axis=0)
    score_scale[score_scale < 1.0e-6] = 1.0

    membership = np.asarray(program_membership, dtype=bool)
    names = np.asarray(program_names).astype(str)
    if membership.shape != (len(names), counts.shape[1]):
        raise ValueError("training program membership is malformed")
    active = membership.sum(axis=1) > 0
    program_thresholds = np.full(len(names), np.nan, dtype=np.float64)
    for index in np.flatnonzero(active):
        program_score = normalized[:, membership[index]].mean(axis=1)
        program_thresholds[index] = np.quantile(program_score, 0.90)

    library_a = np.maximum(np.asarray(train_library_half_a, dtype=np.float64), 1.0)
    library_b = np.maximum(np.asarray(train_library_half_b, dtype=np.float64), 1.0)
    first = np.log1p(np.asarray(train_half_a) * (scale / library_a[:, None]))
    second = np.log1p(np.asarray(train_half_b) * (scale / library_b[:, None]))
    split_covariance = np.mean(
        (first - first.mean(axis=0)) * (second - second.mean(axis=0)), axis=0
    )
    return {
        "diagnostic_normalization_scale": np.asarray(scale, dtype=np.float32),
        "diagnostic_basis_mean": mean.astype(np.float32),
        "diagnostic_basis_components": components.astype(np.float32),
        "diagnostic_basis_score_scale": score_scale.astype(np.float32),
        "diagnostic_program_names": names,
        "diagnostic_program_membership": membership,
        "diagnostic_program_active": active,
        "diagnostic_rare_program_thresholds": program_thresholds.astype(np.float32),
        "diagnostic_training_split_covariance": split_covariance.astype(np.float32),
        "diagnostic_training_reliable_gene": (split_covariance > 0),
    }


def _prediction_receipt_arrays(
    donor: str,
    public: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    rate_variances: Mapping[str, np.ndarray],
    dispersion: np.ndarray,
    m8_parameters: Mapping[str, np.ndarray],
    diagnostic_parameters: Mapping[str, np.ndarray],
    m4_shuffle_index: np.ndarray,
    m4_composition_strata: np.ndarray,
    query_composition: np.ndarray,
    bleep_retrieval_entropy: np.ndarray,
    reference_type_names: Sequence[str],
    matched_observed_type_names: Sequence[str],
    matched_effective_sample_size: float,
    matched_cell_count: int,
) -> dict[str, np.ndarray]:
    expected = (len(public["query_spot_ids"]), len(public["gene_ids"]))
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "heldout_donor": np.asarray(donor),
        "query_spot_ids": np.asarray(public["query_spot_ids"]),
        "gene_ids": np.asarray(public["gene_ids"]),
        "training_only_dispersion": np.asarray(dispersion, dtype=np.float32),
        "prediction_scale": np.asarray("per_unit_actual_ST_library_rate"),
        "wrong_donor_ids": np.asarray(public["wrong_donor_ids"]),
        "composition_target_provenance": np.asarray(
            "training_ST_only_proxy_not_observed_spot_truth"
        ),
        "query_composition_provenance": np.asarray("H_and_E_prediction_only_no_heldout_ST"),
        "query_H_composition": np.asarray(query_composition, dtype=np.float32),
        "m4_shuffle_index": np.asarray(m4_shuffle_index, dtype=np.int64),
        "m4_composition_stratum": np.asarray(m4_composition_strata, dtype=np.int64),
        "BLEEP_retrieval_entropy": np.asarray(
            bleep_retrieval_entropy, dtype=np.float32
        ),
        "reference_model_type_names": np.asarray(reference_type_names),
        "matched_observed_type_names": np.asarray(matched_observed_type_names),
        "matched_reference_effective_sample_size": np.asarray(
            matched_effective_sample_size, dtype=np.float32
        ),
        "matched_reference_cell_count": np.asarray(matched_cell_count, dtype=np.int64),
    }
    for arm in MODEL_ARMS:
        if arm not in predictions:
            raise RuntimeError(f"missing prediction arm: {arm}")
        value = np.asarray(predictions[arm], dtype=np.float32)
        valid_shape = (
            value.shape == expected
            if arm != "M6"
            else value.shape == (len(public["wrong_donor_ids"]), *expected)
        )
        if not valid_shape or not np.isfinite(value).all() or np.any(value <= 0):
            raise RuntimeError(f"{arm} returned invalid count means")
        arrays["rate_M6_candidates" if arm == "M6" else f"rate_{arm}"] = value
    for arm in ("M0", "M1", "M2", "M3"):
        value = np.asarray(rate_variances[arm], dtype=np.float32)
        if value.shape != expected or not np.isfinite(value).all() or np.any(value < 0):
            raise RuntimeError(f"{arm} returned invalid posterior rate variance")
        arrays[f"posterior_rate_variance_{arm}"] = value
    for name, value in m8_parameters.items():
        arrays[f"m8_{name}"] = np.asarray(value)
    for name, value in diagnostic_parameters.items():
        arrays[name] = np.asarray(value)
    return arrays


def fit_predict_one_fold(
    public: Mapping[str, np.ndarray],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    seed: int,
) -> Mapping[str, np.ndarray]:
    """Fit one fold without accepting a held-out ST argument.

    The core classes own the neural details.  This orchestrator enforces that
    M0 and M3 share one composition/state model and the exact same ST decoder.
    Reference construction receives all held-out donor cells and must return a
    mixture distribution; no centroid-only fallback exists.
    """

    validate_public_fold(public)
    core = _import_core()
    seed_everything(seed)
    train_counts = np.asarray(public["train_st_counts"], dtype=np.float32)
    train_library = np.asarray(public["train_st_library"], dtype=np.float32)
    sc_counts = np.asarray(public["train_sc_counts"], dtype=np.float32)
    matched_counts = np.asarray(public["matched_sc_counts"], dtype=np.float32)
    wrong_index = np.asarray(public["wrong_train_sc_index"], dtype=np.int64)
    genes = train_counts.shape[1]
    train_augmented = _append_other_count_bin(train_counts, train_library)
    sc_augmented = _append_other_count_bin(sc_counts, public["train_sc_library"])
    matched_augmented = _append_other_count_bin(matched_counts, public["matched_sc_library"])

    dispersion = np.asarray(
        _call_with_supported_kwargs(
            core.fit_nb2_dispersion,
            train_counts,
            training_observation_ids=np.asarray(public["train_spot_ids"]).astype(str),
            training_donor_ids=np.asarray(public["train_donor_ids"]).astype(str),
            library_size=train_library,
            library_sizes=train_library,
        ),
        dtype=np.float32,
    )
    diagnostic_parameters = dict(
        fit_training_diagnostics(
            train_counts,
            train_library,
            np.asarray(public["train_st_half_a"]),
            np.asarray(public["train_st_half_b"]),
            np.asarray(public["train_st_library_half_a"]),
            np.asarray(public["train_st_library_half_b"]),
            np.asarray(public["program_names"]),
            np.asarray(public["program_gene_membership"]),
            latent_dim=latent_dim,
        )
    )
    vae_dispersion = np.asarray(
        core.fit_nb2_dispersion(
            train_augmented,
            training_observation_ids=np.asarray(public["train_spot_ids"]).astype(str),
            training_donor_ids=np.asarray(public["train_donor_ids"]).astype(str),
            library_size=train_library,
        ),
        dtype=np.float32,
    )
    m8_parameters = fit_m8_cross_half_predictor(
        core,
        train_counts,
        train_library,
        dispersion,
        vae_dispersion,
        seed=seed + 80_000,
    )
    diagnostic_parameters["training_other_transcript_dispersion"] = np.asarray(
        vae_dispersion[-1], dtype=np.float32
    )
    # Concatenation is bounded by raw 256-gene projections, never the broad CSR.
    molecular_counts = np.concatenate((train_augmented, sc_augmented), axis=0)
    modality = np.concatenate(
        (np.ones(len(train_counts), dtype=np.int64), np.zeros(len(sc_counts), dtype=np.int64))
    )
    training_molecular_donors = np.concatenate(
        (
            np.asarray(public["train_donor_ids"]).astype(str),
            np.asarray(public["train_sc_donor_ids"]).astype(str),
        )
    )
    training_molecular_observations = np.concatenate(
        (
            np.char.add("st::", np.asarray(public["train_spot_ids"]).astype(str)),
            np.char.add("reference::", np.asarray(public["train_sc_cell_ids"]).astype(str)),
        )
    )
    heldout_molecular_observations = np.concatenate(
        (
            np.char.add("st::", np.asarray(public["query_spot_ids"]).astype(str)),
            np.char.add("reference::", np.asarray(public["matched_sc_cell_ids"]).astype(str)),
        )
    )
    molecular_exposure = np.concatenate((train_library, public["train_sc_library"]))

    def fit_count_vae(alignment_weight: float) -> object:
        model = _instantiate(
            core.CountVAE,
            n_genes=genes + 1,
            gene_count=genes + 1,
            latent_dim=latent_dim,
            hidden_dim=min(256, max(32, genes)),
            dispersion=vae_dispersion,
        )
        model.to(torch.device(device))
        fit_method = getattr(model, "fit_model", None) or getattr(model, "fit", None)
        if fit_method is None:
            raise RuntimeError("CountVAE exposes no bounded fit method")
        _call_with_supported_kwargs(
            fit_method,
            molecular_counts,
            modality=modality,
            modalities=modality,
            training_donor_ids=training_molecular_donors,
            alignment_weight=alignment_weight,
            observation_ids=training_molecular_observations,
            heldout_observation_ids=heldout_molecular_observations,
            library_size=molecular_exposure,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        return model

    vae = fit_count_vae(1.0)
    alignment = getattr(vae, "alignment_diagnostics", None)
    if alignment is None:
        raise RuntimeError("shared latent fit omitted training-donor cross-assay alignment")
    unaligned_vae = fit_count_vae(0.0)
    unaligned_alignment = getattr(unaligned_vae, "alignment_diagnostics", None)
    if unaligned_alignment is None:
        raise RuntimeError("unaligned shared-latent comparator omitted alignment diagnostics")
    if (
        not bool(alignment.support_criterion_met)
        or not alignment.post_matched_to_mismatched_ratio
        < unaligned_alignment.post_matched_to_mismatched_ratio
    ):
        raise RuntimeError(
            "cross-assay alignment failed the training-only matched/mismatched support criterion"
        )
    diagnostic_parameters.update(
        {
            "cross_assay_alignment_donor_ids": np.asarray(alignment.donor_ids),
            "cross_assay_alignment_pre_MSE": np.asarray(
                alignment.pre_mse, dtype=np.float32
            ),
            "cross_assay_alignment_post_MSE": np.asarray(
                alignment.post_mse, dtype=np.float32
            ),
            "cross_assay_alignment_weight": np.asarray(
                alignment.weight, dtype=np.float32
            ),
            "cross_assay_alignment_pre_mismatched_MSE": np.asarray(
                alignment.pre_mismatched_mse, dtype=np.float32
            ),
            "cross_assay_alignment_post_mismatched_MSE": np.asarray(
                alignment.post_mismatched_mse, dtype=np.float32
            ),
            "cross_assay_alignment_pre_matched_to_mismatched_ratio": np.asarray(
                alignment.pre_matched_to_mismatched_ratio, dtype=np.float32
            ),
            "cross_assay_alignment_post_matched_to_mismatched_ratio": np.asarray(
                alignment.post_matched_to_mismatched_ratio, dtype=np.float32
            ),
            "cross_assay_alignment_pre_separation": np.asarray(
                alignment.pre_separation, dtype=np.float32
            ),
            "cross_assay_alignment_post_separation": np.asarray(
                alignment.post_separation, dtype=np.float32
            ),
            "cross_assay_alignment_optimizer_applications_per_epoch": np.asarray(
                alignment.optimizer_applications_per_epoch, dtype=np.int64
            ),
            "cross_assay_alignment_optimizer_applications_total": np.asarray(
                alignment.optimizer_applications_total, dtype=np.int64
            ),
            "cross_assay_alignment_support_criterion_met": np.asarray(
                alignment.support_criterion_met
            ),
            "cross_assay_alignment_support_criterion": np.asarray(
                alignment.support_criterion
            ),
            "cross_assay_unaligned_post_matched_MSE": np.asarray(
                unaligned_alignment.post_mse, dtype=np.float32
            ),
            "cross_assay_unaligned_post_mismatched_MSE": np.asarray(
                unaligned_alignment.post_mismatched_mse, dtype=np.float32
            ),
            "cross_assay_unaligned_post_matched_to_mismatched_ratio": np.asarray(
                unaligned_alignment.post_matched_to_mismatched_ratio, dtype=np.float32
            ),
            "cross_assay_alignment_beats_unaligned_comparator": np.asarray(True),
        }
    )
    train_st_latent = _encode(vae, train_augmented, modality="st", device=device)
    train_sc_latent = _encode(vae, sc_augmented, modality="scrna", device=device)
    matched_latent = _encode(vae, matched_augmented, modality="scrna", device=device)
    unaligned_train_st_latent = _encode(
        unaligned_vae, train_augmented, modality="st", device=device
    )
    unaligned_train_sc_latent = _encode(
        unaligned_vae, sc_augmented, modality="scrna", device=device
    )
    del unaligned_vae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    train_sc_donor_ids = np.asarray(public["train_sc_donor_ids"]).astype(str)
    train_sc_type_ids = np.asarray(public["train_sc_type_ids"]).astype(str)
    matched_type_ids_all = np.asarray(public["matched_sc_type_ids"]).astype(str)
    wrong_donor_ids_all = train_sc_donor_ids[wrong_index]
    wrong_type_ids_all = train_sc_type_ids[wrong_index]
    wrong_cell_ids_all = np.asarray(public["train_sc_cell_ids"]).astype(str)[wrong_index]
    # M0 capacity and routing vocabulary are training-only.  Query-reference or
    # wrong-bank support must never reconfigure the H&E baseline.
    all_types = tuple(sorted(set(train_sc_type_ids.tolist())))
    if len(all_types) < 2:
        raise RuntimeError("fewer than two training-reference types are available")
    train_type_keep = np.isin(train_sc_type_ids, all_types)
    matched_type_keep = np.isin(matched_type_ids_all, all_types)
    wrong_type_keep = np.isin(wrong_type_ids_all, all_types)
    wrong_latent = train_sc_latent[wrong_index][wrong_type_keep]
    wrong_donor_ids = wrong_donor_ids_all[wrong_type_keep]
    wrong_type_ids = wrong_type_ids_all[wrong_type_keep]
    wrong_cell_ids = wrong_cell_ids_all[wrong_type_keep]

    matched_mixture = _call_with_supported_kwargs(
        core.build_reference_mixture,
        matched_latent[matched_type_keep],
        type_ids=matched_type_ids_all[matched_type_keep],
        donor_ids=np.asarray(public["matched_sc_donor_ids"]).astype(str)[matched_type_keep],
        observation_ids=np.asarray(public["matched_sc_cell_ids"]).astype(str)[
            matched_type_keep
        ],
        source_modality="single_cell",
        n_components=3,
        seed=seed,
    )
    if type(matched_mixture).__name__ != core.ReferenceMixture.__name__:
        raise RuntimeError("matched reference must be a ReferenceMixture, not a centroid")
    wrong_mixtures = {
        donor: _call_with_supported_kwargs(
            core.build_reference_mixture,
            wrong_latent[wrong_donor_ids == donor],
            type_ids=wrong_type_ids[wrong_donor_ids == donor],
            donor_ids=wrong_donor_ids[wrong_donor_ids == donor],
            observation_ids=wrong_cell_ids[wrong_donor_ids == donor],
            source_modality="single_cell",
            n_components=3,
            seed=seed,
        )
        for donor in np.asarray(public["wrong_donor_ids"]).astype(str)
    }
    generic_natural = _call_with_supported_kwargs(
        core.build_reference_mixture,
        wrong_latent,
        type_ids=wrong_type_ids,
        donor_ids=wrong_donor_ids,
        observation_ids=wrong_cell_ids,
        source_modality="single_cell",
        donor_equal=True,
        n_components=3,
        seed=seed,
    )
    generic_mixture = _donor_equal_generic_reference(
        core, generic_natural, _scalar_text(public["heldout_donor"])
    )

    composition_proxy, proxy_types = training_only_composition_proxy(
        train_counts,
        sc_counts[train_type_keep],
        train_sc_type_ids[train_type_keep],
    )
    if proxy_types != all_types:
        raise RuntimeError("training composition proxy type order is inconsistent")
    donor_type_alignment = donor_type_proxy_alignment_statistics(
        train_st_latent,
        np.asarray(public["train_donor_ids"]),
        composition_proxy,
        train_sc_latent[train_type_keep],
        train_sc_donor_ids[train_type_keep],
        train_sc_type_ids[train_type_keep],
        all_types,
    )
    unaligned_donor_type_alignment = donor_type_proxy_alignment_statistics(
        unaligned_train_st_latent,
        np.asarray(public["train_donor_ids"]),
        composition_proxy,
        unaligned_train_sc_latent[train_type_keep],
        train_sc_donor_ids[train_type_keep],
        train_sc_type_ids[train_type_keep],
        all_types,
    )
    donor_type_support = bool(
        donor_type_alignment.get("evaluable") is True
        and unaligned_donor_type_alignment.get("evaluable") is True
        and float(donor_type_alignment["matched_to_mismatched_ratio"]) < 1.0
        and float(donor_type_alignment["matched_to_mismatched_ratio"])
        < float(unaligned_donor_type_alignment["matched_to_mismatched_ratio"])
    )
    diagnostic_parameters.update(
        {
            "cross_assay_donor_type_proxy_evaluable": np.asarray(
                donor_type_alignment.get("evaluable") is True
            ),
            "cross_assay_donor_type_proxy_type_names": np.asarray(
                donor_type_alignment.get("evaluable_types", ()), dtype=str
            ),
            "cross_assay_donor_type_proxy_pairs": np.asarray(
                donor_type_alignment.get("donor_type_pairs", 0), dtype=np.int64
            ),
            "cross_assay_donor_type_proxy_matched_MSE": np.asarray(
                donor_type_alignment.get("matched_MSE", np.nan), dtype=np.float32
            ),
            "cross_assay_donor_type_proxy_mismatched_MSE": np.asarray(
                donor_type_alignment.get("mismatched_MSE", np.nan), dtype=np.float32
            ),
            "cross_assay_donor_type_proxy_ratio": np.asarray(
                donor_type_alignment.get("matched_to_mismatched_ratio", np.nan),
                dtype=np.float32,
            ),
            "cross_assay_donor_type_proxy_unaligned_ratio": np.asarray(
                unaligned_donor_type_alignment.get(
                    "matched_to_mismatched_ratio", np.nan
                ),
                dtype=np.float32,
            ),
            "cross_assay_donor_type_proxy_support_criterion_met": np.asarray(
                donor_type_support
            ),
            "cross_assay_donor_type_proxy_scope": np.asarray(
                "training_only_composition_weighted_ST_proxy_not_observed_type_truth"
            ),
        }
    )
    del unaligned_train_st_latent, unaligned_train_sc_latent
    type_anchor_means = np.vstack(
        [
            np.mean(train_sc_latent[train_sc_type_ids == type_name], axis=0)
            for type_name in all_types
        ]
    ).astype(np.float32)

    state = _instantiate(
        core.CompositionStateModel,
        image_dim=np.asarray(public["train_image"]).shape[1],
        input_dim=np.asarray(public["train_image"]).shape[1],
        type_labels=all_types,
        n_genes=genes + 1,
        latent_dim=latent_dim,
        n_types=len(all_types),
        type_count=len(all_types),
        hidden_dim=min(256, max(32, latent_dim * 4)),
    )
    state.to(torch.device(device))
    state_fit = getattr(state, "fit_model", None) or getattr(state, "fit", None)
    if state_fit is None:
        raise RuntimeError("CompositionStateModel exposes no bounded fit method")
    _call_with_supported_kwargs(
        state_fit,
        np.asarray(public["train_image"], dtype=np.float32),
        train_st_latent,
        composition_targets=composition_proxy,
        routing_targets=composition_proxy,
        type_ids=all_types,
        type_anchor_means=type_anchor_means,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    variance_receipt = state.variance_calibration_receipt()
    diagnostic_parameters.update(
        {
            "H_state_variance_calibration_method": np.asarray(
                variance_receipt["method"]
            ),
            "H_state_variance_calibration_rows": np.asarray(
                variance_receipt["rows"], dtype=np.int64
            ),
            "H_state_variance_calibration_NLL_before": np.asarray(
                variance_receipt["nll_before"], dtype=np.float32
            ),
            "H_state_variance_calibration_NLL_after": np.asarray(
                variance_receipt["nll_after"], dtype=np.float32
            ),
            "H_state_calibrated_variance_by_type": np.asarray(
                variance_receipt["per_type_variance"], dtype=np.float32
            ),
        }
    )

    # M5 coordinates owns a separately trained branch with the same parameter
    # budget as the H&E state branch.  Coordinates are never padded through the
    # H-trained branch.
    coordinate_state = _instantiate(
        core.CompositionStateModel,
        image_dim=np.asarray(public["train_image"]).shape[1],
        input_dim=np.asarray(public["train_image"]).shape[1],
        type_labels=all_types,
        n_genes=genes + 1,
        latent_dim=latent_dim,
        n_types=len(all_types),
        type_count=len(all_types),
        hidden_dim=min(256, max(32, latent_dim * 4)),
    )
    coordinate_state.to(torch.device(device))
    coordinate_fit = getattr(coordinate_state, "fit_model", None) or getattr(
        coordinate_state, "fit", None
    )
    if coordinate_fit is None:
        raise RuntimeError("coordinate CompositionStateModel exposes no bounded fit method")
    train_coordinates, query_coordinates = _normalized_coordinates(
        np.asarray(public["train_coordinates"], dtype=np.float32),
        np.asarray(public["query_coordinates"], dtype=np.float32),
    )
    coordinate_train_input = np.zeros_like(np.asarray(public["train_image"], dtype=np.float32))
    coordinate_query_input = np.zeros_like(np.asarray(public["query_image"], dtype=np.float32))
    coordinate_train_input[:, : train_coordinates.shape[1]] = train_coordinates
    coordinate_query_input[:, : query_coordinates.shape[1]] = query_coordinates
    _call_with_supported_kwargs(
        coordinate_fit,
        coordinate_train_input,
        train_st_latent,
        composition_targets=composition_proxy,
        routing_targets=composition_proxy,
        type_ids=all_types,
        type_anchor_means=type_anchor_means,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        seed=seed + 50_000,
    )
    coordinate_variance_receipt = coordinate_state.variance_calibration_receipt()
    diagnostic_parameters.update(
        {
            "coordinate_state_variance_calibration_NLL_before": np.asarray(
                coordinate_variance_receipt["nll_before"], dtype=np.float32
            ),
            "coordinate_state_variance_calibration_NLL_after": np.asarray(
                coordinate_variance_receipt["nll_after"], dtype=np.float32
            ),
        }
    )

    def state_details(
        image: np.ndarray, reference: object | None, mode: str
    ) -> dict[str, np.ndarray]:
        if mode == "reference_only":
            if reference is None or not all(
                hasattr(reference, name) for name in ("type_names", "type_means", "type_weights")
            ):
                raise RuntimeError("M1 requires a type-aware reference distribution")
            names = tuple(reference.type_names)
            natural = dict(zip(names, np.asarray(reference.type_weights(), dtype=np.float64)))
            means = np.asarray(type_anchor_means, dtype=np.float64).copy()
            variances = np.zeros_like(means)
            weights = np.asarray([natural.get(name, 0.0) for name in all_types])
            entropies = np.zeros(len(all_types), dtype=np.float64)
            for type_index, type_name in enumerate(all_types):
                component_index = np.flatnonzero(reference.type_labels == type_name)
                if not len(component_index):
                    continue
                local_weight = np.asarray(reference.weights[component_index], dtype=np.float64)
                local_weight /= local_weight.sum()
                local_mean = np.asarray(reference.means[component_index], dtype=np.float64)
                local_variance = np.asarray(
                    reference.variances[component_index], dtype=np.float64
                )
                means[type_index] = np.sum(local_weight[:, None] * local_mean, axis=0)
                variances[type_index] = np.sum(
                    local_weight[:, None]
                    * (local_variance + (local_mean - means[type_index]) ** 2),
                    axis=0,
                )
                entropies[type_index] = -np.sum(
                    local_weight * np.log(np.maximum(local_weight, 1.0e-12))
                )
            if weights.sum() <= 0:
                raise RuntimeError("M1 reference has no training-vocabulary type support")
            weights /= weights.sum()
            composition = np.repeat(weights[None], len(image), axis=0)
            type_mean = np.repeat(means[None], len(image), axis=0).astype(np.float32)
            type_variance = np.repeat(variances[None], len(image), axis=0).astype(np.float32)
            latent = np.sum(composition[:, :, None] * type_mean, axis=1)
            return {
                "composition": composition.astype(np.float32),
                "type_mean": type_mean,
                "type_variance": type_variance,
                "latent": latent.astype(np.float32),
                "reference_entropy": np.repeat(entropies[None], len(image), axis=0).astype(
                    np.float32
                ),
            }
        method = getattr(state, "predict_details_numpy", None)
        if method is None:
            raise RuntimeError("CompositionStateModel exposes no type-specific predictor")
        value = _call_with_supported_kwargs(
            method,
            np.asarray(image, dtype=np.float32),
            reference=reference,
            reference_mixture=reference,
            mode=mode,
            device=device,
        )
        if not isinstance(value, Mapping):
            raise RuntimeError("CompositionStateModel details must be a mapping")
        result = {
            name: (
                item.detach().cpu().numpy().astype(np.float32)
                if torch.is_tensor(item)
                else np.asarray(item, dtype=np.float32)
            )
            for name, item in value.items()
        }
        if (
            result.get("latent", np.empty(0)).shape != (len(image), latent_dim)
            or result.get("composition", np.empty(0)).shape != (len(image), len(all_types))
            or result.get("type_mean", np.empty(0)).shape
            != (len(image), len(all_types), latent_dim)
        ):
            raise RuntimeError(f"CompositionStateModel returned malformed {mode} details")
        return result

    def state_predict(image: np.ndarray, reference: object | None, mode: str) -> np.ndarray:
        return state_details(image, reference, mode)["latent"]

    def composition_predict(image: np.ndarray) -> np.ndarray:
        method = getattr(state, "predict_composition_numpy", None) or getattr(
            state, "predict_composition", None
        )
        if method is None:
            details = getattr(state, "predict_details_numpy", None)
            if details is None:
                raise RuntimeError(
                    "CompositionStateModel must expose H-derived composition for exact M4 strata"
                )
            value = _call_with_supported_kwargs(details, image, mode="image_only", device=device)
            if not isinstance(value, Mapping) or "composition" not in value:
                raise RuntimeError("composition details omit the H-derived composition")
            value = value["composition"]
        else:
            value = _call_with_supported_kwargs(method, image, device=device)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (len(image), len(all_types)):
            raise RuntimeError("H-derived composition prediction is malformed")
        return result

    def coordinate_details(
        reference: object | None, mode: str
    ) -> dict[str, np.ndarray]:
        method = getattr(coordinate_state, "predict_details_numpy", None)
        if method is None:
            raise RuntimeError(
                "coordinate CompositionStateModel exposes no type-specific predictor"
            )
        value = _call_with_supported_kwargs(
            method,
            coordinate_query_input,
            reference=reference,
            reference_mixture=reference,
            mode=mode,
            device=device,
        )
        if not isinstance(value, Mapping):
            raise RuntimeError("coordinate details must be a mapping")
        result = {
            name: (
                item.detach().cpu().numpy().astype(np.float32)
                if torch.is_tensor(item)
                else np.asarray(item, dtype=np.float32)
            )
            for name, item in value.items()
        }
        if result.get("type_mean", np.empty(0)).shape != (
            len(coordinate_query_input),
            len(all_types),
            latent_dim,
        ):
            raise RuntimeError("coordinate branch returned malformed type states")
        return result

    query_image = np.asarray(public["query_image"], dtype=np.float32)
    query_composition = composition_predict(query_image)
    train_h_composition = composition_predict(
        np.asarray(public["train_image"], dtype=np.float32)
    )
    diagnostic_parameters["diagnostic_composition_support_threshold"] = np.asarray(
        np.quantile(np.max(train_h_composition, axis=1), 0.01), dtype=np.float32
    )
    shuffle_index, composition_strata = _composition_stratified_derangement(
        np.asarray(public["query_section_ids"]), query_composition, seed + 40_000
    )
    details_m0 = state_details(query_image, None, "image_only")
    details_m3 = state_details(query_image, matched_mixture, "full_poe")
    details_m1 = state_details(
        np.asarray(public["query_blank_image"]), matched_mixture, "reference_only"
    )
    details_m2 = state_details(query_image, matched_mixture, "composition_reference_mean")
    details_m4 = state_details(query_image[shuffle_index], matched_mixture, "full_poe")
    details_blank = state_details(
        np.asarray(public["query_blank_image"]), matched_mixture, "full_poe"
    )
    details_coordinates_image = coordinate_details(None, "image_only")
    details_coordinates = coordinate_details(matched_mixture, "full_poe")
    details_wrong = {
        donor: state_details(query_image, mixture, "full_poe")
        for donor, mixture in wrong_mixtures.items()
    }
    details_generic = state_details(query_image, generic_mixture, "full_poe")

    def component_posterior(
        image_details: Mapping[str, np.ndarray],
        output_details: Mapping[str, np.ndarray],
        reference: object | None,
        mode: str,
    ) -> dict[str, np.ndarray]:
        """Keep every donor/type state component through nonlinear decoding."""

        rows, types = np.asarray(output_details["composition"]).shape
        if mode in {"image_only", "composition_reference_mean"}:
            maximum = 1
        elif reference is None:
            maximum = 1
        else:
            maximum = max(
                1,
                max(
                    np.count_nonzero(np.asarray(reference.type_labels) == name)
                    for name in all_types
                ),
            )
        component_mean = np.zeros((rows, types, maximum, latent_dim), dtype=np.float32)
        component_variance = np.zeros_like(component_mean)
        component_weight = np.zeros((rows, types, maximum), dtype=np.float32)
        image_mean = np.asarray(image_details["type_mean"], dtype=np.float64)
        image_variance = np.asarray(image_details["type_variance"], dtype=np.float64)
        for type_index, type_name in enumerate(all_types):
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
                precision = (
                    1.0 / image_variance[:, type_index, None]
                    + 1.0 / reference_variance[None]
                )
                local_variance = 1.0 / precision
                local_mean = local_variance * (
                    image_mean[:, type_index, None] / image_variance[:, type_index, None]
                    + reference_mean[None] / reference_variance[None]
                )
                overlap_variance = (
                    image_variance[:, type_index, None] + reference_variance[None]
                )
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
                component_variance[:, type_index, :count] = local_variance.astype(
                    np.float32
                )
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

    blank_image_details = state_details(
        np.asarray(public["query_blank_image"]), None, "image_only"
    )
    shuffled_image_details = state_details(query_image[shuffle_index], None, "image_only")
    matched_component_count = np.asarray(
        [np.count_nonzero(matched_mixture.type_labels == name) for name in all_types],
        dtype=np.int64,
    )
    component_details_by_arm = {
        "M0": component_posterior(details_m0, details_m0, None, "image_only"),
        "M1": component_posterior(details_m0, details_m1, matched_mixture, "reference_only"),
        "M2": component_posterior(
            details_m0, details_m2, matched_mixture, "composition_reference_mean"
        ),
        "M3": component_posterior(details_m0, details_m3, matched_mixture, "full_poe"),
        "M4": component_posterior(
            shuffled_image_details, details_m4, matched_mixture, "full_poe"
        ),
        "M5_blank": component_posterior(
            blank_image_details, details_blank, matched_mixture, "full_poe"
        ),
        "M5_coordinates": component_posterior(
            details_coordinates_image,
            details_coordinates,
            matched_mixture,
            "full_poe",
        ),
        "M7": component_posterior(details_m0, details_generic, generic_mixture, "full_poe"),
    }
    # M2 on the full target necessarily falls back to H&E state when a matched
    # donor bank lacks a training-vocabulary type.  Preserve that descriptive
    # arm, but use a separate, support-matched pair for Gate 3 so the comparison
    # isolates continuous state rather than missing-type handling.
    gate3_supported_type = matched_component_count >= 2
    gate3_supported_mass = np.sum(
        query_composition * gate3_supported_type[None], axis=1, dtype=np.float64
    )
    gate3_composition = query_composition * gate3_supported_type[None]
    positive_support = gate3_supported_mass > 0
    gate3_composition[positive_support] /= gate3_supported_mass[positive_support, None]
    gate3_composition[~positive_support] = query_composition[~positive_support]
    for arm in ("M2", "M3"):
        conditional = {
            name: np.asarray(value).copy()
            for name, value in component_details_by_arm[arm].items()
        }
        conditional["composition"] = gate3_composition.astype(np.float32)
        component_details_by_arm[f"{arm}_supported"] = conditional
    wrong_component_details = {
        donor: component_posterior(details_m0, details_wrong[donor], mixture, "full_poe")
        for donor, mixture in wrong_mixtures.items()
    }

    matched_type_cell_count = np.asarray(
        [np.count_nonzero(matched_type_ids_all == name) for name in all_types],
        dtype=np.int64,
    )
    matched_type_ess = np.zeros(len(all_types), dtype=np.float32)
    state_supported = np.zeros((len(query_image), len(all_types)), dtype=bool)
    minimum_distance = np.full((len(query_image), len(all_types)), -1.0, dtype=np.float32)
    for type_index, type_name in enumerate(all_types):
        component_index = np.flatnonzero(matched_mixture.type_labels == type_name)
        if len(component_index) < 2:
            continue
        local_weight = np.asarray(matched_mixture.weights[component_index], dtype=np.float64)
        local_weight /= local_weight.sum()
        matched_type_ess[type_index] = float(1.0 / np.sum(local_weight**2))
        difference = (
            np.asarray(details_m0["type_mean"], dtype=np.float64)[:, type_index, None]
            - np.asarray(matched_mixture.means[component_index], dtype=np.float64)[None]
        )
        denominator = (
            np.asarray(details_m0["type_variance"], dtype=np.float64)[:, type_index, None]
            + np.asarray(matched_mixture.variances[component_index], dtype=np.float64)[None]
        )
        distance = np.mean(difference**2 / np.maximum(denominator, 1.0e-6), axis=2)
        minimum_distance[:, type_index] = np.min(distance, axis=1).astype(np.float32)
        state_supported[:, type_index] = minimum_distance[:, type_index] <= 4.0
    reference_coverage_mass = np.sum(
        np.asarray(details_m0["composition"], dtype=np.float64) * state_supported,
        axis=1,
    )
    reference_abstain = reference_coverage_mass < 0.90
    entropy_normalizer = np.log(np.maximum(matched_component_count, 2))
    m1_supported = matched_component_count > 0
    m1_supported_mass = np.sum(
        np.asarray(details_m1["composition"], dtype=np.float64) * m1_supported[None], axis=1
    )
    m1_entropy_numerator = np.sum(
        np.asarray(details_m1["composition"], dtype=np.float64)
        * np.asarray(details_m1["reference_entropy"], dtype=np.float64)
        / entropy_normalizer[None],
        axis=1,
    )
    m1_reference_entropy = np.divide(
        m1_entropy_numerator,
        m1_supported_mass,
        out=np.full_like(m1_entropy_numerator, np.nan),
        where=m1_supported_mass > 0,
    )
    m3_supported = matched_component_count >= 2
    m3_supported_mass = np.sum(
        np.asarray(details_m3["composition"], dtype=np.float64) * m3_supported[None], axis=1
    )
    m3_entropy_numerator = np.sum(
        np.asarray(details_m3["composition"], dtype=np.float64)
        * np.asarray(details_m3["reference_entropy"], dtype=np.float64)
        / entropy_normalizer[None],
        axis=1,
    )
    m3_reference_entropy = np.divide(
        m3_entropy_numerator,
        m3_supported_mass,
        out=np.full_like(m3_entropy_numerator, np.nan),
        where=m3_supported_mass > 0,
    )
    wrong_type_cell_count = np.asarray(
        [
            [np.count_nonzero((wrong_donor_ids == donor) & (wrong_type_ids == name))
             for name in all_types]
            for donor in np.asarray(public["wrong_donor_ids"]).astype(str)
        ],
        dtype=np.int64,
    )
    wrong_type_component_count = np.asarray(
        [
            [np.count_nonzero(wrong_mixtures[donor].type_labels == name) for name in all_types]
            for donor in np.asarray(public["wrong_donor_ids"]).astype(str)
        ],
        dtype=np.int64,
    )
    diagnostic_parameters.update(
        {
            "matched_reference_component_count_by_type": matched_component_count,
            "matched_reference_cell_count_by_type": matched_type_cell_count,
            "matched_reference_component_ESS_by_type": matched_type_ess,
            "matched_reference_state_support_threshold": np.asarray(4.0, dtype=np.float32),
            "matched_reference_state_support_threshold_status": np.asarray(
                "fixed_development_exploratory_not_inner_fold_calibrated"
            ),
            "matched_reference_abstention_coverage_mass_threshold": np.asarray(
                0.90, dtype=np.float32
            ),
            "matched_reference_abstention_threshold_status": np.asarray(
                "fixed_development_exploratory_not_inner_fold_calibrated"
            ),
            "query_reference_min_normalized_mahalanobis": minimum_distance,
            "query_reference_coverage_mass": reference_coverage_mass.astype(np.float32),
            "query_reference_abstain": reference_abstain,
            "M1_reference_entropy_normalized": m1_reference_entropy.astype(np.float32),
            "M3_reference_entropy_normalized": m3_reference_entropy.astype(np.float32),
            "wrong_reference_cell_count_by_type": wrong_type_cell_count,
            "wrong_reference_component_count_by_type": wrong_type_component_count,
            "gate3_supported_type_mask": gate3_supported_type,
            "gate3_supported_composition_mass": gate3_supported_mass.astype(np.float32),
            "gate3_minimum_supported_composition_mass": np.asarray(
                0.90, dtype=np.float32
            ),
            "gate3_minimum_eligible_spots_per_section": np.asarray(3, dtype=np.int64),
            "gate3_supported_score_eligible": (
                gate3_supported_mass >= 0.90
            ),
            "gate3_comparison_scope": np.asarray(
                "matched_types_with_at_least_two_components_identical_renormalized_"
                "H_and_E_composition_in_M2_supported_and_M3_supported"
            ),
        }
    )

    hard_negative_index = biologically_hard_negative_indices(
        np.asarray(public["train_donor_ids"]),
        np.asarray(public["train_indication_ids"]),
        composition_proxy,
        train_st_latent,
    )
    hard_negative_same_indication = (
        np.asarray(public["train_indication_ids"]).astype(str)[hard_negative_index]
        == np.asarray(public["train_indication_ids"]).astype(str)
    )
    dominant_training_composition = np.argmax(composition_proxy, axis=1)
    hard_negative_same_composition = (
        dominant_training_composition[hard_negative_index]
        == dominant_training_composition
    )
    hard_negative_primary = hard_negative_same_indication & hard_negative_same_composition
    hard_negative_indication_fallback = (
        hard_negative_same_indication & ~hard_negative_same_composition
    )
    hard_negative_global_fallback = ~hard_negative_same_indication
    diagnostic_parameters.update(
        {
            "BLEEP_temperature": np.asarray(0.07, dtype=np.float32),
            "BLEEP_hard_negative_weight": np.asarray(0.5, dtype=np.float32),
            "BLEEP_hard_negative_strategy": np.asarray(
                "prioritize_same_indication_same_dominant_composition_different_"
                "donor_then_same_indication_then_global_emergency_fallback_"
                "max_fixed_latent_projection_difference"
            ),
            "BLEEP_hard_negative_index": hard_negative_index,
            "BLEEP_hard_negative_primary_count": np.asarray(
                hard_negative_primary.sum(), dtype=np.int64
            ),
            "BLEEP_hard_negative_primary_fraction": np.asarray(
                hard_negative_primary.mean(), dtype=np.float32
            ),
            "BLEEP_hard_negative_same_indication_fallback_count": np.asarray(
                hard_negative_indication_fallback.sum(), dtype=np.int64
            ),
            "BLEEP_hard_negative_same_indication_fallback_fraction": np.asarray(
                hard_negative_indication_fallback.mean(), dtype=np.float32
            ),
            "BLEEP_hard_negative_global_fallback_count": np.asarray(
                hard_negative_global_fallback.sum(), dtype=np.int64
            ),
            "BLEEP_hard_negative_global_fallback_fraction": np.asarray(
                hard_negative_global_fallback.mean(), dtype=np.float32
            ),
        }
    )
    retrieval = _instantiate(
        core.ContrastiveRetrieval,
        image_dim=query_image.shape[1],
        latent_dim=latent_dim,
        molecular_dim=latent_dim,
        embedding_dim=latent_dim,
    )
    retrieval.to(torch.device(device))
    retrieval_fit = getattr(retrieval, "fit_model", None) or getattr(retrieval, "fit", None)
    if retrieval_fit is None:
        raise RuntimeError("ContrastiveRetrieval exposes no bounded fit method")
    _call_with_supported_kwargs(
        retrieval_fit,
        np.asarray(public["train_image"], dtype=np.float32),
        train_st_latent,
        hard_negative_molecular=train_st_latent[hard_negative_index],
        hard_negative_weight=0.5,
        observation_ids=np.asarray(public["train_spot_ids"]).astype(str),
        heldout_observation_ids=np.asarray(public["query_spot_ids"]).astype(str),
        epochs=epochs,
        batch_size=batch_size,
        temperature=0.07,
        device=device,
        seed=seed,
    )
    retrieve = getattr(retrieval, "retrieve_numpy", None) or getattr(retrieval, "retrieve", None)
    if retrieve is None:
        raise RuntimeError("ContrastiveRetrieval exposes no retrieval method")
    latent_bleep = _call_with_supported_kwargs(
        retrieve,
        query_image,
        matched_latent,
        device=device,
        return_entropy=True,
        temperature=0.07,
    )
    if not isinstance(latent_bleep, tuple) or len(latent_bleep) != 2:
        raise RuntimeError("ContrastiveRetrieval must return retrieval entropy")
    latent_bleep, retrieval_entropy = latent_bleep
    if torch.is_tensor(latent_bleep):
        latent_bleep = latent_bleep.detach().cpu().numpy()
    latent_bleep = np.asarray(latent_bleep, dtype=np.float32)

    def decode_type_mixture(
        details: Mapping[str, np.ndarray], *, decode_seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        composition = np.asarray(details["composition"], dtype=np.float32)
        component_mean = np.asarray(details["component_mean"], dtype=np.float32)
        component_variance = np.asarray(details["component_variance"], dtype=np.float32)
        component_weight = np.asarray(details["component_weight"], dtype=np.float32)
        rows, types, components, _ = component_mean.shape
        method = getattr(vae, "decode_diagonal_gaussian_numpy", None)
        if method is None:
            raise RuntimeError("CountVAE lacks posterior decoder integration")
        moments = method(
            component_mean.reshape(-1, latent_dim),
            component_variance.reshape(-1, latent_dim),
            library_size=np.ones(rows * types * components, dtype=np.float32),
            modality="st",
            endpoint_gene_indices=np.arange(genes),
            dispersion=dispersion,
            samples=32,
            batch_size=batch_size,
            seed=decode_seed,
        )
        if not isinstance(moments, Mapping):
            raise RuntimeError("posterior decoder integration returned malformed moments")
        component_rate = np.asarray(moments["mean_counts"], dtype=np.float32).reshape(
            rows, types, components, genes
        )
        component_rate_variance = np.asarray(
            moments["latent_variance_counts"], dtype=np.float32
        ).reshape(rows, types, components, genes)
        type_rate = np.sum(component_weight[..., None] * component_rate, axis=2)
        type_rate_variance = np.sum(
            component_weight[..., None]
            * (component_rate_variance + component_rate**2),
            axis=2,
        ) - type_rate**2
        mean_rate = np.sum(composition[:, :, None] * type_rate, axis=1)
        rate_variance = np.sum(
            composition[:, :, None] ** 2 * np.maximum(type_rate_variance, 0.0), axis=1
        )
        return mean_rate.astype(np.float32), rate_variance.astype(np.float32)

    decoded = {
        arm: decode_type_mixture(details, decode_seed=seed + 60_000)
        for arm, details in component_details_by_arm.items()
    }
    predictions = {arm: value[0] for arm, value in decoded.items()}
    rate_variances = {
        arm: decoded[arm][1] for arm in ("M0", "M1", "M2", "M3")
    }
    diagnostic_parameters.update(
        {
            "rate_M2_supported": decoded["M2_supported"][0],
            "rate_M3_supported": decoded["M3_supported"][0],
            "posterior_decode_representation": np.asarray(
                "componentwise_reference_PoE_no_moment_collapse_before_decoder"
            ),
            "posterior_decode_Sobol_samples": np.asarray(32, dtype=np.int64),
            "posterior_decode_seed": np.asarray(seed + 60_000, dtype=np.int64),
            "posterior_uncertainty_scope": np.asarray(
                "state_and_NB2_only_composition_uncertainty_not_propagated"
            ),
        }
    )
    predictions["BLEEP"] = _decode_rate(
        vae, latent_bleep, endpoint_genes=genes, device=device
    )
    predictions["M6"] = np.stack(
        [
            decode_type_mixture(wrong_component_details[donor], decode_seed=seed + 60_000)[0]
            for donor in np.asarray(public["wrong_donor_ids"]).astype(str)
        ],
        axis=0,
    )
    return _prediction_receipt_arrays(
        _scalar_text(public["heldout_donor"]),
        public,
        predictions,
        rate_variances,
        dispersion,
        m8_parameters,
        diagnostic_parameters,
        shuffle_index,
        composition_strata,
        query_composition,
        np.asarray(retrieval_entropy, dtype=np.float32),
        all_types,
        tuple(sorted(set(matched_type_ids_all.tolist()))),
        float(matched_mixture.effective_sample_size()),
        int(np.count_nonzero(matched_type_keep)),
    )


def fit_predict(args: argparse.Namespace) -> Mapping[str, object]:
    if not args.smoke:
        _validate_args(args)
    manifest = _read_prepared_manifest(args.output)
    if int(args.seed) != int(manifest.get("base_seed", -1)):
        raise ValueError("fit-predict seed differs from the hash-bound prepared seed")
    _validate_bound_training_configuration(args, manifest)
    core = _import_core()
    runner_sha256 = _sha256(Path(__file__).resolve())
    core_sha256 = _sha256(Path(core.__file__).resolve())
    receipts: dict[str, object] = {}
    for donor in manifest["donors"]:
        donor = str(donor)
        prepared = manifest["folds"][donor]
        fold_seed = int(prepared["seed"])
        public_path = Path(str(prepared["public_path"]))
        public = _verify_semantic_file(public_path, str(prepared["public_semantic_sha256"]))
        validate_public_fold(public)
        identity = _checkpoint_identity(
            str(prepared["public_semantic_sha256"]),
            donor=donor,
            seed=fold_seed,
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            device=args.device,
            runner_sha256=runner_sha256,
            core_sha256=core_sha256,
            protocol_sha256=str(manifest["protocol_sha256"]),
        )
        fold_dir = args.output / "folds" / donor
        prediction_path = fold_dir / "predictions.npz"
        receipt_path = fold_dir / "fit_predict_receipt.json"
        if args.resume and prediction_path.is_file() and receipt_path.is_file():
            old = json.loads(receipt_path.read_text(encoding="utf-8"))
            if old.get("checkpoint_identity") == identity:
                arrays = _load_arrays(prediction_path)
                if _semantic_array_hash(arrays) == old.get("prediction_semantic_sha256"):
                    receipts[donor] = old
                    continue
        predictions = fit_predict_one_fold(
            public,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            seed=fold_seed,
        )
        _atomic_npz(prediction_path, predictions)
        receipt = {
            "schema": PREDICTION_SCHEMA,
            "heldout_donor": donor,
            "checkpoint_identity": identity,
            "public_semantic_sha256": prepared["public_semantic_sha256"],
            "prediction_path": str(prediction_path),
            "prediction_semantic_sha256": _semantic_array_hash(predictions),
            "heldout_ST_opened": False,
            "decoder_shared_M0_M3": True,
            "matched_reference_representation": "deterministic_multi_component_distribution",
            "reference_assay": "registered_suspension_type_cell_not_verified_snRNA",
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "runner_sha256": runner_sha256,
            "core_sha256": core_sha256,
            "protocol_sha256": manifest["protocol_sha256"],
            "artifact_complete": True,
            "scientific_implementation_complete": False,
            "implementation_status": (
                "primary_arm_predictions_complete; required_secondary_metrics_not_complete"
            ),
        }
        _atomic_json(receipt_path, receipt)
        receipts[donor] = receipt
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    aggregate = {
        "schema": PREDICTION_SCHEMA,
        "analysis_scope": "exposed_development_only_non_confirmatory",
        "folds": receipts,
        "all_folds_complete": len(receipts) == len(manifest["donors"]),
    }
    _atomic_json(args.output / "fit_predict_manifest.json", aggregate)
    return aggregate


def _nb_deviance_rows(
    core: Any, counts: np.ndarray, mean: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    value = _call_with_supported_kwargs(
        core.nb2_deviance,
        counts,
        mean,
        dispersion=theta,
        theta=theta,
        reduction="none",
    )
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape == counts.shape:
        result = result.mean(axis=1)
    if result.shape != (len(counts),):
        raise RuntimeError("nb2_deviance must return rows or a row/gene matrix")
    return result


def _nb_log_likelihood_rows(
    core: Any, counts: np.ndarray, mean: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    value = _call_with_supported_kwargs(
        core.nb2_log_prob,
        counts,
        mean,
        dispersion=theta,
        theta=theta,
        reduction="none",
    )
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape == counts.shape:
        result = result.mean(axis=1)
    if result.shape != (len(counts),):
        raise RuntimeError("nb2_log_prob must return rows or a row/gene matrix")
    return result


def _posterior_predictive_log_score_rows(
    core: Any,
    counts: np.ndarray,
    library: np.ndarray,
    rate_mean: np.ndarray,
    rate_variance: np.ndarray,
    theta: np.ndarray,
    *,
    quadrature_points: int = 8,
) -> np.ndarray:
    """Moment-matched lognormal quadrature for the marginal NB2 log score."""

    mean_rate = np.maximum(np.asarray(rate_mean, dtype=np.float64), 1.0e-10)
    variance_rate = np.maximum(np.asarray(rate_variance, dtype=np.float64), 0.0)
    sigma_squared = np.log1p(variance_rate / np.maximum(mean_rate**2, 1.0e-20))
    log_location = np.log(mean_rate) - 0.5 * sigma_squared
    nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_points))
    log_integral = np.full(mean_rate.shape, -np.inf, dtype=np.float64)
    for node, weight in zip(nodes, weights):
        sampled_rate = np.exp(log_location + np.sqrt(2.0 * sigma_squared) * node)
        sampled_mean = np.maximum(sampled_rate * library[:, None], 1.0e-8)
        value = _call_with_supported_kwargs(
            core.nb2_log_prob,
            counts,
            sampled_mean,
            dispersion=theta,
            theta=theta,
            reduction="none",
        )
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        log_integral = np.logaddexp(
            log_integral,
            np.asarray(value, dtype=np.float64) + math.log(float(weight) / math.sqrt(math.pi)),
        )
    return log_integral.mean(axis=1)


def _section_macro(values: np.ndarray, section_ids: np.ndarray) -> float:
    rows = np.asarray(values, dtype=np.float64)
    sections = np.asarray(section_ids).astype(str)
    if rows.shape != sections.shape or not len(rows):
        raise ValueError("section-macro values must be nonempty and row aligned")
    section_means = [
        float(np.mean(rows[sections == value])) for value in sorted(set(sections.tolist()))
    ]
    return float(np.mean(section_means))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.size < 2:
        return math.nan
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    return math.nan if denominator <= 0 else float(np.sum(first * second) / denominator)


def _finite_or_none(value: float) -> float | None:
    """Return a JSON-safe diagnostic without treating undefined as evidence."""

    number = float(value)
    return number if np.isfinite(number) else None


def _section_balanced_feature_correlation(
    observed: np.ndarray, predicted: np.ndarray, section_ids: np.ndarray
) -> float:
    values = []
    for section in sorted(set(np.asarray(section_ids).astype(str).tolist())):
        keep = np.asarray(section_ids).astype(str) == section
        local = [
            _correlation(observed[keep, gene], predicted[keep, gene])
            for gene in range(observed.shape[1])
        ]
        finite = [value for value in local if np.isfinite(value)]
        if finite:
            values.append(float(np.mean(finite)))
    return float(np.mean(values)) if values else math.nan


def _section_balanced_calibration_slope(
    core: Any, observed: np.ndarray, predicted: np.ndarray, section_ids: np.ndarray
) -> float:
    """Average truth-on-prediction slopes per gene, then per section."""

    section_values = []
    sections = np.asarray(section_ids).astype(str)
    for section in sorted(set(sections.tolist())):
        keep = sections == section
        gene_values = []
        for gene in range(observed.shape[1]):
            try:
                value = float(
                    core.calibration_slope(observed[keep, gene], predicted[keep, gene])
                )
            except ValueError:
                continue
            if np.isfinite(value):
                gene_values.append(value)
        if gene_values:
            section_values.append(float(np.mean(gene_values)))
    return float(np.mean(section_values)) if section_values else math.nan


def _quality_metrics(
    core: Any,
    counts: np.ndarray,
    library: np.ndarray,
    rate: np.ndarray,
    rate_variance: np.ndarray,
    theta: np.ndarray,
    sections: np.ndarray,
    secret: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
) -> Mapping[str, object]:
    """Compute the frozen secondary suite without updating any prediction."""

    scale = float(np.asarray(predictions["diagnostic_normalization_scale"]))
    observed = np.log1p(counts * (scale / library[:, None]))
    predicted = np.log1p(np.asarray(rate, dtype=np.float64) * scale)
    basis_mean = np.asarray(predictions["diagnostic_basis_mean"], dtype=np.float64)
    components = np.asarray(predictions["diagnostic_basis_components"], dtype=np.float64)
    score_scale = np.asarray(predictions["diagnostic_basis_score_scale"], dtype=np.float64)
    observed_latent = ((observed - basis_mean) @ components.T) / score_scale
    predicted_latent = ((predicted - basis_mean) @ components.T) / score_scale
    latent_rows = np.mean((observed_latent - predicted_latent) ** 2, axis=1)

    membership = np.asarray(predictions["diagnostic_program_membership"], dtype=bool)
    active = np.asarray(predictions["diagnostic_program_active"], dtype=bool)
    thresholds = np.asarray(
        predictions["diagnostic_rare_program_thresholds"], dtype=np.float64
    )
    observed_program = np.column_stack(
        [observed[:, membership[index]].mean(axis=1) for index in np.flatnonzero(active)]
    )
    predicted_program = np.column_stack(
        [predicted[:, membership[index]].mean(axis=1) for index in np.flatnonzero(active)]
    )
    program_correlation = _section_balanced_feature_correlation(
        observed_program, predicted_program, sections
    )
    covariance_errors = []
    for section in sorted(set(np.asarray(sections).astype(str).tolist())):
        keep = np.asarray(sections).astype(str) == section
        if int(keep.sum()) < 3 or observed_program.shape[1] < 2:
            continue
        truth_covariance = np.cov(observed_program[keep], rowvar=False)
        prediction_covariance = np.cov(predicted_program[keep], rowvar=False)
        covariance_errors.append(
            float(
                np.linalg.norm(prediction_covariance - truth_covariance)
                / max(np.linalg.norm(truth_covariance), 1.0e-8)
            )
        )

    rare_recalls = []
    active_indices = np.flatnonzero(active)
    for local_index, program_index in enumerate(active_indices):
        truth_positive = observed_program[:, local_index] >= thresholds[program_index]
        if truth_positive.any():
            predicted_positive = predicted_program[:, local_index] >= thresholds[program_index]
            rare_recalls.append(float(np.mean(predicted_positive[truth_positive])))

    first_library = np.maximum(
        np.asarray(secret["heldout_st_library_half_a"], dtype=np.float64)[
            np.asarray(secret["primary_score_eligible"], dtype=bool)
        ],
        1.0,
    )
    second_library = np.maximum(
        np.asarray(secret["heldout_st_library_half_b"], dtype=np.float64)[
            np.asarray(secret["primary_score_eligible"], dtype=bool)
        ],
        1.0,
    )
    first = np.log1p(
        np.asarray(secret["heldout_st_half_a"], dtype=np.float64)[
            np.asarray(secret["primary_score_eligible"], dtype=bool)
        ]
        * (scale / first_library[:, None])
    )
    second = np.log1p(
        np.asarray(secret["heldout_st_half_b"], dtype=np.float64)[
            np.asarray(secret["primary_score_eligible"], dtype=bool)
        ]
        * (scale / second_library[:, None])
    )
    training_reliable = np.asarray(
        predictions["diagnostic_training_reliable_gene"], dtype=bool
    )
    variance_ratios = []
    for section in sorted(set(np.asarray(sections).astype(str).tolist())):
        keep = np.asarray(sections).astype(str) == section
        if int(keep.sum()) < 3:
            continue
        covariance = np.mean(
            (first[keep] - first[keep].mean(axis=0))
            * (second[keep] - second[keep].mean(axis=0)),
            axis=0,
        )
        predicted_variance = np.var(predicted[keep], axis=0, ddof=1)
        valid = training_reliable & np.isfinite(covariance) & (covariance > 0)
        variance_ratios.extend((predicted_variance[valid] / covariance[valid]).tolist())

    mean_counts = np.maximum(rate * library[:, None], 1.0e-8)
    latent_count_variance = (
        np.asarray(rate_variance, dtype=np.float64) * library[:, None] ** 2
    )
    predictive_variance = (
        mean_counts
        + mean_counts**2 / theta[None]
        + latent_count_variance * (1.0 + 1.0 / theta[None])
    )
    standard_deviation = np.sqrt(np.maximum(predictive_variance, 1.0e-8))
    interval_coverage = {}
    for label, z_value in (("50", 0.67448975), ("80", 1.28155157), ("95", 1.95996398)):
        lower = np.maximum(mean_counts - z_value * standard_deviation, 0.0)
        upper = mean_counts + z_value * standard_deviation
        section_coverage = [
            float(np.mean((counts[sections == section] >= lower[sections == section])
                          & (counts[sections == section] <= upper[sections == section])))
            for section in sorted(set(sections.tolist()))
        ]
        interval_coverage[label] = float(np.mean(section_coverage))

    return {
        "latent_MSE_20D": _finite_or_none(_section_macro(latent_rows, sections)),
        "gene_correlation": _finite_or_none(
            _section_balanced_feature_correlation(observed, predicted, sections)
        ),
        "program_correlation": _finite_or_none(program_correlation),
        "calibration_slope": _finite_or_none(
            _section_balanced_calibration_slope(core, observed, predicted, sections)
        ),
        "reliability_adjusted_variance_median": _finite_or_none(
            float(np.median(variance_ratios)) if variance_ratios else math.nan
        ),
        "reliability_adjusted_variance_strata": len(variance_ratios),
        "program_covariance_relative_error": _finite_or_none(
            float(np.mean(covariance_errors)) if covariance_errors else math.nan
        ),
        "rare_program_state_recall": _finite_or_none(
            float(np.mean(rare_recalls)) if rare_recalls else math.nan
        ),
        "rare_programs_evaluable": len(rare_recalls),
        "predictive_interval_coverage": interval_coverage,
        "posterior_latent_count_variance_mean": float(np.mean(latent_count_variance)),
        "interval_method": (
            "latent_integrated_NB2_posterior_predictive_moment_normal_approximation"
        ),
    }


def _donor_effects(
    per_donor: Mapping[str, Mapping[str, float]], left: str, right: str
) -> np.ndarray:
    return np.asarray(
        [values[left] - values[right] for _, values in sorted(per_donor.items())], dtype=float
    )


def _sign_flip(core: Any, values: np.ndarray) -> Mapping[str, object]:
    result = _call_with_supported_kwargs(
        core.exact_sign_flip_test,
        values,
        alternative="greater",
    )
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "p_value"):
        return {
            "p_value": float(result.p_value),
            "observed": float(getattr(result, "statistic", np.mean(values))),
            "confidence_interval": [
                float(value) for value in getattr(result, "confidence_interval", (math.nan,) * 2)
            ],
            "positive_fraction": float(getattr(result, "positive_fraction", np.mean(values > 0))),
            "n_donors": int(getattr(result, "donors", len(values))),
        }
    return {"p_value": float(result), "observed": float(np.mean(values))}


def _donor_bootstrap_interval(
    values: np.ndarray, *, seed: int, replicates: int = 20_000
) -> tuple[float, float]:
    """Deterministic paired-donor percentile interval for a mean effect."""

    effects = np.asarray(values, dtype=np.float64)
    if effects.ndim != 1 or len(effects) < 2 or not np.isfinite(effects).all():
        raise ValueError("donor bootstrap requires at least two finite paired effects")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(replicates), dtype=np.float64)
    block = 2_000
    for start in range(0, int(replicates), block):
        stop = min(start + block, int(replicates))
        indices = rng.integers(0, len(effects), size=(stop - start, len(effects)))
        means[start:stop] = effects[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _score_fold(
    core: Any,
    secret: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> Mapping[str, object]:
    if _scalar_text(secret["heldout_donor"]) != _scalar_text(predictions["heldout_donor"]):
        raise ValueError("prediction and secret donor identities differ")
    if not np.array_equal(secret["heldout_spot_ids"], predictions["query_spot_ids"]):
        raise ValueError("prediction rows do not align to sealed score target")
    keep = np.asarray(secret["primary_score_eligible"], dtype=bool)
    counts = np.asarray(secret["heldout_st_counts"], dtype=np.float32)[keep]
    library = np.asarray(secret["heldout_st_library"], dtype=np.float32)[keep]
    sections = np.asarray(secret["heldout_section_ids"]).astype(str)[keep]
    theta = np.asarray(predictions["training_only_dispersion"], dtype=np.float32)
    losses: dict[str, float] = {}
    log_likelihoods: dict[str, float] = {}
    row_losses: dict[str, list[float]] = {}
    quality: dict[str, Mapping[str, object]] = {}
    posterior_log_scores: dict[str, float] = {}
    m6_control: dict[str, object] = {}
    for arm in MODEL_ARMS:
        if arm == "M6":
            candidate_rates = np.asarray(predictions["rate_M6_candidates"], dtype=np.float32)[
                :, keep
            ]
            candidates = candidate_rates * library[None, :, None]
            candidate_rows = [
                _nb_deviance_rows(core, counts, candidate, theta) for candidate in candidates
            ]
            candidate_log_rows = [
                _nb_log_likelihood_rows(core, counts, candidate, theta)
                for candidate in candidates
            ]
            candidate_macro = np.asarray(
                [_section_macro(value, sections) for value in candidate_rows]
            )
            selected = int(np.argmin(candidate_macro))
            donor_equal_rows = np.mean(np.stack(candidate_rows, axis=0), axis=0)
            losses[arm] = float(np.mean(candidate_macro))
            log_likelihoods[arm] = float(
                np.mean([_section_macro(value, sections) for value in candidate_log_rows])
            )
            row_losses[arm] = donor_equal_rows.tolist()
            m6_control = {
                "primary": "equal_mean_over_eligible_wrong_donor_losses",
                "candidate_donor_ids": np.asarray(predictions["wrong_donor_ids"])
                .astype(str)
                .tolist(),
                "candidate_mean_nb_deviance": candidate_macro.tolist(),
                "best_wrong_sensitivity_donor": str(
                    np.asarray(predictions["wrong_donor_ids"]).astype(str)[selected]
                ),
                "best_wrong_sensitivity_mean_nb_deviance": float(candidate_macro[selected]),
            }
            continue
        rate = np.asarray(predictions[f"rate_{arm}"], dtype=np.float32)[keep]
        mean = rate * library[:, None]
        rows = _nb_deviance_rows(core, counts, mean, theta)
        losses[arm] = _section_macro(rows, sections)
        log_likelihoods[arm] = _section_macro(
            _nb_log_likelihood_rows(core, counts, mean, theta), sections
        )
        row_losses[arm] = rows.tolist()
        if arm in {"M0", "M1", "M2", "M3"}:
            rate_variance = np.asarray(
                predictions[f"posterior_rate_variance_{arm}"], dtype=np.float32
            )[keep]
            posterior_log_scores[arm] = _section_macro(
                _posterior_predictive_log_score_rows(
                    core,
                    counts,
                    library,
                    rate,
                    rate_variance,
                    theta,
                ),
                sections,
            )
            quality[arm] = _quality_metrics(
                core,
                counts,
                library,
                rate,
                rate_variance,
                theta,
                sections,
                secret,
                predictions,
            )

    # Gate 3 is conditional on matched-reference support.  The two conditional
    # arms use the same H&E composition, renormalized over types with at least
    # two matched components, and are scored only where that supported mass was
    # at least 0.90 before the molecular target was opened.
    gate3_eligible = np.asarray(
        predictions["gate3_supported_score_eligible"], dtype=bool
    )[keep]
    minimum_gate3_spots = int(
        predictions["gate3_minimum_eligible_spots_per_section"]
    )
    gate3_section_counts = {
        section: int(np.sum(gate3_eligible & (sections == section)))
        for section in sorted(set(sections.tolist()))
    }
    donor_type_proxy_support = bool(
        predictions["cross_assay_donor_type_proxy_support_criterion_met"]
    )
    gate3_evaluable = bool(
        donor_type_proxy_support
        and gate3_section_counts
        and all(count >= minimum_gate3_spots for count in gate3_section_counts.values())
    )
    gate3_supported_comparison: dict[str, object] = {
        "evaluable": gate3_evaluable,
        "minimum_supported_composition_mass": float(
            predictions["gate3_minimum_supported_composition_mass"]
        ),
        "minimum_eligible_spots_per_section": minimum_gate3_spots,
        "eligible_spots": int(gate3_eligible.sum()),
        "eligible_fraction": float(np.mean(gate3_eligible)),
        "eligible_spots_by_section": gate3_section_counts,
        "training_only_donor_type_proxy_alignment_support": donor_type_proxy_support,
        "scope": _scalar_text(predictions["gate3_comparison_scope"]),
        "full_target_M3_vs_M2_role": "descriptive_mixed_support_not_used_for_gate3",
    }
    if gate3_evaluable:
        conditional_sections = sections[gate3_eligible]
        for arm in ("M2_supported", "M3_supported"):
            rate = np.asarray(predictions[f"rate_{arm}"], dtype=np.float32)[keep][
                gate3_eligible
            ]
            mean = rate * library[gate3_eligible, None]
            rows = _nb_deviance_rows(core, counts[gate3_eligible], mean, theta)
            losses[arm] = _section_macro(rows, conditional_sections)
            log_likelihoods[arm] = _section_macro(
                _nb_log_likelihood_rows(core, counts[gate3_eligible], mean, theta),
                conditional_sections,
            )
        gate3_supported_comparison.update(
            {
                "M2_supported_mean_nb_deviance": losses["M2_supported"],
                "M3_supported_mean_nb_deviance": losses["M3_supported"],
                "M2_supported_heldout_nb_log_likelihood": log_likelihoods[
                    "M2_supported"
                ],
                "M3_supported_heldout_nb_log_likelihood": log_likelihoods[
                    "M3_supported"
                ],
            }
        )
    else:
        gate3_supported_comparison["not_evaluable_reason"] = (
            "training_donor_type_proxy_alignment_failed"
            if not donor_type_proxy_support
            else "at_least_one_section_has_fewer_than_three_prespecified_supported_spots"
        )

    # M8 is deliberately created only after the sealed target is opened.  Both
    # M8 and M3 are evaluated on the same NB-compatible held-out half target.
    # The oracle split is generated from every full-depth-eligible target row.
    # Stored molecule halves are development reliability inputs only and cannot
    # route this independently generated endpoint.  The unselected-transcript
    # bin is split too, so half-depth filtering never conditions on panel genes.
    m8_counts = counts
    m8_library = library
    m8_sections = sections
    augmented = _append_other_count_bin(m8_counts, m8_library)
    augmented_theta = np.concatenate(
        (
            theta,
            np.asarray(predictions["training_other_transcript_dispersion"]).reshape(1),
        )
    )
    half_a_full, half_b_full = _nb_compatible_split(
        core, augmented, augmented_theta, seed=seed
    )
    exposure_a = half_a_full.sum(axis=1, dtype=np.float64)
    exposure_b = half_b_full.sum(axis=1, dtype=np.float64)
    generated_split_keep = (exposure_a > 0) & (exposure_b > 0)
    if not generated_split_keep.any():
        raise RuntimeError("no positive-depth generated split observations remain for M8")
    half_a = half_a_full[generated_split_keep, : counts.shape[1]]
    half_b = half_b_full[generated_split_keep, : counts.shape[1]]
    exposure_a = exposure_a[generated_split_keep]
    exposure_b = exposure_b[generated_split_keep]
    m8_sections = m8_sections[generated_split_keep]
    half_theta = theta * 0.5
    parameters = {
        name.removeprefix("m8_"): value
        for name, value in predictions.items()
        if name.startswith("m8_")
    }
    m8_rate = apply_m8_cross_half_predictor(half_a, exposure_a, parameters)
    m8_mean = np.maximum(m8_rate * exposure_b[:, None], 1.0e-6)
    m3_half_mean = np.maximum(
        np.asarray(predictions["rate_M3"])[keep][generated_split_keep]
        * exposure_b[:, None],
        1.0e-6,
    )
    m8_rows = _nb_deviance_rows(core, half_b, m8_mean, half_theta)
    m3_half_rows = _nb_deviance_rows(core, half_b, m3_half_mean, half_theta)
    losses["M8"] = _section_macro(m8_rows, m8_sections)
    losses["M3_same_M8_target"] = _section_macro(m3_half_rows, m8_sections)
    log_likelihoods["M8"] = _section_macro(
        _nb_log_likelihood_rows(core, half_b, m8_mean, half_theta), m8_sections
    )
    log_likelihoods["M3_same_M8_target"] = _section_macro(
        _nb_log_likelihood_rows(core, half_b, m3_half_mean, half_theta), m8_sections
    )
    model_types = set(np.asarray(predictions["reference_model_type_names"]).astype(str).tolist())
    observed_types = set(
        np.asarray(predictions["matched_observed_type_names"]).astype(str).tolist()
    )
    observed_reference_labels_recognized = len(model_types & observed_types) / max(
        len(observed_types), 1
    )
    model_vocabulary_type_support = len(model_types & observed_types) / max(
        len(model_types), 1
    )
    reference_state_coverage = float(
        np.mean(np.asarray(predictions["query_reference_coverage_mass"], dtype=np.float64))
    )
    abstention_rate = float(
        np.mean(np.asarray(predictions["query_reference_abstain"], dtype=bool))
    )
    bleep_retrieval_entropy = float(
        np.mean(np.asarray(predictions["BLEEP_retrieval_entropy"], dtype=np.float64))
    )
    m1_reference_entropy = float(
        np.mean(np.asarray(predictions["M1_reference_entropy_normalized"], dtype=np.float64))
    )
    m3_reference_entropy = float(
        np.mean(np.asarray(predictions["M3_reference_entropy_normalized"], dtype=np.float64))
    )
    m6_control.update(
        {
            "reference_type_names": sorted(model_types),
            "matched_reference_cell_count_by_type": np.asarray(
                predictions["matched_reference_cell_count_by_type"], dtype=np.int64
            ).tolist(),
            "matched_reference_component_ESS_by_type": np.asarray(
                predictions["matched_reference_component_ESS_by_type"], dtype=np.float64
            ).tolist(),
            "wrong_reference_cell_count_by_type": np.asarray(
                predictions["wrong_reference_cell_count_by_type"], dtype=np.int64
            ).tolist(),
            "wrong_reference_component_count_by_type": np.asarray(
                predictions["wrong_reference_component_count_by_type"], dtype=np.int64
            ).tolist(),
            "bank_size_and_state_quality_fixed": False,
            "specificity_attribution": "natural_bank_diagnostic_not_fixed_ESS_proof",
        }
    )
    for arm in quality:
        quality[arm] = {
            **quality[arm],
            "observed_reference_labels_recognized_fraction": (
                observed_reference_labels_recognized
                if arm in {"M1", "M2", "M3"}
                else "not_applicable"
            ),
            "model_vocabulary_type_support": (
                model_vocabulary_type_support
                if arm in {"M1", "M2", "M3"}
                else "not_applicable"
            ),
            "reference_state_coverage": (
                reference_state_coverage
                if arm in {"M2", "M3"}
                else model_vocabulary_type_support
                if arm == "M1"
                else "not_applicable"
            ),
            "abstention_rate": (
                abstention_rate
                if arm in {"M2", "M3"}
                else 1.0 - model_vocabulary_type_support
                if arm == "M1"
                else "not_applicable"
            ),
            "poe_reference_entropy": (
                m3_reference_entropy if arm == "M3" else m1_reference_entropy
                if arm == "M1"
                else "not_applicable"
            ),
        }
    return {
        "heldout_donor": _scalar_text(secret["heldout_donor"]),
        "indication": _scalar_text(np.unique(secret["heldout_indication_ids"])),
        "scored_spots": int(keep.sum()),
        "scored_sections": int(len(set(sections.tolist()))),
        "zero_depth_excluded": int((~keep).sum()),
        "mean_nb_deviance": losses,
        "heldout_nb_log_likelihood": log_likelihoods,
        "plug_in_NB_log_likelihood": log_likelihoods,
        "posterior_predictive_NB_log_score_M0_to_M3": posterior_log_scores,
        "posterior_predictive_log_score_method": (
            "8_point_Gauss_Hermite_moment_matched_lognormal_rate_marginal"
        ),
        "row_nb_deviance": row_losses,
        "quality_metrics_M0_to_M3": quality,
        "BLEEP_retrieval_entropy": bleep_retrieval_entropy,
        "BLEEP_hard_negative_control": {
            "strategy": _scalar_text(predictions["BLEEP_hard_negative_strategy"]),
            "primary_count": int(predictions["BLEEP_hard_negative_primary_count"]),
            "primary_fraction": float(
                predictions["BLEEP_hard_negative_primary_fraction"]
            ),
            "same_indication_fallback_count": int(
                predictions["BLEEP_hard_negative_same_indication_fallback_count"]
            ),
            "same_indication_fallback_fraction": float(
                predictions["BLEEP_hard_negative_same_indication_fallback_fraction"]
            ),
            "global_fallback_count": int(
                predictions["BLEEP_hard_negative_global_fallback_count"]
            ),
            "global_fallback_fraction": float(
                predictions["BLEEP_hard_negative_global_fallback_fraction"]
            ),
        },
        "gate3_supported_comparison": gate3_supported_comparison,
        "cross_assay_alignment": {
            "training_donor_ids": np.asarray(
                predictions["cross_assay_alignment_donor_ids"]
            )
            .astype(str)
            .tolist(),
            "pre_MSE": float(np.asarray(predictions["cross_assay_alignment_pre_MSE"])),
            "post_MSE": float(np.asarray(predictions["cross_assay_alignment_post_MSE"])),
            "weight": float(np.asarray(predictions["cross_assay_alignment_weight"])),
            "pre_matched_MSE": float(
                np.asarray(predictions["cross_assay_alignment_pre_MSE"])
            ),
            "post_matched_MSE": float(
                np.asarray(predictions["cross_assay_alignment_post_MSE"])
            ),
            "pre_mismatched_MSE": float(
                np.asarray(predictions["cross_assay_alignment_pre_mismatched_MSE"])
            ),
            "post_mismatched_MSE": float(
                np.asarray(predictions["cross_assay_alignment_post_mismatched_MSE"])
            ),
            "pre_matched_to_mismatched_ratio": float(
                np.asarray(
                    predictions[
                        "cross_assay_alignment_pre_matched_to_mismatched_ratio"
                    ]
                )
            ),
            "post_matched_to_mismatched_ratio": float(
                np.asarray(
                    predictions[
                        "cross_assay_alignment_post_matched_to_mismatched_ratio"
                    ]
                )
            ),
            "pre_separation": float(
                np.asarray(predictions["cross_assay_alignment_pre_separation"])
            ),
            "post_separation": float(
                np.asarray(predictions["cross_assay_alignment_post_separation"])
            ),
            "optimizer_applications_per_epoch": int(
                np.asarray(
                    predictions[
                        "cross_assay_alignment_optimizer_applications_per_epoch"
                    ]
                )
            ),
            "optimizer_applications_total": int(
                np.asarray(predictions["cross_assay_alignment_optimizer_applications_total"])
            ),
            "support_criterion_met": bool(
                np.asarray(predictions["cross_assay_alignment_support_criterion_met"])
            ),
            "support_criterion": _scalar_text(
                predictions["cross_assay_alignment_support_criterion"]
            ),
            "unaligned_post_matched_MSE": float(
                np.asarray(predictions["cross_assay_unaligned_post_matched_MSE"])
            ),
            "unaligned_post_mismatched_MSE": float(
                np.asarray(predictions["cross_assay_unaligned_post_mismatched_MSE"])
            ),
            "unaligned_post_matched_to_mismatched_ratio": float(
                np.asarray(
                    predictions[
                        "cross_assay_unaligned_post_matched_to_mismatched_ratio"
                    ]
                )
            ),
            "beats_unaligned_comparator": bool(
                np.asarray(predictions["cross_assay_alignment_beats_unaligned_comparator"])
            ),
            "metric_role": (
                "coarse_training_only_donor_pseudobulk_fit_adequacy_guard_not_"
                "type_state_calibration_evidence"
            ),
            "donor_type_proxy_diagnostic": {
                "evaluable": bool(
                    predictions["cross_assay_donor_type_proxy_evaluable"]
                ),
                "training_type_names": np.asarray(
                    predictions["cross_assay_donor_type_proxy_type_names"]
                )
                .astype(str)
                .tolist(),
                "donor_type_pairs": int(
                    predictions["cross_assay_donor_type_proxy_pairs"]
                ),
                "matched_MSE": _finite_or_none(
                    float(predictions["cross_assay_donor_type_proxy_matched_MSE"])
                ),
                "mismatched_MSE": _finite_or_none(
                    float(predictions["cross_assay_donor_type_proxy_mismatched_MSE"])
                ),
                "matched_to_mismatched_ratio": _finite_or_none(
                    float(predictions["cross_assay_donor_type_proxy_ratio"])
                ),
                "unaligned_matched_to_mismatched_ratio": _finite_or_none(
                    float(predictions["cross_assay_donor_type_proxy_unaligned_ratio"])
                ),
                "support_criterion_met": bool(
                    predictions[
                        "cross_assay_donor_type_proxy_support_criterion_met"
                    ]
                ),
                "scope": _scalar_text(
                    predictions["cross_assay_donor_type_proxy_scope"]
                ),
                "role": "Gate_3_evaluability_guard_not_cell_level_truth",
            },
            "heldout_ST_used": False,
        },
        "H_state_variance_calibration": {
            "method": _scalar_text(predictions["H_state_variance_calibration_method"]),
            "training_rows": int(
                np.asarray(predictions["H_state_variance_calibration_rows"])
            ),
            "NLL_before": float(
                np.asarray(predictions["H_state_variance_calibration_NLL_before"])
            ),
            "NLL_after": float(
                np.asarray(predictions["H_state_variance_calibration_NLL_after"])
            ),
        },
        "matched_reference_support": {
            "observed_reference_labels_recognized_fraction": (
                observed_reference_labels_recognized
            ),
            "model_vocabulary_type_support": model_vocabulary_type_support,
            "H_composition_weighted_state_coverage": reference_state_coverage,
            "abstention_rate": abstention_rate,
            "abstention_semantics": (
                "out_of_support_diagnostic_flag_only_predictions_are_not_withheld"
            ),
            "normalized_mahalanobis_threshold": float(
                np.asarray(predictions["matched_reference_state_support_threshold"])
            ),
            "coverage_mass_threshold": float(
                np.asarray(
                    predictions["matched_reference_abstention_coverage_mass_threshold"]
                )
            ),
            "threshold_status": _scalar_text(
                predictions["matched_reference_state_support_threshold_status"]
            ),
            "effective_sample_size": float(
                np.asarray(predictions["matched_reference_effective_sample_size"])
            ),
            "cell_count": int(np.asarray(predictions["matched_reference_cell_count"])),
        },
        "m6_wrong_reference_control": m6_control,
        "m8_same_target": {
            "target": "NB_compatible_half_B",
            "exposure": "generated_other_inclusive_half_A_and_half_B_library_depths",
            "M8": _section_macro(m8_rows, m8_sections),
            "M3": _section_macro(m3_half_rows, m8_sections),
            "scored_spots": int(generated_split_keep.sum()),
            "zero_split_depth_excluded": int((~generated_split_keep).sum()),
            "routing_mask": "generated_other_inclusive_positive_half_depths_not_panel_counts",
            "scoring_dispersion_fraction_of_full_theta": 0.5,
            "predictor": "training_donor_fitted_cross_half_molecular_ridge",
            "scientific_role": "empirical_molecular_oracle_not_a_hard_ST_floor",
            "full_depth_measurement_noise_risk": (
                "not_estimated_no_registered_exact_replicate_or_full_depth_noise_model"
            ),
        },
    }


def _evaluate_gates(
    core: Any,
    per_donor: Mapping[str, Mapping[str, float]],
    donor_indications: Mapping[str, str],
    *,
    seed: int,
) -> Mapping[str, object]:
    comparisons: dict[str, object] = {}
    pairs = {
        "M3_vs_M0": ("M0", "M3"),
        "M3_vs_M1": ("M1", "M3"),
        "M3_vs_M4": ("M4", "M3"),
        "M3_vs_M2_full_mixed_support_descriptive": ("M2", "M3"),
        "M3_vs_M6": ("M6", "M3"),
        "M3_vs_M7": ("M7", "M3"),
        "M3_vs_BLEEP": ("BLEEP", "M3"),
        "M8_vs_M3": ("M3_same_M8_target", "M8"),
    }
    for name, (left, right) in pairs.items():
        values = _donor_effects(per_donor, left, right)
        sign = _sign_flip(core, values)
        comparison_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little"
        )
        comparisons[name] = {
            "donor_improvement": values.tolist(),
            "mean_improvement": float(np.mean(values)),
            "median_improvement": float(np.median(values)),
            "donor_fraction_improved": float(np.mean(values > 0)),
            "sign_flip": sign,
            "paired_donor_bootstrap_confidence_interval": list(
                _donor_bootstrap_interval(values, seed=comparison_seed)
            ),
            "bootstrap_replicates": 20_000,
        }
    gate3_missing_donors = [
        donor
        for donor, values in sorted(per_donor.items())
        if not _is_finite_number(values.get("M2_supported"))
        or not _is_finite_number(values.get("M3_supported"))
    ]
    gate3_comparison_name = "M3_supported_vs_M2_supported"
    gate3_comparison_evaluable = not gate3_missing_donors
    if gate3_comparison_evaluable:
        gate3_values = _donor_effects(
            per_donor, "M2_supported", "M3_supported"
        )
        gate3_sign = _sign_flip(core, gate3_values)
        gate3_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{gate3_comparison_name}".encode()).digest()[:8],
            "little",
        )
        comparisons[gate3_comparison_name] = {
            "evaluable": True,
            "donor_improvement": gate3_values.tolist(),
            "mean_improvement": float(np.mean(gate3_values)),
            "median_improvement": float(np.median(gate3_values)),
            "donor_fraction_improved": float(np.mean(gate3_values > 0)),
            "sign_flip": gate3_sign,
            "paired_donor_bootstrap_confidence_interval": list(
                _donor_bootstrap_interval(gate3_values, seed=gate3_seed)
            ),
            "bootstrap_replicates": 20_000,
            "scope": "matched_reference_supported_composition_only",
        }
    else:
        comparisons[gate3_comparison_name] = {
            "evaluable": False,
            "missing_donors": gate3_missing_donors,
            "reason": (
                "at_least_one_donor_section_failed_the_prespecified_"
                "supported_composition_spot_minimum"
            ),
        }
    pvalues = {
        name: float(comparisons[name]["sign_flip"]["p_value"])
        for name in ("M3_vs_M1", "M3_vs_M4", "M3_vs_M6", "M3_vs_M7")
    }
    names_a = ("M3_vs_M1", "M3_vs_M4")
    names_b = ("M3_vs_M6", "M3_vs_M7")
    adjusted_a = np.asarray(core.holm_adjust([pvalues[name] for name in names_a]))
    adjusted_b = np.asarray(core.holm_adjust([pvalues[name] for name in names_b]))
    holm_a = {name: float(value) for name, value in zip(names_a, adjusted_a)}
    holm_b = {name: float(value) for name, value in zip(names_b, adjusted_b)}
    gate_payload = {
        "comparisons": comparisons,
        "holm_gate2": holm_a,
        "holm_gate4": holm_b,
        "minimum_relative_gain": 0.05,
        "minimum_donor_fraction": 0.70,
        "familywise_alpha": 0.05,
        "ordered": True,
    }
    donor_order = [donor for donor, _ in sorted(per_donor.items())]
    core_losses = {
        arm: [per_donor[donor][arm] for donor in donor_order]
        for arm in ("M0", "M1", "M2", "M3", "M4", "M6", "M7", "M8")
    }
    try:
        core_decision = _call_with_supported_kwargs(
            core.evaluate_ordered_gates,
            core_losses,
            indication_ids=[donor_indications[donor] for donor in donor_order],
            minimum_relative_gain=0.05,
            minimum_positive_fraction=0.70,
            maximum_indication_reversal=0.02,
        )
        core_crosscheck = (
            dataclasses.asdict(core_decision)
            if dataclasses.is_dataclass(core_decision)
            else core_decision
        )
        if isinstance(core_crosscheck, Mapping):
            sanitized = dict(core_crosscheck)
            raw_gates = sanitized.get("gates")
            if isinstance(raw_gates, (list, tuple)):
                sanitized["gates"] = [
                    gate
                    for gate in raw_gates
                    if not (
                        isinstance(gate, Mapping)
                        and gate.get("name")
                        in {
                            "gate_3_state_beyond_routing",
                            "gate_4_matching_specificity",
                            "gate_5_molecular_headroom",
                        }
                    )
                ]
            sanitized.pop("molecular_headroom_detected", None)
            sanitized.pop("personalization_supported", None)
            sanitized.pop("full_model_chain_supported", None)
            sanitized["scope"] = (
                "descriptive_generic_core_crosscheck_gates_1_to_2_only;_"
                "conditional_gate3_natural_bank_gate4_and_same_target_gate5_"
                "are_computed_only_by_this_runner"
            )
            core_crosscheck = sanitized
    except (TypeError, ValueError):
        core_crosscheck = {"status": "core_gate_contract_not_evaluable"}

    central = comparisons["M3_vs_M0"]
    central_effect = np.asarray(central["donor_improvement"], dtype=np.float64)
    baseline = np.asarray([per_donor[donor]["M0"] for donor in donor_order])
    relative_gain = float(np.mean(central_effect) / np.mean(baseline))
    confidence = central["paired_donor_bootstrap_confidence_interval"]
    indication_ok = True
    indication_relative_gain: dict[str, float] = {}
    indication_vector = np.asarray([donor_indications[donor] for donor in donor_order])
    for indication in sorted(set(indication_vector.tolist())):
        keep = indication_vector == indication
        value = float(np.mean(central_effect[keep]) / np.mean(baseline[keep]))
        indication_relative_gain[indication] = value
        indication_ok &= value >= -0.02
    gate1 = bool(
        relative_gain >= 0.05
        and float(np.mean(central_effect > 0)) >= 0.70
        and float(confidence[0]) > 0
        and float(central["sign_flip"]["p_value"]) <= 0.05
        and indication_ok
    )
    gate2 = bool(
        gate1
        and all(comparisons[name]["mean_improvement"] > 0 for name in names_a)
        and all(holm_a[name] <= 0.05 for name in names_a)
    )
    gate3_comparison = comparisons[gate3_comparison_name]
    gate3 = bool(
        gate2
        and gate3_comparison_evaluable
        and gate3_comparison["mean_improvement"] > 0
        and gate3_comparison["sign_flip"]["p_value"] <= 0.05
    )
    gate4 = bool(
        gate3
        and all(comparisons[name]["mean_improvement"] > 0 for name in names_b)
        and all(holm_b[name] <= 0.05 for name in names_b)
    )
    gate5 = bool(
        comparisons["M8_vs_M3"]["mean_improvement"] > 0
        and comparisons["M8_vs_M3"]["sign_flip"]["p_value"] <= 0.05
    )
    reached = [True, gate1, gate2, gate3, True]
    passed = [gate1, gate2, gate3, gate4, gate5]
    first_failure = next(
        (f"gate_{index}" for index, value in enumerate(passed[:4], start=1) if not value),
        None,
    )
    decision = {
        "gate_1": {
            "reached": reached[0],
            "passed": gate1,
            "relative_gain": relative_gain,
            "confidence_interval": confidence,
            "exact_sign_flip_p_value": float(central["sign_flip"]["p_value"]),
            "donor_fraction_improved": float(np.mean(central_effect > 0)),
            "indication_relative_gain": indication_relative_gain,
            "no_severe_indication_reversal": indication_ok,
        },
        "gate_2": {"reached": reached[1], "passed": gate2, "holm": holm_a},
        "gate_3": {
            "reached": reached[2],
            "evaluable": gate3_comparison_evaluable,
            "passed": gate3,
            "comparison": gate3_comparison_name,
            "missing_donors": gate3_missing_donors,
            "full_target_M3_vs_M2_role": "descriptive_mixed_support_not_gating",
        },
        "gate_4": {"reached": reached[3], "passed": gate4, "holm": holm_b},
        "gate_5": {
            "reached": True,
            "passed": gate5,
            "blocking": False,
            "same_split_target": True,
        },
        "central_development_gate_passed": gate1,
        "incremental_reference_value_development_signal": gate1,
        "histology_reference_synergy_supported": bool(gate1 and gate2),
        "continuous_state_beyond_routing_supported": bool(gate1 and gate2 and gate3),
        "full_model_chain_supported": bool(gate1 and gate2 and gate3),
        "natural_wrong_or_generic_reference_separation_supported": bool(
            gate1 and gate2 and gate3 and gate4
        ),
        "personalized_reference_supported": False,
        "personalization_status": (
            "not_attributable_without_the_unrun_fixed_ESS_type_support_and_"
            "reference_quality_sensitivity"
        ),
        "BLEEP_descriptive_comparison": {
            "mean_improvement": comparisons["M3_vs_BLEEP"]["mean_improvement"],
            "donor_fraction_improved": comparisons["M3_vs_BLEEP"][
                "donor_fraction_improved"
            ],
            "unadjusted_p_value": comparisons["M3_vs_BLEEP"]["sign_flip"][
                "p_value"
            ],
            "passes_prespecified_70_percent_consistency": bool(
                comparisons["M3_vs_BLEEP"]["donor_fraction_improved"] >= 0.70
            ),
            "role": "descriptive_non_Holm_non_gating_complexity_control",
        },
        "stopped_at": first_failure,
    }
    return {
        **gate_payload,
        "decision": decision,
        "core_gate_1_to_2_sanitized_crosscheck": core_crosscheck,
        "core_personalization_crosscheck_disregarded_reason": (
            "the_generic_core_contract_assumes_fixed_M6_M7_reference_banks;_"
            "this_development_run_uses_natural_banks"
        ),
    }


_QUALITY_FIELDS = (
    "latent_MSE_20D",
    "gene_correlation",
    "program_correlation",
    "calibration_slope",
    "reliability_adjusted_variance_median",
    "program_covariance_relative_error",
    "rare_program_state_recall",
)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and bool(
        np.isfinite(float(value))
    )


def _quality_completeness(
    fold_reports: Mapping[str, Mapping[str, object]],
) -> tuple[bool, list[str]]:
    """Fail closed when any prespecified diagnostic is unavailable."""

    missing: list[str] = []
    for donor, fold in sorted(fold_reports.items()):
        quality = fold.get("quality_metrics_M0_to_M3", {})
        if not isinstance(quality, Mapping):
            missing.append(f"{donor}:quality_metrics_M0_to_M3")
            continue
        for arm in ("M0", "M1", "M2", "M3"):
            metrics = quality.get(arm, {})
            if not isinstance(metrics, Mapping):
                missing.append(f"{donor}:{arm}")
                continue
            for field in _QUALITY_FIELDS:
                if not _is_finite_number(metrics.get(field)):
                    missing.append(f"{donor}:{arm}:{field}")
            intervals = metrics.get("predictive_interval_coverage", {})
            if not isinstance(intervals, Mapping):
                missing.append(f"{donor}:{arm}:predictive_interval_coverage")
            else:
                for level in ("50", "80", "95"):
                    if not _is_finite_number(intervals.get(level)):
                        missing.append(f"{donor}:{arm}:interval_{level}")
            if arm in {"M1", "M2", "M3"}:
                for field in (
                    "observed_reference_labels_recognized_fraction",
                    "model_vocabulary_type_support",
                ):
                    if not _is_finite_number(metrics.get(field)):
                        missing.append(f"{donor}:{arm}:{field}")
            if arm in {"M1", "M2", "M3"}:
                for field in ("reference_state_coverage", "abstention_rate"):
                    if not _is_finite_number(metrics.get(field)):
                        missing.append(f"{donor}:{arm}:{field}")
        m3 = quality.get("M3", {})
        if not isinstance(m3, Mapping) or not _is_finite_number(
            m3.get("poe_reference_entropy")
        ):
            missing.append(f"{donor}:M3:poe_reference_entropy")
        if not _is_finite_number(fold.get("BLEEP_retrieval_entropy")):
            missing.append(f"{donor}:BLEEP:retrieval_entropy")
    return not missing, missing


def _donor_balanced_quality(
    fold_reports: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    """Aggregate only evaluable donor metrics and retain their denominators."""

    result: dict[str, object] = {}
    for arm in ("M0", "M1", "M2", "M3"):
        arm_result: dict[str, object] = {}
        fields = (
            *_QUALITY_FIELDS,
            "observed_reference_labels_recognized_fraction",
            "model_vocabulary_type_support",
            "reference_state_coverage",
            "abstention_rate",
            "poe_reference_entropy",
        )
        for field in fields:
            values = [
                fold["quality_metrics_M0_to_M3"][arm].get(field)
                for fold in fold_reports.values()
            ]
            finite = [float(value) for value in values if _is_finite_number(value)]
            arm_result[field] = {
                "mean": float(np.mean(finite)) if finite else None,
                "evaluable_donors": len(finite),
            }
        arm_result["predictive_interval_coverage"] = {
            level: {
                "mean": (
                    float(np.mean(values))
                    if (
                        values := [
                            float(
                                fold["quality_metrics_M0_to_M3"][arm][
                                    "predictive_interval_coverage"
                                ][level]
                            )
                            for fold in fold_reports.values()
                            if _is_finite_number(
                                fold["quality_metrics_M0_to_M3"][arm][
                                    "predictive_interval_coverage"
                                ].get(level)
                            )
                        ]
                    )
                    else None
                ),
                "evaluable_donors": len(values),
            }
            for level in ("50", "80", "95")
        }
        result[arm] = arm_result
    return result


def _quality_guard(
    aggregate: Mapping[str, object], margins: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply the frozen development quality-preservation margins to M3."""

    def mean(arm: str, field: str) -> float:
        value = aggregate[arm][field]["mean"]  # type: ignore[index]
        if not _is_finite_number(value):
            raise ValueError(f"quality guard field is unavailable: {arm}/{field}")
        return float(value)

    m0_variance = mean("M0", "reliability_adjusted_variance_median")
    m3_variance = mean("M3", "reliability_adjusted_variance_median")
    m0_covariance_error = mean("M0", "program_covariance_relative_error")
    m3_covariance_error = mean("M3", "program_covariance_relative_error")
    m0_rare_recall = mean("M0", "rare_program_state_recall")
    m3_rare_recall = mean("M3", "rare_program_state_recall")
    interval_margins = margins["maximum_absolute_interval_coverage_error"]
    if not isinstance(interval_margins, Mapping):
        raise ValueError("interval coverage margins are malformed")

    checks: dict[str, object] = {}
    variance_ratio = m3_variance / max(m0_variance, 1.0e-12)
    variance_threshold = float(
        margins["minimum_M3_to_M0_reliability_adjusted_variance_ratio"]
    )
    checks["reliability_adjusted_variance"] = {
        "observed_M3_to_M0_ratio": variance_ratio,
        "threshold": variance_threshold,
        "passed": variance_ratio >= variance_threshold,
    }
    covariance_ratio = m3_covariance_error / max(m0_covariance_error, 1.0e-12)
    covariance_threshold = float(
        margins["maximum_M3_to_M0_program_covariance_error_ratio"]
    )
    checks["program_covariance"] = {
        "observed_M3_to_M0_error_ratio": covariance_ratio,
        "threshold": covariance_threshold,
        "passed": covariance_ratio <= covariance_threshold,
    }
    rare_drop = m0_rare_recall - m3_rare_recall
    rare_threshold = float(margins["maximum_rare_state_recall_drop"])
    checks["rare_program_state_recall"] = {
        "observed_M0_minus_M3": rare_drop,
        "threshold": rare_threshold,
        "passed": rare_drop <= rare_threshold,
    }
    for level, target in (("50", 0.50), ("80", 0.80), ("95", 0.95)):
        coverage = aggregate["M3"]["predictive_interval_coverage"][level]["mean"]  # type: ignore[index]
        if not _is_finite_number(coverage):
            raise ValueError(f"M3 interval {level} is unavailable")
        error = abs(float(coverage) - target)
        threshold = float(interval_margins[level])
        checks[f"predictive_interval_{level}"] = {
            "observed_coverage": float(coverage),
            "absolute_error": error,
            "threshold": threshold,
            "passed": error <= threshold,
            "blocking": False,
            "reason": "moment_normal_interval_is_exploratory_not_a_predictive_quantile",
        }
    reference_coverage = mean("M3", "reference_state_coverage")
    reference_threshold = float(
        margins["minimum_H_composition_weighted_state_coverage"]
    )
    checks["reference_state_coverage"] = {
        "observed": reference_coverage,
        "threshold": reference_threshold,
        "passed": reference_coverage >= reference_threshold,
        "blocking": False,
        "reason": (
            "fixed_development_support_distance_is_exploratory_and_was_not_"
            "calibrated_by_an_inner_training_donor_search"
        ),
    }
    abstention = mean("M3", "abstention_rate")
    abstention_threshold = float(margins["maximum_abstention_rate"])
    checks["abstention_rate"] = {
        "observed": abstention,
        "threshold": abstention_threshold,
        "passed": abstention <= abstention_threshold,
        "blocking": False,
        "reason": (
            "fixed_development_coverage_mass_is_exploratory_and_was_not_"
            "calibrated_by_an_inner_training_donor_search"
        ),
    }
    passed = all(
        bool(value["passed"])
        for value in checks.values()
        if bool(value.get("blocking", True))  # type: ignore[union-attr]
    )
    return {
        "status": "evaluated_against_frozen_development_margins",
        "passed": passed,
        "checks": checks,
    }


def _validate_prediction_manifest_binding(
    args: argparse.Namespace,
    core: Any,
    prepared: Mapping[str, object],
    prediction_manifest: Mapping[str, object],
) -> None:
    """Reject incomplete or stale predictions before any score target is opened."""

    if (
        prediction_manifest.get("schema") != PREDICTION_SCHEMA
        or prediction_manifest.get("all_folds_complete") is not True
    ):
        raise ValueError("fit-predict manifest is incomplete or malformed")
    donors = tuple(str(value) for value in prepared.get("donors", ()))
    prepared_folds = prepared.get("folds")
    prediction_folds = prediction_manifest.get("folds")
    if (
        not donors
        or not isinstance(prepared_folds, Mapping)
        or not isinstance(prediction_folds, Mapping)
        or set(prediction_folds) != set(donors)
        or set(prepared_folds) != set(donors)
    ):
        raise ValueError("fit-predict donor folds do not exactly match preparation")

    runner_sha256 = _sha256(Path(__file__).resolve())
    core_sha256 = _sha256(Path(core.__file__).resolve())
    protocol_sha256 = str(prepared.get("protocol_sha256", ""))
    for donor in donors:
        fold = prepared_folds[donor]
        receipt = prediction_folds[donor]
        if not isinstance(fold, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError(f"prediction receipt is malformed for {donor}")
        identity = _checkpoint_identity(
            str(fold.get("public_semantic_sha256", "")),
            donor=donor,
            seed=int(fold.get("seed", -1)),
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            device=args.device,
            runner_sha256=runner_sha256,
            core_sha256=core_sha256,
            protocol_sha256=protocol_sha256,
        )
        required = {
            "schema": PREDICTION_SCHEMA,
            "heldout_donor": donor,
            "checkpoint_identity": identity,
            "public_semantic_sha256": fold.get("public_semantic_sha256"),
            "protocol_sha256": protocol_sha256,
            "runner_sha256": runner_sha256,
            "core_sha256": core_sha256,
            "artifact_complete": True,
            "heldout_ST_opened": False,
        }
        mismatched = [
            name for name, expected in required.items() if receipt.get(name) != expected
        ]
        expected_path = (args.output / "folds" / donor / "predictions.npz").resolve()
        observed_path = Path(str(receipt.get("prediction_path", ""))).resolve()
        if observed_path != expected_path:
            mismatched.append("prediction_path")
        if mismatched:
            raise ValueError(
                f"stale prediction receipt for {donor}: {sorted(set(mismatched))}"
            )
        predictions = _verify_semantic_file(
            observed_path, str(receipt.get("prediction_semantic_sha256", ""))
        )
        public = _verify_semantic_file(
            Path(str(fold.get("public_path", ""))),
            str(fold.get("public_semantic_sha256", "")),
        )
        _validate_prediction_artifact(
            predictions,
            public,
            donor=donor,
            epochs=args.epochs,
        )


def _validate_prediction_artifact(
    predictions: Mapping[str, np.ndarray],
    public: Mapping[str, np.ndarray],
    *,
    donor: str,
    epochs: int,
) -> None:
    """Validate every score-time prediction field before opening an outcome.

    The semantic hash proves that an artifact has not changed; it does not prove
    that a re-hashed artifact is complete.  Keep this validator target-free and
    call it both during global preflight and immediately before each score target
    is opened.
    """

    direct_fields = {
        "schema",
        "heldout_donor",
        "query_spot_ids",
        "gene_ids",
        "training_only_dispersion",
        "training_other_transcript_dispersion",
        "prediction_scale",
        "wrong_donor_ids",
        "query_H_composition",
        "m4_shuffle_index",
        "m4_composition_stratum",
        "BLEEP_retrieval_entropy",
        "BLEEP_temperature",
        "BLEEP_hard_negative_weight",
        "BLEEP_hard_negative_strategy",
        "BLEEP_hard_negative_index",
        "BLEEP_hard_negative_primary_count",
        "BLEEP_hard_negative_primary_fraction",
        "BLEEP_hard_negative_same_indication_fallback_count",
        "BLEEP_hard_negative_same_indication_fallback_fraction",
        "BLEEP_hard_negative_global_fallback_count",
        "BLEEP_hard_negative_global_fallback_fraction",
        "reference_model_type_names",
        "matched_observed_type_names",
        "matched_reference_effective_sample_size",
        "matched_reference_cell_count",
        "diagnostic_normalization_scale",
        "diagnostic_basis_mean",
        "diagnostic_basis_components",
        "diagnostic_basis_score_scale",
        "diagnostic_program_names",
        "diagnostic_program_membership",
        "diagnostic_program_active",
        "diagnostic_rare_program_thresholds",
        "diagnostic_training_reliable_gene",
        "m8_coefficient",
        "m8_intercept",
        "m8_input_mean",
        "m8_input_scale",
        "m8_normalization_scale",
        "H_state_variance_calibration_method",
        "H_state_variance_calibration_rows",
        "H_state_variance_calibration_NLL_before",
        "H_state_variance_calibration_NLL_after",
        "H_state_calibrated_variance_by_type",
        "matched_reference_cell_count_by_type",
        "matched_reference_component_count_by_type",
        "matched_reference_component_ESS_by_type",
        "matched_reference_state_support_threshold",
        "matched_reference_state_support_threshold_status",
        "matched_reference_abstention_coverage_mass_threshold",
        "query_reference_min_normalized_mahalanobis",
        "query_reference_coverage_mass",
        "query_reference_abstain",
        "M1_reference_entropy_normalized",
        "M3_reference_entropy_normalized",
        "wrong_reference_cell_count_by_type",
        "wrong_reference_component_count_by_type",
        "rate_M2_supported",
        "rate_M3_supported",
        "gate3_supported_type_mask",
        "gate3_supported_composition_mass",
        "gate3_minimum_supported_composition_mass",
        "gate3_minimum_eligible_spots_per_section",
        "gate3_supported_score_eligible",
        "gate3_comparison_scope",
        "cross_assay_alignment_donor_ids",
        "cross_assay_alignment_pre_MSE",
        "cross_assay_alignment_post_MSE",
        "cross_assay_alignment_weight",
        "cross_assay_alignment_pre_mismatched_MSE",
        "cross_assay_alignment_post_mismatched_MSE",
        "cross_assay_alignment_pre_matched_to_mismatched_ratio",
        "cross_assay_alignment_post_matched_to_mismatched_ratio",
        "cross_assay_alignment_pre_separation",
        "cross_assay_alignment_post_separation",
        "cross_assay_alignment_optimizer_applications_per_epoch",
        "cross_assay_alignment_optimizer_applications_total",
        "cross_assay_alignment_support_criterion_met",
        "cross_assay_alignment_support_criterion",
        "cross_assay_unaligned_post_matched_MSE",
        "cross_assay_unaligned_post_mismatched_MSE",
        "cross_assay_unaligned_post_matched_to_mismatched_ratio",
        "cross_assay_alignment_beats_unaligned_comparator",
        "cross_assay_donor_type_proxy_evaluable",
        "cross_assay_donor_type_proxy_type_names",
        "cross_assay_donor_type_proxy_pairs",
        "cross_assay_donor_type_proxy_matched_MSE",
        "cross_assay_donor_type_proxy_mismatched_MSE",
        "cross_assay_donor_type_proxy_ratio",
        "cross_assay_donor_type_proxy_unaligned_ratio",
        "cross_assay_donor_type_proxy_support_criterion_met",
        "cross_assay_donor_type_proxy_scope",
    }
    rate_fields = {
        "rate_M6_candidates" if arm == "M6" else f"rate_{arm}" for arm in MODEL_ARMS
    }
    variance_fields = {f"posterior_rate_variance_{arm}" for arm in ("M0", "M1", "M2", "M3")}
    missing = sorted((direct_fields | rate_fields | variance_fields) - set(predictions))
    if missing:
        raise ValueError(f"prediction artifact is incomplete for {donor}: {missing}")

    if (
        _scalar_text(predictions["schema"]) != PREDICTION_SCHEMA
        or _scalar_text(predictions["heldout_donor"]) != donor
        or _scalar_text(predictions["prediction_scale"])
        != "per_unit_actual_ST_library_rate"
    ):
        raise ValueError(f"prediction artifact identity is malformed for {donor}")
    for field in ("query_spot_ids", "gene_ids", "wrong_donor_ids"):
        public_field = field
        if not np.array_equal(np.asarray(predictions[field]), np.asarray(public[public_field])):
            raise ValueError(
                f"prediction artifact {field} differs from sealed public fold for {donor}"
            )

    rows = len(public["query_spot_ids"])
    genes = len(public["gene_ids"])
    wrong_donors = len(public["wrong_donor_ids"])
    expected_rate = (rows, genes)
    for arm in MODEL_ARMS:
        field = "rate_M6_candidates" if arm == "M6" else f"rate_{arm}"
        value = np.asarray(predictions[field])
        expected = (wrong_donors, rows, genes) if arm == "M6" else expected_rate
        if value.shape != expected or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"prediction artifact {field} is malformed for {donor}")
    for field in variance_fields:
        value = np.asarray(predictions[field])
        if value.shape != expected_rate or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"prediction artifact {field} is malformed for {donor}")

    dispersion = np.asarray(predictions["training_only_dispersion"])
    other_dispersion = np.asarray(predictions["training_other_transcript_dispersion"])
    if (
        dispersion.shape != (genes,)
        or not np.isfinite(dispersion).all()
        or np.any(dispersion <= 0)
        or other_dispersion.shape != ()
        or not np.isfinite(other_dispersion)
        or float(other_dispersion) <= 0
    ):
        raise ValueError(f"prediction artifact dispersion is malformed for {donor}")

    type_names = np.asarray(predictions["reference_model_type_names"]).astype(str)
    types = len(type_names)
    if types < 2 or len(set(type_names.tolist())) != types:
        raise ValueError(f"prediction artifact reference types are malformed for {donor}")
    required_shapes = {
        "query_H_composition": (rows, types),
        "m4_shuffle_index": (rows,),
        "m4_composition_stratum": (rows,),
        "BLEEP_retrieval_entropy": (rows,),
        "matched_reference_cell_count_by_type": (types,),
        "matched_reference_component_ESS_by_type": (types,),
        "query_reference_min_normalized_mahalanobis": (rows, types),
        "query_reference_coverage_mass": (rows,),
        "query_reference_abstain": (rows,),
        "M1_reference_entropy_normalized": (rows,),
        "M3_reference_entropy_normalized": (rows,),
        "wrong_reference_cell_count_by_type": (wrong_donors, types),
        "wrong_reference_component_count_by_type": (wrong_donors, types),
        "rate_M2_supported": (rows, genes),
        "rate_M3_supported": (rows, genes),
        "gate3_supported_type_mask": (types,),
        "gate3_supported_composition_mass": (rows,),
        "gate3_supported_score_eligible": (rows,),
        "BLEEP_hard_negative_index": (len(public["train_spot_ids"]),),
        "m8_coefficient": (genes, genes),
        "m8_intercept": (genes,),
        "m8_input_mean": (genes,),
        "m8_input_scale": (genes,),
        "diagnostic_basis_mean": (genes,),
        "diagnostic_training_reliable_gene": (genes,),
    }
    for field, expected in required_shapes.items():
        value = np.asarray(predictions[field])
        if value.shape != expected:
            raise ValueError(f"prediction artifact {field} is malformed for {donor}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"prediction artifact {field} is non-finite for {donor}")
    for field in ("rate_M2_supported", "rate_M3_supported"):
        if np.any(np.asarray(predictions[field]) <= 0):
            raise ValueError(f"prediction artifact {field} is non-positive for {donor}")
    gate3_mass = np.asarray(predictions["gate3_supported_composition_mass"])
    gate3_eligible = np.asarray(predictions["gate3_supported_score_eligible"], dtype=bool)
    gate3_type = np.asarray(predictions["gate3_supported_type_mask"], dtype=bool)
    if (
        not np.isclose(
            float(predictions["gate3_minimum_supported_composition_mass"]), 0.90
        )
        or int(predictions["gate3_minimum_eligible_spots_per_section"]) != 3
        or not np.array_equal(gate3_eligible, gate3_mass >= 0.90)
        or not np.array_equal(
            gate3_type,
            np.asarray(predictions["matched_reference_component_count_by_type"]) >= 2,
        )
    ):
        raise ValueError(f"prediction artifact Gate 3 support receipt is malformed for {donor}")

    components = np.asarray(predictions["diagnostic_basis_components"])
    latent = components.shape[0] if components.ndim == 2 else 0
    membership = np.asarray(predictions["diagnostic_program_membership"])
    programs = membership.shape[0] if membership.ndim == 2 else 0
    diagnostic_shapes = {
        "diagnostic_basis_components": (latent, genes),
        "diagnostic_basis_score_scale": (latent,),
        "diagnostic_program_names": (programs,),
        "diagnostic_program_membership": (programs, genes),
        "diagnostic_program_active": (programs,),
        "diagnostic_rare_program_thresholds": (programs,),
        "H_state_calibrated_variance_by_type": (types, latent),
    }
    if latent < 1 or programs < 1:
        raise ValueError(f"prediction artifact diagnostics are empty for {donor}")
    for field, expected in diagnostic_shapes.items():
        value = np.asarray(predictions[field])
        if value.shape != expected or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
            raise ValueError(f"prediction artifact {field} is malformed for {donor}")
    if not np.asarray(predictions["diagnostic_program_active"], dtype=bool).any():
        raise ValueError(f"prediction artifact has no active diagnostic program for {donor}")

    numeric_scalars = (
        "m8_normalization_scale",
        "matched_reference_effective_sample_size",
        "matched_reference_cell_count",
        "matched_reference_state_support_threshold",
        "matched_reference_abstention_coverage_mass_threshold",
        "H_state_variance_calibration_rows",
        "H_state_variance_calibration_NLL_before",
        "H_state_variance_calibration_NLL_after",
        "cross_assay_alignment_pre_MSE",
        "cross_assay_alignment_post_MSE",
        "cross_assay_alignment_weight",
        "cross_assay_alignment_pre_mismatched_MSE",
        "cross_assay_alignment_post_mismatched_MSE",
        "cross_assay_alignment_pre_matched_to_mismatched_ratio",
        "cross_assay_alignment_post_matched_to_mismatched_ratio",
        "cross_assay_alignment_pre_separation",
        "cross_assay_alignment_post_separation",
        "cross_assay_alignment_optimizer_applications_per_epoch",
        "cross_assay_alignment_optimizer_applications_total",
        "cross_assay_unaligned_post_matched_MSE",
        "cross_assay_unaligned_post_mismatched_MSE",
        "cross_assay_unaligned_post_matched_to_mismatched_ratio",
        "BLEEP_hard_negative_primary_count",
        "BLEEP_hard_negative_primary_fraction",
        "BLEEP_hard_negative_same_indication_fallback_count",
        "BLEEP_hard_negative_same_indication_fallback_fraction",
        "BLEEP_hard_negative_global_fallback_count",
        "BLEEP_hard_negative_global_fallback_fraction",
        "BLEEP_temperature",
        "BLEEP_hard_negative_weight",
        "gate3_minimum_supported_composition_mass",
        "gate3_minimum_eligible_spots_per_section",
    )
    for field in numeric_scalars:
        value = np.asarray(predictions[field])
        if value.shape != () or not np.isfinite(value):
            raise ValueError(f"prediction artifact {field} is malformed for {donor}")

    negative_kinds = ("primary", "same_indication_fallback", "global_fallback")
    training_rows = len(public["train_spot_ids"])
    negative_counts = [
        int(predictions[f"BLEEP_hard_negative_{kind}_count"])
        for kind in negative_kinds
    ]
    negative_fractions = [
        float(predictions[f"BLEEP_hard_negative_{kind}_fraction"])
        for kind in negative_kinds
    ]
    hard_negative_index = np.asarray(predictions["BLEEP_hard_negative_index"])
    if (
        any(count < 0 or count > training_rows for count in negative_counts)
        or any(fraction < 0 or fraction > 1 for fraction in negative_fractions)
        or any(
            not np.isclose(fraction, count / training_rows, atol=1.0e-6)
            for count, fraction in zip(negative_counts, negative_fractions)
        )
        or sum(negative_counts) != training_rows
        or not np.isclose(sum(negative_fractions), 1.0, atol=1.0e-5)
        or np.any(hard_negative_index < 0)
        or np.any(hard_negative_index >= training_rows)
    ):
        raise ValueError(f"prediction artifact BLEEP fallback receipt is malformed for {donor}")

    donor_type_evaluable = bool(predictions["cross_assay_donor_type_proxy_evaluable"])
    donor_type_names = np.asarray(
        predictions["cross_assay_donor_type_proxy_type_names"]
    ).astype(str)
    donor_type_scalars = (
        "cross_assay_donor_type_proxy_pairs",
        "cross_assay_donor_type_proxy_matched_MSE",
        "cross_assay_donor_type_proxy_mismatched_MSE",
        "cross_assay_donor_type_proxy_ratio",
        "cross_assay_donor_type_proxy_unaligned_ratio",
    )
    if any(np.asarray(predictions[field]).shape != () for field in donor_type_scalars):
        raise ValueError(f"prediction artifact donor-type alignment is malformed for {donor}")
    if donor_type_evaluable and (
        not len(donor_type_names)
        or int(predictions["cross_assay_donor_type_proxy_pairs"]) < 2
        or not all(np.isfinite(predictions[field]) for field in donor_type_scalars[1:])
    ):
        raise ValueError(f"prediction artifact donor-type alignment is malformed for {donor}")
    donor_type_ratio = float(predictions["cross_assay_donor_type_proxy_ratio"])
    donor_type_unaligned_ratio = float(
        predictions["cross_assay_donor_type_proxy_unaligned_ratio"]
    )
    expected_donor_type_support = bool(
        donor_type_evaluable
        and np.isfinite(donor_type_ratio)
        and np.isfinite(donor_type_unaligned_ratio)
        and donor_type_ratio < 1.0
        and donor_type_ratio < donor_type_unaligned_ratio
    )
    if bool(
        predictions["cross_assay_donor_type_proxy_support_criterion_met"]
    ) != expected_donor_type_support:
        raise ValueError(f"prediction artifact donor-type support receipt is malformed for {donor}")

    pre_ratio = float(predictions["cross_assay_alignment_pre_matched_to_mismatched_ratio"])
    post_ratio = float(predictions["cross_assay_alignment_post_matched_to_mismatched_ratio"])
    unaligned_ratio = float(
        predictions["cross_assay_unaligned_post_matched_to_mismatched_ratio"]
    )
    applications = int(predictions["cross_assay_alignment_optimizer_applications_per_epoch"])
    total_applications = int(predictions["cross_assay_alignment_optimizer_applications_total"])
    if (
        not bool(predictions["cross_assay_alignment_support_criterion_met"])
        or not bool(predictions["cross_assay_alignment_beats_unaligned_comparator"])
        or float(predictions["cross_assay_alignment_weight"]) != 1.0
        or not (0 <= post_ratio < min(1.0, pre_ratio, unaligned_ratio))
        or applications <= 0
        or total_applications != applications * int(epochs)
    ):
        raise ValueError(f"prediction artifact alignment receipt is malformed for {donor}")


def score(args: argparse.Namespace) -> Mapping[str, object]:
    if not args.smoke:
        _validate_args(args)
    core = _import_core()
    prepared = _read_prepared_manifest(args.output)
    if int(args.seed) != int(prepared.get("base_seed", -1)):
        raise ValueError("score seed differs from the hash-bound prepared seed")
    _validate_bound_training_configuration(args, prepared)
    prediction_manifest_path = args.output / "fit_predict_manifest.json"
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    _validate_prediction_manifest_binding(args, core, prepared, prediction_manifest)
    fold_reports: dict[str, object] = {}
    donor_losses: dict[str, Mapping[str, float]] = {}
    donor_log_likelihoods: dict[str, Mapping[str, float]] = {}
    donor_posterior_log_scores: dict[str, Mapping[str, float]] = {}
    for donor in prepared["donors"]:
        donor = str(donor)
        fold = prepared["folds"][donor]
        prediction_receipt = prediction_manifest["folds"][donor]
        predictions = _verify_semantic_file(
            Path(str(prediction_receipt["prediction_path"])),
            str(prediction_receipt["prediction_semantic_sha256"]),
        )
        # Revalidate the complete target-free artifact immediately before this
        # donor's outcome is opened.  Global preflight catches stale folds; this
        # second check also closes a mutation window between preflight and use.
        public = _verify_semantic_file(
            Path(str(fold["public_path"])), str(fold["public_semantic_sha256"])
        )
        _validate_prediction_artifact(
            predictions,
            public,
            donor=donor,
            epochs=args.epochs,
        )
        secret = _verify_semantic_file(
            Path(str(fold["score_target_path"])), str(fold["score_target_semantic_sha256"])
        )
        report = _score_fold(core, secret, predictions, seed=int(fold["seed"]))
        fold_reports[donor] = report
        donor_losses[donor] = report["mean_nb_deviance"]
        donor_log_likelihoods[donor] = report["heldout_nb_log_likelihood"]
        donor_posterior_log_scores[donor] = report[
            "posterior_predictive_NB_log_score_M0_to_M3"
        ]
        _atomic_json(args.output / "folds" / donor / "score_report.json", report)
    gates = _evaluate_gates(
        core,
        donor_losses,
        {donor: str(report["indication"]) for donor, report in fold_reports.items()},
        seed=int(prepared["base_seed"]),
    )
    quality_aggregate = _donor_balanced_quality(fold_reports)
    metric_suite_complete, missing_metrics = _quality_completeness(fold_reports)
    for donor, fold in sorted(fold_reports.items()):
        alignment = fold.get("cross_assay_alignment", {})
        if not isinstance(alignment, Mapping):
            missing_metrics.append(f"{donor}:cross_assay_alignment")
            continue
        if alignment.get("support_criterion_met") is not True:
            missing_metrics.append(f"{donor}:cross_assay_alignment_support")
        if alignment.get("beats_unaligned_comparator") is not True:
            missing_metrics.append(f"{donor}:cross_assay_unaligned_comparator")
    for donor, values in sorted(donor_log_likelihoods.items()):
        for arm in (*MODEL_ARMS, "M8", "M3_same_M8_target"):
            if not _is_finite_number(values.get(arm)):
                missing_metrics.append(f"{donor}:{arm}:heldout_nb_log_likelihood")
    for donor, values in sorted(donor_posterior_log_scores.items()):
        for arm in ("M0", "M1", "M2", "M3"):
            if not _is_finite_number(values.get(arm)):
                missing_metrics.append(f"{donor}:{arm}:posterior_predictive_NB_log_score")
    metric_suite_complete = not missing_metrics
    secondary_status = {
        "20D_latent_MSE": "complete_training_fitted_20D_molecular_basis",
        "gene_and_program_correlation": "complete_section_balanced",
        "calibration_slope": "complete_section_balanced",
        "reliability_adjusted_variance": "complete_using_preexisting_molecule_halves",
        "program_covariance": "complete_on_frozen_prespecified_program_membership",
        "rare_state_recall": "complete_program_extreme_recall_not_cell_composition_truth",
        "predictive_interval_coverage_50_80_95": (
            "exploratory_nonblocking_NB2_mean_variance_normal_approximation"
        ),
        "reference_coverage": (
            "exploratory_nonblocking_fixed_development_distance_threshold"
        ),
        "abstention_rate": (
            "exploratory_nonblocking_out_of_support_flag_not_prediction_abstention"
        ),
        "retrieval_entropy": "complete_BLEEP_style_retrieval_diagnostic",
        "cross_assay_alignment": (
            "complete_training_only_scale_free_matched_vs_mismatched_with_lambda0_comparator"
        ),
    }
    bound_protocol_path = Path(str(prepared["protocol"]))
    protocol = json.loads(bound_protocol_path.read_text(encoding="utf-8"))
    margins = protocol.get("development_quality_margins")
    diagnostic_decision = gates["decision"]
    if metric_suite_complete and isinstance(margins, Mapping):
        quality_guard = dict(_quality_guard(quality_aggregate, margins))
        posterior_m0 = float(
            np.mean([value["M0"] for value in donor_posterior_log_scores.values()])
        )
        posterior_m3 = float(
            np.mean([value["M3"] for value in donor_posterior_log_scores.values()])
        )
        approximate_likelihood_check = {
            "M0": posterior_m0,
            "M3": posterior_m3,
            "passed": posterior_m3 > posterior_m0,
            "blocking": False,
            "reason": (
                "8_point_Gauss_Hermite_over_a_moment_matched_lognormal_rate_"
                "is_an_approximation_not_exact_component_posterior_integration"
            ),
        }
        plug_in_m0 = float(np.mean([value["M0"] for value in donor_log_likelihoods.values()]))
        plug_in_m3 = float(np.mean([value["M3"] for value in donor_log_likelihoods.values()]))
        plug_in_likelihood_check = {
            "M0": plug_in_m0,
            "M3": plug_in_m3,
            "passed": plug_in_m3 > plug_in_m0,
            "blocking": True,
            "reason": "shared_decoder_heldout_NB2_plug_in_log_likelihood",
        }
        quality_guard["checks"][
            "approximate_posterior_predictive_NB_log_score"
        ] = approximate_likelihood_check
        quality_guard["checks"]["heldout_NB_plug_in_log_likelihood"] = (
            plug_in_likelihood_check
        )
        alignment_check = {
            "folds_passing_support_criterion": int(
                sum(
                    bool(fold["cross_assay_alignment"]["support_criterion_met"])
                    for fold in fold_reports.values()
                )
            ),
            "folds_beating_unaligned_comparator": int(
                sum(
                    bool(fold["cross_assay_alignment"]["beats_unaligned_comparator"])
                    for fold in fold_reports.values()
                )
            ),
            "required_folds": len(fold_reports),
            "passed": all(
                bool(fold["cross_assay_alignment"]["support_criterion_met"])
                and bool(fold["cross_assay_alignment"]["beats_unaligned_comparator"])
                for fold in fold_reports.values()
            ),
            "blocking": True,
        }
        quality_guard["checks"]["cross_assay_alignment"] = alignment_check
        quality_guard["passed"] = bool(
            quality_guard["passed"]
            and plug_in_likelihood_check["passed"]
            and alignment_check["passed"]
        )
        gates["quality_guard"] = quality_guard
        gates["decision"] = {
            **diagnostic_decision,
            "status": "development_gates_and_quality_guard_evaluated",
            "quality_preservation_passed": quality_guard["passed"],
            "central_signal_candidate_for_external_test": bool(
                diagnostic_decision["central_development_gate_passed"]
                and quality_guard["passed"]
            ),
            "development_candidate_supported": bool(
                diagnostic_decision["full_model_chain_supported"]
                and quality_guard["passed"]
            ),
            "claim_progression_allowed": bool(
                diagnostic_decision["full_model_chain_supported"]
                and quality_guard["passed"]
            ),
            "claim_progression_scope": (
                "proposed_multimodal_model_independent_preregistered_regional_"
                "validation_only_after_gates_1_to_3_and_quality"
            ),
        }
    else:
        gates["diagnostic_decision_before_completeness_guard"] = diagnostic_decision
        gates["quality_guard"] = {
            "status": "not_evaluable",
            "missing_metric_count": len(missing_metrics),
            "missing_metrics": missing_metrics,
            "frozen_margins_available": isinstance(margins, Mapping),
        }
        gates["decision"] = {
            "supported": False,
            "status": "not_evaluable_incomplete_required_metric_suite",
            "claim_progression_allowed": False,
        }
    gate3_evaluable_donors = [
        donor
        for donor, fold in fold_reports.items()
        if bool(fold["gate3_supported_comparison"]["evaluable"])
    ]
    gate3_summary: dict[str, object] = {
        "all_donors_evaluable": len(gate3_evaluable_donors) == len(fold_reports),
        "evaluable_donors": gate3_evaluable_donors,
        "required_donors": len(fold_reports),
        "comparison": "M3_supported_vs_M2_supported",
        "full_target_M3_vs_M2_role": "descriptive_mixed_support_not_gating",
    }
    if len(gate3_evaluable_donors) == len(fold_reports):
        gate3_summary["donor_balanced_mean_nb_deviance"] = {
            arm: float(np.mean([values[arm] for values in donor_losses.values()]))
            for arm in ("M2_supported", "M3_supported")
        }
        gate3_summary["donor_balanced_mean_nb_log_likelihood"] = {
            arm: float(
                np.mean([values[arm] for values in donor_log_likelihoods.values()])
            )
            for arm in ("M2_supported", "M3_supported")
        }
    report = {
        "schema": SCHEMA,
        "analysis_scope": "exposed_development_only_non_confirmatory",
        "evidence_status": "development_diagnostics_only",
        "can_confirm_scientific_hypothesis": False,
        "can_authorize_regional_HEIR": False,
        "can_authorize_cell_level_HEIR": False,
        "image_encoder": HOPTIMUS_REPOSITORY,
        "image_scale_um": 112,
        "reference_assay": "registered_suspension_type_cell_not_verified_snRNA",
        "uni2_h_run": False,
        "primary_endpoint": "donor_balanced_heldout_NB_deviance_256_gene_panel",
        "artifact_identities": {
            "source_sha256": str(prepared["source_sha256"]),
            "panel_sha256": str(prepared["panel_sha256"]),
            "protocol_sha256": str(prepared["protocol_sha256"]),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "core_sha256": _sha256(Path(core.__file__).resolve()),
            "fit_predict_manifest_sha256": _sha256(prediction_manifest_path),
        },
        "frozen_training_configuration": {
            "base_seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "latent_dim": int(args.latent_dim),
            "device": str(args.device),
        },
        "folds": fold_reports,
        "donor_balanced_full_target_mean_nb_deviance": {
            arm: float(np.mean([values[arm] for values in donor_losses.values()]))
            for arm in MODEL_ARMS
        },
        "donor_balanced_M8_same_half_target_mean_nb_deviance": {
            arm: float(np.mean([values[arm] for values in donor_losses.values()]))
            for arm in ("M3_same_M8_target", "M8")
        },
        "donor_balanced_full_target_mean_nb_log_likelihood": {
            arm: float(np.mean([values[arm] for values in donor_log_likelihoods.values()]))
            for arm in MODEL_ARMS
        },
        "donor_balanced_M8_same_half_target_mean_nb_log_likelihood": {
            arm: float(np.mean([values[arm] for values in donor_log_likelihoods.values()]))
            for arm in ("M3_same_M8_target", "M8")
        },
        "donor_balanced_posterior_predictive_NB_log_score": {
            arm: float(
                np.mean([values[arm] for values in donor_posterior_log_scores.values()])
            )
            for arm in ("M0", "M1", "M2", "M3")
        },
        "indication_equal_full_target_mean_nb_deviance": {
            arm: float(
                np.mean(
                    [
                        np.mean(
                            [
                                donor_losses[donor][arm]
                                for donor, fold in fold_reports.items()
                                if fold["indication"] == indication
                            ]
                        )
                        for indication in sorted(
                            {fold["indication"] for fold in fold_reports.values()}
                        )
                    ]
                )
            )
            for arm in MODEL_ARMS
        },
        "indication_equal_M8_same_half_target_mean_nb_deviance": {
            arm: float(
                np.mean(
                    [
                        np.mean(
                            [
                                donor_losses[donor][arm]
                                for donor, fold in fold_reports.items()
                                if fold["indication"] == indication
                            ]
                        )
                        for indication in sorted(
                            {fold["indication"] for fold in fold_reports.values()}
                        )
                    ]
                )
            )
            for arm in ("M3_same_M8_target", "M8")
        },
        "indication_equal_full_target_mean_nb_log_likelihood": {
            arm: float(
                np.mean(
                    [
                        np.mean(
                            [
                                donor_log_likelihoods[donor][arm]
                                for donor, fold in fold_reports.items()
                                if fold["indication"] == indication
                            ]
                        )
                        for indication in sorted(
                            {fold["indication"] for fold in fold_reports.values()}
                        )
                    ]
                )
            )
            for arm in MODEL_ARMS
        },
        "indication_equal_M8_same_half_target_mean_nb_log_likelihood": {
            arm: float(
                np.mean(
                    [
                        np.mean(
                            [
                                donor_log_likelihoods[donor][arm]
                                for donor, fold in fold_reports.items()
                                if fold["indication"] == indication
                            ]
                        )
                        for indication in sorted(
                            {fold["indication"] for fold in fold_reports.values()}
                        )
                    ]
                )
            )
            for arm in ("M3_same_M8_target", "M8")
        },
        "indication_equal_posterior_predictive_NB_log_score": {
            arm: float(
                np.mean(
                    [
                        np.mean(
                            [
                                donor_posterior_log_scores[donor][arm]
                                for donor, fold in fold_reports.items()
                                if fold["indication"] == indication
                            ]
                        )
                        for indication in sorted(
                            {fold["indication"] for fold in fold_reports.values()}
                        )
                    ]
                )
            )
            for arm in ("M0", "M1", "M2", "M3")
        },
        "aggregation_order": (
            "spots_within_section_then_sections_within_donor_then_donors;"
            " indication_equal_secondary"
        ),
        "donor_balanced_secondary_metrics": quality_aggregate,
        "donor_balanced_BLEEP_retrieval_entropy": float(
            np.mean([fold["BLEEP_retrieval_entropy"] for fold in fold_reports.values()])
        ),
        "gate3_supported_comparison": gate3_summary,
        "donor_balanced_cross_assay_alignment": {
            "mean_pre_matched_to_mismatched_ratio": float(
                np.mean(
                    [
                        fold["cross_assay_alignment"][
                            "pre_matched_to_mismatched_ratio"
                        ]
                        for fold in fold_reports.values()
                    ]
                )
            ),
            "mean_post_matched_to_mismatched_ratio": float(
                np.mean(
                    [
                        fold["cross_assay_alignment"][
                            "post_matched_to_mismatched_ratio"
                        ]
                        for fold in fold_reports.values()
                    ]
                )
            ),
            "mean_unaligned_post_matched_to_mismatched_ratio": float(
                np.mean(
                    [
                        fold["cross_assay_alignment"][
                            "unaligned_post_matched_to_mismatched_ratio"
                        ]
                        for fold in fold_reports.values()
                    ]
                )
            ),
            "all_folds_support_criterion": all(
                bool(fold["cross_assay_alignment"]["support_criterion_met"])
                for fold in fold_reports.values()
            ),
            "all_folds_beat_unaligned_comparator": all(
                bool(fold["cross_assay_alignment"]["beats_unaligned_comparator"])
                for fold in fold_reports.values()
            ),
            "donor_type_proxy_all_folds_evaluable": all(
                bool(
                    fold["cross_assay_alignment"]["donor_type_proxy_diagnostic"][
                        "evaluable"
                    ]
                )
                for fold in fold_reports.values()
            ),
            "donor_type_proxy_all_folds_support_criterion": all(
                bool(
                    fold["cross_assay_alignment"]["donor_type_proxy_diagnostic"][
                        "support_criterion_met"
                    ]
                )
                for fold in fold_reports.values()
            ),
        },
        "ordered_gates": gates,
        "implementation_status": {
            "primary_arm_identity_contracts": "implemented",
            "required_secondary_metrics": secondary_status,
            "primary_metric_suite_complete": metric_suite_complete,
            "central_development_experiment_complete": metric_suite_complete,
            "full_protocol_implementation_complete": False,
            "scientific_implementation_complete": False,
            "central_gate_attribution_allowed": metric_suite_complete,
            "gate_attribution_allowed": False,
            "gate_attribution_scope": (
                "central_exposed_development_only_when_primary_metric_suite_complete"
            ),
            "sensitivity_status": {
                "H_optimus_55um": (
                    "unavailable_registered_source_has_no_55um_features_not_run"
                ),
                "fold_local_gene_panel_LODO": "not_run",
                "soft_composition_weighted_reference": "not_run",
                "fixed_ESS_wrong_and_generic_reference": "not_run",
            },
            "missing_required_metric_count": len(missing_metrics),
            "missing_required_metrics": missing_metrics,
        },
        "iterative_refinement_run": False,
        "limitations": [
            "NatCommun outcomes were used for exposed development and cannot confirm the claim",
            "regional Visium spots cannot authorize cell-level hypotheses",
            "the registered suspension reference is annotated as cell and is not verified snRNA",
            "M8 is an empirical cross-half molecular oracle, not a mathematical ST floor",
            "full-depth measurement-noise risk is unavailable without a registered replicate model",
            (
                "posterior uncertainty propagates state and NB2 variance but not "
                "composition uncertainty"
            ),
            "the reported abstention rate is an out-of-support flag; predictions are not withheld",
            (
                "M6 and M7 use natural reference banks; personalization is not "
                "attributable without fixed-ESS controls"
            ),
            (
                "55-um H-optimus, panel-LODO, soft-weighted-reference, and "
                "fixed-ESS sensitivities were not run"
            ),
            "independent matched regional confirmation remains required",
        ],
    }
    _atomic_json(args.output / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "fit-predict", "score", "all"), default="all"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--projected-source", type=Path, default=DEFAULT_PROJECTED_SOURCE)
    parser.add_argument(
        "--expected-projected-source-sha256",
        default=DEFAULT_PROJECTED_SOURCE_SHA256,
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.60)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true", help="allow a tiny synthetic panel/source")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    reject_uni2(args.device)
    if args.latent_dim != 20 and not args.smoke:
        raise ValueError("the frozen secondary latent dimension is 20")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.smoke and args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if not args.smoke and args.device == "cpu":
        raise ValueError("the real benchmark must use bounded CUDA execution")
    if not args.smoke:
        observed = {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "latent_dim": int(args.latent_dim),
        }
        expected = {
            "seed": FROZEN_BASE_SEED,
            "epochs": FROZEN_EPOCHS,
            "batch_size": FROZEN_BATCH_SIZE,
            "latent_dim": FROZEN_LATENT_DIM,
        }
        mismatched = [name for name in expected if observed[name] != expected[name]]
        if mismatched:
            raise ValueError(
                f"real benchmark arguments differ from frozen protocol: {mismatched}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    configure_resources(
        cpu_threads=args.cpu_threads,
        gpu_memory_fraction=args.gpu_memory_fraction,
        device=args.device,
    )
    seed_everything(args.seed)
    stages = ("prepare", "fit-predict", "score") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "prepare":
            prepare(args)
        elif stage == "fit-predict":
            fit_predict(args)
        else:
            score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
