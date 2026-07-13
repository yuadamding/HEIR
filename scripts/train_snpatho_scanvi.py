#!/usr/bin/env python3
"""Train an FFPE-snPATHO scVI/scANVI teacher with optional true donor holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scvi

from heir.data import RNAReference
from heir.expression import EXPRESSION_SPACE_ID
from heir.prior import SCVIAdapter
from heir.prior.scvi_adapter import SCVI_EXPRESSION_NORMALIZATION_CONTRACT

SAMPLES = ("4066", "4399", "4411")
MOLECULAR_DESIGNS = (
    "no_specimen_correction",
    "technical_batch_only",
    "specimen_batch_sensitivity",
)


@dataclass(frozen=True)
class MolecularTrainingPlan:
    """Declare which donors may influence fitted molecular parameters."""

    training_donors: tuple[str, ...]
    held_out_sample: Optional[str]

    @property
    def mode(self) -> str:
        return (
            "historical_all_donor_negative_control"
            if self.held_out_sample is None
            else "leave_one_donor_out"
        )

    @property
    def uses_frozen_query(self) -> bool:
        return self.held_out_sample is not None

    def latent_role(self, sample: str) -> str:
        if sample not in SAMPLES:
            raise ValueError("unknown snPATHO sample: %s" % sample)
        return "frozen_query_encoder" if sample == self.held_out_sample else "reference_encoder"


def _training_plan(held_out_sample: Optional[str]) -> MolecularTrainingPlan:
    normalized = None if held_out_sample is None else str(held_out_sample).strip()
    if normalized is not None and normalized not in SAMPLES:
        raise ValueError("held-out sample must be one of: %s" % ", ".join(SAMPLES))
    training_donors = tuple(sample for sample in SAMPLES if sample != normalized)
    if normalized is not None and len(training_donors) != len(SAMPLES) - 1:
        raise RuntimeError("leave-one-donor-out planning did not produce two training donors")
    return MolecularTrainingPlan(
        training_donors=training_donors,
        held_out_sample=normalized,
    )


def _validate_molecular_design_for_plan(
    plan: MolecularTrainingPlan,
    molecular_design: str,
) -> None:
    """Reject batch models that cannot be transferred without target leakage."""

    if molecular_design not in MOLECULAR_DESIGNS:
        raise ValueError("unsupported molecular design: %s" % molecular_design)
    if plan.uses_frozen_query and molecular_design == "specimen_batch_sensitivity":
        raise ValueError(
            "specimen_batch_sensitivity is invalid for a held-out donor: its section_id "
            "category is absent from the reference fit and would require query-category "
            "extension"
        )


def _training_partition_provenance(plan: MolecularTrainingPlan) -> dict[str, object]:
    """Build the auditable leakage boundary shared by all emitted artifacts."""

    held_out = plan.held_out_sample
    if held_out is not None and held_out in plan.training_donors:
        raise ValueError("held-out donor cannot appear in molecular training donors")
    return {
        "mode": plan.mode,
        "held_out_sample": held_out,
        "backbone_training_donors": list(plan.training_donors),
        "decoder_training_donors": list(plan.training_donors),
        "all_donor_behavior_role": (
            "historical_negative_control" if held_out is None else "not_applicable"
        ),
        "held_out_mapping": (
            {
                "method": "SCANVI.load_query_data_without_query_training",
                "labels_available_to_query_model": False,
                "query_train_called": False,
                "query_parameters_frozen_before_inference": True,
                "inference_guard_enabled_without_optimization": True,
                "held_out_expression_used_for_fitting": False,
                "held_out_annotation_used_for_label_mapping": False,
                "label_mapping_method": "frozen_training_donor_SCANVI_classifier",
                "label_training_donors": list(plan.training_donors),
            }
            if held_out is not None
            else None
        ),
    }


def _latent_artifact_provenance(
    plan: MolecularTrainingPlan,
    sample: str,
) -> dict[str, object]:
    """Bind each emitted latent to the donors that fit its frozen encoder."""

    role = plan.latent_role(sample)
    if plan.held_out_sample is not None and plan.held_out_sample in plan.training_donors:
        raise ValueError("held-out donor leaked into latent-training provenance")
    return {
        "inference_role": role,
        "latent_training_donors": list(plan.training_donors),
        "sample_expression_used_for_model_fitting": sample in plan.training_donors,
        "sample_labels_used_for_model_fitting": sample in plan.training_donors,
        "cell_type_label_source": (
            "frozen_training_donor_SCANVI_classifier"
            if sample == plan.held_out_sample
            else "published_training_donor_annotation"
        ),
        "cell_type_label_training_donors": list(plan.training_donors),
        "sample_annotation_used_for_cell_type_labels": sample in plan.training_donors,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_input_files(
    inputs: Sequence[tuple[str, Path]],
) -> tuple[dict[str, object], ...]:
    """Hash every immutable molecular input before any long-running read."""

    records = []
    seen_roles = set()
    seen_paths = set()
    for role, value in inputs:
        role = str(role).strip()
        path = value.expanduser().resolve()
        if not role or role in seen_roles:
            raise ValueError("molecular input roles must be unique and non-empty")
        if path in seen_paths:
            raise ValueError("molecular input paths must be unique")
        if not path.is_file():
            raise FileNotFoundError(path)
        before = path.stat()
        sha256 = _file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("molecular input changed while it was being frozen: %s" % path)
        records.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256,
                "size_bytes": int(after.st_size),
            }
        )
        seen_roles.add(role)
        seen_paths.add(path)
    return tuple(records)


def _assert_input_files_unchanged(
    records: Sequence[dict[str, object]],
    *,
    phase: str,
) -> None:
    """Rehash frozen inputs at long-run read/output provenance boundaries."""

    for record in records:
        path = Path(str(record["path"])).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError("molecular input disappeared during %s: %s" % (phase, path))
        expected_size = int(record["size_bytes"])
        expected_sha256 = str(record["sha256"])
        before = path.stat()
        actual_sha256 = _file_sha256(path)
        after = path.stat()
        changed_while_hashing = (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        )
        if (
            changed_while_hashing
            or after.st_size != expected_size
            or actual_sha256 != expected_sha256
        ):
            raise RuntimeError(
                "molecular input changed during %s: %s; discard partial outputs and rerun "
                "from immutable inputs" % (phase, path)
            )


def _input_record(
    records: Sequence[dict[str, object]],
    role: str,
) -> dict[str, object]:
    matches = [record for record in records if record["role"] == role]
    if len(matches) != 1:
        raise RuntimeError("frozen molecular input role is absent or duplicated: %s" % role)
    return matches[0]


def _ordered_string_sha256(values: Sequence[object]) -> str:
    normalized = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    encoded = json.dumps(
        {"dtype": "string", "shape": [len(normalized)], "values": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _align_frozen_query_labels(
    reference_cell_ids: Sequence[object],
    query_cell_ids: Sequence[object],
    predicted_labels: Sequence[object],
) -> np.ndarray:
    """Align frozen classifier outputs without consulting target annotations."""

    reference_ids = tuple(str(value) for value in reference_cell_ids)
    query_ids = tuple(str(value) for value in query_cell_ids)
    labels = tuple(str(value) for value in predicted_labels)
    if len(query_ids) != len(labels) or len(set(query_ids)) != len(query_ids):
        raise ValueError("frozen query label rows are duplicated or misaligned")
    if len(set(reference_ids)) != len(reference_ids) or set(reference_ids) != set(query_ids):
        raise ValueError("frozen query label cells differ from the RNA reference")
    lookup = dict(zip(query_ids, labels))
    aligned = np.asarray([lookup[value] for value in reference_ids], dtype=np.str_)
    if any(not value.strip() for value in aligned.tolist()):
        raise ValueError("frozen query labels must be non-empty")
    return aligned


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _gene_panel(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        genes = tuple(
            line.split("\t", 1)[0].strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        )
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("gene panel must contain unique genes")
    return genes


def _stratified_validation_mask(
    section_values: np.ndarray,
    *,
    fraction: float,
    seed: int,
    samples: Sequence[str] = SAMPLES,
) -> np.ndarray:
    """Select a deterministic validation subset from every specimen."""

    if not 0.0 < fraction < 0.5:
        raise ValueError("decoder validation fraction must lie in (0, 0.5)")
    values = np.asarray(section_values).astype(str)
    mask = np.zeros(len(values), dtype=bool)
    generator = np.random.default_rng(seed)
    normalized_samples = tuple(str(sample) for sample in samples)
    if not normalized_samples or len(set(normalized_samples)) != len(normalized_samples):
        raise ValueError("decoder validation samples must be unique and non-empty")
    for sample in normalized_samples:
        indices = np.flatnonzero(values == sample)
        if len(indices) < 2:
            raise ValueError("decoder validation requires at least two cells for %s" % sample)
        count = min(len(indices) - 1, max(1, int(round(fraction * len(indices)))))
        mask[generator.permutation(indices)[:count]] = True
    return mask


def _technical_batch_contract(combined: ad.AnnData, key: Optional[str]) -> tuple[str, ...]:
    if not key:
        raise ValueError(
            "technical_batch_only requires --technical-batch-key naming an observed run or "
            "chemistry field"
        )
    if key not in combined.obs:
        raise ValueError("technical batch key is absent from the FFPE reference: %s" % key)
    raw_batches = combined.obs[key].to_numpy(dtype=object)
    sections = combined.obs["section_id"].to_numpy(dtype=object)
    levels = SCVIAdapter.validate_crossed_technical_batch(
        sections,
        raw_batches,
        key=key,
    )
    batches = combined.obs[key].astype(str).str.strip()
    combined.obs[key] = batches.astype("category")
    return levels


def _clean_categorical_values(values: Sequence[object], *, label: str) -> np.ndarray:
    cleaned = []
    for value in np.asarray(values, dtype=object).tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError("%s contains missing values" % label)
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            raise ValueError("%s contains missing or empty values" % label)
        cleaned.append(text)
    return np.asarray(cleaned, dtype=np.str_)


def _held_out_technical_batch_contract(
    combined: ad.AnnData,
    *,
    plan: MolecularTrainingPlan,
    key: str,
    reference_levels: Sequence[str],
) -> tuple[str, ...]:
    """Prove that frozen-query mapping needs no categorical level extension."""

    if not plan.uses_frozen_query or plan.held_out_sample is None:
        return ()
    if key not in combined.obs:
        raise ValueError("technical batch key is absent from the FFPE reference: %s" % key)
    sections = _clean_categorical_values(
        combined.obs["section_id"].to_numpy(dtype=object),
        label="section identity",
    )
    batches = _clean_categorical_values(
        combined.obs[key].to_numpy(dtype=object),
        label="technical batch key",
    )
    selected = sections == plan.held_out_sample
    if not selected.any():
        raise ValueError("held-out donor has no rows in the FFPE reference")
    held_out_levels = tuple(sorted(set(batches[selected].tolist())))
    reference_level_set = set(str(level) for level in reference_levels)
    unseen = sorted(set(held_out_levels) - reference_level_set)
    if unseen:
        raise ValueError(
            "held-out technical batch levels are absent from the reference training donors: %s"
            % ", ".join(unseen)
        )
    return held_out_levels


def _lock_query_technical_categories(
    query: ad.AnnData,
    *,
    key: str,
    reference_levels: Sequence[str],
) -> tuple[str, ...]:
    """Install exactly the reference registry categories before query loading."""

    if key not in query.obs:
        raise ValueError("held-out query lacks technical batch key: %s" % key)
    levels = tuple(str(level).strip() for level in reference_levels)
    if not levels or any(not level for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("reference technical batch levels are invalid")
    values = _clean_categorical_values(
        query.obs[key].to_numpy(dtype=object),
        label="held-out technical batch key",
    )
    observed = tuple(sorted(set(values.tolist())))
    unseen = sorted(set(observed) - set(levels))
    if unseen:
        raise ValueError(
            "held-out technical batch levels are absent from the reference training donors: %s"
            % ", ".join(unseen)
        )
    query.obs[key] = pd.Categorical(values, categories=list(levels), ordered=False)
    if tuple(str(value) for value in query.obs[key].cat.categories) != levels:
        raise RuntimeError("held-out technical categories were not locked to the reference")
    if bool(query.obs[key].isna().any()):
        raise RuntimeError("locking held-out technical categories introduced missing values")
    return observed


def _frozen_query_mapping(
    held_out: ad.AnnData,
    *,
    reference_model: Path,
    latent_dim: int,
    technical_batch_key: Optional[str] = None,
    reference_technical_batch_levels: Sequence[str] = (),
) -> tuple[object, ad.AnnData, np.ndarray, np.ndarray, dict[str, object]]:
    """Encode a held-out donor without optimizing any query-model parameter.

    ``load_query_data`` transfers the trained reference weights and registry to
    the query object.  Deliberately omitting ``query_model.train()`` is the
    leakage boundary: held-out counts are used only as encoder input, never as
    an optimization target.  Published labels are removed before registry
    transfer so they cannot reach the classifier or any fitting decision.
    """

    query = held_out.copy()
    technical_category_audit = None
    if technical_batch_key is not None:
        observed_levels = _lock_query_technical_categories(
            query,
            key=technical_batch_key,
            reference_levels=reference_technical_batch_levels,
        )
        technical_category_audit = {
            "key": technical_batch_key,
            "reference_levels": list(reference_technical_batch_levels),
            "held_out_levels": list(observed_levels),
            "novel_levels": [],
            "category_extension_allowed": False,
        }
    elif reference_technical_batch_levels:
        raise ValueError("reference technical levels require a technical batch key")
    labels_removed = "major_annotation" in query.obs
    if labels_removed:
        query.obs.drop(columns=["major_annotation"], inplace=True)
    query_model = scvi.model.SCANVI.load_query_data(
        query,
        str(reference_model),
        inplace_subset_query_vars=True,
        accelerator="gpu",
        device=0,
    )
    query_model.module.eval()
    parameter_count = 0
    for parameter in query_model.module.parameters():
        parameter.requires_grad_(False)
        parameter_count += int(parameter.numel())
    if any(parameter.requires_grad for parameter in query_model.module.parameters()):
        raise RuntimeError("held-out query model still has trainable parameters")
    # scvi-tools marks every load_query_data result untrained because its usual
    # scArches workflow expects a subsequent ``train`` call.  HEIR deliberately
    # skips that optimization.  Once all parameters are frozen, enable only the
    # inference API guard so get_latent_representation/get_normalized_expression
    # can run against the copied reference weights.
    query_model.is_trained = True
    latent = np.asarray(query_model.get_latent_representation(), dtype=np.float32)
    if latent.shape != (query.n_obs, latent_dim) or not np.isfinite(latent).all():
        raise RuntimeError("frozen scANVI query encoder returned an invalid latent representation")
    predicted_labels = np.asarray(query_model.predict(soft=False)).astype(str)
    if predicted_labels.shape != (query.n_obs,) or any(
        not value.strip() for value in predicted_labels.tolist()
    ):
        raise RuntimeError("frozen scANVI classifier returned invalid held-out labels")
    label_counts = {
        str(label): int(count)
        for label, count in zip(*np.unique(predicted_labels, return_counts=True))
    }
    return (
        query_model,
        query,
        latent,
        predicted_labels,
        {
            "labels_removed_before_registry_transfer": labels_removed,
            "query_train_called": False,
            "parameters_frozen_before_inference": True,
            "inference_guard_enabled_without_optimization": True,
            "frozen_parameter_count": parameter_count,
            "cells_mapped": int(query.n_obs),
            "label_predictions_generated_without_target_annotation": True,
            "label_prediction_rule": "SCANVI.predict(soft=False)",
            "predicted_label_sha256": _ordered_string_sha256(predicted_labels),
            "predicted_label_counts": label_counts,
            **(
                {}
                if technical_category_audit is None
                else {"technical_batch_categories": technical_category_audit}
            ),
        },
    )


def _decoder_error(
    decoder: object,
    latent: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    import torch
    from torch.nn import functional as F

    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(selected) == 0:
        raise ValueError("decoder evaluation mask is empty")
    device = torch.device("cuda")
    module = decoder.to(device)
    module.decoder.eval()
    with torch.no_grad():
        prediction = module.decoder(torch.from_numpy(latent[selected]).to(device))
        observed = torch.from_numpy(target[selected]).to(device)
        smooth_l1 = float(F.smooth_l1_loss(prediction, observed).cpu())
        median_absolute_error = float(torch.median(torch.abs(prediction - observed)).cpu())
    module.cpu()
    return {
        "cells": int(len(selected)),
        "smooth_l1": smooth_l1,
        "median_absolute_error": median_absolute_error,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repository / "artifacts" / "snpatho" / "r1_ffpe",
    )
    parser.add_argument(
        "--native-model",
        type=Path,
        default=(
            repository.parent
            / "HEIR_assets"
            / "pretrained"
            / "snpatho_scanvi_r2_preserve_biology_v1"
        ),
    )
    parser.add_argument(
        "--decoder-output",
        type=Path,
        default=repository.parent
        / "HEIR_assets"
        / "pretrained"
        / "snpatho_scanvi_r2_preserve_biology_v1_decoder.pt",
    )
    parser.add_argument(
        "--latent-output-root",
        type=Path,
        default=repository / "artifacts" / "snpatho" / "r2_scanvi",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=repository / "artifacts" / "snpatho" / "r2_scanvi" / "provenance.json",
    )
    parser.add_argument(
        "--gene-panel",
        type=Path,
        default=repository / "manifests" / "gene_panel_snpatho_500.tsv",
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--scvi-epochs", type=int, default=150)
    parser.add_argument("--scanvi-epochs", type=int, default=100)
    parser.add_argument("--decoder-epochs", type=int, default=200)
    parser.add_argument("--decoder-posterior-samples", type=int, default=32)
    parser.add_argument(
        "--held-out-sample",
        choices=SAMPLES,
        default=None,
        help=(
            "fit the scVI/scANVI backbone and distilled decoder on the other two donors, "
            "then map this donor once through a frozen, label-free query encoder; omitting "
            "this option preserves the historical all-donor negative-control workflow"
        ),
    )
    parser.add_argument(
        "--molecular-design",
        choices=MOLECULAR_DESIGNS,
        default="no_specimen_correction",
        help=(
            "preserve specimen biology by default; the historical section_id batch model is "
            "available only as specimen_batch_sensitivity"
        ),
    )
    parser.add_argument(
        "--technical-batch-key",
        help="observed technical run/chemistry column; required only for technical_batch_only",
    )
    parser.add_argument(
        "--decoder-validation-policy",
        choices=("donor_rotated", "single_donor_sensitivity"),
        default="donor_rotated",
        help=(
            "historical all-donor decoder audit policy; --held-out-sample instead uses a "
            "training-donor-only stratified validation split"
        ),
    )
    parser.add_argument("--decoder-validation-sample", choices=SAMPLES, default="4411")
    parser.add_argument("--decoder-validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--decoder-rotation-root",
        type=Path,
        default=None,
        help="external directory for donor-rotated decoder audit checkpoints",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "deprecated safety flag; completed molecular output families are immutable, so "
            "choose fresh versioned paths instead"
        ),
    )
    args = parser.parse_args(argv)

    training_plan = _training_plan(args.held_out_sample)
    _validate_molecular_design_for_plan(training_plan, args.molecular_design)
    scvi.settings.seed = args.seed
    np.random.seed(args.seed)
    input_root = args.input_root.expanduser().resolve()
    native_model = args.native_model.expanduser().resolve()
    decoder_output = args.decoder_output.expanduser().resolve()
    latent_root = args.latent_output_root.expanduser().resolve()
    provenance_output = args.provenance_output.expanduser().resolve()
    panel_path = args.gene_panel.expanduser().resolve()
    rotation_root = (
        decoder_output.with_name(decoder_output.stem + "_donor_rotations")
        if args.decoder_rotation_root is None
        else args.decoder_rotation_root.expanduser().resolve()
    )
    if any(
        value <= 0
        for value in (
            args.latent_dim,
            args.scvi_epochs,
            args.scanvi_epochs,
            args.decoder_epochs,
            args.decoder_posterior_samples,
        )
    ):
        raise ValueError("latent width, epoch counts, and posterior samples must be positive")
    if args.decoder_posterior_samples <= 1:
        raise ValueError("decoder-posterior-samples must exceed one for MC averaging")
    if args.decoder_validation_policy == "donor_rotated" and not training_plan.uses_frozen_query:
        _stratified_validation_mask(
            np.concatenate([np.repeat(sample, 2) for sample in SAMPLES]),
            fraction=args.decoder_validation_fraction,
            seed=args.seed,
        )
    exact_outputs = (native_model, decoder_output, latent_root, provenance_output)
    if not training_plan.uses_frozen_query:
        exact_outputs = (*exact_outputs, rotation_root)
    if len(set(exact_outputs)) != len(exact_outputs):
        raise ValueError("molecular output paths must be distinct")
    output_directories = (native_model, latent_root)
    if not training_plan.uses_frozen_query:
        output_directories = (*output_directories, rotation_root)
    for directory in output_directories:
        if decoder_output.is_relative_to(directory) or directory.is_relative_to(decoder_output):
            raise ValueError("decoder output cannot contain or be contained by an output directory")
    for index, first in enumerate(output_directories):
        for second in output_directories[index + 1 :]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError("molecular output directories cannot be nested")
    existing = [path for path in exact_outputs if path.exists()]
    if existing:
        prefix = (
            "--overwrite cannot replace an existing molecular output family; "
            if args.overwrite
            else "molecular output already exists; "
        )
        raise FileExistsError(
            prefix + "choose fresh versioned output paths: " + ", ".join(map(str, existing))
        )

    h5ad_paths = {sample: (input_root / sample / "reference.h5ad").resolve() for sample in SAMPLES}
    reference500_paths = {
        sample: (input_root / sample / "reference500.npz").resolve() for sample in SAMPLES
    }
    molecular_input_records = _freeze_input_files(
        (
            *(("reference_h5ad:%s" % sample, h5ad_paths[sample]) for sample in SAMPLES),
            *(("reference500:%s" % sample, reference500_paths[sample]) for sample in SAMPLES),
            ("gene_panel", panel_path),
        )
    )
    inputs = []
    input_hashes = {}
    for sample in SAMPLES:
        source = h5ad_paths[sample]
        source_record = _input_record(molecular_input_records, "reference_h5ad:%s" % sample)
        _assert_input_files_unchanged(
            (source_record,),
            phase="immediately_before_reference_h5ad_read:%s" % sample,
        )
        values = ad.read_h5ad(source)
        if "major_annotation" not in values.obs or "processing_method" not in values.obs:
            raise ValueError("R1 AnnData lacks required annotation/workflow fields")
        if set(values.obs["processing_method"].astype(str)) != {"FFPE_snPATHO"}:
            raise ValueError("R1 AnnData includes a non-FFPE-snPATHO workflow")
        values.obs["section_id"] = sample
        values.obs["source_cell_id"] = values.obs_names.astype(str)
        values.obs["source_row"] = np.arange(values.n_obs, dtype=np.int64)
        inputs.append(values)
        input_hashes[sample] = str(source_record["sha256"])
    reference500_by_sample = {}
    reference500_hashes = {}
    for sample in SAMPLES:
        source_record = _input_record(molecular_input_records, "reference500:%s" % sample)
        _assert_input_files_unchanged(
            (source_record,),
            phase="immediately_before_reference500_read:%s" % sample,
        )
        reference500_by_sample[sample] = RNAReference.load_npz(reference500_paths[sample])
        reference500_hashes[sample] = str(source_record["sha256"])
    common_genes = tuple(str(value) for value in inputs[0].var_names)
    if any(
        tuple(str(value) for value in values.var_names) != common_genes for values in inputs[1:]
    ):
        raise ValueError("FFPE R1 references use different full gene orders")
    combined = ad.concat(inputs, join="inner", merge="same", index_unique="-")
    model_mask = combined.obs["section_id"].astype(str).isin(training_plan.training_donors)
    if int(model_mask.sum()) == 0 or (
        int(model_mask.sum()) == combined.n_obs and training_plan.uses_frozen_query
    ):
        raise RuntimeError("leave-one-donor-out partition is empty or failed to exclude its target")
    model_input = combined if not training_plan.uses_frozen_query else combined[model_mask].copy()
    training_label_values = _clean_categorical_values(
        model_input.obs["major_annotation"].to_numpy(dtype=object),
        label="molecular training-donor annotation",
    )
    training_label_ontology = tuple(sorted(set(training_label_values.tolist())))
    labels = pd.Categorical(training_label_values, categories=list(training_label_ontology))
    if "unknown" not in labels.categories:
        labels = labels.add_categories(["unknown"])
    model_input.obs["major_annotation"] = labels

    panel_record = _input_record(molecular_input_records, "gene_panel")
    _assert_input_files_unchanged(
        (panel_record,),
        phase="immediately_before_gene_panel_read",
    )
    panel = _gene_panel(panel_path)
    missing_panel = sorted(set(panel) - set(common_genes))
    if missing_panel:
        raise ValueError(
            "scANVI full reference is missing panel genes: %s" % ", ".join(missing_panel)
        )

    if args.molecular_design == "no_specimen_correction":
        if args.technical_batch_key:
            raise ValueError(
                "--technical-batch-key is only valid with --molecular-design technical_batch_only"
            )
        model_batch_key = None
        transform_batch: Optional[tuple[str, ...]] = None
        batch_correction_mode = "none"
        technical_batch_contingency = None
        technical_batch_query_contract = None
    elif args.molecular_design == "technical_batch_only":
        model_batch_key = str(args.technical_batch_key or "")
        transform_batch = _technical_batch_contract(model_input, args.technical_batch_key)
        held_out_technical_levels = _held_out_technical_batch_contract(
            combined,
            plan=training_plan,
            key=model_batch_key,
            reference_levels=transform_batch,
        )
        batch_correction_mode = "reference_batch_marginalization"
        technical_batch_contingency = {
            sample: {
                level: int(
                    (
                        (model_input.obs["section_id"].astype(str) == sample)
                        & (model_input.obs[model_batch_key].astype(str) == level)
                    ).sum()
                )
                for level in transform_batch
            }
            for sample in training_plan.training_donors
        }
        technical_batch_query_contract = (
            {
                "held_out_sample": training_plan.held_out_sample,
                "reference_levels": list(transform_batch),
                "held_out_levels": list(held_out_technical_levels),
                "novel_levels": [],
                "category_extension_allowed": False,
            }
            if training_plan.uses_frozen_query
            else None
        )
    else:
        if args.technical_batch_key:
            raise ValueError(
                "--technical-batch-key cannot be combined with specimen_batch_sensitivity"
            )
        model_batch_key = "section_id"
        transform_batch = training_plan.training_donors
        batch_correction_mode = "reference_batch_marginalization"
        technical_batch_contingency = None
        technical_batch_query_contract = None

    scvi.model.SCVI.setup_anndata(model_input, batch_key=model_batch_key)
    base = scvi.model.SCVI(
        model_input,
        n_latent=args.latent_dim,
        gene_likelihood="nb",
        n_layers=2,
        n_hidden=256,
    )
    base.train(
        max_epochs=args.scvi_epochs,
        accelerator="gpu",
        devices=1,
        early_stopping=True,
        check_val_every_n_epoch=1,
        batch_size=256,
    )
    model = scvi.model.SCANVI.from_scvi_model(
        base,
        unlabeled_category="unknown",
        labels_key="major_annotation",
    )
    model.train(
        max_epochs=args.scanvi_epochs,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=1,
        batch_size=256,
    )
    _assert_input_files_unchanged(
        molecular_input_records,
        phase="immediately_before_first_molecular_output",
    )
    native_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(native_model), overwrite=False, save_anndata=False)
    model_digest = _directory_sha256(native_model)
    latent_space_id = "sha256:" + model_digest
    reference_latent = np.asarray(model.get_latent_representation(), dtype=np.float32)
    if (
        reference_latent.shape != (model_input.n_obs, args.latent_dim)
        or not np.isfinite(reference_latent).all()
    ):
        raise RuntimeError("scANVI returned an invalid latent representation")

    latent_by_sample: dict[str, np.ndarray] = {}
    encoded_cell_ids: dict[str, np.ndarray] = {}
    model_sections = model_input.obs["section_id"].astype(str).to_numpy()
    model_source_ids = model_input.obs["source_cell_id"].astype(str).to_numpy()
    for sample in training_plan.training_donors:
        selected = np.flatnonzero(model_sections == sample)
        latent_by_sample[sample] = reference_latent[selected]
        encoded_cell_ids[sample] = model_source_ids[selected]

    query_model = None
    query_input = None
    query_audit = None
    query_predicted_labels = None
    if training_plan.held_out_sample is not None:
        held_out_sample = training_plan.held_out_sample
        held_out_mask = combined.obs["section_id"].astype(str).to_numpy() == held_out_sample
        held_out_input = combined[held_out_mask].copy()
        (
            query_model,
            query_input,
            query_latent,
            query_predicted_labels,
            query_audit,
        ) = _frozen_query_mapping(
            held_out_input,
            reference_model=native_model,
            latent_dim=args.latent_dim,
            technical_batch_key=(
                model_batch_key if args.molecular_design == "technical_batch_only" else None
            ),
            reference_technical_batch_levels=(() if transform_batch is None else transform_batch),
        )
        latent_by_sample[held_out_sample] = query_latent
        encoded_cell_ids[held_out_sample] = query_input.obs["source_cell_id"].astype(str).to_numpy()
        unsupported_predictions = sorted(
            set(query_predicted_labels.tolist()) - set(training_label_ontology)
        )
        if unsupported_predictions:
            raise RuntimeError(
                "frozen query classifier emitted labels absent from molecular training donors: %s"
                % ", ".join(unsupported_predictions)
            )
    latent = np.concatenate([latent_by_sample[sample] for sample in SAMPLES], axis=0)
    if latent.shape != (combined.n_obs, args.latent_dim):
        raise RuntimeError("reference and held-out query latents do not cover all snPATHO cells")

    _assert_input_files_unchanged(
        molecular_input_records,
        phase="immediately_before_latent_output_family",
    )
    latent_outputs = {}
    for sample in SAMPLES:
        reference = reference500_by_sample[sample]
        lookup = {
            source_id: latent_value
            for source_id, latent_value in zip(encoded_cell_ids[sample], latent_by_sample[sample])
        }
        missing = [str(value) for value in reference.cell_ids if str(value) not in lookup]
        if missing or len(lookup) != len(reference.cell_ids):
            raise ValueError("scANVI/reference cell alignment failed for %s" % sample)
        aligned = np.stack([lookup[str(value)] for value in reference.cell_ids]).astype(np.float32)
        aligned_labels = np.asarray(reference.cell_type_labels).astype(str)
        if sample == training_plan.held_out_sample:
            if query_predicted_labels is None:
                raise RuntimeError("held-out frozen classifier labels are unavailable")
            aligned_labels = _align_frozen_query_labels(
                reference.cell_ids,
                encoded_cell_ids[sample],
                query_predicted_labels,
            )
            if query_audit is None:
                raise RuntimeError("held-out query audit is unavailable")
            query_audit["predicted_label_sha256"] = _ordered_string_sha256(aligned_labels)
            query_audit["predicted_label_order"] = "RNAReference.cell_ids"
        output = latent_root / sample / "reference500_scanvi.npz"
        replace(
            reference,
            latent=aligned,
            cell_type_labels=aligned_labels,
            latent_space_id=latent_space_id,
            latent_training_donors=training_plan.training_donors,
            latent_transform_sha256=model_digest,
        ).save_npz(output)
        latent_outputs[sample] = {
            "path": str(output),
            "sha256": _file_sha256(output),
            "cells": int(len(aligned)),
            "cell_type_labels_sha256": _ordered_string_sha256(aligned_labels),
            **_latent_artifact_provenance(training_plan, sample),
        }

    adapter = SCVIAdapter(latent_dim=args.latent_dim, likelihood="nb")
    adapter.model = model
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    reference_expression_target = adapter.normalized_expression(
        model_input,
        panel,
        transform_batch=transform_batch,
        batch_correction_mode=batch_correction_mode,
        posterior_samples=args.decoder_posterior_samples,
    )
    held_out_expression_target = None

    decoder_data = model_input
    decoder_latent = reference_latent
    decoder_target = reference_expression_target
    section_values = model_input.obs["section_id"].astype(str).to_numpy()
    shared_decoder_options = {
        "transform_batch": transform_batch,
        "batch_correction_mode": batch_correction_mode,
        "posterior_samples": args.decoder_posterior_samples,
        "latent_target": decoder_latent,
        "max_epochs": args.decoder_epochs,
        "seed": args.seed,
        "device": "cuda",
    }
    shared_decoder_options["expression_target"] = decoder_target
    decoder_rotations = []
    if training_plan.uses_frozen_query:
        validation_mask = _stratified_validation_mask(
            section_values,
            fraction=args.decoder_validation_fraction,
            seed=args.seed,
            samples=training_plan.training_donors,
        )
        decoder_training_donors = training_plan.training_donors
        validation_description = "true_leave_one_donor_out_training_donor_stratified"
    elif args.decoder_validation_policy == "donor_rotated":
        _assert_input_files_unchanged(
            molecular_input_records,
            phase="immediately_before_decoder_rotation_output_family",
        )
        rotation_root.mkdir(parents=True, exist_ok=False)
        for held_out_sample in SAMPLES:
            rotation_path = rotation_root / ("heldout_%s.pt" % held_out_sample)
            held_out_mask = section_values == held_out_sample
            rotated = adapter.export_transferable_decoder_checkpoint(
                str(rotation_path),
                decoder_data,
                panel,
                validation_mask=held_out_mask,
                training_donors=[sample for sample in SAMPLES if sample != held_out_sample],
                latent_space_id=latent_space_id,
                **shared_decoder_options,
            )
            decoder_rotations.append(
                {
                    "held_out_sample": held_out_sample,
                    "scope": "decoder_distillation_only_shared_scanvi_backbone",
                    "path": str(rotation_path),
                    "sha256": _file_sha256(rotation_path),
                    "validation": _decoder_error(
                        rotated,
                        latent,
                        decoder_target,
                        held_out_mask,
                    ),
                }
            )
        validation_mask = _stratified_validation_mask(
            section_values,
            fraction=args.decoder_validation_fraction,
            seed=args.seed,
        )
        decoder_training_donors = SAMPLES
        validation_description = "donor_rotated_audit_plus_stratified_deployment_split"
    else:
        validation_mask = section_values == args.decoder_validation_sample
        decoder_training_donors = tuple(
            sample for sample in SAMPLES if sample != args.decoder_validation_sample
        )
        validation_description = "single_donor_sensitivity"

    _assert_input_files_unchanged(
        molecular_input_records,
        phase="immediately_before_deployed_decoder_output",
    )
    deployed_decoder = adapter.export_transferable_decoder_checkpoint(
        str(decoder_output),
        decoder_data,
        panel,
        validation_mask=validation_mask,
        training_donors=decoder_training_donors,
        latent_space_id=latent_space_id,
        **shared_decoder_options,
    )
    deployed_checkpoint = torch.load(decoder_output, map_location="cpu", weights_only=True)
    deployed_metadata = deployed_checkpoint["metadata"]
    if set(deployed_metadata.get("training_donors", ())) != set(decoder_training_donors):
        raise RuntimeError("distilled decoder training-donor provenance differs from its plan")
    held_out_decoder_evaluation = None
    if training_plan.held_out_sample is not None:
        if query_model is None or query_input is None or query_audit is None:
            raise RuntimeError("held-out decoder evaluation inputs are unavailable")
        # This target is computed only after decoder fitting has completed.  It
        # is used solely for the terminal held-out report and cannot influence
        # optimization or early stopping.
        adapter.model = query_model
        held_out_expression_target = adapter.normalized_expression(
            query_input,
            panel,
            transform_batch=transform_batch,
            batch_correction_mode=batch_correction_mode,
            posterior_samples=args.decoder_posterior_samples,
        )
        adapter.model = model
        held_out_sample = training_plan.held_out_sample
        held_out_decoder_evaluation = _decoder_error(
            deployed_decoder,
            latent_by_sample[held_out_sample],
            held_out_expression_target,
            np.ones(len(latent_by_sample[held_out_sample]), dtype=bool),
        )
    training_partition = _training_partition_provenance(training_plan)
    training_partition["label_ontology"] = list(training_label_ontology)
    training_partition["label_ontology_sha256"] = _ordered_string_sha256(training_label_ontology)
    training_partition["decoder_training_donors"] = list(decoder_training_donors)
    if query_audit is not None:
        held_out_mapping = training_partition["held_out_mapping"]
        if not isinstance(held_out_mapping, dict):
            raise RuntimeError("leave-one-donor-out provenance lacks query mapping")
        held_out_mapping["runtime_audit"] = query_audit
    training_partition["fit_cell_counts"] = {
        sample: int((model_sections == sample).sum()) for sample in training_plan.training_donors
    }
    payload = {
        "schema": "heir.snpatho_scanvi_r2.v1",
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "status": (
            "native_scanvi_true_leave_one_donor_out"
            if training_plan.uses_frozen_query
            else (
                "native_scanvi_specimen_batch_sensitivity"
                if args.molecular_design == "specimen_batch_sensitivity"
                else "native_scanvi_with_specimen_biology_preserved"
            )
        ),
        "analysis_role": (
            "leave_one_donor_out_molecular_audit"
            if training_plan.uses_frozen_query
            else "historical_all_donor_negative_control"
        ),
        "annotation_provenance": (
            "training-donor major_annotation copied from the published integrated-workflow "
            "object; held-out labels generated by the frozen training-donor-only scANVI "
            "classifier after target annotation removal"
            if training_plan.uses_frozen_query
            else "major_annotation copied from the published integrated-workflow object; "
            "not an independent clean reannotation"
        ),
        "workflow_filter": "processing_method == FFPE_snPATHO",
        "samples": list(SAMPLES),
        "training_partition": training_partition,
        "input_h5ad_sha256": input_hashes,
        "input_reference500_sha256": reference500_hashes,
        "molecular_input_files": [dict(record) for record in molecular_input_records],
        "native_model": str(native_model),
        "native_model_sha256": model_digest,
        "latent_space_id": latent_space_id,
        "latent_outputs": latent_outputs,
        "decoder": str(decoder_output),
        "decoder_sha256": _file_sha256(decoder_output),
        "expression_space_id": EXPRESSION_SPACE_ID,
        "expression_transform": SCVI_EXPRESSION_NORMALIZATION_CONTRACT,
        "gene_panel": str(panel_path),
        "decoder_contract": {
            "schema": deployed_metadata["schema"],
            "batch_correction_mode": deployed_metadata["batch_correction_mode"],
            "posterior_samples": deployed_metadata["posterior_samples"],
            "distillation_latent_sha256": deployed_metadata["distillation_latent_sha256"],
            "distillation_target_sha256": deployed_metadata["distillation_target_sha256"],
            "validation_mask_sha256": deployed_metadata["validation_mask_sha256"],
            "training_donors": list(deployed_metadata["training_donors"]),
        },
        "molecular_design": {
            "name": args.molecular_design,
            "model_batch_key": model_batch_key,
            "batch_correction_mode": batch_correction_mode,
            "transform_batch": list(() if transform_batch is None else transform_batch),
            "specimen_identity_is_biological": True,
            "technical_batch_key": args.technical_batch_key,
            "technical_batch_contingency": technical_batch_contingency,
            "technical_batch_query_contract": technical_batch_query_contract,
        },
        "decoder_validation": {
            "policy": validation_description,
            "single_donor_sample": (
                args.decoder_validation_sample
                if args.decoder_validation_policy == "single_donor_sensitivity"
                else None
            ),
            "stratified_fraction": (
                args.decoder_validation_fraction
                if validation_description
                in {
                    "donor_rotated_audit_plus_stratified_deployment_split",
                    "true_leave_one_donor_out_training_donor_stratified",
                }
                else None
            ),
            "deployment_validation": _decoder_error(
                deployed_decoder,
                decoder_latent,
                decoder_target,
                validation_mask,
            ),
            "held_out_evaluation": held_out_decoder_evaluation,
            "rotations": decoder_rotations,
            "rotation_limitation": (
                "not applicable; the backbone and decoder both exclude the held-out donor"
                if training_plan.uses_frozen_query
                else (
                    "decoder-only donor rotation; the shared scANVI backbone still saw all donors"
                    if decoder_rotations
                    else "single held-out donor sensitivity; no rotation audit"
                )
            ),
        },
        "gene_panel_sha256": str(panel_record["sha256"]),
        "scvi_tools_version": scvi.__version__,
        "latent_dim": args.latent_dim,
        "scvi_epochs": args.scvi_epochs,
        "scanvi_epochs": args.scanvi_epochs,
        "decoder_epochs": args.decoder_epochs,
        "decoder_posterior_samples": args.decoder_posterior_samples,
        "seed": args.seed,
        "cuda": True,
    }
    _assert_input_files_unchanged(
        molecular_input_records,
        phase="immediately_before_provenance_manifest_write",
    )
    _atomic_json(provenance_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
