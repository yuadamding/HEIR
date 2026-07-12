#!/usr/bin/env python3
"""Train a joint FFPE-snPATHO-only native scVI/scANVI molecular teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import numpy as np
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
) -> np.ndarray:
    """Select a deterministic validation subset from every specimen."""

    if not 0.0 < fraction < 0.5:
        raise ValueError("decoder validation fraction must lie in (0, 0.5)")
    values = np.asarray(section_values).astype(str)
    mask = np.zeros(len(values), dtype=bool)
    generator = np.random.default_rng(seed)
    for sample in SAMPLES:
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

    scvi.settings.seed = args.seed
    np.random.seed(args.seed)
    input_root = args.input_root.expanduser().resolve()
    native_model = args.native_model.expanduser().resolve()
    decoder_output = args.decoder_output.expanduser().resolve()
    latent_root = args.latent_output_root.expanduser().resolve()
    provenance_output = args.provenance_output.expanduser().resolve()
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
    if args.decoder_validation_policy == "donor_rotated":
        _stratified_validation_mask(
            np.concatenate([np.repeat(sample, 2) for sample in SAMPLES]),
            fraction=args.decoder_validation_fraction,
            seed=args.seed,
        )
    exact_outputs = (native_model, decoder_output, latent_root, provenance_output, rotation_root)
    if len(set(exact_outputs)) != len(exact_outputs):
        raise ValueError("molecular output paths must be distinct")
    for directory in (native_model, latent_root, rotation_root):
        if decoder_output.is_relative_to(directory) or directory.is_relative_to(decoder_output):
            raise ValueError("decoder output cannot contain or be contained by an output directory")
    output_directories = (native_model, latent_root, rotation_root)
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

    inputs = []
    input_hashes = {}
    for sample in SAMPLES:
        source = input_root / sample / "reference.h5ad"
        if not source.is_file():
            raise FileNotFoundError(source)
        values = ad.read_h5ad(source)
        if "major_annotation" not in values.obs or "processing_method" not in values.obs:
            raise ValueError("R1 AnnData lacks required annotation/workflow fields")
        if set(values.obs["processing_method"].astype(str)) != {"FFPE_snPATHO"}:
            raise ValueError("R1 AnnData includes a non-FFPE-snPATHO workflow")
        values.obs["section_id"] = sample
        values.obs["source_cell_id"] = values.obs_names.astype(str)
        values.obs["source_row"] = np.arange(values.n_obs, dtype=np.int64)
        inputs.append(values)
        input_hashes[sample] = _file_sha256(source)
    common_genes = tuple(str(value) for value in inputs[0].var_names)
    if any(
        tuple(str(value) for value in values.var_names) != common_genes for values in inputs[1:]
    ):
        raise ValueError("FFPE R1 references use different full gene orders")
    combined = ad.concat(inputs, join="inner", merge="same", index_unique="-")
    labels = combined.obs["major_annotation"].astype("category")
    if "unknown" not in labels.cat.categories:
        labels = labels.cat.add_categories(["unknown"])
    combined.obs["major_annotation"] = labels

    panel_path = args.gene_panel.expanduser().resolve()
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
    elif args.molecular_design == "technical_batch_only":
        model_batch_key = str(args.technical_batch_key or "")
        transform_batch = _technical_batch_contract(combined, args.technical_batch_key)
        batch_correction_mode = "reference_batch_marginalization"
        technical_batch_contingency = {
            sample: {
                level: int(
                    (
                        (combined.obs["section_id"].astype(str) == sample)
                        & (combined.obs[model_batch_key].astype(str) == level)
                    ).sum()
                )
                for level in transform_batch
            }
            for sample in SAMPLES
        }
    else:
        if args.technical_batch_key:
            raise ValueError(
                "--technical-batch-key cannot be combined with specimen_batch_sensitivity"
            )
        model_batch_key = "section_id"
        transform_batch = SAMPLES
        batch_correction_mode = "reference_batch_marginalization"
        technical_batch_contingency = None

    scvi.model.SCVI.setup_anndata(combined, batch_key=model_batch_key)
    base = scvi.model.SCVI(
        combined,
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
    native_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(native_model), overwrite=False, save_anndata=False)
    model_digest = _directory_sha256(native_model)
    latent_space_id = "sha256:" + model_digest
    latent = np.asarray(model.get_latent_representation(), dtype=np.float32)
    if latent.shape != (combined.n_obs, args.latent_dim) or not np.isfinite(latent).all():
        raise RuntimeError("scANVI returned an invalid latent representation")

    latent_outputs = {}
    section_values = combined.obs["section_id"].astype(str).to_numpy()
    source_ids = combined.obs["source_cell_id"].astype(str).to_numpy()
    for sample in SAMPLES:
        reference_path = input_root / sample / "reference500.npz"
        reference = RNAReference.load_npz(reference_path)
        selected = np.flatnonzero(section_values == sample)
        lookup = {source_ids[index]: latent[index] for index in selected}
        missing = [str(value) for value in reference.cell_ids if str(value) not in lookup]
        if missing or len(lookup) != len(reference.cell_ids):
            raise ValueError("scANVI/reference cell alignment failed for %s" % sample)
        aligned = np.stack([lookup[str(value)] for value in reference.cell_ids]).astype(np.float32)
        output = latent_root / sample / "reference500_scanvi.npz"
        replace(
            reference,
            latent=aligned,
            latent_space_id=latent_space_id,
            latent_training_donors=SAMPLES,
            latent_transform_sha256=model_digest,
        ).save_npz(output)
        latent_outputs[sample] = {
            "path": str(output),
            "sha256": _file_sha256(output),
            "cells": int(len(aligned)),
        }

    adapter = SCVIAdapter(latent_dim=args.latent_dim, likelihood="nb")
    adapter.model = model
    shared_decoder_options = {
        "transform_batch": transform_batch,
        "batch_correction_mode": batch_correction_mode,
        "posterior_samples": args.decoder_posterior_samples,
        "latent_target": latent,
        "max_epochs": args.decoder_epochs,
        "seed": args.seed,
        "device": "cuda",
    }
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    decoder_target = adapter.normalized_expression(
        combined,
        panel,
        transform_batch=transform_batch,
        batch_correction_mode=batch_correction_mode,
        posterior_samples=args.decoder_posterior_samples,
    )
    shared_decoder_options["expression_target"] = decoder_target
    decoder_rotations = []
    if args.decoder_validation_policy == "donor_rotated":
        rotation_root.mkdir(parents=True, exist_ok=False)
        for held_out_sample in SAMPLES:
            rotation_path = rotation_root / ("heldout_%s.pt" % held_out_sample)
            held_out_mask = section_values == held_out_sample
            rotated = adapter.export_transferable_decoder_checkpoint(
                str(rotation_path),
                combined,
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

    deployed_decoder = adapter.export_transferable_decoder_checkpoint(
        str(decoder_output),
        combined,
        panel,
        validation_mask=validation_mask,
        training_donors=decoder_training_donors,
        latent_space_id=latent_space_id,
        **shared_decoder_options,
    )
    deployed_checkpoint = torch.load(decoder_output, map_location="cpu", weights_only=True)
    deployed_metadata = deployed_checkpoint["metadata"]
    payload = {
        "schema": "heir.snpatho_scanvi_r2.v1",
        "status": (
            "native_scanvi_specimen_batch_sensitivity"
            if args.molecular_design == "specimen_batch_sensitivity"
            else "native_scanvi_with_specimen_biology_preserved"
        ),
        "annotation_provenance": (
            "major_annotation copied from the published integrated-workflow object; "
            "not an independent clean reannotation"
        ),
        "workflow_filter": "processing_method == FFPE_snPATHO",
        "samples": list(SAMPLES),
        "input_h5ad_sha256": input_hashes,
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
        },
        "molecular_design": {
            "name": args.molecular_design,
            "model_batch_key": model_batch_key,
            "batch_correction_mode": batch_correction_mode,
            "transform_batch": list(() if transform_batch is None else transform_batch),
            "specimen_identity_is_biological": True,
            "technical_batch_key": args.technical_batch_key,
            "technical_batch_contingency": technical_batch_contingency,
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
                if args.decoder_validation_policy == "donor_rotated"
                else None
            ),
            "deployment_validation": _decoder_error(
                deployed_decoder,
                latent,
                decoder_target,
                validation_mask,
            ),
            "rotations": decoder_rotations,
            "rotation_limitation": (
                "decoder-only donor rotation; the shared scANVI backbone still saw all donors"
                if decoder_rotations
                else "single held-out donor sensitivity; no rotation audit"
            ),
        },
        "gene_panel_sha256": _file_sha256(panel_path),
        "scvi_tools_version": scvi.__version__,
        "latent_dim": args.latent_dim,
        "scvi_epochs": args.scvi_epochs,
        "scanvi_epochs": args.scanvi_epochs,
        "decoder_epochs": args.decoder_epochs,
        "decoder_posterior_samples": args.decoder_posterior_samples,
        "seed": args.seed,
        "cuda": True,
    }
    _atomic_json(provenance_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
