#!/usr/bin/env python3
"""Train a joint FFPE-snPATHO-only native scVI/scANVI molecular teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import scvi

from heir.data import RNAReference
from heir.prior import SCVIAdapter

SAMPLES = ("4066", "4399", "4411")


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
        default=repository.parent / "HEIR_assets" / "pretrained" / "snpatho_scanvi_r1_v1",
    )
    parser.add_argument(
        "--decoder-output",
        type=Path,
        default=repository.parent
        / "HEIR_assets"
        / "pretrained"
        / "snpatho_scanvi_r1_v1_decoder.pt",
    )
    parser.add_argument(
        "--latent-output-root",
        type=Path,
        default=repository / "artifacts" / "snpatho" / "r1_scanvi",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=repository / "artifacts" / "snpatho" / "r1_scanvi" / "provenance.json",
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    scvi.settings.seed = args.seed
    np.random.seed(args.seed)
    input_root = args.input_root.expanduser().resolve()
    native_model = args.native_model.expanduser().resolve()
    decoder_output = args.decoder_output.expanduser().resolve()
    latent_root = args.latent_output_root.expanduser().resolve()
    provenance_output = args.provenance_output.expanduser().resolve()
    if native_model.exists():
        if not args.overwrite:
            raise FileExistsError("native scANVI checkpoint already exists: %s" % native_model)
        shutil.rmtree(native_model)
    if decoder_output.exists() and not args.overwrite:
        raise FileExistsError("distilled decoder already exists: %s" % decoder_output)

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

    scvi.model.SCVI.setup_anndata(combined, batch_key="section_id")
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
        replace(reference, latent=aligned, latent_space_id=latent_space_id).save_npz(output)
        latent_outputs[sample] = {
            "path": str(output),
            "sha256": _file_sha256(output),
            "cells": int(len(aligned)),
        }

    panel = _gene_panel(args.gene_panel.expanduser().resolve())
    missing_panel = sorted(set(panel) - set(common_genes))
    if missing_panel:
        raise ValueError(
            "scANVI full reference is missing panel genes: %s" % ", ".join(missing_panel)
        )
    adapter = SCVIAdapter(latent_dim=args.latent_dim, likelihood="nb")
    adapter.model = model
    adapter.export_transferable_decoder_checkpoint(
        str(decoder_output),
        combined,
        panel,
        validation_mask=section_values == "4411",
        training_donors=SAMPLES,
        latent_space_id=latent_space_id,
        transform_batch=SAMPLES,
        max_epochs=args.decoder_epochs,
        seed=args.seed,
        device="cuda",
    )
    payload = {
        "schema": "heir.snpatho_scanvi_r1.v1",
        "status": "native_scanvi_with_published_integrated_annotation_sensitivity",
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
        "gene_panel_sha256": _file_sha256(args.gene_panel.expanduser().resolve()),
        "scvi_tools_version": scvi.__version__,
        "latent_dim": args.latent_dim,
        "scvi_epochs": args.scvi_epochs,
        "scanvi_epochs": args.scanvi_epochs,
        "decoder_epochs": args.decoder_epochs,
        "seed": args.seed,
        "cuda": True,
    }
    _atomic_json(provenance_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
