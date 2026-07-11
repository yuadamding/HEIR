#!/usr/bin/env python3
"""Train HEIR's frozen RNA decoder against a manifest-bound latent transform.

This is the lightweight, auditable fallback used when scVI is unavailable.  It
does not learn a new latent space: the input RNAReference must already contain
latents produced by the frozen development transform.  Only the decoder is
optimized, and the exported checkpoint is marked ``decoder_only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy import sparse
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from heir.data import RNAReference
from heir.evaluation import expression_metrics
from heir.expression import EXPRESSION_SPACE_ID, EXPRESSION_TARGET_SUM
from heir.models.rna import RNAVAE, RNAVAEConfig
from heir.utils import resolve_device, set_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_panel(reference: RNAReference) -> np.ndarray:
    scale = EXPRESSION_TARGET_SUM / np.maximum(reference.library_sizes, 1.0)
    values = sparse.diags(scale).dot(reference.counts).tocsr().astype(np.float32)
    values.data = np.log1p(values.data)
    return values.toarray().astype(np.float32, copy=False)


def _stratified_split(labels: np.ndarray, validation_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    validation = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(validation_fraction * len(indices))))
        validation[indices[:count]] = True
    if validation.all():
        raise ValueError("validation split consumed every cell")
    return validation


def _atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--latent-transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-donor", action="append", required=True)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if not 0.0 < args.validation_fraction < 1.0 or args.patience <= 0:
        raise ValueError("validation-fraction and patience are invalid")
    reference_path = args.reference.expanduser().resolve()
    transform_path = args.latent_transform.expanduser().resolve()
    reference = RNAReference.load_npz(reference_path)
    if reference.latent.shape[1] == 0 or not reference.latent_space_id:
        raise ValueError("reference must contain a frozen, identified latent representation")
    transform_sha256 = _sha256(transform_path)
    expected_space = "sha256:" + transform_sha256
    if reference.latent_space_id != expected_space:
        raise ValueError("reference latent identity differs from --latent-transform")

    expression = _normalized_panel(reference)
    validation_mask = _stratified_split(
        reference.cell_type_labels,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    model = RNAVAE(
        RNAVAEConfig(
            input_dim=reference.shape[1],
            latent_dim=reference.latent.shape[1],
            hidden_dims=(128,),
            decoder_hidden_dims=(128, 256),
            dropout=0.05,
            nonnegative_output=True,
        )
    ).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)

    train_latent = torch.from_numpy(reference.latent[~validation_mask])
    train_expression = torch.from_numpy(expression[~validation_mask])
    valid_latent = torch.from_numpy(reference.latent[validation_mask]).to(device)
    valid_expression = torch.from_numpy(expression[validation_mask]).to(device)
    loader = DataLoader(
        TensorDataset(train_latent, train_expression),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=use_amp,
    )
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for latent, target in loader:
            latent = latent.to(device, non_blocking=use_amp)
            target = target.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = model.decode(latent)
                loss = F.huber_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(latent)
            count += len(latent)
        model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
            validation_prediction = model.decode(valid_latent)
            validation_loss = float(F.huber_loss(validation_prediction, valid_expression))
        training_loss = total / max(count, 1)
        history.append(
            {
                "epoch": epoch,
                "train_huber": training_loss,
                "validation_huber": validation_loss,
            }
        )
        if validation_loss < best_loss - 1.0e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("decoder training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        held_out_prediction = model.decode(valid_latent).cpu().numpy()
    elapsed = time.perf_counter() - start
    checkpoint = model.checkpoint()
    checkpoint["metadata"] = {
        "decoder_only": True,
        "gene_names": reference.gene_ids.tolist(),
        "training_donors": sorted(set(str(value) for value in args.training_donor)),
        "latent_space_id": reference.latent_space_id,
        "expression_space_id": EXPRESSION_SPACE_ID,
        "source_reference_sha256": _sha256(reference_path),
        "latent_transform_sha256": transform_sha256,
        "seed": args.seed,
    }
    output_path = args.output.expanduser().resolve()
    _atomic_torch_save(checkpoint, output_path)
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_huber": best_loss,
        "held_out_expression": expression_metrics(
            held_out_prediction,
            expression[validation_mask],
        ),
        "train_cells": int((~validation_mask).sum()),
        "validation_cells": int(validation_mask.sum()),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "mixed_precision": use_amp,
        "peak_cuda_memory_bytes": (int(torch.cuda.max_memory_allocated(device)) if use_amp else 0),
        "checkpoint_sha256": _sha256(output_path),
        "history": history,
    }
    metrics_path = args.metrics.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
