#!/usr/bin/env python3
"""Train and score the donor-held-out MorphologyStateGate from frozen NPZ inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from heir.evaluation import evaluate_morphology_state_checkpoint
from heir.models import (
    RNAVAE,
    MorphologyStateGate,
    MorphologyStateGateConfig,
    fit_morphology_state_gate,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions, sha256_file


def _load_npz(path: Path, *, heldout: bool) -> Mapping[str, np.ndarray]:
    required = {"frozen_features", "latent_targets", "type_labels", "donor_ids"}
    if heldout:
        required.add("expression_targets")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("%s is missing: %s" % (path, ", ".join(missing)))
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    return result


def _type_names(
    train: Mapping[str, np.ndarray], heldout: Mapping[str, np.ndarray]
) -> Tuple[str, ...]:
    train_names = tuple(str(value) for value in train.get("type_names", np.asarray([])).tolist())
    heldout_names = tuple(
        str(value) for value in heldout.get("type_names", np.asarray([])).tolist()
    )
    if train_names and heldout_names and train_names != heldout_names:
        raise ValueError("training and held-out type ontologies differ")
    names = train_names or heldout_names
    labels = np.concatenate(
        (
            np.asarray(train["type_labels"], dtype=np.int64),
            np.asarray(heldout["type_labels"], dtype=np.int64),
        )
    )
    if labels.ndim != 1 or not len(labels) or np.any(labels < 0):
        raise ValueError("type_labels must be a non-empty non-negative integer vector")
    num_types = int(labels.max()) + 1
    if not names:
        names = tuple(str(index) for index in range(num_types))
    if len(names) != num_types:
        raise ValueError("type_names do not cover all type indices")
    return names


def _load_decoder(path: Path) -> Tuple[torch.nn.Module, Mapping[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError("decoder checkpoint is not a safe RNAVAE checkpoint") from error
    if not isinstance(payload, Mapping):
        raise ValueError("decoder checkpoint root must be a mapping")
    model = RNAVAE.from_checkpoint(payload)
    model.freeze_decoder(True)
    metadata = payload.get("metadata")
    return model.decoder.eval(), metadata if isinstance(metadata, Mapping) else {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--heldout-data", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--residual-rank", type=int, default=4)
    parser.add_argument("--residual-hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--minimum-within-type-r2", type=float, default=0.05)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-missing-wrong-donor-bank",
        action="store_true",
        help="development-only relaxation; the report records that the required control was waived",
    )
    args = parser.parse_args(argv)

    training_path = args.training_data.expanduser().resolve()
    heldout_path = args.heldout_data.expanduser().resolve()
    decoder_path = args.decoder_checkpoint.expanduser().resolve()
    checkpoint_path = args.checkpoint_output.expanduser().resolve()
    report_path = args.report_output.expanduser().resolve()
    inputs = (training_path, heldout_path, decoder_path)
    if len(set(inputs)) != len(inputs) or any(not path.is_file() for path in inputs):
        raise ValueError("training, held-out, and decoder inputs must be distinct files")
    reject_output_input_collisions(
        (checkpoint_path, report_path),
        inputs,
        label="MorphologyStateGate benchmark",
    )
    if checkpoint_path == report_path:
        raise ValueError("checkpoint and report outputs must be distinct")
    input_sha256 = {str(path): sha256_file(path) for path in inputs}

    train = _load_npz(training_path, heldout=False)
    heldout = _load_npz(heldout_path, heldout=True)
    names = _type_names(train, heldout)
    train_features = np.asarray(train["frozen_features"], dtype=np.float32)
    train_latent = np.asarray(train["latent_targets"], dtype=np.float32)
    train_labels = np.asarray(train["type_labels"], dtype=np.int64)
    train_donors = np.asarray(train["donor_ids"]).astype(str)
    heldout_donors = np.asarray(heldout["donor_ids"]).astype(str)
    overlap = sorted(set(train_donors.tolist()) & set(heldout_donors.tolist()))
    if overlap:
        raise ValueError("training and held-out donors overlap: %s" % ", ".join(overlap))
    if train_features.ndim != 2 or train_latent.ndim != 2:
        raise ValueError("training features and latent targets must be matrices")
    if len(train_features) != len(train_latent) or len(train_labels) != len(train_features):
        raise ValueError("training arrays are not row aligned")

    decoder, decoder_metadata = _load_decoder(decoder_path)
    decoder_training_donors = {str(value) for value in decoder_metadata.get("training_donors", ())}
    decoder_overlap = sorted(decoder_training_donors & set(heldout_donors.tolist()))
    if decoder_overlap:
        raise ValueError(
            "frozen decoder was trained on held-out donors: %s" % ", ".join(decoder_overlap)
        )
    config = MorphologyStateGateConfig(
        feature_dim=int(train_features.shape[1]),
        latent_dim=int(train_latent.shape[1]),
        num_types=len(names),
        residual_rank=args.residual_rank,
        residual_hidden_dim=args.residual_hidden_dim,
        type_names=names,
    )
    model = MorphologyStateGate.from_training_data(
        config,
        train_features,
        train_latent,
        train_labels,
        train_donors,
    )
    training_report = fit_morphology_state_gate(
        model,
        train_features,
        train_latent,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    model.save_checkpoint(checkpoint_path)
    gate_report = evaluate_morphology_state_checkpoint(
        checkpoint_path,
        heldout["frozen_features"],
        heldout["latent_targets"],
        heldout["type_labels"],
        heldout_donors,
        decoder=decoder,
        expression_targets=heldout["expression_targets"],
        roi_ids=heldout.get("roi_ids"),
        seed=args.seed,
        device=args.device,
        minimum_within_type_r2=args.minimum_within_type_r2,
        bootstrap_iterations=args.bootstrap_iterations,
        require_wrong_donor_banks=not args.allow_missing_wrong_donor_bank,
    )
    for path, digest in zip(inputs, input_sha256.values()):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("benchmark input changed during execution: %s" % path)
    report = {
        **gate_report,
        "training": training_report,
        "provenance": {
            "training_data": {
                "path": str(training_path),
                "sha256": input_sha256[str(training_path)],
            },
            "heldout_data": {"path": str(heldout_path), "sha256": input_sha256[str(heldout_path)]},
            "decoder_checkpoint": {
                "path": str(decoder_path),
                "sha256": input_sha256[str(decoder_path)],
                "training_donors": sorted(decoder_training_donors),
            },
            "checkpoint_output": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
        },
        "control_waivers": (
            ["wrong_donor_state_bank"] if args.allow_missing_wrong_donor_bank else []
        ),
    }
    atomic_json_dump(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
