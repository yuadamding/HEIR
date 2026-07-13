#!/usr/bin/env python3
"""Build auditable scale-held-out refinement views from a trained HEIR model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from heir.models import HEIRModel
from heir.training import HEIRTrainingBatch
from heir.utils import reject_output_input_collisions, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_identity_sha256(values: Sequence[object]) -> str:
    """Hash an ordered ontology without relying on NumPy byte layout."""

    encoded = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".npz.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--shared-tail-features", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    batch_path = args.batch.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reject_output_input_collisions(
        output_paths=(output_path,),
        input_paths=(checkpoint_path, batch_path),
        label="refinement-view artifact",
    )
    checkpoint_hash = _sha256(checkpoint_path)
    batch_hash = _sha256(batch_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = HEIRModel.from_checkpoint(checkpoint).eval()
    batch = HEIRTrainingBatch.load_npz(batch_path)
    bound_input_paths = (checkpoint_path, batch_path, *batch.source_artifacts)
    reject_output_input_collisions(
        output_paths=(output_path,),
        input_paths=bound_input_paths,
        label="refinement-view artifact",
    )
    if not batch.nucleus_ids:
        raise ValueError("refinement views require stable nucleus IDs")
    if not batch.source_sha256:
        raise ValueError("refinement views require source SHA-256 provenance in the batch")
    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("refinement views require checkpoint ontology/provenance metadata")
    checkpoint_types = tuple(str(value) for value in checkpoint_metadata.get("type_names", ()))
    if checkpoint_types != batch.type_names:
        raise ValueError("checkpoint and refinement-view batch use different cell types")
    if str(checkpoint_metadata.get("feature_space_id", "")) != batch.feature_space_id:
        raise ValueError("checkpoint and refinement-view batch use different feature spaces")
    if str(checkpoint_metadata.get("latent_space_id", "")) != batch.latent_space_id:
        raise ValueError("checkpoint and refinement-view batch use different latent spaces")
    width = int(batch.morphology.shape[1])
    tail = int(args.shared_tail_features)
    blocks = int(args.encoder_blocks)
    if blocks < 2 or tail < 0 or tail >= width or (width - tail) % blocks:
        raise ValueError("encoder blocks must evenly partition non-tail morphology features")
    block_width = (width - tail) // blocks
    device = resolve_device(args.device)
    model.to(device)
    values = batch.to(device)
    probabilities = []
    source_hashes = []
    for block_index in range(blocks):
        morphology = values.morphology.clone()
        morphology[:, : width - tail] = 0
        start = block_index * block_width
        stop = start + block_width
        morphology[:, start:stop] = values.morphology[:, start:stop]
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ),
        ):
            output = model(
                morphology,
                values.edge_index,
                values.edge_weight,
                prototype_means=values.prototype_means,
                prototype_variances=values.prototype_variances,
                prototype_types=values.prototype_types,
                prototype_weights=values.prototype_weights,
                prototype_mask=values.prototype_mask,
                sample_latent=False,
            )
        probabilities.append(output.type_probabilities.float().cpu().numpy())
        digest = hashlib.sha256()
        digest.update(checkpoint_hash.encode("ascii"))
        digest.update(batch_hash.encode("ascii"))
        digest.update(("encoder_block_%d:%d:%d" % (block_index, start, stop)).encode("ascii"))
        source_hashes.append(digest.hexdigest())

    view_array = np.stack(probabilities).astype(np.float32)
    if any(
        np.array_equal(view_array[left], view_array[right])
        for left in range(blocks)
        for right in range(left + 1, blocks)
    ):
        raise ValueError("scale-held-out views produced identical predictions")
    metadata = {
        "schema": "heir.refinement_views.v2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "batch": str(batch_path),
        "batch_sha256": batch_hash,
        "batch_contract": batch.CONTRACT,
        "batch_contract_version": batch.CONTRACT_VERSION,
        "batch_source_sha256": list(batch.source_sha256),
        "batch_source_roles": list(batch.source_roles),
        "sample_id": batch.sample_id,
        "donor_id": batch.donor_id,
        "bag_id": batch.bag_id,
        "block_id": batch.block_id,
        "feature_space_id": batch.feature_space_id,
        "latent_space_id": batch.latent_space_id,
        "type_names": list(batch.type_names),
        "type_ontology_sha256": _ordered_identity_sha256(batch.type_names),
        "view_construction": "one_encoder_scale_block_plus_shared_explicit_morphology",
        "encoder_blocks": blocks,
        "encoder_block_width": block_width,
        "shared_tail_features": tail,
        "device": str(device),
    }
    if _sha256(checkpoint_path) != checkpoint_hash or _sha256(batch_path) != batch_hash:
        raise RuntimeError(
            "refinement-view checkpoint or batch changed during inference; discard the result"
        )
    reject_output_input_collisions(
        output_paths=(output_path,),
        input_paths=bound_input_paths,
        label="refinement-view artifact",
    )
    _atomic_npz(
        output_path,
        {
            "nucleus_ids": np.asarray(batch.nucleus_ids, dtype=np.str_),
            "view_predictions": view_array,
            "view_ids": np.asarray(
                ["encoder_scale_%d" % index for index in range(blocks)], dtype=np.str_
            ),
            "view_source_sha256": np.asarray(source_hashes, dtype=np.str_),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        },
    )
    print(json.dumps({**metadata, "output": str(output_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
