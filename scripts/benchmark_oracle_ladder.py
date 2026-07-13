#!/usr/bin/env python3
"""Evaluate the HEIR oracle ladder from one pickle-free NPZ fixture."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from heir.evaluation import evaluate_oracle_ladder
from heir.utils import atomic_json_dump, reject_output_input_collisions

FIELDS = (
    "truth_expression",
    "truth_latent",
    "true_types",
    "decoder_expression",
    "type_mean_expression",
    "prototype_means",
    "prototype_expression",
    "prototype_types",
    "predicted_type_probabilities",
    "oracle_type_conditioned_heir_expression",
    "residual_disabled_heir_expression",
    "full_heir_expression",
    "cell_rna_mass",
)
IDENTITY_FIELDS = ("cell_ids", "gene_names", "spot_ids")
DECODER_CHECKPOINT_BINDING_FIELD = "decoder_checkpoint_sha256"
HEIR_CHECKPOINT_BINDING_FIELD = "heir_checkpoint_sha256"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        required=True,
        help="Decoder checkpoint whose hash is bound to the decoder-ceiling fixture.",
    )
    parser.add_argument(
        "--heir-checkpoint",
        type=Path,
        required=True,
        help="HEIR checkpoint used for both full and exact residual-disabled predictions.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    input_path = args.input.expanduser().resolve()
    decoder_checkpoint_path = args.decoder_checkpoint.expanduser().resolve()
    heir_checkpoint_path = args.heir_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reject_output_input_collisions(
        (output_path,),
        (input_path, decoder_checkpoint_path, heir_checkpoint_path),
        label="oracle ladder",
    )
    input_sha256 = _file_sha256(input_path)
    decoder_checkpoint_sha256 = _file_sha256(decoder_checkpoint_path)
    heir_checkpoint_sha256 = _file_sha256(heir_checkpoint_path)
    with np.load(input_path, allow_pickle=False) as archive:
        missing = sorted(
            (
                set(FIELDS)
                | set(IDENTITY_FIELDS)
                | {DECODER_CHECKPOINT_BINDING_FIELD, HEIR_CHECKPOINT_BINDING_FIELD}
            )
            - set(archive.files)
        )
        if missing:
            raise ValueError("oracle ladder input is missing: %s" % ", ".join(missing))
        payload = {name: np.array(archive[name], copy=True) for name in (*FIELDS, *IDENTITY_FIELDS)}
        declared_decoder_checkpoint = np.asarray(archive[DECODER_CHECKPOINT_BINDING_FIELD])
        declared_heir_checkpoint = np.asarray(archive[HEIR_CHECKPOINT_BINDING_FIELD])
        if declared_decoder_checkpoint.size != 1:
            raise ValueError("decoder_checkpoint_sha256 must be one scalar digest")
        if declared_heir_checkpoint.size != 1:
            raise ValueError("heir_checkpoint_sha256 must be one scalar digest")
        declared_decoder_checkpoint_sha256 = str(declared_decoder_checkpoint.reshape(-1)[0])
        declared_heir_checkpoint_sha256 = str(declared_heir_checkpoint.reshape(-1)[0])
    if _file_sha256(input_path) != input_sha256:
        raise ValueError("oracle input artifact changed while it was being loaded")
    if _file_sha256(decoder_checkpoint_path) != decoder_checkpoint_sha256:
        raise ValueError("decoder checkpoint changed while the oracle input was being loaded")
    if _file_sha256(heir_checkpoint_path) != heir_checkpoint_sha256:
        raise ValueError("HEIR checkpoint changed while the oracle input was being loaded")
    if declared_decoder_checkpoint_sha256 != decoder_checkpoint_sha256:
        raise ValueError(
            "oracle fixture decoder checkpoint hash differs from the supplied checkpoint"
        )
    if declared_heir_checkpoint_sha256 != heir_checkpoint_sha256:
        raise ValueError("oracle fixture HEIR checkpoint hash differs from the supplied checkpoint")
    report = dict(
        evaluate_oracle_ladder(
            **payload,
            input_artifact_sha256=input_sha256,
            decoder_checkpoint_sha256=decoder_checkpoint_sha256,
            heir_checkpoint_sha256=heir_checkpoint_sha256,
        )
    )
    provenance = dict(report["provenance"])
    provenance["checkpoint_file_verification"] = {
        "decoder": "fixture declaration matched the supplied decoder checkpoint file sha256",
        "heir": "fixture declaration matched the supplied HEIR checkpoint file sha256",
    }
    provenance["source_sha256"] = {
        "scripts.benchmark_oracle_ladder": _file_sha256(Path(__file__).resolve()),
        "heir.evaluation.oracle": _file_sha256(
            Path(__file__).resolve().parents[1] / "src" / "heir" / "evaluation" / "oracle.py"
        ),
    }
    report["provenance"] = provenance
    endpoints = dict(report["endpoints"])
    decoder_endpoint = dict(endpoints["rna_decoder_ceiling"])
    decoder_endpoint["checkpoint_file_verification"] = True
    endpoints["rna_decoder_ceiling"] = decoder_endpoint
    for endpoint_name in (
        "oracle_type_predicted_state",
        "full_heir_residual_disabled",
        "full_heir",
    ):
        endpoint = dict(endpoints[endpoint_name])
        endpoint["checkpoint_file_verification"] = True
        endpoints[endpoint_name] = endpoint
    report["endpoints"] = endpoints
    if _file_sha256(input_path) != input_sha256:
        raise ValueError("oracle input artifact changed during evaluation")
    if _file_sha256(decoder_checkpoint_path) != decoder_checkpoint_sha256:
        raise ValueError("decoder checkpoint changed during oracle evaluation")
    if _file_sha256(heir_checkpoint_path) != heir_checkpoint_sha256:
        raise ValueError("HEIR checkpoint changed during oracle evaluation")
    reject_output_input_collisions(
        (output_path,),
        (input_path, decoder_checkpoint_path, heir_checkpoint_path),
        label="oracle ladder",
    )
    atomic_json_dump(report, output_path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
