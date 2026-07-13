#!/usr/bin/env python3
"""Create a frozen, independently grounded molecular E-step artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from heir.data import HistologyBag, PrototypeSet, RNAReference
from heir.losses import unbalanced_sinkhorn
from heir.models import HEIRModel
from heir.training import (
    MolecularEStepArtifact,
    ValidatedInitializationReceipt,
    array_content_sha256,
    frozen_transport_telemetry,
    ordered_identity_sha256,
    validate_primary_claim_exclusions,
)
from heir.utils import atomic_json_dump, reject_output_input_collisions, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _freeze_inputs(paths: Sequence[Path]) -> Mapping[Path, str]:
    records = {}
    for path in paths:
        if not path.is_file():
            raise ValueError("molecular E-step input does not exist: %s" % path)
        if path in records:
            raise ValueError("molecular E-step inputs must be distinct: %s" % path)
        records[path] = _sha256(path)
    return records


def _assert_inputs_unchanged(records: Mapping[Path, str]) -> None:
    for path, expected in records.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("molecular E-step input changed during production: %s" % path)


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("teacher checkpoint must contain a mapping")
    return value


def _prototype_type_indices(
    prototype_labels: Sequence[object], type_names: Sequence[str]
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(type_names)}
    labels = tuple(str(value) for value in prototype_labels)
    missing = sorted(set(labels) - set(lookup))
    if missing:
        raise ValueError("prototype bank contains types outside the teacher ontology")
    return np.asarray([lookup[label] for label in labels], dtype=np.int64)


def _teacher_training_donors(metadata: Mapping[str, object]) -> Tuple[str, ...]:
    values = metadata.get("training_donors")
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("teacher checkpoint lacks training_donors provenance")
    result = tuple(sorted(set(str(value).strip() for value in values if str(value).strip())))
    if not result:
        raise ValueError("teacher checkpoint training_donors provenance is empty")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-receipt", type=Path, required=True)
    parser.add_argument("--histology", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--rna-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry-output", type=Path)
    parser.add_argument(
        "--teacher-role",
        choices=tuple(sorted(MolecularEStepArtifact.TRUSTED_TEACHER_ROLES)),
        default="generic_crossmodal_pretraining",
    )
    parser.add_argument("--target-donor")
    parser.add_argument("--type-cost-weight", type=float, default=1.0)
    parser.add_argument("--unknown-cost", type=float, default=1.0)
    parser.add_argument("--artifact-threshold", type=float, default=0.50)
    parser.add_argument("--fixed-unknown-mass", type=float, default=0.05)
    parser.add_argument("--uot-epsilon", type=float, default=0.1)
    parser.add_argument("--uot-marginal-relaxation", type=float, default=1.0)
    parser.add_argument("--uot-iterations", type=int, default=160)
    parser.add_argument("--uot-convergence-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--uot-maximum-marginal-residual", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.type_cost_weight < 0 or args.unknown_cost < 0:
        raise ValueError("type-cost-weight and unknown-cost must be non-negative")
    if not 0.0 <= args.artifact_threshold <= 1.0:
        raise ValueError("artifact-threshold must lie in [0, 1]")

    teacher_path = args.teacher_checkpoint.expanduser().resolve()
    receipt_path = args.initialization_receipt.expanduser().resolve()
    histology_path = args.histology.expanduser().resolve()
    prototypes_path = args.prototypes.expanduser().resolve()
    reference_path = args.rna_reference.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    telemetry_path = (
        None if args.telemetry_output is None else args.telemetry_output.expanduser().resolve()
    )
    input_records = _freeze_inputs(
        (teacher_path, receipt_path, histology_path, prototypes_path, reference_path)
    )
    reject_output_input_collisions(
        (output_path,) if telemetry_path is None else (output_path, telemetry_path),
        tuple(input_records),
        label="molecular E-step",
    )
    checkpoint = _load_checkpoint(teacher_path)
    model = HEIRModel.from_checkpoint(checkpoint)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("teacher checkpoint lacks ontology/provenance metadata")
    validate_primary_claim_exclusions(
        metadata,
        artifact="molecular E-step teacher checkpoint",
    )
    type_names = tuple(str(value) for value in metadata.get("type_names", ()))
    if not type_names or len(set(type_names)) != len(type_names):
        raise ValueError("teacher checkpoint type ontology is malformed")
    training_donors = _teacher_training_donors(metadata)
    feature_space_id = str(metadata.get("feature_space_id", ""))
    latent_space_id = str(metadata.get("latent_space_id", ""))
    if not feature_space_id or not latent_space_id:
        raise ValueError("teacher checkpoint lacks feature/latent-space provenance")

    histology = HistologyBag.load_npz(histology_path)
    prototypes = PrototypeSet.load_npz(prototypes_path)
    reference = RNAReference.load_npz(reference_path)
    target_donor = str(args.target_donor or histology.donor_id).strip()
    if not target_donor:
        raise ValueError("target donor must be explicit in the bag or command")
    if histology.donor_id and histology.donor_id != target_donor:
        raise ValueError("target donor differs from the HistologyBag")
    if not all(
        _is_sha256(value)
        for value in (
            histology.histology_source_sha256,
            histology.nuclei_source_sha256,
            histology.feature_source_sha256,
        )
    ):
        raise ValueError("trusted E-step requires complete HistologyBag source provenance")
    if prototypes.donor_id and prototypes.donor_id != target_donor:
        raise ValueError("matched prototype bank donor differs from the target donor")
    if not prototypes.block_id or prototypes.block_id != histology.block_id:
        raise ValueError("matched prototype bank block differs from the HistologyBag")
    if prototypes.source_reference_sha256 != input_records[reference_path]:
        raise ValueError("prototype bank is not hash-bound to the supplied RNA reference")
    if not prototypes.latent_training_donors or target_donor in set(
        prototypes.latent_training_donors
    ):
        raise ValueError("prototype latent mapping is not donor-held-out from the target")
    if target_donor in set(training_donors):
        raise ValueError("frozen teacher was trained on the target donor")
    if histology.feature_space_id != feature_space_id:
        raise ValueError("HistologyBag feature space differs from the frozen teacher")
    if prototypes.latent_space_id != latent_space_id:
        raise ValueError("prototype latent space differs from the frozen teacher")
    if reference.latent_space_id and reference.latent_space_id != latent_space_id:
        raise ValueError("RNA reference latent space differs from the frozen teacher")
    if model.config.morphology_dim != histology.features.shape[1]:
        raise ValueError("HistologyBag width differs from the frozen teacher")
    if model.config.latent_dim != prototypes.means.shape[1]:
        raise ValueError("prototype latent width differs from the frozen teacher")
    if model.config.num_cell_types != len(type_names):
        raise ValueError("teacher type width differs from its ontology metadata")
    prototype_types = _prototype_type_indices(prototypes.cell_type_labels, type_names)

    receipt = ValidatedInitializationReceipt.load_json(receipt_path)
    teacher_sha256 = input_records[teacher_path]
    receipt.validate_binding(
        checkpoint_sha256=teacher_sha256,
        feature_space_id=feature_space_id,
        latent_space_id=latent_space_id,
        type_names=type_names,
        target_donors=(target_donor,),
        receipt_path=receipt_path,
    )
    if set(receipt.training_donors) != set(training_donors):
        raise ValueError("initialization receipt training donors differ from teacher metadata")

    device = resolve_device(args.device)
    if device.type == "cuda":
        # This artifact is independently replayed on CPU. Keep CUDA execution
        # in IEEE float32 (not TF32 or autocast) so the serialized cost and
        # coupling remain within the strict cross-device replay tolerance.
        torch.set_float32_matmul_precision("highest")
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
    model = model.to(device=device, dtype=torch.float32).eval()
    morphology = torch.from_numpy(np.array(histology.features, dtype=np.float32, copy=True)).to(
        device
    )
    edge_index = torch.from_numpy(np.array(histology.edge_index, dtype=np.int64, copy=True)).to(
        device
    )
    edge_weight = torch.from_numpy(np.array(histology.edge_weight, dtype=np.float32, copy=True)).to(
        device
    )
    # The producer may use CUDA, but the frozen teacher path remains float32 so
    # a CPU verifier can tightly replay its costs and transport without
    # autocast-induced half-precision drift.
    with torch.inference_mode():
        _, type_probabilities, image_latent = model.encode_frozen_morphology(
            morphology,
            edge_index,
            edge_weight,
        )
    image_latent = image_latent.float()
    type_probabilities = type_probabilities.float()
    prototype_means = torch.from_numpy(np.array(prototypes.means, dtype=np.float32, copy=True)).to(
        device
    )
    prototype_variances = torch.from_numpy(
        np.array(prototypes.variances, dtype=np.float32, copy=True)
    ).to(device)
    prototype_type_tensor = torch.from_numpy(prototype_types).to(device)
    variance = prototype_variances.clamp_min(model.config.prototype_variance_floor)
    gaussian_cost = 0.5 * (
        (image_latent.unsqueeze(1) - prototype_means.unsqueeze(0)).square() / variance.unsqueeze(0)
        + variance.unsqueeze(0).log()
    ).mean(dim=2)
    type_cost = -type_probabilities.index_select(1, prototype_type_tensor).clamp_min(1.0e-8).log()
    cost = gaussian_cost + float(args.type_cost_weight) * type_cost
    source_mass_array = np.asarray(
        histology.segmentation_confidence * (1.0 - histology.artifact_probability),
        dtype=np.float32,
    )
    source_mass_array = np.array(source_mass_array, copy=True)
    source_mass_array[
        np.asarray(histology.artifact_probability) >= float(args.artifact_threshold)
    ] = 0.0
    if not np.any(source_mass_array > 0):
        raise ValueError("artifact threshold leaves no positive E/M source mass")
    source_mass = torch.from_numpy(source_mass_array).to(device)
    target_mass = torch.from_numpy(np.array(prototypes.weights, dtype=np.float32, copy=True)).to(
        device
    )
    with torch.inference_mode():
        transport = unbalanced_sinkhorn(
            cost,
            source_mass=source_mass,
            target_mass=target_mass,
            epsilon=float(args.uot_epsilon),
            marginal_relaxation=float(args.uot_marginal_relaxation),
            iterations=int(args.uot_iterations),
            convergence_tolerance=float(args.uot_convergence_tolerance),
            unknown_mass=float(args.fixed_unknown_mass),
            unknown_cost=float(args.unknown_cost),
            add_unknown=True,
        )
    if transport.converged is None or not bool(transport.converged.item()):
        raise ValueError("frozen molecular E-step transport did not converge")
    raw_plan = transport.plan.float()
    row_mass = raw_plan.sum(dim=1, keepdim=True)
    positive_source = source_mass > 0
    if bool((row_mass[positive_source] <= 0).any()):
        raise ValueError("frozen molecular E-step produced an empty positive-mass source row")
    normalized_plan = torch.zeros_like(raw_plan)
    normalized_plan[positive_source] = raw_plan[positive_source] / row_mass[positive_source]
    normalized_plan[~positive_source, -1] = 1.0
    normalized_source_mass = source_mass / source_mass.sum()
    realized_target = (normalized_plan * normalized_source_mass.unsqueeze(1)).sum(dim=0)
    normalized_target_mass = target_mass / target_mass.sum().clamp_min(1.0e-8)
    desired_target = torch.cat(
        (
            normalized_target_mass * (1.0 - float(args.fixed_unknown_mass)),
            normalized_target_mass.new_tensor([float(args.fixed_unknown_mass)]),
        )
    )
    source_marginal_residual = float((normalized_plan.sum(dim=1) - 1.0).abs().max().cpu())
    target_marginal_residual = float((realized_target - desired_target).abs().sum().cpu())
    source_dual_residual = float(transport.source_dual_residual.item())
    target_dual_residual = float(transport.target_dual_residual.item())
    full_transport_cost = torch.cat(
        (cost, cost.new_full((len(cost), 1), float(args.unknown_cost))), dim=1
    )

    morphology_array = np.asarray(histology.features, dtype=np.float32)
    means_array = np.asarray(prototypes.means, dtype=np.float32)
    variances_array = np.asarray(prototypes.variances, dtype=np.float32)
    weights_array = np.asarray(prototypes.weights, dtype=np.float32)
    normalized_plan_array = normalized_plan.cpu().numpy()
    raw_plan_array = raw_plan.cpu().numpy()
    transport_cost_array = full_transport_cost.float().cpu().numpy()
    solver_telemetry = frozen_transport_telemetry(
        raw_transport_plan=raw_plan_array,
        transport_cost=transport_cost_array,
        source_mass=source_mass_array,
        target_weights=weights_array,
        fixed_unknown_mass=float(args.fixed_unknown_mass),
        epsilon=float(args.uot_epsilon),
        marginal_relaxation=float(args.uot_marginal_relaxation),
    )
    artifact = MolecularEStepArtifact(
        transport_plan=normalized_plan_array,
        raw_transport_plan=raw_plan_array,
        transport_cost=transport_cost_array,
        source_mass=source_mass_array,
        nucleus_ids=tuple(str(value) for value in histology.nucleus_ids.tolist()),
        prototype_ids=tuple(str(value) for value in prototypes.prototype_ids.tolist()),
        source_artifacts=(str(histology_path), str(prototypes_path), str(reference_path)),
        source_sha256=(
            input_records[histology_path],
            input_records[prototypes_path],
            input_records[reference_path],
        ),
        source_roles=("histology", "prototype_bank", "rna_reference"),
        teacher_checkpoint=str(teacher_path),
        teacher_checkpoint_sha256=teacher_sha256,
        initialization_receipt=str(receipt_path),
        initialization_receipt_sha256=input_records[receipt_path],
        teacher_role=str(args.teacher_role),
        teacher_training_donors=training_donors,
        target_donor=target_donor,
        feature_space_id=feature_space_id,
        latent_space_id=latent_space_id,
        type_ontology_sha256=ordered_identity_sha256(type_names),
        morphology_sha256=array_content_sha256(morphology_array),
        prototype_means_sha256=array_content_sha256(means_array),
        prototype_variances_sha256=array_content_sha256(variances_array),
        prototype_types_sha256=array_content_sha256(prototype_types),
        prototype_weights_sha256=array_content_sha256(weights_array),
        image_latent_sha256=array_content_sha256(image_latent.cpu().numpy()),
        type_probabilities_sha256=array_content_sha256(type_probabilities.cpu().numpy()),
        transport_cost_sha256=array_content_sha256(transport_cost_array),
        source_mass_sha256=array_content_sha256(source_mass_array),
        artifact_threshold=float(args.artifact_threshold),
        type_cost_weight=float(args.type_cost_weight),
        unknown_cost=float(args.unknown_cost),
        fixed_unknown_mass=float(args.fixed_unknown_mass),
        uot_epsilon=float(args.uot_epsilon),
        uot_marginal_relaxation=float(args.uot_marginal_relaxation),
        uot_iterations=int(args.uot_iterations),
        uot_iterations_run=int(transport.iterations_run),
        uot_convergence_tolerance=float(args.uot_convergence_tolerance),
        uot_maximum_marginal_residual=float(args.uot_maximum_marginal_residual),
        converged=True,
        source_marginal_residual=source_marginal_residual,
        target_marginal_residual=target_marginal_residual,
        solver_source_marginal_error=solver_telemetry["solver_source_marginal_error"],
        solver_target_marginal_error=solver_telemetry["solver_target_marginal_error"],
        source_dual_residual=source_dual_residual,
        target_dual_residual=target_dual_residual,
        transport_objective=solver_telemetry["transport_objective"],
        e_step_round=0,
    )
    _assert_inputs_unchanged(input_records)
    artifact.save_npz(output_path)
    telemetry = {
        "schema": "heir.molecular_e_step_telemetry.v1",
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "teacher_checkpoint_sha256": teacher_sha256,
        "initialization_receipt_sha256": input_records[receipt_path],
        "target_donor": target_donor,
        "cells": len(histology.nucleus_ids),
        "prototypes": len(prototypes.prototype_ids),
        "device": str(device),
        "converged": True,
        "iterations": int(transport.iterations_run),
        "source_dual_residual": source_dual_residual,
        "target_dual_residual": target_dual_residual,
        "source_marginal_residual": source_marginal_residual,
        "target_marginal_residual": target_marginal_residual,
        "solver_source_marginal_error": solver_telemetry["solver_source_marginal_error"],
        "solver_target_marginal_error": solver_telemetry["solver_target_marginal_error"],
        "artifact_threshold": float(args.artifact_threshold),
        "type_cost_weight": float(args.type_cost_weight),
        "unknown_cost": float(args.unknown_cost),
        "realized_unknown_mass": float(realized_target[-1].cpu()),
        "fixed_unknown_mass": float(args.fixed_unknown_mass),
    }
    if telemetry_path is not None:
        _assert_inputs_unchanged(input_records)
        atomic_json_dump(telemetry, telemetry_path)
    print(json.dumps(telemetry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
