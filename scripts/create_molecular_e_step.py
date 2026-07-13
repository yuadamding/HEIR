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
    prototype_target_mass,
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


def _load_numeric_npz(path: Path, names: Sequence[str]) -> Tuple[np.ndarray, ...]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(names) - set(archive.files))
            if missing:
                raise ValueError("%s is missing: %s" % (path, ", ".join(missing)))
            arrays = tuple(np.array(archive[name], dtype=np.float32, copy=True) for name in names)
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and " is missing: " in str(error):
            raise
        raise ValueError(
            "optional E-step input is not a numeric NPZ artifact: %s" % path
        ) from error
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("optional E-step input contains non-finite values: %s" % path)
    return arrays


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-receipt", type=Path, required=True)
    parser.add_argument("--histology", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--rna-reference", type=Path, required=True)
    parser.add_argument(
        "--image-latent-uncertainty",
        type=Path,
        help="NPZ containing non-negative image_latent_variance (cells by latent dimensions)",
    )
    parser.add_argument(
        "--latent-whitening-artifact",
        type=Path,
        help="training-only NPZ containing latent_mean and positive latent_scale vectors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry-output", type=Path)
    parser.add_argument(
        "--teacher-role",
        choices=tuple(sorted(MolecularEStepArtifact.TRUSTED_TEACHER_ROLES)),
        help="compatibility assertion; the authoritative role comes from checkpoint metadata",
    )
    parser.add_argument("--target-donor")
    parser.add_argument("--type-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--type-compatibility-mode",
        choices=tuple(sorted(MolecularEStepArtifact.TYPE_COMPATIBILITY_MODES)),
        default="hard_broad_mask",
        help=("soft type penalty (legacy) or a hard mask to the teacher-selected broad type"),
    )
    parser.add_argument(
        "--prototype-mass-mode",
        choices=tuple(sorted(MolecularEStepArtifact.PROTOTYPE_MASS_MODES)),
        default="uniform_within_type",
        help=(
            "sample-matched reference weights (legacy) or broad/fine type mass distributed "
            "uniformly among within-type states"
        ),
    )
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
    uncertainty_path = (
        None
        if args.image_latent_uncertainty is None
        else args.image_latent_uncertainty.expanduser().resolve()
    )
    whitening_path = (
        None
        if args.latent_whitening_artifact is None
        else args.latent_whitening_artifact.expanduser().resolve()
    )
    output_path = args.output.expanduser().resolve()
    telemetry_path = (
        None if args.telemetry_output is None else args.telemetry_output.expanduser().resolve()
    )
    input_paths = [teacher_path, receipt_path, histology_path, prototypes_path, reference_path]
    input_paths.extend(path for path in (uncertainty_path, whitening_path) if path is not None)
    input_records = _freeze_inputs(tuple(input_paths))
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
    checkpoint_teacher_role = str(metadata.get("teacher_role", "")).strip()
    if checkpoint_teacher_role not in MolecularEStepArtifact.TRUSTED_TEACHER_ROLES:
        raise ValueError("teacher checkpoint lacks a trusted teacher_role")
    if args.teacher_role is not None and args.teacher_role != checkpoint_teacher_role:
        raise ValueError("--teacher-role differs from authoritative checkpoint metadata")
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
    image_latent_variance_array = np.empty((0, 0), dtype=np.float32)
    if uncertainty_path is not None:
        (image_latent_variance_array,) = _load_numeric_npz(
            uncertainty_path, ("image_latent_variance",)
        )
        if image_latent_variance_array.shape != tuple(image_latent.shape):
            raise ValueError("image_latent_variance dimensions differ from the teacher output")
        if np.any(image_latent_variance_array < 0):
            raise ValueError("image_latent_variance must be non-negative")
    latent_whitening_mean = np.empty(0, dtype=np.float32)
    latent_whitening_scale = np.empty(0, dtype=np.float32)
    if whitening_path is not None:
        latent_whitening_mean, latent_whitening_scale = _load_numeric_npz(
            whitening_path, ("latent_mean", "latent_scale")
        )
        if (
            latent_whitening_mean.shape != (model.config.latent_dim,)
            or latent_whitening_scale.shape != latent_whitening_mean.shape
            or np.any(latent_whitening_scale <= 0)
        ):
            raise ValueError("latent whitening mean/scale dimensions or values are invalid")
    cost_image_latent = image_latent
    cost_prototype_means = prototype_means
    cost_prototype_variances = prototype_variances
    image_latent_variance = (
        None
        if not image_latent_variance_array.size
        else torch.from_numpy(image_latent_variance_array).to(device)
    )
    if latent_whitening_mean.size:
        whitening_mean = torch.from_numpy(latent_whitening_mean).to(device)
        whitening_scale = torch.from_numpy(latent_whitening_scale).to(device)
        cost_image_latent = (cost_image_latent - whitening_mean) / whitening_scale
        cost_prototype_means = (cost_prototype_means - whitening_mean) / whitening_scale
        cost_prototype_variances = cost_prototype_variances / whitening_scale.square()
        if image_latent_variance is not None:
            image_latent_variance = image_latent_variance / whitening_scale.square()
    variance = cost_prototype_variances.clamp_min(model.config.prototype_variance_floor)
    total_variance = variance.unsqueeze(0)
    if image_latent_variance is not None:
        total_variance = total_variance + image_latent_variance.unsqueeze(1)
    gaussian_cost = 0.5 * (
        (cost_image_latent.unsqueeze(1) - cost_prototype_means.unsqueeze(0)).square()
        / total_variance
        + total_variance.log()
    ).mean(dim=2)
    type_cost = -type_probabilities.index_select(1, prototype_type_tensor).clamp_min(1.0e-8).log()
    pair_mask = None
    if args.type_compatibility_mode == "hard_broad_mask":
        if model.config.fine_to_parent is None:
            selected_broad_type = type_probabilities.argmax(dim=1)
            prototype_broad_type = prototype_type_tensor
        else:
            fine_to_parent = torch.as_tensor(
                model.config.fine_to_parent,
                dtype=torch.long,
                device=device,
            )
            parent_probabilities = type_probabilities.new_zeros(
                (len(type_probabilities), model.config.num_parent_types)
            )
            parent_probabilities = parent_probabilities.index_add(
                1,
                fine_to_parent,
                type_probabilities,
            )
            selected_broad_type = parent_probabilities.argmax(dim=1)
            prototype_broad_type = fine_to_parent.index_select(0, prototype_type_tensor)
        pair_mask = selected_broad_type.unsqueeze(1) == prototype_broad_type.unsqueeze(0)
        if not bool(pair_mask.any(dim=1).all()):
            raise ValueError("hard broad compatibility leaves a nucleus without prototypes")
        cost = gaussian_cost
    else:
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
    target_mass_array = prototype_target_mass(
        prototypes.weights,
        prototype_types,
        mode=str(args.prototype_mass_mode),
        fine_to_parent=model.config.fine_to_parent,
    )
    target_mass = torch.from_numpy(target_mass_array).to(device)
    with torch.inference_mode():
        transport = unbalanced_sinkhorn(
            cost,
            source_mass=source_mass,
            target_mass=target_mass,
            pair_mask=pair_mask,
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
    raw_real_row_mass = raw_plan[:, :-1].sum(dim=1)
    raw_dustbin_row_mass = raw_plan[:, -1]
    conditional_known = torch.zeros_like(raw_plan[:, :-1])
    positive_real = raw_real_row_mass > 0
    conditional_known[positive_real] = raw_plan[positive_real, :-1] / raw_real_row_mass[
        positive_real
    ].unsqueeze(1)
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
    gaussian_cost_array = gaussian_cost.float().cpu().numpy()
    type_cost_array = type_cost.float().cpu().numpy()
    known_cost_array = cost.float().cpu().numpy()
    compatibility_array = (
        np.ones_like(known_cost_array, dtype=bool)
        if pair_mask is None
        else pair_mask.cpu().numpy().astype(bool, copy=False)
    )
    second_best_gaps = []
    for row_cost, row_compatible in zip(known_cost_array, compatibility_array):
        admissible = np.sort(row_cost[row_compatible].astype(np.float64, copy=False))
        if len(admissible) >= 2:
            second_best_gaps.append(float(admissible[1] - admissible[0]))
    solver_telemetry = frozen_transport_telemetry(
        raw_transport_plan=raw_plan_array,
        transport_cost=transport_cost_array,
        source_mass=source_mass_array,
        target_weights=target_mass_array,
        fixed_unknown_mass=float(args.fixed_unknown_mass),
        epsilon=float(args.uot_epsilon),
        marginal_relaxation=float(args.uot_marginal_relaxation),
    )
    source_artifacts = [str(histology_path), str(prototypes_path), str(reference_path)]
    source_sha256 = [
        input_records[histology_path],
        input_records[prototypes_path],
        input_records[reference_path],
    ]
    source_roles = ["histology", "prototype_bank", "rna_reference"]
    if uncertainty_path is not None:
        source_artifacts.append(str(uncertainty_path))
        source_sha256.append(input_records[uncertainty_path])
        source_roles.append("image_latent_uncertainty")
    if whitening_path is not None:
        source_artifacts.append(str(whitening_path))
        source_sha256.append(input_records[whitening_path])
        source_roles.append("latent_whitening")
    artifact = MolecularEStepArtifact(
        transport_plan=normalized_plan_array,
        raw_transport_plan=raw_plan_array,
        transport_cost=transport_cost_array,
        source_mass=source_mass_array,
        nucleus_ids=tuple(str(value) for value in histology.nucleus_ids.tolist()),
        prototype_ids=tuple(str(value) for value in prototypes.prototype_ids.tolist()),
        source_artifacts=tuple(source_artifacts),
        source_sha256=tuple(source_sha256),
        source_roles=tuple(source_roles),
        teacher_checkpoint=str(teacher_path),
        teacher_checkpoint_sha256=teacher_sha256,
        initialization_receipt=str(receipt_path),
        initialization_receipt_sha256=input_records[receipt_path],
        teacher_role=checkpoint_teacher_role,
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
        raw_real_row_mass=raw_real_row_mass.cpu().numpy(),
        raw_dustbin_row_mass=raw_dustbin_row_mass.cpu().numpy(),
        conditional_known_prototype_distribution=conditional_known.cpu().numpy(),
        type_compatibility_mode=str(args.type_compatibility_mode),
        prototype_mass_mode=str(args.prototype_mass_mode),
        image_latent_variance=image_latent_variance_array,
        latent_whitening_mean=latent_whitening_mean,
        latent_whitening_scale=latent_whitening_scale,
    )
    _assert_inputs_unchanged(input_records)
    artifact.save_npz(output_path)
    telemetry = {
        "schema": "heir.molecular_e_step_telemetry.v2",
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
        "conditional_row_normalization_residual": source_marginal_residual,
        "source_marginal_residual_legacy_alias": source_marginal_residual,
        "target_marginal_residual": target_marginal_residual,
        "solver_source_marginal_error": solver_telemetry["solver_source_marginal_error"],
        "solver_target_marginal_error": solver_telemetry["solver_target_marginal_error"],
        "artifact_threshold": float(args.artifact_threshold),
        "type_cost_weight": float(args.type_cost_weight),
        "type_compatibility_mode": str(args.type_compatibility_mode),
        "prototype_mass_mode": str(args.prototype_mass_mode),
        "image_latent_uncertainty": (
            None
            if uncertainty_path is None
            else {"path": str(uncertainty_path), "sha256": input_records[uncertainty_path]}
        ),
        "latent_whitening_artifact": (
            None
            if whitening_path is None
            else {"path": str(whitening_path), "sha256": input_records[whitening_path]}
        ),
        "target_mass": {
            "minimum": float(target_mass_array.min()),
            "maximum": float(target_mass_array.max()),
            "entropy": float(
                -(
                    target_mass_array.astype(np.float64)
                    * np.log(np.maximum(target_mass_array.astype(np.float64), 1.0e-12))
                ).sum()
            ),
        },
        "unknown_cost": float(args.unknown_cost),
        "cost_distributions": {
            "gaussian_cost_median": float(np.median(gaussian_cost_array)),
            "gaussian_cost_p95": float(np.quantile(gaussian_cost_array, 0.95)),
            "type_cost_median": float(np.median(type_cost_array)),
            "type_cost_p95": float(np.quantile(type_cost_array, 0.95)),
            "cost_gap_to_second_best_median": (
                None if not second_best_gaps else float(np.median(second_best_gaps))
            ),
            "cost_gap_to_second_best_supported_cells": len(second_best_gaps),
        },
        "realized_unknown_mass": float(realized_target[-1].cpu()),
        "fixed_unknown_mass": float(args.fixed_unknown_mass),
        "raw_real_row_mass": {
            "total": float(raw_real_row_mass.sum().cpu()),
            "median": float(raw_real_row_mass.median().cpu()),
            "p05": float(torch.quantile(raw_real_row_mass, 0.05).cpu()),
            "p95": float(torch.quantile(raw_real_row_mass, 0.95).cpu()),
        },
        "raw_dustbin_row_mass_total": float(raw_dustbin_row_mass.sum().cpu()),
    }
    if telemetry_path is not None:
        _assert_inputs_unchanged(input_records)
        atomic_json_dump(telemetry, telemetry_path)
    print(json.dumps(telemetry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
