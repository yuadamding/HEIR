"""Registration-preserving, fully refitted nulls for neural HEST probes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

NULL_FAMILIES = (
    "within_section_type_derangement",
    "different_spatial_block_reassignment",
)
COMPLETE_REFIT_STEPS = (
    "preprocessing",
    "target_fitting",
    "hyperparameter_selection",
    "training",
    "checkpoint_selection",
    "prediction",
    "scoring",
)


@dataclass(frozen=True)
class NeuralNullDesign:
    kind: str
    mappings: tuple[np.ndarray, ...]
    receipts: tuple[Mapping[str, object], ...]
    mapping_set_sha256: str


def mapping_sha256(mapping: np.ndarray, observation_ids: np.ndarray) -> str:
    values = np.asarray(mapping, dtype=np.int64)
    identities = np.asarray(observation_ids).astype(str)
    if values.ndim != 1 or identities.shape != values.shape:
        raise ValueError("null mapping and observation IDs must be aligned vectors")
    if len(set(identities.tolist())) != len(identities):
        raise ValueError("null observation IDs must be unique")
    pairs = sorted((identities[row], identities[source]) for row, source in enumerate(values))
    encoded = "\n".join(f"{destination}\0{source}" for destination, source in pairs)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _aligned_vectors(*values: object) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value).astype(str) for value in values)
    if not arrays or arrays[0].ndim != 1 or not len(arrays[0]):
        raise ValueError("null identities must be non-empty vectors")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("null identities must be row aligned")
    return arrays


def _groups(observation_ids: np.ndarray, *identities: np.ndarray) -> tuple[np.ndarray, ...]:
    keys = list(zip(*(values.tolist() for values in identities)))
    grouped: dict[tuple[str, ...], list[int]] = {}
    for row, key in enumerate(keys):
        grouped.setdefault(key, []).append(row)
    return tuple(
        np.asarray(sorted(grouped[key], key=lambda row: str(observation_ids[row])), dtype=np.int64)
        for key in sorted(grouped)
    )


def within_section_type_derangement(
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    type_ids: np.ndarray,
    observation_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    donors, sections, types = _aligned_vectors(donor_ids, section_ids, type_ids)
    identities = np.asarray(observation_ids).astype(str)
    if identities.shape != donors.shape or len(set(identities.tolist())) != len(identities):
        raise ValueError("null observation IDs must be unique and row aligned")
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(donors), dtype=np.int64)
    groups = _groups(identities, donors, sections, types)
    active_rows = 0
    active_groups = 0
    for group in groups:
        if len(group) < 2:
            continue
        order = group[rng.permutation(len(group))]
        mapping[order] = np.roll(order, 1)
        active_rows += len(group)
        active_groups += 1
    if not (
        np.array_equal(donors, donors[mapping])
        and np.array_equal(sections, sections[mapping])
        and np.array_equal(types, types[mapping])
    ):
        raise RuntimeError("within-section/type derangement crossed a frozen stratum")
    changed = mapping != np.arange(len(mapping))
    return mapping, {
        "kind": "within_section_type_derangement",
        "seed": int(seed),
        "mapping_sha256": mapping_sha256(mapping, identities),
        "groups": len(groups),
        "active_groups": active_groups,
        "eligible_rows": active_rows,
        "changed_fraction": float(np.mean(changed)),
        "cross_block_fraction": None,
    }


def different_spatial_block_reassignment(
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    type_ids: np.ndarray,
    block_ids: np.ndarray,
    observation_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    donors, sections, types, blocks = _aligned_vectors(donor_ids, section_ids, type_ids, block_ids)
    identities = np.asarray(observation_ids).astype(str)
    if identities.shape != donors.shape or len(set(identities.tolist())) != len(identities):
        raise ValueError("null observation IDs must be unique and row aligned")
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(donors), dtype=np.int64)
    groups = _groups(identities, donors, sections, types)
    feasible = 0
    infeasible = 0
    eligible_rows = 0
    for group in groups:
        block_names = sorted(set(blocks[group].tolist()))
        chunks = [group[blocks[group] == block] for block in block_names]
        maximum = max((len(chunk) for chunk in chunks), default=0)
        if len(chunks) < 2 or maximum > len(group) / 2:
            infeasible += 1
            continue
        order = rng.permutation(len(chunks))
        rows = np.concatenate(
            [chunks[index][rng.permutation(len(chunks[index]))] for index in order]
        )
        assigned = None
        for shift in range(maximum, len(group) - maximum + 1):
            candidate = np.roll(rows, shift)
            if np.all(blocks[rows] != blocks[candidate]):
                assigned = candidate
                break
        if assigned is None:
            infeasible += 1
            continue
        mapping[rows] = assigned
        feasible += 1
        eligible_rows += len(group)
    if not (
        np.array_equal(donors, donors[mapping])
        and np.array_equal(sections, sections[mapping])
        and np.array_equal(types, types[mapping])
    ):
        raise RuntimeError("different-block reassignment crossed a frozen stratum")
    changed = mapping != np.arange(len(mapping))
    if np.any(changed & (blocks == blocks[mapping])):
        raise RuntimeError("active different-block reassignment retained a spatial block")
    return mapping, {
        "kind": "different_spatial_block_reassignment",
        "seed": int(seed),
        "mapping_sha256": mapping_sha256(mapping, identities),
        "groups": len(groups),
        "feasible_groups": feasible,
        "infeasible_groups": infeasible,
        "eligible_rows": eligible_rows,
        "changed_fraction": float(np.mean(changed)),
        "cross_block_fraction": float(np.mean(blocks != blocks[mapping])),
    }


def build_neural_null_design(
    kind: str,
    permutations: int,
    donor_ids: np.ndarray,
    section_ids: np.ndarray,
    type_ids: np.ndarray,
    block_ids: np.ndarray,
    observation_ids: np.ndarray,
    *,
    seed: int,
) -> NeuralNullDesign:
    if kind not in NULL_FAMILIES or permutations < 1:
        raise ValueError("unknown neural null family or invalid permutation count")
    mappings = []
    receipts = []
    for index in range(permutations):
        local_seed = int(seed + index * 104729)
        if kind == "within_section_type_derangement":
            mapping, receipt = within_section_type_derangement(
                donor_ids, section_ids, type_ids, observation_ids, seed=local_seed
            )
        else:
            mapping, receipt = different_spatial_block_reassignment(
                donor_ids,
                section_ids,
                type_ids,
                block_ids,
                observation_ids,
                seed=local_seed,
            )
        mappings.append(mapping)
        receipts.append(receipt)
    hashes = [str(receipt["mapping_sha256"]) for receipt in receipts]
    return NeuralNullDesign(
        kind=kind,
        mappings=tuple(mappings),
        receipts=tuple(receipts),
        mapping_set_sha256=hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest(),
    )


def apply_joint_image_mapping(features: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Apply one row map jointly to every carried image view."""

    values = np.asarray(features)
    row_map = np.asarray(mapping, dtype=np.int64)
    if values.ndim not in {2, 3} or row_map.shape != (len(values),):
        raise ValueError("joint image mapping inputs are malformed")
    if sorted(row_map.tolist()) != list(range(len(row_map))):
        raise ValueError("joint image mapping must be a row permutation")
    return values[row_map]


def run_refitted_neural_null(
    observed_statistic: float,
    image_features: np.ndarray,
    design: NeuralNullDesign,
    refit_and_score: Callable[
        [np.ndarray, np.ndarray, Mapping[str, object]], Mapping[str, object]
    ],
) -> Mapping[str, object]:
    """Refit the complete caller-owned pipeline for every registered mapping.

    The callback receives mapped image features, the mapping, and its receipt.
    It is intentionally not given mapped nuisance features or outcomes.  The
    experiment runner owns and must refit preprocessing, target construction,
    nested selection, training, checkpointing, prediction, and scoring.
    """

    if not np.isfinite(observed_statistic):
        raise ValueError("observed neural-null statistic must be finite")
    values = []
    refit_receipts = []
    for mapping, receipt in zip(design.mappings, design.receipts):
        mapped = apply_joint_image_mapping(image_features, mapping)
        refit_receipt = refit_and_score(mapped, mapping.copy(), receipt)
        if not isinstance(refit_receipt, Mapping):
            raise RuntimeError("refitted neural null callback must return an execution receipt")
        if refit_receipt.get("mapping_sha256") != receipt["mapping_sha256"]:
            raise RuntimeError("refitted neural null receipt is bound to the wrong mapping")
        if refit_receipt.get("completed_steps") != list(COMPLETE_REFIT_STEPS):
            raise RuntimeError("refitted neural null did not attest every registered fit step")
        score = float(refit_receipt.get("score", float("nan")))
        if not np.isfinite(score):
            raise RuntimeError("refitted neural null returned a non-finite statistic")
        values.append(score)
        refit_receipts.append(dict(refit_receipt))
    array = np.asarray(values, dtype=np.float64)
    encoded_receipts = json.dumps(
        refit_receipts,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "kind": design.kind,
        "permutations": len(values),
        "values": values,
        "observed_statistic": float(observed_statistic),
        "empirical_p": float(
            (1 + np.count_nonzero(array >= observed_statistic)) / (1 + len(array))
        ),
        "mapping_set_sha256": design.mapping_set_sha256,
        "model_refit_for_every_permutation": True,
        "refit_scope": list(COMPLETE_REFIT_STEPS),
        "refit_receipts": refit_receipts,
        "refit_receipts_sha256": hashlib.sha256(encoded_receipts).hexdigest(),
        "receipts": list(design.receipts),
    }


__all__ = [
    "NULL_FAMILIES",
    "COMPLETE_REFIT_STEPS",
    "NeuralNullDesign",
    "apply_joint_image_mapping",
    "build_neural_null_design",
    "different_spatial_block_reassignment",
    "mapping_sha256",
    "run_refitted_neural_null",
    "within_section_type_derangement",
]
