"""Separate matched-reference specificity gate with equalized molecular banks."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def evaluate_reference_specificity(
    query_latent: np.ndarray,
    query_types: np.ndarray,
    query_donors: Sequence[object],
    matched_bank_latent: np.ndarray,
    matched_bank_types: np.ndarray,
    wrong_banks: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    repeats: int = 100,
    seed: int = 17,
) -> Mapping[str, object]:
    """Compare equal-sized, type-balanced matched and wrong state banks repeatedly."""

    query = np.asarray(query_latent, dtype=np.float64)
    labels = np.asarray(query_types)
    donors = np.asarray([str(value) for value in query_donors])
    matched = np.asarray(matched_bank_latent, dtype=np.float64)
    matched_types = np.asarray(matched_bank_types)
    if repeats < 100:
        raise ValueError("reference-specificity gate requires at least 100 subsampling repeats")
    if (
        query.ndim != 2
        or matched.ndim != 2
        or query.shape[1] != matched.shape[1]
        or labels.shape != (len(query),)
        or donors.shape != (len(query),)
        or matched_types.shape != (len(matched),)
        or not wrong_banks
    ):
        raise ValueError("reference-specificity inputs are malformed")
    banks = {"matched": (matched, matched_types)}
    for name, (latent, types) in wrong_banks.items():
        values = np.asarray(latent, dtype=np.float64)
        bank_types = np.asarray(types)
        if (
            not str(name).strip()
            or values.ndim != 2
            or values.shape[1] != query.shape[1]
            or bank_types.shape != (len(values),)
        ):
            raise ValueError("wrong reference bank is malformed: %s" % name)
        banks[str(name)] = (values, bank_types)
    occupied_types = sorted(set(labels.tolist()))
    per_type_size = {}
    for type_index in occupied_types:
        counts = [int(np.sum(bank_types == type_index)) for _, bank_types in banks.values()]
        if min(counts) < 2:
            raise ValueError("every bank needs at least two states per query type")
        per_type_size[type_index] = min(counts)

    def score(bank: np.ndarray, bank_types: np.ndarray) -> float:
        donor_scores = []
        for donor in sorted(set(donors.tolist())):
            selected_donor = donors == donor
            type_scores = []
            for type_index in occupied_types:
                selected = selected_donor & (labels == type_index)
                if not np.any(selected):
                    continue
                candidates = bank[bank_types == type_index]
                distance = np.square(query[selected, None, :] - candidates[None, :, :]).sum(axis=2)
                type_scores.append(float(np.sqrt(distance.min(axis=1)).mean()))
            donor_scores.append(float(np.mean(type_scores)))
        return float(np.mean(donor_scores))

    rng = np.random.default_rng(seed)
    repeated = {name: [] for name in banks}
    for _ in range(repeats):
        for name, (latent, bank_types) in banks.items():
            chosen = []
            for type_index in occupied_types:
                candidates = np.flatnonzero(bank_types == type_index)
                chosen.extend(
                    rng.choice(candidates, size=per_type_size[type_index], replace=False).tolist()
                )
            selected = np.asarray(chosen, dtype=np.int64)
            repeated[name].append(score(latent[selected], bank_types[selected]))
    comparisons = []
    matched_values = np.asarray(repeated["matched"])
    for name in sorted(wrong_banks):
        wrong_values = np.asarray(repeated[name])
        delta = wrong_values - matched_values
        comparisons.append(
            {
                "wrong_bank": name,
                "mean_wrong_minus_matched_distance": float(delta.mean()),
                "fraction_repeats_matched_better": float(np.mean(delta > 0.0)),
                "matched_mean_distance": float(matched_values.mean()),
                "wrong_mean_distance": float(wrong_values.mean()),
            }
        )
    passed = all(
        row["mean_wrong_minus_matched_distance"] > 0.0
        and row["fraction_repeats_matched_better"] >= 0.95
        for row in comparisons
    )
    return {
        "schema": "heir.reference_specificity_gate.v1",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "repeats": repeats,
        "seed": seed,
        "per_type_bank_size": {str(key): value for key, value in per_type_size.items()},
        "aggregation": "equal_donor_equal_type",
        "comparisons": comparisons,
    }


__all__ = ["evaluate_reference_specificity"]
