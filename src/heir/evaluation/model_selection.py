"""Nested development-donor model selection shared by every feature family."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact

from .ridge_probe import fit_and_score


def subset_artifact(
    artifact: MorphologyRidgeDatasetArtifact, selected: np.ndarray, *, role: str
) -> MorphologyRidgeDatasetArtifact:
    values = {
        name: getattr(artifact, name)
        for name in artifact.__dataclass_fields__
        if name != "role"
    }
    observations = len(artifact.observation_ids)
    for name, value in tuple(values.items()):
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == observations:
            values[name] = value[selected]
    values["role"] = role
    return MorphologyRidgeDatasetArtifact(**values)


def select_hyperparameters(
    development: MorphologyRidgeDatasetArtifact,
    features: np.ndarray,
    *,
    ranks: Sequence[int],
    alphas: Sequence[float],
    minimum_support: int,
    device: str,
    include_composition: bool = False,
) -> Tuple[int, float, Sequence[Mapping[str, object]]]:
    """Select rank/penalty using only leave-one-development-donor-out folds."""

    if features.ndim != 2 or len(features) != len(development.observation_ids):
        raise ValueError("hyperparameter-selection features are not row aligned")
    donors = sorted(set(development.donor_ids.tolist()))
    if len(donors) < 3:
        raise ValueError("nested donor validation requires at least three development donors")
    results = []
    for rank in sorted(set(int(value) for value in ranks)):
        for alpha in sorted(set(float(value) for value in alphas)):
            fold_values = []
            for heldout_donor in donors:
                train_mask = development.donor_ids != heldout_donor
                validation_mask = ~train_mask
                train = subset_artifact(development, train_mask, role="development")
                validation = subset_artifact(development, validation_mask, role="locked_test")
                try:
                    value, *_ = fit_and_score(
                        train,
                        validation,
                        features[train_mask],
                        features[validation_mask],
                        rank=rank,
                        alpha=alpha,
                        minimum_support=minimum_support,
                        device=device,
                        include_composition=include_composition,
                    )
                except ValueError:
                    continue
                fold_values.append(value)
            results.append(
                {
                    "rank": rank,
                    "alpha": alpha,
                    "donor_folds": len(fold_values),
                    "macro_r2": float(np.mean(fold_values)) if fold_values else float("-inf"),
                }
            )
    eligible = [row for row in results if row["donor_folds"] == len(donors)]
    if not eligible:
        raise ValueError("no rank/alpha candidate supports every development donor fold")
    selected = sorted(eligible, key=lambda row: (-row["macro_r2"], row["rank"], row["alpha"]))[0]
    return int(selected["rank"]), float(selected["alpha"]), results


__all__ = ["select_hyperparameters", "subset_artifact"]
