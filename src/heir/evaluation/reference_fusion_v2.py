"""Preregistered v2 molecular-reference primitives for regional validation.

This module deliberately leaves :mod:`reference_fusion` unchanged so frozen v1
reports remain reproducible.  V2 replaces identity-hash averages with molecular
k-means, constrains cross-assay calibration to corresponding latent axes, and
allows the fusion weight to be selected over the full interpolation interval
using training donors only.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np

from heir.evaluation.reference_fusion import (
    PrototypeBank,
    _matrix,
    _stable_hash,
    _strings,
    _weights,
)

MOLECULAR_KMEANS_MAXIMUM_ITERATIONS = 1_000


def _stable_observation_order(observation_ids: np.ndarray, seed: int) -> np.ndarray:
    """Order rows reproducibly without depending on their input positions."""

    return np.asarray(
        sorted(
            range(len(observation_ids)),
            key=lambda index: (
                _stable_hash((observation_ids[index],), seed),
                observation_ids[index],
            ),
        ),
        dtype=np.int64,
    )


def _deterministic_molecular_kmeans(
    values: np.ndarray,
    observation_ids: np.ndarray,
    clusters: int,
    *,
    seed: int,
    maximum_iterations: int = MOLECULAR_KMEANS_MAXIMUM_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster molecular states with deterministic farthest-point initialization."""

    rows, width = values.shape
    requested = int(clusters)
    iterations = int(maximum_iterations)
    if requested <= 0 or iterations <= 0:
        raise ValueError("clusters and maximum_iterations must be positive")
    stable_order = _stable_observation_order(observation_ids, seed)
    unique_rows: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index in stable_order.tolist():
        key = tuple(float(value) for value in values[index])
        if key not in seen:
            seen.add(key)
            unique_rows.append(index)
    count = min(requested, rows, len(unique_rows))
    if count == 0:
        raise ValueError("molecular k-means received no unique finite state")

    # One identity-seeded center fixes orientation.  Remaining centers maximize
    # molecular distance, so observation identity cannot define the clusters.
    centers = [values[unique_rows[0]].copy()]
    selected = {unique_rows[0]}
    stable_rank = {index: rank for rank, index in enumerate(stable_order.tolist())}
    while len(centers) < count:
        center_matrix = np.vstack(centers)
        candidates = [index for index in unique_rows if index not in selected]
        distance = np.min(
            np.sum(
                np.square(values[candidates, None, :] - center_matrix[None, :, :]),
                axis=2,
            ),
            axis=1,
        )
        maximum = float(np.max(distance))
        tied = [
            candidate for candidate, value in zip(candidates, distance.tolist()) if value == maximum
        ]
        chosen = min(tied, key=stable_rank.__getitem__)
        centers.append(values[chosen].copy())
        selected.add(chosen)

    center_matrix = np.vstack(centers)
    previous: np.ndarray | None = None
    seen_assignment_digests: dict[bytes, int] = {}
    labels = np.zeros(rows, dtype=np.int64)
    for _iteration in range(iterations):
        distance = np.sum(np.square(values[:, None, :] - center_matrix[None, :, :]), axis=2)
        labels = np.argmin(distance, axis=1).astype(np.int64)
        counts = np.bincount(labels, minlength=count)
        for empty in np.flatnonzero(counts == 0).tolist():
            movable = [index for index in stable_order.tolist() if counts[labels[index]] > 1]
            if not movable:
                raise RuntimeError("molecular k-means cannot repair an empty cluster")
            assigned_distance = distance[np.arange(rows), labels]
            maximum = max(float(assigned_distance[index]) for index in movable)
            chosen = min(
                (index for index in movable if float(assigned_distance[index]) == maximum),
                key=stable_rank.__getitem__,
            )
            counts[labels[chosen]] -= 1
            labels[chosen] = empty
            counts[empty] += 1

        updated = np.empty((count, width), dtype=np.float64)
        for cluster in range(count):
            # Stable accumulation makes cluster means invariant to input order.
            members = [index for index in stable_order.tolist() if labels[index] == cluster]
            updated[cluster] = np.mean(values[members], axis=0, dtype=np.float64)
        if previous is not None and np.array_equal(labels, previous):
            center_matrix = updated
            break
        canonical_labels = np.ascontiguousarray(labels, dtype="<i8")
        assignment_digest = sha256(canonical_labels.tobytes(order="C")).digest()
        if assignment_digest in seen_assignment_digests:
            first_iteration = seen_assignment_digests[assignment_digest]
            raise RuntimeError(
                "molecular k-means entered a deterministic assignment cycle: "
                f"iteration {first_iteration} repeated at {_iteration + 1}"
            )
        seen_assignment_digests[assignment_digest] = _iteration + 1
        previous = labels.copy()
        center_matrix = updated
    else:
        raise RuntimeError(
            f"molecular k-means did not converge deterministically within {iterations} iterations"
        )
    return center_matrix, labels


def build_reference_prototypes(
    latent: object,
    donor_ids: Sequence[object],
    type_labels: Sequence[object],
    observation_ids: Sequence[object],
    *,
    max_prototypes_per_type: int = 8,
    seed: int = 17,
) -> PrototypeBank:
    """Create deterministic donor/type prototypes from molecular-state clusters.

    Only the intended sc/snRNA inference bank is accepted.  Clustering sees
    only the supplied reference latent; the NatCommun caller calibrates that
    latent using training donors and excludes held-out spatial outcomes.  One
    prototype gives the donor/type-centroid sensitivity; 8--16 prototypes can
    represent within-type state structure.
    """

    values = _matrix(latent, "reference latent")
    rows = len(values)
    donors = _strings(donor_ids, "donor_ids", rows)
    types = _strings(type_labels, "type_labels", rows)
    observations = _strings(observation_ids, "observation_ids", rows)
    if len(set(observations.tolist())) != rows:
        raise ValueError("reference observation IDs must be unique")
    maximum = int(max_prototypes_per_type)
    if maximum <= 0 or maximum != max_prototypes_per_type:
        raise ValueError("max_prototypes_per_type must be a positive integer")

    states: list[np.ndarray] = []
    counts: list[float] = []
    output_donors: list[str] = []
    output_types: list[str] = []
    output_ids: list[str] = []
    for donor in sorted(set(donors.tolist())):
        for type_label in sorted(set(types[donors == donor].tolist())):
            indices = np.flatnonzero((donors == donor) & (types == type_label))
            try:
                local_states, labels = _deterministic_molecular_kmeans(
                    values[indices],
                    observations[indices],
                    min(len(indices), maximum),
                    seed=_stable_hash((donor, type_label), seed),
                    maximum_iterations=MOLECULAR_KMEANS_MAXIMUM_ITERATIONS,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "molecular k-means failed for "
                    f"donor={donor!r}, type={type_label!r}, rows={len(indices)}, "
                    f"prototypes={min(len(indices), maximum)}"
                ) from error
            for cluster, state in enumerate(local_states):
                states.append(state)
                counts.append(float(np.sum(labels == cluster)))
                output_donors.append(donor)
                output_types.append(type_label)
                output_ids.append(f"{donor}::{type_label}::molecular_kmeans::{cluster}")
    return PrototypeBank(
        states=np.vstack(states),
        weights=np.asarray(counts, dtype=np.float64),
        donor_ids=np.asarray(output_donors),
        type_labels=np.asarray(output_types),
        prototype_ids=np.asarray(output_ids),
    )


@dataclass(frozen=True)
class ReferenceCalibrator:
    """Constrained training-donor-only map from reference to target latent."""

    coefficients: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    fit_donors: tuple[str, ...]
    ridge_alpha: float
    mode: str
    pairing_unit: str
    indication_labels: tuple[str, ...]
    indication_slopes: np.ndarray | None
    indication_source_means: np.ndarray | None
    indication_target_means: np.ndarray | None
    donor_indications: tuple[tuple[str, str], ...]
    paired_summary_rows: int
    qualified_indications: tuple[str, ...] = ()
    fallback_indications: tuple[str, ...] = ()

    def transform(
        self,
        values: object,
        *,
        donor_ids: Sequence[object] | None = None,
        indication_ids: Sequence[object] | None = None,
    ) -> np.ndarray:
        matrix = _matrix(values, "reference values")
        if matrix.shape[1] != len(self.source_mean):
            raise ValueError("reference values differ from calibrator width")
        global_result = (matrix - self.source_mean) @ self.coefficients + self.target_mean
        if self.mode == "global_diagonal":
            return global_result
        if self.mode != "indication_diagonal":
            raise ValueError(f"unsupported calibration mode: {self.mode}")
        if indication_ids is not None and donor_ids is not None:
            raise ValueError("provide donor_ids or indication_ids, not both")
        if indication_ids is not None:
            indications = _strings(indication_ids, "indication_ids", len(matrix))
        elif donor_ids is not None:
            donors = _strings(donor_ids, "donor_ids", len(matrix))
            lookup = dict(self.donor_indications)
            missing = sorted(set(donors.tolist()) - set(lookup))
            if missing:
                raise ValueError(f"calibrator lacks donor indications: {missing}")
            indications = np.asarray([lookup[donor] for donor in donors])
        else:
            raise ValueError("indication-aware calibration requires donor_ids or indication_ids")
        qualified = self.qualified_indications or self.indication_labels
        registered = (
            {indication for _donor, indication in self.donor_indications}
            | set(qualified)
            | set(self.fallback_indications)
        )
        unknown = sorted(set(indications.tolist()) - registered)
        if unknown:
            raise ValueError(f"calibrator has no registered indications: {unknown}")
        assert self.indication_slopes is not None
        assert self.indication_source_means is not None
        assert self.indication_target_means is not None
        result = global_result.copy()
        for group_index, indication in enumerate(self.indication_labels):
            selected = indications == indication
            result[selected] = (
                matrix[selected] - self.indication_source_means[group_index]
            ) * self.indication_slopes[group_index] + self.indication_target_means[group_index]
        return result


def _diagonal_calibration_parameters(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit corresponding-axis slopes with an identity-centred ridge prior."""

    normalized = weights / weights.sum(dtype=np.float64)
    source_mean = np.sum(source * normalized[:, None], axis=0)
    target_mean = np.sum(target * normalized[:, None], axis=0)
    x = source - source_mean
    y = target - target_mean
    numerator = np.sum(normalized[:, None] * x * y, axis=0) + ridge_alpha
    denominator = np.sum(normalized[:, None] * np.square(x), axis=0) + ridge_alpha
    return source_mean, target_mean, numerator / denominator


def _normalize_donor_indications(
    donor_indications: Mapping[str, str],
) -> dict[str, str]:
    """Validate one unambiguous, non-empty indication registration per donor."""

    normalized: dict[str, str] = {}
    for raw_donor, raw_indication in donor_indications.items():
        donor = str(raw_donor)
        indication = str(raw_indication)
        if not donor or not indication:
            raise ValueError("donor_indications contains an empty donor or indication")
        if donor in normalized and normalized[donor] != indication:
            raise ValueError(f"donor_indications is ambiguous after normalization: {donor}")
        normalized[donor] = indication
    return normalized


def _paired_calibration_summaries(
    reference: np.ndarray,
    reference_donors: np.ndarray,
    target: np.ndarray,
    target_donors: np.ndarray,
    fit_donors: tuple[str, ...],
    reference_types: np.ndarray | None,
    target_types: np.ndarray | None,
    donor_indications: Mapping[str, str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], str]:
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    pair_donors: list[str] = []
    pair_indications: list[str] = []
    paired_donors: list[str] = []
    for donor in fit_donors:
        source_mask = reference_donors == donor
        target_mask = target_donors == donor
        if not source_mask.any() or not target_mask.any():
            continue
        if donor_indications is not None and donor not in donor_indications:
            raise ValueError(f"donor_indications lacks paired fit donor: {donor}")
        indication = "__all__" if donor_indications is None else str(donor_indications[donor])
        if not indication:
            raise ValueError("donor_indications contains an empty indication")
        local_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        if reference_types is None:
            local_pairs.append(
                (reference[source_mask].mean(axis=0), target[target_mask].mean(axis=0))
            )
        else:
            assert target_types is not None
            shared = sorted(
                set(reference_types[source_mask].tolist()) & set(target_types[target_mask].tolist())
            )
            for type_label in shared:
                local_pairs.append(
                    (
                        reference[source_mask & (reference_types == type_label)].mean(axis=0),
                        target[target_mask & (target_types == type_label)].mean(axis=0),
                    )
                )
        if not local_pairs:
            continue
        paired_donors.append(donor)
        for source_mean, target_mean in local_pairs:
            source_rows.append(source_mean)
            target_rows.append(target_mean)
            pair_donors.append(donor)
            pair_indications.append(indication)
    if len(paired_donors) < 2:
        raise ValueError("reference calibration requires at least two paired training donors")
    donor_pair_counts = {donor: pair_donors.count(donor) for donor in sorted(set(pair_donors))}
    if donor_indications is None:
        indication_donor_counts: Mapping[str, int] = {"__all__": len(paired_donors)}
    else:
        indication_donor_counts = {
            indication: sum(donor_indications[donor] == indication for donor in paired_donors)
            for indication in sorted({donor_indications[donor] for donor in paired_donors})
        }
    weights = np.asarray(
        [
            1.0
            / (
                donor_pair_counts[donor]
                * indication_donor_counts[
                    "__all__" if donor_indications is None else donor_indications[donor]
                ]
            )
            for donor in pair_donors
        ],
        dtype=np.float64,
    )
    return (
        np.vstack(source_rows),
        np.vstack(target_rows),
        np.asarray(pair_indications),
        weights,
        tuple(paired_donors),
        "donor" if reference_types is None else "donor_x_type",
    )


def fit_reference_calibrator(
    reference_values: object,
    reference_donor_ids: Sequence[object],
    target_values: object,
    target_donor_ids: Sequence[object],
    fit_donor_ids: Sequence[object],
    *,
    ridge_alpha: float = 1.0,
    reference_type_labels: Sequence[object] | None = None,
    target_type_labels: Sequence[object] | None = None,
    donor_indications: Mapping[str, str] | None = None,
) -> ReferenceCalibrator:
    """Fit diagonal global or hierarchical indication-aware calibration.

    Supplying both type-label vectors changes the paired unit from donor mean to
    donor-by-type pseudobulk.  Each donor receives equal total weight regardless
    of its number of shared types.  An indication-specific map is qualified only
    by at least two paired fit donors; every other registered indication uses the
    global map.  Neither map can mix molecular axes.
    """

    reference = _matrix(reference_values, "reference values")
    target = _matrix(target_values, "target values")
    if reference.shape[1] != target.shape[1]:
        raise ValueError("reference and target latent widths differ")
    reference_donors = _strings(reference_donor_ids, "reference_donor_ids", len(reference))
    target_donors = _strings(target_donor_ids, "target_donor_ids", len(target))
    fit_donors = tuple(sorted(set(str(value) for value in fit_donor_ids)))
    if not fit_donors:
        raise ValueError("fit_donor_ids cannot be empty")
    if (reference_type_labels is None) != (target_type_labels is None):
        raise ValueError("reference and target type labels must be supplied together")
    reference_types = (
        None
        if reference_type_labels is None
        else _strings(reference_type_labels, "reference_type_labels", len(reference))
    )
    target_types = (
        None
        if target_type_labels is None
        else _strings(target_type_labels, "target_type_labels", len(target))
    )
    alpha = float(ridge_alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge_alpha must be finite and positive")
    normalized_indications = (
        None if donor_indications is None else _normalize_donor_indications(donor_indications)
    )
    source, destination, indications, weights, paired, pairing_unit = _paired_calibration_summaries(
        reference,
        reference_donors,
        target,
        target_donors,
        fit_donors,
        reference_types,
        target_types,
        normalized_indications,
    )
    source_mean, target_mean, slopes = _diagonal_calibration_parameters(
        source, destination, weights, alpha
    )

    labels: tuple[str, ...] = ()
    group_slopes: np.ndarray | None = None
    group_source: np.ndarray | None = None
    group_target: np.ndarray | None = None
    indication_lookup: tuple[tuple[str, str], ...] = ()
    qualified_indications: tuple[str, ...] = ()
    fallback_indications: tuple[str, ...] = ()
    mode = "global_diagonal"
    if normalized_indications is not None:
        registered_indications = tuple(sorted(set(normalized_indications.values())))
        paired_counts = {
            indication: sum(normalized_indications[donor] == indication for donor in paired)
            for indication in registered_indications
        }
        # A donor-by-type fit can contribute several summary rows per donor.  The
        # qualification threshold intentionally counts paired donors, not rows.
        qualified_indications = tuple(
            indication for indication in registered_indications if paired_counts[indication] >= 2
        )
        fallback_indications = tuple(
            indication
            for indication in registered_indications
            if indication not in qualified_indications
        )
        labels = qualified_indications
        group_slopes = np.empty((len(labels), reference.shape[1]), dtype=np.float64)
        group_source = np.empty_like(group_slopes)
        group_target = np.empty_like(group_slopes)
        for group_index, indication in enumerate(labels):
            selected = indications == indication
            (
                group_source[group_index],
                group_target[group_index],
                group_slopes[group_index],
            ) = _diagonal_calibration_parameters(
                source[selected], destination[selected], weights[selected], alpha
            )
        indication_lookup = tuple(sorted(normalized_indications.items()))
        mode = "indication_diagonal"
    return ReferenceCalibrator(
        coefficients=np.diag(slopes),
        source_mean=source_mean,
        target_mean=target_mean,
        fit_donors=paired,
        ridge_alpha=alpha,
        mode=mode,
        pairing_unit=pairing_unit,
        indication_labels=labels,
        indication_slopes=group_slopes,
        indication_source_means=group_source,
        indication_target_means=group_target,
        donor_indications=indication_lookup,
        paired_summary_rows=len(source),
        qualified_indications=qualified_indications,
        fallback_indications=fallback_indications,
    )


def _heldout_calibration_loss(
    calibrator: ReferenceCalibrator,
    reference: np.ndarray,
    reference_donors: np.ndarray,
    target: np.ndarray,
    target_donors: np.ndarray,
    donor: str,
    reference_types: np.ndarray | None,
    target_types: np.ndarray | None,
) -> float:
    reference_mask = reference_donors == donor
    target_mask = target_donors == donor
    transformed = calibrator.transform(
        reference[reference_mask],
        donor_ids=(
            reference_donors[reference_mask] if calibrator.mode == "indication_diagonal" else None
        ),
    )
    losses: list[float] = []
    if reference_types is None:
        losses.append(
            float(np.mean(np.square(transformed.mean(axis=0) - target[target_mask].mean(axis=0))))
        )
    else:
        assert target_types is not None
        local_reference_types = reference_types[reference_mask]
        shared = sorted(
            set(local_reference_types.tolist()) & set(target_types[target_mask].tolist())
        )
        for type_label in shared:
            losses.append(
                float(
                    np.mean(
                        np.square(
                            transformed[local_reference_types == type_label].mean(axis=0)
                            - target[target_mask & (target_types == type_label)].mean(axis=0)
                        )
                    )
                )
            )
    if not losses:
        raise ValueError(f"held-out calibration donor has no paired summary: {donor}")
    return float(np.mean(losses))


def select_reference_calibration_alpha(
    reference_values: object,
    reference_donor_ids: Sequence[object],
    target_values: object,
    target_donor_ids: Sequence[object],
    fit_donor_ids: Sequence[object],
    *,
    candidate_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    reference_type_labels: Sequence[object] | None = None,
    target_type_labels: Sequence[object] | None = None,
    donor_indications: Mapping[str, str] | None = None,
) -> tuple[float, Mapping[str, object]]:
    """Select shrinkage by donor-equal leave-one-training-donor-out loss.

    Every fold refits the same qualified-indication/global-fallback hierarchy
    used by the final fit.  Donors outside ``fit_donor_ids`` are never scored.
    """

    reference = _matrix(reference_values, "reference values")
    target = _matrix(target_values, "target values")
    if reference.shape[1] != target.shape[1]:
        raise ValueError("reference and target latent widths differ")
    reference_donors = _strings(reference_donor_ids, "reference_donor_ids", len(reference))
    target_donors = _strings(target_donor_ids, "target_donor_ids", len(target))
    if (reference_type_labels is None) != (target_type_labels is None):
        raise ValueError("reference and target type labels must be supplied together")
    reference_types = (
        None
        if reference_type_labels is None
        else _strings(reference_type_labels, "reference_type_labels", len(reference))
    )
    target_types = (
        None
        if target_type_labels is None
        else _strings(target_type_labels, "target_type_labels", len(target))
    )
    fit_donors = tuple(sorted(set(str(value) for value in fit_donor_ids)))
    paired = tuple(
        donor
        for donor in fit_donors
        if np.any(reference_donors == donor) and np.any(target_donors == donor)
    )
    if len(paired) < 3:
        raise ValueError("calibration alpha selection requires at least three paired fit donors")
    normalized_indications = (
        None if donor_indications is None else _normalize_donor_indications(donor_indications)
    )
    if normalized_indications is not None:
        missing = sorted(set(paired) - set(normalized_indications))
        if missing:
            raise ValueError(f"donor_indications lacks paired fit donors: {missing}")
    alphas = tuple(sorted(set(float(value) for value in candidate_alphas)))
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("candidate_alphas must be non-empty, finite, and positive")

    losses: dict[str, float] = {}
    donor_equal_losses: dict[str, float] = {}
    donor_losses: dict[str, Mapping[str, float]] = {}
    indication_losses: dict[str, Mapping[str, float]] = {}
    fold_mapping: dict[str, Mapping[str, object]] = {}
    for alpha in alphas:
        local: dict[str, float] = {}
        for validation_donor in paired:
            inner_donors = [donor for donor in paired if donor != validation_donor]
            calibrator = fit_reference_calibrator(
                reference,
                reference_donors,
                target,
                target_donors,
                inner_donors,
                ridge_alpha=alpha,
                reference_type_labels=reference_types,
                target_type_labels=target_types,
                donor_indications=normalized_indications,
            )
            mapping = {
                "qualified_indications": list(calibrator.qualified_indications),
                "global_fallback_indications": list(calibrator.fallback_indications),
                "minimum_paired_donors_per_indication": 2,
            }
            previous_mapping = fold_mapping.setdefault(validation_donor, mapping)
            if previous_mapping != mapping:
                raise RuntimeError("hierarchical calibration mapping changed across alphas")
            local[validation_donor] = _heldout_calibration_loss(
                calibrator,
                reference,
                reference_donors,
                target,
                target_donors,
                validation_donor,
                reference_types,
                target_types,
            )
        key = f"{alpha:g}"
        donor_losses[key] = local
        donor_equal_losses[key] = float(np.mean(tuple(local.values())))
        if normalized_indications is None:
            indication_losses[key] = {}
            losses[key] = donor_equal_losses[key]
        else:
            local_indications = {
                indication: float(
                    np.mean(
                        [
                            loss
                            for donor, loss in local.items()
                            if normalized_indications[donor] == indication
                        ]
                    )
                )
                for indication in sorted({normalized_indications[donor] for donor in local})
            }
            indication_losses[key] = local_indications
            losses[key] = float(np.mean(tuple(local_indications.values())))
    selected_loss, selected = min((losses[f"{alpha:g}"], alpha) for alpha in alphas)
    final_calibrator = fit_reference_calibrator(
        reference,
        reference_donors,
        target,
        target_donors,
        paired,
        ridge_alpha=selected,
        reference_type_labels=reference_types,
        target_type_labels=target_types,
        donor_indications=normalized_indications,
    )
    return selected, {
        "schema": "heir.reference_calibration_selection.v2",
        "fit_donors": list(paired),
        "candidate_alphas": list(alphas),
        "candidate_selection_mse": losses,
        "candidate_donor_equal_mse": donor_equal_losses,
        "candidate_donor_mse": donor_losses,
        "candidate_per_indication_donor_equal_mse": indication_losses,
        "selected_alpha": selected,
        "selected_loss": selected_loss,
        "selection": "leave_one_fit_donor_out",
        "selection_weighting": (
            "indication_equal_then_donor_equal_within_indication"
            if normalized_indications is not None
            else "donor_equal"
        ),
        "calibration_fit_weighting": (
            "indication_equal_then_donor_equal_then_pair_equal"
            if normalized_indications is not None
            else "donor_equal_then_pair_equal"
        ),
        "hierarchical_mapping": (
            "indication_specific_diagonal_with_global_fallback"
            if normalized_indications is not None
            else "global_diagonal"
        ),
        "minimum_paired_donors_per_indication": 2,
        "fold_hierarchical_mapping": fold_mapping,
        "final_qualified_indications": list(final_calibrator.qualified_indications),
        "final_global_fallback_indications": list(final_calibrator.fallback_indications),
        "axis_mapping": "diagonal_only",
        "non_fit_donor_outcomes_used": False,
    }


def residual_fusion(image_state: object, reference_state: object, alpha: float) -> np.ndarray:
    """Apply one prespecified interpolation without imposing an H&E weight floor."""

    image = _matrix(image_state, "image_state")
    reference = _matrix(reference_state, "reference_state")
    if reference.shape != image.shape:
        raise ValueError("image and reference states must align")
    coefficient = float(alpha)
    if not np.isfinite(coefficient) or coefficient < 0 or coefficient > 1:
        raise ValueError("alpha must lie in [0, 1]")
    return image + coefficient * (reference - image)


def adaptive_residual_fusion(
    image_state: object,
    reference_state: object,
    support_weight: object,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a training-selected interpolation to zero outside bank support."""

    image = _matrix(image_state, "image_state")
    reference = _matrix(reference_state, "reference_state")
    if image.shape != reference.shape:
        raise ValueError("image and reference states must align")
    support = np.asarray(support_weight, dtype=np.float64)
    if support.shape != (len(image),) or not np.isfinite(support).all():
        raise ValueError("support_weight must be a finite row vector")
    if np.any((support < 0) | (support > 1)):
        raise ValueError("support_weight must lie in [0, 1]")
    coefficient = float(alpha)
    if not np.isfinite(coefficient) or coefficient < 0 or coefficient > 1:
        raise ValueError("alpha must lie in [0, 1]")
    row_alpha = coefficient * support
    return image + row_alpha[:, None] * (reference - image), row_alpha


def select_fusion_alpha(
    image_state: object,
    reference_state: object,
    truth: object,
    *,
    donor_ids: Sequence[object],
    fit_donor_ids: Sequence[object],
    candidate_alphas: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    sample_weight: object | None = None,
) -> tuple[float, Mapping[str, object]]:
    """Select fusion alpha strictly within the named training donors."""

    image = _matrix(image_state, "image_state")
    reference = _matrix(reference_state, "reference_state")
    target = _matrix(truth, "truth")
    if image.shape != reference.shape or target.shape != image.shape:
        raise ValueError("fusion selection matrices must align")
    donors = _strings(donor_ids, "donor_ids", len(image))
    fit_donors = tuple(sorted(set(str(value) for value in fit_donor_ids)))
    if not fit_donors:
        raise ValueError("fit_donor_ids cannot be empty")
    selected_rows = np.isin(donors, fit_donors)
    if not selected_rows.any():
        raise ValueError("no fusion-selection rows belong to fit_donor_ids")
    weights = _weights(sample_weight, len(image))[selected_rows]
    alphas = tuple(sorted(set(float(value) for value in candidate_alphas)))
    if not alphas or any(not np.isfinite(alpha) or alpha < 0 or alpha > 1 for alpha in alphas):
        raise ValueError("candidate_alphas must be non-empty and lie in [0, 1]")

    losses: dict[str, float] = {}
    donor_losses: dict[str, Mapping[str, float]] = {}
    for alpha in alphas:
        prediction = residual_fusion(image[selected_rows], reference[selected_rows], alpha)
        row_loss = np.mean(np.square(target[selected_rows] - prediction), axis=1)
        local_donors = donors[selected_rows]
        local_weights = weights
        per_donor = {
            donor: float(
                np.average(
                    row_loss[local_donors == donor],
                    weights=local_weights[local_donors == donor],
                )
            )
            for donor in sorted(set(local_donors.tolist()))
        }
        key = f"{alpha:g}"
        donor_losses[key] = per_donor
        losses[key] = float(np.mean(tuple(per_donor.values())))
    selected_loss, selected = min((losses[f"{alpha:g}"], alpha) for alpha in alphas)
    return selected, {
        "schema": "heir.fusion_alpha_selection.v2",
        "fit_donors": list(fit_donors),
        "candidate_alphas": list(alphas),
        "candidate_donor_equal_mse": losses,
        "candidate_donor_mse": donor_losses,
        "selected_alpha": selected,
        "selected_loss": selected_loss,
        "selection_scope": "fit_donors_only",
        "non_fit_donor_outcomes_used": False,
    }
