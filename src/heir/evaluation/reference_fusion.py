"""Leakage-resistant, H&E-centred molecular-reference fusion primitives.

The functions in this module are intentionally small and outcome-agnostic.  They
support the regional matched Chromium FLEX--Visium validation and the explicitly
non-authorizing HEST spatial-reference pilot without embedding cohort paths or
software-product concerns in the scientific core.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _matrix(values: object, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return result


def _strings(values: object, name: str, rows: int | None = None) -> np.ndarray:
    result = np.asarray(values).astype(str)
    if result.ndim != 1 or (rows is not None and len(result) != rows):
        raise ValueError(f"{name} must be a one-dimensional aligned vector")
    if any(not value for value in result.tolist()):
        raise ValueError(f"{name} contains empty identifiers")
    return result


def _weights(values: object | None, rows: int) -> np.ndarray:
    if values is None:
        return np.ones(rows, dtype=np.float64)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (rows,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("weights must be finite, positive, and row aligned")
    return result


def _stable_hash(parts: Sequence[object], seed: int) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"HEIRref1")
    digest.update(str(int(seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little")


def split_unique_molecules(
    section_ids: Sequence[object],
    barcode_ids: Sequence[object],
    feature_ids: Sequence[object],
    umi_ids: Sequence[object],
    *,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Hash unique molecules into disjoint, reconstructing transcript halves.

    Duplicate records with the same section/barcode/feature/UMI identity always
    receive the same assignment, preventing a molecule observed in two FASTQ
    lanes from leaking into both halves.
    """

    section = _strings(section_ids, "section_ids")
    rows = len(section)
    barcode = _strings(barcode_ids, "barcode_ids", rows)
    feature = _strings(feature_ids, "feature_ids", rows)
    umi = _strings(umi_ids, "umi_ids", rows)
    assignment_by_key: dict[tuple[str, str, str, str], bool] = {}
    half_a = np.zeros(rows, dtype=bool)
    for index, key in enumerate(zip(section, barcode, feature, umi)):
        canonical = tuple(str(value) for value in key)
        assigned = assignment_by_key.get(canonical)
        if assigned is None:
            assigned = (_stable_hash(canonical, seed) & 1) == 0
            assignment_by_key[canonical] = assigned
        half_a[index] = assigned
    half_b = ~half_a
    unique_a = sum(assignment_by_key.values())
    return half_a, half_b, {
        "schema": "heir.unique_molecule_split.v1",
        "seed": int(seed),
        "records": rows,
        "unique_molecules": len(assignment_by_key),
        "unique_half_a": int(unique_a),
        "unique_half_b": int(len(assignment_by_key) - unique_a),
        "disjoint": bool(not np.any(half_a & half_b)),
        "reconstructs_all_records": bool(np.all(half_a | half_b)),
        "key": "section|barcode|feature|UMI",
    }


@dataclass(frozen=True)
class TargetBasis:
    """Train-only standardization followed by an optional PCA projection."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    fit_donors: tuple[str, ...]

    def transform(self, values: object) -> np.ndarray:
        matrix = _matrix(values, "target values")
        if matrix.shape[1] != len(self.mean):
            raise ValueError("target values differ from the fitted basis width")
        standardized = (matrix - self.mean) / self.scale
        return standardized @ self.components.T

    def inverse_transform(self, latent: object) -> np.ndarray:
        matrix = _matrix(latent, "target latent")
        if matrix.shape[1] != len(self.components):
            raise ValueError("target latent differs from the fitted basis width")
        return (matrix @ self.components) * self.scale + self.mean


def fit_target_basis(
    values: object,
    donor_ids: Sequence[object],
    fit_donor_ids: Sequence[object],
    *,
    n_components: int | None = None,
    sample_weight: object | None = None,
) -> TargetBasis:
    """Fit a molecular basis without using outcomes from held-out donors."""

    matrix = _matrix(values, "target values")
    donors = _strings(donor_ids, "donor_ids", len(matrix))
    fit_donors = tuple(sorted(set(str(value) for value in fit_donor_ids)))
    if not fit_donors:
        raise ValueError("fit_donor_ids cannot be empty")
    mask = np.isin(donors, fit_donors)
    if not mask.any():
        raise ValueError("no rows belong to the requested basis-fit donors")
    weights = _weights(sample_weight, len(matrix))[mask]
    weights = weights / weights.sum(dtype=np.float64)
    training = matrix[mask]
    mean = np.sum(training * weights[:, None], axis=0)
    variance = np.sum(np.square(training - mean) * weights[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1.0e-8] = 1.0
    standardized = (training - mean) / scale
    width = standardized.shape[1]
    components_count = width if n_components is None else int(n_components)
    if components_count <= 0 or components_count > min(standardized.shape):
        raise ValueError("n_components exceeds the train-only target matrix rank")
    if components_count == width:
        components = np.eye(width, dtype=np.float64)
    else:
        weighted = standardized * np.sqrt(weights)[:, None]
        _u, _singular, vt = np.linalg.svd(weighted, full_matrices=False)
        components = vt[:components_count].copy()
        for component in components:
            pivot = int(np.argmax(np.abs(component)))
            if component[pivot] < 0:
                component *= -1
    return TargetBasis(mean, scale, components, fit_donors)


@dataclass(frozen=True)
class PrototypeBank:
    """Deterministic molecular state prototypes with donor/type provenance."""

    states: np.ndarray
    weights: np.ndarray
    donor_ids: np.ndarray
    type_labels: np.ndarray
    prototype_ids: np.ndarray

    def subset(self, indices: object) -> "PrototypeBank":
        selected = np.asarray(indices)
        return PrototypeBank(
            self.states[selected],
            self.weights[selected],
            self.donor_ids[selected],
            self.type_labels[selected],
            self.prototype_ids[selected],
        )


def build_reference_prototypes(
    latent: object,
    donor_ids: Sequence[object],
    type_labels: Sequence[object],
    observation_ids: Sequence[object],
    *,
    max_prototypes_per_type: int = 4,
    seed: int = 17,
) -> PrototypeBank:
    """Create reproducible state prototypes without outcome-guided clustering."""

    values = _matrix(latent, "reference latent")
    rows = len(values)
    donors = _strings(donor_ids, "donor_ids", rows)
    types = _strings(type_labels, "type_labels", rows)
    observations = _strings(observation_ids, "observation_ids", rows)
    if len(set(observations.tolist())) != rows:
        raise ValueError("reference observation IDs must be unique")
    if max_prototypes_per_type <= 0:
        raise ValueError("max_prototypes_per_type must be positive")

    states: list[np.ndarray] = []
    counts: list[float] = []
    output_donors: list[str] = []
    output_types: list[str] = []
    output_ids: list[str] = []
    for donor in sorted(set(donors.tolist())):
        for type_label in sorted(set(types[donors == donor].tolist())):
            indices = np.flatnonzero((donors == donor) & (types == type_label))
            ordered = sorted(
                indices.tolist(),
                key=lambda index: (
                    _stable_hash((observations[index],), seed),
                    observations[index],
                ),
            )
            groups = np.array_split(
                np.asarray(ordered, dtype=np.int64),
                min(len(ordered), max_prototypes_per_type),
            )
            for group_index, group in enumerate(groups):
                states.append(values[group].mean(axis=0, dtype=np.float64))
                counts.append(float(len(group)))
                output_donors.append(donor)
                output_types.append(type_label)
                output_ids.append(f"{donor}::{type_label}::{group_index}")
    return PrototypeBank(
        np.vstack(states),
        np.asarray(counts, dtype=np.float64),
        np.asarray(output_donors),
        np.asarray(output_types),
        np.asarray(output_ids),
    )


@dataclass(frozen=True)
class ReferenceCalibrator:
    """Small training-donor-only affine map from reference to target latent."""

    coefficients: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    fit_donors: tuple[str, ...]
    ridge_alpha: float

    def transform(self, values: object) -> np.ndarray:
        matrix = _matrix(values, "reference values")
        if matrix.shape[1] != len(self.source_mean):
            raise ValueError("reference values differ from calibrator width")
        return (matrix - self.source_mean) @ self.coefficients + self.target_mean


def fit_reference_calibrator(
    reference_values: object,
    reference_donor_ids: Sequence[object],
    target_values: object,
    target_donor_ids: Sequence[object],
    fit_donor_ids: Sequence[object],
    *,
    ridge_alpha: float = 1.0,
) -> ReferenceCalibrator:
    """Fit sc/snRNA-to-spatial calibration from paired training donor means."""

    reference = _matrix(reference_values, "reference values")
    target = _matrix(target_values, "target values")
    if reference.shape[1] != target.shape[1]:
        raise ValueError("reference and target latent widths differ")
    reference_donors = _strings(reference_donor_ids, "reference_donor_ids", len(reference))
    target_donors = _strings(target_donor_ids, "target_donor_ids", len(target))
    fit_donors = tuple(sorted(set(str(value) for value in fit_donor_ids)))
    paired = [
        donor
        for donor in fit_donors
        if np.any(reference_donors == donor) and np.any(target_donors == donor)
    ]
    if len(paired) < 2:
        raise ValueError("reference calibration requires at least two paired training donors")
    source = np.vstack([reference[reference_donors == donor].mean(axis=0) for donor in paired])
    destination = np.vstack([target[target_donors == donor].mean(axis=0) for donor in paired])
    source_mean = source.mean(axis=0)
    target_mean = destination.mean(axis=0)
    x = source - source_mean
    y = destination - target_mean
    alpha = float(ridge_alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge_alpha must be finite and positive")
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    coefficients = np.linalg.solve(gram, x.T @ y)
    return ReferenceCalibrator(coefficients, source_mean, target_mean, tuple(paired), alpha)


def soft_reference_state(
    queries: object,
    bank_states: object,
    bank_weights: object | None = None,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Use the frozen H&E state to retrieve a soft molecular reference state."""

    query = _matrix(queries, "queries")
    states = _matrix(bank_states, "bank_states")
    if query.shape[1] != states.shape[1]:
        raise ValueError("query and bank latent widths differ")
    weights = _weights(bank_weights, len(states))
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive")
    distance = (
        np.sum(np.square(query), axis=1)[:, None]
        + np.sum(np.square(states), axis=1)[None, :]
        - 2.0 * query @ states.T
    )
    logits = -np.maximum(distance, 0.0) / tau + np.log(weights / weights.sum())[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    attention = np.exp(logits)
    attention /= attention.sum(axis=1, keepdims=True)
    return attention @ states


def reference_only_state(bank_states: object, bank_weights: object | None, rows: int) -> np.ndarray:
    """Return the image-independent, frequency-weighted bank centroid."""

    states = _matrix(bank_states, "bank_states")
    weights = _weights(bank_weights, len(states))
    if rows <= 0:
        raise ValueError("rows must be positive")
    centroid = np.average(states, axis=0, weights=weights)
    return np.repeat(centroid[None, :], rows, axis=0)


def type_routed_reference_state(
    query_type_labels: Sequence[object],
    bank: PrototypeBank,
) -> tuple[np.ndarray, np.ndarray]:
    """Route by H&E-predicted type without using continuous image-state distances.

    This is the mandatory M2 diagnostic.  Queries whose predicted type is absent
    from the bank fall back to the natural bank centroid and are marked uncovered.
    """

    query_types = _strings(query_type_labels, "query_type_labels")
    output = np.empty((len(query_types), bank.states.shape[1]), dtype=np.float64)
    covered = np.zeros(len(query_types), dtype=bool)
    fallback = np.average(bank.states, axis=0, weights=bank.weights)
    for type_label in sorted(set(query_types.tolist())):
        query = query_types == type_label
        available = bank.type_labels == type_label
        if available.any():
            output[query] = np.average(
                bank.states[available], axis=0, weights=bank.weights[available]
            )
            covered[query] = True
        else:
            output[query] = fallback
    return output, covered


def equalize_bank_strata(
    bank_ids: Sequence[object],
    stratum_columns: Sequence[Sequence[object]],
    observation_ids: Sequence[object],
    *,
    seed: int = 17,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Equalize type/depth/quality strata across matched and comparator banks.

    Only strata represented in every bank are retained.  The same minimum count
    per stratum is selected deterministically in every bank, so any remaining
    matched advantage cannot be attributed to bank size or recorded composition.
    """

    banks = _strings(bank_ids, "bank_ids")
    rows = len(banks)
    observations = _strings(observation_ids, "observation_ids", rows)
    if not stratum_columns:
        raise ValueError("at least one equalization stratum column is required")
    columns = [
        _strings(values, f"stratum_columns[{index}]", rows)
        for index, values in enumerate(stratum_columns)
    ]
    bank_names = sorted(set(banks.tolist()))
    if len(bank_names) < 2:
        raise ValueError("bank equalization requires at least two banks")
    strata = np.asarray(
        ["\x1f".join(column[index] for column in columns) for index in range(rows)]
    )
    common = set(strata[banks == bank_names[0]].tolist())
    for bank_name in bank_names[1:]:
        common &= set(strata[banks == bank_name].tolist())
    selected: list[int] = []
    retained: dict[str, int] = {}
    for stratum in sorted(common):
        by_bank = {
            bank_name: np.flatnonzero((banks == bank_name) & (strata == stratum))
            for bank_name in bank_names
        }
        count = min(len(indices) for indices in by_bank.values())
        if count == 0:
            continue
        retained[stratum] = int(count)
        for bank_name, indices in by_bank.items():
            ordered = sorted(
                indices.tolist(),
                key=lambda index: (
                    _stable_hash((bank_name, stratum, observations[index]), seed),
                    observations[index],
                ),
            )
            selected.extend(ordered[:count])
    output = np.asarray(sorted(selected), dtype=np.int64)
    return output, {
        "schema": "heir.reference_bank_equalization.v1",
        "banks": bank_names,
        "common_strata": len(retained),
        "per_bank_rows": int(len(output) // len(bank_names)),
        "selected_rows": int(len(output)),
        "stratum_counts_per_bank": retained,
        "seed": int(seed),
        "query_outcomes_used": False,
    }


def reference_support_audit(
    queries: object,
    bank_states: object,
    bank_weights: object | None = None,
    *,
    maximum_distance: float,
    temperature: float = 1.0,
) -> Mapping[str, object]:
    """Measure reference distance, retrieval uncertainty, and support coverage."""

    query = _matrix(queries, "queries")
    states = _matrix(bank_states, "bank_states")
    if query.shape[1] != states.shape[1]:
        raise ValueError("query and bank latent widths differ")
    weights = _weights(bank_weights, len(states))
    threshold = float(maximum_distance)
    tau = float(temperature)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("maximum_distance must be finite and positive")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive")
    distance = (
        np.sum(np.square(query), axis=1)[:, None]
        + np.sum(np.square(states), axis=1)[None, :]
        - 2.0 * query @ states.T
    )
    distance = np.maximum(distance, 0.0)
    nearest = np.sqrt(distance.min(axis=1))
    logits = -distance / tau + np.log(weights / weights.sum())[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    attention = np.exp(logits)
    attention /= attention.sum(axis=1, keepdims=True)
    entropy = -np.sum(attention * np.log(np.maximum(attention, 1.0e-300)), axis=1)
    if len(states) > 1:
        entropy /= np.log(len(states))
    else:
        entropy[:] = 0.0
    supported = nearest <= threshold
    # Smoothly reduce the correction near the training-fitted support boundary.
    support_weight = np.clip(1.0 - nearest / threshold, 0.0, 1.0)
    return {
        "nearest_distance": nearest,
        "normalized_attention_entropy": entropy,
        "supported": supported,
        "support_weight": support_weight,
        "maximum_distance": threshold,
        "coverage": float(supported.mean()),
        "abstention": float((~supported).mean()),
    }


def adaptive_residual_fusion(
    image_state: object,
    reference_state: object,
    support_weight: object,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink correction to zero out of reference support and return row alphas."""

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
    if not np.isfinite(coefficient) or coefficient < 0 or coefficient > 0.5:
        raise ValueError("alpha must lie in [0, 0.5]")
    row_alpha = coefficient * support
    output = image + row_alpha[:, None] * (reference - image)
    return output, row_alpha


def within_type_reference_residuals(
    values: object,
    donor_ids: Sequence[object],
    type_labels: Sequence[object],
    reference_values: object,
    reference_donor_ids: Sequence[object],
    reference_type_labels: Sequence[object],
    *,
    reference_weights: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """Center outcomes on the matched scRNA donor/type mean.

    The key state endpoint is ``y_ST - mu_sc[d, type]``.  Missing donor/type
    combinations are deliberately left as ``NaN`` and marked uncovered rather
    than silently substituting a generic mean.  Callers must score only covered
    rows and report the returned coverage receipt.
    """

    matrix = _matrix(values, "values")
    donors = _strings(donor_ids, "donor_ids", len(matrix))
    types = _strings(type_labels, "type_labels", len(matrix))
    reference = _matrix(reference_values, "reference_values")
    if reference.shape[1] != matrix.shape[1]:
        raise ValueError("values and reference_values widths differ")
    reference_donors = _strings(
        reference_donor_ids, "reference_donor_ids", len(reference)
    )
    reference_types = _strings(
        reference_type_labels, "reference_type_labels", len(reference)
    )
    weights = _weights(reference_weights, len(reference))
    means = np.full(matrix.shape, np.nan, dtype=np.float64)
    covered = np.zeros(len(matrix), dtype=bool)
    covered_groups: list[str] = []
    missing_groups: list[str] = []
    for donor, type_label in sorted(set(zip(donors.tolist(), types.tolist()))):
        query = (donors == donor) & (types == type_label)
        available = (reference_donors == donor) & (reference_types == type_label)
        group = f"{donor}::{type_label}"
        if not available.any():
            missing_groups.append(group)
            continue
        means[query] = np.average(
            reference[available], axis=0, weights=weights[available]
        )
        covered[query] = True
        covered_groups.append(group)
    residuals = matrix - means
    return residuals, means, covered, {
        "schema": "heir.within_type_reference_residuals.v1",
        "definition": "value - matched_scRNA_donor_type_mean",
        "rows": int(len(matrix)),
        "covered_rows": int(covered.sum()),
        "coverage": float(covered.mean()),
        "covered_groups": covered_groups,
        "missing_groups": missing_groups,
        "missing_group_policy": "nan_exclude_and_report",
        "generic_fallback_used": False,
    }


def residual_fusion(image_state: object, reference_state: object, alpha: float) -> np.ndarray:
    """Apply one bounded correction anchored exactly to the H&E prediction."""

    image = _matrix(image_state, "image_state")
    reference = _matrix(reference_state, "reference_state")
    if reference.shape != image.shape:
        raise ValueError("image and reference states must align")
    coefficient = float(alpha)
    if not np.isfinite(coefficient) or coefficient < 0 or coefficient > 0.5:
        raise ValueError("alpha must lie in [0, 0.5] to keep H&E structurally central")
    if coefficient == 0.0:
        return image.copy()
    return image + coefficient * (reference - image)


def select_fusion_alpha(
    image_state: object,
    reference_state: object,
    truth: object,
    candidate_alphas: Sequence[float] = (0.0, 0.1, 0.25, 0.5),
    *,
    sample_weight: object | None = None,
) -> tuple[float, Mapping[str, float]]:
    """Select the smallest equally good bounded correction on validation rows."""

    image = _matrix(image_state, "image_state")
    reference = _matrix(reference_state, "reference_state")
    target = _matrix(truth, "truth")
    if image.shape != reference.shape or target.shape != image.shape:
        raise ValueError("fusion selection matrices must align")
    weights = _weights(sample_weight, len(image))
    losses: dict[str, float] = {}
    for alpha in sorted(set(float(value) for value in candidate_alphas)):
        prediction = residual_fusion(image, reference, alpha)
        row_loss = np.mean(np.square(target - prediction), axis=1)
        losses[f"{alpha:g}"] = float(np.average(row_loss, weights=weights))
    selected = min((loss, float(alpha)) for alpha, loss in losses.items())[1]
    return selected, losses


def build_matched_wrong_generic_banks(
    bank: PrototypeBank,
    query_donor: str,
    query_indication: str,
    donor_indications: Mapping[str, str],
) -> Mapping[str, object]:
    """Return outcome-free matched, all hard-wrong, and query-excluded banks."""

    donor = str(query_donor)
    indication = str(query_indication)
    if donor_indications.get(donor) != indication:
        raise ValueError("query donor indication is inconsistent")
    available = sorted(set(bank.donor_ids.tolist()))
    matched_indices = np.flatnonzero(bank.donor_ids == donor)
    if not len(matched_indices):
        raise ValueError("matched reference bank is absent")
    wrong_donors = [
        candidate
        for candidate in available
        if candidate != donor and donor_indications.get(candidate) == indication
    ]
    wrong = {
        candidate: np.flatnonzero(bank.donor_ids == candidate)
        for candidate in wrong_donors
    }
    generic_indices = np.flatnonzero(
        np.asarray(
            [
                value != donor and donor_indications.get(value) == indication
                for value in bank.donor_ids
            ],
            dtype=bool,
        )
    )
    if not wrong or not len(generic_indices):
        raise ValueError("query indication lacks a wrong/generic donor bank")
    return {
        "matched": matched_indices,
        "wrong": wrong,
        "generic": generic_indices,
        "wrong_donors": wrong_donors,
        "query_donor_excluded_from_generic": bool(
            not np.any(bank.donor_ids[generic_indices] == donor)
        ),
    }


def deterministic_group_derangement(
    group_ids: Sequence[object],
    observation_ids: Sequence[object],
    *,
    seed: int = 17,
) -> np.ndarray:
    """Return a deterministic no-fixed-point shuffle within each group."""

    groups = _strings(group_ids, "group_ids")
    observations = _strings(observation_ids, "observation_ids", len(groups))
    mapping = np.empty(len(groups), dtype=np.int64)
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            raise ValueError(f"derangement group has fewer than two rows: {group}")
        ordered = np.asarray(
            sorted(
                indices.tolist(),
                key=lambda index: (
                    _stable_hash((group, observations[index]), seed),
                    observations[index],
                ),
            ),
            dtype=np.int64,
        )
        shift = 1 + (_stable_hash((group,), seed) % (len(ordered) - 1))
        mapping[ordered] = np.roll(ordered, shift)
    if np.any(mapping == np.arange(len(mapping))):
        raise RuntimeError("derangement unexpectedly contains a fixed point")
    return mapping


def donor_section_macro_loss(
    truth: object,
    prediction: object,
    donor_ids: Sequence[object],
    section_ids: Sequence[object],
) -> Mapping[str, object]:
    """Compute the prespecified section-balanced, donor-equal standardized MSE."""

    target = _matrix(truth, "truth")
    predicted = _matrix(prediction, "prediction")
    if predicted.shape != target.shape:
        raise ValueError("truth and prediction must align")
    donors = _strings(donor_ids, "donor_ids", len(target))
    sections = _strings(section_ids, "section_ids", len(target))
    row_loss = np.mean(np.square(target - predicted), axis=1)
    donor_losses: dict[str, float] = {}
    section_losses: dict[str, float] = {}
    for donor in sorted(set(donors.tolist())):
        local_sections = sorted(set(sections[donors == donor].tolist()))
        values = []
        for section in local_sections:
            selected = (donors == donor) & (sections == section)
            loss = float(row_loss[selected].mean())
            section_losses[f"{donor}::{section}"] = loss
            values.append(loss)
        donor_losses[donor] = float(np.mean(values))
    return {
        "donor_section_macro_mse": float(np.mean(tuple(donor_losses.values()))),
        "donor_mse": donor_losses,
        "section_mse": section_losses,
        "rows": int(len(target)),
        "donors": int(len(donor_losses)),
        "sections": int(len(section_losses)),
    }


def donor_type_normalized_loss(
    truth: object,
    prediction: object,
    donor_ids: Sequence[object],
    type_labels: Sequence[object],
    *,
    section_ids: Sequence[object] | None = None,
    minimum_total_variance: float = 1.0e-10,
) -> Mapping[str, object]:
    """Compute donor/type-balanced SSE normalized by within-group truth variance.

    This implements the prespecified primary loss exactly at donor/type level.
    When sections are supplied, it additionally equal-weights sections within
    each donor/type using per-row SSE divided by that donor/type's per-row truth
    variance.  Zero-information donor/type groups are excluded and reported.
    """

    target = _matrix(truth, "truth")
    predicted = _matrix(prediction, "prediction")
    if predicted.shape != target.shape:
        raise ValueError("truth and prediction must align")
    donors = _strings(donor_ids, "donor_ids", len(target))
    types = _strings(type_labels, "type_labels", len(target))
    sections = (
        None
        if section_ids is None
        else _strings(section_ids, "section_ids", len(target))
    )
    threshold = float(minimum_total_variance)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("minimum_total_variance must be finite and non-negative")

    donor_type: dict[str, float] = {}
    donor_section_type: dict[str, float] = {}
    donor_values: dict[str, list[float]] = {}
    donor_section_values: dict[str, list[float]] = {}
    excluded: list[str] = []
    for donor in sorted(set(donors.tolist())):
        for type_label in sorted(set(types[donors == donor].tolist())):
            selected = (donors == donor) & (types == type_label)
            local_truth = target[selected]
            center = local_truth.mean(axis=0)
            denominator = float(np.square(local_truth - center).sum())
            key = f"{donor}::{type_label}"
            if denominator <= threshold:
                excluded.append(key)
                continue
            numerator = float(np.square(target[selected] - predicted[selected]).sum())
            loss = numerator / denominator
            donor_type[key] = loss
            donor_values.setdefault(donor, []).append(loss)
            if sections is None:
                continue
            rows = int(selected.sum())
            truth_variance_per_row = denominator / rows
            section_values: list[float] = []
            for section in sorted(set(sections[selected].tolist())):
                local = selected & (sections == section)
                section_sse_per_row = float(
                    np.square(target[local] - predicted[local]).sum() / local.sum()
                )
                section_loss = section_sse_per_row / truth_variance_per_row
                donor_section_type[f"{key}::{section}"] = section_loss
                section_values.append(section_loss)
            donor_section_values.setdefault(donor, []).append(
                float(np.mean(section_values))
            )

    if not donor_values:
        raise ValueError("no donor/type group has enough target variance to score")
    donor_loss = {
        donor: float(np.mean(values)) for donor, values in donor_values.items()
    }
    result: dict[str, object] = {
        "schema": "heir.donor_type_normalized_loss.v1",
        "donor_type_balanced_loss": float(np.mean(tuple(donor_loss.values()))),
        "donor_loss": donor_loss,
        "donor_type_loss": donor_type,
        "donors": int(len(donor_loss)),
        "scored_donor_type_groups": int(len(donor_type)),
        "excluded_zero_variance_groups": excluded,
        "normalization": "SSE / within_donor_type_truth_SST",
    }
    if sections is not None:
        section_donor_loss = {
            donor: float(np.mean(values))
            for donor, values in donor_section_values.items()
        }
        result.update(
            {
                "donor_section_type_balanced_loss": float(
                    np.mean(tuple(section_donor_loss.values()))
                ),
                "donor_section_type_loss": donor_section_type,
                "section_balanced_donor_loss": section_donor_loss,
                "section_normalization": (
                    "mean_section_SSE_per_row / donor_type_truth_SST_per_row"
                ),
            }
        )
    return result


def variance_preservation(
    truth: object,
    prediction: object,
    section_ids: Sequence[object],
) -> Mapping[str, object]:
    """Audit target-wise within-section variance and correlation preservation."""

    target = _matrix(truth, "truth")
    predicted = _matrix(prediction, "prediction")
    if target.shape != predicted.shape:
        raise ValueError("truth and prediction must align")
    sections = _strings(section_ids, "section_ids", len(target))
    ratios: list[float] = []
    correlations: list[float] = []
    for section in sorted(set(sections.tolist())):
        selected = sections == section
        if selected.sum() < 3:
            continue
        target_variance = np.var(target[selected], axis=0)
        predicted_variance = np.var(predicted[selected], axis=0)
        valid = target_variance > 1.0e-10
        ratios.extend((predicted_variance[valid] / target_variance[valid]).tolist())
        for index in np.flatnonzero(valid):
            correlation = np.corrcoef(target[selected, index], predicted[selected, index])[0, 1]
            if np.isfinite(correlation):
                correlations.append(float(correlation))
    return {
        "median_within_section_variance_ratio": (
            float(np.median(ratios)) if ratios else None
        ),
        "median_within_section_correlation": (
            float(np.median(correlations)) if correlations else None
        ),
        "evaluated_section_target_pairs": len(ratios),
    }


def evaluate_stage_gate(
    h_loss_by_donor: Mapping[str, float],
    fusion_loss_by_donor: Mapping[str, float],
    reference_loss_by_donor: Mapping[str, float],
    shuffled_loss_by_donor: Mapping[str, float],
    *,
    floor_loss_by_donor: Mapping[str, float] | None = None,
    median_variance_ratio: float | None = None,
    minimum_relative_gain: float = 0.05,
    minimum_positive_fraction: float = 0.70,
    minimum_variance_ratio: float = 0.50,
) -> Mapping[str, object]:
    """Apply the pre-outcome gate controlling optional iterative refinement."""

    donors = sorted(set(h_loss_by_donor) & set(fusion_loss_by_donor))
    missing_reference = set(donors) - set(reference_loss_by_donor)
    missing_shuffled = set(donors) - set(shuffled_loss_by_donor)
    if not donors or missing_reference or missing_shuffled:
        raise ValueError("stage-gate donor losses are incomplete")
    h = np.asarray([h_loss_by_donor[donor] for donor in donors], dtype=np.float64)
    fusion = np.asarray([fusion_loss_by_donor[donor] for donor in donors], dtype=np.float64)
    reference = np.asarray([reference_loss_by_donor[donor] for donor in donors], dtype=np.float64)
    shuffled = np.asarray([shuffled_loss_by_donor[donor] for donor in donors], dtype=np.float64)
    if not np.isfinite(np.concatenate((h, fusion, reference, shuffled))).all() or np.any(h <= 0):
        raise ValueError("stage-gate losses must be finite and H loss must be positive")
    relative = (h - fusion) / h
    criteria = {
        "relative_gain": bool(relative.mean() >= minimum_relative_gain),
        "positive_donor_fraction": bool(np.mean(relative > 0) >= minimum_positive_fraction),
        "beats_reference_only": bool(np.mean(fusion) < np.mean(reference)),
        "beats_shuffled_h": bool(np.mean(fusion) < np.mean(shuffled)),
        "variance_preserved": bool(
            median_variance_ratio is not None
            and np.isfinite(median_variance_ratio)
            and median_variance_ratio >= minimum_variance_ratio
        ),
    }
    if floor_loss_by_donor is not None:
        if set(donors) - set(floor_loss_by_donor):
            raise ValueError("measurement-floor donor losses are incomplete")
        floor = np.asarray([floor_loss_by_donor[donor] for donor in donors], dtype=np.float64)
        criteria["does_not_beat_measurement_floor"] = bool(np.mean(fusion) >= np.mean(floor))
    passed = bool(all(criteria.values()))
    return {
        "passed": passed,
        "decision": "allow_next_round" if passed else "iteration_not_run_failed_inner_gate",
        "criteria": criteria,
        "mean_relative_gain": float(relative.mean()),
        "positive_donor_fraction": float(np.mean(relative > 0)),
        "donors": donors,
        "thresholds": {
            "minimum_relative_gain": float(minimum_relative_gain),
            "minimum_positive_fraction": float(minimum_positive_fraction),
            "minimum_variance_ratio": float(minimum_variance_ratio),
        },
    }


def anchored_iteration(
    original_image_state: object,
    bank_states: object,
    bank_weights: object | None,
    alphas: Sequence[float],
    *,
    temperature: float = 1.0,
    maximum_rounds: int = 3,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Run at most three corrections, each anchored to the original H&E state."""

    original = _matrix(original_image_state, "original_image_state")
    states = _matrix(bank_states, "bank_states")
    coefficients = tuple(float(value) for value in alphas)
    if not coefficients or len(coefficients) > maximum_rounds or maximum_rounds > 3:
        raise ValueError("anchored refinement permits one to three rounds")
    current = original.copy()
    history: list[np.ndarray] = []
    for coefficient in coefficients:
        reference = soft_reference_state(
            current, states, bank_weights, temperature=temperature
        )
        current = residual_fusion(original, reference, coefficient)
        history.append(current.copy())
    return current, tuple(history)
