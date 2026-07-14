#!/usr/bin/env python3
"""Run the pragmatic, non-authorizing 15-donor HEST hypothesis test.

This analysis is deliberately retrospective.  It tests whether registered H&E
features predict within-type nucleus-overlapping RNA in held-out biological
donors, but it can never authorize H-CELL, H-INTRINSIC, or full HEIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

SCHEMA = "heir.hest_retrospective_report.v2"
SOURCE_SCHEMA = "heir.registered_observations_retrospective.v1"
SOURCE_SCOPE = "all_15_donors_retrospective_exposed_non_authorizing"
EXPECTED_DONORS = (
    "THD0008",
    "THD0011",
    "TILD117",
    "TILD175",
    "VUHD069",
    "VUHD116",
    "VUILD102",
    "VUILD105",
    "VUILD106",
    "VUILD107",
    "VUILD110",
    "VUILD115",
    "VUILD78",
    "VUILD91",
    "VUILD96",
)
EXPECTED_SECTIONS = (
    "NCBI856",
    "NCBI857",
    "NCBI858",
    "NCBI859",
    "NCBI860",
    "NCBI861",
    "NCBI864",
    "NCBI865",
    "NCBI866",
    "NCBI867",
    "NCBI870",
    "NCBI873",
    "NCBI875",
    "NCBI876",
    "NCBI879",
    "NCBI880",
    "NCBI881",
    "NCBI882",
    "NCBI883",
    "NCBI884",
)
EXPOSURE_RECEIPT_SHA256 = "7c9dd840968ec45f2bce4b2e9f6ae315688834da9a17acb94daf24060e24804b"
DATASET_REVISION = "7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec"
ENCODER_NAME = "MahmoodLab/UNI2-h"
ENCODER_REVISION = "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"
ENCODER_MANIFEST_SHA256 = "4ce7aad048abe8be99e6b1542d7eff88dc46e00fdf75057ca01728b21bc2f369"
ENCODER_CONFIG_SHA256 = "8b207fbff3e34884fd225b2d52e8ff51b728a1d0ac2fe8bb2b8db8011308ac98"
ENCODER_CHECKPOINT_SHA256 = "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"
REQUIRED_CROPS = (
    "crop_112um",
    "cell_mask_only",
    "nucleus_mask_only",
    "target_cell_removed_112um",
)
CONTROL_ORDER = (
    "reference_mean",
    "technical",
    "spatial",
    "stain_qc",
    "morphometry_density",
    "combined_nonimage",
)
MINIMUM_FULL_PERMUTATIONS = 100
MAXIMUM_PERMUTATIONS = 1000


def _log(message: str) -> None:
    print(message, flush=True)


def _array(archive: np.lib.npyio.NpzFile, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    raise ValueError("source is missing one of: " + ", ".join(names))


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_train, x_test = _standardize(x_train, x_test)
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ weights


def _lodo_predict(
    features: np.ndarray,
    truth: np.ndarray,
    donors: np.ndarray,
    alpha: float,
) -> np.ndarray:
    prediction = np.empty_like(truth, dtype=np.float64)
    for held_out in sorted(set(donors.tolist())):
        train = donors != held_out
        test = donors == held_out
        if not train.any() or not test.any():
            raise ValueError("leave-one-donor-out fold lacks train or test rows")
        prediction[test] = _ridge_predict(
            features[train], truth[train], features[test], alpha
        )
    return prediction


def _reference_residuals(
    y: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.full_like(y, np.nan, dtype=np.float64)
    evaluation = np.zeros(len(y), dtype=bool)
    for donor, section, label in sorted(set(zip(donors, sections, labels))):
        stratum = (donors == donor) & (sections == section) & (labels == label)
        reference = stratum & np.char.startswith(roles.astype(str), "reference")
        test = stratum & np.char.startswith(roles.astype(str), "evaluation")
        if reference.sum() >= minimum_support and test.sum() >= minimum_support:
            residuals[test] = y[test] - y[reference].mean(axis=0)
            evaluation[test] = True
    return residuals, evaluation


def _finite(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _safe_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


class _HierarchicalScorer:
    """Vectorized donor/type and donor/section/type scoring for repeated nulls."""

    def __init__(
        self,
        truth: np.ndarray,
        donors: np.ndarray,
        sections: np.ndarray,
        labels: np.ndarray,
        minimum_support: int,
    ) -> None:
        self.truth = np.asarray(truth, dtype=np.float64)
        self.donors = np.asarray(donors).astype(str)
        self.sections = np.asarray(sections).astype(str)
        self.labels = np.asarray(labels).astype(str)
        self.minimum_support = int(minimum_support)
        if self.truth.ndim != 2 or any(
            values.shape != (len(self.truth),)
            for values in (self.donors, self.sections, self.labels)
        ):
            raise ValueError("hierarchical scorer inputs are not row aligned")
        self._dt = self._group_spec((self.donors, self.labels))
        self._dst = self._group_spec((self.donors, self.sections, self.labels))

    def _group_spec(self, identities: Sequence[np.ndarray]) -> dict[str, object]:
        row_keys = list(zip(*(values.tolist() for values in identities)))
        keys = sorted(set(row_keys))
        lookup = {key: index for index, key in enumerate(keys)}
        codes = np.asarray([lookup[key] for key in row_keys], dtype=np.int64)
        support = np.bincount(codes, minlength=len(keys)).astype(np.int64)
        reference_denominator = np.bincount(
            codes,
            weights=np.einsum("ij,ij->i", self.truth, self.truth, optimize=True),
            minlength=len(keys),
        )
        centered_denominator = np.zeros(len(keys), dtype=np.float64)
        for index in range(len(keys)):
            selected = codes == index
            centered = self.truth[selected] - self.truth[selected].mean(axis=0)
            centered_denominator[index] = float(
                np.einsum("ij,ij->", centered, centered, optimize=True)
            )
        return {
            "keys": keys,
            "codes": codes,
            "support": support,
            "reference_denominator": reference_denominator,
            "centered_denominator": centered_denominator,
        }

    def _group_values(
        self, spec: Mapping[str, object], row_error: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        codes = np.asarray(spec["codes"], dtype=np.int64)
        support = np.asarray(spec["support"], dtype=np.int64)
        centered = np.asarray(spec["centered_denominator"], dtype=np.float64)
        reference = np.asarray(spec["reference_denominator"], dtype=np.float64)
        error = np.bincount(codes, weights=row_error, minlength=len(support))
        supported = support >= self.minimum_support
        r2 = np.full(len(support), np.nan, dtype=np.float64)
        reduction = np.full(len(support), np.nan, dtype=np.float64)
        valid_r2 = supported & (centered > 1e-12)
        valid_reference = supported & (reference > 1e-12)
        r2[valid_r2] = 1.0 - error[valid_r2] / centered[valid_r2]
        reduction[valid_reference] = 1.0 - error[valid_reference] / reference[valid_reference]
        return r2, reduction

    @staticmethod
    def _mean_by(
        keys: Sequence[str], values: np.ndarray
    ) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for key, value in zip(keys, values):
            if np.isfinite(value):
                grouped.setdefault(str(key), []).append(float(value))
        return {key: float(np.mean(rows)) for key, rows in sorted(grouped.items()) if rows}

    @staticmethod
    def _section_then_donor(
        keys: Sequence[tuple[str, str, str]], values: np.ndarray
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        by_section: dict[tuple[str, str], list[float]] = {}
        for (donor, section, _label), value in zip(keys, values):
            if np.isfinite(value):
                by_section.setdefault((donor, section), []).append(float(value))
        section_values: dict[str, dict[str, float]] = {}
        for (donor, section), rows in sorted(by_section.items()):
            section_values.setdefault(donor, {})[section] = float(np.mean(rows))
        donor_values = {
            donor: float(np.mean(list(sections.values())))
            for donor, sections in sorted(section_values.items())
            if sections
        }
        return donor_values, section_values

    def score(self, prediction: np.ndarray, *, detailed: bool) -> dict[str, object]:
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != self.truth.shape or not np.isfinite(prediction).all():
            raise ValueError("prediction is malformed or non-finite")
        difference = self.truth - prediction
        row_error = np.einsum("ij,ij->i", difference, difference, optimize=True)
        dt_r2, dt_reference = self._group_values(self._dt, row_error)
        dst_r2, dst_reference = self._group_values(self._dst, row_error)
        dt_keys = list(self._dt["keys"])
        dst_keys = list(self._dst["keys"])
        donor_r2 = self._mean_by([key[0] for key in dt_keys], dt_r2)
        donor_reference = self._mean_by([key[0] for key in dt_keys], dt_reference)
        donor_section_r2, section_r2 = self._section_then_donor(dst_keys, dst_r2)
        donor_section_reference, section_reference = self._section_then_donor(
            dst_keys, dst_reference
        )
        type_r2 = self._mean_by([key[1] for key in dt_keys], dt_r2)
        type_reference = self._mean_by([key[1] for key in dt_keys], dt_reference)
        per_donor = {
            donor: {
                "donor_type_r2": _finite(donor_r2.get(donor)),
                "donor_type_reference_error_reduction": _finite(
                    donor_reference.get(donor)
                ),
                "section_balanced_r2": _finite(donor_section_r2.get(donor)),
                "section_balanced_reference_error_reduction": _finite(
                    donor_section_reference.get(donor)
                ),
            }
            for donor in sorted(
                set(donor_r2)
                | set(donor_reference)
                | set(donor_section_r2)
                | set(donor_section_reference)
            )
        }
        result: dict[str, object] = {
            "donor_type_equal_r2": _safe_mean(donor_r2.values()),
            "donor_type_equal_reference_error_reduction": _safe_mean(
                donor_reference.values()
            ),
            "donor_section_type_equal_r2": _safe_mean(donor_section_r2.values()),
            "donor_section_type_equal_reference_error_reduction": _safe_mean(
                donor_section_reference.values()
            ),
            "per_donor": per_donor,
            "per_type": {
                label: {
                    "r2": _finite(type_r2.get(label)),
                    "reference_error_reduction": _finite(type_reference.get(label)),
                }
                for label in sorted(set(type_r2) | set(type_reference))
            },
            "per_donor_section": {
                donor: {
                    section: {
                        "r2": _finite(value),
                        "reference_error_reduction": _finite(
                            section_reference.get(donor, {}).get(section)
                        ),
                    }
                    for section, value in sorted(sections.items())
                }
                for donor, sections in sorted(section_r2.items())
            },
            "support": {
                "rows": int(len(self.truth)),
                "donors": int(len(set(self.donors.tolist()))),
                "sections": int(len(set(self.sections.tolist()))),
                "labels": int(len(set(self.labels.tolist()))),
                "evaluable_donor_type_strata": int(np.isfinite(dt_r2).sum()),
                "evaluable_donor_section_type_strata": int(np.isfinite(dst_r2).sum()),
            },
        }
        if detailed:
            dt_support = np.asarray(self._dt["support"], dtype=np.int64)
            dst_support = np.asarray(self._dst["support"], dtype=np.int64)
            result["donor_type_rows"] = [
                {
                    "donor_id": donor,
                    "type_id": label,
                    "support": int(dt_support[index]),
                    "r2": _finite(dt_r2[index]),
                    "reference_error_reduction": _finite(dt_reference[index]),
                }
                for index, (donor, label) in enumerate(dt_keys)
            ]
            result["donor_section_type_rows"] = [
                {
                    "donor_id": donor,
                    "section_id": section,
                    "type_id": label,
                    "support": int(dst_support[index]),
                    "r2": _finite(dst_r2[index]),
                    "reference_error_reduction": _finite(dst_reference[index]),
                }
                for index, (donor, section, label) in enumerate(dst_keys)
            ]
        return result


def _merge_named_parts(
    parts: Sequence[tuple[np.ndarray, Sequence[str]]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not parts:
        raise ValueError("control family needs at least one matrix")
    rows = len(parts[0][0])
    columns: list[np.ndarray] = []
    names: list[str] = []
    seen: set[str] = set()
    for matrix, local_names in parts:
        values = np.asarray(matrix)
        local_names = tuple(str(value) for value in local_names)
        if values.ndim != 2 or len(values) != rows or values.shape[1] != len(local_names):
            raise ValueError("control feature names and matrix disagree")
        for index, name in enumerate(local_names):
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
            columns.append(values[:, index])
    if not columns:
        raise ValueError("control family is empty after deduplication")
    return np.column_stack(columns).astype(np.float64), tuple(names)


def _load_control_families(
    archive: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    coordinate = _array(archive, ("coordinate_features",))
    coordinate_names = _array(archive, ("coordinate_feature_names",)).astype(str)
    technical = _array(archive, ("technical_covariates",))
    technical_names = _array(archive, ("technical_covariate_names",)).astype(str)
    stain = _array(archive, ("stain_quality_features", "stain_features"))
    stain_names = _array(
        archive, ("stain_quality_feature_names", "stain_feature_names")
    ).astype(str)
    nucleus = _array(
        archive, ("nucleus_geometry_features", "nuclear_morphometric_features")
    )
    nucleus_names = _array(
        archive,
        ("nucleus_geometry_feature_names", "nuclear_morphometric_feature_names"),
    ).astype(str)
    cell = _array(archive, ("cell_geometry_features", "cell_morphometric_features"))
    cell_names = _array(
        archive, ("cell_geometry_feature_names", "cell_morphometric_feature_names")
    ).astype(str)
    density = _array(archive, ("local_density_features",))
    density_names = _array(archive, ("local_density_feature_names",)).astype(str)
    density_set = set(density_names.tolist())
    spatial_indices = [
        index
        for index, name in enumerate(coordinate_names.tolist())
        if name not in density_set and "background" not in name
    ]
    spatial = coordinate[:, spatial_indices]
    spatial_names = coordinate_names[spatial_indices]
    raw = {
        "technical": ((technical, technical_names),),
        "spatial": ((spatial, spatial_names),),
        "stain_qc": ((technical, technical_names), (stain, stain_names)),
        "morphometry_density": (
            (nucleus, nucleus_names),
            (cell, cell_names),
            (density, density_names),
        ),
        "combined_nonimage": (
            (spatial, spatial_names),
            (technical, technical_names),
            (stain, stain_names),
            (nucleus, nucleus_names),
            (cell, cell_names),
            (density, density_names),
        ),
    }
    matrices: dict[str, np.ndarray] = {}
    names: dict[str, tuple[str, ...]] = {}
    for family, parts in raw.items():
        matrices[family], names[family] = _merge_named_parts(parts)
        if len(names[family]) != len(set(names[family])):
            raise AssertionError("control family contains duplicate feature identities")
    return matrices, names


def _mapping_sha256(mapping: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mapping).view(np.uint8)).hexdigest()


def _group_rows(*identities: np.ndarray) -> list[np.ndarray]:
    keys = list(zip(*(np.asarray(values).astype(str).tolist() for values in identities)))
    grouped: dict[tuple[str, ...], list[int]] = {}
    for index, key in enumerate(keys):
        grouped.setdefault(key, []).append(index)
    return [np.asarray(grouped[key], dtype=np.int64) for key in sorted(grouped)]


def _within_section_type_derangement(
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    result = np.arange(len(donors), dtype=np.int64)
    active_rows = 0
    active_groups = 0
    groups = _group_rows(donors, sections, labels, roles)
    for group in groups:
        if len(group) < 2:
            continue
        ordered = group[rng.permutation(len(group))]
        result[ordered] = np.roll(ordered, 1)
        active_rows += len(group)
        active_groups += 1
    if not (
        np.array_equal(np.asarray(donors).astype(str), np.asarray(donors).astype(str)[result])
        and np.array_equal(
            np.asarray(sections).astype(str), np.asarray(sections).astype(str)[result]
        )
        and np.array_equal(np.asarray(labels).astype(str), np.asarray(labels).astype(str)[result])
        and np.array_equal(np.asarray(roles).astype(str), np.asarray(roles).astype(str)[result])
    ):
        raise RuntimeError("local derangement crossed a frozen stratum")
    return result, {
        "seed": int(seed),
        "mapping_sha256": _mapping_sha256(result),
        "groups": len(groups),
        "active_groups": active_groups,
        "active_rows": active_rows,
        "changed_fraction": float(np.mean(result != np.arange(len(result)))),
    }


def _different_spatial_block_reassignment(
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    block_values = np.asarray(blocks).astype(str)
    result = np.arange(len(donors), dtype=np.int64)
    groups = _group_rows(donors, sections, labels, roles)
    feasible_groups = 0
    infeasible_groups = 0
    eligible_rows = 0
    for group in groups:
        local_blocks = sorted(set(block_values[group].tolist()))
        chunks = [group[block_values[group] == block] for block in local_blocks]
        maximum = max((len(chunk) for chunk in chunks), default=0)
        if len(chunks) < 2 or maximum > len(group) / 2:
            infeasible_groups += 1
            continue
        order = rng.permutation(len(chunks))
        ordered = np.concatenate(
            [chunks[index][rng.permutation(len(chunks[index]))] for index in order]
        )
        assigned = None
        for shift in range(maximum, len(group) - maximum + 1):
            candidate = np.roll(ordered, shift)
            if np.all(block_values[ordered] != block_values[candidate]):
                assigned = candidate
                break
        if assigned is None:
            infeasible_groups += 1
            continue
        result[ordered] = assigned
        feasible_groups += 1
        eligible_rows += len(group)
    if not (
        np.array_equal(np.asarray(donors).astype(str), np.asarray(donors).astype(str)[result])
        and np.array_equal(
            np.asarray(sections).astype(str), np.asarray(sections).astype(str)[result]
        )
        and np.array_equal(np.asarray(labels).astype(str), np.asarray(labels).astype(str)[result])
        and np.array_equal(np.asarray(roles).astype(str), np.asarray(roles).astype(str)[result])
    ):
        raise RuntimeError("block reassignment crossed a frozen stratum")
    changed = result != np.arange(len(result))
    if np.any(changed & (block_values == block_values[result])):
        raise RuntimeError("active block reassignment retained a spatial block")
    return result, {
        "seed": int(seed),
        "mapping_sha256": _mapping_sha256(result),
        "groups": len(groups),
        "feasible_groups": feasible_groups,
        "infeasible_groups": infeasible_groups,
        "eligible_rows": eligible_rows,
        "changed_fraction": float(np.mean(changed)),
        "cross_block_fraction": float(np.mean(block_values != block_values[result])),
    }


def _permutation_maps(
    kind: str,
    permutations: int,
    seed: int,
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    roles: np.ndarray,
    blocks: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, object]]:
    maps: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for index in range(permutations):
        local_seed = int(seed + index * 104729)
        if kind == "within_section_type_derangement":
            mapping, report = _within_section_type_derangement(
                donors, sections, labels, roles, seed=local_seed
            )
        elif kind == "different_spatial_block_reassignment":
            mapping, report = _different_spatial_block_reassignment(
                donors, sections, labels, roles, blocks, seed=local_seed
            )
        else:
            raise ValueError("unknown null family")
        maps.append(mapping)
        rows.append(report)
    hashes = [str(row["mapping_sha256"]) for row in rows]
    set_sha = hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()
    return maps, {
        "kind": kind,
        "permutations": permutations,
        "mapping_sha256s": hashes,
        "mapping_set_sha256": set_sha,
        "minimum_changed_fraction": float(
            min(float(row["changed_fraction"]) for row in rows)
        ),
        "minimum_cross_block_fraction": (
            float(min(float(row["cross_block_fraction"]) for row in rows))
            if kind == "different_spatial_block_reassignment"
            else None
        ),
        "minimum_eligible_rows": (
            int(min(int(row["eligible_rows"]) for row in rows))
            if kind == "different_spatial_block_reassignment"
            else int(min(int(row["active_rows"]) for row in rows))
        ),
    }


def _effect_values(
    model: Mapping[str, object], comparator: Mapping[str, object], key: str
) -> dict[str, float]:
    model_donors = model.get("per_donor", {})
    comparator_donors = comparator.get("per_donor", {})
    result = {}
    for donor in sorted(set(model_donors) & set(comparator_donors)):
        left = model_donors[donor].get(key)
        right = comparator_donors[donor].get(key)
        if left is not None and right is not None and np.isfinite(left) and np.isfinite(right):
            result[donor] = float(left - right)
    return result


def _summarize_effects(
    values: Mapping[str, float], largest_donor: str
) -> dict[str, object]:
    ordered = {key: float(value) for key, value in sorted(values.items())}
    array = np.asarray(list(ordered.values()), dtype=np.float64)
    positive = np.clip(array, 0.0, None)
    leave_one_out = {
        donor: _safe_mean(value for key, value in ordered.items() if key != donor)
        for donor in ordered
    }
    return {
        "per_donor": ordered,
        "paired_donors": len(ordered),
        "mean": _finite(array.mean()) if len(array) else None,
        "median": _finite(np.median(array)) if len(array) else None,
        "minimum": _finite(array.min()) if len(array) else None,
        "maximum": _finite(array.max()) if len(array) else None,
        "positive_donor_fraction": (
            float(np.mean(array > 0)) if len(array) else None
        ),
        "largest_positive_donor_share": (
            float(positive.max() / positive.sum()) if positive.sum() > 0 else None
        ),
        "aggregation_excluding_largest_evaluation_donor": _safe_mean(
            value for donor, value in ordered.items() if donor != largest_donor
        ),
        "largest_evaluation_donor": largest_donor,
        "leave_one_donor_aggregation": leave_one_out,
        "sensitivity_kind": "aggregation_only_not_training_refit",
    }


def _paired_effects(
    model: Mapping[str, object],
    comparator: Mapping[str, object],
    largest_donor: str,
) -> dict[str, object]:
    return {
        "donor_type_r2": _summarize_effects(
            _effect_values(model, comparator, "donor_type_r2"), largest_donor
        ),
        "donor_type_reference_error_reduction": _summarize_effects(
            _effect_values(
                model, comparator, "donor_type_reference_error_reduction"
            ),
            largest_donor,
        ),
        "section_balanced_r2": _summarize_effects(
            _effect_values(model, comparator, "section_balanced_r2"), largest_donor
        ),
        "section_balanced_reference_error_reduction": _summarize_effects(
            _effect_values(
                model, comparator, "section_balanced_reference_error_reduction"
            ),
            largest_donor,
        ),
    }


def _best_control(control_scores: Mapping[str, Mapping[str, object]]) -> str:
    best_name = CONTROL_ORDER[0]
    best_value = float("-inf")
    for name in CONTROL_ORDER:
        value = control_scores[name].get("donor_type_equal_r2")
        numeric = float(value) if value is not None and np.isfinite(value) else float("-inf")
        if numeric > best_value:
            best_name = name
            best_value = numeric
    return best_name


def _null_summary(
    observed_increment: float,
    values: Sequence[float],
    design: Mapping[str, object],
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "statistic": "donor_type_equal_r2_increment_over_combined_nonimage",
        "observed_increment": float(observed_increment),
        "values": [float(value) for value in array.tolist()],
        "mean": float(array.mean()),
        "matched_minus_null_mean": float(observed_increment - array.mean()),
        "empirical_p": float(
            (1 + np.count_nonzero(array >= observed_increment)) / (1 + len(array))
        ),
        "mapping_set_sha256": design["mapping_set_sha256"],
        "minimum_changed_fraction": design["minimum_changed_fraction"],
        "minimum_cross_block_fraction": design["minimum_cross_block_fraction"],
        "model_refit_for_every_permutation": True,
    }


def _quality_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    labels: np.ndarray,
    registration_strata: np.ndarray,
    locked_measurement: np.ndarray,
    minimum_support: int,
) -> dict[str, object]:
    masks = {
        "strict_locked_measurement": locked_measurement,
        "registration_best": registration_strata == "best",
        "registration_best_or_intermediate": np.isin(
            registration_strata, ("best", "intermediate")
        ),
        "registration_near_threshold": registration_strata == "near_threshold",
        "registration_failed": registration_strata == "failed",
    }
    result: dict[str, object] = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < minimum_support:
            result[name] = {
                "available": False,
                "rows": int(mask.sum()),
                "reason": "insufficient_rows",
            }
            continue
        scorer = _HierarchicalScorer(
            truth[mask], donors[mask], sections[mask], labels[mask], minimum_support
        )
        score = scorer.score(prediction[mask], detailed=False)
        available = score["donor_type_equal_r2"] is not None
        result[name] = {
            "available": available,
            "rows": int(mask.sum()),
            "score": score if available else None,
            "reason": None if available else "no_supported_donor_type_strata",
        }
    return result


def _criterion(
    name: str, value: float | None, operator: str, threshold: float
) -> dict[str, object]:
    passed = False
    if value is not None and np.isfinite(value):
        passed = bool(value > threshold) if operator == ">" else bool(value <= threshold)
    return {
        "name": name,
        "value": _finite(value),
        "operator": operator,
        "threshold": threshold,
        "pass": passed,
    }


def _evidence_summary(fine_result: Mapping[str, object]) -> dict[str, object]:
    arms = fine_result["arms"]
    contrasts = fine_result["crop_contrasts"]
    full = arms["crop_112um"]
    full_score = full["models"]["combined_plus_image"]
    full_effect = full["nested_increment_over_combined_nonimage"]["donor_type_r2"]
    h_cell_criteria = [
        _criterion(
            "positive_fine_type_donor_equal_r2",
            full_score["donor_type_equal_r2"],
            ">",
            0.0,
        ),
        _criterion(
            "positive_section_balanced_r2",
            full_score["donor_section_type_equal_r2"],
            ">",
            0.0,
        ),
        _criterion(
            "improves_spatial_reference",
            full_score["donor_type_equal_reference_error_reduction"],
            ">",
            0.0,
        ),
        _criterion("beats_combined_nonimage", full_effect["mean"], ">", 0.0),
        _criterion(
            "beats_best_nonimage",
            full["increment_over_best_nonimage"]["effects"]["donor_type_r2"]["mean"],
            ">",
            0.0,
        ),
        _criterion(
            "majority_positive_donor_increment",
            full_effect["positive_donor_fraction"],
            ">",
            0.5,
        ),
        _criterion(
            "within_section_type_null",
            full["nulls"]["within_section_type_derangement"]["empirical_p"],
            "<=",
            0.05,
        ),
        _criterion(
            "different_block_null",
            full["nulls"]["different_spatial_block_reassignment"]["empirical_p"],
            "<=",
            0.05,
        ),
    ]

    def intrinsic(kind: str, crop: str, contrast: str) -> dict[str, object]:
        arm = arms[crop]
        arm_score = arm["models"]["combined_plus_image"]
        arm_effect = arm["nested_increment_over_combined_nonimage"]["donor_type_r2"]
        paired = contrasts[contrast]["donor_type_r2"]
        strict = contrasts[contrast].get("strict_locked_measurement_donor_type_r2")
        criteria = [
            _criterion(
                "positive_fine_type_donor_equal_r2",
                arm_score["donor_type_equal_r2"],
                ">",
                0.0,
            ),
            _criterion(
                "improves_spatial_reference",
                arm_score["donor_type_equal_reference_error_reduction"],
                ">",
                0.0,
            ),
            _criterion("beats_combined_nonimage", arm_effect["mean"], ">", 0.0),
            _criterion(
                "beats_best_nonimage",
                arm["increment_over_best_nonimage"]["effects"]["donor_type_r2"][
                    "mean"
                ],
                ">",
                0.0,
            ),
            _criterion(
                "majority_positive_donor_increment",
                arm_effect["positive_donor_fraction"],
                ">",
                0.5,
            ),
            _criterion("beats_removed_context", paired["mean"], ">", 0.0),
            _criterion(
                "majority_positive_donor_contrast",
                paired["positive_donor_fraction"],
                ">",
                0.5,
            ),
            _criterion(
                "within_section_type_null",
                arm["nulls"]["within_section_type_derangement"]["empirical_p"],
                "<=",
                0.05,
            ),
            _criterion(
                "different_block_null",
                arm["nulls"]["different_spatial_block_reassignment"]["empirical_p"],
                "<=",
                0.05,
            ),
            _criterion("strict_measurement_contrast", strict, ">", 0.0),
        ]
        return {
            "hypothesis_id": kind,
            "evidence_status": (
                "exploratory_support"
                if all(row["pass"] for row in criteria)
                else "not_supported_or_indeterminate_in_this_analysis"
            ),
            "criteria": criteria,
            "authorizes": False,
            "interpretation_limit": (
                "Target-removed images contain a white cell-shaped hole; the contrast is not "
                "a pure context intervention."
            ),
        }

    return {
        "H-CELL-retrospective": {
            "hypothesis_id": "H-CELL-retrospective",
            "evidence_status": (
                "exploratory_support"
                if all(row["pass"] for row in h_cell_criteria)
                else "not_supported_or_indeterminate_in_this_analysis"
            ),
            "criteria": h_cell_criteria,
            "authorizes": False,
        },
        "H-INTRINSIC-cell-retrospective": intrinsic(
            "H-INTRINSIC-cell-retrospective",
            "cell_mask_only",
            "cell_mask_minus_target_removed",
        ),
        "H-INTRINSIC-nucleus-retrospective": intrinsic(
            "H-INTRINSIC-nucleus-retrospective",
            "nucleus_mask_only",
            "nucleus_mask_minus_target_removed",
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def benchmark(
    source: Path,
    output: Path,
    alpha: float,
    permutations: int,
    seed: int,
    projection_dimension: int = 96,
    minimum_support: int = 5,
) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if (
        source == output
        or alpha <= 0
        or not 1 <= permutations <= MAXIMUM_PERMUTATIONS
        or projection_dimension < 1
        or minimum_support < 2
    ):
        raise ValueError("invalid benchmark source, output, or numeric setting")
    _log(f"HEST retrospective benchmark: loading {source}")
    with np.load(source, allow_pickle=False) as archive:
        def scalar(name: str) -> object:
            return np.asarray(_array(archive, (name,))).reshape(()).item()

        identity = {
            "study_stage": str(scalar("study_stage")),
            "schema": str(scalar("schema_version")),
            "scope": str(scalar("source_scope")),
            "status": str(scalar("analysis_status")),
            "authorizes_h_cell": bool(scalar("authorizes_h_cell")),
            "authorizes_h_intrinsic": bool(scalar("authorizes_h_intrinsic")),
            "authorizes_full_heir": bool(scalar("authorizes_full_heir")),
            "exposure_receipt": str(scalar("prior_outcome_exposure_receipt_sha256")),
            "cohort_id": str(scalar("cohort_id")),
            "cohort_release": str(scalar("cohort_release")),
            "encoder_name": str(scalar("encoder_name")),
            "encoder_revision": str(scalar("encoder_revision")),
            "encoder_manifest": str(scalar("encoder_manifest_sha256")),
            "encoder_config": str(scalar("feature_config_sha256")),
            "encoder_checkpoint": str(scalar("feature_checkpoint_sha256")),
        }
        expected_identity = {
            "study_stage": "retrospective_exposed",
            "schema": SOURCE_SCHEMA,
            "scope": SOURCE_SCOPE,
            "status": "retrospective_exposed_non_authorizing",
            "authorizes_h_cell": False,
            "authorizes_h_intrinsic": False,
            "authorizes_full_heir": False,
            "exposure_receipt": EXPOSURE_RECEIPT_SHA256,
            "cohort_id": "HEST",
            "cohort_release": DATASET_REVISION,
            "encoder_name": ENCODER_NAME,
            "encoder_revision": ENCODER_REVISION,
            "encoder_manifest": ENCODER_MANIFEST_SHA256,
            "encoder_config": ENCODER_CONFIG_SHA256,
            "encoder_checkpoint": ENCODER_CHECKPOINT_SHA256,
        }
        if identity != expected_identity:
            raise ValueError("source is not the explicitly exposed HEST artifact")
        donors = _array(archive, ("donor_ids", "donor_id")).astype(str)
        sections = _array(archive, ("section_ids", "section_id")).astype(str)
        fine = _array(archive, ("fine_type_ids", "fine_type")).astype(str)
        broad_names = _array(archive, ("broad_type_names",)).astype(str)
        broad_index = _array(
            archive, ("broad_type_labels", "broad_type_label")
        ).astype(int)
        broad = broad_names[broad_index]
        roles = _array(archive, ("pool_roles", "pool_role")).astype(str)
        blocks = _array(archive, ("block_ids", "block_id")).astype(str)
        registration_strata = _array(archive, ("registration_quality_strata",)).astype(str)
        locked_measurement = _array(archive, ("locked_measurement_qc_pass",)).astype(bool)
        y = _array(archive, ("nucleus_molecular_targets",)).astype(np.float64)
        crop_ids = tuple(_array(archive, ("crop_ids",)).astype(str).tolist())
        images = _array(
            archive, ("image_features_by_crop_and_encoder", "image_features")
        )
        control_families, control_names = _load_control_families(archive)
    if crop_ids != REQUIRED_CROPS:
        raise ValueError("source does not contain exactly the ordered retrospective crop arms")
    if set(donors.tolist()) != set(EXPECTED_DONORS):
        raise ValueError("retrospective HEST analysis requires all 15 biological donors")
    if set(sections.tolist()) != set(EXPECTED_SECTIONS):
        raise ValueError("retrospective HEST analysis requires the exact 20 HEST sections")
    row_arrays = (sections, fine, broad, roles, blocks, registration_strata, locked_measurement)
    if any(len(values) != len(donors) for values in row_arrays) or len(y) != len(donors):
        raise ValueError("source row arrays disagree")
    if images.ndim != 3 or images.shape[:2] != (len(donors), len(REQUIRED_CROPS)):
        raise ValueError("retrospective image feature tensor is malformed")
    if not np.isfinite(y).all() or any(
        not np.isfinite(values).all() for values in control_families.values()
    ):
        raise ValueError("source targets or controls are non-finite")
    projection_width = min(projection_dimension, int(images.shape[2]))
    projection_rng = np.random.default_rng(seed + 1)
    projection = projection_rng.choice(
        np.asarray([-1.0, 1.0]), size=(images.shape[2], projection_width)
    ) / np.sqrt(projection_width)
    projected_images: dict[str, np.ndarray] = {}
    for crop_index, crop_id in enumerate(crop_ids):
        _log(f"HEST projection: crop={crop_id}")
        projected_images[crop_id] = (
            np.asarray(images[:, crop_index], dtype=np.float64) @ projection
        )
    del images
    results: dict[str, object] = {}
    for resolution_index, (resolution, labels) in enumerate(
        (("broad_lineage", broad), ("fine_type", fine))
    ):
        residuals, evaluation = _reference_residuals(
            y, donors, sections, labels, roles, minimum_support
        )
        if set(donors.tolist()) != set(donors[evaluation].tolist()):
            raise ValueError(f"{resolution} lacks reference/evaluation support for every donor")
        local_truth = residuals[evaluation]
        local_donors = donors[evaluation]
        local_sections = sections[evaluation]
        local_labels = labels[evaluation]
        local_roles = roles[evaluation]
        local_blocks = blocks[evaluation]
        local_registration = registration_strata[evaluation]
        local_locked = locked_measurement[evaluation]
        local_control_families = {
            family: values[evaluation] for family, values in control_families.items()
        }
        donor_counts = {
            donor: int(np.count_nonzero(local_donors == donor))
            for donor in sorted(set(local_donors.tolist()))
        }
        largest_donor = sorted(donor_counts, key=lambda donor: (-donor_counts[donor], donor))[0]
        scorer = _HierarchicalScorer(
            local_truth,
            local_donors,
            local_sections,
            local_labels,
            minimum_support,
        )
        _log(
            f"HEST resolution={resolution}: evaluation_rows={len(local_truth)} controls"
        )
        reference_prediction = np.zeros_like(local_truth)
        control_scores: dict[str, dict[str, object]] = {
            "reference_mean": scorer.score(reference_prediction, detailed=True)
        }
        del reference_prediction
        for family in CONTROL_ORDER[1:]:
            prediction = _lodo_predict(
                local_control_families[family],
                local_truth,
                local_donors,
                alpha,
            )
            control_scores[family] = scorer.score(prediction, detailed=True)
            del prediction
        best_control = _best_control(control_scores)
        null_maps: dict[str, list[np.ndarray]] = {}
        null_designs: dict[str, dict[str, object]] = {}
        for null_index, kind in enumerate(
            (
                "within_section_type_derangement",
                "different_spatial_block_reassignment",
            )
        ):
            null_maps[kind], null_designs[kind] = _permutation_maps(
                kind,
                permutations,
                seed + resolution_index * 10_000_019 + null_index * 1_000_003,
                local_donors,
                local_sections,
                local_labels,
                local_roles,
                local_blocks,
            )
        arms: dict[str, object] = {}
        for crop_id in crop_ids:
            _log(f"HEST resolution={resolution} arm={crop_id}: observed models")
            image = projected_images[crop_id][evaluation]
            image_only_prediction = _lodo_predict(image, local_truth, local_donors, alpha)
            combined_prediction = _lodo_predict(
                np.column_stack((local_control_families["combined_nonimage"], image)),
                local_truth,
                local_donors,
                alpha,
            )
            image_only_score = scorer.score(image_only_prediction, detailed=False)
            combined_score = scorer.score(combined_prediction, detailed=True)
            nested_effect = _paired_effects(
                combined_score, control_scores["combined_nonimage"], largest_donor
            )
            best_effect = _paired_effects(
                combined_score, control_scores[best_control], largest_donor
            )
            observed_increment = float(
                combined_score["donor_type_equal_r2"]
                - control_scores["combined_nonimage"]["donor_type_equal_r2"]
            )
            null_results: dict[str, object] = {}
            for kind in null_maps:
                values = []
                for permutation_index, mapping in enumerate(null_maps[kind], start=1):
                    prediction = _lodo_predict(
                        np.column_stack(
                            (
                                local_control_families["combined_nonimage"],
                                image[mapping],
                            )
                        ),
                        local_truth,
                        local_donors,
                        alpha,
                    )
                    null_score = scorer.score(prediction, detailed=False)
                    values.append(
                        float(
                            null_score["donor_type_equal_r2"]
                            - control_scores["combined_nonimage"]["donor_type_equal_r2"]
                        )
                    )
                    if permutation_index % 10 == 0 or permutation_index == permutations:
                        _log(
                            "HEST null: resolution=%s arm=%s kind=%s %d/%d"
                            % (
                                resolution,
                                crop_id,
                                kind,
                                permutation_index,
                                permutations,
                            )
                        )
                null_results[kind] = _null_summary(
                    observed_increment, values, null_designs[kind]
                )
            arms[crop_id] = {
                "crop_role": (
                    "context_plus_target_cell_silhouette_white_hole"
                    if crop_id == "target_cell_removed_112um"
                    else "registered_crop_arm"
                ),
                "models": {
                    "image_only": image_only_score,
                    "combined_plus_image": combined_score,
                },
                "nested_increment_over_combined_nonimage": nested_effect,
                "increment_over_best_nonimage": {
                    "best_nonimage_control": best_control,
                    "effects": best_effect,
                },
                "nulls": null_results,
                "quality_sensitivity": _quality_scores(
                    local_truth,
                    combined_prediction,
                    local_donors,
                    local_sections,
                    local_labels,
                    local_registration,
                    local_locked,
                    minimum_support,
                ),
            }
            del image_only_prediction, combined_prediction
        contrast_specs = {
            "full_context_minus_target_removed": (
                "crop_112um",
                "target_cell_removed_112um",
            ),
            "cell_mask_minus_target_removed": (
                "cell_mask_only",
                "target_cell_removed_112um",
            ),
            "nucleus_mask_minus_target_removed": (
                "nucleus_mask_only",
                "target_cell_removed_112um",
            ),
        }
        contrasts: dict[str, object] = {}
        for name, (left, right) in contrast_specs.items():
            left_score = arms[left]["models"]["combined_plus_image"]
            right_score = arms[right]["models"]["combined_plus_image"]
            effects = _paired_effects(left_score, right_score, largest_donor)
            left_strict = arms[left]["quality_sensitivity"]["strict_locked_measurement"]
            right_strict = arms[right]["quality_sensitivity"]["strict_locked_measurement"]
            strict_difference = None
            if left_strict["available"] and right_strict["available"]:
                strict_difference = float(
                    left_strict["score"]["donor_type_equal_r2"]
                    - right_strict["score"]["donor_type_equal_r2"]
                )
            contrasts[name] = {
                **effects,
                "strict_locked_measurement_donor_type_r2": strict_difference,
                "paired_crop_predictions": True,
            }
        results[resolution] = {
            "evaluation_cells": int(evaluation.sum()),
            "evaluation_donor_counts": donor_counts,
            "largest_evaluation_donor": largest_donor,
            "control_feature_registry": {
                "deduplicated": True,
                **{
                    family: {
                        "width": len(control_names[family]),
                        "feature_names": list(control_names[family]),
                    }
                    for family in CONTROL_ORDER[1:]
                },
            },
            "controls": control_scores,
            "best_nonimage_control": best_control,
            "null_designs": null_designs,
            "arms": arms,
            "crop_contrasts": contrasts,
        }
    _log("HEST retrospective benchmark: hashing source receipt")
    report = {
        "schema": SCHEMA,
        "analysis_status": "retrospective_exposed_non_authorizing",
        "study_stage": "retrospective_exposed",
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_full_heir": False,
        "hypotheses": _evidence_summary(results["fine_type"]),
        "limitations": [
            "RNA-derived final_CT/final_lineage labels may remain ontology-dependent.",
            "All GSE250346/HEST outcomes were previously exposed.",
            "This is internal association evidence, not external or prospective confirmation.",
            "The pooled multi-type ridge can miss type-specific morphology mappings.",
            "The fixed 96-dimensional outcome-free projection can attenuate UNI2-h signal.",
            "Target-removed images retain a white cell-shaped silhouette and are not pure context.",
            "Strict registration/measurement sensitivity limits intrinsic interpretation.",
        ],
        "source": str(source),
        "source_sha256": _sha256(source),
        "donors": sorted(set(donors.tolist())),
        "sections": sorted(set(sections.tolist())),
        "donor_count": len(set(donors.tolist())),
        "section_count": len(set(sections.tolist())),
        "folding": "leave_one_biological_donor_out",
        "reference": "same_donor_section_type_spatial_pool_mean",
        "primary_target": "nucleus_overlapping_xenium_transcripts_log1p_cpm_10000",
        "alpha": alpha,
        "permutations_per_null_family": permutations,
        "permutation_scope": (
            "full_retrospective" if permutations >= MINIMUM_FULL_PERMUTATIONS else "smoke_only"
        ),
        "seed": seed,
        "minimum_support": minimum_support,
        "image_projection": "fixed_outcome_free_rademacher",
        "image_projection_dimension": projection_width,
        "numeric_backend": "numpy_float64",
        "results": results,
    }
    _write_json(output, report)
    _log(f"HEST retrospective benchmark complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--permutations", type=int, default=MINIMUM_FULL_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--projection-dimension", type=int, default=96)
    parser.add_argument("--minimum-support", type=int, default=5)
    args = parser.parse_args()
    benchmark(
        args.source,
        args.output,
        args.alpha,
        args.permutations,
        args.seed,
        args.projection_dimension,
        args.minimum_support,
    )


if __name__ == "__main__":
    main()
