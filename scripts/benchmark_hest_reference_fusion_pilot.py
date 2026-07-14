#!/usr/bin/env python3
"""Run the non-authorizing HEST spatial-reference fusion pilot.

This experiment uses spatially disjoint Xenium cells as a molecular reference.
It is deliberately labelled a retrospective spatial-reference pilot: HEST has
no independent matched sc/snRNA-seq bank, so this runner cannot validate a
matched-scRNA or personalized-reference claim.

All model selection is nested inside leave-one-donor-out (LODO) evaluation.
The frozen UNI2-h ridge prediction is the centre of the one-step fusion:

    H_plus_R = H + alpha * (soft_reference(H) - H), alpha in [0, 0.5].

The runner never launches automatically when imported.  The real 1.8-GB source
is loaded only from ``main`` or an explicit ``run_pilot`` call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.evaluation.hest_measurement import (
    feature_reliability_report,
    normalize_halves,
    ordered_program_scores,
    reference_residualize_halves,
)
from heir.evaluation.hest_nested_ridge import (
    donor_section_type_row_weights,
    fit_weighted_ridge_grid,
    grouped_donor_folds,
)
from heir.evaluation.reference_fusion import (
    PrototypeBank,
    build_matched_wrong_generic_banks,
    build_reference_prototypes,
    deterministic_group_derangement,
    fit_target_basis,
    reference_only_state,
    residual_fusion,
    soft_reference_state,
)

SCHEMA = "heir.hest_reference_fusion_pilot.v1"
SOURCE_SCHEMA = "heir.registered_observations_retrospective.v1"
REGISTERED_SOURCE_SHA256 = (
    "57b77c7be2e30026a2da9ba0f9d5b205cf630f5d138942db6366e15cae2ef7a3"
)
DEFAULT_SOURCE = Path("/mnt/seagate/HEIR_runs/hest_retrospective/source.npz")
DEFAULT_OUTPUT = Path("/mnt/seagate/HEIR_runs/hest_reference_fusion_pilot")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
FUSION_ALPHAS = (0.0, 0.1, 0.25, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _array(archive: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    raise ValueError("source is missing one of: " + ", ".join(names))


def _optional_array(
    archive: np.lib.npyio.NpzFile, *names: str
) -> Optional[np.ndarray]:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    return None


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    return np.asarray(_array(archive, name)).reshape(()).item()


def _identifiers(values: object, name: str, rows: int) -> np.ndarray:
    result = np.asarray(values).astype(str)
    if result.shape != (rows,) or any(not value for value in result.tolist()):
        raise ValueError(f"{name} must be a non-empty row-aligned identifier vector")
    return result


def _matrix(values: object, name: str, rows: int) -> np.ndarray:
    result = np.asarray(values)
    if (
        result.ndim != 2
        or result.shape[0] != rows
        or result.shape[1] == 0
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be a finite row-aligned matrix")
    return result


def _donor_values(donors: np.ndarray, values: np.ndarray, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for donor in sorted(set(donors.tolist())):
        local = sorted(set(values[donors == donor].tolist()))
        if len(local) != 1:
            raise ValueError(f"{name} must have exactly one value per donor")
        result[donor] = local[0]
    return result


@dataclass(frozen=True)
class PilotSource:
    """In-memory arrays needed by the spatial-reference pilot."""

    path: Path
    sha256: str
    observation_ids: np.ndarray
    donors: np.ndarray
    sections: np.ndarray
    fine_types: np.ndarray
    indications: np.ndarray
    donor_indications: Mapping[str, str]
    roles: np.ndarray
    images: np.ndarray
    blank_images: Optional[np.ndarray]
    blank_status: Mapping[str, object]
    coordinates: np.ndarray
    program_names: tuple[str, ...]
    program_total: np.ndarray
    program_half_a: np.ndarray
    program_half_b: np.ndarray
    evaluation_mask: np.ndarray
    reference_mask: np.ndarray
    reference_split_id: str
    support_strata: tuple[Mapping[str, object], ...]


def _reference_mask_from_strata(
    donors: np.ndarray,
    sections: np.ndarray,
    fine_types: np.ndarray,
    roles: np.ndarray,
    strata: Sequence[Mapping[str, object]],
) -> np.ndarray:
    result = np.zeros(len(donors), dtype=bool)
    is_reference = np.char.startswith(np.char.lower(roles.astype(str)), "reference")
    for record in strata:
        if not bool(record["supported"]):
            continue
        result |= (
            is_reference
            & (donors == str(record["donor_id"]))
            & (sections == str(record["section_id"]))
            & (fine_types == str(record["fine_type_id"]))
        )
    return result


def _blank_features(
    archive: np.lib.npyio.NpzFile,
    rows: int,
    image_width: int,
) -> tuple[Optional[np.ndarray], Mapping[str, object]]:
    for name in ("blank_patch_features", "blank_image_features"):
        if name not in archive.files:
            continue
        values = np.asarray(archive[name])
        if values.ndim == 3 and values.shape[0] == rows:
            values = values[:, 0, :]
        if values.shape == (rows, image_width) and np.isfinite(values).all():
            return values.astype(np.float32, copy=False), {
                "available": True,
                "source_key": name,
                "definition": "source-provided frozen-encoder blank-patch features",
            }
        return None, {
            "available": False,
            "source_key": name,
            "reason": "source blank feature shape does not match the frozen H&E feature space",
        }
    if "blank_patch" in archive.files:
        return None, {
            "available": False,
            "source_key": "blank_patch",
            "reason": "raw blank_patch exists but frozen encoder features are absent",
        }
    return None, {
        "available": False,
        "source_key": None,
        "reason": "source contains no blank_patch or frozen blank-patch features",
    }


def load_source(
    path: Path,
    *,
    reference_split_id: str = "primary",
    minimum_support: int = 20,
    enforce_registered_hash: bool = True,
) -> PilotSource:
    """Load and fail-closed validate the retrospective registered source."""

    source_path = path.expanduser().resolve()
    source_hash = _sha256(source_path)
    if enforce_registered_hash and source_hash != REGISTERED_SOURCE_SHA256:
        raise ValueError("source does not match the registered retrospective HEST receipt")
    with np.load(source_path, allow_pickle=False) as archive:
        identity = {
            "schema": str(_scalar(archive, "schema_version")),
            "stage": str(_scalar(archive, "study_stage")),
            "status": str(_scalar(archive, "analysis_status")),
            "authorizes_h_cell": bool(_scalar(archive, "authorizes_h_cell")),
            "authorizes_h_intrinsic": bool(_scalar(archive, "authorizes_h_intrinsic")),
            "authorizes_full_heir": bool(_scalar(archive, "authorizes_full_heir")),
        }
        expected = {
            "schema": SOURCE_SCHEMA,
            "stage": "retrospective_exposed",
            "status": "retrospective_exposed_non_authorizing",
            "authorizes_h_cell": False,
            "authorizes_h_intrinsic": False,
            "authorizes_full_heir": False,
        }
        if identity != expected:
            raise ValueError("source is not the non-authorizing retrospective HEST artifact")

        donors_raw = _array(archive, "donor_ids", "donor_id")
        rows = len(donors_raw)
        donors = _identifiers(donors_raw, "donors", rows)
        sections = _identifiers(
            _array(archive, "section_ids", "section_id"), "sections", rows
        )
        fine_types = _identifiers(
            _array(archive, "fine_type_ids", "fine_type", "type_labels"),
            "fine_types",
            rows,
        )
        indications = _identifiers(
            _array(archive, "disease_statuses", "disease_state"),
            "indications",
            rows,
        )
        observation_ids = _identifiers(
            _array(archive, "observation_ids", "observation_id", "cell_id"),
            "observation_ids",
            rows,
        )
        if len(set(observation_ids.tolist())) != rows:
            raise ValueError("observation IDs must be unique")

        split_names = tuple(
            _array(archive, "reference_split_ids").astype(str).tolist()
        )
        if reference_split_id not in split_names:
            raise ValueError("requested reference split is absent")
        split_index = split_names.index(reference_split_id)
        roles_by_split = np.asarray(_array(archive, "pool_roles_by_split"))
        if roles_by_split.shape != (rows, len(split_names)):
            raise ValueError("pool_roles_by_split is not row/split aligned")
        roles = _identifiers(
            roles_by_split[:, split_index], "pool roles", rows
        )

        crop_ids = tuple(_array(archive, "crop_ids").astype(str).tolist())
        primary_crop = str(_scalar(archive, "primary_crop_id"))
        if primary_crop not in crop_ids:
            raise ValueError("primary crop is absent from image feature tensor")
        # ``frozen_features`` is the receipt-bound primary crop and avoids
        # inflating the four-crop 887-MB tensor when the 222-MB primary matrix
        # is already present.
        if "frozen_features" in archive.files:
            image_tensor = np.asarray(archive["frozen_features"])
        else:
            image_tensor = np.asarray(
                _array(
                    archive,
                    "image_features_by_crop_and_encoder",
                    "image_features",
                )
            )
        if image_tensor.ndim == 3:
            images = image_tensor[:, crop_ids.index(primary_crop), :]
        elif image_tensor.ndim == 2:
            images = image_tensor
        else:
            raise ValueError("image features must be a row-feature or row-crop-feature tensor")
        images = _matrix(images, "primary H&E features", rows).astype(
            np.float32, copy=False
        )
        blank_images, blank_status = _blank_features(archive, rows, images.shape[1])
        coordinates = _matrix(
            _array(archive, "coordinate_features", "spatial_features"),
            "coordinate features",
            rows,
        ).astype(np.float32, copy=False)

        program_names = tuple(_array(archive, "program_names").astype(str).tolist())
        membership = np.asarray(_array(archive, "program_gene_membership"))
        total = ordered_program_scores(
            _array(archive, "normalized_nucleus_targets", "nucleus_molecular_targets"),
            program_names,
            membership,
        )
        half_a, half_b = normalize_halves(
            _array(archive, "nucleus_target_counts_half_a"),
            _array(archive, "nucleus_target_counts_half_b"),
            library_sizes_half_a=_array(archive, "nucleus_library_size_half_a"),
            library_sizes_half_b=_array(archive, "nucleus_library_size_half_b"),
        )
        program_half_a = ordered_program_scores(half_a, program_names, membership)
        program_half_b = ordered_program_scores(half_b, program_names, membership)
        residuals = reference_residualize_halves(
            program_half_a,
            program_half_b,
            donors,
            sections,
            fine_types,
            roles,
            minimum_support=minimum_support,
        )
        reference_mask = _reference_mask_from_strata(
            donors, sections, fine_types, roles, residuals.strata
        )
        if not residuals.evaluation_mask.any() or not reference_mask.any():
            raise ValueError("reference split has no supported disjoint query/reference cells")

    return PilotSource(
        path=source_path,
        sha256=source_hash,
        observation_ids=observation_ids,
        donors=donors,
        sections=sections,
        fine_types=fine_types,
        indications=indications,
        donor_indications=_donor_values(donors, indications, "disease status"),
        roles=roles,
        images=images,
        blank_images=blank_images,
        blank_status=blank_status,
        coordinates=coordinates,
        program_names=program_names,
        program_total=total,
        program_half_a=program_half_a,
        program_half_b=program_half_b,
        evaluation_mask=residuals.evaluation_mask,
        reference_mask=reference_mask,
        reference_split_id=reference_split_id,
        support_strata=residuals.strata,
    )


def _training_reliable_programs(
    source: PilotSource,
    training_mask: np.ndarray,
    *,
    minimum_reliability: float,
    minimum_rows: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Select targets using split halves from outer-training query rows only."""

    report = feature_reliability_report(
        source.program_half_a,
        source.program_half_b,
        source.program_names,
        source.donors,
        source.fine_types,
        evaluation_mask=training_mask,
        minimum_rows=minimum_rows,
    )
    selected = []
    for index, name in enumerate(source.program_names):
        value = report["donor_type_macro"]["features"][name][
            "median_spearman_brown_reliability"
        ]
        if value is not None and float(value) >= minimum_reliability:
            selected.append(index)
    return np.asarray(selected, dtype=np.int64), report


def _row_weights(source: PilotSource, selected: np.ndarray) -> np.ndarray:
    return donor_section_type_row_weights(
        source.donors[selected],
        source.sections[selected],
        source.fine_types[selected],
    )


def _macro_loss_from_rows(
    row_loss: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    types: np.ndarray,
) -> Mapping[str, object]:
    """Average cells, types, sections, and donors in that order."""

    values = np.asarray(row_loss, dtype=np.float64)
    if values.shape != (len(donors),) or not np.isfinite(values).all():
        raise ValueError("row losses must be finite and identity aligned")
    donor_losses: dict[str, float] = {}
    donor_type_losses: dict[str, float] = {}
    for donor in sorted(set(donors.tolist())):
        donor_selected = donors == donor
        section_values = []
        type_values = []
        for section in sorted(set(sections[donor_selected].tolist())):
            local = donor_selected & (sections == section)
            within_section = []
            for type_id in sorted(set(types[local].tolist())):
                loss = float(values[local & (types == type_id)].mean())
                within_section.append(loss)
                type_values.append(loss)
            section_values.append(float(np.mean(within_section)))
        donor_losses[donor] = float(np.mean(section_values))
        donor_type_losses[donor] = float(np.mean(type_values))
    return {
        "donor_section_type_macro_mse": float(np.mean(tuple(donor_losses.values()))),
        "donor_type_macro_mse": float(np.mean(tuple(donor_type_losses.values()))),
        "per_donor": donor_losses,
        "rows": int(len(values)),
        "donors": int(len(donor_losses)),
    }


def score_arm(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    types: np.ndarray,
    target_names: Sequence[str],
) -> Mapping[str, object]:
    """Score one arm without pooling donors or abundant cell types."""

    target = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 2 or predicted.shape != target.shape or not len(target):
        raise ValueError("truth and prediction must be aligned non-empty matrices")
    row_loss = np.mean(np.square(target - predicted), axis=1)
    report = dict(_macro_loss_from_rows(row_loss, donors, sections, types))
    zero_loss = _macro_loss_from_rows(
        np.mean(np.square(target), axis=1), donors, sections, types
    )["donor_section_type_macro_mse"]
    loss = float(report["donor_section_type_macro_mse"])
    report["rmse"] = float(np.sqrt(loss))
    report["residual_coordinate_r2"] = (
        None if zero_loss <= 1.0e-12 else float(1.0 - loss / zero_loss)
    )
    names = tuple(str(value) for value in target_names)
    report["program_mse"] = {
        name: float(
            _macro_loss_from_rows(
                np.square(target[:, index] - predicted[:, index]),
                donors,
                sections,
                types,
            )["donor_section_type_macro_mse"]
        )
        for index, name in enumerate(names)
    }
    report["diagnostics"] = _prediction_diagnostics(
        target, predicted, donors, sections, types, names
    )
    return report


def _prediction_diagnostics(
    truth: np.ndarray,
    prediction: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    types: np.ndarray,
    names: Sequence[str],
) -> Mapping[str, object]:
    variance_ratios: list[float] = []
    correlations: list[float] = []
    rare_recalls: dict[str, list[float]] = {name: [] for name in names}
    for donor in sorted(set(donors.tolist())):
        for section in sorted(set(sections[donors == donor].tolist())):
            for type_id in sorted(
                set(types[(donors == donor) & (sections == section)].tolist())
            ):
                selected = (
                    (donors == donor)
                    & (sections == section)
                    & (types == type_id)
                )
                if int(selected.sum()) < 5:
                    continue
                local_truth = truth[selected]
                local_prediction = prediction[selected]
                true_variance = np.var(local_truth, axis=0)
                predicted_variance = np.var(local_prediction, axis=0)
                valid = true_variance > 1.0e-10
                variance_ratios.extend(
                    (predicted_variance[valid] / true_variance[valid]).tolist()
                )
                for index in np.flatnonzero(valid):
                    if predicted_variance[index] > 1.0e-12:
                        correlation = np.corrcoef(
                            local_truth[:, index], local_prediction[:, index]
                        )[0, 1]
                        if np.isfinite(correlation):
                            correlations.append(float(correlation))
                    cutoff_truth = np.quantile(local_truth[:, index], 0.9)
                    cutoff_prediction = np.quantile(local_prediction[:, index], 0.9)
                    rare = local_truth[:, index] >= cutoff_truth
                    if rare.any():
                        rare_recalls[names[index]].append(
                            float(np.mean(local_prediction[rare, index] >= cutoff_prediction))
                        )
    slopes = []
    for index in range(truth.shape[1]):
        variance = float(np.var(prediction[:, index]))
        if variance > 1.0e-12:
            slopes.append(
                float(
                    np.cov(prediction[:, index], truth[:, index], ddof=0)[0, 1]
                    / variance
                )
            )
    return {
        "median_within_section_type_variance_ratio": (
            None if not variance_ratios else float(np.median(variance_ratios))
        ),
        "median_within_section_type_correlation": (
            None if not correlations else float(np.median(correlations))
        ),
        "median_calibration_slope_truth_on_prediction": (
            None if not slopes else float(np.median(slopes))
        ),
        "rare_state_recall_at_within_stratum_top_decile": {
            name: (None if not values else float(np.mean(values)))
            for name, values in rare_recalls.items()
        },
        "evaluated_rows": int(len(truth)),
    }


def _floor_score(
    half_a: np.ndarray,
    half_b: np.ndarray,
    donors: np.ndarray,
    sections: np.ndarray,
    types: np.ndarray,
) -> Mapping[str, object]:
    row_loss = 0.5 * np.mean(np.square(half_a - half_b), axis=1)
    report = dict(_macro_loss_from_rows(row_loss, donors, sections, types))
    report["rmse"] = float(np.sqrt(report["donor_section_type_macro_mse"]))
    report["definition"] = "0.5 * split-half standardized program MSE"
    return report


def _reference_state_by_donor(
    queries: np.ndarray,
    query_donors: np.ndarray,
    bank: PrototypeBank,
    bank_indices_by_donor: Mapping[str, np.ndarray],
    *,
    temperature: float,
    reference_only: bool = False,
) -> np.ndarray:
    output = np.empty_like(queries, dtype=np.float64)
    for donor in sorted(set(query_donors.tolist())):
        selected = query_donors == donor
        indices = np.asarray(bank_indices_by_donor[donor], dtype=np.int64)
        local = bank.subset(indices)
        if reference_only:
            output[selected] = reference_only_state(
                local.states, local.weights, int(selected.sum())
            )
        else:
            output[selected] = soft_reference_state(
                queries[selected],
                local.states,
                local.weights,
                temperature=temperature,
            )
    return output


def _matched_indices(bank: PrototypeBank, donors: Sequence[str]) -> dict[str, np.ndarray]:
    result = {
        donor: np.flatnonzero(bank.donor_ids == donor)
        for donor in sorted(set(str(value) for value in donors))
    }
    if any(not len(indices) for indices in result.values()):
        raise ValueError("a validation donor lacks a supported matched spatial bank")
    return result


def _ridge_predictions(
    features: np.ndarray,
    latent: np.ndarray,
    source: PilotSource,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alphas: Sequence[float],
    *,
    device: str,
) -> tuple[np.ndarray, str]:
    fit = fit_weighted_ridge_grid(
        features[train_indices],
        latent[train_indices],
        alphas,
        _row_weights(source, train_indices),
        device=device,
    )
    return fit.predict(features[test_indices]), fit.fit_device


def _nested_select(
    features: np.ndarray,
    latent: np.ndarray,
    source: PilotSource,
    outer_training_indices: np.ndarray,
    bank: PrototypeBank,
    *,
    ridge_alphas: Sequence[float],
    fusion_alphas: Sequence[float],
    inner_folds: int,
    temperature: float,
    seed: int,
    device: str,
) -> Mapping[str, object]:
    """Select H first, then alpha, using only inner held-out donors."""

    outer_donors = source.donors[outer_training_indices]
    donor_count = len(set(outer_donors.tolist()))
    folds = grouped_donor_folds(
        outer_donors, n_splits=min(int(inner_folds), donor_count), seed=seed
    )
    ridge_values = tuple(float(value) for value in ridge_alphas)
    fusion_values = tuple(sorted(set(float(value) for value in fusion_alphas)))
    h_loss = {alpha: [] for alpha in ridge_values}
    fold_cache: list[Mapping[str, object]] = []
    devices = []
    for local_train, local_validation in folds:
        train = outer_training_indices[local_train]
        validation = outer_training_indices[local_validation]
        predictions, fit_device = _ridge_predictions(
            features,
            latent,
            source,
            train,
            validation,
            ridge_values,
            device=device,
        )
        devices.append(fit_device)
        for alpha_index, alpha in enumerate(ridge_values):
            report = score_arm(
                latent[validation],
                predictions[alpha_index],
                source.donors[validation],
                source.sections[validation],
                source.fine_types[validation],
                (f"target_{i}" for i in range(latent.shape[1])),
            )
            h_loss[alpha].extend(float(value) for value in report["per_donor"].values())
        fold_cache.append(
            {"validation": validation, "predictions": predictions}
        )
    selected_ridge = min(
        (float(np.mean(h_loss[alpha])), alpha) for alpha in ridge_values
    )[1]
    selected_ridge_index = ridge_values.index(selected_ridge)

    fusion_loss = {alpha: [] for alpha in fusion_values}
    for cached in fold_cache:
        validation = np.asarray(cached["validation"], dtype=np.int64)
        h = np.asarray(cached["predictions"])[selected_ridge_index]
        donors = source.donors[validation]
        matched = _reference_state_by_donor(
            h,
            donors,
            bank,
            _matched_indices(bank, donors.tolist()),
            temperature=temperature,
        )
        for alpha in fusion_values:
            prediction = residual_fusion(h, matched, alpha)
            local = score_arm(
                latent[validation],
                prediction,
                donors,
                source.sections[validation],
                source.fine_types[validation],
                (f"target_{i}" for i in range(latent.shape[1])),
            )
            fusion_loss[alpha].extend(
                float(value) for value in local["per_donor"].values()
            )
    selected_fusion = min(
        (float(np.mean(fusion_loss[alpha])), alpha) for alpha in fusion_values
    )[1]
    return {
        "selected_ridge_alpha": float(selected_ridge),
        "selected_fusion_alpha": float(selected_fusion),
        "ridge_inner_loss": {
            str(alpha): float(np.mean(values)) for alpha, values in h_loss.items()
        },
        "fusion_inner_loss": {
            str(alpha): float(np.mean(values)) for alpha, values in fusion_loss.items()
        },
        "fit_devices": sorted(set(devices)),
        "heldout_outer_outcomes_used": False,
        "purpose": "hyperparameter selection only; not a scientific gate or proxy",
    }


def _equalized_bank(bank: PrototypeBank) -> PrototypeBank:
    """Equalize donors, types within donor, then prototypes within type."""

    weights = np.zeros(len(bank.states), dtype=np.float64)
    donors = sorted(set(bank.donor_ids.tolist()))
    for donor in donors:
        donor_rows = bank.donor_ids == donor
        types = sorted(set(bank.type_labels[donor_rows].tolist()))
        for type_id in types:
            selected = donor_rows & (bank.type_labels == type_id)
            weights[selected] = 1.0 / (
                len(donors) * len(types) * int(selected.sum())
            )
    return PrototypeBank(
        states=bank.states,
        weights=weights,
        donor_ids=bank.donor_ids,
        type_labels=bank.type_labels,
        prototype_ids=bank.prototype_ids,
    )


def _type_routing_only(
    query_donors: np.ndarray,
    query_types: np.ndarray,
    bank: PrototypeBank,
    bank_indices_by_donor: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Route by supplied type only; never inspect H&E or query RNA values."""

    output = np.full((len(query_donors), bank.states.shape[1]), np.nan, dtype=np.float64)
    covered = np.zeros(len(query_donors), dtype=bool)
    missing: list[list[str]] = []
    for donor in sorted(set(query_donors.tolist())):
        donor_indices = np.asarray(bank_indices_by_donor[donor], dtype=np.int64)
        for type_id in sorted(set(query_types[query_donors == donor].tolist())):
            rows = (query_donors == donor) & (query_types == type_id)
            indices = donor_indices[bank.type_labels[donor_indices] == type_id]
            if not len(indices):
                missing.append([donor, type_id])
                continue
            local = bank.subset(indices)
            output[rows] = reference_only_state(
                local.states, local.weights, int(rows.sum())
            )
            covered[rows] = True
    return output, covered, {
        "rows": int(len(query_donors)),
        "covered_rows": int(covered.sum()),
        "abstained_rows": int((~covered).sum()),
        "coverage": float(covered.mean()) if len(covered) else 0.0,
        "missing_donor_type_routes": missing,
        "uses_image": False,
        "uses_query_rna": False,
    }


def _score_if_covered(
    truth: np.ndarray,
    prediction: np.ndarray,
    covered: np.ndarray,
    source: PilotSource,
    indices: np.ndarray,
    target_names: Sequence[str],
) -> Mapping[str, object]:
    if not covered.any():
        return {
            "status": "unavailable_no_covered_rows",
            "coverage": 0.0,
            "abstained_rows": int(len(covered)),
        }
    selected = indices[covered]
    result = dict(
        score_arm(
            truth[covered],
            prediction[covered],
            source.donors[selected],
            source.sections[selected],
            source.fine_types[selected],
            target_names,
        )
    )
    result.update(
        {
            "status": "complete" if covered.all() else "complete_with_abstention",
            "coverage": float(covered.mean()),
            "abstained_rows": int((~covered).sum()),
        }
    )
    return result


def evaluate_outer_donor(
    source: PilotSource,
    query_donor: str,
    *,
    ridge_alphas: Sequence[float] = RIDGE_ALPHAS,
    fusion_alphas: Sequence[float] = FUSION_ALPHAS,
    inner_folds: int = 5,
    minimum_reliability: float = 0.20,
    minimum_reliability_rows: int = 3,
    max_prototypes_per_type: int = 4,
    temperature: float = 1.0,
    seed: int = 17,
    device: str = "auto",
) -> Mapping[str, object]:
    """Evaluate every same-assay dry-run arm for one outer LODO donor."""

    donor = str(query_donor)
    if np.any(source.evaluation_mask & source.reference_mask):
        raise ValueError("query and spatial-reference masks must be disjoint")
    available_donors = sorted(set(source.donors[source.evaluation_mask].tolist()))
    if donor not in available_donors:
        raise ValueError("query donor has no supported evaluation cells")
    training = np.flatnonzero(source.evaluation_mask & (source.donors != donor))
    query = np.flatnonzero(source.evaluation_mask & (source.donors == donor))
    if len(set(source.donors[training].tolist())) < 3:
        raise ValueError("nested donor CV requires at least three outer-training donors")

    selected, reliability = _training_reliable_programs(
        source,
        np.isin(np.arange(len(source.donors)), training),
        minimum_reliability=minimum_reliability,
        minimum_rows=minimum_reliability_rows,
    )
    if not len(selected):
        return {
            "query_donor": donor,
            "status": "abstained_no_outer_training_reliable_nucleus_program",
            "query_rows": int(len(query)),
            "selected_programs": [],
            "reliability": reliability,
        }
    target_names = tuple(source.program_names[index] for index in selected)

    eligible = np.flatnonzero(source.evaluation_mask)
    fit_weights = donor_section_type_row_weights(
        source.donors[eligible],
        source.sections[eligible],
        source.fine_types[eligible],
    )
    basis = fit_target_basis(
        source.program_total[eligible][:, selected],
        source.donors[eligible],
        [value for value in available_donors if value != donor],
        n_components=len(selected),
        sample_weight=fit_weights,
    )
    latent = basis.transform(source.program_total[:, selected])
    half_a = basis.transform(source.program_half_a[:, selected])
    half_b = basis.transform(source.program_half_b[:, selected])
    reference_indices = np.flatnonzero(source.reference_mask)
    bank = build_reference_prototypes(
        latent[reference_indices],
        source.donors[reference_indices],
        source.fine_types[reference_indices],
        source.observation_ids[reference_indices],
        max_prototypes_per_type=max_prototypes_per_type,
        seed=seed,
    )
    banks = build_matched_wrong_generic_banks(
        bank,
        donor,
        source.donor_indications[donor],
        source.donor_indications,
    )

    tuning = _nested_select(
        source.images,
        latent,
        source,
        training,
        bank,
        ridge_alphas=ridge_alphas,
        fusion_alphas=fusion_alphas,
        inner_folds=inner_folds,
        temperature=temperature,
        seed=seed,
        device=device,
    )
    image_fit = fit_weighted_ridge_grid(
        source.images[training],
        latent[training],
        [tuning["selected_ridge_alpha"]],
        _row_weights(source, training),
        device=device,
    )
    h = image_fit.predict(source.images[query])[0]
    matched_bank = bank.subset(banks["matched"])
    equalized_matched_bank = _equalized_bank(matched_bank)
    matched_reference = soft_reference_state(
        h, matched_bank.states, matched_bank.weights, temperature=temperature
    )
    equalized_reference = soft_reference_state(
        h,
        equalized_matched_bank.states,
        equalized_matched_bank.weights,
        temperature=temperature,
    )
    alpha = float(tuning["selected_fusion_alpha"])
    fused = residual_fusion(h, matched_reference, alpha)
    fused_equalized = residual_fusion(h, equalized_reference, alpha)
    matched_only = reference_only_state(
        matched_bank.states, matched_bank.weights, len(query)
    )
    matched_only_equalized = reference_only_state(
        equalized_matched_bank.states,
        equalized_matched_bank.weights,
        len(query),
    )

    generic_bank = bank.subset(banks["generic"])
    generic_reference = soft_reference_state(
        h, generic_bank.states, generic_bank.weights, temperature=temperature
    )
    generic_fused = residual_fusion(h, generic_reference, alpha)
    equalized_generic = _equalized_bank(generic_bank)
    generic_equalized_reference = soft_reference_state(
        h,
        equalized_generic.states,
        equalized_generic.weights,
        temperature=temperature,
    )

    groups = np.char.add(
        np.char.add(source.donors[query], "::"), source.fine_types[query]
    )
    shuffled_indices = deterministic_group_derangement(
        groups, source.observation_ids[query], seed=seed
    )
    shuffled_h = h[shuffled_indices]
    shuffled_reference = soft_reference_state(
        shuffled_h,
        matched_bank.states,
        matched_bank.weights,
        temperature=temperature,
    )
    shuffled_fused = residual_fusion(shuffled_h, shuffled_reference, alpha)

    truth = latent[query]
    arm_predictions = {
        "H": h,
        "R_matched_natural": matched_only,
        "R_matched_equalized": matched_only_equalized,
        "H_plus_R_matched_natural": fused,
        "H_plus_R_matched_equalized": fused_equalized,
        "H_plus_R_generic_natural": generic_fused,
        "H_plus_R_generic_equalized": residual_fusion(
            h, generic_equalized_reference, alpha
        ),
        "shuffled_H_plus_R_matched_natural": shuffled_fused,
    }
    arms = {
        name: score_arm(
            truth,
            prediction,
            source.donors[query],
            source.sections[query],
            source.fine_types[query],
            target_names,
        )
        for name, prediction in arm_predictions.items()
    }

    hard_wrong: dict[str, object] = {}
    hard_wrong_equalized: dict[str, object] = {}
    for wrong_donor in banks["wrong_donors"]:
        wrong_bank = bank.subset(banks["wrong"][wrong_donor])
        wrong_reference = soft_reference_state(
            h, wrong_bank.states, wrong_bank.weights, temperature=temperature
        )
        hard_wrong[wrong_donor] = score_arm(
            truth,
            residual_fusion(h, wrong_reference, alpha),
            source.donors[query],
            source.sections[query],
            source.fine_types[query],
            target_names,
        )
        equalized_wrong_bank = _equalized_bank(wrong_bank)
        equalized_wrong_reference = soft_reference_state(
            h,
            equalized_wrong_bank.states,
            equalized_wrong_bank.weights,
            temperature=temperature,
        )
        hard_wrong_equalized[wrong_donor] = score_arm(
            truth,
            residual_fusion(h, equalized_wrong_reference, alpha),
            source.donors[query],
            source.sections[query],
            source.fine_types[query],
            target_names,
        )

    matched_map = {donor: np.arange(len(matched_bank.states), dtype=np.int64)}
    routed_natural, routed_covered, routed_natural_coverage = _type_routing_only(
        source.donors[query], source.fine_types[query], matched_bank, matched_map
    )
    routed_equalized, routed_equalized_covered, routed_equalized_coverage = (
        _type_routing_only(
            source.donors[query],
            source.fine_types[query],
            equalized_matched_bank,
            matched_map,
        )
    )
    arms["type_routing_only_matched_natural"] = _score_if_covered(
        truth, routed_natural, routed_covered, source, query, target_names
    )
    arms["type_routing_only_matched_equalized"] = _score_if_covered(
        truth,
        routed_equalized,
        routed_equalized_covered,
        source,
        query,
        target_names,
    )

    if source.blank_images is None:
        arms["blank_H_plus_R_matched_natural"] = {
            "status": "unavailable",
            **source.blank_status,
        }
    else:
        blank_h = image_fit.predict(source.blank_images[query])[0]
        blank_reference = soft_reference_state(
            blank_h,
            matched_bank.states,
            matched_bank.weights,
            temperature=temperature,
        )
        arms["blank_H_plus_R_matched_natural"] = score_arm(
            truth,
            residual_fusion(blank_h, blank_reference, alpha),
            source.donors[query],
            source.sections[query],
            source.fine_types[query],
            target_names,
        )
        arms["blank_H_plus_R_matched_natural"]["status"] = "complete"

    coordinate_tuning = _nested_select(
        source.coordinates,
        latent,
        source,
        training,
        bank,
        ridge_alphas=ridge_alphas,
        fusion_alphas=fusion_alphas,
        inner_folds=inner_folds,
        temperature=temperature,
        seed=seed,
        device=device,
    )
    coordinate_fit = fit_weighted_ridge_grid(
        source.coordinates[training],
        latent[training],
        [coordinate_tuning["selected_ridge_alpha"]],
        _row_weights(source, training),
        device=device,
    )
    coordinate_h = coordinate_fit.predict(source.coordinates[query])[0]
    coordinate_reference = soft_reference_state(
        coordinate_h,
        matched_bank.states,
        matched_bank.weights,
        temperature=temperature,
    )
    arms["coordinate_H_plus_R_matched_natural"] = score_arm(
        truth,
        residual_fusion(
            coordinate_h,
            coordinate_reference,
            coordinate_tuning["selected_fusion_alpha"],
        ),
        source.donors[query],
        source.sections[query],
        source.fine_types[query],
        target_names,
    )

    floor = _floor_score(
        half_a[query],
        half_b[query],
        source.donors[query],
        source.sections[query],
        source.fine_types[query],
    )
    h_loss = float(arms["H"]["donor_section_type_macro_mse"])
    fusion_loss = float(
        arms["H_plus_R_matched_natural"]["donor_section_type_macro_mse"]
    )
    wrong_losses = {
        wrong: float(report["donor_section_type_macro_mse"])
        for wrong, report in hard_wrong.items()
    }
    return {
        "query_donor": donor,
        "status": "complete_same_assay_engineering_dry_run",
        "scope": "spatially disjoint HEST Xenium reference; not matched sc/snRNA",
        "query_rows": int(len(query)),
        "reference_rows": int(source.reference_mask.sum()),
        "selected_programs": list(target_names),
        "target_basis_fit_donors": list(basis.fit_donors),
        "leakage_receipt": {
            "outer_query_donor_used_for_target_basis": False,
            "outer_query_outcomes_used_for_reliability_or_hyperparameters": False,
            "outer_same_donor_spatial_reference_used": True,
            "outer_reference_role_contains_query_cells": False,
        },
        "tuning": tuning,
        "coordinate_tuning": coordinate_tuning,
        "arms": arms,
        "hard_wrong_donor_arms": hard_wrong,
        "hard_wrong_donor_equalized_arms": hard_wrong_equalized,
        "split_half_st_floor": floor,
        "coverage_and_abstention": {
            "query_coverage": 1.0,
            "query_abstained_rows": 0,
            "type_routing_natural": routed_natural_coverage,
            "type_routing_equalized": routed_equalized_coverage,
            "blank_H": source.blank_status,
            "hard_wrong_donors_evaluated": list(banks["wrong_donors"]),
            "hard_wrong_definition": (
                "every other donor in the same HEST lung cohort and disease-status stratum"
            ),
            "generic_excludes_query_donor": bool(
                banks["query_donor_excluded_from_generic"]
            ),
        },
        "descriptive_diagnostics_only": {
            "relative_mse_gain_H_to_matched_fusion": float(
                (h_loss - fusion_loss) / h_loss
            ),
            "matched_fusion_beats_H": bool(fusion_loss < h_loss),
            "matched_fusion_beats_reference_only": bool(
                fusion_loss
                < arms["R_matched_natural"]["donor_section_type_macro_mse"]
            ),
            "matched_fusion_beats_every_hard_wrong": bool(
                wrong_losses and fusion_loss < min(wrong_losses.values())
            ),
            "scientific_gate": False,
        },
        "correction_audit": {
            "selected_alpha": alpha,
            "maximum_allowed_alpha": 0.5,
            "mean_correction_l2": float(
                np.mean(np.linalg.norm(fused - h, axis=1))
            ),
            "mean_H_l2": float(np.mean(np.linalg.norm(h, axis=1))),
        },
        "iteration": {
            "status": "not_run_by_design",
            "reason": (
                "HEST is a same-assay engineering dry run, not a scientific "
                "gate or refinement proxy"
            ),
            "rounds": 0,
        },
        "fit_devices": {
            "image_outer": image_fit.fit_device,
            "coordinate_outer": coordinate_fit.fit_device,
        },
    }


def _aggregate_complete_donors(
    donor_reports: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    complete = {
        donor: report
        for donor, report in donor_reports.items()
        if report.get("status") == "complete_same_assay_engineering_dry_run"
    }
    arm_names = sorted(
        set.intersection(
            *(set(report["arms"]) for report in complete.values())
        )
    ) if complete else []
    arms: dict[str, object] = {}
    for name in arm_names:
        per_donor = {}
        for donor, report in complete.items():
            arm = report["arms"][name]
            if "donor_section_type_macro_mse" in arm:
                per_donor[donor] = float(arm["donor_section_type_macro_mse"])
        if per_donor:
            arms[name] = {
                "donor_equal_mean_mse": float(np.mean(tuple(per_donor.values()))),
                "donor_equal_rmse": float(np.sqrt(np.mean(tuple(per_donor.values())))),
                "per_donor_mse": per_donor,
                "donors": len(per_donor),
            }
    wrong_by_query = {
        donor: {
            wrong: float(values["donor_section_type_macro_mse"])
            for wrong, values in report["hard_wrong_donor_arms"].items()
        }
        for donor, report in complete.items()
    }
    equalized_wrong_by_query = {
        donor: {
            wrong: float(values["donor_section_type_macro_mse"])
            for wrong, values in report.get(
                "hard_wrong_donor_equalized_arms", {}
            ).items()
        }
        for donor, report in complete.items()
    }
    diagnostics: dict[str, object] = {"scientific_gate": False}
    if "H" in arms and "H_plus_R_matched_natural" in arms:
        h = arms["H"]["per_donor_mse"]
        fusion = arms["H_plus_R_matched_natural"]["per_donor_mse"]
        donors = sorted(set(h) & set(fusion))
        gains = [(h[donor] - fusion[donor]) / h[donor] for donor in donors]
        diagnostics.update(
            {
                "mean_relative_mse_gain": float(np.mean(gains)),
                "positive_donor_fraction": float(np.mean(np.asarray(gains) > 0)),
                "interpretation": "descriptive same-assay mechanics only",
            }
        )
    return {
        "complete_donors": sorted(complete),
        "abstained_or_failed_donors": sorted(set(donor_reports) - set(complete)),
        "arms": arms,
        "hard_wrong_donor_mse_by_query": wrong_by_query,
        "hard_wrong_donor_equalized_mse_by_query": equalized_wrong_by_query,
        "descriptive_diagnostics_only": diagnostics,
        "iteration": {
            "status": "not_run_by_design",
            "rounds": 0,
            "scientific_gate": False,
        },
    }


def _base_report(source: PilotSource, *, seed: int, device: str) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "status": "in_progress",
        "experiment_class": "retrospective_same_assay_engineering_dry_run",
        "scientific_role": (
            "mechanics/QC only; must not act as a scientific gate, proxy, or validation"
        ),
        "reference_modality": "spatially disjoint Xenium cells from the same HEST assay",
        "matched_scrna_reference": False,
        "matched_snrna_reference": False,
        "personalized_reference_validation": False,
        "authorizes_h_cell": False,
        "authorizes_h_intrinsic": False,
        "authorizes_reference_refinement": False,
        "authorizes_full_heir": False,
        "source": {
            "path": str(source.path),
            "sha256": source.sha256,
            "reference_split_id": source.reference_split_id,
            "rows": int(len(source.donors)),
            "supported_evaluation_rows": int(source.evaluation_mask.sum()),
            "supported_reference_rows": int(source.reference_mask.sum()),
            "donors": sorted(set(source.donors.tolist())),
            "blank_H": source.blank_status,
        },
        "design": {
            "outer_evaluation": "leave one biological donor out",
            "inner_selection": "grouped donor CV within outer-training donors",
            "target": "outer-training-reliable nucleus program space",
            "fusion": "one bounded H&E-centred correction, alpha <= 0.5",
            "natural_and_equalized_bank_controls": True,
            "type_routing_only_diagnostic": True,
            "coverage_and_abstention_reported": True,
            "iteration": "prohibited for this dry run",
        },
        "seed": int(seed),
        "requested_device": str(device),
        "donor_reports": {},
    }


def run_pilot(
    source_path: Path,
    output_dir: Path,
    *,
    reference_split_id: str = "primary",
    minimum_support: int = 20,
    inner_folds: int = 5,
    minimum_reliability: float = 0.20,
    max_prototypes_per_type: int = 4,
    temperature: float = 1.0,
    seed: int = 17,
    device: str = "auto",
    enforce_registered_hash: bool = True,
) -> Mapping[str, object]:
    source = load_source(
        source_path,
        reference_split_id=reference_split_id,
        minimum_support=minimum_support,
        enforce_registered_hash=enforce_registered_hash,
    )
    report = _base_report(source, seed=seed, device=device)
    report_path = output_dir.expanduser().resolve() / "report.json"
    _write_json(report_path, report)
    for donor in sorted(set(source.donors[source.evaluation_mask].tolist())):
        report["donor_reports"][donor] = evaluate_outer_donor(
            source,
            donor,
            inner_folds=inner_folds,
            minimum_reliability=minimum_reliability,
            max_prototypes_per_type=max_prototypes_per_type,
            temperature=temperature,
            seed=seed,
            device=device,
        )
        _write_json(report_path, report)
    report["summary"] = _aggregate_complete_donors(report["donor_reports"])
    report["status"] = "complete_same_assay_engineering_dry_run"
    _write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-split", default="primary")
    parser.add_argument("--minimum-support", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--minimum-reliability", type=float, default=0.20)
    parser.add_argument("--max-prototypes-per-type", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--allow-unregistered-source",
        action="store_true",
        help="testing only: bypass the pinned retrospective source SHA-256",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    run_pilot(
        arguments.source,
        arguments.output_dir,
        reference_split_id=arguments.reference_split,
        minimum_support=arguments.minimum_support,
        inner_folds=arguments.inner_folds,
        minimum_reliability=arguments.minimum_reliability,
        max_prototypes_per_type=arguments.max_prototypes_per_type,
        temperature=arguments.temperature,
        seed=arguments.seed,
        device=arguments.device,
        enforce_registered_hash=not arguments.allow_unregistered_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
