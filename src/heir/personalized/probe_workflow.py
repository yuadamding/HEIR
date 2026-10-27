"""Direct frozen-image readouts on a fixed normalized-log measurement scale.

These diagnostic ridge models have no RNA-state interpretation. Predictions are
scored directly: neither library normalization nor clipping is applied to them.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .error_decomposition import _gene_errors, error_decomposition
from .evaluate import _ranks
from .observation import ObservationOperator, normalized_log_expression
from .protein_features import load_joint_inputs
from .schema import SpatialTargets, identifiers, image_backbone_identity, spatial_metadata
from .serialization import load_prediction
from .study import _sha, validate_folds
from .train import TrainingSample

SCALE = "fixed_panel_log1p_1e4"
SCORE = "mean_replicate_median_gene_spearman"
METRICS = (SCORE, "normalized_log_mse", "squared_bias", "centered_mse")
VARIABILITY = "mean_replicate_gene_sd_ratio_diagnostic"
ENDPOINTS = (*METRICS, VARIABILITY)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _defined_mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _strict_mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def _variability(rows):
    return _defined_mean(
        [
            _defined_mean([r["std_ratio"] for r in rows if r["assay_replicate"] == group])
            for group in sorted({r["assay_replicate"] for r in rows})
        ]
    )


def _region_rows(sample):
    """Require the existing matched regional units, not averaging of log profiles."""
    inputs, targets, operator = sample.inputs, sample.targets, sample.observation
    if inputs.patient_id != targets.patient_id or inputs.specimen_id != targets.specimen_id:
        raise ValueError("probe inputs and targets must identify the same specimen")
    if tuple(operator.observation_ids) != tuple(targets.observation_ids):
        raise ValueError("probe target and observation rows must be ordered identically")
    if set(operator.cell_ids) != set(inputs.cell_ids):
        raise ValueError("probe image and observation region IDs differ")
    positive = np.asarray(operator.weights) > 0
    counts = np.bincount(operator.rows[positive], minlength=len(targets.observation_ids))
    if np.any(counts != 1) or np.any(operator.weights[positive] != 1):
        raise ValueError("direct-log probes require one unit-weight region per observation")
    mapping = np.empty(len(counts), dtype=int)
    lookup = {key: j for j, key in enumerate(inputs.cell_ids)}
    image_columns = np.asarray([lookup[key] for key in operator.cell_ids])
    mapping[operator.rows[positive]] = image_columns[operator.columns[positive]]
    if sample.observation_groups is None:
        raise ValueError("probe observations require explicit assay-replicate groups")
    return mapping


def _truth(sample, gene_ids, normalization_gene_ids):
    lookup = {gene: i for i, gene in enumerate(sample.targets.gene_ids)}
    if not set(gene_ids) | set(normalization_gene_ids) <= set(lookup):
        raise ValueError("probe fixed gene panel is absent from target measurements")
    values = torch.as_tensor(sample.targets.expression, dtype=torch.float32)
    observed = torch.as_tensor(sample.targets.observed, dtype=torch.bool)
    normalized = normalized_log_expression(
        values, observed, normalization_indices=[lookup[g] for g in normalization_gene_ids]
    ).numpy()
    columns = [lookup[g] for g in gene_ids]
    return normalized[:, columns], np.asarray(sample.targets.observed)[:, columns]


def _feature_transform(samples):
    """Equal patients, then specimens, then unique prediction regions; no labels."""
    counts = Counter(sample.inputs.patient_id for sample in samples)
    n_patients = len(counts)
    values, weights = [], []
    for sample in samples:
        image = np.asarray(sample.inputs.image_features, dtype=np.float64)
        values.append(image)
        weights.append(
            np.full(len(image), 1 / (n_patients * counts[sample.inputs.patient_id] * len(image)))
        )
    values, weights = np.concatenate(values), np.concatenate(weights)
    mean = np.sum(values * weights[:, None], axis=0)
    variance = np.sum((values - mean) ** 2 * weights[:, None], axis=0)
    return mean, np.maximum(np.sqrt(variance), 1e-6)


def patient_balanced_weights(labels, observed):
    """Equal patient/specimen/replicate mass over observed outcomes for one gene."""
    labels, observed = list(labels), np.asarray(observed, dtype=bool)
    if observed.shape != (len(labels),):
        raise ValueError("probe training availability must follow observation rows")
    patients = sorted({row[0] for row in labels})
    weights = np.zeros(len(labels), dtype=np.float64)
    for patient in patients:
        specimens = sorted(
            {row[1] for row, keep in zip(labels, observed) if keep and row[0] == patient}
        )
        if not specimens:
            raise ValueError(
                "every training patient must provide an observed value for every probe gene"
            )
        for specimen in specimens:
            groups = sorted(
                {
                    row[2]
                    for row, keep in zip(labels, observed)
                    if keep and row[:2] == (patient, specimen)
                }
            )
            for group in groups:
                selected = (
                    np.asarray([row == (patient, specimen, group) for row in labels]) & observed
                )
                weights[selected] = 1 / (
                    len(patients) * len(specimens) * len(groups) * selected.sum()
                )
    return weights


@dataclass
class FrozenFeatureProbe:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    gene_ids: tuple[str, ...]
    normalization_gene_ids: tuple[str, ...]
    encoder_manifest_sha256: str
    prediction_unit: str
    coordinate_units: str
    alpha: float
    training_patient_ids: tuple[str, ...]
    validation_patient_ids: tuple[str, ...]
    selection: dict

    def __post_init__(self):
        identifiers(self.gene_ids, "probe output genes")
        identifiers(self.normalization_gene_ids, "probe normalization genes")
        image_backbone_identity(self.encoder_manifest_sha256, present=True)
        spatial_metadata(self.prediction_unit, self.coordinate_units)
        if (
            len(self.normalization_gene_ids) < 2
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
            or self.feature_mean.ndim != 1
            or self.feature_scale.shape != self.feature_mean.shape
            or self.coefficients.shape != (len(self.feature_mean), len(self.gene_ids))
            or self.intercept.shape != (len(self.gene_ids),)
            or not all(
                np.isfinite(value).all()
                for value in (
                    self.feature_mean,
                    self.feature_scale,
                    self.coefficients,
                    self.intercept,
                )
            )
            or np.any(self.feature_scale <= 0)
        ):
            raise ValueError(
                "probe readout arrays, regularization or fixed denominator are invalid"
            )

    def predict(self, inputs):
        if inputs.reference is not None or inputs.protein_features is not None:
            raise ValueError("direct feature probes accept image-only inputs")
        if inputs.image_encoder_manifest_sha256 != self.encoder_manifest_sha256:
            raise ValueError("probe encoder identity differs from the fitted representation")
        if (inputs.prediction_unit, inputs.coordinate_units) != (
            self.prediction_unit,
            self.coordinate_units,
        ):
            raise ValueError("probe prediction units differ from the fitted regional frame")
        values = np.asarray(inputs.image_features, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or not np.isfinite(values).all()
        ):
            raise ValueError("probe image features differ from the fitted finite feature axis")
        return (
            (values - self.feature_mean) / self.feature_scale
        ) @ self.coefficients + self.intercept

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(
            directory / "readout.npz",
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            coefficients=self.coefficients,
            intercept=self.intercept,
        )
        metadata = {
            key: getattr(self, key)
            for key in (
                "gene_ids",
                "normalization_gene_ids",
                "encoder_manifest_sha256",
                "prediction_unit",
                "coordinate_units",
                "alpha",
                "training_patient_ids",
                "validation_patient_ids",
                "selection",
            )
        }
        metadata.update(
            schema="heir.frozen_feature_probe.v1",
            prediction_scale=SCALE,
            normalize_prediction_again=False,
            readout_sha256=_sha(directory / "readout.npz"),
        )
        _write(directory / "probe.json", metadata)
        _write(
            directory / "sha256.json",
            {name: _sha(directory / name) for name in ("probe.json", "readout.npz")},
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        inventory = json.loads((directory / "sha256.json").read_text())
        if set(inventory) != {"probe.json", "readout.npz"} or any(
            _sha(directory / name) != digest for name, digest in inventory.items()
        ):
            raise ValueError("probe bundle inventory integrity mismatch")
        metadata = json.loads((directory / "probe.json").read_text())
        if (
            metadata.pop("schema") != "heir.frozen_feature_probe.v1"
            or metadata.pop("prediction_scale") != SCALE
            or metadata.pop("normalize_prediction_again") is not False
            or metadata.pop("readout_sha256") != _sha(directory / "readout.npz")
        ):
            raise ValueError("probe bundle integrity or normalized-log scale contract differs")
        with np.load(directory / "readout.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        for key in (
            "gene_ids",
            "normalization_gene_ids",
            "training_patient_ids",
            "validation_patient_ids",
        ):
            metadata[key] = tuple(metadata[key])
        return cls(**metadata, **arrays)


def score_probe(predicted, sample, gene_ids, normalization_gene_ids, *, row_mask=None):
    """Direct score on normalized-log values; predictions are never normalized."""
    mapping = _region_rows(sample)
    truth, observed = _truth(sample, gene_ids, normalization_gene_ids)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (
        predicted.shape != (len(sample.inputs.cell_ids), len(gene_ids))
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("probe normalized-log prediction must be finite and region/gene aligned")
    predicted = predicted[mapping]
    selected = np.ones(len(mapping), dtype=bool) if row_mask is None else np.asarray(row_mask)
    if selected.shape != (len(mapping),) or selected.dtype != bool:
        raise ValueError("probe scoring support must be a boolean observation-row mask")
    groups = np.asarray(sample.observation_groups)
    replicates, gene_rows, target_eligibility = [], [], []
    for group in sorted(set(groups)):
        rows = (groups == group) & selected
        current, errors = [], []
        n_observed = int(observed[rows].sum())
        for column, gene in enumerate(gene_ids):
            keep = rows & observed[:, column]
            x, y = predicted[keep, column], truth[keep, column].astype(np.float64)
            reason = (
                "insufficient_regions"
                if len(x) < 3
                else "constant_target"
                if np.ptp(y) == 0
                else "constant_prediction"
                if np.ptp(x) == 0
                else None
            )
            current.append(
                {
                    "gene_id": gene,
                    "n_regions": len(x),
                    "spearman": None if reason else float(np.corrcoef(_ranks(x), _ranks(y))[0, 1]),
                    "undefined_reason": reason,
                }
            )
            error = {
                "patient_id": sample.inputs.patient_id,
                "specimen_id": sample.inputs.specimen_id,
                "assay_replicate": group,
                "gene_id": gene,
                **_gene_errors(x, y),
            }
            error["weight_in_replicate"] = len(x) / n_observed if n_observed else 0.0
            error.update(spearman=current[-1]["spearman"], undefined_reason=reason)
            errors.append(error)
        invalid = any(row["undefined_reason"] == "constant_prediction" for row in current)
        correlations = [row["spearman"] for row in current if row["spearman"] is not None]
        metrics = {
            key: sum(
                row["weight_in_replicate"] * row["mse" if key == "normalized_log_mse" else key]
                for row in errors
                if row["n"]
            )
            if n_observed
            else None
            for key in ("normalized_log_mse", "squared_bias", "centered_mse")
        }
        replicates.append(
            {
                "assay_replicate": group,
                "n_rows": int(rows.sum()),
                "median_gene_spearman": None
                if invalid or not correlations
                else float(np.median(correlations)),
                "headline_status": "undefined_prediction_on_evaluable_panel"
                if invalid
                else "evaluable"
                if correlations
                else "no_evaluable_target_genes",
                "gene_scores": current,
                **metrics,
            }
        )
        gene_rows.extend(errors)
        target_eligibility.append(
            {
                "assay_replicate": group,
                "genes": [
                    {
                        "gene_id": row["gene_id"],
                        "n_regions": row["n_regions"],
                        "target_exclusion": row["undefined_reason"]
                        if row["undefined_reason"] in ("insufficient_regions", "constant_target")
                        else None,
                    }
                    for row in current
                ],
            }
        )
    missing = [row["assay_replicate"] for row in replicates if not row["n_rows"]]
    invalid = any(
        r["headline_status"] == "undefined_prediction_on_evaluable_panel" for r in replicates
    )
    rho = None if invalid else _defined_mean([r["median_gene_spearman"] for r in replicates])
    report = {
        "patient_id": sample.inputs.patient_id,
        "specimen_id": sample.inputs.specimen_id,
        "prediction_unit": sample.inputs.prediction_unit,
        "coordinate_units": sample.inputs.coordinate_units,
        "prediction_scale": SCALE,
        "normalize_prediction_again": False,
        "evaluation_gene_ids": list(gene_ids),
        "normalization_gene_ids": list(normalization_gene_ids),
        "evaluation_observation_ids": [
            key for key, keep in zip(sample.targets.observation_ids, selected) if keep
        ],
        "target_gene_eligibility": target_eligibility,
        "replicate_scores": replicates,
        "headline_status": "no_common_protein_coverage_in_some_replicates"
        if missing
        else "undefined_prediction_on_evaluable_panel"
        if invalid
        else "evaluable"
        if rho is not None
        else "no_evaluable_target_genes",
        SCORE: None if missing else rho,
        **{
            key: None if missing else _defined_mean([r[key] for r in replicates])
            for key in ("normalized_log_mse", "squared_bias", "centered_mse")
        },
        "missing_coverage_replicates": missing,
        VARIABILITY: None if missing else _variability(gene_rows),
        "analysis_precision": "float64_readout_and_moments_of_float32_fixed_panel_targets",
        "scientific_qualification": False,
        "interpretation": (
            "direct frozen-image diagnostic; no reference-state or cell-placement claim"
        ),
    }
    return report, gene_rows


def fit_feature_probe(training, validation, gene_ids, normalization_gene_ids, alpha_grid=ALPHAS):
    """Select one of explicitly counted weighted ridge candidates using validation only."""
    training, validation = list(training), list(validation)
    alpha_grid = tuple(float(alpha) for alpha in alpha_grid)
    if (
        not training
        or not validation
        or not alpha_grid
        or any(not np.isfinite(a) or a <= 0 for a in alpha_grid)
        or tuple(sorted(set(alpha_grid))) != alpha_grid
    ):
        raise ValueError(
            "probe fitting needs train/validation specimens and ordered distinct positive alphas"
        )
    train_ids = tuple(sorted({s.inputs.patient_id for s in training}))
    validation_ids = tuple(sorted({s.inputs.patient_id for s in validation}))
    if set(train_ids) & set(validation_ids):
        raise ValueError("probe training and validation patients must be disjoint")
    all_samples = training + validation
    identities = {
        (
            s.inputs.image_encoder_manifest_sha256,
            s.inputs.prediction_unit,
            s.inputs.coordinate_units,
            s.inputs.image_features.shape[1],
        )
        for s in all_samples
    }
    if len(identities) != 1 or any(
        s.inputs.reference is not None or s.inputs.protein_features is not None for s in all_samples
    ):
        raise ValueError(
            "probe samples must share one frozen image-only representation and spatial frame"
        )
    if len({(s.inputs.patient_id, s.inputs.specimen_id) for s in all_samples}) != len(all_samples):
        raise ValueError("duplicate probe training or validation specimens")
    mean, scale = _feature_transform(training)
    features, targets, masks, labels = [], [], [], []
    for sample in training:
        mapping = _region_rows(sample)
        features.append((sample.inputs.image_features[mapping].astype(np.float64) - mean) / scale)
        y, mask = _truth(sample, gene_ids, normalization_gene_ids)
        targets.append(y.astype(np.float64))
        masks.append(mask)
        labels.extend(
            (sample.inputs.patient_id, sample.inputs.specimen_id, group)
            for group in sample.observation_groups
        )
    x, y, observed = np.concatenate(features), np.concatenate(targets), np.concatenate(masks)
    coefficients = [np.zeros((x.shape[1], len(gene_ids))) for _ in alpha_grid]
    intercepts = [np.zeros(len(gene_ids)) for _ in alpha_grid]
    decompositions = {}
    for column in range(len(gene_ids)):
        weights = patient_balanced_weights(labels, observed[:, column])
        key = weights.tobytes()
        if key not in decompositions:
            x_mean = weights @ x
            weighted_x = (x - x_mean) * np.sqrt(weights)[:, None]
            u, singular, vt = np.linalg.svd(weighted_x, full_matrices=False)
            decompositions[key] = x_mean, u, singular, vt
        x_mean, u, singular, vt = decompositions[key]
        values = np.where(observed[:, column], y[:, column], 0.0)
        y_mean = weights @ values
        response = u.T @ ((values - y_mean) * np.sqrt(weights))
        for index, alpha in enumerate(alpha_grid):
            coefficient = vt.T @ (singular / (singular**2 + alpha) * response)
            coefficients[index][:, column] = coefficient
            intercepts[index][column] = y_mean - x_mean @ coefficient
    encoder, unit, coordinate_units, _ = next(iter(identities))
    candidates, reports = [], []
    for alpha, coefficient, intercept in zip(alpha_grid, coefficients, intercepts):
        model = FrozenFeatureProbe(
            mean,
            scale,
            coefficient,
            intercept,
            tuple(gene_ids),
            tuple(normalization_gene_ids),
            encoder,
            unit,
            coordinate_units,
            alpha,
            train_ids,
            validation_ids,
            {},
        )
        per_specimen = [
            score_probe(model.predict(s.inputs), s, gene_ids, normalization_gene_ids)[0]
            for s in validation
        ]
        per_patient = [
            {
                "patient_id": patient,
                **{
                    key: _strict_mean([r[key] for r in per_specimen if r["patient_id"] == patient])
                    for key in METRICS
                },
            }
            for patient in validation_ids
        ]
        report = {
            "alpha": alpha,
            "validation_patient_ids": list(validation_ids),
            "patients": per_patient,
            **{key: _strict_mean([r[key] for r in per_patient]) for key in METRICS},
        }
        report["eligible"] = all(
            report[key] is not None and np.isfinite(report[key]) for key in METRICS
        )
        candidates.append(model)
        reports.append(report)
    eligible = [index for index, report in enumerate(reports) if report["eligible"]]
    if not eligible:
        raise ValueError("no ridge candidate has fully defined validation endpoints")
    best = max(
        eligible,
        key=lambda index: (
            reports[index][SCORE],
            -reports[index]["centered_mse"],
            -reports[index]["normalized_log_mse"],
        ),
    )
    selected = candidates[best]
    selected.selection = {
        "policy": "validation_spatial_correlation_then_centered_error_then_total_mse",
        "alpha_grid": list(alpha_grid),
        "selected_alpha": selected.alpha,
        "candidate_solutions": len(alpha_grid),
        "selected_readouts": 1,
        "additional_refits": 0,
        "weighted_design_decompositions": len(decompositions),
        "validation": reports,
        "loss": (
            "equal_patient_specimen_replicate_observed_row_squared_error_plus_alpha_weight_norm"
        ),
        "intercept_penalized": False,
        "feature_scaling": (
            "training_only_equal_patient_specimen_unique_prediction_region_population_mean_sd"
        ),
    }
    return selected


def _sample_from_source(inputs, path):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    target = SpatialTargets(
        tuple(values["observation_ids"]),
        tuple(values["gene_ids"]),
        values["expression"],
        values["observed"],
        inputs.patient_id,
        inputs.specimen_id,
    )
    operator = ObservationOperator(
        tuple(values["cell_ids"]),
        target.observation_ids,
        values["rows"],
        values["columns"],
        values["weights"],
    )
    return TrainingSample(inputs, target, operator, tuple(values["groups"]))


def _export_probe(model, inputs, path, metadata):
    values = model.predict(inputs)
    record = {
        "schema": "heir.frozen_feature_probe_prediction.v1",
        "prediction_scale": SCALE,
        "normalize_prediction_again": False,
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "prediction_unit": inputs.prediction_unit,
        "coordinate_units": inputs.coordinate_units,
        "encoder_manifest_sha256": model.encoder_manifest_sha256,
        "normalization_gene_ids": list(model.normalization_gene_ids),
        "test_targets_loaded": False,
        "alpha": model.alpha,
        **metadata,
    }
    with Path(path).open("xb") as handle:
        np.savez(
            handle,
            normalized_log_expression=values,
            region_ids=np.asarray(inputs.cell_ids),
            gene_ids=np.asarray(model.gene_ids),
            coordinates=inputs.coordinates,
            metadata=np.asarray(json.dumps(record, allow_nan=False)),
        )


def _load_probe_prediction(path, inputs, gene_ids):
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(archive["metadata"].item())
        if (
            record["schema"] != "heir.frozen_feature_probe_prediction.v1"
            or record["prediction_scale"] != SCALE
            or record["normalize_prediction_again"] is not False
            or tuple(archive["region_ids"]) != inputs.cell_ids
            or tuple(archive["gene_ids"]) != tuple(gene_ids)
            or not np.array_equal(archive["coordinates"], inputs.coordinates)
            or record["patient_id"] != inputs.patient_id
            or record["specimen_id"] != inputs.specimen_id
            or record["encoder_manifest_sha256"] != inputs.image_encoder_manifest_sha256
        ):
            raise ValueError("probe export scale, feature identity or spatial axes differ")
        return archive["normalized_log_expression"].copy()


def _summary(records):
    patients = sorted({row["patient_id"] for row in records})
    result = []
    for patient in patients:
        selected = [r for r in records if r["patient_id"] == patient]
        if len({r["specimen_id"] for r in selected}) != len(selected):
            raise ValueError("duplicate probe patient/specimen evaluation")
        result.append(
            {
                "patient_id": patient,
                **{key: _strict_mean([r[key] for r in selected]) for key in ENDPOINTS},
            }
        )
    return {
        "n_patients": len(patients),
        "patients": result,
        **{key: _strict_mean([r[key] for r in result]) for key in ENDPOINTS},
    }


def _assert_matched_support(probe, baseline):
    for key in (
        "patient_id",
        "specimen_id",
        "evaluation_observation_ids",
        "evaluation_gene_ids",
        "normalization_gene_ids",
    ):
        if probe[key] != baseline[key]:
            raise ValueError(
                "probe and saved HRP comparator require identical target rows and gene eligibility"
            )
    if probe["evaluation_observation_ids"] and sorted(
        probe["target_gene_eligibility"], key=lambda row: row["assay_replicate"]
    ) != sorted(baseline["target_gene_eligibility"], key=lambda row: row["assay_replicate"]):
        raise ValueError("probe and saved HRP comparator require identical target gene eligibility")


def _verify_baseline_truth(path, saved, sample, gene_ids, normalization_gene_ids, row_mask):
    """Recompute old endpoints on current truth before accepting a paired contrast."""
    from .cohort_cli import _subset, replicate_scores

    if row_mask is not None and not row_mask.any():
        if any(saved[key] is not None for key in METRICS):
            raise ValueError(
                "saved HRP comparator reports metrics without common covered observations"
            )
        return None
    target, operator, groups = sample.targets, sample.observation, sample.observation_groups
    missing = False
    if row_mask is not None:
        target, operator = _subset(target, operator, row_mask)
        groups = tuple(group for group, keep in zip(groups, row_mask) if keep)
        missing = set(groups) != set(sample.observation_groups)
    prediction, _ = load_prediction(path)
    score = replicate_scores(prediction, target, operator, groups, gene_ids, normalization_gene_ids)
    moments = error_decomposition(
        prediction, target, operator, groups, gene_ids, normalization_gene_ids
    )
    for key in METRICS:
        recomputed = None if missing else score[key] if key in score else moments[key]
        recorded = saved[key]
        tolerance = 1e-5 if key == "normalized_log_mse" else 1e-9
        if (recorded is None) != (recomputed is None) or (
            recorded is not None and not np.isclose(recorded, recomputed, rtol=0, atol=tolerance)
        ):
            raise ValueError(
                "saved HRP comparator scores differ from the currently loaded target values"
            )
    return None if missing else _variability(moments["per_gene"])


def _protocol(path):
    protocol = json.loads(Path(path).read_text())
    if (
        protocol.get("schema") != "heir.frozen_feature_probe_study.v1"
        or set(protocol.get("backbones", {})) != {"hoptimus1", "uni2h"}
        or protocol.get("outer_folds") != list(range(5))
        or protocol.get("seed") != 0
        or tuple(protocol.get("alpha_grid", ())) != ALPHAS
        or protocol.get("selected_readouts") != 10
        or protocol.get("candidate_solutions") != 50
        or protocol.get("feature_dim") != 1536
        or protocol.get("prediction_scale") != SCALE
        or protocol.get("no_train_validation_refit") is not True
        or protocol.get("selection")
        != "validation_spatial_correlation_then_centered_error_then_total_mse"
    ):
        raise ValueError("unsupported frozen ten-readout/fifty-candidate probe protocol")
    for name in ("evaluation_gene_ids", "normalization_gene_ids"):
        identifiers(tuple(protocol[name]), name)
    return protocol


def run_feature_probes(config_path, output):
    """Fit only the prespecified readouts; all test exports precede test target access."""
    from .joint_workflow import _coverage_target, _tsv

    protocol = _protocol(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(Path(config_path).read_bytes())
    genes, panel = tuple(protocol["evaluation_gene_ids"]), tuple(protocol["normalization_gene_ids"])
    source_records, sources = {}, {}
    for backbone, declaration in protocol["backbones"].items():
        source = Path(declaration["source_study_directory"])
        prepared = Path(declaration["prepared_directory"])
        identity = json.loads((source / "identity.json").read_text())
        metadata = json.loads((prepared / "prepared.json").read_text())
        source_config = json.loads((source / "config.json").read_text())
        if (
            declaration["source_identity_sha256"] != _sha(source / "identity.json")
            or declaration["source_exports_sha256"] != _sha(source / "exports.json")
            or identity["prepared_sha256"] != _sha(prepared / "prepared.json")
            or identity["experiment_config_sha256"] != _sha(source / "config.json")
            or identity["folds_sha256"] != protocol["folds_sha256"]
            or _sha(source / "folds.csv") != protocol["folds_sha256"]
            or source_config["model"]["gene_ids"] != list(genes)
            or source_config["model"]["normalization_gene_ids"] != list(panel)
            or source_config["model"]["image_encoder_manifest_sha256"]
            != declaration["encoder_manifest_sha256"]
        ):
            raise ValueError("probe source study differs from its frozen target/feature identity")
        for biopsy, digest in identity["input_metadata_sha256"].items():
            if _sha(prepared / "inputs" / biopsy / "metadata.json") != digest:
                raise ValueError("probe prepared metadata differs from the completed source study")
        inputs = {
            b: load_joint_inputs(prepared / "inputs" / b, modalities=("H",))[0]
            for b in metadata["biopsies"]
        }
        if any(
            value.image_features.shape[1] != protocol["feature_dim"]
            or value.image_encoder_manifest_sha256 != declaration["encoder_manifest_sha256"]
            for value in inputs.values()
        ):
            raise ValueError("probe inputs differ from the declared frozen full-width encoder")
        folds = validate_folds(
            source / "folds.csv", {i.patient_id for i in inputs.values()}, protocol["outer_folds"]
        )
        inventory = json.loads((source / "exports.json").read_text())
        baseline = {
            (e["biopsy_id"], e["outer_fold"], e["component"]): e
            for e in inventory
            if e["arm"] == "HRP" and e["seed"] == 0 and e["component"] in ("full", "transfer")
        }
        sources[backbone] = (source, prepared, inputs, folds, baseline)
        source_records[backbone] = {
            **declaration,
            "input_metadata_sha256": identity["input_metadata_sha256"],
        }
    left, right = [set(i.patient_id for i in value[2].values()) for value in sources.values()]
    if left != right:
        raise ValueError("probe backbones must cover identical biological patients")
    source_files = {
        str(path): _sha(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
    }
    identity = {
        "protocol_sha256": _sha(config_path),
        "sources": source_records,
        "source_file_sha256": source_files,
        "prior_cohort_exposure": True,
        "test_targets_used_for_model_selection": False,
        "seed": 0,
    }
    _write(output / "identity.json", identity)
    all_scores, gene_rows, contrasts, exports, selections = [], [], [], [], []
    for backbone, (source, prepared, inputs, folds, baseline) in sources.items():
        for fold, roles in folds.items():
            directory = output / backbone / f"fold-{fold}"
            directory.mkdir(parents=True)
            by_role = {
                role: [b for b, i in inputs.items() if roles[i.patient_id] == role]
                for role in ("train", "validation", "test")
            }
            target_identity = {}

            def samples(role):
                result = []
                for biopsy in by_role[role]:
                    path = source / f"fold-{fold}" / "targets" / f"{biopsy}.npz"
                    result.append(_sample_from_source(inputs[biopsy], path))
                    target_identity[biopsy] = {
                        "role": role,
                        "path": str(path),
                        "sha256": _sha(path),
                    }
                return result

            print(
                f"frozen feature probe: backbone={backbone} fold={fold} candidates={len(ALPHAS)}",
                flush=True,
            )
            fitted = fit_feature_probe(
                samples("train"), samples("validation"), genes, panel, protocol["alpha_grid"]
            )
            fitted.save(directory / "probe.bundle")
            restored = FrozenFeatureProbe.load(directory / "probe.bundle")
            if not np.array_equal(
                fitted.predict(inputs[by_role["train"][0]]),
                restored.predict(inputs[by_role["train"][0]]),
            ):
                raise ValueError("probe save/reload changed direct normalized-log predictions")
            selections.append({"backbone": backbone, "outer_fold": fold, **fitted.selection})
            fold_exports = []
            for biopsy in by_role["test"]:
                path = directory / f"test-{biopsy}.probe.npz"
                ids = {"backbone": backbone, "outer_fold": fold, "biopsy_id": biopsy, "seed": 0}
                _export_probe(restored, inputs[biopsy], path, ids)
                fold_exports.append({"path": str(path.resolve()), **ids})
            _write(
                directory / "test_predictions_complete.json",
                {
                    "test_targets_loaded": False,
                    "test_biopsies": by_role["test"],
                    "prediction_sha256": {
                        Path(e["path"]).name: _sha(e["path"]) for e in fold_exports
                    },
                    "selected_readouts": 1,
                    "candidate_solutions": len(ALPHAS),
                    "additional_refits": 0,
                },
            )
            # Only now load these held-out targets and the already completed HRP scores.
            for export in fold_exports:
                biopsy = export["biopsy_id"]
                target_path = source / f"fold-{fold}" / "targets" / f"{biopsy}.npz"
                sample = _sample_from_source(inputs[biopsy], target_path)
                target_identity[biopsy] = {
                    "role": "test",
                    "path": str(target_path),
                    "sha256": _sha(target_path),
                }
                predicted = _load_probe_prediction(export["path"], inputs[biopsy], genes)
                protein_inputs, _ = load_joint_inputs(
                    prepared / "inputs" / biopsy, modalities=("P",)
                )
                _, coverage = _coverage_target(
                    (sample.targets, sample.observation, sample.observation_groups, {}),
                    protein_inputs,
                )
                retained = set(coverage["evaluation_observation_ids"])
                common_mask = np.asarray(
                    [row in retained for row in sample.targets.observation_ids]
                )
                for support, row_mask in (
                    ("all_target_eligible", None),
                    ("common_original_protein_coverage", common_mask),
                ):
                    report, errors = score_probe(predicted, sample, genes, panel, row_mask=row_mask)
                    ids = {k: v for k, v in export.items() if k != "path"}
                    report.update(
                        ids,
                        arm="frozen_feature_ridge",
                        evaluation_support=support,
                        prediction_sha256=_sha(export["path"]),
                    )
                    if row_mask is not None:
                        report["coverage_support"] = coverage
                    label = "all" if row_mask is None else "common-coverage"
                    _write(directory / f"test-{biopsy}.{label}.scores.json", report)
                    all_scores.append(report)
                    gene_rows.extend({**ids, "evaluation_support": support, **r} for r in errors)
                    for component in ("full", "transfer"):
                        entry = baseline[(biopsy, fold, component)]
                        path = Path(entry["path"])
                        suffix = (
                            ".scores.json" if row_mask is None else ".common-coverage.scores.json"
                        )
                        saved = json.loads(path.with_suffix(suffix).read_text())
                        if saved["prediction_sha256"] != _sha(path):
                            raise ValueError("saved HRP comparator prediction checksum differs")
                        _assert_matched_support(report, saved)
                        saved[VARIABILITY] = _verify_baseline_truth(
                            path, saved, sample, genes, panel, row_mask
                        )
                        contrasts.append(
                            {
                                **ids,
                                "patient_id": sample.inputs.patient_id,
                                "specimen_id": sample.inputs.specimen_id,
                                "evaluation_support": support,
                                "contrast": f"probe-minus-HRP-{component}",
                                "baseline_prediction_sha256": saved["prediction_sha256"],
                                **{
                                    "delta_" + key: report[key] - saved[key]
                                    if report[key] is not None and saved[key] is not None
                                    else None
                                    for key in ENDPOINTS
                                },
                            }
                        )
            _write(directory / "target_identity.json", target_identity)
            if any(_sha(record["path"]) != record["sha256"] for record in target_identity.values()):
                raise ValueError(
                    "a consumed probe target table changed during fitting or evaluation"
                )
            exports.extend(fold_exports)
    summary = {
        "schema": "heir.frozen_feature_probe_result.v1",
        "selected_readouts": len(selections),
        "candidate_solutions": sum(r["candidate_solutions"] for r in selections),
        "additional_refits": 0,
        "seed": 0,
        "prediction_scale": SCALE,
        "normalize_prediction_again": False,
        "selections": selections,
        "backbones": {},
        "paired_summary": {},
        "scientific_qualification": False,
        "independent_confirmation": False,
    }
    for backbone in sources:
        summary["backbones"][backbone] = {}
        summary["paired_summary"][backbone] = {}
        for support in ("all_target_eligible", "common_original_protein_coverage"):
            selected = [
                r
                for r in all_scores
                if r["backbone"] == backbone and r["evaluation_support"] == support
            ]
            summary["backbones"][backbone][support] = _summary(selected)
            summary["paired_summary"][backbone][support] = {}
            for component in ("full", "transfer"):
                rows = [
                    r
                    for r in contrasts
                    if r["backbone"] == backbone
                    and r["evaluation_support"] == support
                    and r["contrast"] == f"probe-minus-HRP-{component}"
                ]
                patient_rows = [
                    {**r, **{key: r["delta_" + key] for key in ENDPOINTS}} for r in rows
                ]
                summary["paired_summary"][backbone][support][f"probe-minus-HRP-{component}"] = (
                    _summary(patient_rows)
                )
    if summary["selected_readouts"] != 10 or summary["candidate_solutions"] != 50:
        raise ValueError("completed probe inventory differs from its exact declared budget")
    if any(_sha(path) != digest for path, digest in source_files.items()):
        raise ValueError("probe source code changed during execution")
    _write(output / "exports.json", exports)
    _write(output / "summary.json", summary)
    _tsv(
        output / "patient_scores.tsv",
        [
            {
                key: r[key]
                for key in (
                    "backbone",
                    "outer_fold",
                    "biopsy_id",
                    "patient_id",
                    "specimen_id",
                    "evaluation_support",
                    "headline_status",
                    *ENDPOINTS,
                )
            }
            for r in all_scores
        ],
    )
    _tsv(output / "gene_scores.tsv", gene_rows)
    _tsv(output / "paired_contrasts.tsv", contrasts)
    return summary
