"""Saved residual energy/alignment diagnostics; no fitting or new predictor selection.

Every multiplier below is computed using evaluation outcomes and is a diagnostic,
never a deployable correction. Repeated outer-training appearances are averaged
within specimen before equal-specimen/patient aggregation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .crossfit import _target_identity, load_crossfit_baseline
from .joint_workflow import _tsv
from .probe_workflow import (
    ENDPOINTS,
    SCALE,
    FrozenFeatureProbe,
    _region_rows,
    _sample_from_source,
    _truth,
    _write,
    score_probe,
)
from .protein_features import load_joint_inputs
from .residual_train import FittedResidualModel
from .residual_workflow import ARMS, SUPPORTS, _protocol, _source_contract, load_residual_prediction
from .study import _sha

MOMENTS = (
    "baseline_mse",
    "final_mse",
    "residual_energy",
    "error_alignment",
    "baseline_error_mean",
    "residual_mean",
    "centered_baseline_mse",
    "centered_final_mse",
    "centered_residual_energy",
    "centered_error_alignment",
)
SCORES = tuple(prefix + key for prefix in ("baseline_", "final_") for key in ENDPOINTS)
NUMERIC = (*MOMENTS, *SCORES)


def _mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def _derived(moments):
    result = dict(moments)
    for prefix in ("", "centered_"):
        energy, alignment, base, final = (
            moments[prefix + key]
            for key in ("residual_energy", "error_alignment", "baseline_mse", "final_mse")
        )
        if any(value is None for value in (energy, alignment, base, final)):
            values = dict(
                mse_change=None,
                identity_error=None,
                diagnostic_optimal_multiplier=None,
                diagnostic_shrink_multiplier=None,
                diagnostic_optimal_mse_change=None,
                diagnostic_shrink_mse_change=None,
                direction="undefined_support",
            )
        else:
            change, expected = final - base, energy - 2 * alignment
            error = abs(change - expected)
            if not np.isfinite([energy, alignment, base, final]).all() or not np.isclose(
                change, expected, atol=1e-10, rtol=1e-10
            ):
                raise ValueError("residual energy/alignment identity differs")
            alpha = alignment / energy if energy else None
            shrink = float(np.clip(alpha, 0, 1)) if alpha is not None else None
            direction = (
                "zero_residual"
                if energy == 0
                else "noncorrective_direction"
                if alignment <= 0
                else "beneficial_at_multiplier_one"
                if 2 * alignment > energy
                else "aligned_but_multiplier_one_too_large"
                if 2 * alignment < energy
                else "break_even_at_multiplier_one"
            )
            values = dict(
                mse_change=change,
                identity_error=error,
                diagnostic_optimal_multiplier=alpha,
                diagnostic_shrink_multiplier=shrink,
                diagnostic_optimal_mse_change=-alignment * alpha if energy else 0.0,
                diagnostic_shrink_mse_change=energy * shrink**2 - 2 * alignment * shrink
                if energy
                else 0.0,
                direction=direction,
            )
        result.update({prefix + key: value for key, value in values.items()})
    result["squared_bias_change"] = (
        result["mse_change"] - result["centered_mse_change"]
        if result["mse_change"] is not None and result["centered_mse_change"] is not None
        else None
    )
    return result


def _moments(error, delta):
    if not len(error):
        return dict.fromkeys(MOMENTS)
    final = error - delta
    centered_e, centered_d = error - error.mean(), delta - delta.mean()
    return dict(
        baseline_mse=float(np.mean(error**2)),
        final_mse=float(np.mean(final**2)),
        residual_energy=float(np.mean(delta**2)),
        error_alignment=float(np.mean(error * delta)),
        baseline_error_mean=float(error.mean()),
        residual_mean=float(delta.mean()),
        centered_baseline_mse=float(np.mean(centered_e**2)),
        centered_final_mse=float(np.mean((final - final.mean()) ** 2)),
        centered_residual_energy=float(np.mean(centered_d**2)),
        centered_error_alignment=float(np.mean(centered_e * centered_d)),
    )


def residual_decomposition(baseline, residual, sample, gene_ids, panel, *, row_mask=None):
    """Use the existing evaluator's row/gene weights and within-replicate centering."""
    mapping = _region_rows(sample)
    truth, observed = _truth(sample, gene_ids, panel)
    baseline, residual = np.asarray(baseline), np.asarray(residual)
    shape = (len(sample.inputs.cell_ids), len(gene_ids))
    if any(
        values.shape != shape or values.dtype != np.float64 or not np.isfinite(values).all()
        for values in (baseline, residual)
    ):
        raise ValueError("diagnostics require finite float64 direct outcomes on the named axes")
    selected = np.ones(len(mapping), bool) if row_mask is None else np.asarray(row_mask)
    if selected.shape != (len(mapping),) or selected.dtype != bool:
        raise ValueError("diagnostic support must be a boolean observation-row mask")
    baseline, residual = baseline[mapping], residual[mapping]
    labels, rows, replicates = np.asarray(sample.observation_groups), [], []
    for group in sorted(set(labels)):
        group_rows = (labels == group) & selected
        n_observed = int(observed[group_rows].sum())
        current = []
        for column, gene in enumerate(gene_ids):
            keep = group_rows & observed[:, column]
            error = truth[keep, column].astype(np.float64) - baseline[keep, column]
            delta = residual[keep, column]
            current.append(
                {
                    "assay_replicate": group,
                    "gene_id": gene,
                    "n_observed": len(error),
                    "weight_in_replicate": len(error) / n_observed if n_observed else 0.0,
                    **_derived(_moments(error, delta)),
                }
            )
        moments = {
            key: sum(r[key] * r["weight_in_replicate"] for r in current if r["n_observed"])
            if n_observed
            else None
            for key in MOMENTS
        }
        replicates.append(
            {"assay_replicate": group, "n_rows": int(group_rows.sum()), **_derived(moments)}
        )
        rows.extend(current)
    missing = any(not row["n_rows"] for row in replicates)
    moments = {key: None if missing else _mean([r[key] for r in replicates]) for key in MOMENTS}
    for row in rows:
        row["weight_in_specimen"] = row["weight_in_replicate"] / len(replicates)
    return (
        {
            **_derived(moments),
            "n_observations": int(selected.sum()),
            "n_replicates": len(replicates),
            "n_observed_entries": int(observed[selected].sum()),
            "missing_coverage_replicate": missing,
        },
        replicates,
        rows,
    )


def aggregate_residual_reports(records):
    """Average repeated fold instances within specimen, then specimens and patients.

    This preserves equal-patient weight when training patients recur in unequal
    numbers of outer folds. Fold-specific summaries use this same function.
    """
    records = list(records)
    keys = [(r["outer_fold"], r["patient_id"], r["specimen_id"]) for r in records]
    if not records or len(set(keys)) != len(keys):
        raise ValueError("residual aggregation requires unique nonempty specimen/fold instances")
    specimens = defaultdict(list)
    for row in records:
        specimens[(row["patient_id"], row["specimen_id"])].append(row)
    patients = defaultdict(list)
    for (patient, _), current in sorted(specimens.items()):
        patients[patient].append({key: _mean([r.get(key) for r in current]) for key in NUMERIC})
    patient_rows = [
        {
            "patient_id": patient,
            "n_specimens": len(current),
            "n_fold_instances": sum(r["patient_id"] == patient for r in records),
            **_derived({key: _mean([r[key] for r in current]) for key in NUMERIC}),
        }
        for patient, current in sorted(patients.items())
    ]
    return {
        "n_patients": len(patients),
        "n_specimens": len(specimens),
        "n_fold_instances": len(records),
        **_derived({key: _mean([r[key] for r in patient_rows]) for key in NUMERIC}),
        "patients": patient_rows,
    }


class _Sources:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = _sha(path)
        if (expected is not None and digest != expected) or (
            str(path) in self.hashes and digest != self.hashes[str(path)]
        ):
            raise ValueError(f"diagnostic source identity changed: {path}")
        self.hashes[str(path)] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())

    def inventory(self, directory):
        directory = Path(directory)
        for name, digest in self.json(directory / "sha256.json").items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("diagnostic inventory path escapes its bundle")
            self.bind(path, digest)


def _historical_module_hash(path, hashes):
    """Resolve a package module independently of its checkout/installation root.

    Historical receipts use absolute filenames. Match the complete package-relative
    identity, never a bare basename, and reject duplicate identities even when their
    hashes agree. The caller still verifies the current file's exact bytes.
    """
    module = ("heir", "personalized", Path(path).name)
    matches = [digest for name, digest in hashes.items() if Path(name).parts[-3:] == module]
    if len(matches) != 1:
        raise ValueError(
            f"historical module identity requires exactly one source: {'/'.join(module)} "
            f"({len(matches)} matches)"
        )
    digest = matches[0]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"historical module identity requires a SHA-256: {'/'.join(module)}")
    return digest


def _compare_scores(actual, expected, keys=ENDPOINTS):
    for key in keys:
        a, b = actual[key], expected[key]
        if (a is None) != (b is None) or (
            a is not None and not np.isclose(a, b, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError(f"saved residual scores differ for {key}")


def diagnose_residual_study(study, output):
    """Replay selected v0.12 bundles and diagnose their saved outcomes without fitting."""
    study, output, sources = Path(study).resolve(), Path(output).resolve(), _Sources()
    if output.exists():
        raise FileExistsError(output)
    summary = sources.json(study / "summary.json")
    identity = sources.json(study / "identity.json")
    protocol_path = sources.bind(study / "protocol.json", identity["protocol_sha256"])
    protocol, config = _protocol(protocol_path)
    prepared, source, baseline_root, metadata, h_inputs, folds, _ = _source_contract(
        protocol, config
    )
    for directory in (study, prepared, source, baseline_root):
        if output.is_relative_to(directory.resolve()) or directory.resolve().is_relative_to(output):
            raise ValueError("diagnostic output must be separate from all historical sources")
    if summary["schema"] != "heir.residual_study_result.v1" or (
        summary["neural_fits"],
        summary["inner_readouts"],
        summary["reused_deployment_readouts"],
    ) != (30, 15, 5):
        raise ValueError("diagnostics require the completed fixed residual study")
    if (study.parent / "execution.json").is_file():
        execution = sources.json(study.parent / "execution.json")
        if execution["status"] != "complete" or execution["summary_sha256"] != _sha(
            study / "summary.json"
        ):
            raise ValueError("completed execution receipt differs")
    numerical_names = (
        "residual_model",
        "residual_train",
        "residual_workflow",
        "crossfit",
        "probe_workflow",
        "joint_model",
        "train",
        "observation",
        "schema",
        "protein_features",
        "serialization",
        "study",
        "joint_workflow",
        "error_decomposition",
        "evaluate",
        "reference",
        "regions",
        "geometry",
        "correspondence",
        "model",
        "data",
    )
    old_hashes = identity["source_file_sha256"]
    for name in numerical_names:
        path = Path(__file__).with_name(name + ".py")
        sources.bind(path, _historical_module_hash(path, old_hashes))
    for path in (
        __file__,
        Path(__file__).with_name("cohort_cli.py"),
        source / "identity.json",
        source / "config.json",
        source / "exports.json",
        source / "folds.csv",
        prepared / "prepared.json",
    ):
        sources.bind(path)
    exports = sources.json(study / "exports.json")
    n_test = sum(
        roles[item.patient_id] == "test" for roles in folds.values() for item in h_inputs.values()
    )
    expected_exports = {
        "baseline": n_test,
        "final": n_test * len(ARMS),
        "transfer": n_test * sum("R" in arm.split("_")[0] for arm in ARMS),
    }
    if Counter(e["component"] for e in exports) != expected_exports:
        raise ValueError("completed study export inventory differs")
    export_index = {(e["outer_fold"], e["arm"], e["biopsy_id"], e["component"]): e for e in exports}
    if len(export_index) != len(exports):
        raise ValueError("duplicate residual export identity")
    selections = {(s["outer_fold"], s["arm"]): s for s in summary["selections"]}
    if set(selections) != {(fold, arm) for fold in folds for arm in ARMS}:
        raise ValueError("selected bundle inventory differs from frozen folds and arms")
    cache = {}

    def inputs(biopsy, modalities):
        key = biopsy, tuple(modalities)
        if key not in cache:
            directory = prepared / "inputs" / biopsy
            record = sources.json(
                directory / "metadata.json", identity["input_metadata_sha256"][biopsy]
            )
            names = [
                "geometry",
                *(
                    name
                    for code, name in (("H", "image"), ("R", "reference"), ("P", "protein"))
                    if code in modalities
                ),
            ]
            if "P" in modalities and "protein_correspondence" in record["payloads"]:
                names.append("protein_correspondence")
            for name in names:
                payload = record["payloads"][name]
                sources.bind(directory / payload["file"], payload["sha256"])
            cache[key] = load_joint_inputs(directory, modalities=tuple(modalities))[0]
        return cache[key]

    records, gene_rows, replicate_rows, shift_rows, selection_rows = [], [], [], [], []
    role_counts, test_replays, validation_checks = Counter(), 0, 0
    genes, panel = config.gene_ids, config.normalization_gene_ids
    for fold, roles in folds.items():
        directory = study / f"fold-{fold}"
        barrier = sources.json(directory / "test_predictions_complete.json")
        current_exports = [e for e in exports if e["outer_fold"] == fold]
        if barrier["test_targets_loaded"] is not False or barrier["prediction_sha256"] != {
            Path(e["path"]).name: e["sha256"] for e in current_exports
        }:
            raise ValueError("residual test prediction barrier differs")
        for entry in current_exports:
            sources.bind(entry["path"], entry["sha256"])
        targets = sources.json(directory / "target_identity.json")
        if set(targets) != set(h_inputs):
            raise ValueError("target inventory differs from prepared specimens")
        samples = {}
        for biopsy, declared in targets.items():
            if declared["role"] != roles[h_inputs[biopsy].patient_id]:
                raise ValueError("target patient role differs from frozen folds")
            if (
                Path(declared["path"]).resolve()
                != (source / f"fold-{fold}/targets/{biopsy}.npz").resolve()
            ):
                raise ValueError("target source path differs from the completed study")
            path = sources.bind(declared["path"], declared["sha256"])
            samples[biopsy] = _sample_from_source(inputs(biopsy, ("H",)), path)
            role_counts[declared["role"]] += 1
        sources.inventory(directory / "crossfit")
        crossfit = load_crossfit_baseline(directory / "crossfit")
        if set(crossfit.training_patient_ids) != {
            p for p, role in roles.items() if role == "train"
        }:
            raise ValueError("crossfit patient inventory differs from frozen outer-training roles")
        retained = baseline_root / f"fold-{fold}" / "probe.bundle"
        sources.bind(
            retained / "sha256.json",
            protocol["baseline_folds"][str(fold)]["bundle_inventory_sha256"],
        )
        sources.inventory(retained)
        deployment = FrozenFeatureProbe.load(retained)
        for record in crossfit.source_records:
            sample = next(
                s
                for s in samples.values()
                if (s.inputs.patient_id, s.inputs.specimen_id)
                == (record["patient_id"], record["specimen_id"])
            )
            if record["target_identity"] != _target_identity(sample):
                raise ValueError("crossfit target identity differs from current training outcomes")
            expected = crossfit.models[record["inner_fold"]].predict(sample.inputs)
            if not np.array_equal(crossfit.prediction_for(sample.inputs), expected):
                raise ValueError("crossfit values differ from the patient-excluding saved model")
        for arm in ARMS:
            entry = selections[(fold, arm)]
            bundle = Path(entry["bundle"])
            if bundle.resolve() != (directory / f"{arm}.bundle").resolve():
                raise ValueError("selected residual bundle path differs")
            sources.inventory(bundle)
            fitted = FittedResidualModel.load(bundle)
            if fitted.baseline.alpha != protocol["alpha"] or any(
                not np.array_equal(getattr(fitted.baseline, name), getattr(deployment, name))
                for name in ("feature_mean", "feature_scale", "coefficients", "intercept")
            ):
                raise ValueError("residual deployment baseline differs from the retained ridge")
            if (
                fitted.selection != entry["selection"]
                or set(fitted.training_patient_ids)
                != {p for p, role in roles.items() if role == "train"}
                or set(fitted.validation_patient_ids)
                != {p for p, role in roles.items() if role == "validation"}
            ):
                raise ValueError("selected residual bundle roles or checkpoint identity differs")
            selected_epoch = fitted.selection["selected_epoch"]
            history = fitted.history
            if [row["epoch"] for row in history] != list(range(101)):
                raise ValueError("residual checkpoint history is incomplete")
            selection_rows.append(
                {
                    "outer_fold": fold,
                    "arm": arm,
                    "selected_epoch": selected_epoch,
                    "selected_zero_residual": selected_epoch == 0,
                    **{
                        f"zero_validation_{key}": history[0]["validation"][key] for key in ENDPOINTS
                    },
                    **{
                        f"selected_validation_{key}": fitted.selection["validation"][key]
                        for key in ENDPOINTS
                    },
                }
            )
            validation = []
            for biopsy, sample in samples.items():
                role = targets[biopsy]["role"]
                modalities = ("H", "P") if arm == "HRP_generic" else fitted.model.config.modalities
                prediction = fitted.predict(inputs(biopsy, modalities))
                baseline, residual = (
                    getattr(prediction, key).cpu().numpy() for key in ("image_baseline", "residual")
                )
                if role == "test":
                    export = export_index[(fold, arm, biopsy, "final")]
                    saved, _ = load_residual_prediction(export["path"], sample.inputs, genes)
                    if any(
                        not np.array_equal(saved[key], getattr(prediction, key).cpu().numpy())
                        for key in saved
                    ):
                        raise ValueError("selected residual replay differs from saved test output")
                    test_replays += 1
                views = {"deployment": baseline}
                if role == "train":
                    views["inner_crossfit"] = crossfit.prediction_for(sample.inputs)
                mapping = _region_rows(sample)
                covered = inputs(biopsy, ("H", "P")).protein_features.observed.any(1)[mapping]
                for support, mask in zip(SUPPORTS, (None, covered)):
                    ids = dict(
                        outer_fold=fold,
                        arm=arm,
                        patient_id=sample.inputs.patient_id,
                        specimen_id=sample.inputs.specimen_id,
                        biopsy_id=biopsy,
                        role=role,
                        evaluation_support=support,
                    )
                    for view, base in views.items():
                        moments, replicates, detailed = residual_decomposition(
                            base, residual, sample, genes, panel, row_mask=mask
                        )
                        base_score = score_probe(base, sample, genes, panel, row_mask=mask)[0]
                        final_score = score_probe(
                            base + residual, sample, genes, panel, row_mask=mask
                        )[0]
                        _compare_scores(
                            {
                                "normalized_log_mse": moments["baseline_mse"],
                                "centered_mse": moments["centered_baseline_mse"],
                            },
                            base_score,
                            ("normalized_log_mse", "centered_mse"),
                        )
                        _compare_scores(
                            {
                                "normalized_log_mse": moments["final_mse"],
                                "centered_mse": moments["centered_final_mse"],
                            },
                            final_score,
                            ("normalized_log_mse", "centered_mse"),
                        )
                        row = {
                            **ids,
                            "baseline_view": view,
                            **moments,
                            **{
                                prefix + key: score[key]
                                for prefix, score in (
                                    ("baseline_", base_score),
                                    ("final_", final_score),
                                )
                                for key in ENDPOINTS
                            },
                        }
                        records.append(row)
                        replicate_rows.extend(
                            {**ids, "baseline_view": view, **r} for r in replicates
                        )
                        gene_rows.extend({**ids, "baseline_view": view, **r} for r in detailed)
                        if role == "validation" and support == SUPPORTS[0]:
                            validation.append(row)
                        if role == "test":
                            path = Path(export_index[(fold, arm, biopsy, "final")]["path"])
                            _compare_scores(
                                final_score,
                                sources.json(directory / f"{path.stem}.{support}.scores.json"),
                            )
                    if role == "train" and arm == ARMS[0]:
                        inner = views["inner_crossfit"]
                        shift = residual_decomposition(
                            inner, baseline - inner, sample, genes, panel, row_mask=mask
                        )[0]
                        shift_rows.append({**{k: v for k, v in ids.items() if k != "arm"}, **shift})
            validation_summary = aggregate_residual_reports(validation)
            _compare_scores(
                {key: validation_summary["final_" + key] for key in ENDPOINTS},
                fitted.selection["validation"],
            )
            _compare_scores(
                {key: validation_summary["baseline_" + key] for key in ENDPOINTS},
                history[0]["validation"],
            )
            validation_checks += 1
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["arm"], row["role"], row["baseline_view"], row["evaluation_support"])].append(
            row
        )
    summaries, patient_rows, fold_summaries = [], [], []
    for key, selected in sorted(grouped.items()):
        ids = dict(zip(("arm", "role", "baseline_view", "evaluation_support"), key))
        result = aggregate_residual_reports(selected)
        patient_rows.extend({**ids, **r} for r in result.pop("patients"))
        summaries.append({**ids, **result})
        for fold in sorted({r["outer_fold"] for r in selected}):
            value = aggregate_residual_reports([r for r in selected if r["outer_fold"] == fold])
            value.pop("patients")
            fold_summaries.append({**ids, "outer_fold": fold, **value})
    for row in summaries:
        if row["role"] == "test":
            retained = summary["scores"][row["evaluation_support"]][row["arm"] + ":final"]
            _compare_scores({key: row["final_" + key] for key in ENDPOINTS}, retained)
    for path, digest in sources.hashes.copy().items():
        sources.bind(path, digest)
    result = {
        "schema": "heir.saved_residual_diagnostics.v1",
        "prediction_scale": SCALE,
        "model_fits": 0,
        "checkpoint_selection_changes": 0,
        "predictions_modified": False,
        "diagnostic_multipliers_use_evaluation_targets": True,
        "diagnostic_multipliers_are_validated_predictors": False,
        "scientific_qualification": False,
        "independent_confirmation": False,
        "source_study": str(study),
        "source_summary_sha256": _sha(study / "summary.json"),
        "source_roles": dict(role_counts),
        "selected_bundles": len(selections),
        "exact_test_prediction_replays": test_replays,
        "selected_validation_and_zero_candidate_replays": validation_checks,
        "weighting": (
            "observed entries within replicate; equal replicates; repeated folds within specimen; "
            "equal specimens within patient; equal patients"
        ),
        "centering": "within each observed replicate/gene before aggregation",
        "baseline_shift": (
            "inner and deployment errors compared on identical training observations; "
            "deployment training errors are in-sample; no causal attribution"
        ),
        "baseline_shift_table_columns": (
            "baseline_mse=inner_crossfit_error; final_mse=deployment_error; "
            "residual_energy=mean_squared_deployment_minus_inner_prediction"
        ),
        "interpretation": (
            "selected-model failure diagnosis on exposed development data; "
            "no retraining, altered predictor, or independent confirmation"
        ),
        "summaries": summaries,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "summary.json", result)
    _write(output / "consumed_sha256.json", sources.hashes)
    for name, rows in (
        ("specimen_moments", records),
        ("replicate_moments", replicate_rows),
        ("replicate_gene_moments", gene_rows),
        ("patient_moments", patient_rows),
        ("fold_moments", fold_summaries),
        ("selection", selection_rows),
        ("baseline_shift", shift_rows),
    ):
        _tsv(output / (name + ".tsv"), rows)
    return result
