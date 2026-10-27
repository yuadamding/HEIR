"""Independent saved-array residual audit. No diagnostic/scoring helper imports or fits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

# Existing loaders/forward model only; normalization and moments below are independent.
from heir.personalized.protein_features import load_joint_inputs
from heir.personalized.residual_train import FittedResidualModel

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
CORRELATION = "mean_replicate_median_gene_spearman"
AMPLITUDE = "mean_replicate_gene_sd_ratio_diagnostic"
ENDPOINTS = (CORRELATION, "normalized_log_mse", "squared_bias", "centered_mse", AMPLITUDE)
NUMERIC = MOMENTS + tuple(prefix + key for prefix in ("baseline_", "final_") for key in ENDPOINTS)
GROUP = ("arm", "role", "baseline_view", "evaluation_support")
CASE = ("outer_fold", "arm", "biopsy_id", "role", "baseline_view", "evaluation_support")
SUPPORTS = ("all_target_eligible", "common_original_protein_coverage")
ARMS = ("H", "HR", "HP", "HRP", "HRP_coverage_only", "HRP_generic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.bound, self.maximum, self.checks = {}, defaultdict(float), 0

    def require(self, condition, message):
        self.checks += 1
        if not condition:
            raise AssertionError(message)

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        self.require(expected is None or digest == expected, f"source digest: {path}")
        self.require(
            str(path) not in self.bound or self.bound[str(path)] == digest,
            f"source changed: {path}",
        )
        self.bound[str(path)] = digest
        return path

    def js(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())

    def npz(self, path, expected=None):
        with np.load(self.bind(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def numeric(self, actual, expected, label):
        if isinstance(expected, str):
            expected = None if expected == "" else float(expected)
        self.require((actual is None) == (expected is None), f"definedness {label}")
        if actual is None:
            return
        difference = abs(float(actual) - float(expected))
        self.maximum[label.split(":")[-1]] = max(self.maximum[label.split(":")[-1]], difference)
        self.require(
            np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-10, atol=1e-10),
            f"numeric {label}: {actual} != {expected}",
        )


def derived(row):
    row = dict(row)
    for prefix in ("", "centered_"):
        b, f, energy, alignment = (
            row[prefix + k]
            for k in ("baseline_mse", "final_mse", "residual_energy", "error_alignment")
        )
        if None in (b, f, energy, alignment):
            values = {
                key: None
                for key in (
                    "mse_change",
                    "diagnostic_optimal_multiplier",
                    "diagnostic_shrink_multiplier",
                    "diagnostic_optimal_mse_change",
                    "diagnostic_shrink_mse_change",
                )
            }
            values["direction"] = "undefined_support"
        else:
            alpha = alignment / energy if energy > 0 else None
            shrink = min(1.0, max(0.0, alpha)) if alpha is not None else None
            direction = "zero_residual"
            if energy > 0:
                direction = (
                    "noncorrective_direction"
                    if alignment <= 0
                    else "beneficial_at_multiplier_one"
                    if energy < 2 * alignment
                    else "aligned_but_multiplier_one_too_large"
                    if energy > 2 * alignment
                    else "break_even_at_multiplier_one"
                )
            values = {
                "mse_change": f - b,
                "diagnostic_optimal_multiplier": alpha,
                "diagnostic_shrink_multiplier": shrink,
                "diagnostic_optimal_mse_change": -(alignment**2) / energy if energy else 0.0,
                "diagnostic_shrink_mse_change": energy * shrink * shrink - 2 * alignment * shrink
                if energy
                else 0.0,
                "direction": direction,
            }
        row.update({prefix + key: value for key, value in values.items()})
    row["squared_bias_change"] = (
        row["mse_change"] - row["centered_mse_change"]
        if row["mse_change"] is not None and row["centered_mse_change"] is not None
        else None
    )
    return row


def vector_moments(error, delta):
    if not len(error):
        return dict.fromkeys(MOMENTS)
    # Dot-product moments use a different reduction from production elementwise means.
    ec, dc = error - np.mean(error), delta - np.mean(delta)
    final = error - delta
    fc = final - np.mean(final)
    n = len(error)
    return dict(
        zip(
            MOMENTS,
            (
                float(error @ error / n),
                float(final @ final / n),
                float(delta @ delta / n),
                float(error @ delta / n),
                float(sum(error) / n),
                float(sum(delta) / n),
                float(ec @ ec / n),
                float(fc @ fc / n),
                float(dc @ dc / n),
                float(ec @ dc / n),
            ),
        )
    )


def mean(values, strict=True):
    if not values or (strict and any(v is None for v in values)):
        return None
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def evaluate(baseline, delta, truth, observed, groups, selected, genes):
    reps, details, replicate_scores = [], [], []
    for label in sorted(set(groups)):
        in_group = (groups == label) & selected
        count = int(observed[in_group].sum())
        gene_moments, gene_weights = [], []
        scores = {
            prefix: {"correlations": [], "ratios": [], "invalid": False}
            for prefix in ("baseline_", "final_")
        }
        for column, gene in enumerate(genes):
            keep = in_group & observed[:, column]
            y, b, d = truth[keep, column], baseline[keep, column], delta[keep, column]
            moments = vector_moments(y - b, d)
            weight = len(y) / count if count else 0.0
            details.append(
                {
                    "assay_replicate": label,
                    "gene_id": gene,
                    "n_observed": len(y),
                    "weight_in_replicate": weight,
                    "weight_in_specimen": weight / len(set(groups)),
                    **derived(moments),
                }
            )
            if len(y):
                gene_moments.append(moments)
                gene_weights.append(weight)
            for prefix, prediction in (("baseline_", b), ("final_", b + d)):
                if len(y) >= 3 and np.ptp(y) != 0:
                    if np.ptp(prediction) == 0:
                        scores[prefix]["invalid"] = True
                    else:
                        scores[prefix]["correlations"].append(
                            float(
                                np.corrcoef(
                                    rankdata(y, method="average"),
                                    rankdata(prediction, method="average"),
                                )[0, 1]
                            )
                        )
                if len(y) and y.std() > 0:
                    scores[prefix]["ratios"].append(float(prediction.std() / y.std()))
        combined = {
            key: float(sum(w * g[key] for g, w in zip(gene_moments, gene_weights)))
            if count
            else None
            for key in MOMENTS
        }
        reps.append({"assay_replicate": label, "n_rows": int(in_group.sum()), **derived(combined)})
        per_rep = {}
        for prefix in ("baseline_", "final_"):
            state = scores[prefix]
            per_rep[prefix + CORRELATION] = (
                None
                if state["invalid"] or not state["correlations"]
                else float(np.median(state["correlations"]))
            )
            per_rep[prefix + AMPLITUDE] = mean(state["ratios"], strict=False)
            per_rep[prefix + "normalized_log_mse"] = combined[
                "baseline_mse" if prefix == "baseline_" else "final_mse"
            ]
            per_rep[prefix + "centered_mse"] = combined[
                "centered_baseline_mse" if prefix == "baseline_" else "centered_final_mse"
            ]
            per_rep[prefix + "squared_bias"] = (
                per_rep[prefix + "normalized_log_mse"] - per_rep[prefix + "centered_mse"]
                if count
                else None
            )
            per_rep[prefix + "invalid"] = state["invalid"]
        replicate_scores.append(per_rep)
    missing = any(r["n_rows"] == 0 for r in reps)
    specimen = {key: None if missing else mean([r[key] for r in reps]) for key in MOMENTS}
    for prefix in ("baseline_", "final_"):
        for key in ENDPOINTS:
            values = [r[prefix + key] for r in replicate_scores]
            bad = key == CORRELATION and any(r[prefix + "invalid"] for r in replicate_scores)
            specimen[prefix + key] = None if missing or bad else mean(values, strict=False)
    specimen.update(
        n_observations=int(selected.sum()),
        n_replicates=len(reps),
        n_observed_entries=int(observed[selected].sum()),
    )
    return derived(specimen), reps, details


def aggregate(rows):
    """Compute explicit row weights instead of nested production means."""
    patients = sorted({r["patient_id"] for r in rows})
    specimen_sets = {p: {r["specimen_id"] for r in rows if r["patient_id"] == p} for p in patients}
    multiplicity = Counter((r["patient_id"], r["specimen_id"]) for r in rows)
    weighted = [
        (
            r,
            1
            / len(patients)
            / len(specimen_sets[r["patient_id"]])
            / multiplicity[(r["patient_id"], r["specimen_id"])],
        )
        for r in rows
    ]
    result = {
        key: float(sum(weight * r[key] for r, weight in weighted))
        if all(r.get(key) is not None for r in rows)
        else None
        for key in NUMERIC
    }
    return derived(result)


def read_table(audit, path):
    with audit.bind(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def record_key(row, names):
    return tuple(str(row[name]) for name in names)


def compare_record(audit, actual, expected, label):
    for key, value in actual.items():
        if key in ("direction", "centered_direction"):
            audit.require(value == expected[key], label + ":" + key)
        elif (
            key in MOMENTS
            or key in NUMERIC
            or key.startswith(("diagnostic_", "centered_diagnostic_"))
            or key
            in (
                "mse_change",
                "centered_mse_change",
                "squared_bias_change",
                "n_observed",
                "weight_in_replicate",
                "weight_in_specimen",
                "n_observations",
                "n_replicates",
                "n_observed_entries",
                "n_rows",
            )
        ):
            audit.numeric(value, expected[key], label + ":" + key)
    for prefix in ("", "centered_"):
        if actual[prefix + "mse_change"] is not None:
            audit.numeric(
                actual[prefix + "mse_change"],
                actual[prefix + "residual_energy"] - 2 * actual[prefix + "error_alignment"],
                label + ":" + prefix + "independent_identity",
            )


def verify_aggregates(directory):
    """Check published aggregate algebra; this does not replay underlying arrays."""
    audit = Audit()
    publication = audit.js(directory / "provenance.json")
    audit.require(
        publication["schema"] == "heir.residual_aggregate_publication.v1", "publication schema"
    )
    audit.bind(directory / "aggregates.tsv", publication["aggregate_sha256"])
    rows = read_table(audit, directory / "aggregates.tsv")
    keys = {record_key(row, GROUP) for row in rows}
    audit.require(
        len(rows) == len(keys) == publication["historical_audit"]["summary_groups"],
        "complete unique aggregate groups",
    )
    for row in rows:
        moments = {key: None if row[key] == "" else float(row[key]) for key in NUMERIC}
        compare_record(audit, derived(moments), row, "published_aggregate")
    return {
        "status": "passed",
        "scope": "aggregate algebra only; no array reconstruction or new biological audit",
        "summary_groups": len(rows),
        "checks": audit.checks,
        "maximum_absolute_discrepancy": max(audit.maximum.values()),
        "historical_array_reconstruction_maximum_absolute_discrepancy": publication[
            "historical_audit"
        ]["maximum_absolute_discrepancy"],
    }


def main(args):
    audit = Audit()
    result = {
        "schema": "heir.independent_residual_moment_audit.v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__),
        "status": "running",
        "model_fits": 0,
        "predictions_modified": False,
        "runtime": {
            "device": "cpu",
            "torch_threads": 2,
            "torch_interop_threads": 2,
            "openblas_threads": 2,
        },
        "math_implementation": (
            "independent dot-product moments and explicit patient/specimen/fold weights; "
            "no new diagnostic/scoring helper imports"
        ),
        "tolerance": {"atol": 1e-10, "rtol": 1e-10},
    }
    if args.output.exists():
        raise SystemExit("Audit output already exists; preserve existing evidence.")
    try:
        diagnostic = audit.js(args.diagnostics / "summary.json")
        audit.require(
            diagnostic["schema"] == "heir.saved_residual_diagnostics.v1"
            and diagnostic["model_fits"] == 0,
            "completed no-fit diagnostic schema",
        )
        targets_by_table = {}
        for name, key in (
            ("specimen_moments", CASE),
            ("replicate_moments", CASE + ("assay_replicate",)),
            ("replicate_gene_moments", CASE + ("assay_replicate", "gene_id")),
            ("patient_moments", GROUP + ("patient_id",)),
            ("fold_moments", GROUP + ("outer_fold",)),
            ("baseline_shift", ("outer_fold", "biopsy_id", "evaluation_support")),
        ):
            rows = read_table(audit, args.diagnostics / (name + ".tsv"))
            index = {record_key(row, key): row for row in rows}
            audit.require(len(index) == len(rows), "unique " + name)
            targets_by_table[name] = index
        audit.js(args.study / "summary.json")
        protocol = audit.js(args.study / "protocol.json")
        identity = audit.js(args.study / "identity.json")
        audit.require(
            sha(args.study / "summary.json") == diagnostic["source_summary_sha256"],
            "bound original summary",
        )
        exports = audit.js(args.study / "exports.json")
        index = {(r["outer_fold"], r["arm"], r["biopsy_id"], r["component"]): r for r in exports}
        prepared = Path(protocol["prepared_directory"])
        genes, panel = (
            tuple(protocol["model"]["gene_ids"]),
            tuple(protocol["model"]["normalization_gene_ids"]),
        )
        input_cache, metadata_cache, arrays_cache = {}, {}, {}

        def raw(biopsy):
            if biopsy not in metadata_cache:
                folder = prepared / "inputs" / biopsy
                metadata = audit.js(
                    folder / "metadata.json", identity["input_metadata_sha256"][biopsy]
                )
                arrays = {}
                for key, row in metadata["payloads"].items():
                    audit.bind(folder / row["file"], row["sha256"])
                    if key in ("image", "protein", "geometry"):
                        arrays[key] = audit.npz(folder / row["file"])
                metadata_cache[biopsy], arrays_cache[biopsy] = metadata, arrays
            return metadata_cache[biopsy], arrays_cache[biopsy]

        def inputs(biopsy, modalities):
            key = biopsy, tuple(modalities)
            if key not in input_cache:
                raw(biopsy)
                input_cache[key] = load_joint_inputs(
                    prepared / "inputs" / biopsy, modalities=tuple(modalities)
                )[0]
            return input_cache[key]

        records, shifts, roles = [], [], Counter()
        test_count = validation_count = train_count = 0
        for fold in range(5):
            folder = args.study / f"fold-{fold}"
            target_contract = audit.js(folder / "target_identity.json")
            cf = audit.js(folder / "crossfit/crossfit.json")
            cf_index = {(r["patient_id"], r["specimen_id"]): r for r in cf["source_records"]}
            deployment = audit.npz(folder / "H.bundle/baseline.bundle/readout.npz")
            samples = {}
            for biopsy, source in target_contract.items():
                meta, arrays = raw(biopsy)
                roles[source["role"]] += 1
                target = audit.npz(source["path"], source["sha256"])
                axis = tuple(map(str, target["gene_ids"]))
                value = torch.tensor(target["expression"], dtype=torch.float32)
                mask = torch.tensor(target["observed"], dtype=torch.bool)
                columns = [axis.index(g) for g in panel]
                audit.require(bool(mask[:, columns].all()), "complete target normalization panel")
                exposure = value[:, columns].sum(1, keepdim=True)
                audit.require(bool((exposure > 0).all()), "positive target exposure")
                normalized = (
                    torch.log1p(torch.where(mask, value, 0) / exposure * 10000)
                    .numpy()
                    .astype(np.float64)
                )
                order = [axis.index(g) for g in genes]
                truth, observed = normalized[:, order], target["observed"][:, order]
                positive = target["weights"] > 0
                audit.require(
                    np.all(target["weights"][positive] == 1)
                    and np.all(np.bincount(target["rows"][positive], minlength=len(truth)) == 1),
                    "unit region observation mapping",
                )
                lookup = {v: i for i, v in enumerate(meta["cell_ids"])}
                mapping = np.empty(len(truth), int)
                mapping[target["rows"][positive]] = [
                    lookup[str(target["cell_ids"][i])] for i in target["columns"][positive]
                ]
                image = arrays["image"]["image_features"].astype(np.float64)
                baseline = (
                    (image - deployment["feature_mean"]) / deployment["feature_scale"]
                ) @ deployment["coefficients"] + deployment["intercept"]
                inner = None
                if source["role"] == "train":
                    c = cf_index[(meta["patient_id"], meta["specimen_id"])]
                    prediction = audit.npz(
                        folder / "crossfit" / c["prediction_file"], c["prediction_file_sha256"]
                    )
                    audit.require(
                        tuple(prediction["region_ids"]) == tuple(meta["cell_ids"]),
                        "crossfit region order",
                    )
                    inner = prediction["normalized_log_expression"]
                    train_count += 1
                samples[biopsy] = {
                    "meta": meta,
                    "role": source["role"],
                    "truth": truth,
                    "observed": observed,
                    "mapping": mapping,
                    "groups": target["groups"],
                    "covered": arrays["protein"]["observed"].any(1)[mapping],
                    "baseline": baseline,
                    "inner": inner,
                }
            for arm in ARMS:
                bundle = folder / f"{arm}.bundle"
                for name, digest in audit.js(bundle / "sha256.json").items():
                    audit.bind(bundle / name, digest)
                fitted = FittedResidualModel.load(bundle, device="cpu")
                validation = []
                for biopsy, s in samples.items():
                    role, mapping, baseline = s["role"], s["mapping"], s["baseline"]
                    meta = s["meta"]
                    if role == "test":
                        e = index[(fold, arm, biopsy, "final")]
                        saved = audit.npz(e["path"], e["sha256"])
                        audit.require(
                            tuple(saved["gene_ids"]) == genes
                            and tuple(saved["region_ids"]) == tuple(meta["cell_ids"]),
                            "test named axes",
                        )
                        audit.require(
                            np.array_equal(baseline, saved["image_baseline"]),
                            "independent test baseline replay",
                        )
                        audit.require(
                            np.array_equal(
                                baseline + saved["residual"], saved["normalized_log_expression"]
                            ),
                            "test final decomposition",
                        )
                        delta = saved["residual"]
                        test_count += 1
                    else:
                        modalities = (
                            ("H", "P") if arm == "HRP_generic" else fitted.model.config.modalities
                        )
                        forward = fitted.predict(inputs(biopsy, modalities))
                        audit.require(
                            np.array_equal(baseline, forward.image_baseline.cpu().numpy()),
                            "independent selected-model baseline",
                        )
                        delta = forward.residual.cpu().numpy()
                    views = {"deployment": baseline}
                    if role == "train":
                        views["inner_crossfit"] = s["inner"]
                    for support, selected in zip(
                        SUPPORTS, (np.ones(len(mapping), bool), s["covered"])
                    ):
                        ids = dict(
                            outer_fold=fold,
                            arm=arm,
                            biopsy_id=biopsy,
                            role=role,
                            evaluation_support=support,
                            patient_id=meta["patient_id"],
                            specimen_id=meta["specimen_id"],
                        )
                        for view, base in views.items():
                            case = {**ids, "baseline_view": view}
                            moments, reps, gene_rows = evaluate(
                                base[mapping],
                                delta[mapping],
                                s["truth"],
                                s["observed"],
                                s["groups"],
                                selected,
                                genes,
                            )
                            actual = {**case, **moments}
                            compare_record(
                                audit,
                                actual,
                                targets_by_table["specimen_moments"][record_key(case, CASE)],
                                "specimen",
                            )
                            for row in reps:
                                compare_record(
                                    audit,
                                    row,
                                    targets_by_table["replicate_moments"][
                                        record_key({**case, **row}, CASE + ("assay_replicate",))
                                    ],
                                    "replicate",
                                )
                            for row in gene_rows:
                                compare_record(
                                    audit,
                                    row,
                                    targets_by_table["replicate_gene_moments"][
                                        record_key(
                                            {**case, **row}, CASE + ("assay_replicate", "gene_id")
                                        )
                                    ],
                                    "gene",
                                )
                            records.append(actual)
                            if role == "validation" and support == SUPPORTS[0]:
                                validation.append(actual)
                        if role == "train" and arm == "H":
                            shift, _, _ = evaluate(
                                s["inner"][mapping],
                                (baseline - s["inner"])[mapping],
                                s["truth"],
                                s["observed"],
                                s["groups"],
                                selected,
                                genes,
                            )
                            compare_record(
                                audit,
                                {
                                    k: v
                                    for k, v in shift.items()
                                    if k not in NUMERIC or k in MOMENTS
                                },
                                targets_by_table["baseline_shift"][
                                    record_key(
                                        ids, ("outer_fold", "biopsy_id", "evaluation_support")
                                    )
                                ],
                                "baseline_shift",
                            )
                            shifts.append({**ids, **shift})
                val = aggregate(validation)
                for prefix, saved_metrics in (
                    ("final_", fitted.selection["validation"]),
                    ("baseline_", fitted.history[0]["validation"]),
                ):
                    for metric in ENDPOINTS:
                        audit.numeric(
                            val[prefix + metric],
                            saved_metrics[metric],
                            "selected_validation:" + metric,
                        )
                validation_count += 1
        grouped = defaultdict(list)
        for row in records:
            grouped[record_key(row, GROUP)].append(row)
        summaries = {record_key(row, GROUP): row for row in diagnostic["summaries"]}
        audit.require(set(grouped) == set(summaries), "summary groups")
        paired = []
        for key, rows in grouped.items():
            ids = dict(zip(GROUP, key))
            all_values = aggregate(rows)
            compare_record(audit, all_values, summaries[key], "summary")
            patients = sorted({r["patient_id"] for r in rows})
            patient_values = []
            for patient in patients:
                value = aggregate([r for r in rows if r["patient_id"] == patient])
                patient_values.append(value)
                compare_record(
                    audit,
                    value,
                    targets_by_table["patient_moments"][
                        record_key({**ids, "patient_id": patient}, GROUP + ("patient_id",))
                    ],
                    "patient",
                )
            for fold in sorted({r["outer_fold"] for r in rows}):
                value = aggregate([r for r in rows if r["outer_fold"] == fold])
                compare_record(
                    audit,
                    value,
                    targets_by_table["fold_moments"][
                        record_key({**ids, "outer_fold": fold}, GROUP + ("outer_fold",))
                    ],
                    "fold",
                )
            for metric in (
                "mse_change",
                "centered_mse_change",
                "residual_energy",
                "error_alignment",
                "centered_residual_energy",
                "centered_error_alignment",
            ):
                values = [r[metric] for r in patient_values if r[metric] is not None]
                paired.append(
                    {
                        **ids,
                        "metric": metric,
                        "n_patients": len(values),
                        "mean": mean(values),
                        "median": float(np.median(values)) if values else None,
                        "positive": sum(v > 0 for v in values),
                        "negative": sum(v < 0 for v in values),
                        "zero": sum(v == 0 for v in values),
                    }
                )
        audit.require(
            test_count == 78 and validation_count == 30 and train_count == 42,
            "complete saved prediction/validation/crossfit counts",
        )
        audit.require(
            len(records) == len(targets_by_table["specimen_moments"]), "full specimen coverage"
        )
        for path, digest in list(audit.bound.items()):
            audit.bind(path, digest)
        result.update(
            status="passed",
            saved_test_predictions=test_count,
            selected_validation_models=validation_count,
            crossfit_training_instances=train_count,
            original_role_counts=dict(roles),
            specimen_moment_records=len(records),
            baseline_shift_records=len(shifts),
            summary_groups=len(grouped),
            maximum_absolute_discrepancy=dict(audit.maximum),
            checks=audit.checks,
            consumed_sha256=audit.bound,
            diagnostic_summary_sha256=sha(args.diagnostics / "summary.json"),
            patient_breakdowns=paired,
            interpretation=(
                "Analytical target-informed multipliers remain descriptive; no new predictor, "
                "checkpoint choice, fit, or independent confirmation."
            ),
        )
    except BaseException as error:
        result.update(
            status="failed",
            checks=audit.checks,
            error_type=type(error).__name__,
            error=str(error),
            consumed_sha256=audit.bound,
        )
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregates", type=Path, help="Check published aggregates only")
    parser.add_argument("--study", type=Path, help="Original completed v0.12 study")
    parser.add_argument("--diagnostics", type=Path, help="Original v0.13 diagnostic directory")
    parser.add_argument("--output", type=Path, help="New full-audit receipt; must not exist")
    args = parser.parse_args()
    if args.aggregates is not None:
        if any(value is not None for value in (args.study, args.diagnostics, args.output)):
            parser.error("--aggregates cannot be combined with array-reconstruction arguments")
        print(json.dumps(verify_aggregates(args.aggregates), indent=2, allow_nan=False))
    else:
        if any(value is None for value in (args.study, args.diagnostics, args.output)):
            parser.error("array reconstruction requires --study, --diagnostics, and --output")
        if any(os.environ.get(name) != "2" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
            parser.error("array reconstruction requires OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        main(args)
