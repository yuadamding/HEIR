"""One bounded measurement screen; completed model studies remain read-only."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .capability import _EXPORTS, _sha, _table, _write
from .cohort import (
    discover_cohort,
    load_cohort_targets,
    load_measurement_gene_axes,
    load_measurement_substrate,
)
from .cohort_cli import _subset, replicate_scores
from .diagnostics import _region_rows
from .inspection import write_measurement_overlay
from .measurement import (
    aggregate_measurements,
    common_geometry_support,
)
from .measurement_scores import compare_measurement_normalizers, measurement_pair_scores
from .serialization import load_prediction


def _aligned_pair(measurement, replicates):
    """Keep absent parent/replicate rows separate from observed all-zero rows."""
    from scipy.sparse import csr_matrix

    counts, present = [], []
    for label in replicates:
        rows = np.flatnonzero(np.asarray(measurement.observation_replicates) == label)
        parents = measurement.region_indices[rows]
        if len(set(parents)) != len(parents):
            raise ValueError("duplicate replicate/parent measurements")
        operator = csr_matrix(
            (np.ones(len(rows), dtype=np.int64), (parents, rows)),
            shape=(len(measurement.region_ids), measurement.counts.shape[0]),
        )
        counts.append(operator @ measurement.counts)
        mask = np.zeros(len(measurement.region_ids), dtype=bool)
        mask[parents] = True
        present.append(mask)
    return counts, present


def _pair_arguments(measurement, replicates, entry, genes):
    counts, present = _aligned_pair(measurement, replicates)
    return counts, dict(
        region_ids=measurement.region_ids,
        gene_ids=measurement.gene_ids,
        outcome_gene_ids=genes,
        left_present=present[0],
        right_present=present[1] if len(present) == 2 else None,
        patient_id=entry["patient_id"],
        specimen_id=entry["specimen_id"],
        replicate_ids=replicates,
    )


def _panel_exposure(counts, gene_ids, panel):
    lookup = {gene: i for i, gene in enumerate(gene_ids)}
    return np.asarray(counts[:, [lookup[g] for g in panel]].sum(axis=1)).ravel()


def _support_rows(measurement, panels, biopsy, scope, scale, offset):
    total = np.asarray(measurement.counts.sum(axis=1)).ravel()
    nine = _panel_exposure(measurement.counts, measurement.gene_ids, panels["original_nine"])
    rows = []
    for panel, ids in panels.items():
        exposure = _panel_exposure(measurement.counts, measurement.gene_ids, ids)
        for i, name in enumerate(measurement.observation_ids):
            parent = measurement.region_indices[i]
            rows.append(
                dict(
                    biopsy_id=biopsy,
                    scope=scope,
                    scale_pixels=scale,
                    panel=panel,
                    dx_pixels=offset[0],
                    dy_pixels=offset[1],
                    observation_id=name,
                    replicate=str(measurement.observation_replicates[i]),
                    region_id=measurement.region_ids[parent],
                    total_counts=float(total[i]),
                    nine_gene_counts=float(nine[i]),
                    nine_gene_fraction=float(nine[i] / total[i]) if total[i] else None,
                    panel_counts=float(exposure[i]),
                    positive_exposure=bool(exposure[i] > 0),
                    bead_count=int(measurement.bead_counts[i]),
                    declared_support_area_pixels=float(
                        measurement.grid.support_area_pixels[parent]
                    ),
                    declared_tissue_area_pixels=float(measurement.grid.tissue_area_pixels[parent]),
                    occupied_fine_area_pixels=float(measurement.occupied_area_pixels[i]),
                    occupied_fine_tissue_area_pixels=float(measurement.tissue_area_pixels[i]),
                    measured_zero=bool(total[i] == 0),
                    support_note="occupied_fine_bin_area_not_actual_bead_sampled_tissue_area",
                )
            )
    return rows


def _save_membership(arrays, prefix, measurement):
    arrays[prefix + "__region_ids"] = np.asarray(measurement.region_ids)
    arrays[prefix + "__fine_to_parent"] = measurement.grid.fine_to_parent
    arrays[prefix + "__coordinates"] = measurement.coordinates
    arrays[prefix + "__support_area_pixels"] = measurement.grid.support_area_pixels
    arrays[prefix + "__tissue_area_pixels"] = measurement.grid.tissue_area_pixels
    arrays[prefix + "__bead_parent"] = measurement.bead_parent_indices


def _append_scores(
    records, tables, arrays, report, *, biopsy, condition, scope, eligibility, dx=0, dy=0
):
    base = dict(
        biopsy_id=biopsy,
        patient_id=report["patient_id"],
        specimen_id=report["specimen_id"],
        condition=condition,
        scope=scope,
        eligibility=eligibility,
        dx_pixels=dx,
        dy_pixels=dy,
        structurally_paired=report["structurally_paired"],
        pair_left_replicate=report["replicate_ids"][0],
        pair_right_replicate=report["replicate_ids"][1]
        if report["structurally_paired"]
        else "unavailable",
    )
    key = f"{biopsy}__{scope}__{condition}__{eligibility}__{dx}_{dy}"
    for reason, ids in report["eligibility"]["exclusions"].items():
        arrays[key + "__excluded__" + reason] = np.asarray(ids, dtype=str)
    strata = [("all", report["scores"])] + [
        (r["stratum"], r["scores"]) for r in report["exposure_strata"]["strata"]
    ]
    for stratum, score in strata:
        arrays[key + f"__{stratum}__regions"] = np.asarray(score["region_ids"], dtype=str)
        records.append(
            {
                **base,
                "stratum": stratum,
                **{
                    k: score[k]
                    for k in (
                        "n_regions",
                        "median_gene_spearman",
                        "defined_gene_median_diagnostic",
                        "normalized_log_mse",
                    )
                },
                "defined_genes": sum(g["spearman"] is not None for g in score["gene_scores"]),
                "exposure_cutpoints": report["exposure_strata"]["cutpoints"],
            }
        )
        for direction in score["directions"] or [
            {"source_replicate": report["replicate_ids"][0], "evaluation_replicate": "unavailable"}
        ]:
            for gene in score["gene_scores"]:
                tables.append(
                    {
                        **base,
                        "stratum": stratum,
                        "source_replicate": direction["source_replicate"],
                        "evaluation_replicate": direction["evaluation_replicate"],
                        **gene,
                    }
                )


def _summaries(records):
    """Equal patients; structurally single assays stay in the registry, never disappear."""
    summaries = []
    fields = ("scope", "condition", "eligibility", "stratum", "dx_pixels", "dy_pixels")
    keys = sorted({tuple(row[k] for k in fields) for row in records})
    for key in keys:
        rows = [r for r in records if tuple(r[k] for k in fields) == key]
        if len({r["patient_id"] for r in rows}) != len(rows):
            raise ValueError("measurement summary cannot duplicate patients")
        paired = [r for r in rows if r["structurally_paired"]]
        values = [r["median_gene_spearman"] for r in paired]
        diagnostic = [r["defined_gene_median_diagnostic"] for r in paired]
        errors = [r["normalized_log_mse"] for r in paired]
        summaries.append(
            {
                **dict(zip(fields, key)),
                "n_registered_patients": len(rows),
                "n_paired_patients": len(paired),
                "n_defined_patients": sum(v is not None for v in values),
                "equal_patient_spearman": float(np.mean(values))
                if values and all(v is not None for v in values)
                else None,
                "equal_patient_defined_gene_median_diagnostic": float(np.mean(diagnostic))
                if diagnostic and all(v is not None for v in diagnostic)
                else None,
                "equal_patient_mse": float(np.mean(errors))
                if errors and all(v is not None for v in errors)
                else None,
                "n_regions_across_paired_patients": sum(r["n_regions"] for r in paired),
                "patient_ids": [r["patient_id"] for r in rows],
                "undefined_paired_patients": [
                    r["patient_id"] for r in paired if r["median_gene_spearman"] is None
                ],
            }
        )
    return summaries


def _paired_differences(records):
    rows = [r for r in records if r["scope"] == "screen" and r["stratum"] == "all"]
    lookup = {(r["patient_id"], r["condition"], r["eligibility"]): r for r in rows}
    differences = []
    for row in rows:
        scale, panel = row["condition"].split("_", 1)
        references = []
        if scale != "100":
            references.append((f"100_{panel}", "scale_same_declared_fine_footprint"))
        if panel != "original_nine":
            references.append((f"{scale}_original_nine", "normalizer"))
        for condition, comparison in references:
            base = lookup[(row["patient_id"], condition, row["eligibility"])]
            result = {k: row[k] for k in ("biopsy_id", "patient_id", "eligibility")}
            result.update(
                condition=row["condition"],
                reference_condition=condition,
                comparison=comparison,
                n_regions=row["n_regions"],
                reference_n_regions=base["n_regions"],
                identical_evaluation_rows=(
                    comparison == "normalizer" and row["eligibility"] == "common_eligible"
                ),
                interpretation="patient difference; scales change units and eligibility",
            )
            for metric in (
                "median_gene_spearman",
                "defined_gene_median_diagnostic",
                "normalized_log_mse",
            ):
                result["delta_" + metric] = (
                    row[metric] - base[metric]
                    if row[metric] is not None and base[metric] is not None
                    else None
                )
            differences.append(result)
    return differences


def _reference_check(root, biopsy, entry, substrate, base, study, fold, genes, old_capability):
    """Compare the new substrate to the unchanged v0.7 target-loading path and saved coverage."""
    path = study / f"fold-{fold}" / f"test-{biopsy}-he_rna.npz"
    prediction, metadata = load_prediction(path)
    provenance = metadata["cohort_inputs"]
    if prediction.cell_ids != substrate.grid.region_ids or not np.array_equal(
        prediction.coordinates, substrate.grid.coordinates
    ):
        raise ValueError("measurement base grid differs from completed study")
    target, operator, details = load_cohort_targets(
        root, biopsy, prediction, provenance, gene_ids=genes
    )
    columns = [base.gene_ids.index(g) for g in genes]
    lookup = {name: i for i, name in enumerate(base.observation_ids)}
    order = [lookup[name] for name in target.observation_ids]
    actual = base.counts[order][:, columns].toarray()
    if not np.array_equal(actual, target.expression) or not np.array_equal(
        base.bead_counts[order], details["region_bead_counts"]
    ):
        raise ValueError("original raw nine-gene counts or bead counts differ")
    eligible = actual.sum(1) > 0
    old = json.loads((old_capability / biopsy / "consistency.json").read_text())
    if {r["observation_id"] for r in old["exact_eligible_observations"]} != {
        name for name, keep in zip(target.observation_ids, eligible) if keep
    }:
        raise ValueError("original positive-exposure eligibility differs")
    return (
        target,
        operator,
        details,
        {
            "biopsy_id": biopsy,
            "original_raw_counts_equal": True,
            "original_bead_counts_equal": True,
            "original_eligibility_equal": True,
            "n_observations": len(order),
            "n_positive_exposure": int(eligible.sum()),
        },
    )


def _saved_models(study, fold, biopsy, target, operator, details, regions, genes):
    if not regions:
        return [
            {
                "biopsy_id": biopsy,
                "patient_id": target.patient_id,
                "arm": arm,
                "condition": "100_original_nine",
                "status": "unavailable_shared_eligible_pair",
                "n_shared_regions": 0,
                "spearman": None,
                "normalized_log_mse": None,
            }
            for arm in _EXPORTS
        ]
    directory = study / f"fold-{fold}"
    receipt = json.loads((directory / "test_predictions_complete.json").read_text())
    groups = np.asarray(details["observation_replicates"])
    region_set = set(regions)
    eligible = np.array([operator.cell_ids[j] in region_set for j in _region_rows(operator)])
    selected, geometry = _subset(target, operator, eligible)
    rows = []
    for arm in _EXPORTS:
        path = directory / f"test-{biopsy}-{arm}.npz"
        if _sha(path) != receipt["prediction_sha256"][path.name]:
            raise ValueError("completed prediction differs from its completion record")
        prediction, _ = load_prediction(path)
        result = replicate_scores(prediction, selected, geometry, groups[eligible], genes, genes)
        rows.append(
            {
                "biopsy_id": biopsy,
                "patient_id": target.patient_id,
                "arm": arm,
                "condition": "100_original_nine",
                "status": "supported",
                "n_shared_regions": len(regions),
                "spearman": result["mean_replicate_median_gene_spearman"],
                "normalized_log_mse": result["normalized_log_mse"],
            }
        )
    return rows


def _report(output, summary):
    lines = [
        "# HTAPP measurement qualification (v0.8)",
        "",
        "This is a measurement-only internal development screen. The completed v0.6/v0.7 "
        "results remain unchanged. No mapper, reference, correction, protein model "
        "or encoder was fitted.",
        "",
        "Raw counts are summed over nested bins before fixed-panel normalization. Common support "
        "is defined by bead-occupied H&E-selected 100-pixel bins before expression filtering. "
        "Intrabin sampled tissue may differ. Physical dimensions and exact section correspondence "
        "remain unqualified.",
        "",
        f"The broader panel contains {summary['normalizer_sizes']['shared_measured']} "
        "ordered gene IDs, "
        "selected solely by measured-axis intersection across all spatial "
        "and independent RNA sources.",
        "",
        "| Condition | Paired patients | Defined patients (all 9 genes) | "
        "Mean correlation, strict | "
        "Mean defined-gene median (diagnostic) | Mean normalized error | Regions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value):
        return "Undefined" if value is None else f"{value:.5f}"

    for row in summary["conditions"]:
        if (
            row["scope"] == "screen"
            and row["eligibility"] == "all_eligible"
            and row["stratum"] == "all"
        ):
            lines.append(
                f"| {row['condition']} | {row['n_paired_patients']} | "
                f"{row['n_defined_patients']} | "
                f"{fmt(row['equal_patient_spearman'])} | "
                f"{fmt(row['equal_patient_defined_gene_median_diagnostic'])} | "
                f"{fmt(row['equal_patient_mse'])} | {row['n_regions_across_paired_patients']} |"
            )
    lines += [
        "",
        "All 13 patients remain in the tables; three have only one registered replicate. "
        "Strict summaries require every outcome to be defined "
        "within each paired patient. "
        "The defined-gene diagnostic is labeled separately and must not replace "
        "missing strict results. Displacement summaries retain the ten paired patients.",
        "",
        "Exposure strata use the lower paired panel count and fixed within-patient tertiles; "
        "ties stay together. All-eligible and common-eligible normalizer comparisons are retained. "
        "Displacements reuse zero-offset cutpoints, although exposure changes can move rows "
        "between strata. All-stratum comparisons retain the fixed row set. "
        "The displacement screen shifts only the second registered replicate by the five "
        "prespecified offsets on one common footprint. These are pixel stress tests, not "
        "landmark-derived uncertainty or a transform-selection procedure.",
        "",
        "Saved models are scored only for the original 100-pixel nine-gene condition on the "
        "identical shared rows. Broader normalization is unsupported because the exports lack "
        "the required genes; coarser prediction representations require a later protocol. "
        "Saved-model headlines retain the original target-defined gene eligibility; assay "
        "strict headlines require all nine correlations. Defined-gene assay medians are "
        "diagnostic companions, not equivalent strict headlines.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Exact outputs",
        "",
        "- `measurement_conditions.tsv`: all six prespecified definitions and panels.",
        "- `patient_measurement_scores.tsv` and `patient_summaries.tsv`: every patient, gene, "
        "direction, stratum and eligibility view; all displacement results are retained.",
        "- `paired_condition_differences.tsv`: within-patient scale and normalizer contrasts.",
        "- `exposure_support.tsv`: counts, bead occupancy, declared tissue "
        "support and eligibility.",
        "- `region_membership.npz`: original bead coordinates/IDs, "
        "fine/parent mappings and exact scoring rows.",
        "- `linkage_review.tsv` and `overlays/`: source linkage, uncertainty "
        "and morphology review.",
        "- `saved_model_scores.tsv`, `summary.json`, `identity.json`, `panels.json`: "
        "model compatibility, "
        "patient summaries, source bindings, and fixed ordered gene axes.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def run_measurement(root, study, capability, config_path, output):
    root, study, capability, output = map(Path, (root, study, capability, output))
    config = json.loads(Path(config_path).read_text())
    if config.get("schema") != "heir.cohort_measurement.v1" or config.get("scales_pixels") != [
        100,
        200,
        400,
    ]:
        raise ValueError("unsupported measurement configuration")
    normalizers = config.get("normalizers", {})
    exposure = config.get("exposure_strata", {})
    policies = {
        "support": (
            "intersection_of_bead_occupied_selected_100_pixel_he_bins_across_"
            "registered_replicates_before_exposure_filtering"
        ),
        "aggregation": "sum_raw_counts_within_fixed_fine_support_then_normalize; no_log_averaging",
        "normalization": "log1p(10000*outcome_counts/sum_fixed_panel_counts)",
        "eligibility": "both_replicates_present_and_positive_complete_panel_exposure",
    }
    displacement = config.get("displacement", {})
    if (
        config.get("base_region_size_pixels") != 100
        or config.get("minimum_regions") != 3
        or normalizers.get("original_nine") != config.get("outcome_gene_ids")
        or normalizers.get("shared_measured", {}).get("rule")
        != (
            "sorted_intersection_of_retained_spatial_and_independent_rna_gene_ids_"
            "across_all_thirteen_patients"
        )
        or normalizers.get("shared_measured", {}).get("selection_uses_expression_or_agreement")
        is not False
        or exposure.get("quantiles") != [1 / 3, 2 / 3]
        or exposure.get("quantile_method") != "linear"
        or exposure.get("ties") != "equal_values_never_split"
        or exposure.get("statistic") != "minimum_of_paired_panel_counts"
        or config.get("displacement", {}).get("offsets_pixels")
        != [[0, 0], [-25, 0], [25, 0], [0, -25], [0, 25]]
        or any(config.get(key) != value for key, value in policies.items())
        or displacement.get("shifted_replicate") != "second_sorted_registered_label"
        or displacement.get("support")
        != "intersection_of_geometry_support_across_all_offsets_fixed_before_scoring"
        or displacement.get("select_transform") is not False
    ):
        raise ValueError("unsupported measurement panel, geometry, exposure or displacement policy")
    if (
        config.get("new_model_fitting") is not False
        or config.get("new_reference_fitting") is not False
    ):
        raise ValueError("measurement workflow cannot authorize fitting")
    if any(
        output.resolve().is_relative_to(p.resolve())
        for p in (root, study, capability, Path(__file__).parents[3])
    ):
        raise ValueError("measurement outputs must be outside source, data and completed studies")
    for name, digest in config["completed_study_files_sha256"].items():
        if Path(name).name != name or _sha(study / name) != digest:
            raise ValueError("completed study binding differs")
    if _sha(capability / "summary.json") != config["completed_capability_summary_sha256"]:
        raise ValueError("completed capability summary differs")
    if _sha(root / "metadata/selected_13_biopsies.tsv") != config["patient_metadata_sha256"]:
        raise ValueError("cohort metadata binding differs")
    cohort, verification = discover_cohort(root), {}
    genes = tuple(config["outcome_gene_ids"])
    axes = {
        b: load_measurement_gene_axes(root, b, verification_cache=verification)
        for b in sorted(cohort)
    }
    broad = set.intersection(
        *(
            set(a[k])
            for a in axes.values()
            for k in ("spatial_gene_ids", "independent_rna_gene_ids")
        )
    )
    if not set(genes) <= broad or len(broad) <= len(genes):
        raise ValueError("shared measured panel must include the outcomes and be broader")
    panels = {"original_nine": genes, "shared_measured": tuple(sorted(broad))}
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    (output / "config.json").write_bytes(Path(config_path).read_bytes())
    _write(output / "panels.json", {k: list(v) for k, v in panels.items()})
    package = Path(__file__).resolve().parents[1]
    _write(
        output / "identity.json",
        {
            "config_sha256": _sha(config_path),
            "panels_sha256": _sha(output / "panels.json"),
            "source_file_sha256": {
                str(p.relative_to(package.parent)): _sha(p) for p in sorted(package.rglob("*.py"))
            },
            "completed_study": str(study.resolve()),
            "completed_capability": str(capability.resolve()),
            "gene_axes_and_sources": axes,
            "new_model_fitting": False,
        },
    )
    previous = json.loads((study / "summary.json").read_text())
    folds = {
        p: r["outer_fold"]
        for r in previous["selections"]
        for p, role in r["fold_roles"].items()
        if role == "test"
    }
    arrays = {}
    score_rows, records, support, linkage, checks, models, conservation = ([] for _ in range(7))
    conditions = []
    for scale, (panel, ids) in itertools.product(config["scales_pixels"], panels.items()):
        conditions.append(
            dict(
                condition=f"{scale}_{panel}",
                scale_registered_pixels=scale,
                panel=panel,
                n_normalizer_genes=len(ids),
                panel_file="panels.json",
                panel_sha256=_sha(output / "panels.json"),
                n_outcome_genes=len(genes),
                physical_size="unknown",
                support_rule=config["support"],
                eligibility=config["eligibility"],
                saved_model_status="supported"
                if scale == 100 and panel == "original_nine"
                else "unsupported_missing_normalizer_genes"
                if panel == "shared_measured"
                else "unsupported_coarse_prediction_representation",
            )
        )
    for biopsy, entry in sorted(cohort.items()):
        print(f"measurement: {biopsy}, raw support and six fixed conditions", flush=True)
        substrate = load_measurement_substrate(
            root,
            biopsy,
            base_region_size=100,
            min_tissue_fraction=config["min_tissue_fraction"],
            verification_cache=verification,
        )
        reps = tuple(sorted(set(substrate.replicates)))
        if len(reps) not in (1, 2):
            raise ValueError("this fixed cohort screen expects one or two registered replicates")
        common = common_geometry_support(substrate, reps)
        arrays[biopsy + "__fine_region_ids"] = np.asarray(substrate.grid.region_ids)
        arrays[biopsy + "__fine_bin_keys"] = substrate.grid.bin_keys
        arrays[biopsy + "__bead_ids"] = np.asarray(substrate.bead_ids)
        arrays[biopsy + "__bead_xy"] = substrate.coordinates
        arrays[biopsy + "__bead_replicates"] = np.asarray(substrate.replicates)
        arrays[biopsy + "__bead_fine_region"] = substrate.base_region_indices
        arrays[biopsy + "__common_fine_indices"] = common
        base = aggregate_measurements(substrate, scale_pixels=100)
        target, operator, details, check = _reference_check(
            root,
            biopsy,
            entry,
            substrate,
            base,
            study,
            folds[entry["patient_id"]],
            genes,
            capability,
        )
        checks.append(check)
        overlay = write_measurement_overlay(
            output / "overlays" / f"{biopsy}.png",
            root / entry["role_sources"]["he"]["working_path"],
            substrate.grid,
            substrate.coordinates,
            substrate.replicates,
            common,
        )
        review = config.get("morphology_review", {}).get(biopsy, {})
        linkage.append(
            dict(
                biopsy_id=biopsy,
                patient_id=entry["patient_id"],
                registered_replicates=";".join(reps),
                linkage="unknown",
                exact_section_ids="unknown",
                core_relationship="unknown",
                physical_pixel_size="unknown",
                registration_qualified=False,
                correspondence=review.get("category", "unresolved"),
                morphology_note=review.get("note", "overlay available; review pending"),
                evidence=(
                    "selected_Table_S2_assay_rows.tsv; verified_file_manifest "
                    "section_identity_status; pinned 00_matchSpatial.ipynb"
                ),
                limitation=(
                    "biopsy membership and broad outline overlap "
                    "do not establish same-section tissue"
                ),
            )
        )
        _write(output / "overlays" / f"{biopsy}.json", overlay)
        for scale in config["scales_pixels"]:
            available = (
                base if scale == 100 else aggregate_measurements(substrate, scale_pixels=scale)
            )
            support.extend(_support_rows(available, panels, biopsy, "all_available", scale, (0, 0)))
            measurement = aggregate_measurements(
                substrate, scale_pixels=scale, support_fine_indices=common
            )
            conservation.append(
                {
                    "biopsy": biopsy,
                    "scope": "screen",
                    "scale": scale,
                    "checks": measurement.conservation,
                }
            )
            prefix = f"{biopsy}__screen__{scale}"
            _save_membership(arrays, prefix, measurement)
            support.extend(
                _support_rows(measurement, panels, biopsy, "common_geometry", scale, (0, 0))
            )
            counts, args = _pair_arguments(measurement, reps, entry, genes)
            comparison = compare_measurement_normalizers(
                counts[0],
                counts[1] if len(reps) == 2 else None,
                normalization_panels=panels,
                **args,
            )
            for panel in panels:
                for eligibility, report in comparison["normalizers"][panel].items():
                    _append_scores(
                        records,
                        score_rows,
                        arrays,
                        report,
                        biopsy=biopsy,
                        condition=f"{scale}_{panel}",
                        scope="screen",
                        eligibility=eligibility,
                    )
            if scale == 100:
                regions = comparison["normalizers"]["original_nine"]["all_eligible"]["scores"][
                    "region_ids"
                ]
                models.extend(
                    _saved_models(
                        study,
                        folds[entry["patient_id"]],
                        biopsy,
                        target,
                        operator,
                        details,
                        regions,
                        genes,
                    )
                )
        if len(reps) < 2:
            continue
        offsets = config["displacement"]["offsets_pixels"]
        footprints = [
            common_geometry_support(substrate, reps, displacement_by_replicate={reps[1]: delta})
            for delta in offsets
        ]
        stress_support = np.asarray(
            sorted(set.intersection(*(set(f) for f in footprints))), dtype=int
        )
        arrays[biopsy + "__stress_common_fine_indices"] = stress_support
        for scale in config["scales_pixels"]:
            shifted = [
                aggregate_measurements(
                    substrate,
                    scale_pixels=scale,
                    support_fine_indices=stress_support,
                    displacement_by_replicate={reps[1]: delta},
                )
                for delta in offsets
            ]
            paired = [_pair_arguments(m, reps, entry, genes) for m in shifted]
            for panel, ids in panels.items():
                baseline_counts, baseline_args = paired[0]
                baseline = measurement_pair_scores(
                    baseline_counts[0],
                    baseline_counts[1],
                    **baseline_args,
                    normalization_gene_ids=ids,
                )
                fixed_cutpoints = baseline["exposure_strata"]["cutpoints"] or [0.0, 0.0]
                masks = [
                    args["left_present"]
                    & args["right_present"]
                    & (_panel_exposure(counts[0], args["gene_ids"], ids) > 0)
                    & (_panel_exposure(counts[1], args["gene_ids"], ids) > 0)
                    for counts, args in paired
                ]
                stable = np.logical_and.reduce(masks)
                for delta, (counts, args) in zip(offsets, paired):
                    for eligibility, mask in (
                        ("all_eligible", None),
                        ("common_across_offsets", stable),
                    ):
                        report = measurement_pair_scores(
                            counts[0],
                            counts[1],
                            **args,
                            normalization_gene_ids=ids,
                            row_mask=mask,
                            exposure_cutpoints=fixed_cutpoints,
                        )
                        _append_scores(
                            records,
                            score_rows,
                            arrays,
                            report,
                            biopsy=biopsy,
                            condition=f"{scale}_{panel}",
                            scope="displacement",
                            eligibility=eligibility,
                            dx=delta[0],
                            dy=delta[1],
                        )
            for delta, measurement in zip(offsets, shifted):
                _save_membership(
                    arrays, f"{biopsy}__shift__{scale}__{delta[0]}_{delta[1]}", measurement
                )
                support.extend(
                    _support_rows(measurement, panels, biopsy, "displacement", scale, delta)
                )
                conservation.append(
                    {
                        "biopsy": biopsy,
                        "scope": "displacement",
                        "scale": scale,
                        "offset": delta,
                        "checks": measurement.conservation,
                    }
                )
    summary = dict(
        schema="heir.cohort_measurement_result.v1",
        normalizer_sizes={k: len(v) for k, v in panels.items()},
        conditions=_summaries(records),
        original_condition_reconciliation=checks,
        aggregation_conservation=conservation,
        new_model_fitting=False,
        physical_calibration="unknown",
        exact_section_linkage="unknown",
        independent_confirmation=False,
        scientific_qualification=False,
        decision="No measurement recipe is automatically promoted. Review all six conditions, "
        "patient-level consistency, exposure, loss of resolution, and displacement sensitivity "
        "together. Unknown section linkage and unqualified pixel calibration keep this an "
        "exploratory internal screen; a stronger spatial claim requires documented correspondence.",
    )
    _table(output / "measurement_conditions.tsv", conditions)
    _table(output / "patient_measurement_scores.tsv", score_rows)
    _table(output / "patient_summaries.tsv", records)
    _table(output / "paired_condition_differences.tsv", _paired_differences(records))
    _table(output / "exposure_support.tsv", support)
    _table(output / "linkage_review.tsv", linkage)
    _table(output / "saved_model_scores.tsv", models)
    with (output / "region_membership.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    _write(output / "summary.json", summary)
    _report(output, summary)
    print("measurement: completed all thirteen patients", flush=True)
    return summary
