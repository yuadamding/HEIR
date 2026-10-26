"""Measurement orchestration keeps sparse axes, common support, and undefined patients."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, isspmatrix_csr

from heir.personalized import measurement_workflow as workflow
from heir.personalized.measurement import (
    MeasurementSubstrate,
    aggregate_measurements,
    common_geometry_support,
)
from heir.personalized.measurement_scores import compare_measurement_normalizers
from heir.personalized.observation import ObservationOperator
from heir.personalized.regions import build_region_grid, point_to_grid
from heir.personalized.schema import SpatialTargets


def _substrate(patient="P1", *, paired=True, constant=False):
    image = np.full((100, 900, 3), [170, 75, 120], dtype=np.uint8)
    grid = build_region_grid(image, compute_features=False)
    x = np.arange(9) * 100
    xy = np.column_stack((x + 50, np.full(9, 50)))
    if paired:
        xy = np.concatenate((xy, np.column_stack((x + 5, np.full(9, 50)))))
    genes = ("A", "B", "C")
    counts = np.column_stack((np.arange(1, 10), np.arange(9, 0, -1), np.full(9, 5)))
    if constant:
        counts[:] = [1, 1, 5]
    if paired:
        counts = np.concatenate((counts, counts * 2))
    labels = ("1",) * 9 + (("2",) * 9 if paired else ())
    return MeasurementSubstrate(
        biopsy_id=patient,
        patient_id=patient,
        specimen_id=f"S-{patient}",
        grid=grid,
        gene_ids=genes,
        independent_rna_gene_ids=genes + ("independent-only",),
        bead_ids=tuple(f"{patient}-bead{i}" for i in range(len(xy))),
        coordinates=xy,
        replicates=labels,
        counts=csr_matrix(counts),
        base_region_indices=point_to_grid(grid, xy),
        provenance={},
    )


def test_sparse_alignment_preserves_unobserved_rows_and_observed_zero_rows():
    substrate = _substrate()
    # Replicate1 observes a real zero; replicate2 has no bead in the last region.
    counts = substrate.counts.toarray()
    counts[0] = 0
    xy = substrate.coordinates.copy()
    xy[-1] = [1000, 50]
    substrate = replace(
        substrate,
        coordinates=xy,
        base_region_indices=point_to_grid(substrate.grid, xy),
        counts=csr_matrix(counts),
    )
    measurement = aggregate_measurements(substrate)
    aligned, present = workflow._aligned_pair(measurement, ("1", "2"))
    assert all(isspmatrix_csr(values) for values in aligned)
    assert aligned[0].shape == aligned[1].shape == (9, 3)
    assert aligned[0][0].nnz == 0 and present[0][0]
    assert aligned[1][-1].nnz == 0 and not present[1][-1]
    with pytest.raises(ValueError, match="duplicate replicate/parent"):
        workflow._aligned_pair(
            replace(measurement, region_indices=np.zeros(len(measurement.observation_ids), int)),
            ("1", "2"),
        )


def test_normalizer_common_rows_do_not_change_geometry_or_conflate_zero_with_missing():
    substrate = _substrate()
    counts = substrate.counts.toarray()
    counts[[0, 9], :2] = 0  # Broad-panel exposure remains positive in the observed region.
    substrate = replace(substrate, counts=csr_matrix(counts))
    common = common_geometry_support(substrate, ("1", "2"))
    measurement = aggregate_measurements(substrate, support_fine_indices=common)
    values, args = workflow._pair_arguments(
        measurement, ("1", "2"), {"patient_id": "P1", "specimen_id": "S-P1"}, ("A", "B")
    )
    result = compare_measurement_normalizers(
        *values, normalization_panels={"narrow": ("A", "B"), "broad": ("A", "B", "C")}, **args
    )
    narrow, broad = (result["normalizers"][name] for name in ("narrow", "broad"))
    assert narrow["all_eligible"]["scores"]["n_regions"] == 8
    assert broad["all_eligible"]["scores"]["n_regions"] == 9
    assert (
        narrow["common_eligible"]["scores"]["region_ids"]
        == broad["common_eligible"]["scores"]["region_ids"]
    )
    assert narrow["all_eligible"]["eligibility"]["exclusions"]["absent_left"] == []
    assert common.tolist() == list(range(9))


def test_cohort_summary_retains_undefined_paired_and_structurally_single_patients():
    rows = []
    for patient, paired, score, regions in (
        ("P1", True, 0.8, 900),
        ("P2", True, None, 3),
        ("P3", False, None, 0),
    ):
        rows.append(
            dict(
                patient_id=patient,
                scope="screen",
                condition="100_original_nine",
                eligibility="all_eligible",
                stratum="all",
                dx_pixels=0,
                dy_pixels=0,
                structurally_paired=paired,
                median_gene_spearman=score,
                defined_gene_median_diagnostic=score,
                normalized_log_mse=2 if paired else None,
                n_regions=regions,
            )
        )
    summary = workflow._summaries(rows)[0]
    assert summary["n_registered_patients"] == 3 and summary["n_paired_patients"] == 2
    assert summary["equal_patient_spearman"] is None
    assert summary["undefined_paired_patients"] == ["P2"]
    assert summary["patient_ids"] == ["P1", "P2", "P3"]
    rows[1]["median_gene_spearman"] = -0.8
    assert workflow._summaries(rows)[0]["equal_patient_spearman"] == 0  # Not region weighted.


def test_paired_condition_differences_preserve_undefined_and_name_common_row_comparisons():
    records = []
    for patient in ("defined", "undefined", "single"):
        for scale in (100, 200, 400):
            for panel in ("original_nine", "shared_measured"):
                for eligibility in ("all_eligible", "common_eligible"):
                    value = 0.1 + scale / 1000 + (0.2 if panel == "shared_measured" else 0)
                    records.append(
                        {
                            "biopsy_id": patient,
                            "patient_id": patient,
                            "scope": "screen",
                            "stratum": "all",
                            "condition": f"{scale}_{panel}",
                            "eligibility": eligibility,
                            "n_regions": 0 if patient == "single" else 1200 // scale,
                            "structurally_paired": patient != "single",
                            "median_gene_spearman": value if patient == "defined" else None,
                            "defined_gene_median_diagnostic": value
                            if patient != "single"
                            else None,
                            "normalized_log_mse": 5 - value if patient != "single" else None,
                        }
                    )
    result = workflow._paired_differences(records)
    assert {r["patient_id"] for r in result} == {"defined", "undefined", "single"}
    for row in result:
        assert row["identical_evaluation_rows"] == (
            row["comparison"] == "normalizer" and row["eligibility"] == "common_eligible"
        )
        if row["patient_id"] != "defined":
            assert row["delta_median_gene_spearman"] is None
        if row["patient_id"] == "single":
            assert row["delta_normalized_log_mse"] is None
            assert row["delta_defined_gene_median_diagnostic"] is None
        if row["patient_id"] == "defined" and row["comparison"] == "normalizer":
            assert row["delta_median_gene_spearman"] == pytest.approx(0.2)
            assert row["delta_normalized_log_mse"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "keys,value",
    [
        (("base_region_size_pixels",), 50),
        (("minimum_regions",), 2),
        (("normalizers", "original_nine"), ["wrong", "panel"]),
        (("normalizers", "shared_measured", "rule"), "select_highest_agreement_genes"),
        (("normalizers", "shared_measured", "selection_uses_expression_or_agreement"), True),
        (("exposure_strata", "quantiles"), [0.25, 0.75]),
        (("exposure_strata", "quantile_method"), "nearest"),
        (("exposure_strata", "ties"), "split_equal_values"),
        (("exposure_strata", "statistic"), "mean_of_paired_panel_counts"),
        (("support",), "all_available_tissue"),
        (("aggregation",), "average_log_values"),
        (("normalization",), "observed_outcomes_only"),
        (("eligibility",), "either_replicate_positive"),
        (("displacement", "offsets_pixels"), [[25, 0], [-25, 0], [0, 0], [0, -25], [0, 25]]),
        (("displacement", "offsets_pixels"), [[0, 0], [-25, 0], [25, 0], [0, -25]]),
        (("displacement", "offsets_pixels"), [[0, 0], [-50, 0], [50, 0], [0, -25], [0, 25]]),
        (("displacement", "shifted_replicate"), "first_sorted_registered_label"),
        (("displacement", "support"), "union_across_offsets"),
        (("displacement", "select_transform"), True),
    ],
)
def test_fixed_measurement_configuration_rejects_mismatches_before_data_access(
    tmp_path, keys, value
):
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/personalized/measurement.json").read_text()
    )
    section = config
    for key in keys[:-1]:
        section = section[key]
    section[keys[-1]] = value
    path = tmp_path / "mismatched.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        workflow.run_measurement(
            tmp_path / "absent-data",
            tmp_path / "absent-study",
            tmp_path / "absent-capability",
            path,
            output,
        )
    assert not output.exists()


def test_empty_common_footprint_stays_an_explicit_undefined_paired_patient():
    substrate = _substrate()
    xy = substrate.coordinates.copy()
    xy[9:, 0] += 1000
    substrate = replace(
        substrate, coordinates=xy, base_region_indices=point_to_grid(substrate.grid, xy)
    )
    common = common_geometry_support(substrate, ("1", "2"))
    assert len(common) == 0
    measurement = aggregate_measurements(substrate, scale_pixels=400, support_fine_indices=common)
    values, args = workflow._pair_arguments(
        measurement, ("1", "2"), {"patient_id": "P1", "specimen_id": "S-P1"}, ("A", "B")
    )
    comparison = compare_measurement_normalizers(
        *values, normalization_panels={"original_nine": ("A", "B")}, **args
    )
    report = comparison["normalizers"]["original_nine"]["all_eligible"]
    records, rows, arrays = [], [], {}
    workflow._append_scores(
        records,
        rows,
        arrays,
        report,
        biopsy="P1",
        condition="400_original_nine",
        scope="screen",
        eligibility="all_eligible",
    )
    summary = next(r for r in workflow._summaries(records) if r["stratum"] == "all")
    assert summary["n_registered_patients"] == summary["n_paired_patients"] == 1
    assert summary["n_regions_across_paired_patients"] == 0
    assert summary["undefined_paired_patients"] == ["P1"]
    assert summary["equal_patient_spearman"] is summary["equal_patient_mse"] is None
    assert {row["undefined_reason"] for row in rows} == {"insufficient_regions"}


def _original_targets(substrate, genes):
    base = aggregate_measurements(substrate)
    order = [substrate.gene_ids.index(gene) for gene in genes]
    target = SpatialTargets(
        base.observation_ids,
        tuple(genes),
        base.counts[:, order].toarray(),
        np.ones((len(base.observation_ids), len(genes)), dtype=bool),
        substrate.patient_id,
        substrate.specimen_id,
    )
    operator = ObservationOperator(
        substrate.grid.region_ids,
        base.observation_ids,
        np.arange(len(base.observation_ids)),
        base.region_indices.copy(),
        np.ones(len(base.observation_ids)),
    )
    return target, operator, {"observation_replicates": base.observation_replicates}


def test_saved_models_score_the_same_named_rows_even_if_sparse_triplets_are_reordered(
    tmp_path, monkeypatch
):
    substrate = _substrate()
    target, operator, details = _original_targets(substrate, ("A", "B"))
    order = np.arange(len(operator.rows))[::-1]
    operator = replace(operator, rows=operator.rows[order], columns=operator.columns[order])
    directory = tmp_path / "fold-0"
    directory.mkdir()
    (directory / "test_predictions_complete.json").write_text(
        json.dumps({"prediction_sha256": {"test-P1-he.npz": "digest"}})
    )
    monkeypatch.setattr(workflow, "_EXPORTS", ("he",))
    monkeypatch.setattr(workflow, "_sha", lambda path: "digest")
    monkeypatch.setattr(workflow, "load_prediction", lambda path: (SimpleNamespace(), {}))
    selected = []

    def score(prediction, actual_target, actual_operator, groups, genes, normal):
        selected.extend(actual_target.observation_ids)
        assert set(groups) == {"1", "2"}
        return {"mean_replicate_median_gene_spearman": 0.5, "normalized_log_mse": 1.0}

    monkeypatch.setattr(workflow, "replicate_scores", score)
    names = substrate.grid.region_ids[:3]
    workflow._saved_models(tmp_path, 0, "P1", target, operator, details, names, ("A", "B"))
    expected = [name for name in target.observation_ids if name.split(":", 2)[2] in names]
    assert selected == expected


def test_tiny_cohort_runs_all_conditions_and_fixed_displacement_support(tmp_path, monkeypatch):
    root, study, capability, output = (
        tmp_path / name for name in ("data", "study", "capability", "output")
    )
    for path in (root / "metadata", study, capability):
        path.mkdir(parents=True)
    (root / "metadata/selected_13_biopsies.tsv").write_text("synthetic registry\n")
    (study / "summary.json").write_text(
        json.dumps(
            {
                "selections": [
                    {"outer_fold": 0, "fold_roles": {p: "test" for p in ("P1", "P2", "P3")}}
                ]
            }
        )
    )
    (capability / "summary.json").write_text("{}\n")
    specimens = {
        "P1": _substrate(),
        "P2": _substrate("P2", constant=True),
        "P3": _substrate("P3", paired=False),
    }
    cohort = {
        p: {
            "patient_id": p,
            "specimen_id": s.specimen_id,
            "role_sources": {"he": {"working_path": "unused.png"}},
        }
        for p, s in specimens.items()
    }
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/personalized/measurement.json").read_text()
    )
    config.update(
        outcome_gene_ids=["A", "B"],
        completed_study_files_sha256={"summary.json": workflow._sha(study / "summary.json")},
        completed_capability_summary_sha256=workflow._sha(capability / "summary.json"),
        patient_metadata_sha256=workflow._sha(root / "metadata/selected_13_biopsies.tsv"),
    )
    config["normalizers"]["original_nine"] = config["outcome_gene_ids"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(workflow, "discover_cohort", lambda root: cohort)
    monkeypatch.setattr(
        workflow,
        "load_measurement_gene_axes",
        lambda root, biopsy, **kwargs: {
            "spatial_gene_ids": specimens[biopsy].gene_ids,
            "independent_rna_gene_ids": specimens[biopsy].independent_rna_gene_ids,
        },
    )
    monkeypatch.setattr(
        workflow, "load_measurement_substrate", lambda root, biopsy, **kwargs: specimens[biopsy]
    )
    monkeypatch.setattr(
        workflow, "write_measurement_overlay", lambda *args: {"registration_qualified": False}
    )

    def reference_check(root, biopsy, entry, substrate, base, study, fold, genes, old_capability):
        return (
            *_original_targets(substrate, genes),
            {"biopsy_id": biopsy, "original_raw_counts_equal": True},
        )

    monkeypatch.setattr(workflow, "_reference_check", reference_check)
    model_regions = {}

    def saved_models(study, fold, biopsy, target, operator, details, regions, genes):
        model_regions[biopsy] = list(regions)
        return [{"biopsy_id": biopsy, "status": "supported", "region_ids": list(regions)}]

    monkeypatch.setattr(workflow, "_saved_models", saved_models)
    summary = workflow.run_measurement(root, study, capability, config_path, output)
    screen = [
        r
        for r in summary["conditions"]
        if r["scope"] == "screen" and r["eligibility"] == "all_eligible" and r["stratum"] == "all"
    ]
    assert len(screen) == 6
    assert all(r["n_registered_patients"] == 3 and r["n_paired_patients"] == 2 for r in screen)
    assert all(
        r["equal_patient_spearman"] is None and "P2" in r["undefined_paired_patients"]
        for r in screen
    )
    assert summary["new_model_fitting"] is False
    with np.load(output / "region_membership.npz", allow_pickle=False) as data:
        for patient in ("P1", "P2"):
            fine_support = data[patient + "__stress_common_fine_indices"]
            assert fine_support.tolist() == list(range(8))
            for scale in (100, 200, 400):
                mappings = [
                    data[f"{patient}__shift__{scale}__{dx}_{dy}__fine_to_parent"]
                    for dx, dy in config["displacement"]["offsets_pixels"]
                ]
                for mapping in mappings[1:]:
                    np.testing.assert_array_equal(mapping, mappings[0])
            score_key = f"{patient}__screen__100_original_nine__all_eligible__0_0__all__regions"
            assert data[score_key].tolist() == model_regions[patient]
    with (output / "patient_summaries.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {r["patient_id"] for r in rows} == {"P1", "P2", "P3"}
    assert all(r["median_gene_spearman"] == "" for r in rows if r["patient_id"] == "P2")
