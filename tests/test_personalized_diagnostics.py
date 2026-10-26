"""Fixed-fit diagnostic regressions: evaluation scale, replicate axes and interventions."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_cohort_cli import regional_demo as regional_demo
from test_cohort_cli import regional_workflow as regional_workflow
from test_personalized_cohort import published_cohort as published_cohort
from test_personalized_model import model, specimen

from heir.personalized.cli import main
from heir.personalized.diagnostics import (
    correction_diagnostics,
    permute_protein_values,
    replicate_agreement,
    write_pilot_diagnostics,
)
from heir.personalized.observation import ObservationOperator
from heir.personalized.schema import ProteinEvidence, ProteinObservationOperator, SpatialTargets


def _example():
    prediction = model()(specimen())
    transfer = torch.tensor([[1.0, 4.0, 2.0], [2.0, 3.0, 4.0], [4.0, 1.0, 3.0]])
    prediction = replace(prediction, transfer=transfer, expression=transfer)
    ids = tuple(f"rep:{rep}:c{i}" for rep in ("1", "2") for i in range(3))
    operator = ObservationOperator(
        prediction.cell_ids, ids, np.arange(6), np.tile(np.arange(3), 2), np.ones(6)
    )
    targets = SpatialTargets(
        observation_ids=ids,
        gene_ids=prediction.gene_ids,
        expression=np.concatenate([transfer.numpy(), 13 * transfer.numpy()]),
        observed=np.ones((6, 3), dtype=bool),
        patient_id=prediction.patient_id,
        specimen_id=prediction.specimen_id,
    )
    return prediction, targets, operator, ("1",) * 3 + ("2",) * 3


def test_common_multiplicative_correction_disappears_after_fixed_normalization():
    prediction, targets, operator, groups = _example()
    prediction = replace(
        prediction,
        expression=prediction.transfer * torch.exp(torch.tensor(0.7)),
        log_correction=torch.full((3, 3), 0.7),
    )
    report = correction_diagnostics(
        prediction, targets, operator, groups, ("A", "C"), ("A", "B", "C"), residual_bound=1
    )
    assert all(row["mean_absolute"] < 1e-6 for row in report["normalized_effects"])
    assert all(
        row["mean_absolute_raw_log_correction"] == pytest.approx(0.7)
        for row in report["raw_correction"]
    )
    assert {row["scope"] for row in report["raw_correction"]} == {
        "all_he_regions",
        "unique_scored_regions",
        "scored_replicate:1",
        "scored_replicate:2",
    }
    assert all(row["n_regions"] == 3 for row in report["raw_correction"])


def test_saturation_scopes_do_not_count_target_replicates_as_new_regions():
    prediction, targets, operator, groups = _example()
    delta = torch.tensor([[1.0, -1.0, 0.0], [0.96, 0.0, -0.96], [0.0, 1.0, -1.0]])
    prediction = replace(
        prediction, log_correction=delta, expression=prediction.transfer * delta.exp()
    )
    report = correction_diagnostics(
        prediction, targets, operator, groups, ("A",), ("A", "B", "C"), residual_bound=1
    )
    first = report["raw_correction"][0]
    assert first["positive_bound_fraction"] == pytest.approx(1 / 3)
    assert first["negative_bound_fraction"] == 0
    assert first["near_either_bound_fraction"] == pytest.approx(2 / 3)
    assert report["normalized_effects"][0]["mean_absolute"] > 0
    zero = correction_diagnostics(
        prediction, targets, operator, groups, ("A",), ("A", "B", "C"), residual_bound=0
    )
    assert zero["raw_correction"][0]["positive_bound_fraction"] is None


def test_replicate_agreement_uses_named_region_join_and_fixed_panel():
    prediction, targets, operator, groups = _example()
    # Storage order differs: region identities, rather than array order, establish pairs.
    order = np.array([0, 1, 2, 5, 3, 4])
    targets = replace(
        targets,
        observation_ids=tuple(targets.observation_ids[i] for i in order),
        expression=targets.expression[order],
        observed=targets.observed[order],
    )
    operator = replace(
        operator, observation_ids=targets.observation_ids, columns=operator.columns[order]
    )
    report = replicate_agreement(prediction, targets, operator, groups, ("A", "C"), ("A", "B", "C"))
    pair = report["pairs"][0]
    assert pair["shared_eligible_region_ids"] == list(prediction.cell_ids)
    assert all(row["spearman"] == pytest.approx(1) for row in pair["gene_scores"])
    assert all(
        row["normalized_log_mse"] == pytest.approx(0, abs=1e-10) for row in pair["gene_scores"]
    )


def test_protein_permutation_preserves_replicate_mask_quality_and_marker_axes():
    ids = tuple(f"c{i}" for i in range(6))
    observations = tuple(f"codex:{'1' if i < 3 else '2'}:{name}" for i, name in enumerate(ids))
    indices = np.arange(6)
    values = np.column_stack([indices + 1.0, indices + 11.0])
    observed = np.ones((6, 2), dtype=bool)
    observed[[2, 5], 1] = False
    quality = np.arange(12).reshape(6, 2) / 12
    protein = ProteinEvidence(
        marker_ids=("M1", "M2"),
        values=values,
        observed=observed,
        state_values=np.ones((2, 2)),
        correspondence="region",
        operator=ProteinObservationOperator(ids, observations, indices, indices, np.ones(6)),
        quality=quality,
    )
    permuted, metadata = permute_protein_values(protein, seed=3)
    assert metadata["n_movable_rows"] == 4
    assert permuted.marker_ids == protein.marker_ids
    np.testing.assert_array_equal(permuted.observed, observed)
    np.testing.assert_array_equal(permuted.quality, quality)
    np.testing.assert_array_equal(protein.values, values)
    for destination, donor in enumerate(metadata["donor_observation_ids"]):
        source = observations.index(donor)
        assert observations[destination].split(":")[1] == donor.split(":")[1]
        np.testing.assert_array_equal(observed[destination], observed[source])
        np.testing.assert_array_equal(permuted.values[destination], values[source])
    assert metadata == permute_protein_values(protein, seed=3)[1]


def test_saved_demo_diagnostics_reproduce_five_main_scores_without_fitting(
    regional_demo, tmp_path_factory, monkeypatch
):
    from heir.personalized import train

    root, run, _ = regional_demo
    monkeypatch.setattr(train, "fit", lambda *a, **k: pytest.fail("diagnostics must not fit"))
    output = tmp_path_factory.mktemp("diagnostic-output") / "report"
    main(["cohort-diagnose", "--root", str(root), "--run", str(run), "--output", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert set(report["main_result"]["scores"]) == {
        "he",
        "he_rna",
        "he_rna.transfer",
        "he_rna_protein",
        "he_rna_protein.transfer",
    }
    for name, result in report["main_result"]["scores"].items():
        original = json.loads((run / f"{name}.scores.json").read_text())
        for metric in ("mean_replicate_median_gene_spearman", "normalized_log_mse"):
            assert result[metric] == pytest.approx(original[metric])
    assert report["new_fitting_performed"] is False
    assert report["main_result"]["n_region_replicates"] == 8
    assert all(row["n_scored_regions"] == 4 for row in report["coverage"])
    assert all(row["codex_covered"] == 3 for row in report["coverage"])
    assert all(row["all_markers_observed"] == 2 for row in report["coverage"])
    assert report["diagnostic_subsets"]["common_complete_codex"]["n_region_replicates"] == 4
    assert report["diagnostic_subsets"]["partial_boundary"]["status"] == "no_regions"
    with (output / "scored_regions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 8
    assert all(row["target_bead_count"] == "1" for row in rows)
    assert json.loads((output / "report.json").read_text())["coverage"] == report["coverage"]
    conditions = report["protein_interventions"]["conditions"]
    assert (
        conditions["original"]["prediction_changes"]["all_he_regions"]["expression"][
            "max_absolute_entry_change"
        ]
        == 0
    )
    for condition in conditions.values():
        assert (
            condition["prediction_changes"]["all_he_regions"]["log_correction"][
                "max_absolute_entry_change"
            ]
            == 0
        )
    with pytest.raises(FileExistsError):
        write_pilot_diagnostics(root, run, output)
    with pytest.raises(ValueError, match="immutable"):
        write_pilot_diagnostics(root, run, root / "report")
