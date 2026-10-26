"""Cohort consistency and actual fixed-model protein pathway diagnostics."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_personalized_model import model, specimen

from heir.personalized.cohort_cli import _subset
from heir.personalized.consistency import measurement_consistency, protein_pathway_diagnostics
from heir.personalized.observation import ObservationOperator
from heir.personalized.regions import build_region_grid
from heir.personalized.schema import ProteinEvidence, ProteinObservationOperator, SpatialTargets
from heir.personalized.train import FittedModel


@pytest.fixture
def prepared():
    grid = build_region_grid(
        np.full((10, 40, 3), [170, 75, 120], dtype=np.uint8), bin_size_pixels=10
    )
    inputs = replace(
        specimen(),
        cell_ids=grid.region_ids,
        coordinates_um=grid.coordinates,
        image_features=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.8, 0.2]]),
        graph=None,
        prediction_unit="tissue_region",
        coordinate_units="registered_pixel",
    )
    protein = ProteinEvidence(
        marker_ids=("M1", "M2"),
        values=np.array([[0.9, 0.1], [0.8, 0.0], [0.1, 0.9]]),
        observed=np.array([[True, True], [True, False], [True, True]]),
        quality=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
        state_values=np.array([[0.9, 0.1], [0.1, 0.9]]),
        correspondence="region",
        operator=ProteinObservationOperator(
            inputs.cell_ids,
            tuple(f"codex:1:{name}" for name in inputs.cell_ids[:3]),
            np.arange(3),
            np.arange(3),
            np.ones(3),
        ),
    )
    inputs = replace(inputs, proteins=protein)
    truth = np.array(
        [[8.0, 1.0, 2.0], [3.0, 4.0, 1.0], [1.0, 9.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float32
    )
    prediction = model()(inputs)
    predicted = truth.copy()
    predicted[3] = [900, 1, 1]  # Poor unshared region must not enter the common-row comparison.
    prediction = replace(prediction, expression=torch.from_numpy(predicted))
    columns = np.array([0, 1, 2, 3, 0, 1, 2])
    groups = ("1",) * 4 + ("2",) * 3
    ids = tuple(
        f"slide_seq:{rep}:{inputs.cell_ids[column]}" for rep, column in zip(groups, columns)
    )
    targets = SpatialTargets(
        observation_ids=ids,
        gene_ids=prediction.gene_ids,
        expression=np.concatenate([truth, 13 * truth[:3]]),
        observed=np.ones((7, 3), dtype=bool),
        patient_id=inputs.patient_id,
        specimen_id=inputs.specimen_id,
    )
    operator = ObservationOperator(inputs.cell_ids, ids, np.arange(7), columns, np.ones(7))
    metadata = {
        "all_observation_ids": list(ids),
        "region_bead_counts": [1, 2, 3, 4, 5, 6, 7],
        "observation_replicates": list(groups),
        "source_replicates": {"1": 10, "2": 18},
    }
    provenance = {
        "schema": "heir.htapp_biopsy_regions.v1",
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "grid": {
            "settings": grid.settings,
            "region_ids": list(grid.region_ids),
            "bin_keys": grid.bin_keys.tolist(),
            "coordinates": grid.coordinates.tolist(),
            "tissue_fraction": grid.tissue_fraction.tolist(),
            "component_ids": grid.component_ids,
        },
    }
    return inputs, prediction, targets, operator, groups, metadata, provenance


def _report(prepared, **kwargs):
    inputs, prediction, targets, operator, groups, metadata, provenance = prepared
    return measurement_consistency(
        inputs,
        kwargs.pop("predictions", {f"arm:{i}": prediction for i in range(10)}),
        kwargs.pop("targets", targets),
        kwargs.pop("operator", operator),
        kwargs.pop("groups", groups),
        gene_ids=("A", "B", "C"),
        normalization_gene_ids=("A", "B", "C"),
        target_metadata=kwargs.pop("metadata", metadata),
        provenance=provenance,
        **kwargs,
    )


def test_all_saved_arms_use_identical_shared_regions_and_fixed_normalizer(prepared):
    report = _report(prepared, patient_metadata={"reference_modality": "snRNA", "site": "site-a"})
    pair = report["pairs"][0]
    assert pair["n_shared_eligible_regions"] == 3
    assert pair["shared_eligible_region_ids"] == list(prepared[0].cell_ids[:3])
    assert len(pair["same_region_model_scores"]) == 10
    for score in pair["same_region_model_scores"].values():
        assert score["mean_replicate_median_gene_spearman"] == pytest.approx(1)
        assert score["normalized_log_mse"] == pytest.approx(0, abs=1e-10)
        assert [r["assay_replicate"] for r in score["replicate_scores"]] == ["1", "2"]
        assert all(
            gene["normalized_log_mse"] == pytest.approx(0, abs=1e-10)
            for rep in score["replicate_scores"]
            for gene in rep["gene_scores"]
        )
    assert report["all_eligible_model_scores"]["arm:0"]["normalized_log_mse"] > 0
    assert all(r["spearman"] == pytest.approx(1) for r in pair["gene_scores"])
    assert pair["coverage_by_replicate"]["1"]["target_bead_count"]["mean"] == 2
    assert pair["coverage_by_replicate"]["2"]["target_bead_count"]["mean"] == 6
    for coverage in pair["coverage_by_replicate"].values():
        assert coverage["codex_any_usable"] == 2
        assert coverage["codex_complete_observed"] == 2
        assert coverage["codex_complete_usable"] == 1
    assert report["descriptive_patient_metadata"]["reference_modality"] == "snRNA"
    # Changing model outcomes cannot change the geometry-selected measurement overlap.
    changed = replace(prepared[1], expression=torch.flip(prepared[1].expression, (0,)))
    altered = _report(prepared, predictions={"adversarial": changed})
    assert altered["pairs"][0]["shared_eligible_region_ids"] == pair["shared_eligible_region_ids"]
    assert altered["pairs"][0]["gene_scores"] == pair["gene_scores"]


def test_single_replicate_missing_eligibility_and_disjoint_regions_are_distinct(prepared):
    _, _, targets, operator, groups, metadata, _ = prepared
    keep = np.array(groups) == "1"
    target, mapping = _subset(targets, operator, keep)
    report = _report(prepared, targets=target, operator=mapping, groups=("1",) * 4)
    assert report["status"] == "replicate_without_eligible_regions"
    assert report["pairs"][0]["status"] == "replicate_without_eligible_regions"
    single = {
        "all_observation_ids": metadata["all_observation_ids"][:4],
        "region_bead_counts": metadata["region_bead_counts"][:4],
        "observation_replicates": ["1"] * 4,
        "source_replicates": {"1": 10},
    }
    report = _report(prepared, targets=target, operator=mapping, groups=("1",) * 4, metadata=single)
    assert report["status"] == "single_replicate"
    assert report["pairs"] == []
    keep = np.array([True, False, False, False, False, True, True])
    target, mapping = _subset(targets, operator, keep)
    report = _report(prepared, targets=target, operator=mapping, groups=("1", "2", "2"))
    assert report["status"] == "no_shared_eligible_regions"
    assert report["pairs"][0]["same_region_model_scores"] == {}
    assert report["pairs"][0]["coverage_by_replicate"]["1"]["n_region_replicates"] == 0


def test_missing_or_misaligned_bead_metadata_and_unqualified_exposure_fail(prepared):
    bad = {**prepared[5], "region_bead_counts": [1, 2]}
    with pytest.raises(ValueError, match="bead-count"):
        _report(prepared, metadata=bad)
    bad = {**prepared[5], "observation_replicates": ["2"] * 7}
    with pytest.raises(ValueError, match="replicate metadata"):
        _report(prepared, metadata=bad)
    values = prepared[2].expression.copy()
    values[0] = 0
    with pytest.raises(ValueError, match="qualified positive"):
        _report(prepared, targets=replace(prepared[2], expression=values))


def _fitted(inputs):
    return FittedModel(
        model("he_rna_protein"),
        np.zeros(2),
        np.ones(2),
        ("other_patient",),
        [],
        {},
        training_config={
            "prediction_unit": inputs.prediction_unit,
            "coordinate_units": inputs.coordinate_units,
        },
    )


def test_actual_protein_update_and_interventions_preserve_fitted_state(prepared):
    inputs, _, _, operator, _, _, _ = prepared
    fitted = _fitted(inputs)
    saved = fitted.predict(inputs)
    weights = {k: v.clone() for k, v in fitted.model.state_dict().items()}
    config = fitted.model.config.to_dict()
    report = protein_pathway_diagnostics(
        fitted, inputs, scored_operator=operator, expected_prediction=saved
    )
    assert report["expected_saved_prediction_verified"]
    assert "_protein_logits" not in fitted.model.__dict__
    assert fitted.model.config.to_dict() == config
    for key, value in fitted.model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    for marker in report["state_marker_templates"]:
        assert marker["range"] == pytest.approx(0.8)
        assert marker["std_across_states"] == pytest.approx(0.4)
        assert marker["reference_proportion_weighted_std"] == pytest.approx(0.4)
    logits = report["logit_magnitude"]["unique_scored_regions"]
    assert logits["raw"]["actual_protein_adjustment"]["root_mean_square"] > 0
    assert logits["centered_within_region"]["actual_protein_adjustment"]["root_mean_square"] > 0
    off = report["prediction_interventions"]["zero_confidence"]["unique_scored_regions"]
    assert off["conditional_assignment"]["mean_row_l1_change"] > 0
    assert off["log_correction"]["max_absolute_entry_change"] == 0
    corrupt = replace(saved, expression=saved.expression + 1)
    with pytest.raises(ValueError, match="replay differs"):
        protein_pathway_diagnostics(fitted, inputs, expected_prediction=corrupt)
    assert "_protein_logits" not in fitted.model.__dict__


def test_constant_templates_and_zero_compatibility_are_explicit(prepared):
    inputs = prepared[0]
    inputs = replace(inputs, proteins=replace(inputs.proteins, state_values=np.full((2, 2), 0.5)))
    fitted = _fitted(inputs)
    with torch.no_grad():
        fitted.model.image_mapper[0].weight.zero_()
        fitted.model.image_mapper[0].bias.zero_()
    report = protein_pathway_diagnostics(fitted, inputs)
    assert all(row["std_across_states"] == 0 for row in report["state_marker_templates"])
    for scope in ("raw", "centered_within_region"):
        row = report["logit_magnitude"]["all_he_regions"][scope]
        assert row["adjustment_to_base_rms_ratio"] is None
        assert row["ratio_status"] == "undefined_zero_base_rms"
    assert (
        report["logit_magnitude"]["all_he_regions"]["centered_within_region"][
            "actual_protein_adjustment"
        ]["root_mean_square"]
        == 0
    )
    assert (
        report["prediction_interventions"]["zero_confidence"]["all_he_regions"][
            "conditional_assignment"
        ]["max_absolute_entry_change"]
        == 0
    )


def test_actual_update_capture_restores_existing_override_on_failure(prepared, monkeypatch):
    inputs = prepared[0]
    fitted = _fitted(inputs)

    def failure(logits, actual_inputs):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(fitted.model, "_protein_logits", failure)
    with pytest.raises(RuntimeError, match="sentinel"):
        protein_pathway_diagnostics(fitted, inputs)
    assert fitted.model._protein_logits is failure
