from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.evaluate import aligned_observations, score_patient
from heir.personalized.graph import LRInteraction, build_graph
from heir.personalized.model import ModelConfig
from heir.personalized.observation import ObservationOperator, normalized_log_expression
from heir.personalized.schema import (
    Prediction,
    PredictionInputs,
    ProteinEvidence,
    RNAReference,
    SpatialTargets,
)
from heir.personalized.train import (
    FittedModel,
    TrainingConfig,
    TrainingSample,
    fit,
    make_patient_folds,
    read_fold_roles,
)


def _prediction(values, *, cells=("c1", "c2", "c3"), genes=("A", "B", "C")):
    expression = torch.as_tensor(values, dtype=torch.float32)
    n, g = expression.shape
    return Prediction(
        cell_ids=cells,
        gene_ids=genes,
        state_ids=("unsupported",),
        coordinates_um=torch.zeros((n, 2)),
        assignment=torch.ones((n, 1)),
        transfer=expression.clone(),
        log_correction=torch.zeros((n, g)),
        expression=expression,
        assignment_entropy=torch.zeros(n),
        reference_support=torch.zeros(n),
        registration_sensitivity=torch.full((n,), float("nan")),
        supervised_genes=torch.ones(g, dtype=torch.bool),
        independently_evaluated_genes=torch.zeros(g, dtype=torch.bool),
        specimen_id="S1",
        patient_id="P1",
    )


def _sample(patient="P1", shift=0.0, missing=False):
    cells = tuple(f"c{i}" for i in range(6))
    xy = np.column_stack((np.arange(6) * 10.0, np.zeros(6)))
    reference = RNAReference(
        state_ids=("s1", "s2"),
        gene_ids=("A", "B", "C"),
        expression=np.asarray([[8.0, 1.0, 3.0], [1.0, 8.0, 2.0]]),
        embedding=np.log1p([[8.0, 1.0], [1.0, 8.0]]),
        embedding_gene_ids=("A", "B"),
        proportions=np.asarray([0.5, 0.5]),
        state_types=("t1", "t2"),
        variance=np.zeros((2, 3)),
        cell_counts=np.asarray([10, 10]),
    )
    features = np.column_stack((np.arange(6) / 5, [0, 1, 0, 1, 0, 1])) + shift
    inputs = PredictionInputs(
        specimen_id=f"S-{patient}",
        patient_id=patient,
        cell_ids=cells,
        coordinates_um=xy,
        image_features=features,
        reference=reference,
        proteins=ProteinEvidence(
            marker_ids=("markerA",),
            values=np.linspace(1, 0, 6)[:, None],
            observed=np.ones((6, 1), dtype=bool),
            state_values=np.asarray([[1.0], [0.0]]),
            correspondence="same_cell",
        ),
        graph=build_graph(xy, 25, tissue_component_ids=("tissue",) * 6),
    )
    targets = np.asarray([[16.0, 2.0, 6.0], [8.0, 10.0, 5.0], [2.0, 16.0, 4.0]])
    mask = np.ones((3, 3), dtype=bool)
    if missing:
        mask[1, 0] = False
        targets[1, 0] = np.nan
    target = SpatialTargets(
        observation_ids=("r0", "r1", "r2"),
        gene_ids=("A", "B", "C"),
        expression=targets,
        observed=mask,
        patient_id=patient,
        specimen_id=f"S-{patient}",
    )
    operator = ObservationOperator(
        cell_ids=cells,
        observation_ids=target.observation_ids,
        rows=np.repeat(np.arange(3), 2),
        columns=np.arange(6),
        weights=np.ones(6),
    )
    return TrainingSample(inputs, target, operator)


def _config(variant="he_rna"):
    interactions = (LRInteraction(("A",), ("B",), radius_um=25),)
    return ModelConfig(
        feature_dim=2,
        state_dim=2,
        state_gene_ids=("A", "B"),
        gene_ids=("A", "B", "C"),
        supervised_gene_ids=("A", "B"),
        hidden_dim=4,
        residual_rank=2,
        variant=variant,
        lr_interactions=interactions if variant.startswith(("lr_", "generic_")) else (),
    )


def test_sparse_observation_joins_cells_and_genes_before_linear_aggregation():
    prediction = _prediction(
        [[6.0, 4.0], [2.0, 1.0], [4.0, 3.0]],
        cells=("c3", "c1", "c2"),
        genes=("B", "A"),
    )
    operator = ObservationOperator(
        cell_ids=("c1", "c2", "c3"),
        observation_ids=("r1", "r2"),
        rows=np.asarray([0, 0, 1]),
        columns=np.asarray([0, 1, 2]),
        weights=np.asarray([1.0, 0.5, 2.0]),
    )
    targets = SpatialTargets(
        observation_ids=("r2", "r1"),
        gene_ids=("A", "B"),
        expression=np.asarray([[8.0, 12.0], [2.5, 4.0]]),
        observed=np.ones((2, 2), dtype=bool),
        patient_id="P1",
        specimen_id="S1",
    )
    predicted, truth, mask = aligned_observations(prediction, targets, operator, ("A", "B"))
    expected = torch.tensor([[2.5, 4.0], [8.0, 12.0]])
    torch.testing.assert_close(predicted, expected)
    torch.testing.assert_close(truth, expected)
    normalized = normalized_log_expression(predicted, mask)
    torch.testing.assert_close(normalized[0], torch.log1p(torch.tensor([2.5, 4.0]) / 6.5 * 1e4))
    # A log-before-sum pipeline gives a different number even for this simple footprint.
    wrong = torch.log1p(torch.tensor([1.0, 2.0])) + 0.5 * torch.log1p(torch.tensor([3.0, 4.0]))
    assert not torch.allclose(torch.log1p(predicted[0]), wrong)
    with pytest.raises(ValueError, match="only linear expression"):
        operator.aggregate(prediction, component="log_correction")


def test_masked_nan_targets_never_change_fitted_model():
    sample = _sample(missing=True)
    settings = TrainingConfig(epochs=3, seed=7)
    first = fit([sample], _config(), settings)
    changed = sample.targets.expression.copy()
    changed[~sample.targets.observed] = 1e30
    second = fit(
        [replace(sample, targets=replace(sample.targets, expression=changed))], _config(), settings
    )
    assert first.history == second.history
    torch.testing.assert_close(
        first.predict(sample.inputs).expression,
        second.predict(sample.inputs).expression,
        rtol=0,
        atol=0,
    )


def test_unmeasured_training_gene_cannot_enable_a_learned_residual():
    sample = _sample()
    mask = sample.targets.observed.copy()
    values = sample.targets.expression.copy()
    mask[:, 0] = False
    values[:, 0] = np.nan
    missing_a = replace(sample, targets=replace(sample.targets, observed=mask, expression=values))
    with pytest.raises(ValueError, match="supervised genes have no observed training targets"):
        fit([missing_a], _config(), TrainingConfig(epochs=1))


def test_saved_inference_reuses_training_only_feature_transform(tmp_path):
    samples = [_sample("P1", shift=0), _sample("P2", shift=2)]
    fitted = fit(samples, _config(), TrainingConfig(epochs=3, seed=9))
    training_features = np.concatenate([sample.inputs.image_features for sample in samples])
    np.testing.assert_allclose(fitted.feature_mean, training_features.mean(axis=0))
    np.testing.assert_allclose(fitted.feature_scale, training_features.std(axis=0))
    held_out = _sample("P3", shift=1000)
    mean_before = fitted.feature_mean.copy()
    before = fitted.predict(held_out.inputs)
    assert fitted.training_patient_ids == ("P1", "P2")
    np.testing.assert_array_equal(fitted.feature_mean, mean_before)
    np.testing.assert_allclose(
        fitted.transform(held_out.inputs).image_features,
        (held_out.inputs.image_features - mean_before) / fitted.feature_scale,
    )
    fitted.save(tmp_path / "bundle")
    loaded = FittedModel.load(tmp_path / "bundle")
    after = loaded.predict(held_out.inputs)
    torch.testing.assert_close(after.expression, before.expression, rtol=0, atol=0)
    torch.testing.assert_close(after.assignment, before.assignment, rtol=0, atol=0)
    np.testing.assert_array_equal(loaded.feature_mean, fitted.feature_mean)
    assert loaded.model.config.state_gene_ids == ("A", "B")


def test_context_requires_fitted_baseline_and_load_preserves_frozen_assignment(tmp_path):
    sample = _sample()
    settings = TrainingConfig(epochs=2, seed=12)
    with pytest.raises(ValueError, match="requires a protein baseline"):
        fit([sample], _config("lr_fixed"), settings)
    baseline = fit([sample], _config("he_rna_protein"), settings)
    fixed = fit([sample], _config("lr_fixed"), settings, baseline=baseline)
    torch.testing.assert_close(
        fixed.predict(sample.inputs).assignment,
        baseline.predict(sample.inputs).assignment,
        rtol=0,
        atol=0,
    )
    fixed.save(tmp_path / "fixed")
    loaded = FittedModel.load(tmp_path / "fixed")
    assert loaded.model.fixed_mapper_initialized
    for name in ("image_mapper", "state_mapper", "unsupported_head"):
        parameters = getattr(loaded.model, name).parameters()
        assert all(not parameter.requires_grad for parameter in parameters)
    assert not loaded.model.protein_log_strength.requires_grad
    torch.testing.assert_close(
        loaded.predict(sample.inputs).expression,
        fixed.predict(sample.inputs).expression,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="same patients"):
        fit([_sample("P2")], _config("generic_fixed"), settings, baseline=baseline)
    with pytest.raises(ValueError, match="inference diagnostic"):
        fit([sample], replace(_config("lr_fixed"), transfer_only=True), settings, baseline=baseline)


def test_thirteen_patient_folds_are_disjoint_and_cover_every_patient_once_in_test(tmp_path):
    patients = tuple(f"patient-{i:02d}" for i in range(13))
    rows = make_patient_folds(patients, seed=42)
    path = tmp_path / "folds.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("outer_fold", "patient_id", "role"))
        writer.writeheader()
        writer.writerows(rows)
    sizes, all_test = [], []
    for outer in range(5):
        roles = read_fold_roles(path, outer)
        grouped = {
            role: {p for p, value in roles.items() if value == role}
            for role in ("train", "validation", "test")
        }
        assert set.union(*grouped.values()) == set(patients)
        assert not grouped["train"] & grouped["test"]
        assert not grouped["train"] & grouped["validation"]
        assert not grouped["validation"] & grouped["test"]
        assert len(grouped["validation"]) == 2
        assert len(grouped["train"]) in {8, 9}
        sizes.append(len(grouped["test"]))
        all_test.extend(grouped["test"])
    assert sizes == [3, 3, 3, 2, 2]
    assert sorted(all_test) == list(patients)
    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow([0, patients[0], "test"])
    with pytest.raises(ValueError, match="exactly one role"):
        read_fold_roles(path, 0)


def test_constant_prediction_cannot_improve_headline_by_dropping_a_gene():
    prediction = _prediction([[1, 1, 2], [1, 2, 1], [1, 2.5, 0.5]])
    targets = SpatialTargets(
        observation_ids=("r1", "r2", "r3"),
        gene_ids=("A", "B", "C"),
        expression=np.asarray([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [0.5, 2.5, 1.0]]),
        observed=np.ones((3, 3), dtype=bool),
        patient_id="P1",
        specimen_id="S1",
    )
    operator = ObservationOperator(
        cell_ids=prediction.cell_ids,
        observation_ids=targets.observation_ids,
        rows=np.arange(3),
        columns=np.arange(3),
        weights=np.ones(3),
    )
    report = score_patient([(prediction, targets, operator)], ("A", "B", "C"))
    assert report["gene_scores"][0]["undefined_reason"] == "constant_prediction"
    assert report["requested_genes"] == 3
    assert report["defined_subset_median_diagnostic"] is not None
    assert report["median_gene_spearman"] is None
    assert report["headline_status"] == "undefined_prediction_on_evaluable_panel"
