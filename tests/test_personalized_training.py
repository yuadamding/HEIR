from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.evaluate import (
    aligned_observations,
    normalized_observations,
    score,
    score_patient,
)
from heir.personalized.graph import LRInteraction, build_graph
from heir.personalized.model import ModelConfig, PersonalizedModel
from heir.personalized.observation import ObservationOperator, normalized_log_expression
from heir.personalized.schema import (
    Prediction,
    PredictionInputs,
    ProteinEvidence,
    ProteinObservationOperator,
    RNAReference,
    SpatialTargets,
)
from heir.personalized.train import (
    FittedModel,
    TrainingConfig,
    TrainingSample,
    _abundance_penalty,
    _expression_loss,
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
        conditional_assignment=torch.empty((n, 0)),
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
            operator=ProteinObservationOperator(
                cell_ids=cells,
                observation_ids=cells,
                rows=np.arange(6),
                columns=np.arange(6),
                weights=np.ones(6),
            ),
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


def _change_units(inputs, **changes):
    graph = replace(
        inputs.graph, distance_units=changes.get("coordinate_units", inputs.coordinate_units)
    )
    return replace(inputs, graph=graph, **changes)


def _config(variant="he_rna", *, radius_units="um"):
    interactions = (LRInteraction(("A",), ("B",), radius_um=25, radius_units=radius_units),)
    return ModelConfig(
        feature_dim=2,
        state_dim=2,
        state_gene_ids=("A", "B"),
        gene_ids=("A", "B", "C"),
        supervised_gene_ids=("A", "B"),
        normalization_gene_ids=("B", "C"),
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
    normalized = normalized_log_expression(predicted, mask, normalization_indices=(0, 1))
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
    report = score(
        prediction, targets, operator, ("A", "B", "C"), normalization_gene_ids=("A", "B", "C")
    )
    assert report["gene_scores"][0]["undefined_reason"] == "constant_prediction"
    assert report["requested_genes"] == 3
    assert report["defined_subset_median_diagnostic"] is not None
    assert report["median_gene_spearman"] is None
    assert report["headline_status"] == "undefined_prediction_on_evaluable_panel"


def _identity_case(predicted, measured, *, observed=None, specimen="S1"):
    predicted, measured = np.asarray(predicted), np.asarray(measured)
    cells = tuple(f"c{i}" for i in range(len(predicted)))
    genes = tuple("ABC"[: predicted.shape[1]])
    prediction = replace(_prediction(predicted, cells=cells, genes=genes), specimen_id=specimen)
    targets = SpatialTargets(
        observation_ids=cells,
        gene_ids=genes,
        expression=measured,
        observed=np.ones(measured.shape, dtype=bool) if observed is None else observed,
        patient_id="P1",
        specimen_id=specimen,
    )
    operator = ObservationOperator(
        cell_ids=cells,
        observation_ids=cells,
        rows=np.arange(len(cells)),
        columns=np.arange(len(cells)),
        weights=np.ones(len(cells)),
    )
    return prediction, targets, operator


def test_missing_denominator_gene_cannot_manufacture_spatial_correlation():
    sample = _identity_case(
        [[1.0, 1.0]] * 6,
        [[1.0, 10.0]] * 6,
        observed=np.asarray([[True, False], [True, True]] * 3),
    )
    with pytest.raises(ValueError, match="incomplete normalization coverage"):
        score(*sample, ("A", "B"), normalization_gene_ids=("A", "B"))
    with pytest.raises(ValueError, match="incomplete normalization coverage"):
        score(*sample, ("A",), normalization_gene_ids=("A", "B"))


def test_outcome_subset_cannot_redefine_normalized_values_or_gene_scores():
    sample = _identity_case(
        [[2, 3, 5], [4, 2, 2], [5, 6, 3], [3, 8, 7]],
        [[3, 4, 2], [7, 2, 1], [4, 9, 3], [5, 7, 3]],
    )
    full = normalized_observations(*sample, ("A", "B", "C"), normalization_gene_ids=("B", "C"))
    subset = normalized_observations(*sample, ("A",), normalization_gene_ids=("B", "C"))
    for all_values, selected_values in zip(full, subset):
        torch.testing.assert_close(all_values[:, :1], selected_values, rtol=0, atol=0)
    full_report = score(*sample, ("A", "B", "C"), normalization_gene_ids=("B", "C"))
    subset_report = score(*sample, ("A",), normalization_gene_ids=("B", "C"))
    assert full_report["gene_scores"][0] == subset_report["gene_scores"][0]


def test_single_outcome_has_gradient_with_separate_normalization_basis():
    values = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], requires_grad=True)
    truth = torch.tensor([[2.0, 1.0], [1.0, 3.0], [1.0, 4.0]])
    observed = torch.ones_like(values, dtype=torch.bool)
    prediction = normalized_log_expression(values, observed, normalization_indices=(0, 1))
    target = normalized_log_expression(truth, observed, normalization_indices=(0, 1))
    loss = (prediction[:, 0] - target[:, 0]).square().mean()
    loss.backward()
    assert loss.item() > 0
    assert torch.all(values.grad[:, 0].abs() > 1e-6)
    fitted = fit(
        [_sample()],
        replace(_config(), supervised_gene_ids=("A",)),
        TrainingConfig(epochs=1, protein_weight=0, abundance_weight=0, residual_weight=0),
    )
    assert fitted.history[0]["loss"] > 0
    with pytest.raises(ValueError, match="normalization"):
        normalized_log_expression(values[:, :1], observed[:, :1], normalization_indices=(0,))
    with pytest.raises(ValueError, match="normalization"):
        replace(_config(), normalization_gene_ids=("A",))


def test_missing_outcome_does_not_change_another_genes_normalization():
    values = torch.tensor([[2.0, 3.0, 5.0], [2.0, 3.0, float("nan")]])
    observed = torch.tensor([[True, True, True], [True, True, False]])
    normalized = normalized_log_expression(values, observed, normalization_indices=(0, 1))
    torch.testing.assert_close(normalized[0, :2], normalized[1, :2], rtol=0, atol=0)


def test_empty_denominator_is_rejected_instead_of_replacing_exposure():
    with pytest.raises(ValueError, match="denominator must be finite and positive"):
        normalized_log_expression(
            torch.zeros((3, 2)), torch.ones((3, 2), dtype=torch.bool), normalization_indices=(0, 1)
        )


@pytest.mark.parametrize("target_anchor", [0.0, 4.0])
def test_effectively_one_gene_training_is_rejected_when_prediction_anchor_is_zero(target_anchor):
    sample = _sample()
    expression = sample.inputs.reference.expression.copy()
    expression[:, 1] = 0
    targets = sample.targets.expression.copy()
    targets[:, 1] = target_anchor
    sample = replace(
        sample,
        inputs=replace(
            sample.inputs, reference=replace(sample.inputs.reference, expression=expression)
        ),
        targets=replace(sample.targets, expression=targets),
    )
    config = replace(_config(), supervised_gene_ids=("A",), normalization_gene_ids=("A", "B"))
    with pytest.raises(ValueError, match="degenerate normalization.*\\['A'\\]"):
        fit([sample], config, TrainingConfig(epochs=1))


def test_single_sparse_region_does_not_invalidate_a_usable_training_panel():
    sample = _sample()
    targets = sample.targets.expression.copy()
    targets[0, 1] = 0
    config = replace(_config(), supervised_gene_ids=("A",), normalization_gene_ids=("A", "B"))
    fit(
        [replace(sample, targets=replace(sample.targets, expression=targets))],
        config,
        TrainingConfig(epochs=1),
    )


def test_between_specimen_means_cannot_create_within_specimen_spatial_accuracy():
    first = _identity_case([[1.0, 9.0]] * 4, [[1.0, 9.0]] * 4, specimen="S1")
    second = _identity_case([[9.0, 1.0]] * 4, [[9.0, 1.0]] * 4, specimen="S2")
    report = score_patient([first, second], ("A", "B"), normalization_gene_ids=("A", "B"))
    assert report["mean_specimen_median_gene_spearman"] is None
    assert report["headline_status"] == "no_evaluable_target_genes"
    assert report["evaluable_specimens"] == 0
    assert all(item["median_gene_spearman"] is None for item in report["specimen_scores"])


def test_patient_endpoint_weights_specimens_equally_not_by_region_count():
    first = _identity_case([[1, 3], [2, 2], [3, 1]], [[1, 3], [2, 2], [3, 1]], specimen="S1")
    second = _identity_case(
        [[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]],
        [[6, 1], [5, 2], [4, 3], [3, 4], [2, 5], [1, 6]],
        specimen="S2",
    )
    report = score_patient([first, second], ("A", "B"), normalization_gene_ids=("A", "B"))
    assert report["mean_specimen_median_gene_spearman"] == pytest.approx(0)
    assert [item["median_gene_spearman"] for item in report["specimen_scores"]] == pytest.approx(
        [1, -1]
    )


def test_support_head_cannot_reweight_conditional_abundance():
    sample = _sample()
    model = PersonalizedModel(_config())
    before = model(sample.inputs)
    with torch.no_grad():
        model.unsupported_head.weight.fill_(100)
        model.unsupported_head.bias.fill_(50)
    after = model(sample.inputs)
    assert not torch.allclose(before.reference_support, after.reference_support)
    torch.testing.assert_close(
        before.conditional_assignment, after.conditional_assignment, rtol=0, atol=0
    )
    torch.testing.assert_close(
        _abundance_penalty(before.conditional_assignment, sample.inputs.reference.proportions),
        _abundance_penalty(after.conditional_assignment, sample.inputs.reference.proportions),
        rtol=0,
        atol=0,
    )


def test_legacy_bundle_cannot_silently_acquire_new_numerical_semantics(tmp_path):
    fitted = fit([_sample()], _config(), TrainingConfig(epochs=1))
    bundle = tmp_path / "bundle"
    fitted.save(bundle)
    metadata_path = bundle / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema"] == "heir.personalized.bundle.v2"
    metadata["schema"] = "heir.personalized.bundle.v1"
    metadata_path.write_text(json.dumps(metadata))
    hashes_path = bundle / "sha256.json"
    hashes = json.loads(hashes_path.read_text())
    hashes["bundle.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="retrain legacy bundles"):
        FittedModel.load(bundle)


@pytest.mark.parametrize(
    "changes", [{"prediction_unit": "tissue_region"}, {"coordinate_units": "registered_pixel"}]
)
def test_fit_rejects_mixed_prediction_or_coordinate_units(changes):
    first, second = _sample("P1"), _sample("P2")
    second = replace(second, inputs=_change_units(second.inputs, **changes))
    with pytest.raises(ValueError, match="share prediction_unit and coordinate_units"):
        fit([first, second], _config(), TrainingConfig(epochs=1))


def test_registered_pixel_region_fit_and_context_keep_unit_contract(tmp_path):
    sample = _sample()
    regions = replace(
        sample,
        inputs=_change_units(
            sample.inputs, prediction_unit="tissue_region", coordinate_units="registered_pixel"
        ),
    )
    settings = TrainingConfig(epochs=1, abundance_weight=0)
    baseline = fit([regions], _config("he_rna_protein"), settings)
    assert baseline.training_config["prediction_unit"] == "tissue_region"
    assert baseline.training_config["coordinate_units"] == "registered_pixel"
    fixed = fit(
        [regions], _config("lr_fixed", radius_units="registered_pixel"), settings, baseline=baseline
    )
    assert fixed.predict(regions.inputs).prediction_unit == "tissue_region"
    fixed.save(tmp_path / "region_bundle")
    restored = FittedModel.load(tmp_path / "region_bundle")
    assert restored.training_config["coordinate_units"] == "registered_pixel"
    assert restored.model.config.lr_interactions[0].radius_units == "registered_pixel"
    assert restored.model.config == fixed.model.config
    torch.testing.assert_close(
        restored.predict(regions.inputs).expression,
        fixed.predict(regions.inputs).expression,
        atol=0,
        rtol=0,
    )
    for incompatible in (
        replace(regions.inputs, prediction_unit="nucleus"),
        _change_units(regions.inputs, coordinate_units="um"),
    ):
        for operation in (restored.transform, restored.predict):
            with pytest.raises(ValueError, match="differs from fitted model"):
                operation(incompatible)
    with pytest.raises(ValueError, match="baseline prediction_unit or coordinate_units differs"):
        fit([sample], _config("lr_fixed"), settings, baseline=baseline)
    with pytest.raises(ValueError, match="baseline preparation settings differ"):
        fit(
            [regions],
            _config("lr_fixed"),
            settings,
            baseline=baseline,
            preparation={"different_regions": True},
        )


def test_legacy_bundle_without_units_keeps_nucleus_um_inference_contract(tmp_path):
    sample = _sample()
    fitted = fit([sample], _config(), TrainingConfig(epochs=1))
    del fitted.training_config["prediction_unit"], fitted.training_config["coordinate_units"]
    fitted.save(tmp_path / "legacy_v2")
    loaded = FittedModel.load(tmp_path / "legacy_v2")
    expected = fitted.predict(sample.inputs)
    actual = loaded.predict(sample.inputs)
    torch.testing.assert_close(actual.expression, expected.expression, atol=0, rtol=0)
    assert actual.prediction_unit == "nucleus" and actual.coordinate_units == "um"
    with pytest.raises(ValueError, match="differs from fitted model"):
        loaded.predict(
            _change_units(
                sample.inputs, prediction_unit="tissue_region", coordinate_units="registered_pixel"
            )
        )


def test_scores_preserve_region_unit_metadata():
    first = _identity_case([[1, 3], [2, 2], [3, 1]], [[1, 3], [2, 2], [3, 1]], specimen="S1")
    first = (
        replace(first[0], prediction_unit="tissue_region", coordinate_units="registered_pixel"),
        first[1],
        first[2],
    )
    individual = score(*first, ("A", "B"), normalization_gene_ids=("A", "B"))
    patient = score_patient([first], ("A", "B"), normalization_gene_ids=("A", "B"))
    for report in (individual, patient, patient["specimen_scores"][0]):
        assert report["prediction_unit"] == "tissue_region"
        assert report["coordinate_units"] == "registered_pixel"
    assert individual["median_gene_spearman"] == pytest.approx(1)


@pytest.mark.parametrize(
    "changes", [{"prediction_unit": "tissue_region"}, {"coordinate_units": "registered_pixel"}]
)
def test_patient_score_rejects_mixed_units(changes):
    first = _identity_case([[1, 3], [2, 2], [3, 1]], [[1, 3], [2, 2], [3, 1]], specimen="S1")
    second = _identity_case([[1, 3], [2, 2], [3, 1]], [[1, 3], [2, 2], [3, 1]], specimen="S2")
    second = (replace(second[0], **changes), second[1], second[2])
    with pytest.raises(ValueError, match="share prediction_unit and coordinate_units"):
        score_patient([first, second], ("A", "B"), normalization_gene_ids=("A", "B"))


@pytest.mark.parametrize("groups", [("one",), ("one", "", "two"), ("one", None, "two"), "abc"])
def test_training_observation_groups_require_named_aligned_rows(groups):
    with pytest.raises(ValueError, match="observation_groups"):
        replace(_sample(), observation_groups=groups)


def test_expression_loss_equalizes_replicates_with_unequal_coverage_and_missingness():
    error = torch.tensor([[4.0, 100.0], [1.0, 9.0], [1.0, 100.0], [1.0, 100.0]], requires_grad=True)
    observed = torch.tensor([[True, False], [True, True], [True, False], [True, False]])
    grouped = _expression_loss(
        error, observed, ("replicate1", "replicate2", "replicate2", "replicate2")
    )
    # Replicate 1 contributes 4; replicate 2 contributes mean(gene A=1, gene B=9)=5.
    assert grouped.item() == pytest.approx(4.5)
    assert _expression_loss(error, observed).item() == pytest.approx(5.375)
    grouped.backward()
    assert not error.grad[~observed].any()
    torch.testing.assert_close(error.grad[:, 0], torch.tensor([0.5, 1 / 12, 1 / 12, 1 / 12]))
    assert error.grad[1, 1].item() == pytest.approx(0.25)


def test_fit_uses_replicate_groups_without_changing_ungrouped_behavior():
    sample = _sample(missing=True)
    settings = TrainingConfig(
        epochs=1, seed=17, protein_weight=0, abundance_weight=0, residual_weight=0
    )
    ungrouped = fit([sample], _config(), settings)
    one_group = fit([replace(sample, observation_groups=("all",) * 3)], _config(), settings)
    assert one_group.history == ungrouped.history
    for key, expected in ungrouped.model.state_dict().items():
        torch.testing.assert_close(one_group.model.state_dict()[key], expected, atol=0, rtol=0)
    torch.manual_seed(settings.seed)
    initial = PersonalizedModel(_config())
    transformed = ungrouped.transform(sample.inputs)
    predicted, truth, observed = normalized_observations(
        initial(transformed),
        sample.targets,
        sample.observation,
        _config().supervised_gene_ids,
        normalization_gene_ids=_config().normalization_gene_ids,
    )
    error = (predicted - truth).square()
    assert not observed[1, 0]
    expected = (error[0].mean() + (error[2, 0] + error[1:, 1].mean()) / 2) / 2
    grouped = fit(
        [replace(sample, observation_groups=("replicate1", "replicate2", "replicate2"))],
        _config(),
        settings,
    )
    assert grouped.history[0]["loss"] == pytest.approx(expected.item())
    assert grouped.history[0]["loss"] != pytest.approx(ungrouped.history[0]["loss"])
