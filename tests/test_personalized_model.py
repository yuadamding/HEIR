from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.graph import LRInteraction, build_graph, compute_messages
from heir.personalized.model import VARIANTS, ModelConfig, PersonalizedModel
from heir.personalized.schema import (
    PredictionInputs,
    ProteinEvidence,
    ProteinObservationOperator,
    RNAReference,
    SpatialGraph,
)


def identity_protein_operator(cell_ids):
    indices = np.arange(len(cell_ids))
    return ProteinObservationOperator(cell_ids, cell_ids, indices, indices, np.ones(len(cell_ids)))


def specimen() -> PredictionInputs:
    reference = RNAReference(
        state_ids=("s1", "s2"),
        gene_ids=("A", "B", "C"),
        expression=np.asarray([[4.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        embedding=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        embedding_gene_ids=("A", "B"),
        proportions=np.asarray([0.5, 0.5]),
        state_types=("type1", "type2"),
        variance=np.zeros((2, 3)),
        cell_counts=np.asarray([3, 3]),
    )
    xy = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    graph = build_graph(xy, 15.0, tissue_component_ids=("a", "a", "a"))
    return PredictionInputs(
        specimen_id="sample",
        patient_id="patient",
        cell_ids=("c1", "c2", "c3"),
        coordinates_um=xy,
        image_features=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        reference=reference,
        graph=graph,
    )


def model(variant="he_rna", **kwargs) -> PersonalizedModel:
    torch.manual_seed(23)
    if variant.startswith(("generic", "lr")):
        kwargs["lr_interactions"] = (LRInteraction(("A", "B"), ("C",), radius_um=15),)
    return PersonalizedModel(
        ModelConfig(
            feature_dim=2,
            state_dim=2,
            gene_ids=("A", "B", "C"),
            supervised_gene_ids=("A", "B"),
            normalization_gene_ids=("A", "B"),
            hidden_dim=4,
            residual_rank=2,
            state_gene_ids=("A", "B"),
            variant=variant,
            **kwargs,
        )
    )


def test_reference_gene_and_state_order_do_not_change_prediction():
    inputs = specimen()
    net = model()
    before = net(inputs)
    reference = inputs.reference
    genes, states = [2, 0, 1], [1, 0]
    permuted = replace(
        reference,
        state_ids=tuple(reference.state_ids[i] for i in states),
        gene_ids=tuple(reference.gene_ids[i] for i in genes),
        expression=reference.expression[np.ix_(states, genes)],
        embedding=reference.embedding[states],
        proportions=reference.proportions[states],
        state_types=tuple(reference.state_types[i] for i in states),
        variance=reference.variance[np.ix_(states, genes)],
        cell_counts=reference.cell_counts[states],
    )
    after = net(replace(inputs, reference=permuted))
    torch.testing.assert_close(after.expression, before.expression)
    torch.testing.assert_close(after.assignment[:, [0, 2, 1]], before.assignment)
    assert after.state_ids == ("unsupported", "s2", "s1")


def test_messages_reject_unconditional_support_weighted_assignments():
    inputs = specimen()
    with pytest.raises(ValueError, match="conditional known-state probabilities"):
        compute_messages(
            inputs.graph,
            torch.full((3, 2), 0.1),
            inputs.reference,
            (LRInteraction(("A",), ("C",)),),
        )


def test_reference_embedding_feature_order_is_joined_by_gene_identity():
    inputs = specimen()
    net = model()
    before = net(inputs)
    permuted = replace(
        inputs.reference,
        embedding=inputs.reference.embedding[:, [1, 0]],
        embedding_gene_ids=("B", "A"),
    )
    after = net(replace(inputs, reference=permuted))
    torch.testing.assert_close(after.assignment, before.assignment)
    torch.testing.assert_close(after.expression, before.expression)


def test_same_sized_different_embedding_panel_is_rejected():
    inputs = specimen()
    wrong_panel = replace(inputs.reference, embedding_gene_ids=("A", "C"))
    with pytest.raises(ValueError, match="embedding panel differs.*missing genes"):
        model()(replace(inputs, reference=wrong_panel))
    with pytest.raises(ValueError, match="state_gene_ids must identify"):
        ModelConfig(
            feature_dim=2,
            state_dim=2,
            gene_ids=("A", "B"),
            supervised_gene_ids=("A",),
            normalization_gene_ids=("A", "B"),
        )


def test_unsupervised_genes_and_transfer_diagnostic_are_exact_transfer():
    inputs = specimen()
    net = model()
    predicted = net(inputs)
    assert torch.equal(predicted.expression[:, 2], predicted.transfer[:, 2])
    assert torch.equal(predicted.log_correction[:, 2], torch.zeros(3))
    assert predicted.log_correction.abs().max() <= net.config.residual_bound
    diagnostic = model(transfer_only=True)(inputs)
    assert torch.equal(diagnostic.expression, diagnostic.transfer)
    assert not diagnostic.log_correction.any()


def test_missing_protein_values_are_never_observations():
    inputs = specimen()
    evidence = ProteinEvidence(
        marker_ids=("markerA", "markerB"),
        values=np.asarray([[0.8, np.nan], [0.1, np.nan], [0.4, np.nan]]),
        observed=np.asarray([[True, False]] * 3),
        state_values=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        correspondence="serial_section",
        operator=identity_protein_operator(inputs.cell_ids),
    )
    net = model("he_rna_protein")
    before = net(replace(inputs, proteins=evidence))
    changed = evidence.values.copy()
    changed[:, 1] = [-1e9, 1e30, np.nan]
    after = net(replace(inputs, proteins=replace(evidence, values=changed)))
    torch.testing.assert_close(after.expression, before.expression)
    assert torch.isfinite(net.protein_loss(replace(inputs, proteins=evidence), before))
    without_graph = net(replace(inputs, proteins=evidence, graph=None))
    torch.testing.assert_close(without_graph.expression, before.expression)


def test_reference_deletion_changes_uncalibrated_softmax_diagnostic_mechanically():
    inputs = specimen()
    net = model()
    before = net(inputs)
    reference = inputs.reference
    depleted = replace(
        reference,
        state_ids=("s1",),
        expression=reference.expression[:1],
        embedding=reference.embedding[:1],
        proportions=np.asarray([1.0]),
        state_types=("type1",),
        variance=reference.variance[:1],
        cell_counts=np.asarray([3]),
    )
    after = net(replace(inputs, reference=depleted))
    assert torch.all(after.assignment[:, 0] > before.assignment[:, 0])
    torch.testing.assert_close(after.assignment.sum(dim=1), torch.ones(3))
    assert torch.isnan(after.registration_sensitivity).all()
    assert after.reference_support_calibrated is False
    assert after.unsupported_policy == "conditional_reference_unvalidated"


def test_he_baseline_does_not_use_reference_or_proteins():
    inputs = specimen()
    net = model("he")
    before = net(inputs)
    reference = replace(
        inputs.reference,
        expression=inputs.reference.expression * 1e6,
        embedding=inputs.reference.embedding * -5,
    )
    after = net(replace(inputs, reference=reference))
    assert torch.equal(after.expression, before.expression)
    assert after.assignment.shape == (3, 1)
    assert after.conditional_assignment.shape == (3, 0)
    assert after.unsupported_policy == "he_only"
    assert not after.reference_support.any()
    assert not after.expression[:, 2].any()


def test_direction_and_complex_members_are_preserved():
    inputs = specimen()
    graph = SpatialGraph(
        senders=np.asarray([0]),
        receivers=np.asarray([1]),
        distances_um=np.asarray([10.0]),
        weights=np.asarray([1.0]),
        n_cells=2,
    )
    q = torch.eye(2, requires_grad=True)
    interaction = LRInteraction(("A", "B"), ("C",), radius_um=15)
    lr = compute_messages(graph, q, inputs.reference, (interaction,), mode="lr")
    generic = compute_messages(graph, q, inputs.reference, (interaction,), mode="generic")
    torch.testing.assert_close(lr, torch.tensor([[0.0, 0.0], [6.0, 3.0]]))
    torch.testing.assert_close(generic, torch.tensor([[0.0, 0.0], [2.0, 3.0]]))
    lr.sum().backward()
    assert torch.isfinite(q.grad).all()
    missing = replace(interaction, ligand=("A", "missing"))
    assert not compute_messages(graph, q, inputs.reference, (missing,))[:, 0].any()
    contact = replace(interaction, mode="contact", radius_um=5)
    assert not compute_messages(graph, q, inputs.reference, (contact,))[:, 0].any()


def test_graph_prevents_component_crossing_and_allows_tissue_mask_filter():
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.1, 0.0]])
    graph = build_graph(
        xy,
        10,
        tissue_component_ids=("a", "a", "a", "b"),
        max_neighbors=1,
        edge_allowed=lambda sender, receiver: {sender, receiver} != {0, 1},
    )
    edges = set(zip(graph.senders, graph.receivers))
    assert (0, 1) not in edges and (1, 0) not in edges
    assert all(3 not in edge for edge in edges)
    assert max(np.bincount(graph.receivers)) == 1


def test_staged_context_freezes_mapper_and_bounds_refinement():
    inputs = specimen()
    baseline = model("he_rna_protein")
    fixed = model("lr_fixed")
    refined = model("lr_refined", refinement_steps=2)
    fixed.initialize_fixed_mapper(baseline)
    refined.initialize_fixed_mapper(baseline)
    assert fixed.fixed_mapper_initialized
    assert all(not value.requires_grad for value in fixed.image_mapper.parameters())
    before = baseline(inputs)
    torch.testing.assert_close(fixed(inputs).assignment, before.assignment)
    after = refined(inputs)
    # Relative odds against unsupported can move by at most the per-step bound.
    old_odds = before.assignment[:, 1:].log() - before.assignment[:, :1].log()
    new_odds = after.assignment[:, 1:].log() - after.assignment[:, :1].log()
    assert (new_odds - old_odds).abs().max() <= 2 * refined.config.refinement_bound + 1e-6
    after.expression.sum().backward()
    assert refined.refinement_message.weight.grad is not None
    restored = ModelConfig.from_dict(refined.config.to_dict())
    assert restored == refined.config


def test_predictions_accept_no_spatial_target_argument():
    net = model()
    with pytest.raises(TypeError):
        net(specimen(), spatial_targets=np.ones((3, 3)))


@pytest.mark.parametrize("variant", VARIANTS)
def test_tissue_region_prediction_preserves_declared_coordinate_units(variant):
    inputs = specimen()
    net = model(variant)
    expected = net(inputs)
    regions = replace(
        inputs,
        cell_ids=("region1", "region2", "region3"),
        prediction_unit="tissue_region",
        coordinate_units="registered_pixel",
    )
    actual = net(regions)
    assert actual.prediction_unit == "tissue_region"
    assert actual.coordinate_units == "registered_pixel"
    assert actual.cell_ids == regions.cell_ids
    np.testing.assert_array_equal(regions.coordinates, regions.coordinates_um)
    torch.testing.assert_close(actual.coordinates, actual.coordinates_um, atol=0, rtol=0)
    torch.testing.assert_close(actual.expression, expected.expression, atol=0, rtol=0)
    assert expected.prediction_unit == "nucleus"
    assert expected.coordinate_units == "um"


@pytest.mark.parametrize(
    "changes",
    [{"prediction_unit": "protein_cell"}, {"coordinate_units": "pixel"}],
)
def test_prediction_input_requires_known_unit_metadata(changes):
    with pytest.raises(ValueError):
        replace(specimen(), **changes)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("unsupported_bias", [3.0, 1000.0])
def test_unsupported_logits_cannot_weight_expression_messages_or_protein_loss(
    variant, unsupported_bias
):
    inputs = specimen()
    evidence = ProteinEvidence(
        marker_ids=("m1", "m2"),
        values=np.asarray([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]]),
        observed=np.ones((3, 2), dtype=bool),
        state_values=np.eye(2),
        correspondence="same_cell",
        operator=identity_protein_operator(inputs.cell_ids),
    )
    inputs = replace(inputs, proteins=evidence)
    net = model(variant)
    assert not any(parameter.requires_grad for parameter in net.unsupported_head.parameters())
    assert not net.unsupported_head.weight.any() and not net.unsupported_head.bias.any()
    before = net(inputs)
    with torch.no_grad():
        net.unsupported_head.weight.copy_(torch.tensor([[4.0, -3.0, 2.0, -1.0]]))
        net.unsupported_head.bias.fill_(unsupported_bias)
    after = net(inputs)
    for name in ("conditional_assignment", "transfer", "expression", "log_correction"):
        torch.testing.assert_close(getattr(after, name), getattr(before, name), atol=0, rtol=0)
    torch.testing.assert_close(net.protein_loss(inputs, after), net.protein_loss(inputs, before))
    if variant != "he":
        assert not torch.equal(after.reference_support, before.reference_support)
        torch.testing.assert_close(after.conditional_assignment.sum(1), torch.ones(3))
        if unsupported_bias == 1000:
            assert not after.reference_support.any()
            assert not after.assignment[:, 1:].any()
            assert torch.all(after.transfer.sum(1) > 0)
    if net.config.lr_interactions:
        for mode in ("generic", "lr"):
            torch.testing.assert_close(
                compute_messages(
                    inputs.graph,
                    before.conditional_assignment,
                    inputs.reference,
                    net.config.lr_interactions,
                    mode=mode,
                ),
                compute_messages(
                    inputs.graph,
                    after.conditional_assignment,
                    inputs.reference,
                    net.config.lr_interactions,
                    mode=mode,
                ),
                atol=0,
                rtol=0,
            )
    # Even deliberately re-enabling the diagnostic head cannot give expression
    # or protein objectives a gradient through that head.
    net.unsupported_head.requires_grad_(True)
    predicted = net(inputs)
    (predicted.expression.sum() + net.protein_loss(inputs, predicted)).backward()
    assert all(
        parameter.grad is None or not parameter.grad.any()
        for parameter in net.unsupported_head.parameters()
    )


@pytest.mark.parametrize("coarse", [False, True])
def test_noiseless_protein_operator_never_prefers_spurious_mixing(coarse):
    original = specimen()
    cells = original.cell_ids[:2]
    operator = (
        ProteinObservationOperator(
            cells,
            ("common_region",),
            np.asarray([0, 0]),
            np.asarray([0, 1]),
            np.asarray([0.5, 0.5]),
        )
        if coarse
        else identity_protein_operator(cells)
    )
    measured = np.asarray([[0.5]]) if coarse else np.asarray([[1.0], [0.0]])
    evidence = ProteinEvidence(
        marker_ids=("m1",),
        values=measured,
        observed=np.ones_like(measured, dtype=bool),
        state_values=np.asarray([[1.0], [0.0]]),
        correspondence="serial_section",
        operator=operator,
    )
    inputs = replace(
        original,
        cell_ids=cells,
        coordinates_um=original.coordinates_um[:2],
        image_features=original.image_features[:2],
        graph=None,
        proteins=evidence,
    )
    net = model("he_rna_protein")
    predicted = net(inputs)
    correct = replace(predicted, conditional_assignment=torch.eye(2))
    mixed = replace(predicted, conditional_assignment=torch.full((2, 2), 0.5))
    assert net.protein_loss(inputs, correct).item() == 0
    mixed_loss = net.protein_loss(inputs, mixed).item()
    assert mixed_loss == 0 if coarse else mixed_loss > 0
    logits = torch.tensor([[0.0, 1000.0, -1000.0], [0.0, -1000.0, 1000.0]])
    torch.testing.assert_close(net._protein_logits(logits, inputs), logits, atol=0, rtol=0)


def test_protein_baseline_does_not_depend_on_communication_radius_or_degree_cap():
    inputs = specimen()
    evidence = ProteinEvidence(
        marker_ids=("m1",),
        values=np.asarray([[0.5]]),
        observed=np.ones((1, 1), dtype=bool),
        state_values=np.asarray([[1.0], [0.0]]),
        correspondence="region",
        operator=ProteinObservationOperator(
            inputs.cell_ids,
            ("region",),
            np.asarray([0, 0, 0]),
            np.arange(3),
            np.full(3, 1 / 3),
        ),
    )
    inputs = replace(inputs, proteins=evidence)
    net = model("he_rna_protein")
    expected = net(inputs)
    for radius, degree in [(1, 1), (1000, 1), (1000, 100)]:
        graph = build_graph(
            inputs.coordinates_um,
            radius,
            tissue_component_ids=("a",) * 3,
            max_neighbors=degree,
        )
        changed = replace(inputs, graph=graph)
        actual = net(changed)
        torch.testing.assert_close(actual.expression, expected.expression, atol=0, rtol=0)
        torch.testing.assert_close(actual.assignment, expected.assignment, atol=0, rtol=0)
        torch.testing.assert_close(
            net.protein_loss(changed, actual), net.protein_loss(inputs, expected)
        )


@pytest.mark.parametrize("correspondence", ["same_cell", "serial_section"])
def test_absolute_protein_quality_attenuates_loss_and_assignment_correction(correspondence):
    inputs = specimen()
    evidence = ProteinEvidence(
        marker_ids=tuple(f"m{i}" for i in range(20)),
        values=np.zeros((3, 20)),
        observed=np.ones((3, 20), dtype=bool),
        state_values=np.asarray([[1.0] * 20, [0.0] * 20]),
        correspondence=correspondence,
        operator=identity_protein_operator(inputs.cell_ids),
    )
    net = model("he_rna_protein")
    logits = torch.zeros((3, 3))
    prediction = replace(net(inputs), conditional_assignment=torch.full((3, 2), 0.5))
    adjustments, losses = [], []
    for quality in (1.0, 0.1, 0.01, 0.0):
        changed = replace(inputs, proteins=replace(evidence, quality=np.full((3, 20), quality)))
        adjustments.append(net._protein_logits(logits, changed)[:, 1:].abs().sum().item())
        losses.append(net.protein_loss(changed, prediction).item())
    assert all(left > right for left, right in zip(adjustments, adjustments[1:]))
    np.testing.assert_allclose(losses, np.asarray([1.0, 0.1, 0.01, 0.0]) * losses[0], rtol=1e-6)


@pytest.mark.parametrize("panel", [(), ("A",), ("A", "A"), ("A", "missing")])
def test_normalization_panel_must_be_fixed_and_non_degenerate(panel):
    with pytest.raises(ValueError, match="normalization_gene_ids"):
        ModelConfig(
            feature_dim=2,
            state_dim=2,
            gene_ids=("A", "B"),
            supervised_gene_ids=("A",),
            state_gene_ids=("A", "B"),
            normalization_gene_ids=panel,
        )


def test_he_normalization_panel_must_have_predicted_genes():
    with pytest.raises(ValueError, match="normalization genes must be supervised"):
        replace(model().config, variant="he", supervised_gene_ids=("A",))


@pytest.mark.parametrize(
    "changes",
    [
        {"rows": np.asarray([0.0, 1.0, 2.0])},
        {"columns": np.asarray([0, 1, 3])},
        {"weights": np.asarray([1.0, 1.0, 0.0])},
        {"weights": np.asarray([1.0, 1.0, 0.5])},
        {"rows": np.asarray([0, 1, 1])},
        {"observation_ids": ("c1", "c2", "c2")},
    ],
)
def test_protein_operator_rejects_incomplete_or_invalid_spatial_axes(changes):
    operator = identity_protein_operator(specimen().cell_ids)
    with pytest.raises(ValueError):
        replace(operator, **changes)


def test_protein_operator_maps_subset_rows_and_checks_cell_identity():
    inputs = specimen()
    operator = ProteinObservationOperator(
        inputs.cell_ids,
        ("protein_c3", "protein_c1"),
        np.asarray([0, 1]),
        np.asarray([2, 0]),
        np.ones(2),
    )
    values = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    torch.testing.assert_close(operator.aggregate(values), torch.tensor([[3.0], [1.0]]))
    torch.testing.assert_close(
        operator.transpose(torch.tensor([[5.0], [6.0]])), torch.tensor([[6.0], [0.0], [5.0]])
    )
    operator.aggregate(values).sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[1.0], [0.0], [1.0]]))
    evidence = ProteinEvidence(
        marker_ids=("m1",),
        values=np.asarray([[0.0], [1.0]]),
        observed=np.ones((2, 1), dtype=bool),
        state_values=np.asarray([[1.0], [0.0]]),
        correspondence="same_cell",
        operator=operator,
    )
    net = model("he_rna_protein")
    valid = replace(inputs, proteins=evidence)
    assert torch.isfinite(net.protein_loss(valid, net(valid)))
    with pytest.raises(ValueError, match="ordered H&E cell IDs"):
        replace(
            inputs,
            proteins=replace(evidence, operator=replace(operator, cell_ids=("c3", "c2", "c1"))),
        )
