from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.graph import LRInteraction, build_graph, compute_messages
from heir.personalized.model import ModelConfig, PersonalizedModel
from heir.personalized.schema import PredictionInputs, ProteinEvidence, RNAReference, SpatialGraph


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
        ModelConfig(feature_dim=2, state_dim=2, gene_ids=("A", "B"), supervised_gene_ids=("A",))


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
    )
    net = model("he_rna_protein")
    before = net(replace(inputs, proteins=evidence))
    changed = evidence.values.copy()
    changed[:, 1] = [-1e9, 1e30, np.nan]
    after = net(replace(inputs, proteins=replace(evidence, values=changed)))
    torch.testing.assert_close(after.expression, before.expression)
    assert torch.isfinite(net.protein_loss(replace(inputs, proteins=evidence), before))
    with pytest.raises(ValueError, match="requires a spatial graph"):
        net(replace(inputs, proteins=evidence, graph=None))


def test_reference_deletion_increases_explicit_unsupported_mass():
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
