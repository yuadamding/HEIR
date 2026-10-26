"""Broader outcomes are selected from named training patients before held-out access."""

from dataclasses import replace

import numpy as np
import pytest
from test_personalized_training import _sample

from heir.personalized.reference import select_training_outcomes


def test_panel_retains_historical_genes_and_requires_training_detection():
    ref = _sample().inputs.reference
    references = {"P1": ref, "P2": replace(ref, variance=np.full((2, 3), 5.0))}
    detection = {patient: (("C", "B", "A"), [1, 0, 0]) for patient in references}
    panel, record = select_training_outcomes(
        references, detection, ("P2", "P1"), required_gene_ids=("A",), max_genes=2
    )
    assert panel == ("A", "C")
    assert record["normalization_gene_ids"] == list(panel)
    assert record["training_patient_ids"] == ["P1", "P2"]
    assert not record["selection_uses_validation_or_test"]
    assert (
        next(row for row in record["candidate_scores"] if row["gene_id"] == "B")["eligible"]
        is False
    )
    with pytest.raises(ValueError, match="exactly the declared training"):
        select_training_outcomes(
            {**references, "held-out": ref}, detection, ("P1", "P2"), required_gene_ids=("A",)
        )


def test_panel_uses_donor_weights_and_named_axes_without_changing_references():
    ref = _sample().inputs.reference
    high_c = replace(ref, variance=np.array([[0, 0, 10000], [0, 0, 10000]]))
    references = {"P1": ref, "P2": high_c}
    detection = {patient: (ref.gene_ids, np.ones(3)) for patient in references}
    original = ref.proportions.copy()
    panel, record = select_training_outcomes(
        references, detection, ("P1", "P2"), required_gene_ids=("A",), max_genes=2
    )
    assert panel == ("A", "C")
    permuted = {patient: (tuple(reversed(ref.gene_ids)), np.ones(3)) for patient in references}
    reordered, record2 = select_training_outcomes(
        dict(reversed(list(references.items()))),
        permuted,
        ("P2", "P1"),
        required_gene_ids=("A",),
        max_genes=2,
    )
    assert reordered == panel
    assert record2["candidate_scores"] == record["candidate_scores"]
    np.testing.assert_array_equal(ref.proportions, original)
