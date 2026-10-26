"""Reference-analysis orchestration must preserve source-only fits and original studies."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_personalized_model import model, specimen

from heir.personalized.capability import _linear_score, reconstruct_patient
from heir.personalized.cli import main
from heir.personalized.observation import ObservationOperator
from heir.personalized.schema import SpatialTargets


def _linear_fixture():
    inputs = specimen()
    prediction = model()(inputs)
    expression = np.array([[2, 8, 3], [5, 3, 6], [9, 2, 1]], dtype=np.float32)
    ids = tuple(f"assay:{i}" for i in range(3))
    target = SpatialTargets(
        observation_ids=ids,
        gene_ids=prediction.gene_ids,
        expression=expression,
        observed=np.ones(expression.shape, dtype=bool),
        patient_id=prediction.patient_id,
        specimen_id=prediction.specimen_id,
    )
    operator = ObservationOperator(inputs.cell_ids, ids, np.arange(3), np.arange(3), np.ones(3))
    return prediction, expression, target, operator


def test_linear_score_joins_complete_scaffold_permutation_by_region_id():
    prediction, expression, target, operator = _linear_fixture()
    genes = prediction.gene_ids
    expected = _linear_score(prediction, expression, target, operator, genes, genes)
    order = [2, 0, 1]
    region_fields = (
        "coordinates_um",
        "assignment",
        "conditional_assignment",
        "transfer",
        "log_correction",
        "expression",
        "assignment_entropy",
        "reference_support",
        "registration_sensitivity",
    )
    reordered = replace(
        prediction,
        cell_ids=tuple(prediction.cell_ids[i] for i in order),
        **{name: getattr(prediction, name)[order] for name in region_fields},
    )
    actual = _linear_score(reordered, expression, target, operator, genes, genes)
    assert actual == expected
    assert actual["normalized_log_mse"] == 0
    assert actual["median_gene_spearman"] == pytest.approx(1)
    # Also permute sparse triplets and the operator's full region axis independently.
    remapped = replace(
        operator,
        cell_ids=tuple(operator.cell_ids[i] for i in order),
        rows=np.array([2, 0, 1]),
        columns=np.array([0, 1, 2]),
    )
    assert _linear_score(reordered, expression, target, remapped, genes, genes) == expected


def test_linear_score_rejects_duplicate_and_unknown_region_identities():
    prediction, expression, target, operator = _linear_fixture()
    genes = prediction.gene_ids
    duplicated_scaffold = replace(prediction, cell_ids=("c0", "c0", "c2"))
    with pytest.raises(ValueError, match="duplicate"):
        _linear_score(duplicated_scaffold, expression, target, operator, genes, genes)
    duplicated_observations = replace(operator, columns=np.array([0, 0, 2]))
    with pytest.raises(ValueError, match="unique regions"):
        _linear_score(prediction, expression, target, duplicated_observations, genes, genes)
    unknown = replace(operator, cell_ids=operator.cell_ids[:2] + ("unknown-region",))
    with pytest.raises(ValueError, match="unknown scaffold region IDs"):
        _linear_score(prediction, expression, target, unknown, genes, genes)


def test_reconstruction_report_keeps_fit_and_cross_replicate_scopes_separate(tmp_path):
    inputs = specimen()
    prediction = model()(inputs)
    q = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    expression = q @ inputs.reference.expression
    ids = tuple(f"rep:{rep}:c{i}" for rep in ("1", "2") for i in range(3))
    targets = SpatialTargets(
        observation_ids=ids,
        gene_ids=prediction.gene_ids,
        expression=np.concatenate([expression, expression * 7]),
        observed=np.ones((6, 3), dtype=bool),
        patient_id=inputs.patient_id,
        specimen_id=inputs.specimen_id,
    )
    operator = ObservationOperator(
        inputs.cell_ids, ids, np.arange(6), np.tile(np.arange(3), 2), np.ones(6)
    )
    groups = ("1",) * 3 + ("2",) * 3
    protocol = {
        "normalization_gene_ids": list(prediction.gene_ids),
        "reconstruction": {"steps": 30, "learning_rate": 0.05, "seed": 0},
        "cross_replicate": {"minimum_shared_regions": 3},
    }
    first = tmp_path / "first"
    first.mkdir()
    report = reconstruct_patient(
        inputs,
        {"he_rna": prediction},
        targets,
        operator,
        groups,
        {"matched": inputs.reference, "generic": inputs.reference},
        protocol,
        first,
    )
    assert report["target_informed"] and not report["he_mapper"] and not report["correction"]
    assert len(report["within_replicate"]) == 4
    assert len(report["cross_replicate"]) == 4
    assert len(report["direct_replicate_transfer"]) == 2
    assert all(
        r["scores"]["normalized_log_mse"] < 1e-10 for r in report["direct_replicate_transfer"]
    )
    assert all(
        r["scores"]["median_gene_spearman"] == pytest.approx(1)
        for r in report["direct_replicate_transfer"]
    )
    second = tmp_path / "changed-evaluation"
    second.mkdir()
    changed = targets.expression.copy()
    changed[3:] = changed[3:][::-1]
    altered = reconstruct_patient(
        inputs,
        {"he_rna": prediction},
        replace(targets, expression=changed),
        operator,
        groups,
        {"matched": inputs.reference, "generic": inputs.reference},
        protocol,
        second,
    )
    with np.load(first / "matched-replicate-1.npz", allow_pickle=False) as before:
        with np.load(second / "matched-replicate-1.npz", allow_pickle=False) as after:
            np.testing.assert_array_equal(before["q"], after["q"])
    a = next(
        r
        for r in report["cross_replicate"]
        if r["reference"] == "matched" and r["source_replicate"] == "1"
    )
    b = next(
        r
        for r in altered["cross_replicate"]
        if r["reference"] == "matched" and r["source_replicate"] == "1"
    )
    assert a["scores"]["normalized_log_mse"] < b["scores"]["normalized_log_mse"]


def test_capability_cli_rejects_unbound_study_before_output_creation(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema": "unbound"}))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="unsupported reference-capability"):
        main(
            [
                "cohort-capability",
                "--root",
                str(tmp_path / "data"),
                "--study",
                str(tmp_path / "study"),
                "--protocol",
                str(protocol),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()
