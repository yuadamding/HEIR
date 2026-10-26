"""Reference-analysis orchestration must preserve source-only fits and original studies."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_personalized_model import model, specimen

from heir.personalized.capability import reconstruct_patient
from heir.personalized.cli import main
from heir.personalized.observation import ObservationOperator
from heir.personalized.schema import SpatialTargets


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
