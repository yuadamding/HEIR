from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from heir.evaluation.gene_panel import (
    CSRMatrix,
    PanelMomentBundle,
    normalized_group_moments,
    panel_artifact,
    project_csr_columns,
    select_gene_panel,
    validate_development_protocol,
    validate_panel_artifact,
)


def _csr(dense: np.ndarray) -> CSRMatrix:
    values = np.asarray(dense, dtype=np.int32)
    data = []
    indices = []
    indptr = [0]
    for row in values:
        columns = np.flatnonzero(row)
        indices.extend(columns.tolist())
        data.extend(row[columns].tolist())
        indptr.append(len(data))
    return CSRMatrix(
        np.asarray(data, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(indptr, dtype=np.int64),
        values.shape,
    ).validate()


def _bundle(*, heldout_multiplier: int = 1) -> PanelMomentBundle:
    genes = ("GENE0", "GENE1", "GENE2", "GENE3", "GENE4", "MT-X", "RPL1", "HBA1")
    donors = np.repeat(np.asarray(["D1", "D2", "D3"]), 4)
    first = np.asarray(
        [
            [8, 1, 1, 1, 1, 2, 2, 2],
            [1, 8, 1, 1, 1, 2, 2, 2],
            [7, 2, 1, 1, 1, 2, 2, 2],
            [2, 7, 1, 1, 1, 2, 2, 2],
        ]
        * 3,
        dtype=np.int32,
    )
    second = np.asarray(
        [
            [7, 1, 2, 1, 1, 1, 1, 1],
            [1, 7, 2, 1, 1, 1, 1, 1],
            [6, 2, 2, 1, 1, 1, 1, 1],
            [2, 6, 2, 1, 1, 1, 1, 1],
        ]
        * 3,
        dtype=np.int32,
    )
    first[donors == "D3"] *= heldout_multiplier
    second[donors == "D3"] *= heldout_multiplier
    full = first + second
    libraries = full.sum(axis=1).astype(np.float64)

    sc_donors = np.repeat(np.asarray(["D1", "D2", "D3"]), 6)
    sc_types = np.tile(np.repeat(np.asarray(["A", "B"]), 3), 3)
    sc = np.tile(
        np.asarray(
            [
                [8, 2, 1, 1, 1, 2, 2, 2],
                [7, 3, 1, 1, 1, 2, 2, 2],
                [9, 1, 1, 1, 1, 2, 2, 2],
                [2, 8, 1, 2, 1, 2, 2, 2],
                [3, 7, 1, 2, 1, 2, 2, 2],
                [1, 9, 1, 2, 1, 2, 2, 2],
            ],
            dtype=np.int32,
        ),
        (3, 1),
    )
    sc[sc_donors == "D3"] *= heldout_multiplier
    sc_groups = np.char.add(np.char.add(sc_donors, "|"), sc_types)
    return PanelMomentBundle(
        genes,
        normalized_group_moments(_csr(full), libraries, donors),
        normalized_group_moments(_csr(first), libraries, donors),
        normalized_group_moments(_csr(second), libraries, donors),
        normalized_group_moments(_csr(sc), sc.sum(axis=1), sc_groups),
    ).validate()


def test_csr_projection_preserves_requested_order_and_bounds_memory() -> None:
    matrix = _csr(np.asarray([[1, 0, 3], [0, 2, 4], [5, 0, 0]], dtype=np.int32))
    observed = project_csr_columns(
        matrix,
        [2, 0],
        row_mask=np.asarray([True, False, True]),
        max_output_bytes=16,
    )
    np.testing.assert_array_equal(observed, np.asarray([[3, 1], [0, 5]], dtype=np.int32))
    with pytest.raises(MemoryError, match="above max_output_bytes"):
        project_csr_columns(matrix, [2, 0], max_output_bytes=8)


def test_panel_selection_is_deterministic_and_preserves_eligible_programs() -> None:
    bundle = _bundle()
    arguments = {
        "training_donor_ids": ["D3", "D1", "D2"],
        "program_genes": ["GENE4", "HBA1"],
        "panel_size": 4,
    }
    first = select_gene_panel(bundle, **arguments)
    second = select_gene_panel(bundle, **arguments)
    assert first == second
    assert len(first.gene_ids) == 4
    assert "GENE4" in first.gene_ids
    assert "HBA1" not in first.gene_ids
    assert all(not gene.startswith(("MT-", "RPL", "RPS", "HBA", "HBB")) for gene in first.gene_ids)


def test_fold_local_selection_cannot_see_heldout_donor_counts() -> None:
    baseline = select_gene_panel(
        _bundle(heldout_multiplier=1),
        training_donor_ids=["D1", "D2"],
        held_out_donor_id="D3",
        panel_size=3,
        program_genes=["GENE4"],
    )
    perturbed = select_gene_panel(
        _bundle(heldout_multiplier=1000),
        training_donor_ids=["D1", "D2"],
        held_out_donor_id="D3",
        panel_size=3,
        program_genes=["GENE4"],
    )
    assert baseline.gene_ids == perturbed.gene_ids
    assert baseline.broad_column_indices == perturbed.broad_column_indices
    np.testing.assert_allclose(baseline.scores, perturbed.scores, rtol=0, atol=0)


def test_panel_receipt_is_hash_bound_and_training_only() -> None:
    selection = select_gene_panel(
        _bundle(),
        training_donor_ids=["D1", "D2"],
        held_out_donor_id="D3",
        panel_size=3,
    )
    artifact = panel_artifact(
        selection,
        source_sha256="a" * 64,
        source_path="/frozen/source.npz",
        mode="lodo_fold_local",
        program_gene_source="synthetic_test",
    )
    validate_panel_artifact(artifact, expected_size=3)
    assert artifact["gene_ids"] == artifact["selection"]["gene_ids"]
    assert "development" in artifact["scope"]
    tampered = copy.deepcopy(artifact)
    tampered["selection"]["gene_ids"][0] = "CHANGED"
    with pytest.raises(ValueError, match="panel identity|identity hash"):
        validate_panel_artifact(tampered, expected_size=3)


def test_committed_development_protocol_preserves_scientific_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs/natcommun_generative_development_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    validate_development_protocol(protocol)
    assert protocol["encoders"]["UNI2_h"] == "forbidden_not_run"
    assert protocol["claim_boundaries"]["cell_level_claims"] == "prohibited"
    assert protocol["zero_depth_policy"]["full_library_zero_spots"].startswith("exclude")
