from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation.neural_nulls import (
    COMPLETE_REFIT_STEPS,
    apply_joint_image_mapping,
    build_neural_null_design,
    different_spatial_block_reassignment,
    run_refitted_neural_null,
    within_section_type_derangement,
)


def _identities() -> tuple[np.ndarray, ...]:
    ids = np.asarray([f"r{index}" for index in range(8)])
    donors = np.repeat("D", 8)
    sections = np.repeat("S", 8)
    types = np.repeat("T", 8)
    blocks = np.repeat(["a", "b"], 4)
    return ids, donors, sections, types, blocks


def test_nulls_preserve_strata_cross_blocks_and_hash_by_observation_id() -> None:
    ids, donors, sections, types, blocks = _identities()
    local, local_receipt = within_section_type_derangement(donors, sections, types, ids, seed=17)
    block, block_receipt = different_spatial_block_reassignment(
        donors, sections, types, blocks, ids, seed=17
    )
    assert np.all(local != np.arange(len(local)))
    assert np.all(blocks != blocks[block])
    assert np.array_equal(donors, donors[block])

    order = np.asarray([7, 2, 5, 0, 3, 6, 1, 4])
    _, reordered_receipt = within_section_type_derangement(
        donors[order], sections[order], types[order], ids[order], seed=17
    )
    assert reordered_receipt["mapping_sha256"] == local_receipt["mapping_sha256"]
    assert block_receipt["cross_block_fraction"] == pytest.approx(1.0)


def test_joint_views_move_together_and_every_null_is_refitted() -> None:
    ids, donors, sections, types, blocks = _identities()
    images = np.arange(8 * 3 * 2).reshape(8, 3, 2)
    design = build_neural_null_design(
        "within_section_type_derangement",
        3,
        donors,
        sections,
        types,
        blocks,
        ids,
        seed=9,
    )
    mapped = apply_joint_image_mapping(images, design.mappings[0])
    assert np.array_equal(mapped, images[design.mappings[0]])
    calls = []

    def refit(values, mapping, receipt):
        calls.append((values.copy(), mapping.copy(), receipt["mapping_sha256"]))
        return {
            "mapping_sha256": receipt["mapping_sha256"],
            "completed_steps": list(COMPLETE_REFIT_STEPS),
            "score": float(len(calls) - 1),
        }

    result = run_refitted_neural_null(2.0, images, design, refit)
    assert len(calls) == 3
    assert result["model_refit_for_every_permutation"] is True
    assert result["empirical_p"] == pytest.approx(0.5)
    assert len(result["refit_receipts_sha256"]) == 64


def test_null_callback_must_attest_the_complete_refit() -> None:
    ids, donors, sections, types, blocks = _identities()
    images = np.arange(8 * 2).reshape(8, 2)
    design = build_neural_null_design(
        "within_section_type_derangement",
        1,
        donors,
        sections,
        types,
        blocks,
        ids,
        seed=5,
    )

    def incomplete(values, mapping, receipt):
        del values, mapping
        return {
            "mapping_sha256": receipt["mapping_sha256"],
            "completed_steps": ["training", "scoring"],
            "score": 0.0,
        }

    with pytest.raises(RuntimeError, match="every registered fit step"):
        run_refitted_neural_null(0.0, images, design, incomplete)
