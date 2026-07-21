from __future__ import annotations

import numpy as np

from heir.evaluation.neural_control_models import (
    REQUIRED_NEURAL_CONTROL_FAMILIES,
    build_multiview_arms,
    build_neural_control_arms,
    deduplicate_named_feature_parts,
)


def test_controls_are_deduplicated_and_missing_blank_fails_closed() -> None:
    first = np.arange(12, dtype=np.float32).reshape(4, 3)
    second = np.arange(8, dtype=np.float32).reshape(4, 2)
    nuisance, names, receipt = deduplicate_named_feature_parts(
        ((first, ("a", "shared", "b")), (second, ("shared", "c")))
    )
    assert names == ("a", "shared", "b", "c")
    assert nuisance.shape == (4, 4)
    assert receipt["duplicate_names"] == ["shared"]

    image = np.ones((4, 6), dtype=np.float32)
    removed = np.zeros((4, 6), dtype=np.float32)
    arms = build_neural_control_arms(image, removed, nuisance)
    assert tuple(arms) == REQUIRED_NEURAL_CONTROL_FAMILIES
    assert arms["neural_blank_patch"].available is False
    assert arms["neural_blank_patch"].features is None
    assert arms["neural_target_removed"].features is removed


def test_n7_has_a_separate_width_aware_nuisance_branch() -> None:
    rng = np.random.default_rng(4)
    nuisance = rng.normal(size=(5, 3))
    full = rng.normal(size=(5, 7))
    nucleus = rng.normal(size=(5, 7))
    cell = rng.normal(size=(5, 7))
    arms = build_multiview_arms(full, nucleus, cell, nuisance)
    assert arms["N6"].view_dims == (7, 7, 7)
    assert arms["N7"].view_dims == (3, 7, 7, 7)
    assert arms["N7"].features.shape == (5, 24)
    assert not np.shares_memory(arms["N6"].features, arms["N7"].features)
