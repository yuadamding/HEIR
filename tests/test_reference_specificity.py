from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation import evaluate_reference_specificity


def test_reference_specificity_equalizes_type_banks_and_repeats() -> None:
    report = evaluate_reference_specificity(
        np.asarray([[0.0], [10.0], [0.1], [10.1]]),
        np.asarray([0, 1, 0, 1]),
        ("d1", "d1", "d2", "d2"),
        np.asarray([[0.0], [0.2], [10.0], [10.2], [99.0]]),
        np.asarray([0, 0, 1, 1, 1]),
        {"wrong": (np.asarray([[3.0], [3.2], [13.0], [13.2]]), np.asarray([0, 0, 1, 1]))},
        repeats=100,
    )
    assert report["pass"] is True
    assert report["aggregation"] == "equal_donor_equal_type"
    assert report["per_type_bank_size"] == {"0": 2, "1": 2}


def test_reference_specificity_fails_closed_below_100_repeats() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_reference_specificity(
            np.ones((2, 1)),
            np.asarray([0, 0]),
            ("d1", "d2"),
            np.ones((2, 1)),
            np.asarray([0, 0]),
            {"wrong": (np.zeros((2, 1)), np.asarray([0, 0]))},
            repeats=99,
        )
