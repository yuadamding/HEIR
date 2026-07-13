from __future__ import annotations

import numpy as np
import pytest

from heir.evaluation import evaluate_reference_utility


def _queries() -> dict[str, np.ndarray]:
    donors = np.repeat(np.asarray(["d1", "d2", "d3"]), 2)
    types = np.tile(np.asarray([0, 1]), 3)
    target = np.column_stack((types * 10.0, np.repeat([0.0, 0.4, 0.8], 2)))
    return {
        "image": target.copy(),
        "target": target,
        "types": types,
        "donors": donors,
        "observations": np.asarray([f"query_{index}" for index in range(6)]),
        "sections": np.asarray([f"query_section_{index}" for index in range(6)]),
        "disease": np.repeat("healthy", 6),
        "site": np.repeat("site", 6),
        "assay": np.repeat("Xenium", 6),
        "quality": np.repeat("high", 6),
        "depth": np.repeat("middle", 6),
    }


def _bank(role: str, shift: float, source: str, *, donors: tuple[str, ...]) -> dict[str, object]:
    latent = []
    bank_donors = []
    types = []
    for donor_index, donor in enumerate(donors):
        for type_index in (0, 1):
            for offset in (-0.05, 0.05):
                latent.append([type_index * 10.0 + shift + offset, donor_index * 0.4 + shift])
                bank_donors.append(donor)
                types.append(type_index)
    rows = len(latent)
    return {
        "role": role,
        "latent": np.asarray(latent),
        "type_labels": np.asarray(types),
        "donor_ids": np.asarray(bank_donors),
        "observation_ids": np.asarray([f"{role}_{source}_{index}" for index in range(rows)]),
        "section_ids": np.asarray([f"{role}_{source}_section_{index}" for index in range(rows)]),
        "disease_states": np.repeat("healthy", rows),
        "site_ids": np.repeat("site", rows),
        "assay_ids": np.repeat("Xenium", rows),
        "quality_bins": np.repeat("high", rows),
        "depth_bins": np.repeat("middle", rows),
        "latent_model_sha256": "a" * 64,
        "source_sha256": source * 64,
    }


def _evaluate(banks: dict[str, dict[str, object]]) -> dict[str, object]:
    query = _queries()
    return dict(
        evaluate_reference_utility(
            query["image"],
            query["target"],
            query["types"],
            query["donors"],
            query["observations"],
            query["sections"],
            query["disease"],
            query["site"],
            query["assay"],
            query["quality"],
            query["depth"],
            banks,
            repeats=100,
            minimum_effect=0.1,
            bootstrap_samples=1000,
            seed=11,
        )
    )


def test_reference_utility_is_image_conditioned_and_donor_paired() -> None:
    report = _evaluate(
        {
            "matched": _bank("matched", 0.0, "1", donors=("d1", "d2", "d3")),
            "hard_wrong": _bank("hard_wrong", 2.0, "2", donors=("d1", "d2", "d3")),
            "generic": _bank("generic", 4.0, "3", donors=("g1", "g2", "g3")),
        }
    )
    assert report["pass"] is True
    assert report["image_conditioned"] is True
    assert report["only_reference_bank_changes"] is True
    assert report["aggregation"] == "equal_donor_equal_type"
    assert {row["bank_role"] for row in report["comparisons"]} == {
        "hard_wrong",
        "generic",
    }
    assert all(row["donor_confidence_interval"][0] > 0.1 for row in report["comparisons"])


def test_reference_utility_rejects_query_or_section_overlap() -> None:
    banks = {
        "matched": _bank("matched", 0.0, "1", donors=("d1", "d2", "d3")),
        "hard_wrong": _bank("hard_wrong", 2.0, "2", donors=("d1", "d2", "d3")),
        "generic": _bank("generic", 4.0, "3", donors=("g1", "g2", "g3")),
    }
    banks["matched"]["observation_ids"][0] = "query_0"
    with pytest.raises(ValueError, match="query observations overlap"):
        _evaluate(banks)


def test_reference_utility_requires_complete_bank_roles_and_repeated_sampling() -> None:
    banks = {
        "matched": _bank("matched", 0.0, "1", donors=("d1", "d2", "d3")),
        "wrong": _bank("hard_wrong", 2.0, "2", donors=("d1", "d2", "d3")),
    }
    with pytest.raises(ValueError, match="hard-wrong and a generic"):
        _evaluate(banks)

    query = _queries()
    banks["generic"] = _bank("generic", 4.0, "3", donors=("g1", "g2", "g3"))
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_reference_utility(
            query["image"],
            query["target"],
            query["types"],
            query["donors"],
            query["observations"],
            query["sections"],
            query["disease"],
            query["site"],
            query["assay"],
            query["quality"],
            query["depth"],
            banks,
            repeats=99,
            minimum_effect=0.1,
        )
