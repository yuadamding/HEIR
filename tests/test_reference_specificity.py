from __future__ import annotations

import hashlib
import json

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
        "source_samples": np.asarray([f"query_sample_{index}" for index in range(6)]),
        "source_materials": np.asarray([f"query_material_{index}" for index in range(6)]),
        "specimens": np.asarray([f"query_specimen_{index}" for index in range(6)]),
        "preservation": np.repeat("FFPE", 6),
        "disease": np.repeat("healthy", 6),
        "site": np.repeat("lung", 6),
        "institution": np.repeat("institution", 6),
        "assay": np.repeat("Xenium", 6),
        "quality": np.repeat("high", 6),
        "depth": np.repeat("middle", 6),
    }


def _bank(
    role: str,
    shift: float,
    source: str,
    *,
    donors: tuple[str, ...],
    assay_mode: str = "same_assay",
    assay: str = "Xenium",
    calibrated_pairs: tuple[str, ...] = ("same_assay",),
) -> dict[str, object]:
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
        "observation_ids": np.asarray([f"{role}_{source}_observation_{i}" for i in range(rows)]),
        "section_ids": np.asarray([f"{role}_{source}_section_{i}" for i in range(rows)]),
        "source_sample_ids": np.asarray([f"{role}_{source}_sample_{i}" for i in range(rows)]),
        "source_material_ids": np.asarray([f"{role}_{source}_material_{i}" for i in range(rows)]),
        "specimen_ids": np.asarray([f"{role}_{source}_specimen_{i}" for i in range(rows)]),
        "preservation_methods": np.repeat("fresh_frozen", rows),
        "disease_states": np.repeat("healthy", rows),
        "site_ids": np.repeat("lung", rows),
        "institution_ids": np.repeat("institution", rows),
        "assay_ids": np.repeat(assay, rows),
        "quality_bins": np.repeat("high", rows),
        "depth_bins": np.repeat("middle", rows),
        "latent_model_sha256": "a" * 64,
        "normalization_sha256": "b" * 64,
        "assay_harmonization_sha256": "c" * 64,
        "assay_mode": assay_mode,
        "latent_fit_donor_ids": ("development_1", "development_2"),
        "assay_harmonization_fit_donor_ids": ("development_1", "development_2"),
        "assay_harmonization_source_sha256": "e" * 64,
        "calibrated_assay_pairs": calibrated_pairs,
        "material_relationship_to_query": (
            "independent_aliquot" if role == "matched" else "external_independent_donor_material"
        ),
        "independent_tissue_material": True,
        "contains_registered_query_cells": False,
        "selection_uses_query_truth": False,
        "source_sha256": source * 64,
    }


def _banks(
    *, assay_mode: str = "same_assay", assay: str = "Xenium"
) -> dict[str, dict[str, object]]:
    pairs = ("same_assay",) if assay_mode == "same_assay" else (f"Xenium::{assay}",)
    settings = {"assay_mode": assay_mode, "assay": assay, "calibrated_pairs": pairs}
    return {
        "matched": _bank("matched", 0.0, "1", donors=("d1", "d2", "d3"), **settings),
        "wrong_w1": _bank("hard_wrong", 2.0, "2", donors=("w1",), **settings),
        "wrong_w2": _bank("hard_wrong", 2.5, "3", donors=("w2",), **settings),
        "generic": _bank("generic", 4.0, "4", donors=("g1", "g2", "g3"), **settings),
        "population": _bank(
            "population_leave_query_out", 3.0, "5", donors=("p1", "p2", "p3"), **settings
        ),
    }


def _power_receipt(*, comparison_minimum: int = 3) -> dict[str, object]:
    core = {
        "schema": "heir.reference_utility_power.v1",
        "simulation_sha256": "1" * 64,
        "thresholds_sha256": "2" * 64,
        "minimum_relative_effect": 0.1,
        "minimum_query_donors": 3,
        "minimum_comparison_query_donors": comparison_minimum,
        "maximum_familywise_false_positive_probability": 0.05,
        "power_at_minimum_relative_effect": 0.8,
        "uses_locked_query_outcomes": False,
        "scenario_families": [
            "donor_count",
            "query_population_coverage",
            "sparse_exact_matching",
            "wrong_donor_eligibility",
            "assay_harmonization",
        ],
    }
    return {
        **core,
        "receipt_content_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _evaluate(
    banks: dict[str, dict[str, object]],
    *,
    query_eligibility: dict[str, object] | None = None,
    power_receipt: dict[str, object] | None = None,
    repeats: int = 100,
) -> dict[str, object]:
    query = _queries()
    eligibility = query_eligibility or {
        "total_eligible_query_count": 6,
        "excluded_query_count_by_reason": {},
        "eligible_donor_ids": ["d1", "d2", "d3"],
        "eligible_type_labels": [0, 1],
        "power_analysis_sha256": "f" * 64,
        "power_justified_minimum_donors": 3,
    }
    assay_mode = str(next(iter(banks.values()))["assay_mode"])
    reference_assay = str(np.asarray(next(iter(banks.values()))["assay_ids"])[0])
    calibrated_pairs = (
        ("same_assay",) if assay_mode == "same_assay" else (f"Xenium::{reference_assay}",)
    )
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
            query_source_sample_ids=query["source_samples"],
            query_source_material_ids=query["source_materials"],
            query_specimen_ids=query["specimens"],
            query_preservation_methods=query["preservation"],
            query_institution_ids=query["institution"],
            query_latent_model_sha256="a" * 64,
            query_normalization_sha256="b" * 64,
            query_assay_harmonization_sha256="c" * 64,
            query_assay_harmonization_fit_donor_ids=("development_1", "development_2"),
            query_assay_harmonization_source_sha256="e" * 64,
            query_calibrated_assay_pairs=calibrated_pairs,
            eligible_hard_wrong_donor_ids=("w1", "w2"),
            query_eligibility=eligibility,
            power_analysis_receipt=power_receipt or _power_receipt(),
            power_analysis_receipt_sha256="f" * 64,
            frozen_image_model_sha256="6" * 64,
            query_source_sha256="7" * 64,
            morphology_evidence_binding={
                "primary_feature_checkpoint_sha256": "6" * 64,
                "primary_query_source_sha256": "7" * 64,
                "primary_report_sha256": "8" * 64,
                "external_report_sha256s": ["9" * 64],
            },
            repeats=repeats,
            minimum_relative_effect=0.1,
            bootstrap_samples=1000,
            seed=11,
        )
    )


def test_reference_utility_is_image_conditioned_donor_paired_and_complete() -> None:
    report = _evaluate(_banks())
    assert report["pass"] is True
    assert report["image_conditioned"] is True
    assert report["only_reference_bank_changes"] is True
    assert report["aggregation"] == "equal_donor_equal_type"
    assert report["all_eligible_hard_wrong_donors_tested"] is True
    assert report["query_population_coverage"]["retained_fraction"] == 1.0
    assert report["query_population_coverage"]["power_analysis_sha256"] == "f" * 64
    assert report["power_analysis"]["uses_locked_query_outcomes"] is False
    assert {row["bank_role"] for row in report["comparisons"]} == {
        "hard_wrong",
        "generic",
        "population_leave_query_out",
    }
    assert all(
        row["relative_error_reduction_donor_confidence_interval"][0] > 0.1
        for row in report["comparisons"]
    )


def test_reference_utility_rejects_query_material_overlap() -> None:
    banks = _banks()
    banks["matched"]["source_material_ids"][0] = "query_material_0"
    with pytest.raises(ValueError, match="query tissue material overlaps"):
        _evaluate(banks)


def test_reference_utility_rejects_any_eligible_donor_in_latent_fit() -> None:
    banks = _banks()
    for bank in banks.values():
        bank["latent_fit_donor_ids"] = ("development_1", "d4")
    eligibility = {
        "total_eligible_query_count": 7,
        "excluded_query_count_by_reason": {"missing_reference_stratum": 1},
        "eligible_donor_ids": ["d1", "d2", "d3", "d4"],
        "eligible_type_labels": [0, 1],
        "power_analysis_sha256": "f" * 64,
        "power_justified_minimum_donors": 3,
    }
    with pytest.raises(ValueError, match="cannot fit or select"):
        _evaluate(banks, query_eligibility=eligibility)


def test_reference_utility_rejects_scalar_latent_fit_donors() -> None:
    banks = _banks()
    banks["matched"]["latent_fit_donor_ids"] = "development_1"
    with pytest.raises(ValueError, match="latent-fit donors are malformed"):
        _evaluate(banks)


def test_reference_utility_requires_every_wrong_and_population_bank() -> None:
    banks = _banks()
    del banks["wrong_w2"]
    with pytest.raises(ValueError, match="every eligible wrong donor"):
        _evaluate(banks)

    banks = _banks()
    del banks["population"]
    with pytest.raises(ValueError, match="leave-query-donor-out population"):
        _evaluate(banks)


def test_reference_utility_cross_assay_requires_explicit_calibration_mode() -> None:
    report = _evaluate(_banks(assay_mode="cross_assay_development_calibrated", assay="snRNA"))
    assert report["pass"] is True
    assert report["assay_mode"] == "cross_assay_development_calibrated"
    assert report["calibrated_assay_pairs"] == ["Xenium::snRNA"]

    banks = _banks(assay_mode="cross_assay_development_calibrated", assay="snRNA")
    for bank in banks.values():
        bank["calibrated_assay_pairs"] = ("Xenium::bulk_RNA",)
    with pytest.raises(ValueError, match="calibrated assay-pair contract"):
        _evaluate(banks)


def test_reference_utility_reports_and_gates_query_population_coverage() -> None:
    eligibility = {
        "total_eligible_query_count": 12,
        "excluded_query_count_by_reason": {"missing_reference_stratum": 6},
        "eligible_donor_ids": ["d1", "d2", "d3"],
        "eligible_type_labels": [0, 1],
        "power_analysis_sha256": "f" * 64,
        "power_justified_minimum_donors": 3,
    }
    report = _evaluate(_banks(), query_eligibility=eligibility)
    assert report["pass"] is False
    assert report["query_population_coverage"]["retained_fraction"] == 0.5
    assert report["query_population_coverage"]["pass"] is False


def test_reference_utility_enforces_power_minimum_per_comparison() -> None:
    with pytest.raises(ValueError, match="power-justified donor minimum"):
        _evaluate(_banks(), power_receipt=_power_receipt(comparison_minimum=4))


def test_reference_utility_requires_repeated_sampling() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        _evaluate(_banks(), repeats=99)
