from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from heir.evaluation import MORPHOLOGY_RIDGE_REPORT_SCHEMA
from heir.utils import sha256_file


def _load_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "benchmark_reference_specificity.py"
    spec = importlib.util.spec_from_file_location("benchmark_reference_specificity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input_v2() -> dict[str, np.ndarray]:
    rows = 2
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray("heir.reference_utility_input.v2"),
        "query_image_state_latent": np.zeros((rows, 2)),
        "query_molecular_target_latent": np.zeros((rows, 2)),
        "query_types": np.asarray(["type_a", "type_a"]),
        "query_donors": np.asarray(["query_donor", "query_donor"]),
        "query_observation_ids": np.asarray(["query_1", "query_2"]),
        "query_section_ids": np.asarray(["query_section", "query_section"]),
        "query_source_sample_ids": np.asarray(["query_sample", "query_sample"]),
        "query_source_material_ids": np.asarray(["query_material", "query_material"]),
        "query_specimen_ids": np.asarray(["query_specimen", "query_specimen"]),
        "query_preservation_methods": np.repeat("FFPE", rows),
        "query_disease_states": np.repeat("healthy", rows),
        "query_site_ids": np.repeat("lung", rows),
        "query_institution_ids": np.repeat("institution", rows),
        "query_assay_ids": np.repeat("Xenium", rows),
        "query_quality_bins": np.repeat("high", rows),
        "query_depth_bins": np.repeat("middle", rows),
        "query_latent_model_sha256": np.asarray("a" * 64),
        "query_normalization_sha256": np.asarray("b" * 64),
        "query_assay_harmonization_sha256": np.asarray("c" * 64),
        "query_assay_harmonization_fit_donor_ids": np.asarray(["development_donor"]),
        "query_assay_harmonization_source_sha256": np.asarray("2" * 64),
        "query_calibrated_assay_pairs": np.asarray(["same_assay"]),
        "eligible_hard_wrong_donor_ids": np.asarray(["wrong_donor"]),
        "total_eligible_query_count": np.asarray(rows),
        "excluded_query_reason_names": np.asarray([], dtype="U1"),
        "excluded_query_reason_counts": np.asarray([], dtype=np.int64),
        "eligible_query_donor_ids": np.asarray(["query_donor"]),
        "eligible_query_type_labels": np.asarray(["type_a"]),
        "power_analysis_sha256": np.asarray("0" * 64),
        "power_justified_minimum_donors": np.asarray(3),
        "frozen_image_model_sha256": np.asarray("e" * 64),
        "query_source_sha256": np.asarray("f" * 64),
        "bank_names": np.asarray(["matched"]),
        "bank_0_role": np.asarray("matched"),
        "bank_0_latent": np.zeros((rows, 2)),
        "bank_0_type_labels": np.asarray(["type_a", "type_a"]),
        "bank_0_donor_ids": np.asarray(["query_donor", "query_donor"]),
        "bank_0_observation_ids": np.asarray(["bank_1", "bank_2"]),
        "bank_0_section_ids": np.asarray(["bank_section", "bank_section"]),
        "bank_0_source_sample_ids": np.asarray(["bank_sample", "bank_sample"]),
        "bank_0_source_material_ids": np.asarray(["bank_material", "bank_material"]),
        "bank_0_specimen_ids": np.asarray(["bank_specimen", "bank_specimen"]),
        "bank_0_preservation_methods": np.repeat("fresh_frozen", rows),
        "bank_0_disease_states": np.repeat("healthy", rows),
        "bank_0_site_ids": np.repeat("lung", rows),
        "bank_0_institution_ids": np.repeat("institution", rows),
        "bank_0_assay_ids": np.repeat("Xenium", rows),
        "bank_0_quality_bins": np.repeat("high", rows),
        "bank_0_depth_bins": np.repeat("middle", rows),
        "bank_0_latent_model_sha256": np.asarray("a" * 64),
        "bank_0_normalization_sha256": np.asarray("b" * 64),
        "bank_0_assay_harmonization_sha256": np.asarray("c" * 64),
        "bank_0_assay_mode": np.asarray("same_assay"),
        "bank_0_latent_fit_donor_ids": np.asarray(["development_donor"]),
        "bank_0_assay_harmonization_fit_donor_ids": np.asarray(["development_donor"]),
        "bank_0_assay_harmonization_source_sha256": np.asarray("2" * 64),
        "bank_0_calibrated_assay_pairs": np.asarray(["same_assay"]),
        "bank_0_material_relationship_to_query": np.asarray("independent_aliquot"),
        "bank_0_independent_tissue_material": np.asarray(True),
        "bank_0_contains_registered_query_cells": np.asarray(False),
        "bank_0_selection_uses_query_truth": np.asarray(False),
        "bank_0_source_sha256": np.asarray("1" * 64),
    }
    return values


def _write_prerequisites(tmp_path: Path) -> tuple[Path, Path]:
    paths = (tmp_path / "primary.json", tmp_path / "external.json")
    roles = ("primary_hest_uni2h", "external_confirmation_independent")
    for path, role in zip(paths, roles):
        path.write_text(
            json.dumps(
                {
                    "schema_version": MORPHOLOGY_RIDGE_REPORT_SCHEMA,
                    "component_pass": True,
                    "oracle_type_only": True,
                    "experiment_role": role,
                    "provenance": {
                        "feature_checkpoint_sha256": "e" * 64,
                        "locked_test_data": {"sha256": "f" * 64},
                    },
                }
            ),
            encoding="utf-8",
        )
    return paths


def _write_power_receipt(tmp_path: Path) -> Path:
    core = {
        "schema": "heir.reference_utility_power.v1",
        "simulation_sha256": "1" * 64,
        "thresholds_sha256": "2" * 64,
        "minimum_relative_effect": 0.1,
        "minimum_query_donors": 3,
        "minimum_comparison_query_donors": 3,
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
    receipt = {
        **core,
        "receipt_content_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    path = tmp_path / "power.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_reference_cli_maps_the_complete_v2_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source = tmp_path / "input.npz"
    output = tmp_path / "report.json"
    power = _write_power_receipt(tmp_path)
    values = _input_v2()
    values["power_analysis_sha256"] = np.asarray(sha256_file(power))
    np.savez(source, **values)
    primary, external = _write_prerequisites(tmp_path)
    captured: dict[str, object] = {}

    def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"schema": "heir.matched_reference_utility.v2", "pass": True}

    monkeypatch.setattr(module, "evaluate_reference_utility", fake_evaluate)
    code = module.main(
        [
            "--input",
            str(source),
            "--prerequisite-report",
            str(primary),
            "--prerequisite-report",
            str(external),
            "--report-output",
            str(output),
            "--power-analysis-receipt",
            str(power),
            "--minimum-relative-effect",
            "0.1",
        ]
    )

    assert code == 0
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["query_latent_model_sha256"] == "a" * 64
    assert kwargs["eligible_hard_wrong_donor_ids"].tolist() == ["wrong_donor"]
    assert kwargs["query_eligibility"] == {
        "total_eligible_query_count": 2,
        "excluded_query_count_by_reason": {},
        "eligible_donor_ids": np.asarray(["query_donor"]),
        "eligible_type_labels": np.asarray(["type_a"]),
        "power_analysis_sha256": sha256_file(power),
        "power_justified_minimum_donors": 3,
    }
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [item["role"] for item in result["morphology_prerequisites"]] == [
        "primary_hest_uni2h",
        "external_confirmation_independent",
    ]


@pytest.mark.parametrize(
    ("names", "counts"),
    [
        (np.asarray(["missing_bank"]), np.asarray([-1], dtype=np.int64)),
        (np.asarray(["missing_bank"]), np.asarray([], dtype=np.int64)),
        (np.asarray(["missing_bank", "missing_bank"]), np.asarray([1, 1], dtype=np.int64)),
    ],
)
def test_reference_cli_rejects_malformed_exclusion_denominators(
    tmp_path: Path,
    names: np.ndarray,
    counts: np.ndarray,
) -> None:
    module = _load_module()
    values = _input_v2()
    values["excluded_query_reason_names"] = names
    values["excluded_query_reason_counts"] = counts
    source = tmp_path / "input.npz"
    power = _write_power_receipt(tmp_path)
    values["power_analysis_sha256"] = np.asarray(sha256_file(power))
    np.savez(source, **values)
    primary, external = _write_prerequisites(tmp_path)

    with pytest.raises(ValueError, match="exclusion denominator is malformed"):
        module.main(
            [
                "--input",
                str(source),
                "--prerequisite-report",
                str(primary),
                "--prerequisite-report",
                str(external),
                "--report-output",
                str(tmp_path / "report.json"),
                "--power-analysis-receipt",
                str(power),
                "--minimum-relative-effect",
                "0.1",
            ]
        )
