from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _script():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_hest_retrospective.py"
    spec = importlib.util.spec_from_file_location("benchmark_hest_retrospective", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _script()


def _source(
    path: Path,
    donor_count: int = 15,
    *,
    complete_sections: bool = True,
) -> None:
    rng = np.random.default_rng(4)
    donor_values = []
    section_values = []
    role_values = []
    block_values = []
    section_cursor = 0
    for donor_index, donor in enumerate(RUNNER.EXPECTED_DONORS[:donor_count]):
        donor_sections = 2 if donor_index < 5 else 1
        for _ in range(donor_sections):
            section = RUNNER.EXPECTED_SECTIONS[section_cursor]
            section_cursor += 1
            donor_values.extend([donor] * 12)
            section_values.extend([section] * 12)
            role_values.extend(["reference"] * 6 + ["evaluation"] * 6)
            block_values.extend(
                [f"{section}/b0"] * 3
                + [f"{section}/b1"] * 3
                + [f"{section}/b0"] * 3
                + [f"{section}/b1"] * 3
            )
    donors = np.asarray(donor_values)
    sections = np.asarray(section_values)
    if not complete_sections and len(sections):
        sections[sections == RUNNER.EXPECTED_SECTIONS[-1]] = RUNNER.EXPECTED_SECTIONS[0]
    roles = np.asarray(role_values)
    blocks = np.asarray(block_values)
    rows = len(donors)
    signal = rng.normal(size=(rows, 2))
    targets = signal @ np.asarray([[0.7, -0.2], [0.1, 0.5]]) + rng.normal(
        scale=0.05, size=(rows, 2)
    )
    image_base = signal + rng.normal(scale=0.02, size=signal.shape)
    images = np.stack([image_base.copy() for _ in range(4)], axis=1)
    common = rng.normal(size=(rows, 1))
    np.savez_compressed(
        path,
        study_stage=np.asarray("retrospective_exposed"),
        schema_version=np.asarray(RUNNER.SOURCE_SCHEMA),
        source_scope=np.asarray(RUNNER.SOURCE_SCOPE),
        analysis_status=np.asarray("retrospective_exposed_non_authorizing"),
        authorizes_h_cell=np.asarray(False),
        authorizes_h_intrinsic=np.asarray(False),
        authorizes_full_heir=np.asarray(False),
        prior_outcome_exposure_receipt_sha256=np.asarray(RUNNER.EXPOSURE_RECEIPT_SHA256),
        cohort_id=np.asarray("HEST"),
        cohort_release=np.asarray(RUNNER.DATASET_REVISION),
        encoder_name=np.asarray(RUNNER.ENCODER_NAME),
        encoder_revision=np.asarray(RUNNER.ENCODER_REVISION),
        encoder_manifest_sha256=np.asarray(RUNNER.ENCODER_MANIFEST_SHA256),
        feature_config_sha256=np.asarray(RUNNER.ENCODER_CONFIG_SHA256),
        feature_checkpoint_sha256=np.asarray(RUNNER.ENCODER_CHECKPOINT_SHA256),
        donor_ids=donors,
        section_ids=sections,
        fine_type_ids=np.repeat("T", rows),
        broad_type_names=np.asarray(["L"]),
        broad_type_labels=np.zeros(rows, dtype=int),
        pool_roles=roles,
        block_ids=blocks,
        registration_quality_strata=np.repeat("best", rows),
        locked_measurement_qc_pass=np.ones(rows, dtype=bool),
        nucleus_molecular_targets=targets,
        crop_ids=np.asarray(RUNNER.REQUIRED_CROPS),
        image_features_by_crop_and_encoder=images,
        coordinate_features=np.column_stack((common, common**2)),
        coordinate_feature_names=np.asarray(["he_x_normalized", "he_x_squared"]),
        technical_covariates=common,
        technical_covariate_names=np.asarray(["log1p_library_size"]),
        stain_quality_features=common,
        stain_quality_feature_names=np.asarray(["local_rgb_mean_r"]),
        nucleus_geometry_features=common,
        nucleus_geometry_feature_names=np.asarray(["nucleus_area_um2"]),
        cell_geometry_features=common,
        cell_geometry_feature_names=np.asarray(["cell_area_um2"]),
        local_density_features=common,
        local_density_feature_names=np.asarray(["nearest_neighbor_distance_um"]),
    )


def test_retrospective_report_is_permanently_non_authorizing(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "report.json"
    _source(source)
    RUNNER.benchmark(source, output, alpha=1.0, permutations=2, seed=9)
    report = json.loads(output.read_text())
    assert report["schema"] == RUNNER.SCHEMA
    assert report["analysis_status"] == "retrospective_exposed_non_authorizing"
    assert report["authorizes_h_cell"] is False
    assert report["authorizes_h_intrinsic"] is False
    assert report["authorizes_full_heir"] is False
    assert report["donor_count"] == 15
    assert report["section_count"] == 20
    assert report["permutation_scope"] == "smoke_only"
    assert set(report["results"]) == {"broad_lineage", "fine_type"}
    fine = report["results"]["fine_type"]
    assert set(fine["arms"]) == set(RUNNER.REQUIRED_CROPS)
    assert set(report["hypotheses"]) == {
        "H-CELL-retrospective",
        "H-INTRINSIC-cell-retrospective",
        "H-INTRINSIC-nucleus-retrospective",
    }
    assert fine["control_feature_registry"]["deduplicated"] is True
    assert set(fine["controls"]) == set(RUNNER.CONTROL_ORDER)
    hashes = {
        arm["nulls"]["within_section_type_derangement"]["mapping_set_sha256"]
        for arm in fine["arms"].values()
    }
    assert len(hashes) == 1
    for contrast in fine["crop_contrasts"].values():
        assert contrast["donor_type_r2"]["mean"] == pytest.approx(0.0)


def test_retrospective_report_rejects_incomplete_donor_cohort(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    _source(source, donor_count=14)
    with pytest.raises(ValueError, match="all 15 biological donors"):
        RUNNER.benchmark(source, tmp_path / "report.json", alpha=1.0, permutations=1, seed=1)


def test_retrospective_report_rejects_incomplete_section_cohort(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    _source(source, complete_sections=False)
    with pytest.raises(ValueError, match="exact 20 HEST sections"):
        RUNNER.benchmark(source, tmp_path / "report.json", alpha=1.0, permutations=1, seed=1)


def test_hierarchical_scorer_separates_centered_r2_from_reference_reduction() -> None:
    truth = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    donors = np.repeat("D", 4)
    sections = np.repeat("S", 4)
    labels = np.repeat("T", 4)
    scorer = RUNNER._HierarchicalScorer(truth, donors, sections, labels, 2)
    perfect = scorer.score(truth, detailed=True)
    reference = scorer.score(np.zeros_like(truth), detailed=True)
    assert perfect["donor_type_equal_r2"] == pytest.approx(1.0)
    assert perfect["donor_type_equal_reference_error_reduction"] == pytest.approx(1.0)
    assert reference["donor_type_equal_reference_error_reduction"] == pytest.approx(0.0)
    assert reference["donor_type_equal_r2"] < 0.0


def test_positive_fraction_uses_model_minus_control() -> None:
    model = {
        "per_donor": {
            "D1": {"donor_type_r2": 0.2},
            "D2": {"donor_type_r2": 0.3},
        }
    }
    control = {
        "per_donor": {
            "D1": {"donor_type_r2": 0.1},
            "D2": {"donor_type_r2": 0.4},
        }
    }
    values = RUNNER._effect_values(model, control, "donor_type_r2")
    summary = RUNNER._summarize_effects(values, "D1")
    assert summary["per_donor"] == pytest.approx({"D1": 0.1, "D2": -0.1})
    assert summary["positive_donor_fraction"] == pytest.approx(0.5)


def test_named_control_merge_deduplicates_features() -> None:
    first = np.arange(6).reshape(3, 2)
    second = np.arange(6, 12).reshape(3, 2)
    merged, names = RUNNER._merge_named_parts(
        ((first, ("a", "shared")), (second, ("shared", "b")))
    )
    assert names == ("a", "shared", "b")
    assert merged.shape == (3, 3)
    assert np.array_equal(merged[:, 1], first[:, 1])


def test_null_maps_preserve_strata_and_block_null_changes_blocks() -> None:
    donors = np.repeat("D", 6)
    sections = np.repeat("S", 6)
    labels = np.repeat("T", 6)
    roles = np.repeat("evaluation", 6)
    blocks = np.asarray(["a", "a", "a", "b", "b", "b"])
    local, local_report = RUNNER._within_section_type_derangement(
        donors, sections, labels, roles, seed=3
    )
    block, block_report = RUNNER._different_spatial_block_reassignment(
        donors, sections, labels, roles, blocks, seed=3
    )
    assert np.all(local != np.arange(6))
    assert sorted(local.tolist()) == list(range(6))
    assert local_report["changed_fraction"] == pytest.approx(1.0)
    assert np.all(blocks != blocks[block])
    assert sorted(block.tolist()) == list(range(6))
    assert block_report["cross_block_fraction"] == pytest.approx(1.0)


def test_block_null_reports_infeasible_groups_without_crossing() -> None:
    donors = np.repeat("D", 5)
    sections = np.repeat("S", 5)
    labels = np.repeat("T", 5)
    roles = np.repeat("evaluation", 5)
    blocks = np.asarray(["a", "a", "a", "a", "b"])
    mapping, report = RUNNER._different_spatial_block_reassignment(
        donors, sections, labels, roles, blocks, seed=7
    )
    assert np.array_equal(mapping, np.arange(5))
    assert report["infeasible_groups"] == 1
    assert report["eligible_rows"] == 0


def test_intrinsic_summary_requires_controls_and_both_nulls() -> None:
    def arm(block_p: float, increment: float = 0.01) -> dict[str, object]:
        effect = {"mean": increment, "positive_donor_fraction": 1.0}
        return {
            "models": {
                "combined_plus_image": {
                    "donor_type_equal_r2": 0.02,
                    "donor_section_type_equal_r2": 0.02,
                    "donor_type_equal_reference_error_reduction": 0.02,
                }
            },
            "nested_increment_over_combined_nonimage": {"donor_type_r2": effect},
            "increment_over_best_nonimage": {
                "effects": {"donor_type_r2": effect}
            },
            "nulls": {
                "within_section_type_derangement": {"empirical_p": 0.01},
                "different_spatial_block_reassignment": {"empirical_p": block_p},
            },
        }

    result = {
        "arms": {
            "crop_112um": arm(0.01),
            "cell_mask_only": arm(0.01),
            "nucleus_mask_only": arm(0.06),
            "target_cell_removed_112um": arm(0.5),
        },
        "crop_contrasts": {
            "cell_mask_minus_target_removed": {
                "donor_type_r2": {"mean": 0.01, "positive_donor_fraction": 1.0},
                "strict_locked_measurement_donor_type_r2": 0.01,
            },
            "nucleus_mask_minus_target_removed": {
                "donor_type_r2": {"mean": 0.01, "positive_donor_fraction": 1.0},
                "strict_locked_measurement_donor_type_r2": 0.01,
            },
        },
    }
    summary = RUNNER._evidence_summary(result)
    assert summary["H-INTRINSIC-cell-retrospective"]["evidence_status"] == (
        "exploratory_support"
    )
    nucleus = summary["H-INTRINSIC-nucleus-retrospective"]
    assert nucleus["evidence_status"] == "not_supported_or_indeterminate_in_this_analysis"
    assert next(
        row for row in nucleus["criteria"] if row["name"] == "different_block_null"
    )["pass"] is False

    result["arms"]["nucleus_mask_only"] = arm(0.01, increment=-0.01)
    nucleus = RUNNER._evidence_summary(result)["H-INTRINSIC-nucleus-retrospective"]
    assert nucleus["evidence_status"] == "not_supported_or_indeterminate_in_this_analysis"
    assert next(
        row for row in nucleus["criteria"] if row["name"] == "beats_combined_nonimage"
    )["pass"] is False
