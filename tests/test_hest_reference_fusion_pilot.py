from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _script():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "benchmark_hest_reference_fusion_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_hest_reference_fusion_pilot", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _script()


def _pilot_source(tmp_path: Path, *, blank: bool = False):
    rng = np.random.default_rng(31)
    donors = []
    sections = []
    types = []
    roles = []
    indications = []
    observations = []
    image_rows = []
    target_rows = []
    for donor_index in range(4):
        donor = f"D{donor_index}"
        for type_index, type_id in enumerate(("T0", "T1")):
            for role in ("evaluation", "reference"):
                for cell in range(5):
                    signal = np.asarray(
                        [
                            -0.8 + 0.45 * donor_index + 0.20 * cell,
                            -0.5 + 0.70 * type_index + 0.12 * cell,
                        ]
                    )
                    image = np.r_[signal, signal.sum()]
                    target = signal + rng.normal(scale=0.01, size=2)
                    donors.append(donor)
                    sections.append(f"S{donor_index}")
                    types.append(type_id)
                    roles.append(role)
                    indications.append("disease")
                    observations.append(
                        f"{donor}-{type_id}-{role}-{cell}"
                    )
                    image_rows.append(image)
                    target_rows.append(target)
    donors = np.asarray(donors)
    sections = np.asarray(sections)
    types = np.asarray(types)
    roles = np.asarray(roles)
    target = np.asarray(target_rows)
    image = np.asarray(image_rows, dtype=np.float32)
    evaluation = roles == "evaluation"
    reference = roles == "reference"
    # Highly repeatable split halves, with independent tiny perturbations.
    half_a = target + rng.normal(scale=0.005, size=target.shape)
    half_b = target + rng.normal(scale=0.005, size=target.shape)
    blank_images = np.zeros_like(image) if blank else None
    return RUNNER.PilotSource(
        path=tmp_path / "synthetic.npz",
        sha256="synthetic",
        observation_ids=np.asarray(observations),
        donors=donors,
        sections=sections,
        fine_types=types,
        indications=np.asarray(indications),
        donor_indications={f"D{i}": "disease" for i in range(4)},
        roles=roles,
        images=image,
        blank_images=blank_images,
        blank_status=(
            {
                "available": True,
                "source_key": "blank_patch_features",
                "definition": "synthetic",
            }
            if blank
            else {
                "available": False,
                "source_key": None,
                "reason": "source contains no blank_patch or frozen blank-patch features",
            }
        ),
        coordinates=image[:, :2],
        program_names=("program_a", "program_b"),
        program_total=target,
        program_half_a=half_a,
        program_half_b=half_b,
        evaluation_mask=evaluation,
        reference_mask=reference,
        reference_split_id="primary",
        support_strata=(),
    )


def test_source_loader_uses_registered_primary_features_and_supported_pools(
    tmp_path: Path,
) -> None:
    rows = 8
    donors = np.repeat(["D0", "D1"], 4)
    sections = np.repeat(["S0", "S1"], 4)
    fine_types = np.repeat("T", rows)
    roles = np.tile(["reference", "reference", "evaluation", "evaluation"], 2)
    counts_a = np.tile([[2, 1], [3, 1], [2, 2], [4, 1]], (2, 1)).astype(np.uint32)
    counts_b = counts_a + 1
    total = np.log1p((counts_a + counts_b) * 100.0)
    source_path = tmp_path / "source.npz"
    np.savez_compressed(
        source_path,
        schema_version=np.asarray(RUNNER.SOURCE_SCHEMA),
        study_stage=np.asarray("retrospective_exposed"),
        analysis_status=np.asarray("retrospective_exposed_non_authorizing"),
        authorizes_h_cell=np.asarray(False),
        authorizes_h_intrinsic=np.asarray(False),
        authorizes_full_heir=np.asarray(False),
        donor_ids=donors,
        section_ids=sections,
        fine_type_ids=fine_types,
        disease_statuses=np.repeat("disease", rows),
        observation_ids=np.asarray([f"o{i}" for i in range(rows)]),
        reference_split_ids=np.asarray(["primary"]),
        pool_roles_by_split=roles[:, None],
        crop_ids=np.asarray(["crop_112um"]),
        primary_crop_id=np.asarray("crop_112um"),
        frozen_features=np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        # This deliberately has a different width: the loader must prefer the
        # compact receipt-bound frozen_features matrix.
        image_features_by_crop_and_encoder=np.zeros((rows, 1, 7), dtype=np.float32),
        coordinate_features=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        program_names=np.asarray(["program"]),
        program_gene_membership=np.asarray([[True, True]]),
        normalized_nucleus_targets=total,
        nucleus_target_counts_half_a=counts_a,
        nucleus_target_counts_half_b=counts_b,
        nucleus_library_size_half_a=counts_a.sum(axis=1),
        nucleus_library_size_half_b=counts_b.sum(axis=1),
    )
    source = RUNNER.load_source(
        source_path,
        minimum_support=2,
        enforce_registered_hash=False,
    )
    assert source.images.shape == (rows, 3)
    assert source.evaluation_mask.sum() == 4
    assert source.reference_mask.sum() == 4
    assert source.blank_status["available"] is False
    assert source.donor_indications == {"D0": "disease", "D1": "disease"}


def test_macro_loss_equalizes_types_sections_and_donors() -> None:
    row_loss = np.asarray([1.0, 1.0, 9.0, 4.0, 4.0])
    result = RUNNER._macro_loss_from_rows(
        row_loss,
        np.asarray(["A", "A", "A", "B", "B"]),
        np.asarray(["a1", "a1", "a2", "b1", "b1"]),
        np.asarray(["x", "y", "x", "x", "x"]),
    )
    # A: section a1 mean(type losses 1,1)=1, a2=9, hence 5. B=4.
    assert result["per_donor"] == {"A": 5.0, "B": 4.0}
    assert result["donor_section_type_macro_mse"] == 4.5


def test_type_routing_only_reports_abstention_and_never_uses_image() -> None:
    bank = RUNNER.PrototypeBank(
        states=np.asarray([[1.0], [3.0]]),
        weights=np.asarray([1.0, 3.0]),
        donor_ids=np.asarray(["D", "D"]),
        type_labels=np.asarray(["T0", "T0"]),
        prototype_ids=np.asarray(["p0", "p1"]),
    )
    prediction, covered, receipt = RUNNER._type_routing_only(
        np.asarray(["D", "D"]),
        np.asarray(["T0", "missing"]),
        bank,
        {"D": np.asarray([0, 1])},
    )
    assert covered.tolist() == [True, False]
    assert prediction[0, 0] == 2.5
    assert np.isnan(prediction[1, 0])
    assert receipt["coverage"] == 0.5
    assert receipt["abstained_rows"] == 1
    assert receipt["uses_image"] is False


def test_equalized_bank_removes_natural_type_frequency() -> None:
    bank = RUNNER.PrototypeBank(
        states=np.arange(3, dtype=float)[:, None],
        weights=np.asarray([100.0, 1.0, 1.0]),
        donor_ids=np.asarray(["D", "D", "D"]),
        type_labels=np.asarray(["common", "rare", "rare"]),
        prototype_ids=np.asarray(["p0", "p1", "p2"]),
    )
    equalized = RUNNER._equalized_bank(bank)
    assert equalized.weights[0] == 0.5
    np.testing.assert_allclose(equalized.weights[1:], [0.25, 0.25])


def test_outer_donor_runs_all_same_assay_arms_without_iteration(tmp_path: Path) -> None:
    source = _pilot_source(tmp_path)
    report = RUNNER.evaluate_outer_donor(
        source,
        "D0",
        ridge_alphas=(0.01, 1.0),
        fusion_alphas=(0.0, 0.25),
        inner_folds=3,
        minimum_reliability=0.0,
        minimum_reliability_rows=3,
        max_prototypes_per_type=2,
        seed=4,
        device="cpu",
    )
    assert report["status"] == "complete_same_assay_engineering_dry_run"
    assert report["scope"].endswith("not matched sc/snRNA")
    assert report["target_basis_fit_donors"] == ["D1", "D2", "D3"]
    assert sorted(report["hard_wrong_donor_arms"]) == ["D1", "D2", "D3"]
    assert sorted(report["hard_wrong_donor_equalized_arms"]) == ["D1", "D2", "D3"]
    assert report["coverage_and_abstention"]["generic_excludes_query_donor"] is True
    assert report["coverage_and_abstention"]["type_routing_natural"]["coverage"] == 1.0
    assert report["arms"]["blank_H_plus_R_matched_natural"]["status"] == "unavailable"
    assert "coordinate_H_plus_R_matched_natural" in report["arms"]
    assert "shuffled_H_plus_R_matched_natural" in report["arms"]
    assert report["iteration"] == {
        "status": "not_run_by_design",
        "reason": (
            "HEST is a same-assay engineering dry run, not a scientific gate or "
            "refinement proxy"
        ),
        "rounds": 0,
    }
    assert report["descriptive_diagnostics_only"]["scientific_gate"] is False
    json.dumps(report, allow_nan=False)


def test_blank_arm_uses_source_blank_features_when_present(tmp_path: Path) -> None:
    source = _pilot_source(tmp_path, blank=True)
    report = RUNNER.evaluate_outer_donor(
        source,
        "D0",
        ridge_alphas=(0.1,),
        fusion_alphas=(0.0,),
        inner_folds=3,
        minimum_reliability=0.0,
        max_prototypes_per_type=1,
        device="cpu",
    )
    blank = report["arms"]["blank_H_plus_R_matched_natural"]
    assert blank["status"] == "complete"
    assert blank["rows"] == 10


def test_base_report_is_strictly_non_authorizing(tmp_path: Path) -> None:
    source = _pilot_source(tmp_path)
    report = RUNNER._base_report(source, seed=2, device="cpu")
    json.dumps(report, allow_nan=False)
    assert report["experiment_class"] == "retrospective_same_assay_engineering_dry_run"
    assert report["matched_scrna_reference"] is False
    assert report["personalized_reference_validation"] is False
    assert report["authorizes_reference_refinement"] is False
    assert report["authorizes_full_heir"] is False
    assert report["design"]["iteration"] == "prohibited for this dry run"


def test_aggregate_remains_descriptive_not_a_scientific_gate() -> None:
    report = {
        "D0": {
            "status": "complete_same_assay_engineering_dry_run",
            "arms": {
                "H": {"donor_section_type_macro_mse": 1.0},
                "H_plus_R_matched_natural": {
                    "donor_section_type_macro_mse": 0.8
                },
            },
            "hard_wrong_donor_arms": {
                "D1": {"donor_section_type_macro_mse": 0.9}
            },
        }
    }
    summary = RUNNER._aggregate_complete_donors(report)
    assert summary["descriptive_diagnostics_only"]["scientific_gate"] is False
    assert summary["iteration"]["status"] == "not_run_by_design"
    assert np.isclose(
        summary["descriptive_diagnostics_only"]["mean_relative_mse_gain"], 0.2
    )
