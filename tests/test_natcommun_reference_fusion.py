from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/benchmark_natcommun_reference_fusion.py"
SPEC = importlib.util.spec_from_file_location("benchmark_natcommun_reference_fusion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _csr_payload(prefix: str, matrix: np.ndarray) -> dict[str, np.ndarray]:
    rows, columns = matrix.shape
    data = []
    indices = []
    indptr = [0]
    for row in range(rows):
        local = np.flatnonzero(matrix[row])
        indices.extend(local.tolist())
        data.extend(matrix[row, local].tolist())
        indptr.append(len(data))
    return {
        f"{prefix}_data": np.asarray(data, dtype=np.int32),
        f"{prefix}_indices": np.asarray(indices, dtype=np.int32),
        f"{prefix}_indptr": np.asarray(indptr, dtype=np.int64),
        f"{prefix}_shape": np.asarray([rows, columns], dtype=np.int64),
    }


def _source_archive(tmp_path: Path, *, donors: int = 4) -> Path:
    rng = np.random.default_rng(41)
    donor_ids = np.asarray([f"D{donor}" for donor in range(donors) for _ in range(6)] + ["B2"] * 2)
    primary = donor_ids != "B2"
    rows = len(donor_ids)
    section_ids = np.asarray(
        [f"S{donor}" for donor in range(donors) for _ in range(6)] + ["SB2"] * 2
    )
    indication_ids = np.asarray(["tumor"] * rows)
    gene_ids = np.asarray([f"G{index:02d}" for index in range(24)])

    full = rng.poisson(4.0, size=(rows, len(gene_ids))).astype(np.int32)
    half_a = rng.binomial(full, 0.5).astype(np.int32)
    half_b = full - half_a
    totals_full = full.sum(axis=1).astype(np.int64) + 500
    totals_half_a = half_a.sum(axis=1).astype(np.int64) + 250
    totals_half_b = totals_full - totals_half_a

    cells_per_primary_type = runner.REFERENCE_MINIMUM_QUALIFIED_CELLS
    primary_types = ("T", "B")
    sc_donors = np.asarray(
        [
            f"D{donor}"
            for donor in range(donors)
            for _type_name in primary_types
            for _ in range(cells_per_primary_type)
        ]
        + ["B2"] * 50
    )
    sc_rows = len(sc_donors)
    sc_types = np.asarray(
        [
            type_name
            for _donor in range(donors)
            for type_name in primary_types
            for _ in range(cells_per_primary_type)
        ]
        + ["B"] * 5
        + ["Myeloid"] * 24
        + ["Stroma"] * 21
    )
    sc_counts = rng.poisson(4.0, size=(sc_rows, len(gene_ids))).astype(np.int32)
    # Give types reproducibly distinct state profiles.
    sc_counts[sc_types == "T", :4] += 7
    sc_counts[sc_types == "B", 4:8] += 7
    sc_totals = sc_counts.sum(axis=1).astype(np.int64) + 300
    per_type_depth = np.resize(np.asarray([100, 200, 300]), cells_per_primary_type)
    per_type_quality = np.resize(np.asarray([20, 40, 60]), cells_per_primary_type)
    repeated_depth = np.tile(per_type_depth, donors * len(primary_types))
    repeated_quality = np.tile(per_type_quality, donors * len(primary_types))
    n_count = np.concatenate((repeated_depth, np.resize([100, 200, 300], 50))).astype(float)
    n_feature = np.concatenate((repeated_quality, np.resize([20, 40, 60], 50))).astype(float)

    coverage_pairs = sorted(set(zip(sc_donors.tolist(), sc_types.tolist())))
    coverage_donors = np.asarray([donor for donor, _type_name in coverage_pairs])
    coverage_types = np.asarray([type_name for _donor, type_name in coverage_pairs])
    coverage_counts = np.asarray(
        [
            int(np.sum((sc_donors == donor) & (sc_types == type_name)))
            for donor, type_name in coverage_pairs
        ],
        dtype=np.int32,
    )

    membership = np.zeros((3, len(gene_ids)), dtype=bool)
    membership[0, :8] = True
    membership[1, 8:16] = True
    membership[2, 16:] = True
    parity_payload = {
        "schema": "heir.hoptimus1_official_local_parity.v1",
        "repository": runner.EXPECTED_ENCODER_REPOSITORY,
        "revision": "pinned-test-revision",
        "encoder_manifest_sha256": "test-manifest-sha256",
        "status": "passed",
        "passed": True,
    }
    parity_path = tmp_path / "official_local_parity.json"
    parity_path.write_text(json.dumps(parity_payload), encoding="utf-8")
    parity_sha256 = hashlib.sha256(parity_path.read_bytes()).hexdigest()
    receipt = {
        "schema": runner.SOURCE_RECEIPT_SCHEMA,
        "analysis_scope": runner.SCOPE,
        "builder_implementation_sha256": runner._sha256(runner.FROZEN_SOURCE_BUILDER),
        "protocol_sha256": runner._sha256(runner.FROZEN_PROTOCOL),
        "encoder": {
            "repository": runner.EXPECTED_ENCODER_REPOSITORY,
            "revision": "pinned-test-revision",
            "fine_tuning": "prohibited",
            "device": "cuda",
            "manifest_sha256": "test-manifest-sha256",
            "official_local_parity": {
                "status": "passed",
                "schema": "heir.hoptimus1_official_local_parity.v1",
                "encoder_manifest_sha256": "test-manifest-sha256",
                "receipt_path": str(parity_path),
                "receipt_sha256": parity_sha256,
            },
        },
        "encoder_roles": {
            "primary": {"repository": runner.EXPECTED_ENCODER_REPOSITORY},
            "secondary_comparator": {
                "repository": runner.SECONDARY_ENCODER_REPOSITORY,
                "status": "prespecified_not_run_in_primary_source",
            },
        },
    }
    blank_receipt = {
        "schema": "heir.natcommun_blank_image_control.v1",
        "pixels": "all-white 224x224 RGB",
        "encoder": runner.EXPECTED_ENCODER_REPOSITORY,
    }
    path = tmp_path / "source.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray(runner.SOURCE_SCHEMA),
        source_receipt_json=np.asarray(json.dumps(receipt)),
        blank_image_receipt_json=np.asarray(json.dumps(blank_receipt)),
        spot_ids=np.asarray([f"spot{index}" for index in range(rows)]),
        barcode_ids=np.asarray([f"bc{index}" for index in range(rows)]),
        donor_ids=donor_ids,
        section_ids=section_ids,
        indication_ids=indication_ids,
        spot_primary_eligible=primary,
        image_features=rng.normal(size=(rows, 4)).astype(np.float32),
        blank_image_feature_vector=np.zeros(4, dtype=np.float32),
        coordinate_features=rng.normal(size=(rows, 3)).astype(np.float32),
        gene_ids=gene_ids,
        broad_gene_ids=gene_ids,
        st_counts_full=full,
        st_counts_half_a=half_a,
        st_counts_half_b=half_b,
        st_total_umi_counts_full=totals_full,
        st_total_umi_counts_half_a=totals_half_a,
        st_total_umi_counts_half_b=totals_half_b,
        sc_counts=sc_counts,
        sc_total_umi_counts=sc_totals,
        sc_cell_ids=np.asarray([f"cell{index}" for index in range(sc_rows)]),
        sc_donor_ids=sc_donors,
        sc_indication_ids=np.asarray(["tumor"] * sc_rows),
        sc_primary_eligible=sc_donors != "B2",
        sc_level1_type_ids=sc_types,
        sc_n_count=n_count,
        sc_n_features_rna=n_feature,
        sc_percent_mt=np.zeros(sc_rows),
        reference_coverage_donor_ids=coverage_donors,
        reference_coverage_level1_type_ids=coverage_types,
        reference_coverage_cell_counts=coverage_counts,
        reference_coverage_primary_eligible=coverage_donors != "B2",
        program_names=np.asarray(["p1", "p2", "p3"]),
        program_gene_membership=membership,
        **_csr_payload("st_broad_counts_full", full),
        **_csr_payload("st_broad_counts_half_a", half_a),
        **_csr_payload("st_broad_counts_half_b", half_b),
        **_csr_payload("sc_broad_counts", sc_counts),
    )
    return path


def _hest_qualification_report(tmp_path: Path) -> tuple[Path, str]:
    source_sha256 = "a" * 64
    positive_controls = {
        target: {
            "model": {
                "donor_balanced": {"donor_macro_balanced_accuracy": model}
            },
            "training_majority_baseline": {
                "donor_balanced": {"donor_macro_balanced_accuracy": baseline}
            },
        }
        for target, model, baseline in (
            ("broad_lineage", 0.60, 0.25),
            ("fine_type", 0.30, 0.10),
        )
    }
    positive_controls["nuclear_morphology"] = {
        "full_context": {
            "scores": {
                "targets": {
                    target: {"donor_type_macro_reference_error_reduction": 0.10}
                    for target in runner.HEST_REQUIRED_MORPHOLOGY_TARGETS
                }
            }
        }
    }
    payload = {
        "schema": "heir.hest_scientific_reanalysis.v2",
        "analysis_status": "retrospective_exposed_non_authorizing",
        "study_stage": "retrospective_exposed",
        "requested_phase": "full",
        "execution_status": "scientific_reanalysis_complete",
        "encoder": runner.EXPECTED_ENCODER_REPOSITORY,
        "encoder_revision": runner.HOPTIMUS1_REVISION,
        "encoder_manifest_sha256": runner.HOPTIMUS1_MANIFEST_SHA256,
        "encoder_role": "primary_Hoptimus1_qualification",
        "encoder_feature_width": 1536,
        "source_sha256": source_sha256,
        "source_receipt_expected_sha256": source_sha256,
        "positive_controls": positive_controls,
        "positive_control_gate": runner._recompute_hest_positive_control_gate(
            positive_controls
        ),
        "same_runner_uni2_comparator_preflight": {
            "schema": "heir.hest_same_runner_uni2_preflight.v1",
            "passed": True,
        },
        "implementation_receipt": {
            "file_sha256": {
                "scripts/benchmark_hest_scientific_reanalysis.py": runner._sha256(
                    runner.FROZEN_HEST_RUNNER
                ),
                "src/heir/evaluation/hest_nested_ridge.py": runner._sha256(
                    runner.FROZEN_NESTED_RIDGE
                ),
                "src/heir/evaluation/hest_scoring.py": runner._sha256(
                    runner.FROZEN_SCORING
                ),
                "src/heir/evaluation/hest_measurement.py": runner._sha256(
                    runner.FROZEN_HEST_MEASUREMENT
                ),
            },
            "command": [
                str(runner.FROZEN_HEST_RUNNER),
                "--phase",
                "full",
                "--device",
                "cuda",
                "--representation-profile",
                "full",
                "--expected-source-sha256",
                source_sha256,
                "--expected-encoder",
                runner.EXPECTED_ENCODER_REPOSITORY,
                "--comparison-report",
                "registered_uni2_report.json",
            ],
        },
        "numeric_backend": {
            "requested_device": "cuda",
            "cuda_available": True,
            "deterministic_algorithms_enabled": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    path = tmp_path / "hest_hoptimus_qualification.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, source_sha256


def _contract_args(tmp_path: Path, source_path: Path) -> SimpleNamespace:
    hest_report, hest_source_sha = _hest_qualification_report(tmp_path)
    values = {
        "expected_source_sha256": runner._sha256(source_path),
        "expected_protocol_sha256": runner._sha256(runner.FROZEN_PROTOCOL),
        "expected_builder_sha256": runner._sha256(runner.FROZEN_SOURCE_BUILDER),
        "expected_runner_sha256": runner._sha256(SCRIPT),
        "expected_reference_fusion_sha256": runner._sha256(
            runner.FROZEN_REFERENCE_FUSION
        ),
        "expected_nested_ridge_sha256": runner._sha256(runner.FROZEN_NESTED_RIDGE),
        "expected_scoring_sha256": runner._sha256(runner.FROZEN_SCORING),
        "expected_hest_measurement_sha256": runner._sha256(
            runner.FROZEN_HEST_MEASUREMENT
        ),
        "hest_hoptimus_qualification_report": hest_report,
        "expected_hest_hoptimus_qualification_report_sha256": runner._sha256(
            hest_report
        ),
        "expected_hest_hoptimus_source_sha256": hest_source_sha,
        "expected_hest_runner_sha256": runner._sha256(runner.FROZEN_HEST_RUNNER),
    }
    command = [sys.executable, str(SCRIPT)]
    for option, key in (
        ("--expected-source-sha256", "expected_source_sha256"),
        ("--expected-protocol-sha256", "expected_protocol_sha256"),
        ("--expected-builder-sha256", "expected_builder_sha256"),
        ("--expected-runner-sha256", "expected_runner_sha256"),
        ("--expected-reference-fusion-sha256", "expected_reference_fusion_sha256"),
        ("--expected-nested-ridge-sha256", "expected_nested_ridge_sha256"),
        ("--expected-scoring-sha256", "expected_scoring_sha256"),
        ("--expected-hest-measurement-sha256", "expected_hest_measurement_sha256"),
        ("--hest-hoptimus-qualification-report", "hest_hoptimus_qualification_report"),
        (
            "--expected-hest-hoptimus-qualification-report-sha256",
            "expected_hest_hoptimus_qualification_report_sha256",
        ),
        (
            "--expected-hest-hoptimus-source-sha256",
            "expected_hest_hoptimus_source_sha256",
        ),
        ("--expected-hest-runner-sha256", "expected_hest_runner_sha256"),
    ):
        command.extend((option, str(values[key])))
    command.extend(("--device", "cuda"))
    return SimpleNamespace(**values, command=command)


def test_source_contract_filters_sensitivity_donor_and_checks_split(tmp_path):
    path = _source_archive(tmp_path)
    source_sha256 = runner._sha256(path)
    source = runner.load_source(
        path,
        expected_primary_donors=4,
        expected_source_sha256=source_sha256,
    )
    assert set(source.donor_ids) == {"D0", "D1", "D2", "D3"}
    assert set(source.sc_donor_ids) == {"D0", "D1", "D2", "D3"}
    np.testing.assert_array_equal(source.st_full, source.st_half_a + source.st_half_b)
    np.testing.assert_array_equal(
        source.st_total_full, source.st_total_half_a + source.st_total_half_b
    )
    assert source.source_receipt["encoder"]["repository"] == "bioptimus/H-optimus-1"
    with pytest.raises(ValueError, match="expected-source-sha256"):
        runner.load_source(
            path,
            expected_primary_donors=4,
            expected_source_sha256="0" * 64,
        )

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["st_counts_half_a"] = payload["st_counts_half_a"].copy()
    payload["st_counts_half_a"][0, 0] += 1
    broken = tmp_path / "broken.npz"
    np.savez_compressed(broken, **payload)
    with pytest.raises(ValueError, match="do not exactly reconstruct"):
        runner.load_source(broken, expected_primary_donors=4)


def test_source_rejects_non_hoptimus_image_features(tmp_path):
    path = _source_archive(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    receipt = json.loads(str(payload["source_receipt_json"]))
    receipt["encoder"]["repository"] = "MahmoodLab/UNI2-h"
    payload["source_receipt_json"] = np.asarray(json.dumps(receipt))
    wrong = tmp_path / "wrong_encoder.npz"
    np.savez_compressed(wrong, **payload)
    with pytest.raises(ValueError, match="H-optimus-1"):
        runner.load_source(wrong, expected_primary_donors=4)


def test_source_rejects_hoptimus_without_passed_exact_manifest_parity(tmp_path):
    path = _source_archive(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    receipt = json.loads(str(payload["source_receipt_json"]))
    receipt["encoder"]["official_local_parity"]["status"] = "failed"
    payload["source_receipt_json"] = np.asarray(json.dumps(receipt))
    failed = tmp_path / "failed_parity.npz"
    np.savez_compressed(failed, **payload)
    with pytest.raises(ValueError, match="parity must pass"):
        runner.load_source(failed, expected_primary_donors=4)


def test_composition_equalization_uses_identical_common_strata_quota(tmp_path):
    source = runner.load_source(_source_archive(tmp_path), expected_primary_donors=4)
    first, receipt_first = runner._equalized_indices(source, "D0", ["D0"], pooled=False, seed=3)
    wrong, receipt_wrong = runner._equalized_indices(source, "D0", ["D1"], pooled=False, seed=3)
    generic, receipt_generic = runner._equalized_indices(
        source, "D0", ["D1", "D2", "D3"], pooled=True, seed=3
    )
    assert len(first) == len(wrong) == len(generic)
    assert receipt_first["quota_per_stratum"] == receipt_wrong["quota_per_stratum"]
    assert receipt_first["quota_per_stratum"] == receipt_generic["quota_per_stratum"]
    assert not np.any(source.sc_donor_ids[generic] == "D0")
    assert receipt_generic["outcome_used"] is False


def test_adaptive_fusion_falls_back_exactly_to_h_outside_support():
    image = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    reference = np.zeros_like(image)
    prediction, receipt = runner._adaptive_fusion(
        image,
        reference,
        {
            "support_distance": np.asarray([0.0, 100.0]),
            "type_coverage": np.asarray([1.0, 1.0]),
            "reference_uncertainty": np.asarray([0.0, 0.0]),
        },
        base_alpha=0.5,
        support_threshold=10.0,
    )
    np.testing.assert_array_equal(prediction[1], image[1])
    assert receipt["adaptive_alpha"][0] == pytest.approx(0.5)
    assert receipt["adaptive_alpha"][1] == 0.0
    assert receipt["abstained_fallback_to_H"].tolist() == [False, True]


def test_program_reliability_gate_uses_only_training_donors():
    donors = np.asarray([donor for donor in ("A", "B", "C", "HELD") for _ in range(5)])
    first = np.zeros((len(donors), 4), dtype=float)
    second = np.zeros_like(first)
    for donor in ("A", "B", "C", "HELD"):
        selected = donors == donor
        increasing = np.arange(5, dtype=float)
        first[selected] = np.column_stack((increasing, increasing, increasing, increasing))
        second[selected] = np.column_stack((increasing, increasing, increasing, increasing[::-1]))
    gate = runner._program_reliability_gate(
        first,
        second,
        donors,
        ["A", "B", "C"],
        np.asarray(["p1", "p2", "p3", "bad"]),
    )
    assert gate["status"] == "feasible"
    assert gate["retained_programs"] == ["p1", "p2", "p3"]
    assert gate["heldout_donor_outcomes_used"] is False
    assert "HELD" not in gate["programs"]["p1"]["training_donor_spearman"]


def test_sparse_variance_and_column_slice_do_not_use_masked_rows():
    dense = np.asarray([[1, 0, 2], [0, 3, 0], [1000, 1000, 1000]], dtype=np.int32)
    payload = _csr_payload("x", dense)
    csr = runner.CSRCounts(
        payload["x_data"],
        payload["x_indices"],
        payload["x_indptr"],
        (3, 3),
    )
    mask = np.asarray([True, True, False])
    first = csr.weighted_log_variance(mask, np.asarray([10, 10, 3000]), np.ones(3))
    altered = dense.copy()
    altered[2] *= 1000
    changed = _csr_payload("x", altered)
    changed_csr = runner.CSRCounts(
        changed["x_data"], changed["x_indices"], changed["x_indptr"], (3, 3)
    )
    second = changed_csr.weighted_log_variance(mask, np.asarray([10, 10, 3_000_000]), np.ones(3))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(csr.dense_columns(np.asarray([2, 0])), dense[:, [2, 0]])


def test_program_benchmark_has_exact_M0_M8_and_no_iteration(tmp_path):
    source = runner.load_source(_source_archive(tmp_path), expected_primary_donors=4)
    result = runner.run_endpoint(
        source,
        "program_total",
        "natural",
        ridge_alphas=(0.1,),
        fusion_alphas=(0.0, 0.25),
        temperatures=(1.0,),
        pca_components=20,
        pca_genes=24,
        prototypes_per_type=2,
        bootstrap_iterations=32,
        seed=7,
        device="cpu",
    )
    assert set(result["models"]) == {f"M{index}" for index in range(9)}
    assert {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"} <= set(result["scores"])
    assert result["iteration"] == {
        "status": "not_run_prohibited_by_first_experiment_protocol",
        "maximum_rounds": 1,
        "implementation_present": False,
    }
    assert result["within_type_residual_state_endpoint"]["status"] == "blocked_unavailable"
    assert all(len(value) == 3 for value in result["wrong_donor_conditions"].values())
    for donor, receipt in result["fold_receipts"].items():
        assert donor not in receipt["target_basis"]["fit_donors"]
        assert receipt["reference_calibration_excludes_heldout"] is True
        assert receipt["heldout_ST_used_for_fit_selection_or_support_threshold"] is False
        assert receipt["shuffle_receipt"]["fixed_points"] == 0
        exclusion_key = (
            "failed_reference_sensitivity_donors_excluded_from_all_primary_"
            "fit_selection_inference"
        )
        assert receipt[exclusion_key] == ["B2"]
        assert "B2" not in receipt["outer_training_donors"]
        bank_receipts = [receipt["matched_bank"], receipt["generic_bank"]]
        bank_receipts.extend(receipt["inner_bank_receipts"].values())
        bank_receipts.extend(receipt["wrong_bank_receipts"].values())
        for bank_receipt in bank_receipts:
            qualified = bank_receipt["type_qualification"]["qualified_type_counts"]
            prototypes = bank_receipt.get("prototype_type_counts", {})
            assert set(prototypes) <= set(qualified)
            assert all(
                count >= runner.REFERENCE_MINIMUM_QUALIFIED_CELLS
                for count in qualified.values()
            )
    assert result["scores"]["M8"]["full_depth_correction_factor"] == 0.25
    assert result["scores"]["M0"]["rows"] == len(source.spot_ids)
    assert "donor_section_macro_R2" in result["scores"]["M0"]
    assert result["scores"]["M0"]["exact_donor_type_normalized_loss"] is None
    assert result["scores"]["M0"]["exact_type_balanced_loss_status"].startswith("blocked_")
    assert result["scores"]["M8_raw_cross_half"]["rows"] == 2 * len(source.spot_ids)
    assert result["evaluation_rows"] == {
        "M0_through_M7": "one prediction row per held-out Visium spot",
        "M8": "two cross-fitted split-half rows per same underlying held-out spot",
        "same_underlying_heldout_spot_identities": True,
        "M8_rows": 2 * len(source.spot_ids),
        "underlying_spots": len(source.spot_ids),
    }
    assert result["all_candidate_program_scores_role"] == "secondary"
    assert result["reliability_qualified_program_primary"]["status"].startswith(
        ("feasible", "blocked_")
    )


def test_broad_pca_uses_sparse_outer_training_gene_selection(tmp_path):
    source = runner.load_source(_source_archive(tmp_path), expected_primary_donors=4)
    result = runner.run_endpoint(
        source,
        "pca_total",
        "composition_equalized",
        ridge_alphas=(0.1,),
        fusion_alphas=(0.0,),
        temperatures=(1.0,),
        pca_components=5,
        pca_genes=12,
        prototypes_per_type=1,
        bootstrap_iterations=16,
        seed=11,
        device="cpu",
    )
    assert result["headline"]["loss_role"] == "primary_outer_training_donor_PCA"
    for donor, receipt in result["fold_receipts"].items():
        basis = receipt["target_basis"]
        assert basis["representation"].endswith("broad_common_gene_CSR")
        assert basis["global_broad_matrix_densified"] is False
        assert donor not in basis["fit_donors"]
        assert basis["heldout_outcomes_or_scRNA_used_for_gene_selection"] is False


def test_cli_enforces_broad_latent_dimension_and_bounded_fusion(tmp_path):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--source",
                str(tmp_path / "source.npz"),
                "--output-dir",
                str(tmp_path / "out"),
                "--pca-components",
                "19",
            ]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--source",
                str(tmp_path / "source.npz"),
                "--output-dir",
                str(tmp_path / "out"),
                "--fusion-alphas",
                "0,0.75",
            ]
        )


def test_frozen_reference_threshold_excludes_49_and_retains_50():
    types = np.asarray(["forty_nine"] * 49 + ["fifty"] * 50)
    retained, receipt = runner._qualify_reference_indices(types, np.arange(len(types)))
    assert set(types[retained]) == {"fifty"}
    assert receipt["selected_type_counts"] == {"fifty": 50, "forty_nine": 49}
    assert receipt["qualified_type_counts"] == {"fifty": 50}
    assert receipt["excluded_subthreshold_type_counts"] == {"forty_nine": 49}
    assert receipt["rule_timing"] == "after_exact_natural_or_equalized_bank_selection"


@pytest.mark.parametrize("bank_mode", ["natural", "composition_equalized"])
def test_bank_threshold_is_applied_after_selection_and_no_subthreshold_prototype(
    tmp_path, bank_mode
):
    source = runner.load_source(_source_archive(tmp_path), expected_primary_donors=4)
    bank, receipt = runner._fold_bank(
        source,
        np.ones((len(source.sc_cell_ids), 3)),
        "D0",
        ["D0"],
        bank_mode,
        pooled=False,
        prototypes_per_type=2,
        seed=3,
        bank_role="test_primary",
        fail_if_no_qualified_types=True,
    )
    qualification = receipt["type_qualification"]
    assert qualification["selected_cells_before_type_qualification"] == receipt["selected_cells"]
    assert all(
        count >= runner.REFERENCE_MINIMUM_QUALIFIED_CELLS
        for count in qualification["qualified_type_counts"].values()
    )
    assert set(bank.type_labels) <= set(qualification["qualified_type_counts"])
    assert set(receipt["prototype_type_counts"]) <= set(
        qualification["qualified_type_counts"]
    )


def test_primary_bank_without_a_qualified_type_fails_closed(tmp_path):
    source = runner.load_source(_source_archive(tmp_path), expected_primary_donors=4)
    with pytest.raises(ValueError, match="primary reference bank"):
        runner._fold_bank(
            source,
            np.ones((len(source.sc_cell_ids), 2)),
            "D0",
            ["not-a-donor"],
            "natural",
            pooled=False,
            prototypes_per_type=1,
            seed=3,
            bank_role="matched_primary",
            fail_if_no_qualified_types=True,
        )


def test_empty_or_unsupported_reference_is_bitwise_H_only_with_zero_alpha():
    image = np.asarray([[0.125, -2.5], [3.25, 8.0]], dtype=np.float64)
    route = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    reference, diagnostics = runner._retrieve(
        image,
        route,
        runner._empty_bank(image.shape[1]),
        np.asarray(["T", "B"]),
        1.0,
    )
    prediction, receipt = runner._adaptive_fusion(
        image,
        reference,
        diagnostics,
        base_alpha=0.5,
        support_threshold=1.0,
    )
    assert np.array_equal(prediction, image)
    assert prediction.tobytes() == image.tobytes()
    np.testing.assert_array_equal(receipt["adaptive_alpha"], np.zeros(len(image)))
    assert receipt["abstained_fallback_to_H"].all()


def test_B2_count_audit_forces_descriptive_zero_alpha_fallback(tmp_path):
    source = runner.load_source(_source_archive(tmp_path), expected_primary_donors=4)
    audit = source.failed_reference_sensitivity
    assert audit["selected_reference_cells"] == 50
    assert audit["selected_type_counts"] == {"B": 5, "Myeloid": 24, "Stroma": 21}
    assert audit["qualified_type_count"] == 0
    report = runner._failed_reference_sensitivity_report(source)
    assert report["qualified_type_count"] == 0
    assert report["unsupported_spot_fraction"] == 1.0
    assert report["adaptive_alpha"]["unique_values"] == [0.0]
    assert report["M3_equals_M0"]["exact"] is True
    assert report["used_for_any_primary_fit_selection_support_threshold_or_inference"] is False
    serialized = json.dumps(report, sort_keys=True).lower()
    assert "p_value" not in serialized
    assert "p-value" not in serialized


def test_permuting_B2_spatial_outcomes_leaves_primary_analysis_unchanged(tmp_path):
    path = _source_archive(tmp_path)
    source = runner.load_source(path, expected_primary_donors=4)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    for name in (
        "st_counts_full",
        "st_counts_half_a",
        "st_counts_half_b",
        "st_total_umi_counts_full",
        "st_total_umi_counts_half_a",
        "st_total_umi_counts_half_b",
    ):
        payload[name] = payload[name].copy()
        payload[name][-2:] = payload[name][-2:][::-1]
    for prefix, dense_name in (
        ("st_broad_counts_full", "st_counts_full"),
        ("st_broad_counts_half_a", "st_counts_half_a"),
        ("st_broad_counts_half_b", "st_counts_half_b"),
    ):
        payload.update(_csr_payload(prefix, payload[dense_name]))
    permuted = tmp_path / "permuted_B2_outcomes.npz"
    np.savez_compressed(permuted, **payload)
    altered = runner.load_source(permuted, expected_primary_donors=4)
    np.testing.assert_array_equal(source.st_full, altered.st_full)
    np.testing.assert_array_equal(source.st_half_a, altered.st_half_a)
    assert source.failed_reference_sensitivity == altered.failed_reference_sensitivity
    kwargs = {
        "ridge_alphas": (0.1,),
        "fusion_alphas": (0.0,),
        "temperatures": (1.0,),
        "pca_components": 20,
        "pca_genes": 24,
        "prototypes_per_type": 1,
        "bootstrap_iterations": 16,
        "seed": 9,
        "device": "cpu",
    }
    original_result = runner.run_endpoint(source, "program_total", "natural", **kwargs)
    altered_result = runner.run_endpoint(altered, "program_total", "natural", **kwargs)
    assert original_result["headline"] == altered_result["headline"]
    assert original_result["paired_inference"] == altered_result["paired_inference"]
    assert (
        original_result["measurement_floor_inference"]
        == altered_result["measurement_floor_inference"]
    )


def test_execution_receipt_binds_exact_protocol_builder_runner_and_command(tmp_path):
    source_path = _source_archive(tmp_path)
    source = runner.load_source(source_path, expected_primary_donors=4)
    args = _contract_args(tmp_path, source_path)
    receipt = runner._implementation_receipt(args, source)
    assert receipt["file_sha256"] == receipt["expected_file_sha256"]
    assert receipt["command"] == args.command
    args.expected_runner_sha256 = "0" * 64
    with pytest.raises(ValueError, match="hashes do not match"):
        runner._implementation_receipt(args, source)


@pytest.mark.parametrize(
    "attribute",
    [
        "expected_reference_fusion_sha256",
        "expected_nested_ridge_sha256",
        "expected_scoring_sha256",
        "expected_hest_measurement_sha256",
    ],
)
def test_each_imported_evaluation_hash_mismatch_fails_closed(tmp_path, attribute):
    source_path = _source_archive(tmp_path)
    source = runner.load_source(source_path, expected_primary_donors=4)
    args = _contract_args(tmp_path, source_path)
    setattr(args, attribute, "0" * 64)
    with pytest.raises(ValueError, match="hashes do not match"):
        runner._implementation_receipt(args, source)


def test_hest_hoptimus_prerequisite_is_hash_and_gate_bound(tmp_path):
    source_path = _source_archive(tmp_path)
    args = _contract_args(tmp_path, source_path)
    receipt = runner._load_hest_hoptimus_qualification(args)
    assert receipt["molecular_interpretation_prerequisite_satisfied"] is True
    assert receipt["execution_status"] == "scientific_reanalysis_complete"
    args.expected_hest_hoptimus_qualification_report_sha256 = "0" * 64
    with pytest.raises(ValueError, match="report hash"):
        runner._load_hest_hoptimus_qualification(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution_status", "positive_controls_complete", "identity"),
        ("encoder_role", "wrong_role", "identity"),
        ("encoder_revision", "floating", "identity"),
    ],
)
def test_hest_hoptimus_prerequisite_rejects_incomplete_identity(
    tmp_path, field, value, message
):
    source_path = _source_archive(tmp_path)
    args = _contract_args(tmp_path, source_path)
    payload = json.loads(args.hest_hoptimus_qualification_report.read_text(encoding="utf-8"))
    payload[field] = value
    args.hest_hoptimus_qualification_report.write_text(
        json.dumps(payload), encoding="utf-8"
    )
    args.expected_hest_hoptimus_qualification_report_sha256 = runner._sha256(
        args.hest_hoptimus_qualification_report
    )
    with pytest.raises(ValueError, match=message):
        runner._load_hest_hoptimus_qualification(args)


def test_hest_hoptimus_prerequisite_rejects_failed_positive_gate(tmp_path):
    source_path = _source_archive(tmp_path)
    args = _contract_args(tmp_path, source_path)
    payload = json.loads(args.hest_hoptimus_qualification_report.read_text(encoding="utf-8"))
    payload["positive_control_gate"]["passed"] = False
    payload["positive_control_gate"]["molecular_interpretation_allowed"] = False
    args.hest_hoptimus_qualification_report.write_text(
        json.dumps(payload), encoding="utf-8"
    )
    args.expected_hest_hoptimus_qualification_report_sha256 = runner._sha256(
        args.hest_hoptimus_qualification_report
    )
    with pytest.raises(ValueError, match="incomplete, stale, or failed"):
        runner._load_hest_hoptimus_qualification(args)


@pytest.mark.parametrize("failure", ["missing_natural", "missing_target", "raw_gate_failure"])
def test_hest_positive_gate_requires_complete_recomputed_structure(tmp_path, failure):
    source_path = _source_archive(tmp_path)
    args = _contract_args(tmp_path, source_path)
    payload = json.loads(args.hest_hoptimus_qualification_report.read_text(encoding="utf-8"))
    if failure == "missing_natural":
        del payload["positive_control_gate"]["natural_unmasked_112um_is_primary"]
    elif failure == "missing_target":
        del payload["positive_controls"]["nuclear_morphology"]["full_context"]["scores"][
            "targets"
        ][runner.HEST_REQUIRED_MORPHOLOGY_TARGETS[0]]
    else:
        payload["positive_controls"]["broad_lineage"]["model"]["donor_balanced"][
            "donor_macro_balanced_accuracy"
        ] = 0.1
    args.hest_hoptimus_qualification_report.write_text(
        json.dumps(payload), encoding="utf-8"
    )
    args.expected_hest_hoptimus_qualification_report_sha256 = runner._sha256(
        args.hest_hoptimus_qualification_report
    )
    with pytest.raises(ValueError, match="positive"):
        runner._load_hest_hoptimus_qualification(args)


def test_hest_report_hash_and_json_use_the_same_single_read_bytes(tmp_path, monkeypatch):
    source_path = _source_archive(tmp_path)
    args = _contract_args(tmp_path, source_path)
    target = args.hest_hoptimus_qualification_report.resolve()
    original_read_bytes = Path.read_bytes
    malicious = json.dumps({"schema": "replaced_after_read"}).encode()
    reads = 0

    def replacing_read_bytes(path):
        nonlocal reads
        data = original_read_bytes(path)
        if path.resolve() == target:
            reads += 1
            path.write_bytes(malicious)
        return data

    monkeypatch.setattr(Path, "read_bytes", replacing_read_bytes)
    receipt = runner._load_hest_hoptimus_qualification(args)
    assert receipt["molecular_interpretation_prerequisite_satisfied"] is True
    assert reads == 1
    assert original_read_bytes(target) == malicious


def test_hest_hoptimus_prerequisite_rejects_runner_or_source_mismatch(tmp_path):
    source_path = _source_archive(tmp_path)
    args = _contract_args(tmp_path, source_path)
    args.expected_hest_runner_sha256 = "0" * 64
    with pytest.raises(ValueError, match="current HEST runner"):
        runner._load_hest_hoptimus_qualification(args)
    args = _contract_args(tmp_path, source_path)
    args.expected_hest_hoptimus_source_sha256 = "b" * 64
    with pytest.raises(ValueError, match="identity"):
        runner._load_hest_hoptimus_qualification(args)


def test_numeric_backend_prohibits_cpu_fallback():
    with pytest.raises(ValueError, match="requires --device cuda"):
        runner._configure_numeric_backend(2, "cpu")


def test_cuda_ridge_runtime_fallback_is_rejected(monkeypatch):
    class CpuFit:
        fit_device = "cpu"

        def predict(self, values):
            return np.zeros((1, len(values), 1))

    monkeypatch.setattr(runner, "fit_weighted_ridge_grid", lambda *args, **kwargs: CpuFit())
    with pytest.raises(RuntimeError, match="silently fell back"):
        runner._fit_predict_ridge(
            np.ones((4, 2)),
            np.ones((4, 1)),
            np.ones((2, 2)),
            np.asarray(["A", "A", "B", "B"]),
            np.asarray(["A", "A", "B", "B"]),
            0.1,
            "cuda",
        )


def test_decisive_hypothesis_is_global_holm_aware_and_fail_closed():
    experiment_names = [
        f"{endpoint}::{bank}"
        for endpoint in ("program_total", "pca_total")
        for bank in ("natural", "composition_equalized")
    ]
    experiments = {
        name: {
            "headline": {"status": "evaluable"},
            "paired_inference": {
                comparison: {"mean_effect": 1.0}
                for comparison in runner.DECISIVE_COMPARISONS
            },
        }
        for name in experiment_names
    }
    adjusted = {
        f"{name}::{comparison}": 0.01
        for name in experiment_names
        for comparison in runner.DECISIVE_COMPARISONS
    }
    supported = runner._decisive_hypothesis_decision(experiments, adjusted)
    assert supported["status"] == "evaluable"
    assert supported["supported"] is True
    first = next(iter(adjusted))
    adjusted[first] = 0.051
    not_supported = runner._decisive_hypothesis_decision(experiments, adjusted)
    assert not_supported["status"] == "evaluable"
    assert not_supported["supported"] is False
    del adjusted[first]
    blocked = runner._decisive_hypothesis_decision(experiments, adjusted)
    assert blocked["status"] == "blocked_fail_closed"
    assert blocked["decision"] == "blocked_indeterminate"
    assert blocked["supported"] is False


def test_floor_inequality_is_separate_and_required_for_overall_support():
    experiment_names = [
        f"{endpoint}::{bank}"
        for endpoint in ("program_total", "pca_total")
        for bank in ("natural", "composition_equalized")
    ]
    experiments = {
        name: {
            "headline": {"status": "evaluable"},
            "measurement_floor_inference": {
                "mean_effect": 1.0,
                "exact_sign_flip_p": 0.01,
            },
        }
        for name in experiment_names
    }
    floor = runner._measurement_floor_decision(experiments)
    assert floor["supported"] is True
    controls = {"status": "evaluable", "supported": True}
    effect = {"status": "evaluable", "supported": True}
    overall = runner._overall_scientific_decision(controls, floor, effect)
    assert overall["supported"] is True
    assert overall["scientific_authorization_requires_all_components"] is True

    # M8 >= M3 makes M3-loss minus M8-loss nonpositive even when all controls pass.
    experiments[experiment_names[0]]["measurement_floor_inference"]["mean_effect"] = -0.1
    failed_floor = runner._measurement_floor_decision(experiments)
    assert failed_floor["status"] == "evaluable"
    assert failed_floor["decision"] == "not_supported"
    assert failed_floor["supported"] is False
    failed_overall = runner._overall_scientific_decision(controls, failed_floor, effect)
    assert failed_overall["decision"] == "not_supported"
    assert failed_overall["supported"] is False


def test_blocked_floor_evidence_is_indeterminate_not_negative():
    floor = runner._measurement_floor_decision({})
    assert floor["status"] == "blocked_fail_closed"
    assert floor["decision"] == "blocked_indeterminate"
    controls = {"status": "evaluable", "supported": True}
    effect = {"status": "evaluable", "supported": True}
    overall = runner._overall_scientific_decision(controls, floor, effect)
    assert overall["status"] == "blocked_fail_closed"
    assert overall["decision"] == "blocked_indeterminate"
    assert overall["supported"] is False


def _effect_gate_experiments(
    *, relative_gain: float = 0.06, positive_donors: int = 10
) -> dict[str, object]:
    donors = runner.EXPECTED_PRIMARY_DONOR_IDS
    target_m3_mean = 1.0 - relative_gain
    negative_loss = 1.05
    positive_loss = (
        len(donors) * target_m3_mean
        - (len(donors) - positive_donors) * negative_loss
    ) / positive_donors
    donor_m3 = [
        positive_loss if index < positive_donors else negative_loss
        for index in range(len(donors))
    ]
    recomputed_m3 = float(np.mean(donor_m3, dtype=np.float64))
    recomputed_relative = 1.0 - recomputed_m3
    positive_fraction = positive_donors / len(donors)
    experiments = {}
    for endpoint in ("program_total", "pca_total"):
        for bank in ("natural", "composition_equalized"):
            per_donor = {
                donor: {
                    "M0_loss": 1.0,
                    "M3_loss": donor_m3[index],
                }
                for index, donor in enumerate(donors)
            }
            experiments[f"{endpoint}::{bank}"] = {
                "headline": {
                    "status": "evaluable",
                    "M0_loss": 1.0,
                    "M3_loss": recomputed_m3,
                    "relative_MSE_gain_M3_vs_M0": recomputed_relative,
                    "positive_donor_fraction_M3_vs_M0": positive_fraction,
                    "per_donor": per_donor,
                },
                "research_prototype_thresholds": {
                    "at_least_5_percent_relative_MSE_gain": recomputed_relative >= 0.05,
                    "at_least_70_percent_donors_positive": positive_fraction >= 0.70,
                },
            }
    return experiments


def test_effect_size_and_positive_donor_criteria_are_required_in_every_experiment():
    passing = runner._effect_size_consistency_decision(_effect_gate_experiments())
    assert passing["status"] == "evaluable"
    assert passing["decision"] == "supported"
    assert passing["supported"] is True

    low_gain = runner._effect_size_consistency_decision(
        _effect_gate_experiments(relative_gain=0.049)
    )
    assert low_gain["status"] == "evaluable"
    assert low_gain["decision"] == "not_supported"
    assert low_gain["supported"] is False

    low_donor_fraction = runner._effect_size_consistency_decision(
        _effect_gate_experiments(positive_donors=6)
    )
    assert low_donor_fraction["status"] == "evaluable"
    assert low_donor_fraction["decision"] == "not_supported"
    assert low_donor_fraction["supported"] is False


@pytest.mark.parametrize(
    ("headline_field", "delta"),
    [
        ("M0_loss", 0.01),
        ("M3_loss", -0.01),
        ("relative_MSE_gain_M3_vs_M0", 0.01),
    ],
)
def test_stale_or_manipulated_headline_means_block_as_indeterminate(
    headline_field, delta
):
    experiments = _effect_gate_experiments()
    experiments["program_total::natural"]["headline"][headline_field] += delta
    decision = runner._effect_size_consistency_decision(experiments)
    assert decision["status"] == "blocked_fail_closed"
    assert decision["decision"] == "blocked_indeterminate"
    assert decision["supported"] is False


def test_effect_gate_requires_exact_frozen_donor_set_and_count():
    experiments = _effect_gate_experiments()
    per_donor = experiments["pca_total::natural"]["headline"]["per_donor"]
    per_donor["unexpected"] = per_donor.pop(runner.EXPECTED_PRIMARY_DONOR_IDS[0])
    decision = runner._effect_size_consistency_decision(experiments)
    assert decision["status"] == "blocked_fail_closed"
    assert decision["decision"] == "blocked_indeterminate"
    assert decision["supported"] is False


def test_missing_effect_size_evidence_blocks_overall_as_indeterminate():
    experiments = _effect_gate_experiments()
    del experiments["program_total::natural"]["headline"][
        "positive_donor_fraction_M3_vs_M0"
    ]
    effect = runner._effect_size_consistency_decision(experiments)
    assert effect["status"] == "blocked_fail_closed"
    assert effect["decision"] == "blocked_indeterminate"
    controls = {"status": "evaluable", "supported": True}
    floor = {"status": "evaluable", "supported": True}
    overall = runner._overall_scientific_decision(controls, floor, effect)
    assert overall["decision"] == "blocked_indeterminate"
    assert overall["supported"] is False


def test_failed_effect_size_gate_prevents_support_when_other_families_pass():
    effect = runner._effect_size_consistency_decision(
        _effect_gate_experiments(relative_gain=0.04)
    )
    controls = {"status": "evaluable", "supported": True}
    floor = {"status": "evaluable", "supported": True}
    overall = runner._overall_scientific_decision(controls, floor, effect)
    assert overall["status"] == "evaluable"
    assert overall["decision"] == "not_supported"
    assert overall["supported"] is False
