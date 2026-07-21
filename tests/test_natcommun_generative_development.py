from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "scripts/benchmark_natcommun_generative_development.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_natcommun_generative_development", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _csr_payload(prefix: str, matrix: np.ndarray) -> dict[str, np.ndarray]:
    rows, columns = np.nonzero(matrix)
    order = np.lexsort((columns, rows))
    rows, columns = rows[order], columns[order]
    data = matrix[rows, columns]
    indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr)
    return {
        f"{prefix}_data": data.astype(np.int32),
        f"{prefix}_indices": columns.astype(np.int32),
        f"{prefix}_indptr": indptr,
        f"{prefix}_shape": np.asarray(matrix.shape, dtype=np.int64),
    }


def _write_synthetic_source(path: Path, *, encoder: str = runner.HOPTIMUS_REPOSITORY) -> None:
    donors = np.repeat(np.asarray(["A", "B", "C"]), 3)
    spot_ids = np.asarray([f"spot-{index}" for index in range(len(donors))])
    sections = np.asarray([f"section-{donor}" for donor in donors])
    indications = np.repeat("same_indication", len(donors))
    genes = np.asarray(["G1", "G2", "G3", "G4"])
    counts = np.asarray(
        [
            [3, 0, 1, 0],
            [0, 0, 0, 0],  # explicit zero-depth score exclusion
            [1, 2, 0, 1],
            [0, 2, 3, 0],
            [2, 1, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 4, 0],
            [0, 3, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    st_library = counts.sum(axis=1).astype(np.float64)
    # Four cells per donor and two coarse types give each reference a distribution.
    sc_donors = np.repeat(np.asarray(["A", "B", "C"]), 4)
    # Donor-distinct pseudobulks make matched-vs-mismatched assay alignment
    # identifiable in every two-training-donor synthetic outer fold.
    sc_counts = np.asarray(
        [
            [4, 1, 1, 0],
            [4, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [2, 1, 2, 0],
            [2, 1, 2, 0],
            [0, 2, 1, 1],
            [0, 2, 1, 1],
            [1, 1, 3, 0],
            [1, 1, 3, 0],
            [0, 2, 2, 1],
            [0, 2, 2, 1],
        ],
        dtype=np.int32,
    )
    receipt = {
        "encoder": {
            "repository": encoder,
            "revision": "synthetic",
            "fine_tuning": "none",
        },
        "encoder_roles": {
            "primary": {"repository": encoder},
            "secondary_comparator": {
                "repository": "MahmoodLab/UNI2-h",
                "status": "prespecified_not_run_in_primary_source",
            },
        },
    }
    payload: dict[str, object] = {
        "schema_version": np.asarray("synthetic"),
        "source_receipt_json": np.asarray(json.dumps(receipt)),
        "broad_gene_ids": genes,
        "spot_ids": spot_ids,
        "spot_primary_eligible": np.ones(len(donors), dtype=bool),
        "donor_ids": donors,
        "section_ids": sections,
        "indication_ids": indications,
        "image_features": np.arange(len(donors) * 6, dtype=np.float32).reshape(len(donors), 6),
        "coordinate_features": np.tile(
            np.asarray([[0.0, 1.0]], dtype=np.float32), (len(donors), 1)
        ),
        "blank_image_feature_vector": np.zeros(6, dtype=np.float32),
        "st_total_umi_counts_full": st_library,
        "sc_primary_eligible": np.ones(len(sc_donors), dtype=bool),
        "sc_donor_ids": sc_donors,
        "sc_total_umi_counts": sc_counts.sum(axis=1).astype(np.float64),
        "sc_cell_ids": np.asarray([f"cell-{index}" for index in range(len(sc_donors))]),
        "sc_indication_ids": np.repeat("same_indication", len(sc_donors)),
        "sc_level1_type_ids": np.tile(np.asarray(["T1", "T1", "T2", "T2"]), 3),
        **_csr_payload("st_broad_counts_full", counts),
        **_csr_payload("sc_broad_counts", sc_counts),
    }
    np.savez_compressed(path, **payload)


def _write_panel(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "synthetic",
                "scope": "exposed_development_only",
                "selected_gene_ids": ["G1", "G2", "G3", "G4"],
            }
        ),
        encoding="utf-8",
    )


def _write_protocol(path: Path, gene_count: int = 4) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "synthetic",
                "analysis_status": "exposed_development_only_non_confirmatory",
                "gene_panel_size": gene_count,
                "latent_dimensions": 3,
                "immutable_inputs": {},
                "encoders": {"UNI2_h": "forbidden_not_run"},
                "image_inputs": {
                    "primary": "natural_registered_112um_H_optimus_1",
                    "held_out_ST_may_route_image_query": False,
                },
                "outer_validation": {"held_out_donor_ST_use": "endpoint_scoring_only"},
                "generative_model": {
                    "M0_M3_capacity_rule": "same_H_and_E_query_encoder_and_same_ST_decoder",
                    "iterative_updates": 0,
                },
                "resource_limits": {
                    "maximum_CPU_threads": 4,
                    "maximum_projected_counts_GiB": 1,
                },
                "claim_boundaries": {"cell_level_claims": "prohibited"},
                "M8_split_policy": {"Poisson_split_assumption": "prohibited_under_overdispersion"},
            }
        ),
        encoding="utf-8",
    )


def _args(source: Path, panel: Path, output: Path) -> argparse.Namespace:
    protocol = output.parent / "protocol.json"
    _write_protocol(protocol)
    return argparse.Namespace(
        source=source,
        projected_source=None,
        expected_projected_source_sha256=None,
        panel=panel,
        protocol=protocol,
        output=output,
        expected_source_sha256=None,
        smoke=True,
        seed=2718,
    )


def test_prepare_physically_separates_heldout_st_and_is_deterministic(tmp_path: Path) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    output = tmp_path / "run"

    first = runner.prepare(_args(source, panel, output))
    second = runner.prepare(_args(source, panel, output))

    assert first["donors"] == ["A", "B", "C"]
    assert first["uni2_h_run"] is False
    for donor in first["donors"]:
        one = first["folds"][donor]
        two = second["folds"][donor]
        assert one["public_semantic_sha256"] == two["public_semantic_sha256"]
        assert one["score_target_semantic_sha256"] == two["score_target_semantic_sha256"]
        public = runner._load_arrays(Path(one["public_path"]))
        secret = runner._load_arrays(Path(one["score_target_path"]))
        runner.validate_public_fold(public)
        assert not any(name.startswith("heldout_st") for name in public)
        assert "heldout_st_counts" in secret
        assert set(public["matched_sc_donor_ids"].astype(str)) == {donor}
        assert donor not in set(public["train_sc_donor_ids"].astype(str))


def test_zero_depth_rows_are_removed_from_fit_and_flagged_only_for_score(tmp_path: Path) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    manifest = runner.prepare(_args(source, panel, tmp_path / "run"))

    for donor, receipt in manifest["folds"].items():
        public = runner._load_arrays(Path(receipt["public_path"]))
        secret = runner._load_arrays(Path(receipt["score_target_path"]))
        assert np.all(public["train_st_library"] > 0)
        assert int(secret["zero_depth_excluded_count"]) == 1
        assert secret["primary_score_eligible"].tolist().count(False) == 1
        assert runner._scalar_text(secret["heldout_donor"]) == donor


def test_public_fold_guard_rejects_target_and_donor_leakage(tmp_path: Path) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    manifest = runner.prepare(_args(source, panel, tmp_path / "run"))
    receipt = manifest["folds"]["A"]
    public = runner._load_arrays(Path(receipt["public_path"]))

    target_leak = dict(public)
    target_leak["heldout_st_counts"] = np.ones((1, 4))
    with pytest.raises(ValueError, match="held-out ST leaked"):
        runner.validate_public_fold(target_leak)

    donor_leak = dict(public)
    donor_leak["train_sc_donor_ids"] = np.append(donor_leak["train_sc_donor_ids"], "A")
    with pytest.raises(ValueError, match="held-out donor leaked"):
        runner.validate_public_fold(donor_leak)


@pytest.mark.parametrize("encoder", ["MahmoodLab/UNI2-h", "UNI2_h", "uni2"])
def test_uni2_is_rejected_explicitly(tmp_path: Path, encoder: str) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source, encoder=encoder)
    _write_panel(panel)
    with pytest.raises(ValueError, match="UNI2-h is explicitly prohibited"):
        runner.prepare(_args(source, panel, tmp_path / "run"))


def test_checkpoint_identity_captures_fold_and_hyperparameters() -> None:
    kwargs = {
        "donor": "A",
        "seed": 1,
        "epochs": 3,
        "latent_dim": 20,
        "batch_size": 64,
        "device": "cuda:0",
        "runner_sha256": "b" * 64,
        "core_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
    }
    base = runner._checkpoint_identity("a" * 64, **kwargs)
    assert base == runner._checkpoint_identity("a" * 64, **kwargs)
    for field, value in (
        ("donor", "B"),
        ("epochs", 4),
        ("batch_size", 32),
        ("runner_sha256", "e" * 64),
        ("core_sha256", "f" * 64),
        ("protocol_sha256", "0" * 64),
    ):
        changed = {**kwargs, field: value}
        assert base != runner._checkpoint_identity("a" * 64, **changed)


def test_frozen_protocol_rejects_panel_semantic_replacement() -> None:
    root = Path(__file__).parents[1]
    panel_path = root / "configs/natcommun_generative_gene_panel.json"
    protocol_path = root / "configs/natcommun_generative_development_protocol.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    changed = {**panel, "artifact_sha256": "0" * 64}
    with pytest.raises(ValueError, match="semantic identity"):
        runner.load_protocol(
            protocol_path,
            source_path=Path(protocol["immutable_inputs"]["source_path"]),
            source_sha256=protocol["immutable_inputs"]["source_sha256"],
            panel_path=panel_path,
            panel_payload=changed,
            gene_count=256,
            smoke=False,
        )


def test_prepared_seed_is_immutable_for_fit_and_score(tmp_path: Path) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    args = _args(source, panel, tmp_path / "run")
    runner.prepare(args)
    args.seed += 1
    with pytest.raises(ValueError, match="prepared seed"):
        runner.fit_predict(args)


def test_m0_type_vocabulary_does_not_depend_on_matched_bank_support(tmp_path: Path) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    args = _args(source, panel, tmp_path / "run")
    manifest = runner.prepare(args)
    public = runner._load_arrays(Path(manifest["folds"]["A"]["public_path"]))
    keep = np.asarray(public["matched_sc_type_ids"]).astype(str) == "T1"
    for name in (
        "matched_sc_counts",
        "matched_sc_library",
        "matched_sc_cell_ids",
        "matched_sc_donor_ids",
        "matched_sc_indication_ids",
        "matched_sc_type_ids",
    ):
        public[name] = np.asarray(public[name])[keep]
    prediction = runner.fit_predict_one_fold(
        public,
        device="cpu",
        epochs=8,
        batch_size=16,
        latent_dim=3,
        seed=123,
    )
    assert prediction["reference_model_type_names"].astype(str).tolist() == ["T1", "T2"]
    assert prediction["matched_observed_type_names"].astype(str).tolist() == ["T1"]
    assert np.isfinite(prediction["rate_M3"]).all()


def test_composition_proxy_is_training_only_deterministic_and_on_simplex() -> None:
    st = np.asarray([[8, 1, 0, 0], [0, 1, 7, 1]], dtype=float)
    sc = np.asarray(
        [[6, 1, 0, 0], [5, 2, 0, 0], [0, 0, 5, 2], [0, 1, 6, 1]],
        dtype=float,
    )
    types = np.asarray(["T1", "T1", "T2", "T2"])

    first, order = runner.training_only_composition_proxy(st, sc, types)
    second, second_order = runner.training_only_composition_proxy(st, sc, types)

    assert order == second_order == ("T1", "T2")
    assert np.array_equal(first, second)
    assert np.allclose(first.sum(axis=1), 1.0)
    assert np.all(first > 0)
    assert first[0, 0] > first[0, 1]
    assert first[1, 1] > first[1, 0]


def test_resource_limits_fail_closed() -> None:
    with pytest.raises(ValueError, match="CPU threads"):
        runner.configure_resources(cpu_threads=5, gpu_memory_fraction=0.6, device="cpu")
    with pytest.raises(ValueError, match="GPU memory"):
        runner.configure_resources(cpu_threads=4, gpu_memory_fraction=0.61, device="cpu")


@pytest.mark.parametrize(
    ("field", "value"),
    (("seed", 1730), ("epochs", 79), ("batch_size", 128), ("latent_dim", 19)),
)
def test_real_benchmark_hyperparameters_are_frozen(field: str, value: int) -> None:
    args = argparse.Namespace(
        smoke=False,
        device="cuda:0",
        seed=runner.FROZEN_BASE_SEED,
        epochs=runner.FROZEN_EPOCHS,
        batch_size=runner.FROZEN_BATCH_SIZE,
        latent_dim=runner.FROZEN_LATENT_DIM,
    )
    setattr(args, field, value)
    with pytest.raises(ValueError, match="frozen protocol|frozen secondary"):
        runner._validate_args(args)


def test_all_stage_cpu_smoke_preserves_scoring_boundary(tmp_path: Path) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    args = _args(source, panel, tmp_path / "run")
    args.device = "cpu"
    args.epochs = 8
    args.batch_size = 16
    args.latent_dim = 3
    args.resume = False
    args.cpu_threads = 1
    args.gpu_memory_fraction = 0.6

    runner.configure_resources(cpu_threads=1, gpu_memory_fraction=0.6, device="cpu")
    runner.prepare(args)
    fitted = runner.fit_predict(args)
    report = runner.score(args)

    assert fitted["all_folds_complete"] is True
    assert report["ordered_gates"]["decision"]["status"].startswith("not_evaluable")
    assert report["implementation_status"]["gate_attribution_allowed"] is False
    for donor in ("A", "B", "C"):
        prediction = runner._load_arrays(args.output / "folds" / donor / "predictions.npz")
        assert "rate_M0" in prediction and "rate_M3" in prediction
        assert "posterior_rate_variance_M0" in prediction
        assert "posterior_rate_variance_M3" in prediction
        assert np.all(prediction["posterior_rate_variance_M3"] >= 0)
        assert np.any(prediction["posterior_rate_variance_M3"] > 0)
        assert runner._scalar_text(prediction["posterior_decode_representation"]) == (
            "componentwise_reference_PoE_no_moment_collapse_before_decoder"
        )
        assert "BLEEP_retrieval_entropy" in prediction
        assert "M3_reference_entropy_normalized" in prediction
        assert "retrieval_entropy" not in prediction
        assert float(prediction["cross_assay_alignment_weight"]) == 1.0
        pre_ratio = float(
            prediction["cross_assay_alignment_pre_matched_to_mismatched_ratio"]
        )
        post_ratio = float(
            prediction["cross_assay_alignment_post_matched_to_mismatched_ratio"]
        )
        unaligned_ratio = float(
            prediction["cross_assay_unaligned_post_matched_to_mismatched_ratio"]
        )
        applications = int(
            prediction["cross_assay_alignment_optimizer_applications_per_epoch"]
        )
        assert bool(prediction["cross_assay_alignment_support_criterion_met"])
        assert bool(prediction["cross_assay_alignment_beats_unaligned_comparator"])
        assert 0 <= post_ratio < min(1.0, pre_ratio, unaligned_ratio)
        assert applications > 0
        assert int(prediction["cross_assay_alignment_optimizer_applications_total"]) == (
            applications * args.epochs
        )
        assert "heldout_st_counts" not in prediction
        assert runner._scalar_text(prediction["prediction_scale"]) == (
            "per_unit_actual_ST_library_rate"
        )
        fold = report["folds"][donor]
        assert fold["m6_wrong_reference_control"]["primary"] == (
            "equal_mean_over_eligible_wrong_donor_losses"
        )
        assert fold["m8_same_target"]["predictor"] == (
            "training_donor_fitted_cross_half_molecular_ridge"
        )
        assert fold["m8_same_target"]["zero_split_depth_excluded"] == 0
        assert fold["m8_same_target"]["scoring_dispersion_fraction_of_full_theta"] == 0.5
        assert fold["cross_assay_alignment"]["heldout_ST_used"] is False
        assert fold["cross_assay_alignment"]["support_criterion_met"] is True
        assert fold["cross_assay_alignment"]["beats_unaligned_comparator"] is True
        assert fold["H_state_variance_calibration"]["method"] == (
            "Gaussian_training_residual_likelihood"
        )
    assert "donor_balanced_full_target_mean_nb_log_likelihood" in report
    assert "donor_balanced_M8_same_half_target_mean_nb_log_likelihood" in report
    assert "donor_balanced_BLEEP_retrieval_entropy" in report
    alignment = report["donor_balanced_cross_assay_alignment"]
    assert alignment["all_folds_support_criterion"] is True
    assert alignment["all_folds_beat_unaligned_comparator"] is True
    quality_guard = report["ordered_gates"]["quality_guard"]
    if "checks" in quality_guard:
        assert quality_guard["checks"]["cross_assay_alignment"]["passed"] is True
    else:
        assert not any(
            "cross_assay" in value for value in quality_guard.get("missing_metrics", ())
        )
    assert set(report["donor_balanced_full_target_mean_nb_deviance"]) == set(
        runner.MODEL_ARMS
    )
    assert set(report["donor_balanced_full_target_mean_nb_log_likelihood"]) == set(
        runner.MODEL_ARMS
    )
    assert set(report["donor_balanced_M8_same_half_target_mean_nb_deviance"]) == {
        "M3_same_M8_target",
        "M8",
    }
    assert set(report["donor_balanced_M8_same_half_target_mean_nb_log_likelihood"]) == {
        "M3_same_M8_target",
        "M8",
    }


def test_score_rejects_stale_prediction_receipt_after_reprepare(tmp_path: Path) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    args = _args(source, panel, tmp_path / "run")
    args.device = "cpu"
    args.epochs = 8
    args.batch_size = 16
    args.latent_dim = 3
    args.resume = False
    args.cpu_threads = 1
    args.gpu_memory_fraction = 0.6

    runner.configure_resources(cpu_threads=1, gpu_memory_fraction=0.6, device="cpu")
    runner.prepare(args)
    runner.fit_predict(args)
    manifest_path = args.output / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"]["A"]["public_semantic_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stale prediction receipt"):
        runner.score(args)


def test_score_verifies_all_predictions_before_opening_any_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    args = _args(source, panel, tmp_path / "run")
    args.device = "cpu"
    args.epochs = 8
    args.batch_size = 16
    args.latent_dim = 3
    args.resume = False
    args.cpu_threads = 1
    args.gpu_memory_fraction = 0.6

    runner.configure_resources(cpu_threads=1, gpu_memory_fraction=0.6, device="cpu")
    runner.prepare(args)
    runner.fit_predict(args)
    prediction_path = args.output / "folds" / "B" / "predictions.npz"
    corrupted = runner._load_arrays(prediction_path)
    corrupted["rate_M0"] = np.asarray(corrupted["rate_M0"]) + 1.0
    runner._atomic_npz(prediction_path, corrupted)

    opened: list[Path] = []
    original_verify = runner._verify_semantic_file

    def recording_verify(path: Path, expected: str):
        opened.append(Path(path))
        return original_verify(path, expected)

    monkeypatch.setattr(runner, "_verify_semantic_file", recording_verify)
    with pytest.raises(ValueError, match="semantic identity mismatch"):
        runner.score(args)
    assert not any(path.name == "score_target.npz" for path in opened)


def test_score_rejects_rehashed_incomplete_prediction_before_any_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    args = _args(source, panel, tmp_path / "run")
    args.device = "cpu"
    args.epochs = 8
    args.batch_size = 16
    args.latent_dim = 3
    args.resume = False
    args.cpu_threads = 1
    args.gpu_memory_fraction = 0.6

    runner.configure_resources(cpu_threads=1, gpu_memory_fraction=0.6, device="cpu")
    runner.prepare(args)
    runner.fit_predict(args)
    prediction_path = args.output / "folds" / "B" / "predictions.npz"
    incomplete = runner._load_arrays(prediction_path)
    incomplete.pop("rate_M3")
    runner._atomic_npz(prediction_path, incomplete)
    manifest_path = args.output / "fit_predict_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"]["B"]["prediction_semantic_sha256"] = runner._semantic_array_hash(
        incomplete
    )
    runner._atomic_json(manifest_path, manifest)

    opened: list[Path] = []
    original_verify = runner._verify_semantic_file

    def recording_verify(path: Path, expected: str):
        opened.append(Path(path))
        return original_verify(path, expected)

    monkeypatch.setattr(runner, "_verify_semantic_file", recording_verify)
    with pytest.raises(ValueError, match="prediction artifact is incomplete"):
        runner.score(args)
    assert not any(path.name == "score_target.npz" for path in opened)


def test_immediate_prediction_revalidation_closes_post_preflight_mutation_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, panel = tmp_path / "source.npz", tmp_path / "panel.json"
    _write_synthetic_source(source)
    _write_panel(panel)
    args = _args(source, panel, tmp_path / "run")
    args.device = "cpu"
    args.epochs = 8
    args.batch_size = 16
    args.latent_dim = 3
    args.resume = False
    args.cpu_threads = 1
    args.gpu_memory_fraction = 0.6

    runner.configure_resources(cpu_threads=1, gpu_memory_fraction=0.6, device="cpu")
    runner.prepare(args)
    runner.fit_predict(args)
    original_preflight = runner._validate_prediction_manifest_binding

    def mutate_after_preflight(
        local_args: argparse.Namespace,
        core: object,
        prepared: dict[str, object],
        prediction_manifest: dict[str, object],
    ) -> None:
        original_preflight(local_args, core, prepared, prediction_manifest)
        prediction_path = local_args.output / "folds" / "A" / "predictions.npz"
        incomplete = runner._load_arrays(prediction_path)
        incomplete.pop("rate_M3")
        runner._atomic_npz(prediction_path, incomplete)
        prediction_manifest["folds"]["A"]["prediction_semantic_sha256"] = (
            runner._semantic_array_hash(incomplete)
        )

    opened: list[Path] = []
    original_verify = runner._verify_semantic_file

    def recording_verify(path: Path, expected: str):
        opened.append(Path(path))
        return original_verify(path, expected)

    monkeypatch.setattr(runner, "_validate_prediction_manifest_binding", mutate_after_preflight)
    monkeypatch.setattr(runner, "_verify_semantic_file", recording_verify)
    with pytest.raises(ValueError, match="prediction artifact is incomplete"):
        runner.score(args)
    assert not any(path.name == "score_target.npz" for path in opened)


def test_gate_1_requires_exact_significance_even_when_effect_margins_pass() -> None:
    per_donor = {
        donor: {
            "M0": 10.0,
            "M1": 10.0,
            "M2": 10.0,
            "M2_supported": 10.0,
            "M3": 9.0,
            "M3_supported": 9.0,
            "M4": 10.0,
            "M6": 10.0,
            "M7": 10.0,
            "BLEEP": 10.0,
            "M8": 8.0,
            "M3_same_M8_target": 9.0,
        }
        for donor in ("A", "B", "C")
    }
    gates = runner._evaluate_gates(
        runner._import_core(),
        per_donor,
        {donor: "one_indication" for donor in per_donor},
        seed=2718,
    )

    gate_1 = gates["decision"]["gate_1"]
    assert gate_1["relative_gain"] == pytest.approx(0.10)
    assert gate_1["donor_fraction_improved"] == 1.0
    assert gate_1["confidence_interval"][0] > 0
    assert gate_1["exact_sign_flip_p_value"] == pytest.approx(0.125)
    assert gate_1["passed"] is False
    assert gates["decision"]["central_development_gate_passed"] is False
    assert gates["decision"]["stopped_at"] == "gate_1"


def test_gate_5_uses_only_the_same_half_target_and_core_crosscheck_is_sanitized() -> None:
    per_donor = {
        donor: {
            "M0": 2.0,
            "M1": 2.0,
            # Full-target M2 has the opposite direction and is descriptive only.
            "M2": 0.5,
            "M2_supported": 2.0,
            "M3": 1.0,
            "M3_supported": 1.0,
            "M4": 2.0,
            "M6": 2.0,
            "M7": 2.0,
            "BLEEP": 2.0,
            # Full-target M3 would incorrectly imply the opposite result.
            "M8": 2.0,
            "M3_same_M8_target": 3.0,
        }
        for donor in tuple("ABCDEFGH")
    }
    gates = runner._evaluate_gates(
        runner._import_core(),
        per_donor,
        {donor: "one_indication" for donor in per_donor},
        seed=2718,
    )

    comparison = gates["comparisons"]["M8_vs_M3"]
    assert comparison["mean_improvement"] == pytest.approx(1.0)
    assert gates["decision"]["gate_5"]["same_split_target"] is True
    assert gates["decision"]["gate_5"]["passed"] is True
    assert gates["comparisons"]["M3_vs_M2_full_mixed_support_descriptive"][
        "mean_improvement"
    ] == pytest.approx(-0.5)
    assert gates["comparisons"]["M3_supported_vs_M2_supported"][
        "mean_improvement"
    ] == pytest.approx(1.0)
    assert gates["decision"]["gate_3"]["passed"] is True
    crosscheck = gates["core_gate_1_to_2_sanitized_crosscheck"]
    assert "molecular_headroom_detected" not in crosscheck
    assert "personalization_supported" not in crosscheck
    assert "full_model_chain_supported" not in crosscheck
    assert "conditional_gate3" in crosscheck["scope"]
    assert all(
        gate.get("name")
        not in {
            "gate_3_state_beyond_routing",
            "gate_4_matching_specificity",
            "gate_5_molecular_headroom",
        }
        for gate in crosscheck["gates"]
    )
