from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heir.evaluation.gene_panel import (
    GenePanelSelection,
    canonical_sha256,
    panel_artifact,
)


def _load_validator():
    path = Path(__file__).parents[1] / "scripts/validate_natcommun_fold_local_panels.py"
    spec = importlib.util.spec_from_file_location("validate_natcommun_fold_local_panels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()
runner = validator._load_runner()


def _panel(
    donor: str,
    genes: tuple[str, ...],
    training: tuple[str, ...],
    columns: dict[str, int],
):
    selection = GenePanelSelection(
        gene_ids=genes,
        broad_column_indices=tuple(columns[gene] for gene in genes),
        scores=tuple(float(index) for index in range(len(genes))),
        training_donor_ids=training,
        held_out_donor_id=donor,
        eligible_gene_count=len(genes),
        retained_program_genes=(),
        minimum_split_reliability=0.05,
        metrics_by_gene={gene: {} for gene in genes},
    )
    return panel_artifact(
        selection,
        source_sha256="a" * 64,
        source_path="/sealed/source.npz",
        mode="lodo_fold_local",
        program_gene_source="synthetic_test",
    )


def _panel_set(*, leaked_a: bool = False):
    donors = ("A", "B", "C")
    columns = {"G1": 4, "G2": 2, "G3": 9}
    panels = {
        "A": _panel(
            "A",
            ("G2", "G1"),
            (("A", "B", "C") if leaked_a else ("B", "C")),
            columns,
        ),
        "B": _panel("B", ("G3", "G2"), ("A", "C"), columns),
        "C": _panel("C", ("G1", "G3"), ("A", "B"), columns),
    }
    value = {
        "schema": "heir.natcommun_generative_gene_panel_set.v1",
        "analysis_status": "exposed_development_only_non_confirmatory",
        "scope": "exposed_development_only_non_confirmatory",
        "source": {"path": "/sealed/source.npz", "sha256": "a" * 64},
        "panel_size": 2,
        "lodo_fold_local_panels": panels,
    }
    value["artifact_sha256"] = canonical_sha256(value)
    return value, donors


def test_lodo_receipts_define_training_only_panels_and_one_stable_union() -> None:
    payload, donors = _panel_set()
    panels, genes, columns = validator._validate_panel_set(
        payload,
        expected_source_sha256="a" * 64,
        expected_donors=donors,
        expected_panel_size=2,
        expected_union_size=3,
    )
    assert tuple(panels) == donors
    assert genes == ("G1", "G2", "G3")
    assert columns == (4, 2, 9)
    assert all(
        set(panel["selection"]["training_donor_ids"]) == set(donors) - {donor}
        for donor, panel in panels.items()
    )


def test_lodo_receipts_reject_heldout_donor_in_panel_selection() -> None:
    payload, donors = _panel_set(leaked_a=True)
    with pytest.raises(ValueError, match="other donors only"):
        validator._validate_panel_set(
            payload,
            expected_source_sha256="a" * 64,
            expected_donors=donors,
            expected_panel_size=2,
            expected_union_size=3,
        )


def test_union_slice_preserves_each_fold_panel_order() -> None:
    source = runner.SourceArrays(
        spot_ids=np.asarray(["S1", "S2"]),
        donor_ids=np.asarray(["A", "B"]),
        section_ids=np.asarray(["A1", "B1"]),
        indication_ids=np.asarray(["x", "x"]),
        image=np.ones((2, 4), dtype=np.float32),
        coordinates=np.ones((2, 2), dtype=np.float32),
        blank_image=np.zeros(4, dtype=np.float32),
        st_counts=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        st_library=np.asarray([6, 15], dtype=np.float64),
        st_half_a=np.asarray([[1, 0, 2], [2, 3, 1]], dtype=np.float32),
        st_half_b=np.asarray([[0, 2, 1], [2, 2, 5]], dtype=np.float32),
        st_library_half_a=np.asarray([3, 6], dtype=np.float64),
        st_library_half_b=np.asarray([3, 9], dtype=np.float64),
        sc_counts=np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float32),
        sc_library=np.asarray([24, 33], dtype=np.float64),
        sc_cell_ids=np.asarray(["C1", "C2"]),
        sc_donor_ids=np.asarray(["A", "B"]),
        sc_indication_ids=np.asarray(["x", "x"]),
        sc_type_ids=np.asarray(["T1", "T2"]),
        gene_ids=np.asarray(["G1", "G2", "G3"]),
        program_names=np.asarray(["P1", "P2"]),
        program_gene_membership=np.asarray([[True, False, True], [False, True, False]], dtype=bool),
        source_receipt={"encoder": {"repository": "bioptimus/H-optimus-1"}},
    )
    sliced = validator._slice_source(runner, source, ("G3", "G1"))
    assert sliced.gene_ids.tolist() == ["G3", "G1"]
    np.testing.assert_array_equal(sliced.st_counts, np.asarray([[3, 1], [6, 4]]))
    np.testing.assert_array_equal(sliced.sc_counts, np.asarray([[9, 7], [12, 10]]))
    np.testing.assert_array_equal(
        sliced.program_gene_membership,
        np.asarray([[True, True], [False, False]], dtype=bool),
    )


def test_comparison_effect_is_baseline_loss_minus_fusion_loss() -> None:
    class FakeRunner:
        @staticmethod
        def _donor_bootstrap_interval(values, *, seed):
            assert seed == 7
            return float(np.min(values)), float(np.max(values))

        @staticmethod
        def _sign_flip(core, values):
            assert core == "core"
            return {"observed": float(np.mean(values)), "p_value": 0.25}

    per_donor = {
        "A": {"M0": 3.0, "M3": 2.0},
        "B": {"M0": 2.0, "M3": 2.5},
    }
    result = validator._comparison_summary(
        FakeRunner,
        "core",
        per_donor,
        baseline_arm="M0",
        model_arm="M3",
        seed=7,
    )
    assert result["donor_improvement"] == [1.0, -0.5]
    assert result["mean_improvement"] == pytest.approx(0.25)
    assert result["donor_fraction_improved"] == 0.5


def _prediction_identity_inputs():
    manifest = {
        "schema": validator.PREPARED_SCHEMA,
        "manifest_path": "/sealed/prepared_manifest.json",
        "smoke": False,
        "training_configuration": {"base_seed": 1729, "cpu_threads": 4},
        "frozen_runner_sha256": "runner",
        "frozen_core_sha256": "core",
        "development_protocol_sha256": "development",
        "validation_protocol_sha256": "validation",
        "orchestrator_sha256": "orchestrator",
    }
    fold = {
        "seed": 11,
        "public_file_sha256": "public-file",
        "public_semantic_sha256": "public",
        "panel_artifact_sha256": "panel-artifact",
        "panel_identity_sha256": "panel",
        "panel_gene_ids": ["g1", "g2"],
    }
    return manifest, fold


def test_prediction_identity_binds_seed_public_panel_and_resources_not_target() -> None:
    manifest, fold = _prediction_identity_inputs()
    identity = validator._prediction_identity(manifest, fold, "A")
    changed_seed = dict(fold, seed=12)
    changed_public = dict(fold, public_semantic_sha256="other")
    changed_resources = copy.deepcopy(manifest)
    changed_resources["training_configuration"]["cpu_threads"] = 3
    assert validator._prediction_identity(manifest, changed_seed, "A") != identity
    assert validator._prediction_identity(manifest, changed_public, "A") != identity
    assert validator._prediction_identity(changed_resources, fold, "A") != identity
    with pytest.raises(ValueError, match="prohibited authority"):
        validator._assert_fit_manifest_target_free(
            {"folds": {"A": {**fold, "score_target_path": "/sealed/target.npz"}}}
        )


class _TargetRunner:
    PREPARED_SCHEMA = "prepared"

    @staticmethod
    def _scalar_text(value):
        return str(np.asarray(value).item())


def test_fold_local_target_identity_rejects_reordered_sections() -> None:
    public = {
        "heldout_donor": np.asarray("A"),
        "query_spot_ids": np.asarray(["s1", "s2"]),
        "query_section_ids": np.asarray(["x", "y"]),
        "query_indication_ids": np.asarray(["i", "i"]),
        "gene_ids": np.asarray(["g1", "g2"]),
    }
    full = np.asarray([[1, 1], [0, 0]], dtype=np.float32)
    secret = {
        "schema": np.asarray("prepared"),
        "heldout_donor": np.asarray("A"),
        "heldout_spot_ids": public["query_spot_ids"].copy(),
        "heldout_section_ids": np.asarray(["y", "x"]),
        "heldout_indication_ids": public["query_indication_ids"].copy(),
        "gene_ids": public["gene_ids"].copy(),
        "heldout_st_counts": full,
        "heldout_st_half_a": full.copy(),
        "heldout_st_half_b": np.zeros_like(full),
        "heldout_st_library": np.asarray([2, 0], dtype=np.float32),
        "heldout_st_library_half_a": np.asarray([2, 0], dtype=np.float32),
        "heldout_st_library_half_b": np.asarray([0, 0], dtype=np.float32),
        "primary_score_eligible": np.asarray([True, False]),
        "zero_depth_excluded_count": np.asarray(1),
    }
    predictions = {
        "heldout_donor": np.asarray("A"),
        "query_spot_ids": public["query_spot_ids"].copy(),
        "gene_ids": public["gene_ids"].copy(),
    }
    with pytest.raises(ValueError, match="heldout_section_ids"):
        validator._validate_score_target_identity(
            _TargetRunner, secret, public, predictions, donor="A"
        )


def test_tampered_panel_artifact_is_rejected_before_union_construction() -> None:
    payload, donors = _panel_set()
    tampered = copy.deepcopy(payload)
    tampered["lodo_fold_local_panels"]["A"]["gene_ids"][0] = "CHANGED"
    top = dict(tampered)
    top.pop("artifact_sha256")
    tampered["artifact_sha256"] = canonical_sha256(top)
    with pytest.raises(ValueError, match="panel identity|identity hash"):
        validator._validate_panel_set(
            tampered,
            expected_source_sha256="a" * 64,
            expected_donors=donors,
            expected_panel_size=2,
            expected_union_size=3,
        )


def test_target_identity_requires_exact_target_public_prediction_gene_axis() -> None:
    public = {
        "heldout_donor": np.asarray("A"),
        "query_spot_ids": np.asarray(["s1"]),
        "query_section_ids": np.asarray(["x"]),
        "query_indication_ids": np.asarray(["i"]),
        "gene_ids": np.asarray(["g1", "g2"]),
    }
    secret = {
        "schema": np.asarray("prepared"),
        "heldout_donor": np.asarray("A"),
        "heldout_spot_ids": public["query_spot_ids"].copy(),
        "heldout_section_ids": public["query_section_ids"].copy(),
        "heldout_indication_ids": public["query_indication_ids"].copy(),
        "gene_ids": np.asarray(["g2", "g1"]),
        "heldout_st_counts": np.asarray([[1, 1]], dtype=np.float32),
        "heldout_st_half_a": np.asarray([[1, 0]], dtype=np.float32),
        "heldout_st_half_b": np.asarray([[0, 1]], dtype=np.float32),
        "heldout_st_library": np.asarray([2], dtype=np.float32),
        "heldout_st_library_half_a": np.asarray([1], dtype=np.float32),
        "heldout_st_library_half_b": np.asarray([1], dtype=np.float32),
        "primary_score_eligible": np.asarray([True]),
        "zero_depth_excluded_count": np.asarray(0),
    }
    predictions = {
        "heldout_donor": np.asarray("A"),
        "query_spot_ids": public["query_spot_ids"].copy(),
        "gene_ids": public["gene_ids"].copy(),
    }
    with pytest.raises(ValueError, match="gene axis differs"):
        validator._validate_score_target_identity(
            _TargetRunner, secret, public, predictions, donor="A"
        )


def test_prediction_receipt_binds_canonical_paths_fit_and_resources(tmp_path: Path) -> None:
    manifest, fold = _prediction_identity_inputs()
    manifest["smoke"] = True
    args = SimpleNamespace(
        output=tmp_path,
        device="cuda:0",
        cpu_threads=4,
        gpu_memory_fraction=0.6,
        epochs=80,
        batch_size=256,
        latent_dim=20,
    )
    required = validator._prediction_receipt_required(args, manifest, fold, "A")
    assert required["receipt_path"] == str(
        (tmp_path / "folds/A/fit_predict_receipt.json").resolve()
    )
    assert required["prediction_path"] == str((tmp_path / "folds/A/predictions.npz").resolve())
    assert required["frozen_fit_function"] == "fit_predict_one_fold"
    assert required["smoke"] is True
    receipt = {
        **required,
        "prediction_file_sha256": "a" * 64,
        "prediction_semantic_sha256": "b" * 64,
    }
    assert validator._receipt_matches_required(receipt, required)
    receipt["cpu_threads"] = 3
    assert not validator._receipt_matches_required(receipt, required)


def test_canonical_path_binding_rejects_noncanonical_authority(tmp_path: Path) -> None:
    expected = tmp_path / "artifact.npz"
    assert (
        validator._canonical_path(str(expected.resolve()), expected, label="artifact")
        == expected.resolve()
    )
    with pytest.raises(ValueError, match="canonical registered path"):
        validator._canonical_path("artifact.npz", expected, label="artifact")


def test_smoke_preparation_cannot_enter_scientific_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validator,
        "_read_prepared",
        lambda args, local_runner: {"smoke": True},
    )
    monkeypatch.setattr(
        validator,
        "_read_prediction_manifest",
        lambda *args: pytest.fail("prediction manifest must not open for smoke score"),
    )
    with pytest.raises(ValueError, match="smoke artifacts are prohibited"):
        validator.score(SimpleNamespace(), SimpleNamespace())


def test_relative_reduction_is_computed_within_donor_before_summary() -> None:
    class FakeRunner:
        @staticmethod
        def _donor_bootstrap_interval(values, *, seed):
            assert seed == 9
            return float(np.min(values)), float(np.max(values))

        @staticmethod
        def _sign_flip(core, values):
            assert core == "core"
            return {"observed": float(np.mean(values)), "p_value": 0.25}

    result = validator._relative_reduction_summary(
        FakeRunner,
        "core",
        {
            "A": {"M0": 10.0, "M3": 8.0},
            "B": {"M0": 4.0, "M3": 3.0},
        },
        seed=9,
        original_relative_gain=validator.ORIGINAL_EXTERNAL_PANEL_RELATIVE_GAIN,
    )
    assert result["donor_relative_reduction"] == pytest.approx([0.2, 0.25])
    assert result["mean_relative_reduction"] == pytest.approx(0.225)
    assert result["median_relative_reduction"] == pytest.approx(0.225)
    assert result["paired_donor_bootstrap_confidence_interval"] == pytest.approx([0.2, 0.25])
    assert result["mean_gain_retained_vs_original"] == pytest.approx(
        0.225 / validator.ORIGINAL_EXTERNAL_PANEL_RELATIVE_GAIN
    )


def _inactive_program_predictions() -> dict[str, np.ndarray]:
    return {
        "diagnostic_program_membership": np.asarray(
            [[False, False, False], [True, False, True]], dtype=bool
        ),
        "diagnostic_program_active": np.asarray([False, True], dtype=bool),
        "diagnostic_rare_program_thresholds": np.asarray([np.nan, 1.25], dtype=np.float32),
        "rate_M0": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
    }


def test_inactive_zero_gene_program_gets_unscored_finite_sentinel_only() -> None:
    raw = _inactive_program_predictions()
    original_rate = raw["rate_M0"].copy()
    normalized = validator._normalize_inactive_program_thresholds(raw)
    np.testing.assert_array_equal(normalized["rate_M0"], original_rate)
    np.testing.assert_array_equal(
        normalized["diagnostic_rare_program_thresholds"],
        np.asarray([0.0, 1.25], dtype=np.float32),
    )
    assert normalized["fold_local_inactive_program_threshold_policy"].item() == (
        validator.INACTIVE_PROGRAM_THRESHOLD_POLICY
    )
    assert int(normalized["fold_local_inactive_program_threshold_count"]) == 1

    class FakeRunner:
        called = False

        @classmethod
        def _validate_prediction_artifact(cls, predictions, public, *, donor, epochs):
            cls.called = True
            assert donor == "D1"
            assert epochs == 80

    validator._validate_fold_local_prediction(FakeRunner, normalized, {}, donor="D1", epochs=80)
    assert FakeRunner.called


def test_active_program_threshold_must_remain_finite() -> None:
    raw = _inactive_program_predictions()
    raw["diagnostic_rare_program_thresholds"][1] = np.nan
    with pytest.raises(ValueError, match="diagnostic program receipt"):
        validator._normalize_inactive_program_thresholds(raw)


def test_inactive_program_sentinel_receipt_fails_closed_on_corruption() -> None:
    normalized = validator._normalize_inactive_program_thresholds(_inactive_program_predictions())
    normalized["diagnostic_rare_program_thresholds"][0] = 0.5

    class FakeRunner:
        @staticmethod
        def _validate_prediction_artifact(*args, **kwargs):
            pytest.fail("frozen validator must not run after wrapper receipt corruption")

    with pytest.raises(ValueError, match="inactive-program threshold receipt"):
        validator._validate_fold_local_prediction(FakeRunner, normalized, {}, donor="D1", epochs=80)
