from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_validator():
    path = Path(__file__).parents[1] / "scripts/validate_natcommun_frozen_predictions.py"
    spec = importlib.util.spec_from_file_location("validate_natcommun_frozen_predictions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def test_weighted_count_alignment_identifies_positive_over_applied_correction() -> None:
    correction = np.asarray([[2.0], [2.0], [2.0]])
    residual = np.asarray([[1.0], [1.0], [1.0]])
    variance = np.ones_like(correction)
    result = validator._section_weighted_count_alignment(
        correction,
        residual,
        variance,
        variance,
        np.asarray(["section", "section", "section"]),
    )

    summary = result["section_balanced"]
    assert summary["predictive_variance_weighted_inner_product"] > 0
    assert np.isclose(summary["predictive_variance_weighted_cosine"], 1.0)
    assert np.isclose(summary["optimal_correction_scale"], 0.5)


def test_weighted_effect_is_section_balanced_and_requires_effective_rows() -> None:
    values = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    sections = np.asarray(["A", "A", "A", "B", "B", "B"])
    result = validator._weighted_section_effect(values, sections)

    assert result["evaluable"] is True
    assert np.isclose(result["mean_effect"], 0.0)
    assert len(result["sections"]) == 2

    none = validator._weighted_section_effect(values[:2], sections[:2])
    assert none["evaluable"] is False

    negligible = validator._weighted_section_effect(
        values,
        sections,
        weights=np.full(len(values), 0.01),
        minimum_total_weight=3.0,
    )
    assert negligible["evaluable"] is False


def test_dynamic_range_uses_within_section_quantiles() -> None:
    observed = np.arange(40, dtype=float).reshape(20, 2)
    predicted = observed * 0.5
    sections = np.asarray(["A"] * 10 + ["B"] * 10)
    ratio = validator._dynamic_ranges(observed, predicted, sections)

    assert np.allclose(ratio, 0.5)


def test_array_digest_binds_shape_dtype_and_values() -> None:
    base = np.asarray([True, False, True])
    assert validator._array_digest(base) == validator._array_digest(base.copy())
    assert validator._array_digest(base) != validator._array_digest(~base)
    assert validator._array_digest(base) != validator._array_digest(base.astype(np.int8))


class _RunnerStub:
    PREPARED_SCHEMA = "prepared"

    @staticmethod
    def _scalar_text(value):
        return str(np.asarray(value).item())


def _target_fixture():
    public = {
        "query_spot_ids": np.asarray(["s1", "s2"]),
        "query_section_ids": np.asarray(["A", "A"]),
        "query_indication_ids": np.asarray(["lung", "lung"]),
        "gene_ids": np.asarray(["g1", "g2"]),
    }
    full = np.asarray([[2, 1], [0, 0]], dtype=np.float32)
    half_a = np.asarray([[1, 1], [0, 0]], dtype=np.float32)
    half_b = full - half_a
    secret = {
        "schema": np.asarray("prepared"),
        "heldout_donor": np.asarray("D1"),
        "heldout_spot_ids": public["query_spot_ids"].copy(),
        "heldout_section_ids": public["query_section_ids"].copy(),
        "heldout_indication_ids": public["query_indication_ids"].copy(),
        "heldout_st_counts": full,
        "heldout_st_half_a": half_a,
        "heldout_st_half_b": half_b,
        "heldout_st_library": np.asarray([3, 0], dtype=np.float32),
        "heldout_st_library_half_a": np.asarray([2, 0], dtype=np.float32),
        "heldout_st_library_half_b": np.asarray([1, 0], dtype=np.float32),
        "primary_score_eligible": np.asarray([True, False]),
        "zero_depth_excluded_count": np.asarray(1, dtype=np.int64),
    }
    predictions = {
        "query_spot_ids": public["query_spot_ids"].copy(),
        "gene_ids": public["gene_ids"].copy(),
    }
    return public, secret, predictions


def test_score_target_identity_accepts_exact_bound_target() -> None:
    public, secret, predictions = _target_fixture()
    validator._validate_score_target_identity(
        _RunnerStub, secret, public, predictions, donor="D1"
    )


def test_score_target_identity_rejects_row_or_axis_mismatch() -> None:
    public, secret, predictions = _target_fixture()
    secret["heldout_spot_ids"] = np.asarray(["s2", "s1"])
    with np.testing.assert_raises_regex(ValueError, "heldout_spot_ids"):
        validator._validate_score_target_identity(
            _RunnerStub, secret, public, predictions, donor="D1"
        )

    public, secret, predictions = _target_fixture()
    secret["heldout_st_counts"] = secret["heldout_st_counts"][:, :1]
    with np.testing.assert_raises_regex(ValueError, "heldout_st_counts shape"):
        validator._validate_score_target_identity(
            _RunnerStub, secret, public, predictions, donor="D1"
        )
