"""Focused tests for RNA-derived residual bases and calibrated bounds."""

import numpy as np
import pytest

from heir.prior import RNAResidualGeometry, fit_rna_residual_geometry


def test_type_pca_bases_are_orthonormal_rank_safe_and_deterministic() -> None:
    type_a = np.asarray(
        [
            [-4.0, -1.0, 0.0, 0.0],
            [-2.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [2.0, -1.0, 0.0, 0.0],
            [4.0, 1.0, 0.0, 0.0],
        ]
    )
    type_b = np.asarray(
        [
            [0.0, 0.0, -3.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    latent = np.concatenate((type_a, type_b, np.asarray([[0.0, 0.0, 0.0, 7.0]])))
    labels = np.asarray(["a"] * len(type_a) + ["b"] * len(type_b) + ["rare"])
    order = np.asarray([8, 2, 10, 5, 0, 7, 4, 1, 9, 3, 6])

    first = fit_rna_residual_geometry(
        latent,
        labels,
        rank=2,
        type_names=["b", "a", "rare"],
    )
    permuted = fit_rna_residual_geometry(
        latent[order],
        labels[order],
        rank=2,
        type_names=["b", "a", "rare"],
    )

    np.testing.assert_array_equal(first.type_names, ["b", "a", "rare"])
    np.testing.assert_array_equal(first.effective_ranks, [1, 2, 0])
    np.testing.assert_allclose(
        np.swapaxes(first.residual_type_basis, 1, 2) @ first.residual_type_basis,
        np.broadcast_to(np.eye(2), (3, 2, 2)),
        atol=1.0e-6,
    )
    # The locally supported A subspace is exactly the first two latent axes.
    a_projection = first.residual_type_basis[1] @ first.residual_type_basis[1].T
    np.testing.assert_allclose(a_projection, np.diag([1.0, 1.0, 0.0, 0.0]), atol=1.0e-6)
    np.testing.assert_allclose(permuted.residual_type_basis, first.residual_type_basis, atol=1e-7)
    np.testing.assert_allclose(
        permuted.residual_type_max_norm,
        first.residual_type_max_norm,
        atol=1e-7,
    )


def test_bounds_prefer_state_geometry_then_covariance_then_pooled_residual() -> None:
    latent = np.asarray(
        [
            [-0.2, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.2, -0.1, 0.0],
            [3.8, 0.0, 0.0],
            [4.0, 0.1, 0.0],
            [4.2, -0.1, 0.0],
            [0.0, 5.0, -0.4],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.4],
            [8.0, 8.0, 8.0],
        ]
    )
    labels = np.asarray(["state"] * 6 + ["covariance"] * 3 + ["rare"])
    means = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    mean_labels = np.asarray(["state", "state", "covariance"])
    variances = np.asarray([[0.04, 0.04, 0.04], [0.04, 0.04, 0.04], [1.0, 1.0, 1.0]])

    geometry = fit_rna_residual_geometry(
        latent,
        labels,
        rank=2,
        type_names=["state", "covariance", "rare"],
        prototype_means=means,
        prototype_labels=mean_labels,
        prototype_variances=variances,
        calibration_quantile=0.9,
        bound_fraction=0.25,
    )

    assert geometry.state_scales[0] == pytest.approx(4.0)
    assert geometry.residual_type_max_norm[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(
        geometry.scale_sources,
        ["state", "covariance", "pooled_residual"],
    )
    assert geometry.covariance_scales[1] > 0
    assert geometry.residual_type_max_norm[1] > geometry.minimum_bound
    assert geometry.residual_type_max_norm[2] > geometry.minimum_bound


def test_covariance_aware_assignment_uses_regularized_gaussian_nll() -> None:
    cells = np.asarray([[2.4], [2.5], [2.6]])
    labels = ["state"] * len(cells)
    means = np.asarray([[0.0], [3.0]])
    mean_labels = ["state", "state"]

    gaussian = fit_rna_residual_geometry(
        cells,
        labels,
        rank=1,
        prototype_means=means,
        prototype_labels=mean_labels,
        prototype_variances=np.asarray([[100.0], [0.01]]),
        scale_priority=["residual"],
        bound_fraction=1.0,
    )
    euclidean = fit_rna_residual_geometry(
        cells,
        labels,
        rank=1,
        prototype_means=means,
        prototype_labels=mean_labels,
        scale_priority=["residual"],
        bound_fraction=1.0,
    )

    assert gaussian.prototype_assignment_method == "diagonal_gaussian_nll"
    assert euclidean.prototype_assignment_method == "euclidean"
    # The broad zero-centered state wins under Gaussian NLL, whereas Euclidean
    # assignment selects the narrow state at 3.0.
    assert gaussian.residual_scales[0] > 2.0
    assert euclidean.residual_scales[0] == pytest.approx(0.58)

    repeated = fit_rna_residual_geometry(
        cells,
        labels,
        rank=1,
        prototype_means=means,
        prototype_labels=mean_labels,
        scale_priority=["residual"],
        bound_fraction=1.0,
    )
    np.testing.assert_array_equal(repeated.residual_scales, euclidean.residual_scales)


def test_residual_scale_is_calibrated_after_projection() -> None:
    geometry = fit_rna_residual_geometry(
        np.asarray([[-1.0, 100.0], [0.0, 100.0], [1.0, 100.0]]),
        ["state", "state", "state"],
        rank=1,
        prototype_means=np.asarray([[0.0, 0.0]]),
        prototype_labels=["state"],
        scale_priority=["residual"],
        calibration_quantile=0.9,
        bound_fraction=1.0,
    )

    # The constant 100-unit offset is orthogonal to the learned residual
    # direction and must not inflate a correction HEIR cannot express.
    assert geometry.residual_scales[0] == pytest.approx(1.0)
    assert geometry.residual_type_max_norm[0] == pytest.approx(1.0)
    assert geometry.residual_scale_method == "projected_quantile"


def test_state_bound_uses_low_quantile_of_projected_nearest_neighbors() -> None:
    geometry = fit_rna_residual_geometry(
        np.asarray([[-0.1], [0.1], [1.9], [2.1], [99.9], [100.1]]),
        ["state"] * 6,
        rank=1,
        prototype_means=np.asarray([[0.0], [2.0], [100.0]]),
        prototype_labels=["state"] * 3,
        state_separation_quantile=0.1,
        bound_fraction=0.5,
    )

    # Nearest-state separations are [2, 2, 98]; the distant third state does
    # not inflate the two-unit local basin as an all-pairs median would.
    assert geometry.state_scales[0] == pytest.approx(2.0)
    assert geometry.residual_type_max_norm[0] == pytest.approx(1.0)
    assert geometry.state_scale_method == "projected_nearest_neighbor_quantile"


def test_single_prototype_uses_projected_covariance_or_residual_fallback() -> None:
    geometry = fit_rna_residual_geometry(
        np.asarray(
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 4.0],
                [0.0, 5.0],
                [0.0, 6.0],
            ]
        ),
        ["covariance"] * 3 + ["residual"] * 3,
        rank=1,
        type_names=["covariance", "residual"],
        prototype_means=np.asarray([[0.0, 0.0], [0.0, 5.0]]),
        prototype_labels=["covariance", "residual"],
        prototype_variances=np.asarray([[0.25, 0.0], [0.0, 0.0]]),
    )

    np.testing.assert_allclose(geometry.state_scales, 0.0)
    np.testing.assert_array_equal(geometry.scale_sources, ["covariance", "residual"])
    assert geometry.covariance_scales[0] > 0
    assert geometry.residual_scales[1] > 0


def test_zero_variation_rare_types_receive_deterministic_minimum_geometry() -> None:
    geometry = fit_rna_residual_geometry(
        np.zeros((3, 3), dtype=np.float32),
        ["a", "b", "c"],
        rank=2,
        minimum_bound=0.02,
    )

    np.testing.assert_array_equal(geometry.effective_ranks, [0, 0, 0])
    np.testing.assert_array_equal(geometry.scale_sources, ["minimum", "minimum", "minimum"])
    np.testing.assert_allclose(geometry.residual_type_max_norm, 0.02)
    expected = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    for basis in geometry.residual_type_basis:
        np.testing.assert_array_equal(basis, expected)


def test_artifact_round_trip_and_model_order_alignment(tmp_path) -> None:
    geometry = fit_rna_residual_geometry(
        np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        ["a", "a", "a", "b", "b", "b"],
        rank=2,
        latent_space_id="scanvi:test",
        source_reference_sha256="a" * 64,
        training_donors=["donor-2", "donor-1"],
        latent_transform_sha256="b" * 64,
    )
    path = tmp_path / "residual_geometry.npz"
    geometry.to_npz(path)
    loaded = RNAResidualGeometry.from_npz(path)

    assert loaded.latent_space_id == "scanvi:test"
    assert loaded.source_reference_sha256 == "a" * 64
    assert loaded.training_donors == ("donor-2", "donor-1")
    assert loaded.latent_transform_sha256 == "b" * 64
    assert loaded.prototype_assignment_method == "euclidean"
    assert loaded.state_scale_method == "projected_nearest_neighbor_quantile"
    assert loaded.residual_scale_method == "projected_quantile"
    assert not loaded.residual_type_basis.flags.writeable
    np.testing.assert_array_equal(loaded.residual_type_basis, geometry.residual_type_basis)
    basis, bounds = loaded.model_parameters(["b", "a"])
    np.testing.assert_array_equal(basis, geometry.residual_type_basis[[1, 0]])
    np.testing.assert_array_equal(bounds, geometry.residual_type_max_norm[[1, 0]])
    assert basis.flags.writeable
    with pytest.raises(ValueError, match="missing types: absent"):
        loaded.model_parameters(["a", "absent"])

    with np.load(path, allow_pickle=False) as values:
        legacy_values = {name: values[name] for name in values.files}
    legacy_values["schema"] = np.asarray(RNAResidualGeometry.LEGACY_SCHEMA, dtype=np.str_)
    legacy_path = tmp_path / "legacy_residual_geometry.npz"
    np.savez_compressed(legacy_path, **legacy_values)
    with pytest.raises(ValueError, match="allow_legacy_geometry=True"):
        RNAResidualGeometry.from_npz(legacy_path)
    migrated = RNAResidualGeometry.from_npz(
        legacy_path,
        allow_legacy_geometry=True,
    )
    assert migrated.prototype_assignment_method == "legacy_euclidean"
    assert migrated.state_scale_method == "legacy_all_pair_median"
    assert migrated.residual_scale_method == "legacy_full_latent_quantile"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 4}, "rank must be"),
        ({"rank": 1, "calibration_quantile": 1.0}, "calibration_quantile"),
        ({"rank": 1, "prototype_variances": np.ones((1, 3))}, "require prototype"),
        ({"rank": 1, "scale_priority": ["state", "state"]}, "scale_priority"),
        ({"rank": 1, "prototype_variance_floor": 0.0}, "prototype_variance_floor"),
        ({"rank": 1, "state_separation_quantile": 1.0}, "state_separation_quantile"),
    ],
)
def test_invalid_geometry_inputs_fail_loudly(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_rna_residual_geometry(
            np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ["a", "a"],
            **kwargs,
        )
