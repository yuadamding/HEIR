import numpy as np
import pytest

from heir.image.coordinates import (
    AffineTransform2D,
    PixelMicronTransform,
    microns_to_native_pixels,
    native_pixels_to_microns,
    normalize_mpp,
)
from heir.image.graph import build_spatial_graph
from heir.image.nuclei import (
    ConservationError,
    aggregate_nuclei_to_spots,
    assign_nuclei_to_visium_spots,
    check_spot_conservation,
)


def test_native_pixel_micron_affine_round_trip():
    transform = PixelMicronTransform(
        native_mpp=(0.25, 0.5),
        pixel_origin=(100.0, 50.0),
        micron_origin=(12.0, -7.0),
        rotation_degrees=31.0,
        flip_y=True,
    )
    pixels = np.array([[100.0, 50.0], [120.5, 80.25], [-2.0, 9.0]])
    microns = transform.native_to_microns(pixels)
    recovered = transform.microns_to_native(microns)
    np.testing.assert_allclose(recovered, pixels, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(microns[0], [12.0, -7.0])


def test_general_affine_and_convenience_conversions():
    matrix = np.array([[0.5, 0.1, 4.0], [-0.2, 0.75, -3.0], [0.0, 0.0, 1.0]])
    affine = AffineTransform2D(matrix)
    points = np.array([[[0.0, 0.0], [2.0, 3.0]]])
    np.testing.assert_allclose(
        affine.inverse_transform(affine.transform(points)), points, atol=1e-12
    )

    pixels = np.array([[2.0, 4.0], [8.0, 10.0]])
    physical = native_pixels_to_microns(pixels, (0.5, 0.25))
    np.testing.assert_allclose(physical, [[1.0, 1.0], [4.0, 2.5]])
    np.testing.assert_allclose(
        microns_to_native_pixels(physical, (0.5, 0.25)),
        pixels,
    )


@pytest.mark.parametrize("value", [0.0, -0.5, (0.5,), (0.5, np.nan)])
def test_invalid_mpp_is_rejected(value):
    with pytest.raises(ValueError):
        normalize_mpp(value)


def test_knn_radius_graph_geometry_degree_and_boundary_weights():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [5.0, 5.0]])
    graph = build_spatial_graph(
        coordinates,
        k=3,
        radius=1.1,
        max_degree=2,
        symmetric=True,
        distance_scale=1.0,
        boundary_weights=[1.0, 1.0, 0.0, 1.0, 1.0],
    )
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_features.shape == (graph.num_edges, 4)
    assert np.all(graph.in_degree() <= 2)
    assert np.all(graph.distance <= 1.1 + 1e-6)
    source, target = graph.edge_index
    displacement = coordinates[target] - coordinates[source]
    np.testing.assert_allclose(graph.dx, displacement[:, 0], atol=1e-6)
    np.testing.assert_allclose(graph.dy, displacement[:, 1], atol=1e-6)
    np.testing.assert_allclose(graph.distance, np.linalg.norm(displacement, axis=1), atol=1e-6)
    np.testing.assert_allclose(graph.angle, np.arctan2(graph.dy, graph.dx), atol=1e-6)
    incident_to_unreliable = (source == 2) | (target == 2)
    assert incident_to_unreliable.any()
    assert np.all(graph.edge_weight[incident_to_unreliable] == 0.0)
    assert graph.in_degree()[4] == 0


def test_pure_radius_graph_and_empty_graph():
    graph = build_spatial_graph(
        np.array([[0.0, 0.0], [0.4, 0.0], [1.2, 0.0]]),
        k=None,
        radius=0.5,
        symmetric=False,
    )
    assert graph.num_edges == 2
    np.testing.assert_array_equal(graph.edge_index, [[1, 0], [0, 1]])

    empty = build_spatial_graph(np.empty((0, 2)), k=3)
    assert empty.num_nodes == 0
    assert empty.num_edges == 0


def test_visium_assignment_and_conservation_checked_aggregation():
    nuclei = np.array([[0.0, 0.0], [4.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    spots = np.array([[0.0, 0.0], [10.0, 0.0]])
    assignment = assign_nuclei_to_visium_spots(
        nuclei,
        spots,
        spot_radius=5.0,
        spot_ids=["spot-a", "spot-b"],
    )
    np.testing.assert_array_equal(assignment.spot_index, [0, 0, 1, -1])
    assert assignment.assigned_count == 3
    assert assignment.unassigned_count == 1

    values = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [99.0, 99.0]])
    summed = aggregate_nuclei_to_spots(values, assignment, reduction="sum")
    np.testing.assert_allclose(summed.values, [[4.0, 6.0], [10.0, 20.0]])
    np.testing.assert_array_equal(summed.counts, [2, 1])
    assert summed.assigned_count == 3
    assert summed.unassigned_count == 1

    averaged = aggregate_nuclei_to_spots(
        values[:, 0],
        assignment,
        reduction="weighted_mean",
        weights=[1.0, 3.0, 2.0, 1.0],
    )
    np.testing.assert_allclose(averaged.values, [2.5, 10.0])
    np.testing.assert_allclose(averaged.sums, [10.0, 20.0])
    np.testing.assert_allclose(averaged.weight_sums, [4.0, 2.0])


def test_conservation_error_is_descriptive():
    with pytest.raises(ConservationError, match="failed conservation"):
        check_spot_conservation(np.array([1.0, 2.0]), np.array([1.0, 2.1]))
