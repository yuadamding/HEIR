"""Raw-count support, nested geometry, and measurement-only adapter regressions."""

from dataclasses import replace

import numpy as np
import pytest
from test_personalized_cohort import published_cohort as published_cohort

from heir.personalized import cohort
from heir.personalized.measurement import (
    MeasurementSubstrate,
    aggregate_measurements,
    common_geometry_support,
    mapped_base_regions,
    measurement_values,
)
from heir.personalized.regions import build_region_grid, nested_grid_membership, point_to_grid


@pytest.fixture
def substrate():
    from scipy.sparse import csr_matrix

    image = np.full((100, 350, 3), [170, 75, 120], dtype=np.uint8)
    image[:, 200:300] = 255  # A tissue gap must never be filled by the parent rectangle.
    grid = build_region_grid(image, compute_features=False)
    xy = np.array(
        [[5, 5], [15, 5], [150, 5], [310, 5], [250, 5], [8, 5], [160, 5], [320, 5], [400, 5]],
        dtype=float,
    )
    replicates = ("1",) * 5 + ("2",) * 4
    values = np.array(
        [
            [2, 0, 1],
            [3, 1, 0],
            [0, 0, 0],
            [1, 4, 2],
            [900, 900, 900],
            [1, 2, 3],
            [0, 0, 0],
            [4, 1, 2],
            [800, 800, 800],
        ],
        dtype=np.int64,
    )
    return MeasurementSubstrate(
        "biopsy",
        "patient",
        "specimen",
        grid,
        ("A", "B", "C"),
        ("A", "B", "C", "RNA_only"),
        tuple(f"bead{i}" for i in range(len(xy))),
        xy,
        replicates,
        csr_matrix(values),
        point_to_grid(grid, xy),
        {},
    )


def test_geometry_only_grid_preserves_default_selection_without_invoking_encoder(monkeypatch):
    from heir.personalized import regions

    image = np.full((103, 205, 3), [170, 75, 120], dtype=np.uint8)
    original = build_region_grid(image)
    monkeypatch.setattr(
        regions, "HandcraftedPatchEncoder", lambda: pytest.fail("no encoder allowed")
    )
    measured = build_region_grid(image, compute_features=False)
    assert measured.region_ids == original.region_ids
    np.testing.assert_array_equal(measured.coordinates, original.coordinates)
    np.testing.assert_array_equal(measured.tissue_fraction, original.tissue_fraction)
    assert measured.image_features.shape == (6, 0)
    assert measured.tissue_pixel_counts.sum() == 103 * 205


def test_nested_aggregation_conserves_counts_with_exact_clipped_support(substrate):
    fine = aggregate_measurements(substrate, scale_pixels=100)
    for scale in (100, 200, 400):
        coarse = aggregate_measurements(substrate, scale_pixels=scale)
        assert coarse.conservation["direct_bead_aggregation_equal"]
        assert coarse.grid.support_area_pixels.sum() == 25000
        assert coarse.grid.tissue_area_pixels.sum() == 25000
        for rep in ("1", "2"):
            first = fine.counts[np.asarray(fine.observation_replicates) == rep].sum(0)
            after = coarse.counts[np.asarray(coarse.observation_replicates) == rep].sum(0)
            np.testing.assert_array_equal(first, after)
        assert coarse.bead_observation_indices[4] == coarse.bead_observation_indices[8] == -1
    coarse = aggregate_measurements(substrate, scale_pixels=200)
    assert coarse.region_ids == ("he_region_200:0:0", "he_region_200:1:0")
    assert coarse.grid.parent_fine_region_ids == (
        ("he_region:0:0", "he_region:1:0"),
        ("he_region:3:0",),
    )
    assert coarse.grid.bin_area_pixels.tolist() == [20000, 15000]
    assert coarse.grid.support_area_pixels.tolist() == [20000, 5000]
    assert coarse.bead_counts.tolist() == [3, 1, 2, 1]


def test_measured_zero_and_missing_rows_genes_remain_distinct(substrate):
    fine = aggregate_measurements(substrate, scale_pixels=100)
    values = measurement_values(fine, outcome_gene_ids=("A",), normalization_gene_ids=("A", "B"))
    assert len(fine.observation_ids) == 6  # Zero-count bead-containing bins remain measured.
    assert values["observed"].all()
    assert values["eligible"].tolist() == [True, False, True, True, False, True]
    assert np.isnan(values["panel_fraction"][[1, 4]]).all()
    with pytest.raises(ValueError, match="missing is not zero"):
        measurement_values(fine, outcome_gene_ids=("absent",), normalization_gene_ids=("A", "B"))
    changed = replace(substrate, coordinates=np.array(substrate.coordinates, copy=True))
    xy = changed.coordinates.copy()
    xy[7] = [400, 5]
    changed = replace(changed, coordinates=xy, base_region_indices=point_to_grid(changed.grid, xy))
    result = aggregate_measurements(changed)
    assert result.region_ids == fine.region_ids  # Full parent geometry remains declared.
    assert "slide_seq:2:he_region:3:0" not in result.observation_ids
    assert len(result.observation_ids) == 5


def test_shared_support_is_geometry_only_and_displacements_remap_original_beads(substrate):
    support = common_geometry_support(substrate, ("1", "2"))
    assert support.tolist() == [0, 1, 2]
    zeros = replace(substrate, counts=substrate.counts * 0)
    np.testing.assert_array_equal(common_geometry_support(zeros, ("1", "2")), support)
    shifted = mapped_base_regions(substrate, {"2": [-25, 0]})
    assert shifted[5] == -1 and shifted[7] == -1  # Leave image and enter unselected white bin.
    np.testing.assert_array_equal(shifted[:5], substrate.base_region_indices[:5])
    shared = common_geometry_support(
        substrate, ("1", "2"), displacement_by_replicate={"2": [-25, 0]}
    )
    assert shared.tolist() == [1]
    result = aggregate_measurements(
        substrate,
        scale_pixels=400,
        support_fine_indices=shared,
        displacement_by_replicate={"2": [-25, 0]},
    )
    assert result.bead_counts.tolist() == [1, 1]
    assert result.occupied_fine_indices == ((1,), (1,))
    assert result.grid.support_area_pixels.tolist() == [10000]
    assert not result.counts.nnz  # Geometry retained even though expression is all zero.
    np.testing.assert_array_equal(
        substrate.base_region_indices, point_to_grid(substrate.grid, substrate.coordinates)
    )
    empty = aggregate_measurements(substrate, support_fine_indices=[])
    assert empty.counts.shape == (0, 3) and empty.region_ids == ()
    assert all(
        record["included_counts_total"] == 0
        for record in empty.conservation["by_replicate"].values()
    )


def test_normalizer_is_fixed_under_outcome_subsetting_and_aggregation_first(substrate):
    measurements = aggregate_measurements(substrate, scale_pixels=200)
    one = measurement_values(
        measurements, outcome_gene_ids=("A",), normalization_gene_ids=("A", "B", "C")
    )
    two = measurement_values(
        measurements, outcome_gene_ids=("A", "B"), normalization_gene_ids=("A", "B", "C")
    )
    np.testing.assert_array_equal(one["normalization_exposure"], two["normalization_exposure"])
    assert one["outcome_counts"][0, 0] == 5
    assert one["normalization_exposure"][0] == 7
    assert one["total_exposure"][0] == 7
    with pytest.raises(ValueError, match="integer multiple"):
        nested_grid_membership(substrate.grid, 150)
    with pytest.raises(ValueError, match="unique valid"):
        nested_grid_membership(substrate.grid, 200, fine_indices=[0, 0])
    with pytest.raises(ValueError, match="unmeasured replicate"):
        mapped_base_regions(substrate, {"absent": [0, 0]})


def test_loader_gene_preflight_and_base_counts_match_existing_target_adapter(
    published_cohort, monkeypatch
):
    root = published_cohort
    original_inputs, original_provenance = cohort.prepare_cohort_inputs(
        root, "313-932", region_size=10, max_states=2
    )
    original_targets, original_operator, original_metadata = cohort.load_cohort_targets(
        root, "313-932", original_inputs, original_provenance, gene_ids=("EPCAM", "KRT8")
    )
    original_sha256 = cohort.hashlib.sha256
    source_hash_calls = []

    def count_source_hashes(*args, **kwargs):
        if not args:
            source_hash_calls.append(1)
        return original_sha256(*args, **kwargs)

    monkeypatch.setattr(cohort.hashlib, "sha256", count_source_hashes)
    cache = {}
    axes = cohort.load_measurement_gene_axes(root, "313-932", verification_cache=cache)
    assert axes["spatial_gene_ids"] == ("EPCAM", "KRT8")
    assert len(axes["independent_rna_gene_ids"]) == 19
    # Reference fitting and any image encoder are forbidden on the measurement route.
    from heir.personalized import regions

    monkeypatch.setattr(
        cohort, "build_reference", lambda *a, **k: pytest.fail("no fitted reference")
    )
    monkeypatch.setattr(
        regions, "HandcraftedPatchEncoder", lambda: pytest.fail("no image features")
    )
    measured = cohort.load_measurement_substrate(
        root, "313-932", base_region_size=10, verification_cache=cache
    )
    assert len(cache) == 3  # Only independent RNA, Slide-seq and H&E were consumed.
    assert len(source_hash_calls) == 3  # Preflight and loader reuse each verified source.
    assert measured.grid.image_features.shape == (4, 0)
    result = aggregate_measurements(measured, scale_pixels=10)
    assert result.observation_ids == (
        "slide_seq:1:he_region:0:0",
        "slide_seq:1:he_region:1:0",
        "slide_seq:2:he_region:0:0",
        "slide_seq:2:he_region:1:1",
    )
    np.testing.assert_array_equal(result.counts.toarray(), [[4, 6], [5, 6], [7, 8], [9, 10]])
    assert result.bead_counts.tolist() == [2, 1, 1, 1]
    assert result.observation_ids == original_targets.observation_ids
    np.testing.assert_array_equal(result.counts.toarray(), original_targets.expression)
    np.testing.assert_array_equal(result.region_indices, original_operator.columns)
    assert result.bead_counts.tolist() == original_metadata["region_bead_counts"]
    assert measured.provenance["replicate_linkage"] == "unknown"
    assert measured.provenance["physical_calibration"] == "unknown"
    # A cached verification is rejected after a same-size mutation, not silently reused.
    source = axes["sources"]["rna"]
    path = root / source["working_path"]
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        previous = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([previous[0] ^ 1]))
    with pytest.raises(ValueError, match="changed after verification"):
        cohort.load_measurement_gene_axes(root, "313-932", verification_cache=cache)
