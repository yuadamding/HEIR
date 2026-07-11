import csv
import json

import numpy as np
import pytest
from scipy import io as scipy_io
from scipy import sparse

from heir.cli import main
from heir.data import (
    SpatialTruthArtifact,
    TissuePositions,
    VisiumCounts,
    align_visium_barcodes,
    build_spatial_truth,
    normalize_panel_counts,
    read_tissue_positions,
    read_visium_counts,
)
from heir.data.manifest import MANIFEST_COLUMNS, ManifestRecord


def _sources():
    names = np.asarray(["counts", "positions", "scales", "nuclei", "panel", "manifest"])
    hashes = np.asarray([str(index) * 64 for index in range(1, 7)])
    roles = np.asarray(
        [
            "locked_spatial_counts",
            "locked_spatial_coordinates",
            "locked_spatial_scalefactors",
            "sample_segmentation",
            "canonical_gene_panel",
            "shared_manifest",
        ]
    )
    return names, hashes, roles


def test_barcode_auto_strips_seurat_export_suffix_globally():
    positions = TissuePositions(
        barcodes=np.asarray(["AAAC-1", "CCGT-1"]),
        coordinates_px=np.asarray([[0.0, 0.0], [20.0, 0.0]]),
        in_tissue=np.asarray([True, True]),
    )
    alignment = align_visium_barcodes(
        ["AAAC-1_1", "CCGT-1_1"],
        positions,
        policy="auto",
    )
    assert alignment.policy == "strip-export"
    np.testing.assert_array_equal(alignment.count_indices, [0, 1])
    np.testing.assert_array_equal(alignment.position_indices, [0, 1])

    with pytest.raises(ValueError, match="collisions"):
        align_visium_barcodes(
            ["AAAC-1", "AAAC-2"],
            TissuePositions(
                ["AAAC-1", "AAAC-2"],
                [[0.0, 0.0], [1.0, 0.0]],
                [True, True],
            ),
            policy="strip-gem",
        )


def test_positions_reader_supports_headered_and_legacy_files(tmp_path):
    headered = tmp_path / "tissue_positions.csv"
    headered.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "AA-1,1,0,0,12,34\n"
    )
    parsed = read_tissue_positions(headered, coordinate_scale=2.0)
    np.testing.assert_allclose(parsed.coordinates_px, [[68.0, 24.0]])

    legacy = tmp_path / "tissue_positions_list.csv"
    legacy.write_text("BB-1,1,0,0,5,7\n")
    parsed_legacy = read_tissue_positions(legacy)
    np.testing.assert_allclose(parsed_legacy.coordinates_px, [[7.0, 5.0]])


def test_full_library_normalization_and_versioned_round_trip(tmp_path):
    counts = VisiumCounts(
        barcodes=np.asarray(["AA-1_1", "BB-1_1"]),
        gene_names=np.asarray(["g2", "g1"]),
        counts=sparse.csr_matrix([[0, 10], [20, 0]], dtype=np.float32),
        library_sizes=np.asarray([20.0, 40.0]),
        matrix_source="test",
    )
    positions = TissuePositions(
        np.asarray(["AA-1", "BB-1"]),
        np.asarray([[0.0, 0.0], [20.0, 0.0]]),
        np.asarray([True, True]),
    )
    sources, hashes, roles = _sources()
    artifact = build_spatial_truth(
        counts=counts,
        positions=positions,
        nucleus_ids=["sample::n0", "sample::n1", "sample::n2"],
        nucleus_coordinates_px=np.asarray([[0.0, 0.0], [20.0, 0.0], [100.0, 0.0]]),
        spot_radius_px=5.0,
        barcode_suffix_policy="auto",
        metadata={
            "analysis_role": "locked_validation",
            "cohort_id": "cohort",
            "donor_id": "donor",
            "specimen_id": "sample",
            "block_id": "block",
            "section_id": "section",
            "outer_fold": "fold_0",
            "inner_fold": "inner_0",
        },
        source_artifacts=sources,
        source_sha256=hashes,
        source_roles=roles,
    )
    expected = np.log1p(np.asarray([[0.0, 5000.0], [5000.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(artifact.observed_expression, expected)
    np.testing.assert_array_equal(artifact.nucleus_spot_index, [0, 1, -1])
    assert artifact.barcode_suffix_policy == "strip-export"
    path = tmp_path / "truth.npz"
    artifact.save_npz(path)
    loaded = SpatialTruthArtifact.from_npz(path)
    np.testing.assert_allclose(loaded.observed_expression, expected)
    np.testing.assert_array_equal(loaded.spot_library_sizes, [20.0, 40.0])
    with np.load(path, allow_pickle=False) as archive:
        assert str(archive["__contract__"].item()) == "heir.spatial_truth"
        assert int(archive["__version__"].item()) == 1
        assert all(array.dtype != object for array in archive.values())

    with pytest.raises(ValueError, match="locked target"):
        SpatialTruthArtifact(
            **{
                **artifact.__dict__,
                "analysis_role": "development",
            }
        )


def test_normalization_uses_full_transcriptome_not_panel_mass():
    counts = sparse.csr_matrix([[5.0, 5.0]], dtype=np.float32)
    normalized = normalize_panel_counts(counts, [20.0])
    np.testing.assert_allclose(normalized, np.log1p([[2500.0, 2500.0]]), rtol=1e-6)


def test_h5ad_reader_preserves_full_library_sizes_and_panel_order(tmp_path):
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    adata = anndata.AnnData(
        X=sparse.csr_matrix([[10, 0, 10], [0, 20, 20]], dtype=np.float32),
        obs=pd.DataFrame(index=["AA-1_1", "BB-1_1"]),
        var=pd.DataFrame(
            {"feature_name": ["g1", "g2", "off_panel"]},
            index=["id1", "id2", "id3"],
        ),
    )
    source = tmp_path / "visium.h5ad"
    adata.write_h5ad(source)
    loaded = read_visium_counts(
        source,
        genes=["g2", "g1"],
        gene_key="feature_name",
        chunk_size=1,
    )
    np.testing.assert_array_equal(loaded.gene_names, ["g2", "g1"])
    np.testing.assert_array_equal(loaded.library_sizes, [20.0, 40.0])
    np.testing.assert_array_equal(loaded.counts.toarray(), [[0.0, 10.0], [20.0, 0.0]])


def test_prepare_spatial_truth_cli_from_10x_matrix(tmp_path):
    matrix_dir = tmp_path / "filtered_feature_bc_matrix"
    matrix_dir.mkdir()
    (matrix_dir / "barcodes.tsv").write_text("AA-1_1\nBB-1_1\n")
    (matrix_dir / "features.tsv").write_text(
        "id1\tg1\tGene Expression\nid2\tg2\tGene Expression\nid3\toff_panel\tGene Expression\n"
    )
    # Matrix Market is feature-by-barcode; off-panel counts verify that the
    # canonical denominator uses the complete Gene Expression library.
    scipy_io.mmwrite(
        matrix_dir / "matrix.mtx",
        sparse.coo_matrix([[10, 0], [0, 20], [10, 20]], dtype=np.int32),
    )
    positions = tmp_path / "tissue_positions.csv"
    positions.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "AA-1,1,0,0,0,0\n"
        "BB-1,1,0,1,0,20\n"
    )
    scales = tmp_path / "scalefactors_json.json"
    scales.write_text(json.dumps({"spot_diameter_fullres": 10.0}))
    nuclei = tmp_path / "nuclei.csv"
    nuclei.write_text("nucleus_id,x,y\nn0,0,0\nn1,20,0\nn2,100,0\n")
    panel = tmp_path / "genes.tsv"
    panel.write_text("g2\ng1\n")

    manifest_path = tmp_path / "manifest.tsv"
    record = ManifestRecord(
        cohort_id="cohort",
        donor_id="donor",
        specimen_id="sample",
        block_id="block",
        section_id="section",
        modality="histology+spatial_transcriptomics",
        assay_platform="Visium",
        preservation="FFPE",
        tissue="breast",
        matching_tier="tier_1",
        matching_notes="test fixture",
        analysis_role="locked_validation",
        outer_fold="fold_0",
        inner_fold="inner_0",
        spatial_count_matrix_file=str(matrix_dir),
        spatial_coordinate_file=str(positions),
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerow(record.to_mapping())

    output = tmp_path / "spatial_truth.npz"
    assert (
        main(
            [
                "prepare-spatial-truth",
                "--manifest",
                str(manifest_path),
                "--section-id",
                "section",
                "--counts",
                str(matrix_dir),
                "--positions",
                str(positions),
                "--scalefactors",
                str(scales),
                "--nuclei",
                str(nuclei),
                "--genes",
                str(panel),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    truth = SpatialTruthArtifact.from_npz(output)
    np.testing.assert_array_equal(truth.gene_names, ["g2", "g1"])
    np.testing.assert_array_equal(truth.nucleus_ids, ["sample::n0", "sample::n1", "sample::n2"])
    np.testing.assert_allclose(
        truth.observed_expression,
        np.log1p([[0.0, 5000.0], [5000.0, 0.0]]),
        rtol=1e-6,
    )
    counts_loaded = read_visium_counts(matrix_dir, genes=["g2", "g1"])
    np.testing.assert_array_equal(counts_loaded.library_sizes, [20.0, 40.0])
