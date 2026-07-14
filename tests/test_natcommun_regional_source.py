from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_natcommun_regional_source.py"
    spec = importlib.util.spec_from_file_location("build_natcommun_regional_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


@pytest.mark.parametrize("value", ("B1", "B1_2", "breast", "dlbcl"))
def test_constant_text_preserves_multicharacter_cohort_labels(value: str) -> None:
    observed = builder._constant_text(3, value)
    assert observed.tolist() == [value, value, value]
    assert observed.dtype.itemsize >= len(value) * np.dtype("U1").itemsize


def test_frozen_protocol_keeps_regional_one_step_scope_and_b2_sensitivity(tmp_path: Path) -> None:
    protocol_path = (
        Path(__file__).parents[1] / "configs" / "natcommun_matched_regional_protocol.json"
    )
    protocol = builder._load_protocol(protocol_path)
    assert protocol["observation_level"] == "Visium_v2_spot_regional"
    assert protocol["iteration_gate"]["iterative_rounds_in_first_experiment"] == 0
    assert protocol["primary_endpoints"][-1] == "within_type_residual_state"
    assert protocol["failed_reference_sensitivity_donors"] == ["B2"]
    assert protocol["h_and_e_encoder"]["repository"] == "bioptimus/H-optimus-1"
    assert builder.DEFAULT_ENCODER_MANIFEST.name == "hoptimus1.json"
    assert builder.DEFAULT_MODEL_DIR.name == "H-optimus-1"

    weakened = dict(protocol)
    weakened["bank_conditions"] = ["natural_composition"]
    changed = tmp_path / "weakened.json"
    changed.write_text(json.dumps(weakened), encoding="utf-8")
    with pytest.raises(ValueError, match="scope/endpoints/bank conditions"):
        builder._load_protocol(changed)

    remapped = json.loads(protocol_path.read_text(encoding="utf-8"))
    remapped["sections"][0]["h5ad_donor"] = "0"
    changed.write_text(json.dumps(remapped), encoding="utf-8")
    with pytest.raises(ValueError, match="section-to-donor/H5AD/H&E mapping"):
        builder._load_protocol(changed)


def _write_filtered_matrix(path: Path) -> None:
    # Gene-by-barcode CSC:
    #              A  B
    # ENSG1 / G1   2  0
    # ENSG2 / G2   1  2
    # ENSG3 / O    1  1
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray([b"A-1", b"B-1"]))
        matrix.create_dataset("shape", data=np.asarray([3, 2], dtype=np.int64))
        matrix.create_dataset("data", data=np.asarray([2, 1, 1, 2, 1], dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray([0, 1, 2, 1, 2], dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray([0, 3, 5], dtype=np.int64))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray([b"ENSG1", b"ENSG2", b"ENSG3"]))
        features.create_dataset("name", data=np.asarray([b"G1", b"G2", b"OTHER"]))
        features.create_dataset(
            "feature_type",
            data=np.asarray([b"Gene Expression", b"Gene Expression", b"Gene Expression"]),
        )


def _write_feature_catalog_h5ad(path: Path, genes: list[str], ensembl: list[str]) -> None:
    with h5py.File(path, "w") as handle:
        var = handle.create_group("var")
        var.create_dataset("_index", data=np.asarray(ensembl, dtype="S"))
        var.create_dataset("feature_name", data=np.asarray(genes, dtype="S"))


def _write_feature_catalog_visium(path: Path, genes: list[str], ensembl: list[str]) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        features = handle.create_group("matrix").create_group("features")
        features.create_dataset("id", data=np.asarray(ensembl, dtype="S"))
        features.create_dataset("name", data=np.asarray(genes, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray([b"Gene Expression"] * len(genes)))


def test_broad_panel_is_metadata_only_and_common_to_every_modal_input(tmp_path: Path) -> None:
    genes = list(builder.SELECTED_GENES)
    genes.extend(f"FILLER_{index:03d}" for index in range(320 - len(genes)))
    ensembl = [f"ENSG{index:011d}" for index in range(len(genes))]
    # Two filler symbols sharing one Ensembl identity are not a one-to-one molecular target.
    ensembl[-1] = ensembl[-2]
    h5ad_files = {}
    for kind in ("breast", "dlbcl", "lung"):
        path = tmp_path / f"{kind}.h5ad"
        _write_feature_catalog_h5ad(path, genes, ensembl)
        h5ad_files[kind] = path.name
    matrices = []
    for section in ("S1", "S2"):
        path = tmp_path / section / "outs" / "filtered_feature_bc_matrix.h5"
        _write_feature_catalog_visium(path, genes, ensembl)
        matrices.append(path)

    panel = builder._broad_gene_panel({"h5ad_files": h5ad_files}, tmp_path, tuple(matrices))
    assert len(panel.gene_names) == 318
    assert set(builder.SELECTED_GENES) <= set(panel.gene_names)
    assert len(set(panel.ensembl_ids)) == len(panel.ensembl_ids)
    assert panel.receipt["uses_expression_values"] is False
    assert panel.receipt["uses_spatial_outcomes"] is False
    assert panel.receipt["outer_training_only_variance_selection_and_PCA_required"] is True
    assert len(panel.receipt["catalogs"]) == 5


def _write_molecule_info(path: Path) -> None:
    # One row per corrected unique UMI.  ``count`` is read support and must not weight expression.
    barcodes = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.uint32)
    features = np.asarray([0, 0, 1, 2, 1, 1, 2], dtype=np.uint32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("barcodes", data=np.asarray([b"A-1", b"B-1"]))
        handle.create_dataset("barcode_idx", data=barcodes)
        handle.create_dataset("feature_idx", data=features)
        handle.create_dataset("umi", data=np.asarray([11, 12, 21, 31, 41, 42, 51], dtype=np.uint32))
        handle.create_dataset("library_idx", data=np.zeros(len(barcodes), dtype=np.uint16))
        handle.create_dataset("gem_group", data=np.ones(len(barcodes), dtype=np.uint16))
        handle.create_dataset("count", data=np.asarray([2, 9, 1, 4, 7, 1, 3], dtype=np.uint32))
        feature_group = handle.create_group("features")
        feature_group.create_dataset("id", data=np.asarray([b"ENSG1", b"ENSG2", b"ENSG3"]))


def test_filtered_matrix_positions_and_unique_umi_halves_reconstruct_exactly(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "filtered_feature_bc_matrix.h5"
    molecule_path = tmp_path / "molecule_info.h5"
    positions_path = tmp_path / "tissue_positions.csv"
    _write_filtered_matrix(matrix_path)
    _write_molecule_info(molecule_path)
    positions_path.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "A-1,1,2,3,100.5,200.5\n"
        "B-1,1,4,5,300.5,400.5\n",
        encoding="utf-8",
    )

    filtered = builder._read_filtered_matrix(matrix_path, ("G1", "G2"))
    np.testing.assert_array_equal(filtered.selected_counts, [[2, 1], [0, 2]])
    np.testing.assert_array_equal(filtered.total_counts, [4, 3])
    assert filtered.selected_feature_ids == ("ENSG1", "ENSG2")
    positions = builder._read_tissue_positions(positions_path, filtered.barcodes)
    np.testing.assert_array_equal(positions.array_row_col, [[2, 3], [4, 5]])
    np.testing.assert_allclose(positions.pixel_xy, [[200.5, 100.5], [400.5, 300.5]])

    first = builder._split_molecule_info(
        molecule_path,
        positions.barcodes,
        filtered.selected_feature_ids,
        filtered.broad_feature_ids,
        filtered.all_gene_feature_ids,
        filtered.selected_counts,
        filtered.broad_counts,
        filtered.total_counts,
        section="fixture_A",
        chunk_size=2,
    )
    second = builder._split_molecule_info(
        molecule_path,
        positions.barcodes,
        filtered.selected_feature_ids,
        filtered.broad_feature_ids,
        filtered.all_gene_feature_ids,
        filtered.selected_counts,
        filtered.broad_counts,
        filtered.total_counts,
        section="fixture_A",
        chunk_size=5,
    )
    np.testing.assert_array_equal(first.half_a, second.half_a)
    np.testing.assert_array_equal(first.half_b, second.half_b)
    np.testing.assert_array_equal(first.half_a + first.half_b, filtered.selected_counts)
    np.testing.assert_array_equal(first.total_half_a + first.total_half_b, filtered.total_counts)
    assert builder._csr_sum_equals(filtered.broad_counts, first.broad_half_a, first.broad_half_b)
    subset = builder._subset_csr_rows(
        filtered.broad_counts, np.asarray([1], dtype=np.int64), "fixture subset"
    )
    np.testing.assert_array_equal(_csr_dense(subset), [[0, 2]])
    assert first.receipt["section"] == "fixture_A"
    assert first.receipt["gem_group_in_identity"] is True
    assert first.receipt["halves_are_disjoint_by_construction"] is True
    assert first.receipt["selected_reconstruction_exact"] is True
    assert first.receipt["source_semantics"].startswith("one molecule_info row")

    other_section = builder._split_molecule_info(
        molecule_path,
        positions.barcodes,
        filtered.selected_feature_ids,
        filtered.broad_feature_ids,
        filtered.all_gene_feature_ids,
        filtered.selected_counts,
        filtered.broad_counts,
        filtered.total_counts,
        section="fixture_B",
        chunk_size=5,
    )
    assert other_section.receipt["salt"] != first.receipt["salt"]
    assert not np.array_equal(other_section.half_a, first.half_a)


def test_positions_fail_closed_on_filtered_in_tissue_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    path.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "A-1,1,0,0,1,1\n"
        "B-1,1,0,1,1,2\n"
        "C-1,1,0,2,1,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="barcodes and in-tissue positions differ"):
        builder._read_tissue_positions(path, np.asarray(["A-1", "B-1"]))


def test_unique_umi_split_fails_closed_when_matrix_cannot_be_reconstructed(tmp_path: Path) -> None:
    matrix_path = tmp_path / "filtered.h5"
    molecule_path = tmp_path / "molecule.h5"
    _write_filtered_matrix(matrix_path)
    _write_molecule_info(molecule_path)
    filtered = builder._read_filtered_matrix(matrix_path, ("G1", "G2"))
    wrong = filtered.selected_counts.copy()
    wrong[0, 0] += 1
    with pytest.raises(ValueError, match="do not reconstruct selected"):
        builder._split_molecule_info(
            molecule_path,
            filtered.barcodes,
            filtered.selected_feature_ids,
            filtered.broad_feature_ids,
            filtered.all_gene_feature_ids,
            wrong,
            filtered.broad_counts,
            filtered.total_counts,
            section="fixture_A",
            chunk_size=3,
        )


def _categorical(group: h5py.Group, name: str, values: list[str]) -> None:
    categories = sorted(set(values))
    lookup = {value: index for index, value in enumerate(categories)}
    node = group.create_group(name)
    node.create_dataset("categories", data=np.asarray(categories, dtype="S"))
    node.create_dataset(
        "codes", data=np.asarray([lookup[value] for value in values], dtype=np.int8)
    )


def _write_h5ad(path: Path) -> None:
    # Cell-by-gene CSR: [[2, 0, 4], [0, 3, 0], [1, 1, 0]].
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("X")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = np.asarray([3, 3], dtype=np.int64)
        matrix.create_dataset("data", data=np.asarray([2, 4, 3, 1, 1], dtype=np.float32))
        matrix.create_dataset("indices", data=np.asarray([0, 2, 1, 0, 1], dtype=np.int64))
        matrix.create_dataset("indptr", data=np.asarray([0, 2, 3, 5], dtype=np.int64))
        var = handle.create_group("var")
        var.create_dataset("_index", data=np.asarray([b"ENSG1", b"ENSG2", b"ENSG3"]))
        var.create_dataset("feature_name", data=np.asarray([b"G1", b"G2", b"OTHER"]))
        obs = handle.create_group("obs")
        obs.create_dataset("_index", data=np.asarray([b"c1", b"c2", b"c3"]))
        for name, values in {
            "donor_id": ["raw1", "raw1", "raw2"],
            "sample_id": ["sample1", "sample1", "sample2"],
            "Level1": ["Tumor", "Immune", "Tumor"],
            "Level2": ["T", "I", "T"],
            "Level3": ["T1", "I1", "T2"],
            "Harmonised_Level4": ["T1_state", "I1_state", "T2_state"],
            "DV200_percent": ["50", "50", "40"],
            "block_age_months": ["12", "12", "24"],
        }.items():
            _categorical(obs, name, values)
        obs.create_dataset("nFeature_RNA", data=np.asarray([2, 1, 2], dtype=np.int32))
        obs.create_dataset("percent_mt", data=np.asarray([1.0, 2.0, 3.0]))
        obs.create_dataset("percent_ribo", data=np.asarray([4.0, 5.0, 6.0]))
        obs.create_dataset("percent_hb", data=np.asarray([0.0, 0.5, 1.0]))


def test_h5ad_reader_preserves_raw_counts_types_qc_and_b2_sensitivity(tmp_path: Path) -> None:
    path = tmp_path / "one.h5ad"
    _write_h5ad(path)
    protocol = {
        "primary_donors": ["D1"],
        "failed_reference_sensitivity_donors": ["B2"],
        "h5ad_files": {"test": path.name},
        "sections": [
            {"h5ad": "test", "h5ad_donor": "raw1", "donor": "D1"},
            {"h5ad": "test", "h5ad_donor": "raw2", "donor": "B2"},
        ],
    }
    data = builder._read_chromium_h5ads(protocol, tmp_path, ("G1", "G2"))
    np.testing.assert_array_equal(data.counts, [[2, 0], [0, 3], [1, 1]])
    np.testing.assert_array_equal(data.total_counts, [6, 3, 2])
    assert data.donor_ids.tolist() == ["D1", "D1", "B2"]
    assert data.raw_h5ad_donor_ids.tolist() == ["raw1", "raw1", "raw2"]
    assert data.sample_ids.tolist() == ["sample1", "sample1", "sample2"]
    assert data.indication_ids.tolist() == ["test", "test", "test"]
    assert data.primary_eligible.tolist() == [True, True, False]
    assert data.level1.tolist() == ["Tumor", "Immune", "Tumor"]
    assert data.level4.tolist() == ["T1_state", "I1_state", "T2_state"]
    np.testing.assert_allclose(data.percent_mt, [1, 2, 3])
    assert data.dv200.tolist() == ["50", "50", "40"]
    assert data.input_receipts[0]["raw_count_matrix"] == "X"
    assert data.input_receipts[0]["ambient_corrected_layer_not_used"] == "layers/SoupX"
    assert data.broad_counts.data.dtype == np.int32
    assert data.broad_counts.indices.dtype == np.int32
    assert data.broad_counts.indptr.dtype == np.int64
    assert data.broad_counts.shape == (3, 2)
    np.testing.assert_array_equal(
        _csr_dense(data.broad_counts), np.asarray([[2, 0], [0, 3], [1, 1]])
    )


def test_real_cohort_mapping_requires_independent_h5ad_cell_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "breast.h5ad"
    _write_h5ad(path)
    protocol = {
        "primary_donors": ["B1"],
        "failed_reference_sensitivity_donors": ["B2"],
        "h5ad_files": {"breast": path.name},
        "sections": [
            {"h5ad": "breast", "h5ad_donor": "raw1", "donor": "B1"},
            {"h5ad": "breast", "h5ad_donor": "raw2", "donor": "B2"},
        ],
    }
    with pytest.raises(ValueError, match="cell prefixes contradict the frozen mapping"):
        builder._read_chromium_h5ads(protocol, tmp_path, ("G1", "G2"))


def _csr_dense(value) -> np.ndarray:
    output = np.zeros(value.shape, dtype=np.int32)
    for row in range(value.shape[0]):
        start, stop = int(value.indptr[row]), int(value.indptr[row + 1])
        output[row, value.indices[start:stop]] = value.data[start:stop]
    return output


class _FixtureEncoder:
    feature_width = 3
    manifest_sha256 = "a" * 64

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, patches: np.ndarray) -> np.ndarray:
        self.calls += 1
        means = patches.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return np.column_stack((means, means + 1, means + 2)).astype(np.float32)


def test_112um_embedding_cache_is_identity_bound_and_resumable(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("zarr")
    image_path = tmp_path / "section.tif"
    tifffile.imwrite(
        image_path,
        np.full((64, 64, 3), 100, dtype=np.uint8),
        photometric="rgb",
        tile=(16, 16),
    )
    scalefactors = tmp_path / "scalefactors_json.json"
    scalefactors.write_text(json.dumps({"spot_diameter_fullres": 11.0}), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = SimpleNamespace(
        path=manifest_path,
        sha256="a" * 64,
        encoder_id="fixture",
        feature_width=3,
        input_pixels=224,
    )
    barcodes = np.asarray(["A-1", "B-1"])
    pixels = np.asarray([[32.0, 32.0], [2.0, 2.0]])
    encoder = _FixtureEncoder()
    parity = {"status": "passed", "receipt_sha256": "b" * 64}

    first, first_receipt = builder._section_embeddings(
        section="S",
        image_path=image_path,
        scalefactors_path=scalefactors,
        barcodes=barcodes,
        pixel_xy=pixels,
        encoder=encoder,
        encoder_manifest=manifest,
        cache_dir=tmp_path / "cache",
        batch_size=1,
        encoder_parity=parity,
    )
    assert encoder.calls == 2
    assert first.dtype == np.float16
    assert first_receipt["crop"]["physical_width_um"] == 112.0
    assert first_receipt["crop"]["crop_width_fullres_pixels"] == 22
    assert first_receipt["encoder_resampling"]["implementation"] == "Pillow bicubic"
    assert first_receipt["encoder_resampling"]["resampling_count"] == 1
    assert first_receipt["cache_status"] == "created"

    second, second_receipt = builder._section_embeddings(
        section="S",
        image_path=image_path,
        scalefactors_path=scalefactors,
        barcodes=barcodes,
        pixel_xy=pixels,
        encoder=encoder,
        encoder_manifest=manifest,
        cache_dir=tmp_path / "cache",
        batch_size=2,
        encoder_parity=parity,
    )
    assert encoder.calls == 2
    np.testing.assert_array_equal(second, first)
    assert second_receipt["cache_status"] == "reused"

    _, third_receipt = builder._section_embeddings(
        section="S",
        image_path=image_path,
        scalefactors_path=scalefactors,
        barcodes=barcodes,
        pixel_xy=pixels,
        encoder=encoder,
        encoder_manifest=manifest,
        cache_dir=tmp_path / "cache",
        batch_size=2,
        encoder_parity={"status": "passed", "receipt_sha256": "c" * 64},
    )
    assert encoder.calls == 3
    assert third_receipt["cache_status"] == "created"


def test_large_strip_reader_decodes_once_and_registration_fails_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("zarr")
    path = tmp_path / "strip.tif"
    tifffile.imwrite(path, np.full((64, 64, 3), 90, dtype=np.uint8), photometric="rgb")
    monkeypatch.setattr(builder._TiffRegionReader, "MAX_REGION_CHUNK_BYTES", 1)
    with builder._TiffRegionReader(path) as reader:
        assert reader.storage_mode == "whole_image_decoded_once_for_large_strip"
        assert reader._whole_image is None
        first = reader.crop((32, 32), 16)
        cached_id = id(reader._whole_image)
        second = reader.crop((24, 24), 16)
        assert id(reader._whole_image) == cached_id
        np.testing.assert_array_equal(first, second)
        with pytest.raises(ValueError, match="centers fall outside"):
            builder._registration_qc(reader, np.asarray([[100.0, 100.0]]), 16)
    retained, receipt = builder._h_and_e_observation_selection(
        path,
        np.asarray(["inside-1", "outside-1"]),
        np.asarray([[32.0, 32.0], [100.0, 100.0]]),
    )
    np.testing.assert_array_equal(retained, [True, False])
    assert receipt["excluded_barcode_ids"] == ["outside-1"]
    assert receipt["uses_spatial_outcomes"] is False


def test_pyramidal_tiff_reader_selects_level_zero_array(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("zarr")
    path = tmp_path / "pyramid.tif"
    level_zero = np.full((64, 64, 3), 77, dtype=np.uint8)
    level_one = np.full((32, 32, 3), 88, dtype=np.uint8)
    with tifffile.TiffWriter(path) as writer:
        writer.write(level_zero, photometric="rgb", tile=(16, 16), subifds=1)
        writer.write(level_one, photometric="rgb", tile=(16, 16), subfiletype=1)
    with builder._TiffRegionReader(path) as reader:
        assert (reader.height, reader.width) == (64, 64)
        np.testing.assert_array_equal(reader.crop((32, 32), 16), level_zero[24:40, 24:40])


def test_manual_pathology_is_explicitly_unavailable_without_barcode_mapping() -> None:
    labels, groups, receipt = builder._manual_pathology_labels(
        None, np.asarray(["B1_2", "D1"]), np.asarray(["A-1", "B-1"])
    )
    assert labels.tolist() == ["__unavailable__", "__unavailable__"]
    assert groups.tolist() == ["__unavailable__", "__unavailable__"]
    assert receipt["status"] == "unavailable"
    assert "aggregate" in receipt["reason"]
    assert receipt["outcome_independent_manual_labels_only"] is True


def test_manual_pathology_requires_blinded_h_and_e_provenance(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "section_id,barcode_id,pathologist_annotation,grouped_pathology_annotation\n"
        "S,A-1,tumor,tumor\nS,B-1,stroma,stroma\n",
        encoding="utf-8",
    )
    sections = np.asarray(["S", "S"])
    barcodes = np.asarray(["A-1", "B-1"])
    with pytest.raises(ValueError, match="provenance manifest"):
        builder._manual_pathology_labels(csv_path, sections, barcodes)
    provenance = tmp_path / "labels.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema": "heir.h_and_e_pathology_annotations.v1",
                "annotation_file_sha256": builder._sha256_file(csv_path),
                "source_modality": "H&E_only",
                "barcode_keyed": True,
                "blinded_to_spatial_transcriptomics": True,
                "uses_spatial_expression": False,
                "uses_Visium_clusters": False,
                "uses_Cell2location": False,
            }
        ),
        encoding="utf-8",
    )
    labels, grouped, receipt = builder._manual_pathology_labels(
        csv_path, sections, barcodes, provenance
    )
    assert labels.tolist() == ["tumor", "stroma"]
    assert grouped.tolist() == ["tumor", "stroma"]
    assert receipt["status"] == "available_complete"
    assert receipt["blinded_to_spatial_transcriptomics"] is True


def test_hoptimus_parity_receipt_is_mandatory_and_manifest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = SimpleNamespace(revision="rev", sha256="a" * 64, input_pixels=224)
    path = tmp_path / "parity.json"
    with pytest.raises(ValueError, match="parity receipt is required"):
        builder._load_encoder_parity_receipt(path, manifest)
    native_pixels = int(round(112.0 / 0.2125))
    probe_input = (
        np.arange(native_pixels * native_pixels * 3, dtype=np.uint32).reshape(
            native_pixels, native_pixels, 3
        )
        % 251
    ).astype(np.uint8)
    probe_output = builder._resize_hoptimus_batch(probe_input[None], 224)[0]
    path.write_text(
        json.dumps(
            {
                "schema": "heir.hoptimus1_official_local_parity.v1",
                "status": "passed",
                "passed": True,
                "repository": "bioptimus/H-optimus-1",
                "revision": "rev",
                "encoder_manifest_sha256": "a" * 64,
                "implementation_sha256": builder._sha256_file(builder.ENCODER_PARITY_QUALIFIER),
                "production_runtime_contract": {
                    "code_sha256": dict(builder.FROZEN_ENCODER_RUNTIME_SHA256),
                    "resampling_probe": {
                        "input_shape": list(probe_input.shape),
                        "input_dtype": str(probe_input.dtype),
                        "input_sha256": hashlib.sha256(probe_input.tobytes()).hexdigest(),
                        "output_shape": list(probe_output.shape),
                        "output_dtype": str(probe_output.dtype),
                        "output_sha256": hashlib.sha256(probe_output.tobytes()).hexdigest(),
                        "implementation": "Pillow.Image.Resampling.BICUBIC",
                        "resampling_count": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "FROZEN_ENCODER_PARITY_RECEIPT_SHA256", builder._sha256_file(path))
    monkeypatch.setattr(
        builder,
        "FROZEN_ENCODER_PARITY_IMPLEMENTATION_SHA256",
        builder._sha256_file(builder.ENCODER_PARITY_QUALIFIER),
    )
    assert builder._load_encoder_parity_receipt(path, manifest)["status"] == "passed"


def test_fastq_provenance_binds_invocation_paths_and_ena_sizes(tmp_path: Path) -> None:
    sample = "SAMPLE"
    directory = tmp_path / "arrayexpress" / "E-MTAB-14560" / "ENA_submitted" / "ERR000001"
    directory.mkdir(parents=True)
    names = [f"{sample}_S1_L001_{kind}_001.fastq.gz" for kind in ("I1", "I2", "R1", "R2")]
    sizes = []
    for index, name in enumerate(names, start=1):
        payload = bytes([index]) * index
        (directory / name).write_bytes(payload)
        sizes.append(len(payload))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    report = metadata / "ERP165490_ena_run_filereport.tsv"
    report.write_text(
        "submitted_ftp\tsubmitted_bytes\n"
        + ";".join(f"ftp.example/{name}" for name in names)
        + "\t"
        + ";".join(map(str, sizes))
        + "\n",
        encoding="utf-8",
    )
    invocation = f'            read_path:      "{directory.resolve()}",\n'
    receipt = builder._fastq_input_provenance(tmp_path, sample, invocation)
    assert receipt["read_paths_match_invocation"] is True
    assert receipt["all_local_sizes_match_ENA_submitted_bytes"] is True
    assert len(receipt["files"]) == 4


def test_program_membership_is_fixed_and_coordinate_controls_are_section_local() -> None:
    membership = builder._program_membership()
    assert 3 <= len(builder.PROGRAM_NAMES) <= 8
    assert builder.BROAD_TRAINING_ONLY_PCA_DIMENSION_RANGE == (20, 50)
    assert membership.dtype == np.bool_
    assert membership.shape == (len(builder.PROGRAM_NAMES), len(builder.SELECTED_GENES))
    assert membership.all(axis=1).sum() == 0  # no program is the entire union
    np.testing.assert_array_equal(
        membership.sum(axis=1), [len(builder.FROZEN_PROGRAMS[x]) for x in builder.PROGRAM_NAMES]
    )
    assert set(builder.PROGRAM_CLASSIFICATIONS) == set(builder.PROGRAM_NAMES)
    assert all(
        "within_type_state" in builder.PROGRAM_CLASSIFICATIONS[name]
        for name in builder.PROGRAM_NAMES
    )

    features = builder._coordinate_features(
        np.asarray(["A", "A", "B", "B"]),
        np.asarray([[10, 20], [20, 40], [1000, 2000], [1100, 2200]], dtype=float),
    )
    np.testing.assert_allclose(features[[0, 2], :2], 0)
    np.testing.assert_allclose(features[[1, 3], :2], 1)
