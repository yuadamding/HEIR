"""Scientific contracts of the local HTAPP preparation adapter."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from heir.personalized.data import (
    load_spatial_targets,
    prepare_sample,
    specimen_eligibility,
    validate_matching,
)
from heir.personalized.geometry import (
    local_protein_evidence,
    prepare_geometry,
    region_bins,
    tissue_mask_edge_filter,
    write_overlay,
)
from heir.personalized.reference import build_reference, extract_raw_counts


def test_counts_var_not_processed_gene_axis():
    adata = SimpleNamespace(
        n_obs=2,
        obsm={"counts": np.array([[1, 2, 3], [2, 1, 0]])},
        uns={"counts_var": {"gene_id": ["A", "B", "C"]}},
        var={"gene_id": ["processed_only"]},
        X=np.ones((2, 1)),
        layers={},
    )
    counts, genes = extract_raw_counts(adata, gene_id_column="gene_id")
    assert counts.shape == (2, 3)
    assert genes == ("A", "B", "C")
    with pytest.raises(ValueError, match="raw-count declaration"):
        extract_raw_counts(adata, source="X", gene_id_column="gene_id")
    adata.uns["counts_var"] = {"gene_id": ["A", "B"]}
    with pytest.raises(ValueError, match="gene axis"):
        extract_raw_counts(adata, gene_id_column="gene_id")


def test_reference_uses_linear_means_and_keeps_rare_type():
    counts = np.array([[9, 1], [1, 9], [2, 2], [0, 0]])
    reference = build_reference(
        counts,
        ["A", "B"],
        ["abundant", "abundant", "rare", "empty"],
        state_labels=["one", "one", "one", "one"],
        embedding_gene_ids=["B", "A"],
        max_states=2,
        library_size=10,
    )
    assert reference.state_types == ("abundant", "rare")
    np.testing.assert_allclose(reference.expression, [[5, 5], [5, 5]])
    np.testing.assert_allclose(reference.variance, [[16, 16], [0, 0]])
    np.testing.assert_allclose(reference.proportions, [2 / 3, 1 / 3])
    np.testing.assert_allclose(reference.embedding, np.log1p([[5, 5], [5, 5]]))
    with pytest.raises(ValueError, match="broad types"):
        build_reference(
            counts[:3], ["A", "B"], ["a", "a", "b"], embedding_gene_ids=["A"], max_states=1
        )


def test_registration_direction_units_and_global_bins():
    registration = {
        "source_frame": "protein",
        "target_frame": "he",
        "source_units": "pixel",
        "micrometres_per_pixel": [0.5, 2],
        "target_units": "um",
        "direction": "source_to_target",
        "uncertainty_um": 3,
        "affine": [[1, 0, 10], [0, 1, -10], [0, 0, 1]],
    }
    np.testing.assert_allclose(prepare_geometry([[20, 20]], registration), [[20, 30]])
    assert region_bins([[99.9, 0], [100, 0], [-0.1, 0]]) == (
        "bin:0:0",
        "bin:1:0",
        "bin:-1:0",
    )
    with pytest.raises(ValueError, match="direction"):
        prepare_geometry([[20, 20]], {**registration, "direction": "target_to_source"})


def test_local_proteins_ignore_missing_values_and_other_tissue_islands():
    args = dict(
        he_coordinates_um=[[0, 0], [100, 0]],
        protein_coordinates_um=[[1, 0], [2, 0], [0, 0]],
        observed=np.array([[True, False], [True, False], [True, True]]),
        radius_um=10,
        he_component_ids=["a", "a"],
        protein_component_ids=["a", "a", "b"],
    )
    first = local_protein_evidence(values=[[2, np.nan], [6, -999], [100, 100]], **args)
    second = local_protein_evidence(values=[[2, 1e9], [6, np.inf], [100, 100]], **args)
    np.testing.assert_allclose(first[0], [[4, 0], [0, 0]])
    np.testing.assert_array_equal(first[1], [[True, False], [False, False]])
    for left, right in zip(first, second):
        np.testing.assert_allclose(left, right)


def test_tissue_mask_excludes_thin_internal_gaps():
    mask = np.ones((3, 10), dtype=bool)
    mask[:, 5] = False
    allowed = tissue_mask_edge_filter(
        [[0.1, 0.5], [4.9, 0.5], [6.1, 0.5], [-0.1, 0.5]],
        mask,
        origin_um=[0, 0],
        pixel_size_um=[1, 1],
    )
    assert allowed(0, 1)
    assert not allowed(0, 2)
    assert not allowed(1, 2)
    assert not allowed(3, 0)


@pytest.fixture
def htapp_manifest(tmp_path):
    anndata = pytest.importorskip("anndata")
    pandas = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    parquet.write_table(
        pa.table(
            {
                "cell_id": ["n1", "n2", "n3"],
                "x_um": [0.0, 20.0, 110.0],
                "y_um": [0.0, 0.0, 0.0],
                "segmentation_quality": [1.0, 0.9, 0.9],
                "tissue_component_id": ["island"] * 3,
            }
        ),
        tmp_path / "cells.parquet",
    )
    np.savez(
        tmp_path / "features.npz",
        cell_ids=["n3", "n1", "n2"],
        image_features=np.array([[3.0, 0], [1.0, 0], [2.0, 0]]),
    )
    rna = anndata.AnnData(
        X=np.zeros((3, 1)),
        obs=pandas.DataFrame(
            {
                "broad_type": ["a", "a", "b"],
                "patient_id": ["p1"] * 3,
                "sampling_episode_id": ["episode"] * 3,
                "lesion_id": ["lesion"] * 3,
                "represented_region_id": ["region"] * 3,
            },
            index=["r1", "r2", "r3"],
        ),
        var=pandas.DataFrame(index=["processed_gene"]),
    )
    rna.obsm["counts"] = np.array([[9, 1], [8, 2], [1, 9]])
    rna.uns["counts_var"] = pandas.DataFrame({"gene_id": ["A", "B"]})
    rna.write_h5ad(tmp_path / "rna.h5ad")
    proteins = anndata.AnnData(
        X=np.array([[1.0, np.nan], [3.0, 4.0]]),
        obs=pandas.DataFrame({"tissue_component_id": ["island", "island"]}, index=["p1", "p2"]),
        var=pandas.DataFrame(index=["mA", "mB"]),
    )
    proteins.obsm["spatial"] = np.array([[0.0, 0.0], [20.0, 0.0]])
    proteins.layers["observed"] = np.array([[True, False], [True, True]])
    proteins.write_h5ad(tmp_path / "proteins.h5ad")
    registration = {
        "source_frame": "protein",
        "target_frame": "he",
        "source_units": "um",
        "target_units": "um",
        "direction": "source_to_target",
        "uncertainty_um": 5,
        "affine": np.eye(3).tolist(),
        "correspondence": "serial_section",
    }
    (tmp_path / "registration.json").write_text(json.dumps(registration))
    match = dict(
        patient_id="p1",
        sampling_episode_id="episode",
        lesion_id="lesion",
        represented_region_id="region",
        core_id="core1",
        section_id="section1",
        assay="HE",
    )
    manifest = {
        "schema_version": 1,
        "specimen_id": "s1",
        "patient_id": "p1",
        "matching": {
            "query": match,
            "reference": {
                **match,
                "core_id": "core2",
                "section_id": "dissociated",
                "assay": "snRNA-seq",
            },
            "reference_independent_of_spatial_targets": True,
            "evidence": "source metadata",
        },
        "paths": {
            "cells": "cells.parquet",
            "image_features": "features.npz",
            "rna": "rna.h5ad",
            "proteins": "proteins.h5ad",
            "registration": "registration.json",
            "spatial_targets": "targets.h5ad",
            "observation": "observation.npz",
        },
        "feature_provenance": {
            "encoder_name": "synthetic-test",
            "checkpoint_sha256": "0" * 64,
            "frozen": True,
            "contains_spatial_rna_derived_features": False,
        },
        "reference": {
            "annotation_provenance": "matched_dissociated_rna",
            "broad_type_column": "broad_type",
            "gene_id_column": "gene_id",
            "embedding_gene_ids": ["A", "B"],
            "max_states": 2,
        },
        "proteins": {
            "radius_um": 30,
            "marker_table": [
                {
                    "marker_id": f"m{gene}",
                    "gene_id": gene,
                    "interpretation": "identity",
                    "intensity_half_saturation": 1,
                    "rna_half_saturation": 1000,
                }
                for gene in ("A", "B")
            ],
        },
        "targets": {"coordinate_units": "um", "expression_scale": "nonnegative_linear"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_real_htapp_import_is_id_aligned_and_target_free(htapp_manifest):
    first = prepare_sample(htapp_manifest)
    np.testing.assert_array_equal(first.image_features[:, 0], [1, 2, 3])
    assert first.reference.gene_ids == ("A", "B")
    assert first.proteins.correspondence == "serial_section"
    np.testing.assert_allclose(first.proteins.values[:2, 0], [0.625, 0.625])
    assert not first.proteins.observed[2].any()
    # Deliberately invalid target data must not be opened by preparation.
    (htapp_manifest.parent / "targets.h5ad").write_bytes(b"not a target file")
    second = prepare_sample(htapp_manifest)
    np.testing.assert_array_equal(first.image_features, second.image_features)
    np.testing.assert_array_equal(first.reference.expression, second.reference.expression)
    rows = specimen_eligibility([htapp_manifest])
    assert rows[0]["prediction_eligible"] and rows[0]["payload_checked"]
    assert not rows[0]["evaluation_files_present"]
    assert rows[0]["n_rna_genes"] == 2


def test_target_loading_is_explicit_and_requires_availability(htapp_manifest):
    import anndata
    import pandas

    target = anndata.AnnData(
        X=np.array([[10.0, 20.0], [5.0, 0.0]]),
        obs=pandas.DataFrame(
            {"patient_id": ["p1", "p1"], "specimen_id": ["s1", "s1"]}, index=["bin:0:0", "bin:1:0"]
        ),
        var=pandas.DataFrame(index=["A", "B"]),
    )
    target.obsm["spatial"] = np.array([[50.0, 50.0], [150.0, 50.0]])
    target.write_h5ad(htapp_manifest.parent / "targets.h5ad")
    with pytest.raises(ValueError, match="gene availability"):
        load_spatial_targets(htapp_manifest)
    target.layers["observed"] = np.array([[True, True], [True, False]])
    target.write_h5ad(htapp_manifest.parent / "targets.h5ad")
    actual = load_spatial_targets(htapp_manifest)
    assert actual.observation_ids == ("bin:0:0", "bin:1:0")
    assert actual.patient_id == "p1"
    np.testing.assert_array_equal(actual.observed, target.layers["observed"])
    target.obs["patient_id"] = ["wrong", "wrong"]
    target.write_h5ad(htapp_manifest.parent / "targets.h5ad")
    with pytest.raises(ValueError, match="target payload patient_id"):
        load_spatial_targets(htapp_manifest)


def test_matching_is_more_specific_than_patient_and_forbids_transferred_labels(htapp_manifest):
    manifest = json.loads(htapp_manifest.read_text())
    manifest["matching"]["reference"]["lesion_id"] = "different lesion"
    with pytest.raises(ValueError, match="lesion_id differs"):
        validate_matching(manifest)
    manifest["matching"]["reference"]["lesion_id"] = "lesion"
    manifest["reference"]["broad_type_column"] = "RCTD"
    htapp_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="spatial-RNA-derived"):
        prepare_sample(htapp_manifest)


def test_rna_payload_identity_is_checked_not_only_manifest(htapp_manifest):
    import anndata

    path = htapp_manifest.parent / "rna.h5ad"
    rna = anndata.read_h5ad(path)
    rna.obs["patient_id"] = ["wrong"] * 3
    rna.write_h5ad(path)
    with pytest.raises(ValueError, match="RNA payload patient"):
        prepare_sample(htapp_manifest)
    rna.obs["patient_id"] = ["p1"] * 3
    rna.obs["lesion_id"] = ["wrong"] * 3
    rna.write_h5ad(path)
    with pytest.raises(ValueError, match="RNA payload lesion_id"):
        prepare_sample(htapp_manifest)


def test_overlay_labels_coverage_without_claiming_registration(tmp_path):
    path = tmp_path / "overlay.svg"
    write_overlay(
        path,
        [[0, 0], [200, 100]],
        protein_coordinates_um=[[10, 10]],
        spatial_coordinates_um=[[50, 50]],
        title="synthetic & only",
    )
    content = path.read_text()
    assert "synthetic &amp; only" in content
    assert "100 micrometres" in content
    assert "Protein measurements" in content and "Spatial RNA observations" in content
    assert "Registration requires visual review" in content
