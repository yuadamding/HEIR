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
    mask_component_ids,
    prepare_geometry,
    region_bins,
    regional_protein_evidence,
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


def test_regional_proteins_keep_common_geometry_and_report_excluded_tissue():
    args = dict(
        cell_ids=("c1", "c2", "c3"),
        he_coordinates_um=[[0, 0], [20, 0], [100, 0]],
        protein_coordinates_um=[[1, 0], [2, 0], [0, 0]],
        observed=np.array([[True, False], [True, False], [True, True]]),
        bin_size_um=100,
        he_component_ids=["a", "a", "a"],
        protein_component_ids=["a", "a", "b"],
    )
    first = regional_protein_evidence(values=[[2, np.nan], [6, -999], [100, 100]], **args)
    second = regional_protein_evidence(values=[[2, 1e9], [6, np.inf], [100, 100]], **args)
    np.testing.assert_allclose(first[0], [[4, 0]])
    np.testing.assert_array_equal(first[1], [[True, False]])
    np.testing.assert_array_equal(first[3].columns, [0, 1])
    np.testing.assert_allclose(first[3].weights, [0.5, 0.5])
    assert first[4] == {
        "shared_regions": 1,
        "excluded_he_regions": 1,
        "excluded_protein_regions": 1,
        "unobserved_region_markers": 1,
    }
    for left, right in zip(first[:3], second[:3]):
        np.testing.assert_allclose(left, right)


def test_region_marker_missingness_does_not_change_spatial_kernel():
    args = dict(
        cell_ids=("c1", "c2"),
        he_coordinates_um=[[0, 0], [50, 0]],
        protein_coordinates_um=[[0, 0], [50, 0]],
        values=[[1, np.nan], [0, 1]],
        observed=np.array([[True, False], [True, True]]),
        he_component_ids=["a", "a"],
        protein_component_ids=["a", "a"],
        quality=np.array([[1.0, 0], [0.2, 1.0]]),
    )
    values, observed, quality, operator, _ = regional_protein_evidence(**args)
    np.testing.assert_allclose(values, [[0.5, 0]])
    np.testing.assert_array_equal(observed, [[True, False]])
    np.testing.assert_allclose(quality, [[0.6, 0]])
    np.testing.assert_allclose(operator.weights, [0.5, 0.5])
    args["quality"][1, 0] = 0
    values, observed, quality, operator, _ = regional_protein_evidence(**args)
    assert not observed.any()
    np.testing.assert_allclose(values, 0)
    np.testing.assert_allclose(quality, 0)
    np.testing.assert_allclose(operator.weights, [0.5, 0.5])


def test_region_means_do_not_prefer_artificial_mixing():
    import torch

    args = dict(
        cell_ids=("c1", "c2"),
        he_coordinates_um=[[0, 0], [50, 0]],
        protein_coordinates_um=[[0, 0], [50, 0]],
        values=[[1], [0]],
        observed=np.ones((2, 1), dtype=bool),
        he_component_ids=["a", "a"],
        protein_component_ids=["a", "a"],
    )
    actual = torch.tensor([[1.0], [0.0]])
    mixed = torch.tensor([[0.5], [0.5]])
    coarse = regional_protein_evidence(**args)
    np.testing.assert_allclose(coarse[3].aggregate(actual), coarse[0])
    np.testing.assert_allclose(coarse[3].aggregate(mixed), coarse[0])
    fine = regional_protein_evidence(**args, bin_size_um=25)
    np.testing.assert_allclose(fine[3].aggregate(actual), fine[0])
    assert np.any(np.abs(fine[3].aggregate(mixed).numpy() - fine[0]) > 0)


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
    components = mask_component_ids(
        [[0.1, 0.5], [4.9, 0.5], [6.1, 0.5]],
        ["one_declared_component"] * 3,
        mask,
        origin_um=[0, 0],
        pixel_size_um=[1, 1],
    )
    assert components[0] == components[1]
    assert components[1] != components[2]


@pytest.fixture
def htapp_manifest(tmp_path):
    anndata = pytest.importorskip("anndata")
    pandas = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    match = dict(
        patient_id="p1",
        sampling_episode_id="episode",
        lesion_id="lesion",
        represented_region_id="region",
        core_id="core1",
        section_id="section1",
        specimen_id="s1",
    )
    protein_identity = {**match, "section_id": "section2", "specimen_id": "protein_section"}
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
        obs=pandas.DataFrame(
            {
                "tissue_component_id": ["island", "island"],
                **{field: [value] * 2 for field, value in protein_identity.items()},
            },
            index=["p1", "p2"],
        ),
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
        "source_identity": protein_identity,
        "target_identity": match,
    }
    (tmp_path / "registration.json").write_text(json.dumps(registration))
    manifest = {
        "schema_version": 2,
        "specimen_id": "s1",
        "patient_id": "p1",
        "matching": {
            "query": {**match, "assay": "HE"},
            "reference": {
                **match,
                "core_id": "core2",
                "section_id": "dissociated",
                "assay": "snRNA-seq",
            },
            "protein": {**protein_identity, "assay": "multiplex_IF"},
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
            "bin_size_um": 100,
            "origin_um": [0, 0],
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
    np.testing.assert_allclose(first.proteins.values[:, 0], [0.625])
    assert not first.proteins.observed[:, 1].any()
    assert first.proteins.coverage["excluded_he_regions"] == 1
    np.testing.assert_allclose(first.proteins.operator.weights, [0.5, 0.5])
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


@pytest.mark.parametrize(
    "field",
    [
        "patient_id",
        "sampling_episode_id",
        "lesion_id",
        "represented_region_id",
        "core_id",
        "section_id",
        "specimen_id",
    ],
)
def test_protein_payload_identity_rejected_before_evidence(htapp_manifest, monkeypatch, field):
    import anndata

    from heir.personalized import data

    def unexpected_evidence(**kwargs):
        pytest.fail("wrong biological identity reached ProteinEvidence construction")

    monkeypatch.setattr(data, "ProteinEvidence", unexpected_evidence)
    path = htapp_manifest.parent / "proteins.h5ad"
    protein = anndata.read_h5ad(path)
    protein.obs[field] = ["wrong_identity"] * protein.n_obs
    protein.write_h5ad(path)
    with pytest.raises(ValueError, match=f"protein payload {field}"):
        prepare_sample(htapp_manifest)


@pytest.mark.parametrize(
    "measurement_ids, message",
    [
        (["duplicate", "duplicate"], "protein measurement IDs must be unique"),
        (["", "valid"], "protein measurement ID is unresolved"),
    ],
)
@pytest.mark.filterwarnings("ignore:Observation names are not unique:UserWarning")
def test_protein_measurement_ids_rejected_before_evidence(
    htapp_manifest, monkeypatch, measurement_ids, message
):
    import anndata

    from heir.personalized import data

    def unexpected_evidence(**kwargs):
        pytest.fail("invalid protein measurement IDs reached ProteinEvidence construction")

    monkeypatch.setattr(data, "ProteinEvidence", unexpected_evidence)
    path = htapp_manifest.parent / "proteins.h5ad"
    protein = anndata.read_h5ad(path)
    protein.obs_names = measurement_ids
    protein.write_h5ad(path)
    with pytest.raises(ValueError, match=message):
        prepare_sample(htapp_manifest)


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("field", ["patient_id", "section_id", "specimen_id"])
def test_registration_identity_rejected_before_evidence(htapp_manifest, monkeypatch, side, field):
    from heir.personalized import data

    def unexpected_evidence(**kwargs):
        pytest.fail("wrong registration identity reached ProteinEvidence construction")

    monkeypatch.setattr(data, "ProteinEvidence", unexpected_evidence)
    path = htapp_manifest.parent / "registration.json"
    registration = json.loads(path.read_text())
    registration[f"{side}_identity"][field] = "wrong_identity"
    path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match=f"registration {side}_identity {field}"):
        prepare_sample(htapp_manifest)


@pytest.mark.parametrize("field", ["patient_id", "specimen_id"])
def test_unscoped_registration_identity_is_rejected(htapp_manifest, field):
    path = htapp_manifest.parent / "registration.json"
    registration = json.loads(path.read_text())
    registration[field] = "wrong_identity"
    path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match=f"ambiguous registration {field}"):
        prepare_sample(htapp_manifest)


def test_missing_protein_identity_and_legacy_manifest_require_migration(htapp_manifest):
    manifest = json.loads(htapp_manifest.read_text())
    manifest["schema_version"] = 1
    htapp_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version must be 2"):
        prepare_sample(htapp_manifest)
    manifest["schema_version"] = 2
    manifest["matching"].pop("protein")
    htapp_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="matching.protein.patient_id is unresolved"):
        prepare_sample(htapp_manifest)


def test_query_specimen_must_match_manifest(htapp_manifest):
    manifest = json.loads(htapp_manifest.read_text())
    manifest["matching"]["query"]["specimen_id"] = "wrong"
    with pytest.raises(ValueError, match="matching.query.specimen_id differs"):
        validate_matching(manifest)


def test_protein_operator_independent_of_graph_parameters(htapp_manifest):
    first = prepare_sample(htapp_manifest)
    manifest = json.loads(htapp_manifest.read_text())
    manifest["graph"] = {"radius_um": 1, "max_neighbors": 1}
    htapp_manifest.write_text(json.dumps(manifest))
    second = prepare_sample(htapp_manifest)
    np.testing.assert_array_equal(first.proteins.values, second.proteins.values)
    for field in ("rows", "columns", "weights"):
        np.testing.assert_array_equal(
            getattr(first.proteins.operator, field), getattr(second.proteins.operator, field)
        )
    assert len(first.graph.senders) > len(second.graph.senders)


def test_verified_same_cell_uses_only_explicit_pairs(htapp_manifest):
    import anndata

    manifest = json.loads(htapp_manifest.read_text())
    identity = {
        key: value for key, value in manifest["matching"]["query"].items() if key != "assay"
    }
    manifest["matching"]["protein"] = {**identity, "assay": "multiplex_IF"}
    htapp_manifest.write_text(json.dumps(manifest))
    path = htapp_manifest.parent / "proteins.h5ad"
    protein = anndata.read_h5ad(path)
    for field, value in identity.items():
        protein.obs[field] = [value] * protein.n_obs
    protein.obs["paired_he_cell_id"] = ["n2", "n1"]
    protein.write_h5ad(path)
    path = htapp_manifest.parent / "registration.json"
    registration = json.loads(path.read_text())
    registration.update(
        source_identity=identity,
        correspondence="same_cell",
        same_cell_verified=True,
    )
    path.write_text(json.dumps(registration))
    evidence = prepare_sample(htapp_manifest).proteins
    np.testing.assert_array_equal(evidence.operator.columns, [1, 0])
    np.testing.assert_array_equal(evidence.operator.rows, [0, 1])
    np.testing.assert_allclose(evidence.operator.weights, 1)
    np.testing.assert_allclose(evidence.values[:, 0], [0.5, 0.75])
    np.testing.assert_array_equal(evidence.observed[:, 1], [False, True])
    assert evidence.coverage["excluded_he_regions"] == 1


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
