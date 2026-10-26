"""Measured proteins and genuine modality separation for the joint model."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_personalized_cohort import _tsv
from test_personalized_cohort import published_cohort as published_cohort
from test_region_features import StubEncoder

from heir.features import load_encoder_manifest
from heir.personalized import cohort
from heir.personalized.protein_features import (
    ProteinFeatureScaler,
    aggregate_protein_features,
    load_joint_inputs,
    make_protein_correspondence,
    permute_protein_features,
    permute_protein_intensities,
    qualify_marker_panel,
    replay_protein_features,
    save_joint_inputs,
)
from heir.personalized.region_features import write_feature_cache
from heir.personalized.regions import build_region_grid
from heir.personalized.schema import JointInputs, RegionalProteinFeatures


def _grid(size=10):
    image = np.full((10, 20, 3), [170, 75, 120], dtype=np.uint8)
    return build_region_grid(image, bin_size_pixels=size, compute_features=False)


def _sample(features, patient="train"):
    return JointInputs(
        patient,
        patient + ":biopsy",
        features.region_ids,
        np.zeros((len(features.region_ids), 2)),
        protein_features=features,
    )


def _features():
    return aggregate_protein_features(
        _grid(),
        [[0, np.nan], [3, 5], [8, 4], [-2, 8]],
        [[1, 1], [2, 1], [11, 1], [1, 1]],
        ["1", "1", "1", "2"],
        ("A", "B"),
        cofactor=1,
    )


def test_raw_cell_summaries_preserve_zeros_signed_values_counts_and_equal_replicates():
    features = _features()
    first_rep_mean = np.arcsinh([0, 3]).mean()
    assert features.valid_counts[0, 0] == 3
    assert features.replicate_valid_counts[0, 0, 1] == 1
    assert features.observed.all()
    np.testing.assert_allclose(features.replicate_mean[0, 0, 0], first_rep_mean)
    np.testing.assert_allclose(features.replicate_q90[0, 0, 0], 0.9 * np.arcsinh(3))
    # Replicate 1 has two cells and replicate 2 has one; consensus weights both
    # replicate means equally, not according to their cell counts.
    np.testing.assert_allclose(features.mean[0, 0], (first_rep_mean + np.arcsinh(-2)) / 2)
    assert not np.isclose(features.mean[0, 0], np.arcsinh([0, 3, -2]).mean())
    np.testing.assert_allclose(
        features.replicate_disagreement[0, 0], np.std([first_rep_mean, np.arcsinh(-2)])
    )
    assert not features.replicate_observed[1, 1].any()


def test_coarse_proteins_aggregate_raw_cells_not_fine_mean_or_fine_quantiles():
    values = np.array([[0], [2], [4], [100]], dtype=float)
    xy = [[1, 1], [2, 1], [3, 1], [11, 1]]
    fine = aggregate_protein_features(_grid(), values, xy, ["1"] * 4, ("A",), cofactor=1)
    coarse = aggregate_protein_features(_grid(20), values, xy, ["1"] * 4, ("A",), cofactor=1)
    transformed = np.arcsinh(values[:, 0])
    np.testing.assert_allclose(coarse.mean[0, 0], transformed.mean())
    np.testing.assert_allclose(coarse.q90[0, 0], np.quantile(transformed, 0.9))
    np.testing.assert_allclose(
        coarse.mean[0, 0], np.average(fine.mean[:, 0], weights=fine.valid_counts[:, 0])
    )
    assert not np.isclose(coarse.mean[0, 0], fine.mean.mean())
    assert not np.isclose(coarse.q90[0, 0], fine.q90.mean())


def test_scaler_fit_training_only_named_axes_and_ignores_arbitrary_missing_values():
    features = RegionalProteinFeatures(
        ("r1", "r2", "r3"),
        ("B", "A"),
        np.array([[1.0, 2], [5, 4], [np.nan, np.inf]]),
        np.array([[2.0, 3], [6, 5], [np.inf, -999.0]]),
        np.array([[True, True], [True, True], [False, False]]),
        np.array([[2.0, 3], [4, 5], [np.nan, -1.0]]),
    )
    training = _sample(features)
    heldout = _sample(replace(features, mean=np.full((3, 2), 1e9)), "heldout")
    scaler = ProteinFeatureScaler.fit([training])
    assert scaler.training_patient_ids == ("train",)
    assert scaler.marker_ids == ("A", "B")
    np.testing.assert_allclose(scaler.mean_center, [3, 3])
    transformed = scaler.transform(features)
    np.testing.assert_allclose(transformed.mean[:2], [[-1, -1], [1, 1]])
    assert np.array_equal(transformed.mean[2], [0, 0])
    assert np.array_equal(transformed.valid_counts[2], [0, 0])
    assert scaler.transform(heldout.protein_features).mean[0].min() > 1e8
    restored = ProteinFeatureScaler.from_dict(scaler.to_dict())
    np.testing.assert_array_equal(restored.transform(features).q90, transformed.q90)
    permutation = [1, 0]
    reordered = replace(
        features,
        marker_ids=("A", "B"),
        **{
            name: getattr(features, name)[:, permutation]
            for name in ("mean", "q90", "observed", "valid_counts")
        },
    )
    output = scaler.transform(reordered)
    assert output.marker_ids == ("A", "B")
    np.testing.assert_array_equal(output.mean, transformed.mean[:, permutation])


def test_permutation_updates_every_protein_path_within_replicate_availability():
    grid = _grid(5)
    xy = grid.coordinates
    features = aggregate_protein_features(
        grid,
        np.arange(1, len(xy) * 2 + 1).reshape(len(xy), 2),
        xy,
        ["1"] * len(xy),
        ("A", "B"),
        cofactor=1,
    )
    changed, metadata = permute_protein_features(features, seed=12)
    order = metadata["permutations"]["1"]
    assert order != list(range(len(xy)))
    np.testing.assert_array_equal(changed.replicate_mean[0], features.replicate_mean[0, order])
    np.testing.assert_array_equal(changed.mean, features.mean[order])
    np.testing.assert_array_equal(changed.q90, features.q90[order])
    np.testing.assert_array_equal(changed.valid_counts, features.valid_counts[order])
    np.testing.assert_array_equal(changed.observed, features.observed)
    assert metadata["kind"] == "fixed_model_prediction_intervention"


@pytest.fixture
def frozen_cohort(published_cohort, tmp_path):
    root = published_cohort
    source = cohort.discover_cohort(root)["313-932"]["role_sources"]["he"]
    path = root / source["working_path"]
    grid = build_region_grid(path, bin_size_pixels=10)
    manifest = load_encoder_manifest(
        Path(__file__).parents[1] / "manifests/encoders/hoptimus1.json"
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    write_feature_cache(
        path,
        grid,
        cache / "313-932.npz",
        StubEncoder(manifest),
        manifest,
        expected_image_sha256=source["sha256"],
    )
    return root, {
        "region_size": 10,
        "max_states": 2,
        "feature_cache_dir": cache,
        "feature_encoder_manifest_sha256": manifest.sha256,
    }


def test_preparation_and_bundle_load_only_requested_modalities(
    frozen_cohort, tmp_path, monkeypatch
):
    root, options = frozen_cohort
    sources = cohort.discover_cohort(root)["313-932"]["role_sources"]
    # Ordinary prediction preparation cannot open spatial RNA.
    (root / sources["targets"]["working_path"]).unlink()
    all_inputs, provenance = cohort.prepare_joint_inputs(root, "313-932", **options)
    assert all_inputs.image_features.shape == (4, 1536)
    assert all_inputs.protein_features.mean.shape == (4, 5)
    assert all_inputs.protein_features.valid_counts[0, 0] == 1  # one other cell is NaN
    destination = tmp_path / "prepared"
    save_joint_inputs(all_inputs, destination, provenance)
    restored, restored_provenance = load_joint_inputs(destination, modalities=("H", "R", "P"))
    np.testing.assert_array_equal(restored.reference.expression, all_inputs.reference.expression)
    assert restored_provenance == provenance
    # The HP arm needs neither the original RNA source nor its prepared payload.
    (root / sources["rna"]["working_path"]).unlink()
    (destination / "reference.npz").unlink()
    hp, metadata = cohort.prepare_joint_inputs(root, "313-932", modalities=("H", "P"), **options)
    assert hp.reference is None and metadata["consumed_sources"] == ["he", "protein"]
    restored, _ = load_joint_inputs(destination, modalities=("H", "P"))
    assert restored.reference is None
    # RP can load with the image feature payload absent. It must not invoke an
    # encoder/cache read during fresh preparation either (tested with R absent).
    (destination / "image.npz").unlink()
    restored, _ = load_joint_inputs(destination, modalities=("P",))
    assert restored.image_features is None and restored.reference is None
    import heir.personalized.region_features as cache_module

    monkeypatch.setattr(
        cache_module, "load_feature_cache", lambda *a, **k: pytest.fail("image cache opened")
    )
    measured, metadata = cohort.prepare_joint_inputs(root, "313-932", modalities=("P",), **options)
    assert measured.image_features is None
    assert metadata["grid"]["settings"]["encoder_name"] == "none_measurement_only"
    assert metadata["image_features"] is None and metadata["rna"] is None
    with pytest.raises(FileExistsError):
        save_joint_inputs(all_inputs, destination, provenance)


def test_rp_preparation_and_serialization_without_image_cache(frozen_cohort, tmp_path, monkeypatch):
    root, options = frozen_cohort
    (options["feature_cache_dir"] / "313-932.npz").unlink()
    import heir.personalized.region_features as cache_module

    monkeypatch.setattr(
        cache_module, "load_feature_cache", lambda *a, **k: pytest.fail("image cache opened")
    )
    inputs, provenance = cohort.prepare_joint_inputs(
        root, "313-932", modalities=("R", "P"), **options
    )
    assert inputs.image_features is None and inputs.reference is not None
    assert "other_gene" in inputs.reference.gene_ids  # gene_ids=None retains full measured RNA axis
    directory = tmp_path / "rp"
    save_joint_inputs(inputs, directory, provenance)
    assert not (directory / "image.npz").exists()
    restored, _ = load_joint_inputs(directory, modalities=("R", "P"))
    np.testing.assert_array_equal(restored.protein_features.q90, inputs.protein_features.q90)
    with pytest.raises(ValueError, match="no requested image"):
        load_joint_inputs(directory, modalities=("H",))


def test_marker_panel_separates_measurement_alias_from_biological_gene_targets(tmp_path):
    path = tmp_path / "panel.tsv"
    path.write_text(
        "Target\tMatching gene\tClone\ncytokeratin\tKRT19\tC-11\n"
        "HLA-ABC\tHLA-B\tEMR8-5\nHLA-DR\tHLA-DRA\tTu36\n"
    )
    panel, metadata = qualify_marker_panel(path, ("KRT19", "HLA-B", "HLA-DRA", "DAPI"), "expanded")
    assert panel[0]["measurement_id"] == "KRT19"
    assert set(panel[0]["gene_ids"]) >= {"KRT8", "KRT18"}
    assert "KRT19" not in panel[0]["gene_ids"]
    assert panel[0]["possible_direct_gene_ids"] == ["KRT19"]
    assert panel[1]["gene_ids"] == ["HLA-A", "HLA-B", "HLA-C"]
    assert panel[2]["gene_ids"] == []  # valid antibody feature, unresolved complete RNA mapping
    assert metadata["unmatched_retained_channels"] == ["DAPI"]
    assert all(item["biological_feature_qualified"] for item in panel)
    with pytest.raises(ValueError, match="supporting evidence"):
        qualify_marker_panel(
            path, ("KRT19",), [{"marker_id": "cytokeratin", "measurement_id": "KRT19"}]
        )


def test_raw_marker_and_cell_permutations_preserve_named_summaries():
    values = np.array([[0.0, 3], [2, 5], [np.nan, 9], [7, -4]])
    xy = np.array([[1, 1], [2, 1], [11, 1], [12, 1]])
    replicates = np.array(["b", "a", "b", "a"])
    original = aggregate_protein_features(_grid(), values, xy, replicates, ("A", "B"))
    rows, columns = [3, 1, 0, 2], [1, 0]
    changed = aggregate_protein_features(
        _grid(), values[rows][:, columns], xy[rows], replicates[rows], ("B", "A")
    )
    assert changed.replicate_labels == original.replicate_labels == ("a", "b")
    for name in ("mean", "q90", "observed", "valid_counts", "replicate_disagreement"):
        np.testing.assert_array_equal(getattr(changed, name), getattr(original, name)[:, columns])
    for name in ("replicate_mean", "replicate_q90", "replicate_observed", "replicate_valid_counts"):
        np.testing.assert_array_equal(
            getattr(changed, name), getattr(original, name)[:, :, columns]
        )


def test_h_only_ignores_missing_molecular_files_and_protein_bundle_hash_is_enforced(
    frozen_cohort, tmp_path
):
    root, options = frozen_cohort
    inputs, provenance = cohort.prepare_joint_inputs(root, "313-932", **options)
    saved = tmp_path / "hash_bound"
    save_joint_inputs(inputs, saved, provenance)
    payload = saved / "protein.npz"
    with payload.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="protein payload missing or checksum differs"):
        load_joint_inputs(saved, modalities=("P",))
    h, _ = load_joint_inputs(saved, modalities=("H",))
    assert h.protein_features is None and h.reference is None
    sources = cohort.discover_cohort(root)["313-932"]["role_sources"]
    for role in ("rna", "protein", "targets"):
        (root / sources[role]["working_path"]).unlink()
    h, provenance = cohort.prepare_joint_inputs(root, "313-932", modalities=("H",), **options)
    assert h.protein_features is None and h.reference is None
    assert provenance["consumed_sources"] == ["he"]


def test_all_missing_protein_regions_stay_on_geometry_axis():
    features = aggregate_protein_features(
        _grid(), [[np.nan, np.inf], [0.0, -1]], [[1, 1], [99, 99]], ["1", "1"], ("A", "B")
    )
    assert features.region_ids == _grid().region_ids
    assert not features.observed.any()
    assert features.metadata["excluded_source_cells"] == 1
    assert not features.valid_counts.any()
    scaler = ProteinFeatureScaler.fit([_sample(features)])
    assert scaler.observed_training_regions == (0, 0)
    assert scaler.mean_center == (0, 0) and scaler.mean_scale == (1, 1)
    assert not scaler.transform(features).observed.any()


def test_declared_marker_axis_keeps_absent_channels_masked(published_cohort):
    import anndata

    root = published_cohort
    source = cohort.discover_cohort(root)["313-932"]["role_sources"]["protein"]
    path = root / source["working_path"]
    payload = anndata.read_h5ad(path)
    payload.obsm["counts"] = payload.obsm["counts"][:, 1:]
    payload.uns["counts_var"] = payload.uns["counts_var"][1:]
    payload.write_h5ad(path)
    manifest_path = root / "metadata/selected_13_verified_file_manifest.tsv"
    rows = cohort._table(manifest_path)
    for row in rows:
        if row["working_path"] == source["working_path"]:
            row.update(
                bytes=str(path.stat().st_size), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
    _tsv(manifest_path, rows)
    markers = tuple(marker for marker, _ in cohort.DEFAULT_PROTEIN_MARKERS)
    inputs, provenance = cohort.prepare_joint_inputs(
        root, "313-932", modalities=("P",), region_size=10, protein_marker_ids=markers
    )
    features = inputs.protein_features
    assert features.marker_ids == markers
    assert not features.observed[:, 0].any()
    assert not features.valid_counts[:, 0].any()
    assert features.observed[:, 1:].any()
    selected = provenance["protein"]["panel"]["qualified_features"]
    assert selected[0]["measurement_available"] is False
    assert selected[0]["biological_feature_qualified"] is True


def test_correspondence_replays_raw_named_observations_and_preserves_exclusions():
    grid = _grid()
    values = np.array([[0.0, np.nan], [3, 5], [8, 4], [-2, 8], [20, 40]])
    xy = np.array([[1, 1], [2, 1], [11, 1], [1, 1], [100, 100]])
    replicates = ("1", "1", "1", "2", "2")
    ids = tuple(f"original:{i}" for i in range(len(values)))
    mapping = make_protein_correspondence(
        grid,
        ids,
        xy,
        replicates,
        {
            "patient_id": "train",
            "specimen_id": "train:biopsy",
            "working_path": "codex.h5ad",
            "sha256": "a" * 64,
        },
    )
    np.testing.assert_array_equal(mapping.region_indices, [0, 0, 1, 0, -1])
    assert mapping.observation_ids == ids and mapping.replicate_ids == replicates
    assert mapping.coordinate_units == "registered_pixel"
    assert mapping.qualification == "published_coarse_registration_unvalidated"
    original = aggregate_protein_features(grid, values, xy, replicates, ("A", "B"), cofactor=1)
    mapped = aggregate_protein_features(
        grid, values, xy, replicates, ("A", "B"), cofactor=1, correspondence=mapping
    )
    order = [4, 1, 3, 0, 2]
    replayed = replay_protein_features(
        grid, values[order], tuple(ids[i] for i in order), ("A", "B"), mapping, cofactor=1
    )
    for name in (
        "mean",
        "q90",
        "observed",
        "valid_counts",
        "replicate_mean",
        "replicate_q90",
        "replicate_observed",
        "replicate_valid_counts",
        "replicate_disagreement",
    ):
        np.testing.assert_array_equal(getattr(original, name), getattr(mapped, name))
        np.testing.assert_array_equal(getattr(original, name), getattr(replayed, name))
    assert replayed.metadata["excluded_source_cells"] == 1
    with pytest.raises(ValueError, match="complete original"):
        replay_protein_features(grid, values[:-1], ids[:-1], ("A", "B"), mapping)
    broken = replace(mapping, region_indices=np.array([1, 0, 1, 0, -1]))
    with pytest.raises(ValueError, match="declared H&E geometry"):
        replay_protein_features(grid, values, ids, ("A", "B"), broken)
    with pytest.raises(ValueError, match="patient, specimen"):
        _sample(mapped, patient="another_patient")


def test_correspondence_bundle_roundtrip_is_optional_and_keeps_numeric_payload_identical(
    frozen_cohort, tmp_path
):
    root, options = frozen_cohort
    inputs, provenance = cohort.prepare_joint_inputs(root, "313-932", **options)
    mapping = inputs.protein_features.correspondence
    assert len(mapping.observation_ids) == 4
    assert mapping.source_identity["sha256"] == provenance["sources"]["protein"]["sha256"]
    modern, legacy = tmp_path / "mapped", tmp_path / "unmapped"
    record = save_joint_inputs(inputs, modern, provenance)
    unmapped = replace(
        inputs, protein_features=replace(inputs.protein_features, correspondence=None)
    )
    old_record = save_joint_inputs(unmapped, legacy, provenance)
    assert record["payloads"]["protein"]["sha256"] == old_record["payloads"]["protein"]["sha256"]
    assert (modern / "protein.npz").read_bytes() == (legacy / "protein.npz").read_bytes()
    restored, _ = load_joint_inputs(modern, modalities=("P",))
    restored_mapping = restored.protein_features.correspondence
    assert restored_mapping.observation_ids == mapping.observation_ids
    assert restored_mapping.source_identity == mapping.source_identity
    np.testing.assert_array_equal(restored_mapping.region_indices, mapping.region_indices)
    np.testing.assert_array_equal(restored_mapping.coordinates, mapping.coordinates)
    old, _ = load_joint_inputs(legacy, modalities=("P",))
    assert old.protein_features.correspondence is None
    # A^P is not an H/R input: an absent correspondence payload cannot prevent
    # loading those arms, while a requested P input must check its file hash.
    (modern / "protein_correspondence.npz").unlink()
    image, _ = load_joint_inputs(modern, modalities=("H",))
    assert image.protein_features is None
    with pytest.raises(ValueError, match="protein_correspondence payload missing"):
        load_joint_inputs(modern, modalities=("P",))


def test_saved_correspondence_rejects_changed_mapping_even_with_updated_file_hash(
    frozen_cohort, tmp_path
):
    root, options = frozen_cohort
    inputs, provenance = cohort.prepare_joint_inputs(root, "313-932", **options)
    directory = tmp_path / "corrupt_map"
    record = save_joint_inputs(inputs, directory, provenance)
    path = directory / "protein_correspondence.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["region_indices"][0] = 1
    np.savez_compressed(path, **values)
    record["payloads"]["protein_correspondence"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (directory / "metadata.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="declared H&E geometry"):
        load_joint_inputs(directory, modalities=("P",))


def test_correspondence_binds_he_image_frame_even_when_geometry_matches(frozen_cohort, tmp_path):
    root, options = frozen_cohort
    inputs, provenance = cohort.prepare_joint_inputs(root, "313-932", **options)
    directory = tmp_path / "changed_he_frame"
    record = save_joint_inputs(inputs, directory, provenance)
    record["provenance"]["sources"]["he"]["sha256"] = "b" * 64
    (directory / "metadata.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="declared H&E image frame"):
        load_joint_inputs(directory, modalities=("P",))


def test_intensity_only_permutation_preserves_original_local_coverage():
    grid = _grid(5)
    xy, values, replicates = [], [], []
    for r in range(2):
        for i, coordinate in enumerate(grid.coordinates):
            for cell in range(i + 1):
                xy.append(coordinate)
                values.append([10 * r + i + cell, np.nan if i % 2 else 20 * r + i + cell])
                replicates.append(str(r))
    original = aggregate_protein_features(grid, values, xy, replicates, ("A", "B"))
    changed, metadata = permute_protein_intensities(original, seed=7)
    legacy, legacy_metadata = permute_protein_features(original, seed=7)
    for name in ("observed", "valid_counts", "replicate_observed", "replicate_valid_counts"):
        np.testing.assert_array_equal(getattr(changed, name), getattr(original, name))
    assert not np.array_equal(changed.mean, original.mean)
    assert not np.array_equal(changed.q90, original.q90)
    assert not np.array_equal(legacy.valid_counts, original.valid_counts)
    assert metadata["counts_permuted_with_intensities"] is False
    assert legacy_metadata["counts_permuted_with_intensities"] is True
    for r, label in enumerate(original.replicate_labels):
        order = metadata["permutations"][label]
        np.testing.assert_array_equal(changed.replicate_mean[r], original.replicate_mean[r, order])
