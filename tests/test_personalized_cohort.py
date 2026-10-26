"""Actual-format cohort fixtures without biological identity inventions."""

import csv
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from heir.personalized import cohort


def _tsv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def published_cohort(tmp_path):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    sparse = pytest.importorskip("scipy.sparse")
    image = pytest.importorskip("PIL.Image")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    rows, manifest = [], []
    pilot = "313-932"
    genes = (*cohort.DEFAULT_EMBEDDING_GENES, "other_gene")
    for biopsy in cohort.COHORT_BIOPSIES:
        participant, specimen = f"HTA1_{biopsy.split('-')[0]}", f"HTA1_{biopsy.replace('-', '_')}"
        rows.append(
            {
                "biopsy_id": biopsy,
                "participant_group_id": participant,
                "participant_group_basis": "published_prefix_grouping",
                "htan_biopsy_id": specimen,
            }
        )
        directory = tmp_path / "working/registered" / biopsy
        directory.mkdir(parents=True)
        for modality in ("HE", "scRNAseq", "codex", "slide_seq"):
            path = directory / (modality + (".jpg" if modality == "HE" else ".h5ad"))
            replicate = "1;2" if modality == "slide_seq" else "1"
            if biopsy != pilot:
                path.write_bytes(b"unused fixture source")
            elif modality == "HE":
                image.fromarray(np.full((20, 20, 3), [170, 75, 120], dtype=np.uint8)).save(path)
            else:
                if modality == "scRNAseq":
                    values = np.arange(1, 4 * len(genes) + 1).reshape(4, len(genes))
                    axis = genes
                    obs = pd.DataFrame({"cell_type": ["A", "A", "B", "B"], "replicate": ["1"] * 4})
                elif modality == "codex":
                    values = np.tile(np.arange(1, 6), (4, 1)).astype(float)
                    values[1, 0] = np.nan
                    axis = tuple(g for _, g in cohort.DEFAULT_PROTEIN_MARKERS)
                    obs = pd.DataFrame(
                        {
                            "x": [1, 3, 11, 12],
                            "y": [1, 4, 2, 13],
                            "replicate": ["1"] * 4,
                        }
                    )
                else:
                    values = np.arange(1, 13).reshape(6, 2)
                    axis = ("EPCAM", "KRT8")
                    obs = pd.DataFrame(
                        {
                            "x": [1, 2, 11, 1, 12, 100],
                            "y": [1, 2, 1, 1, 12, 100],
                            "replicate": ["1", "1", "1", "2", "2", "2"],
                        }
                    )
                obs.index = [f"original:{modality}:{i}" for i in range(len(obs))]
                obj = ad.AnnData(X=np.full((len(obs), 1), -999), obs=obs)
                obj.obsm["counts"] = sparse.csr_matrix(values)
                obj.uns["counts_var"] = list(axis)
                obj.write_h5ad(path)
            manifest.append(
                {
                    "biopsy_id": biopsy,
                    "participant_id": participant,
                    "htan_biopsy_id": specimen,
                    "selected_cohort": "True",
                    "included_in_working_subset": "True",
                    "modality": modality,
                    "replicate_id": replicate,
                    "working_path": str(path.relative_to(tmp_path)),
                    "bytes": str(path.stat().st_size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "measurement_key": "" if modality == "HE" else "obsm/counts",
                    "feature_key": "" if modality == "HE" else "uns/counts_var",
                }
            )
    _tsv(metadata / "selected_13_biopsies.tsv", rows)
    _tsv(metadata / "selected_13_verified_file_manifest.tsv", manifest)
    _tsv(
        metadata / "Table_S5_CODEX_panel_original.tsv",
        [
            {"Target": marker, "Matching gene": gene}
            for marker, gene in cohort.DEFAULT_PROTEIN_MARKERS
        ],
    )
    return tmp_path


def _prepare(root, **kwargs):
    return cohort.prepare_cohort_inputs(root, "313-932", region_size=10, max_states=2, **kwargs)


def test_discovery_exact_cohort_and_unavailable_targets(published_cohort):
    root = published_cohort
    inventory = cohort.discover_cohort(root)
    assert tuple(inventory) == cohort.COHORT_BIOPSIES
    record = inventory["313-932"]
    assert record["assay_replicates"]["targets"] == ["1", "2"]
    target = root / record["role_sources"]["targets"]["working_path"]
    target.unlink()
    assert (
        cohort.discover_cohort(root)["313-932"]["file_checks"][str(target.relative_to(root))]
        == "missing"
    )
    path = root / "metadata/selected_13_biopsies.tsv"
    _tsv(path, cohort._table(path)[:-1])
    with pytest.raises(ValueError, match="exact 13"):
        cohort.discover_cohort(root)


def test_inputs_do_not_read_or_require_targets_and_protein_geometry_is_uniform(
    published_cohort, monkeypatch
):
    root = published_cohort
    source = cohort.discover_cohort(root)["313-932"]["role_sources"]["targets"]
    (root / source["working_path"]).unlink()
    read = cohort._read_h5ad
    opened = []

    def allowed(path):
        opened.append(path.name)
        assert "slide_seq" not in path.name
        return read(path)

    monkeypatch.setattr(cohort, "_read_h5ad", allowed)
    inputs, provenance = _prepare(root, protein_cofactor=1.0, rna_cofactor=1.0)
    assert opened == ["scRNAseq.h5ad", "codex.h5ad"]
    assert inputs.prediction_unit == "tissue_region"
    assert inputs.coordinate_units == "registered_pixel"
    assert inputs.image_features.shape == (4, 12)
    assert inputs.patient_id == "HTA1_313"
    assert inputs.specimen_id == "HTA1_313_932"
    assert provenance["section_identity"] == "unknown"
    assert provenance["matching_scope"] == "published_biopsy"
    assert "targets" not in provenance["sources"]
    assert provenance["rna"]["annotation_fit_scope"] == "published_cohort_not_training_fold_fitted"
    assert not inputs.proteins.observed[0, 0]
    assert inputs.proteins.values[0, 0] == 0
    np.testing.assert_allclose(inputs.proteins.values[0, 1:], np.arange(2, 6) / np.arange(3, 7))
    np.testing.assert_array_equal(inputs.proteins.operator.columns, [0, 1, 3])
    assert len(inputs.graph.senders) == 12
    json.dumps(provenance)


def test_consumed_source_checksum_rejects_same_size_change(published_cohort):
    root = published_cohort
    source = cohort.discover_cohort(root)["313-932"]["role_sources"]["rna"]
    path = root / source["working_path"]
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum differs"):
        _prepare(root)


def test_target_loader_retains_replicates_sums_counts_and_requires_no_input_payload(
    published_cohort,
):
    root = published_cohort
    inputs, provenance = _prepare(root)
    for source in provenance["sources"].values():
        (root / source["working_path"]).unlink()
    targets, operator, evidence = cohort.load_cohort_targets(
        root, "313-932", inputs, provenance, gene_ids=["KRT8", "EPCAM"]
    )
    np.testing.assert_array_equal(targets.expression, [[6, 4], [6, 5], [8, 7], [10, 9]])
    np.testing.assert_array_equal(operator.columns, [0, 1, 0, 3])
    assert evidence["observation_replicates"] == ["1", "1", "2", "2"]
    assert evidence["region_bead_counts"] == [2, 1, 1, 1]
    assert evidence["excluded_source_observations"] == 1
    assert targets.patient_id == inputs.patient_id
    assert targets.specimen_id == inputs.specimen_id
    assert targets.observation_ids[0] != targets.observation_ids[2]
    assert targets.observed.all()
    # Evaluation accepts the prediction interface without a reference or image features.
    from types import SimpleNamespace

    prediction = SimpleNamespace(
        **{
            name: getattr(inputs, name)
            for name in (
                "cell_ids",
                "coordinates",
                "prediction_unit",
                "coordinate_units",
                "patient_id",
                "specimen_id",
            )
        }
    )
    reloaded = cohort.load_cohort_targets(
        root, "313-932", prediction, provenance, gene_ids=["EPCAM"]
    )[0]
    np.testing.assert_array_equal(reloaded.expression[:, 0], targets.expression[:, 1])


def test_reference_output_subset_keeps_full_library_normalization(published_cohort):
    full, _ = _prepare(published_cohort)
    restricted, _ = _prepare(published_cohort, gene_ids=["EPCAM"])
    assert "other_gene" not in restricted.reference.gene_ids
    np.testing.assert_array_equal(
        full.reference.expression[:, full.reference.gene_ids.index("EPCAM")],
        restricted.reference.expression[:, restricted.reference.gene_ids.index("EPCAM")],
    )


def test_saved_grid_binding_rejects_other_identity_and_changed_geometry(published_cohort):
    inputs, provenance = _prepare(published_cohort)
    with pytest.raises(ValueError, match="provenance differ"):
        cohort.load_cohort_targets(
            published_cohort,
            "313-932",
            replace(inputs, patient_id="different"),
            provenance,
            gene_ids=["EPCAM"],
        )
    changed = json.loads(json.dumps(provenance))
    changed["grid"]["settings"]["bin_size_pixels"] = 20
    with pytest.raises(ValueError, match="provenance differ"):
        cohort.load_cohort_targets(published_cohort, "313-932", inputs, changed, gene_ids=["EPCAM"])
