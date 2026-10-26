"""Regional cohort CLI acceptance and patient/replicate isolation regressions."""

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_personalized_cohort import _tsv
from test_personalized_cohort import published_cohort as published_cohort

from heir.personalized import cohort
from heir.personalized.cli import main
from heir.personalized.cohort_cli import DEFAULT_OUTCOME_GENES, default_config, replicate_scores
from heir.personalized.evaluate import score
from heir.personalized.observation import ObservationOperator
from heir.personalized.schema import SpatialTargets
from heir.personalized.serialization import load_prediction
from heir.personalized.train import FittedModel


@pytest.fixture
def regional_workflow(published_cohort):
    import anndata
    import pandas
    from PIL import Image
    from scipy.sparse import csr_matrix

    root = published_cohort
    directory = root / "working/registered/313-932"
    image = np.empty((20, 20, 3), dtype=np.uint8)
    for (x, y), color in zip(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [[170, 75, 120], [210, 110, 160], [130, 60, 110], [180, 80, 150]],
    ):
        image[y * 10 : (y + 1) * 10, x * 10 : (x + 1) * 10] = color
    Image.fromarray(image).save(directory / "HE.jpg")
    xy = np.tile([[1, 1], [11, 1], [1, 11], [11, 11]], (2, 1))
    target = anndata.AnnData(
        X=np.full((8, 1), -999),
        obs=pandas.DataFrame(
            {"x": xy[:, 0], "y": xy[:, 1], "replicate": ["1"] * 4 + ["2"] * 4},
            index=[f"bead:{i}" for i in range(8)],
        ),
    )
    target.obsm["counts"] = csr_matrix(
        [[8, 1], [5, 2], [2, 5], [1, 8], [7, 2], [4, 3], [3, 4], [2, 7]]
    )
    target.uns["counts_var"] = ["EPCAM", "KRT8"]
    target.write_h5ad(directory / "slide_seq.h5ad")
    manifest_path = root / "metadata/selected_13_verified_file_manifest.tsv"
    manifest = cohort._table(manifest_path)
    for source in manifest:
        if source["biopsy_id"] == "313-932" and source["modality"] in {"HE", "slide_seq"}:
            path = root / source["working_path"]
            source.update(
                bytes=str(path.stat().st_size),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    _tsv(manifest_path, manifest)
    config = default_config()
    config["preparation"].update(region_size=10, max_states=2)
    config["model"].update(
        gene_ids=["EPCAM", "KRT8"],
        supervised_gene_ids=["EPCAM", "KRT8"],
        normalization_gene_ids=["EPCAM", "KRT8"],
        hidden_dim=8,
        residual_rank=2,
    )
    config_path = root / "workflow.json"
    config_path.write_text(json.dumps(config))
    return root, config_path, directory / "slide_seq.h5ad"


@pytest.fixture
def regional_demo(regional_workflow):
    root, config, target = regional_workflow
    output = root / "demo"
    main(
        [
            "cohort-demo",
            "--root",
            str(root),
            "--config",
            str(config),
            "--biopsy",
            "313-932",
            "--epochs",
            "2",
            "--output",
            str(output),
        ]
    )
    return root, output, target


def test_demo_exports_three_comparisons_and_predict_ignores_unavailable_targets(
    regional_demo, monkeypatch
):
    root, output, target = regional_demo
    run = json.loads((output / "run.json").read_text())
    assert set(run["comparisons"]) == {"he", "he_rna", "he_rna_protein"}
    assert run["prediction_unit"] == "tissue_region"
    assert run["coordinate_units"] == "registered_pixel"
    assert run["evaluation_role"] == "development_in_sample"
    assert not run["scientific_qualification"]
    for variant in run["comparisons"]:
        prediction, provenance = load_prediction(output / f"{variant}.npz")
        assert prediction.expression.shape == (4, 2)
        assert prediction.prediction_unit == "tissue_region"
        assert not provenance["target_payload_read"]
        if variant != "he":
            transfer, _ = load_prediction(output / f"{variant}.transfer.npz")
            np.testing.assert_array_equal(transfer.expression, transfer.transfer)

    original, _ = load_prediction(output / "he_rna_protein.npz")
    read = cohort._read_h5ad

    def prediction_input_only(path):
        assert "slide_seq" not in path.name
        return read(path)

    monkeypatch.setattr(cohort, "_read_h5ad", prediction_input_only)
    command = [
        "cohort-predict",
        "--root",
        str(root),
        "--biopsy",
        "313-932",
        "--bundle",
        str(output / "he_rna_protein.bundle"),
    ]
    target.unlink()
    for name in ("target_absent", "target_changed"):
        if name == "target_changed":
            target.write_bytes(b"deliberately invalid changed target payload")
        path = root / f"{name}.npz"
        main([*command, "--output", str(path)])
        predicted, provenance = load_prediction(path)
        np.testing.assert_array_equal(predicted.assignment, original.assignment)
        np.testing.assert_array_equal(predicted.expression, original.expression)
        assert not provenance["target_payload_read"]


def test_evaluation_gene_subset_preserves_fixed_panel_gene_scores(regional_demo):
    root, output, _ = regional_demo
    requested = root / "one_gene.json"
    requested.write_text(json.dumps(["EPCAM"]))
    command = [
        "cohort-evaluate",
        "--root",
        str(root),
        "--prediction",
        str(output / "he_rna.npz"),
    ]
    main([*command, "--output", str(root / "all_scores.json")])
    main([*command, "--genes", str(requested), "--output", str(root / "one_score.json")])
    full = json.loads((root / "all_scores.json").read_text())
    subset = json.loads((root / "one_score.json").read_text())
    assert full["normalization_gene_ids"] == subset["normalization_gene_ids"] == ["EPCAM", "KRT8"]
    assert subset["evaluation_gene_ids"] == ["EPCAM"]
    for all_genes, one_gene in zip(full["replicate_scores"], subset["replicate_scores"]):
        gene = next(row for row in all_genes["gene_scores"] if row["gene_id"] == "EPCAM")
        assert gene == one_gene["gene_scores"][0]
        assert gene["n_regions"] == 4
        assert gene["spearman"] is not None
    assert not full["held_out_from_fit"]


def test_replicate_scoring_cannot_use_between_replicate_mean_differences():
    ids = tuple(f"region:{i}" for i in range(6))
    values = np.array([[1, 9]] * 3 + [[9, 1]] * 3, dtype=np.float32)
    prediction = SimpleNamespace(
        cell_ids=ids,
        gene_ids=("A", "B"),
        expression=torch.from_numpy(values),
        patient_id="patient",
        specimen_id="biopsy",
        prediction_unit="tissue_region",
        coordinate_units="registered_pixel",
    )
    targets = SpatialTargets(
        ids, ("A", "B"), values, np.ones(values.shape, dtype=bool), "patient", "biopsy"
    )
    operator = ObservationOperator(ids, ids, np.arange(6), np.arange(6), np.ones(6))
    pooled = score(prediction, targets, operator, ["A"], normalization_gene_ids=["A", "B"])
    assert pooled["gene_scores"][0]["spearman"] == pytest.approx(1)
    result = replicate_scores(
        prediction, targets, operator, ("1",) * 3 + ("2",) * 3, ["A"], ["A", "B"]
    )
    assert result["mean_replicate_median_gene_spearman"] is None
    assert result["headline_status"] == "no_evaluable_target_genes"
    assert all(
        report["gene_scores"][0]["undefined_reason"] == "constant_target"
        for report in result["replicate_scores"]
    )


def test_cohort_training_opens_targets_only_for_training_patients(regional_workflow, monkeypatch):
    root, config, _ = regional_workflow
    inventory = cohort.discover_cohort(root)
    folds = root / "folds.csv"
    with folds.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outer_fold", "patient_id", "role"])
        writer.writeheader()
        for biopsy, entry in inventory.items():
            role = (
                "train" if biopsy == "313-932" else "test" if biopsy == "997-7789" else "validation"
            )
            writer.writerow({"outer_fold": 0, "patient_id": entry["patient_id"], "role": role})
            if role != "train":
                (root / entry["role_sources"]["targets"]["working_path"]).unlink()
    read = cohort._read_h5ad
    opened = []

    def training_patient_only(path):
        assert path.parent.name == "313-932"
        opened.append(path.name)
        return read(path)

    monkeypatch.setattr(cohort, "_read_h5ad", training_patient_only)
    bundle = root / "trained.bundle"
    main(
        [
            "cohort-train",
            "--root",
            str(root),
            "--config",
            str(config),
            "--folds",
            str(folds),
            "--outer-fold",
            "0",
            "--epochs",
            "1",
            "--output",
            str(bundle),
        ]
    )
    assert opened == ["scRNAseq.h5ad", "codex.h5ad", "slide_seq.h5ad"]
    metadata = json.loads((bundle / "bundle.json").read_text())
    assert metadata["training_patient_ids"] == ["HTA1_313"]
    assert len(metadata["preparation"]["training_sources"]) == 1
    assert metadata["training_config"]["abundance_weight"] == 0


def test_demo_he_arm_supervises_entire_normalization_panel_atomically(regional_workflow):
    import anndata
    from scipy.sparse import csr_matrix

    root, config_path, target_path = regional_workflow
    target = anndata.read_h5ad(target_path)
    existing = target.obsm["counts"].toarray()
    extra = np.arange(1, 8 * 7 + 1).reshape(8, 7)
    target.obsm["counts"] = csr_matrix(np.column_stack((existing, extra)))
    target.uns["counts_var"] = list(DEFAULT_OUTCOME_GENES)
    target.write_h5ad(target_path)
    manifest_path = root / "metadata/selected_13_verified_file_manifest.tsv"
    manifest = cohort._table(manifest_path)
    for source in manifest:
        if source["working_path"] == str(target_path.relative_to(root)):
            source.update(
                bytes=str(target_path.stat().st_size),
                sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
            )
    _tsv(manifest_path, manifest)
    config = json.loads(config_path.read_text())
    config["model"].update(
        gene_ids=list(DEFAULT_OUTCOME_GENES),
        normalization_gene_ids=list(DEFAULT_OUTCOME_GENES),
        supervised_gene_ids=["EPCAM"],
    )
    config_path.write_text(json.dumps(config))
    output = root / "partially_supervised_demo"
    main(
        [
            "cohort-demo",
            "--root",
            str(root),
            "--config",
            str(config_path),
            "--biopsy",
            "313-932",
            "--epochs",
            "1",
            "--output",
            str(output),
        ]
    )
    he = FittedModel.load(output / "he.bundle")
    assert he.model.config.supervised_gene_ids == DEFAULT_OUTCOME_GENES
    assert he.model.config.normalization_gene_ids == DEFAULT_OUTCOME_GENES
    assert he.model.config.gene_ids == DEFAULT_OUTCOME_GENES
    for variant in ("he_rna", "he_rna_protein"):
        personalized = FittedModel.load(output / f"{variant}.bundle")
        assert personalized.model.config.supervised_gene_ids == ("EPCAM",)
        assert personalized.model.config.normalization_gene_ids == DEFAULT_OUTCOME_GENES


def test_context_training_reuses_protein_baseline_with_same_preparation(regional_workflow):
    root, config_path, _ = regional_workflow
    inventory = cohort.discover_cohort(root)
    folds = root / "context_folds.csv"
    with folds.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outer_fold", "patient_id", "role"])
        writer.writeheader()
        for biopsy, entry in inventory.items():
            role = (
                "train" if biopsy == "313-932" else "test" if biopsy == "997-7789" else "validation"
            )
            writer.writerow({"outer_fold": 0, "patient_id": entry["patient_id"], "role": role})
    command = [
        "cohort-train",
        "--root",
        str(root),
        "--config",
        str(config_path),
        "--folds",
        str(folds),
        "--outer-fold",
        "0",
        "--epochs",
        "1",
    ]
    baseline_path, context_path = root / "protein_baseline.bundle", root / "context.bundle"
    main([*command, "--output", str(baseline_path)])
    config = json.loads(config_path.read_text())
    config["model"].update(
        variant="generic_fixed",
        lr_interactions=[{"ligand": ["EPCAM"], "receptor": ["KRT8"], "radius_um": 20}],
    )
    config_path.write_text(json.dumps(config))
    main([*command, "--baseline", str(baseline_path), "--output", str(context_path)])
    baseline, context = FittedModel.load(baseline_path), FittedModel.load(context_path)
    assert context.model.config.variant == "generic_fixed"
    assert context.model.fixed_mapper_initialized
    assert context.preparation == baseline.preparation
    assert "model" not in context.preparation["cohort_workflow"]
    assert context.training_patient_ids == baseline.training_patient_ids == ("HTA1_313",)
    np.testing.assert_array_equal(context.feature_mean, baseline.feature_mean)
    np.testing.assert_array_equal(context.feature_scale, baseline.feature_scale)
