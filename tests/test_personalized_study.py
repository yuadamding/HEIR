"""Patient-held-out selection, generic-reference provenance, and paired reporting."""

import csv
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from heir.personalized import study
from heir.personalized.cli import main
from heir.personalized.model import ModelConfig, PersonalizedModel
from heir.personalized.observation import ObservationOperator
from heir.personalized.schema import (
    PredictionInputs,
    ProteinEvidence,
    ProteinObservationOperator,
    RNAReference,
    SpatialTargets,
)
from heir.personalized.serialization import load_prediction
from heir.personalized.train import FittedModel, make_patient_folds


def _reference(expression, proportions=None):
    values = np.asarray(expression, dtype=np.float32)
    n = len(values)
    genes = ("EPCAM", "CD8A", "MKI67")[: values.shape[1]]
    return RNAReference(
        state_ids=tuple(f"state:{i}" for i in range(n)),
        gene_ids=genes,
        expression=values,
        embedding=np.log1p(values),
        proportions=np.asarray(proportions if proportions is not None else [1 / n] * n),
        state_types=("type",) * n,
        variance=np.ones(values.shape),
        cell_counts=np.arange(1, n + 1),
        embedding_gene_ids=genes,
    )


def _inputs(patient, reference):
    ids = tuple(f"he_region:{i}:0" for i in range(6))
    xy = np.column_stack((np.arange(6) * 10.0, np.zeros(6)))
    return PredictionInputs(
        patient_id=patient,
        specimen_id=f"specimen-{patient}",
        cell_ids=ids,
        coordinates_um=xy,
        image_features=np.column_stack((np.arange(6) / 5, np.arange(6) % 2)),
        reference=reference,
        proteins=ProteinEvidence(
            marker_ids=("CD8",),
            values=np.linspace(0, 1, 6)[:, None],
            observed=np.ones((6, 1), dtype=bool),
            state_values=np.linspace(0, 1, len(reference.state_ids))[:, None],
            correspondence="region",
            operator=ProteinObservationOperator(ids, ids, np.arange(6), np.arange(6), np.ones(6)),
        ),
        prediction_unit="tissue_region",
        coordinate_units="registered_pixel",
    )


def _report(patient, score, error=1.0, specimen=None):
    return {
        "patient_id": patient,
        "specimen_id": specimen or f"specimen-{patient}",
        "mean_replicate_median_gene_spearman": score,
        "normalized_log_mse": error,
    }


def test_generic_pool_equal_patient_weights_total_variance_and_no_heldout_inputs():
    a = _inputs("A", _reference([[1, 9], [3, 7]], [0.25, 0.75]))
    b = _inputs("B", _reference([[9, 1]], [1.0]))
    arguments = dict(gene_ids=("EPCAM", "CD8A"), embedding_gene_ids=("EPCAM", "CD8A"), max_states=1)
    pool, metadata = study.pool_reference([b, a], ["A", "B"], **arguments)
    np.testing.assert_allclose(pool.expression, [[5.75, 4.25]])
    # Each source state has variance one, plus the weighted variance of its mean.
    expected_variance = 1 + np.average(
        (np.array([1, 3, 9]) - 5.75) ** 2, weights=[0.125, 0.375, 0.5]
    )
    np.testing.assert_allclose(pool.variance, expected_variance)
    np.testing.assert_allclose(pool.proportions, [1])
    assert metadata["training_patient_ids"] == ["A", "B"]
    assert metadata["target_expression_used"] is False
    assert (
        metadata["reference_sha256"]
        == study.pool_reference([a, b], ["B", "A"], **arguments)[1]["reference_sha256"]
    )
    heldout = _inputs("test", _reference([[1000, 1]]))
    with pytest.raises(ValueError, match="exactly the training"):
        study.pool_reference([a, b, heldout], ["A", "B"], **arguments)


def test_patient_macro_and_paired_reports_never_pool_regions_or_drop_undefined():
    first = [_report("A", 1, 4, "A-1"), _report("A", 1, 4, "A-2"), _report("B", -1, 16)]
    summary = study.patient_summary(first)
    assert summary["mean_patient_spearman"] == 0
    assert summary["mean_patient_normalized_log_mse"] == 10
    second = [_report("A", 0.5, 2), _report("B", -0.5, 20)]
    contrast = study.paired_patient_contrasts(
        {"full": first, "control": second}, ["full-minus-control"]
    )["full-minus-control"]
    assert contrast["mean_paired_spearman_difference"] == 0
    assert contrast["mean_paired_normalized_log_mse_difference"] == -1
    first[-1]["mean_replicate_median_gene_spearman"] = None
    contrast = study.paired_patient_contrasts(
        {"full": first, "control": second}, ["full-minus-control"]
    )["full-minus-control"]
    assert contrast["mean_paired_spearman_difference"] is None
    assert contrast["undefined_patient_ids"] == ["B"]
    assert contrast["defined_pairs_mean_diagnostic"] == 0.5


def test_rank_selection_validation_only_equal_patient_and_declared_tiebreaks():
    two = [_report("A", 0.25, 2), _report("B", 0.75, 2)]
    sixteen = [_report("A", 0.75, 3), _report("B", 0.25, 3)]
    rank, record = study.select_rank({2: two, 16: sixteen}, ["A", "B"])
    assert rank == 2  # Equal score, lower MSE.
    sixteen[0]["normalized_log_mse"] = sixteen[1]["normalized_log_mse"] = 2
    assert study.select_rank({16: sixteen, 2: two}, ["A", "B"])[0] == 2
    sixteen[0]["mean_replicate_median_gene_spearman"] = None
    assert (
        study.select_rank({2: two, 16: sixteen}, ["A", "B"])[1]["candidates"][1]["eligible"]
        is False
    )
    with pytest.raises(ValueError, match="exactly the validation"):
        study.select_rank({2: two + [_report("test", 1)]}, ["A", "B"])
    with pytest.raises(ValueError, match="no rank"):
        study.select_rank({16: sixteen}, ["A", "B"])
    assert record["candidates"][0]["n_patients"] == 2


def _fold_file(path, patients):
    rows = make_patient_folds(patients, validation_size=1)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outer_fold", "patient_id", "role"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_fold_roles_disjoint_complete_and_test_once(tmp_path):
    path = tmp_path / "folds.csv"
    patients = [f"P{i}" for i in range(5)]
    rows = _fold_file(path, patients)
    result = study.validate_folds(path, patients)
    assert len(result) == 5
    for patient in patients:
        assert sum(roles[patient] == "test" for roles in result.values()) == 1
    with path.open("a") as handle:
        handle.write(f"0,{rows[0]['patient_id']},train\n")
    with pytest.raises(ValueError, match="exactly one role"):
        study.validate_folds(path, patients)


@pytest.fixture
def study_case(tmp_path, monkeypatch):
    patients = [f"P{i}" for i in range(5)]
    inputs = {
        patient: _inputs(patient, _reference([[8 + i, 1, 2], [1, 8, 3 + i]]))
        for i, patient in enumerate(patients)
    }
    base = ModelConfig(
        feature_dim=2,
        state_dim=3,
        gene_ids=("EPCAM", "CD8A", "MKI67"),
        supervised_gene_ids=("EPCAM", "CD8A", "MKI67"),
        normalization_gene_ids=("EPCAM", "CD8A", "MKI67"),
        state_gene_ids=("EPCAM", "CD8A", "MKI67"),
        hidden_dim=4,
    )
    config = {
        "schema": "heir.htapp_region_workflow.v1",
        "preparation": {"embedding_gene_ids": list(base.state_gene_ids)},
        "model": base.to_dict(),
    }
    config_path, folds_path, protocol_path = (
        tmp_path / n for n in ("config.json", "folds.csv", "protocol.json")
    )
    config_path.write_text(json.dumps(config))
    _fold_file(folds_path, patients)
    protocol = {
        "schema": "heir.internal_patient_study.v1",
        "outer_folds": list(range(5)),
        "ranks": [2, 16],
        "training": asdict(study.TrainingConfig(epochs=100, abundance_weight=0)),
        "arms": study._ARMS,
        "selection": study._SELECTION,
        "max_reference_states": 32,
        "prior_cohort_exposure": True,
        "independent_confirmation": False,
        "test_prediction_before_target_loading": True,
        "no_train_validation_refit": True,
        "outcome_gene_ids": list(base.gene_ids),
        "normalization_gene_ids": list(base.normalization_gene_ids),
        "config_sha256": study._sha(config_path),
        "folds_sha256": study._sha(folds_path),
        "paired_contrasts": [
            "he_rna-minus-he",
            "he_rna-minus-generic_reference",
            "he_rna-minus-constant_reference",
        ],
    }
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "study-output"
    folds = study.validate_folds(folds_path, patients)
    targets_opened, fits = [], []

    def prepare(root, biopsy, **kwargs):
        return inputs[biopsy], {"biopsy_id": biopsy}

    def target_loader(root, biopsy, prediction, provenance, model):
        fold = max(int(p.name.split("-")[1]) for p in output.glob("fold-*"))
        if folds[fold][biopsy] == "test":
            directory = output / f"fold-{fold}"
            assert (directory / "selection.json").is_file()
            receipt = json.loads((directory / "test_predictions_complete.json").read_text())
            assert len(receipt["prediction_sha256"]) == 10
            for name, digest in receipt["prediction_sha256"].items():
                assert study._sha(directory / name) == digest
        targets_opened.append((fold, biopsy))
        ids = inputs[biopsy].cell_ids
        target = SpatialTargets(
            ids,
            base.gene_ids,
            np.asarray(
                [[8, 1, 2], [6, 2, 3], [3, 7, 1], [7, 2, 1], [5, 4, 3], [1, 8, 2]], dtype=np.float32
            ),
            np.ones((6, 3), dtype=bool),
            biopsy,
            inputs[biopsy].specimen_id,
        )
        operator = ObservationOperator(ids, ids, np.arange(6), np.arange(6), np.ones(6))
        return target, operator, ("1",) * 3 + ("2",) * 3, {"biopsy_id": biopsy}

    def fast_fit(samples, model_config, training_config, *, preparation):
        # Exercise real initialization/prediction/serialization while avoiding a
        # redundant 3,500 optimization steps in the orchestration boundary test.
        train_patients = sorted({s.inputs.patient_id for s in samples})
        assert train_patients == sorted(
            p for p, role in preparation["fold_roles"].items() if role == "train"
        )
        assert training_config.epochs == 100
        if preparation.get("study_arm") == "generic_reference":
            assert all(s.inputs.proteins is None for s in samples)
            assert preparation["reference_pool"]["training_patient_ids"] == train_patients
        fits.append((preparation["outer_fold"], model_config, train_patients))
        torch.manual_seed(training_config.seed)
        return FittedModel(
            PersonalizedModel(model_config),
            np.zeros(2),
            np.ones(2),
            tuple(train_patients),
            [],
            preparation=preparation,
            training_config={
                **asdict(training_config),
                "prediction_unit": "tissue_region",
                "coordinate_units": "registered_pixel",
            },
        )

    monkeypatch.setattr(
        study, "discover_cohort", lambda root: {p: {"patient_id": p} for p in patients}
    )
    monkeypatch.setattr(study, "prepare_cohort_inputs", prepare)
    monkeypatch.setattr(study, "qualified_targets", target_loader)
    monkeypatch.setattr(study, "fit", fast_fit)
    return (
        tmp_path,
        config_path,
        folds_path,
        protocol_path,
        output,
        inputs,
        fits,
        targets_opened,
        prepare,
    )


def test_study_boundaries_bundles_and_generic_cli_reload(study_case, monkeypatch):
    root, config, folds, protocol, output, inputs, fits, opened, prepare = study_case
    result = study.run_study(root, config, folds, protocol, output)
    assert len(fits) == 35  # Two candidates plus five new fits per fold; selected RNA is reused.
    assert len(result["selections"]) == 5
    assert all(item["n_patients"] == 5 for item in result["arms"].values())
    assert result["postfit_transfer"]["constant_reference"]["mean_patient_spearman"] is None
    assert len(opened) == 25
    identity = json.loads((output / "identity.json").read_text())
    assert "heir/personalized/model.py" in identity["source_file_sha256"]
    roles = study.validate_folds(folds, inputs)[0]
    patient = next(p for p, role in roles.items() if role == "test")
    bundle = output / "fold-0/generic_reference.bundle"
    pool = study.load_pooled_reference(bundle)
    fitted = FittedModel.load(bundle)
    expected, _ = load_prediction(output / f"fold-0/test-{patient}-generic_reference.npz")
    actual = fitted.predict(replace(inputs[patient], reference=pool, proteins=None))
    np.testing.assert_array_equal(actual.expression, expected.expression)
    monkeypatch.setattr("heir.personalized.cohort_cli.prepare_cohort_inputs", prepare)
    exported = root / "generic-reloaded.npz"
    main(
        [
            "cohort-predict",
            "--root",
            str(root),
            "--biopsy",
            patient,
            "--bundle",
            str(bundle),
            "--output",
            str(exported),
        ]
    )
    reloaded, provenance = load_prediction(exported)
    np.testing.assert_array_equal(reloaded.expression, expected.expression)
    assert provenance["reference_control"]["training_patient_ids"] == list(
        fitted.training_patient_ids
    )
    path = bundle / "pooled_reference.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        study.load_pooled_reference(bundle)


def test_protocol_hashes_checked_before_any_data_read(study_case, monkeypatch):
    root, config, folds, protocol, output, *_ = study_case
    monkeypatch.setattr(
        study, "discover_cohort", lambda _: pytest.fail("data discovery preceded bindings")
    )
    config.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="config_sha256"):
        study.run_study(root, config, folds, protocol, output)
    assert not output.exists()
