from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.personalized.schema import Prediction
from heir.personalized.serialization import load_prediction, save_prediction


def prediction():
    q = torch.tensor([[0.1, 0.6, 0.3], [0.3, 0.2, 0.5]])
    transfer = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return Prediction(
        cell_ids=("c1", "c2"),
        gene_ids=("g1", "g2"),
        state_ids=("unsupported", "s1", "s2"),
        coordinates_um=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        assignment=q,
        conditional_assignment=q[:, 1:] / q[:, 1:].sum(1, keepdim=True),
        transfer=transfer,
        log_correction=torch.zeros((2, 2)),
        expression=transfer,
        assignment_entropy=-(q * q.log()).sum(1),
        reference_support=1 - q[:, 0],
        registration_sensitivity=torch.tensor([float("nan"), 0.5]),
        supervised_genes=torch.tensor([True, False]),
        independently_evaluated_genes=torch.tensor([False, True]),
        patient_id="patient1",
        specimen_id="specimen1",
    )


@pytest.mark.parametrize("prediction_unit", ["nucleus", "tissue_region"])
@pytest.mark.parametrize("coordinate_units", ["um", "registered_pixel"])
def test_exact_prediction_roundtrip_preserves_flags_and_provenance(
    tmp_path, prediction_unit, coordinate_units
):
    original = replace(
        prediction(), prediction_unit=prediction_unit, coordinate_units=coordinate_units
    )
    path = tmp_path / "prediction.npz"
    provenance = {"training_patient_ids": ["patient2"], "fold_roles": {"patient1": "test"}}
    save_prediction(original, path, provenance)
    restored, actual_provenance = load_prediction(path)
    assert actual_provenance == provenance
    for name in (
        "cell_ids",
        "gene_ids",
        "state_ids",
        "patient_id",
        "specimen_id",
        "expression_scale",
        "reference_support_calibrated",
        "unsupported_policy",
        "prediction_unit",
        "coordinate_units",
    ):
        assert getattr(restored, name) == getattr(original, name)
    for name in (
        "coordinates_um",
        "assignment",
        "conditional_assignment",
        "transfer",
        "expression",
        "log_correction",
        "assignment_entropy",
        "reference_support",
        "registration_sensitivity",
        "supervised_genes",
        "independently_evaluated_genes",
    ):
        torch.testing.assert_close(
            getattr(restored, name), getattr(original, name), equal_nan=True, rtol=0, atol=0
        )
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["reference_transferred_genes"].tolist() == [True, True]
        np.testing.assert_array_equal(arrays["unsupported_probability"], original.assignment[:, 0])
        np.testing.assert_array_equal(arrays["expression_adjustment"], np.zeros((2, 2)))
        assert all(arrays[name].dtype.kind in "fiubU" for name in arrays.files)
        metadata = json.loads(arrays["metadata_json"].item())
        assert metadata["schema"] == "heir.personalized.prediction.v3"
        assert metadata["prediction_unit"] == prediction_unit
        assert metadata["coordinate_units"] == coordinate_units
        assert "coordinates_um" not in arrays.files
        np.testing.assert_array_equal(arrays["coordinates"], original.coordinates)
        assert metadata["reference_support_calibrated"] is False
        assert metadata["unsupported_policy"] == "conditional_reference_unvalidated"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_prediction(original, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_ids": ("c1", "c1")},
        {"gene_ids": ("g1", "g1")},
        {"state_ids": ("unsupported", "s1", "s1")},
        {"assignment": torch.ones((2, 3))},
        {"conditional_assignment": torch.ones((2, 2))},
        {"conditional_assignment": torch.tensor([[0.1, 0.9], [0.9, 0.1]])},
        {"reference_support_calibrated": True},
        {"unsupported_policy": "silently_zero_rna"},
        {"expression": torch.ones((2, 1))},
        {"expression": torch.tensor([[float("nan"), 1.0], [1.0, 1.0]])},
        {"transfer": torch.full((2, 2), -1.0)},
        {"reference_support": torch.ones(2)},
        {"supervised_genes": torch.ones(2)},
        {"registration_sensitivity": torch.tensor([float("inf"), 0.0])},
        {"expression_scale": "absolute_counts"},
        {"patient_id": ""},
        {"prediction_unit": "protein_cell"},
        {"coordinate_units": "pixel"},
    ],
)
def test_invalid_prediction_cannot_be_saved(tmp_path, changes):
    path = tmp_path / "prediction.npz"
    with pytest.raises(ValueError):
        save_prediction(replace(prediction(), **changes), path)
    assert not path.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cell_ids", np.asarray(["c1", "c1"])),
        ("gene_ids", np.asarray(["g1", "g1"])),
        ("expression", np.ones((2, 3))),
        ("expression_adjustment", np.ones((2, 2), dtype=np.float32)),
        ("reference_transferred_genes", np.zeros(2, dtype=bool)),
    ],
)
def test_malformed_archive_axes_or_aliases_cannot_be_scored(tmp_path, field, value):
    path = tmp_path / "original.npz"
    save_prediction(prediction(), path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    data[field] = value
    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, **data)
    with pytest.raises(ValueError):
        load_prediction(malformed)


def test_he_only_export_does_not_claim_reference_transfer(tmp_path):
    original = replace(
        prediction(),
        state_ids=("unsupported",),
        assignment=torch.ones((2, 1)),
        conditional_assignment=torch.empty((2, 0)),
        assignment_entropy=torch.zeros(2),
        reference_support=torch.zeros(2),
        transfer=torch.zeros((2, 2)),
        unsupported_policy="he_only",
    )
    path = tmp_path / "he.npz"
    save_prediction(original, path)
    with np.load(path, allow_pickle=False) as archive:
        assert not archive["reference_transferred_genes"].any()
    restored, _ = load_prediction(path)
    assert restored.state_ids == ("unsupported",)


def test_conditional_distribution_survives_zero_joint_support_export(tmp_path):
    original = replace(
        prediction(),
        assignment=torch.tensor([[1.0, 0.0, 0.0]] * 2),
        reference_support=torch.zeros(2),
        assignment_entropy=torch.zeros(2),
    )
    path = tmp_path / "underflow.npz"
    save_prediction(original, path)
    restored, _ = load_prediction(path)
    torch.testing.assert_close(restored.conditional_assignment, original.conditional_assignment)
    torch.testing.assert_close(restored.transfer, original.transfer)
    assert not restored.reference_support.any()


def test_old_prediction_schema_cannot_be_silently_reinterpreted(tmp_path):
    path = tmp_path / "original.npz"
    save_prediction(prediction(), path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(data["metadata_json"].item())
    metadata["schema"] = "heir.personalized.prediction.v1"
    data["metadata_json"] = np.asarray(json.dumps(metadata))
    old = tmp_path / "old.npz"
    np.savez(old, **data)
    with pytest.raises(ValueError, match="unsupported prediction archive schema"):
        load_prediction(old)


def test_legacy_v2_load_preserves_explicit_nucleus_micrometre_contract(tmp_path):
    original = prediction()
    current = tmp_path / "current.npz"
    provenance = {"model_config": {"normalization_gene_ids": ["g1", "g2"]}}
    save_prediction(original, current, provenance)
    with np.load(current, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(data["metadata_json"].item())
    metadata["schema"] = "heir.personalized.prediction.v2"
    del metadata["prediction_unit"], metadata["coordinate_units"]
    data["metadata_json"] = np.asarray(json.dumps(metadata))
    data["coordinates_um"] = data.pop("coordinates")
    legacy = tmp_path / "v2.npz"
    np.savez(legacy, **data)
    restored, restored_provenance = load_prediction(legacy)
    assert restored.prediction_unit == "nucleus"
    assert restored.coordinate_units == "um"
    assert restored_provenance == provenance
    torch.testing.assert_close(restored.coordinates, original.coordinates_um, atol=0, rtol=0)
    torch.testing.assert_close(restored.expression, original.expression, atol=0, rtol=0)
    metadata["coordinate_units"] = "registered_pixel"
    data["metadata_json"] = np.asarray(json.dumps(metadata))
    contradictory = tmp_path / "contradictory-v2.npz"
    np.savez(contradictory, **data)
    with pytest.raises(ValueError, match="v2 predictions require nucleus units and um"):
        load_prediction(contradictory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction_unit", None),
        ("prediction_unit", "cell"),
        ("prediction_unit", []),
        ("coordinate_units", None),
        ("coordinate_units", "unknown"),
        ("coordinate_units", {}),
    ],
)
def test_v3_archive_requires_valid_prediction_and_coordinate_units(tmp_path, field, value):
    current = tmp_path / "current.npz"
    save_prediction(prediction(), current)
    with np.load(current, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(data["metadata_json"].item())
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    data["metadata_json"] = np.asarray(json.dumps(metadata))
    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, **data)
    with pytest.raises(ValueError, match=field):
        load_prediction(malformed)


def test_mutated_invalid_units_are_rejected_at_export(tmp_path):
    original = prediction()
    original.coordinate_units = "not_calibrated"
    with pytest.raises(ValueError, match="coordinate_units"):
        save_prediction(original, tmp_path / "invalid.npz")
