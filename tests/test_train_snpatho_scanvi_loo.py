import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_snpatho_scanvi.py"


def _load_script():
    fake_anndata = types.ModuleType("anndata")
    fake_anndata.AnnData = object
    fake_scvi = types.ModuleType("scvi")
    fake_scvi.__version__ = "test"
    fake_scvi.settings = types.SimpleNamespace(seed=None)
    fake_scvi.model = types.SimpleNamespace(SCANVI=object, SCVI=object)
    previous = {name: sys.modules.get(name) for name in ("anndata", "scvi")}
    sys.modules["anndata"] = fake_anndata
    sys.modules["scvi"] = fake_scvi
    try:
        spec = importlib.util.spec_from_file_location("train_snpatho_scanvi_loo", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


TRAIN = _load_script()


def test_molecular_input_freeze_binds_every_file_and_rejects_long_run_mutation(tmp_path):
    h5ad = tmp_path / "reference.h5ad"
    reference = tmp_path / "reference500.npz"
    panel = tmp_path / "panel.tsv"
    h5ad.write_bytes(b"h5ad-v1")
    reference.write_bytes(b"reference-v1")
    panel.write_text("GENE1\n", encoding="utf-8")

    records = TRAIN._freeze_input_files(
        (
            ("reference_h5ad:4066", h5ad),
            ("reference500:4066", reference),
            ("gene_panel", panel),
        )
    )

    assert [record["role"] for record in records] == [
        "reference_h5ad:4066",
        "reference500:4066",
        "gene_panel",
    ]
    assert all(len(str(record["sha256"])) == 64 for record in records)
    TRAIN._assert_input_files_unchanged(records, phase="before_test_output")

    reference.write_bytes(b"reference-v2")
    with pytest.raises(RuntimeError, match="molecular input changed during before_manifest"):
        TRAIN._assert_input_files_unchanged(records, phase="before_manifest")


@pytest.mark.parametrize("held_out", TRAIN.SAMPLES)
def test_leave_one_donor_out_plan_excludes_target_from_every_fit_provenance(held_out):
    plan = TRAIN._training_plan(held_out)
    provenance = TRAIN._training_partition_provenance(plan)

    assert plan.mode == "leave_one_donor_out"
    assert plan.training_donors == tuple(sample for sample in TRAIN.SAMPLES if sample != held_out)
    assert held_out not in provenance["backbone_training_donors"]
    assert held_out not in provenance["decoder_training_donors"]
    assert plan.latent_role(held_out) == "frozen_query_encoder"
    assert all(plan.latent_role(sample) == "reference_encoder" for sample in plan.training_donors)
    held_out_latent = TRAIN._latent_artifact_provenance(plan, held_out)
    assert held_out_latent == {
        "inference_role": "frozen_query_encoder",
        "latent_training_donors": list(plan.training_donors),
        "sample_expression_used_for_model_fitting": False,
        "sample_labels_used_for_model_fitting": False,
        "cell_type_label_source": "frozen_training_donor_SCANVI_classifier",
        "cell_type_label_training_donors": list(plan.training_donors),
        "sample_annotation_used_for_cell_type_labels": False,
    }
    assert all(
        TRAIN._latent_artifact_provenance(plan, sample)["sample_expression_used_for_model_fitting"]
        is True
        for sample in plan.training_donors
    )
    assert provenance["held_out_mapping"] == {
        "method": "SCANVI.load_query_data_without_query_training",
        "labels_available_to_query_model": False,
        "query_train_called": False,
        "query_parameters_frozen_before_inference": True,
        "inference_guard_enabled_without_optimization": True,
        "held_out_expression_used_for_fitting": False,
        "held_out_annotation_used_for_label_mapping": False,
        "label_mapping_method": "frozen_training_donor_SCANVI_classifier",
        "label_training_donors": list(plan.training_donors),
    }


def test_missing_held_out_sample_is_explicit_historical_negative_control():
    plan = TRAIN._training_plan(None)
    provenance = TRAIN._training_partition_provenance(plan)

    assert plan.training_donors == TRAIN.SAMPLES
    assert plan.mode == "historical_all_donor_negative_control"
    assert provenance["all_donor_behavior_role"] == "historical_negative_control"
    assert provenance["held_out_mapping"] is None


def test_frozen_query_labels_replace_target_annotations_in_reference_order():
    aligned = TRAIN._align_frozen_query_labels(
        ["cell-2", "cell-1"],
        ["cell-1", "cell-2"],
        ["training-donor-T", "training-donor-B"],
    )

    assert aligned.tolist() == ["training-donor-B", "training-donor-T"]
    assert "published-target-label" not in aligned.tolist()
    with pytest.raises(ValueError, match="differ from the RNA reference"):
        TRAIN._align_frozen_query_labels(
            ["cell-1", "cell-2"],
            ["cell-1", "foreign-cell"],
            ["T", "B"],
        )


def test_training_plan_rejects_unknown_donor_and_two_donor_validation_needs_no_target_rows():
    with pytest.raises(ValueError, match="held-out sample"):
        TRAIN._training_plan("not-a-donor")

    sections = np.asarray(["4066", "4066", "4399", "4399"])
    mask = TRAIN._stratified_validation_mask(
        sections,
        fraction=0.2,
        seed=17,
        samples=("4066", "4399"),
    )
    repeated = TRAIN._stratified_validation_mask(
        sections,
        fraction=0.2,
        seed=17,
        samples=("4066", "4399"),
    )
    assert np.array_equal(mask, repeated)
    assert mask[:2].sum() == mask[2:].sum() == 1


def test_held_out_plan_rejects_specimen_batch_sensitivity():
    plan = TRAIN._training_plan("4411")
    with pytest.raises(ValueError, match="specimen_batch_sensitivity is invalid"):
        TRAIN._validate_molecular_design_for_plan(plan, "specimen_batch_sensitivity")

    TRAIN._validate_molecular_design_for_plan(plan, "no_specimen_correction")
    TRAIN._validate_molecular_design_for_plan(plan, "technical_batch_only")
    TRAIN._validate_molecular_design_for_plan(
        TRAIN._training_plan(None),
        "specimen_batch_sensitivity",
    )

    with pytest.raises(ValueError, match="specimen_batch_sensitivity is invalid"):
        TRAIN.main(
            [
                "--held-out-sample",
                "4411",
                "--molecular-design",
                "specimen_batch_sensitivity",
            ]
        )


class _CovariateAnnData:
    def __init__(self, sections, batches):
        self.obs = pd.DataFrame({"section_id": sections, "chemistry_run": batches})


def test_held_out_technical_levels_must_exist_in_reference_training_donors():
    plan = TRAIN._training_plan("4411")
    valid = _CovariateAnnData(
        ["4066", "4066", "4399", "4399", "4411", "4411"],
        ["run-a", "run-b", "run-a", "run-b", "run-a", "run-b"],
    )
    assert TRAIN._held_out_technical_batch_contract(
        valid,
        plan=plan,
        key="chemistry_run",
        reference_levels=("run-a", "run-b"),
    ) == ("run-a", "run-b")

    unseen = _CovariateAnnData(
        ["4066", "4066", "4399", "4399", "4411", "4411"],
        ["run-a", "run-b", "run-a", "run-b", "run-a", "run-c"],
    )
    with pytest.raises(ValueError, match="absent from the reference training donors: run-c"):
        TRAIN._held_out_technical_batch_contract(
            unseen,
            plan=plan,
            key="chemistry_run",
            reference_levels=("run-a", "run-b"),
        )


def test_query_technical_categories_are_locked_to_reference_levels():
    query = _CovariateAnnData(["4411", "4411"], ["run-b", "run-a"])
    observed = TRAIN._lock_query_technical_categories(
        query,
        key="chemistry_run",
        reference_levels=("run-a", "run-b"),
    )

    assert observed == ("run-a", "run-b")
    assert tuple(query.obs["chemistry_run"].cat.categories) == ("run-a", "run-b")
    assert query.obs["chemistry_run"].isna().sum() == 0


class _FakeAnnData:
    def __init__(self):
        self.obs = pd.DataFrame(
            {
                "major_annotation": ["T", "B"],
                "source_cell_id": ["cell-1", "cell-2"],
            }
        )

    @property
    def n_obs(self):
        return len(self.obs)

    def copy(self):
        copied = _FakeAnnData()
        copied.obs = self.obs.copy(deep=True)
        return copied


class _FakeParameter:
    def __init__(self, size):
        self.size = size
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return self.size


class _FakeModule:
    def __init__(self):
        self.values = [_FakeParameter(3), _FakeParameter(5)]
        self.evaluation = False

    def eval(self):
        self.evaluation = True

    def parameters(self):
        return iter(self.values)


class _FakeQueryModel:
    def __init__(self):
        self.module = _FakeModule()

    def train(self, *args, **kwargs):
        raise AssertionError("query training must never be called")

    def get_latent_representation(self):
        return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    def predict(self, *, soft):
        assert soft is False
        return np.asarray(["T", "B"])


def test_frozen_query_mapping_scrubs_labels_and_freezes_every_parameter(monkeypatch, tmp_path):
    observed = {}
    query_model = _FakeQueryModel()

    class _FakeSCANVI:
        @classmethod
        def load_query_data(cls, query, reference_model, **kwargs):
            observed["query"] = query
            observed["reference_model"] = reference_model
            observed["kwargs"] = kwargs
            assert "major_annotation" not in query.obs
            return query_model

    monkeypatch.setattr(TRAIN.scvi.model, "SCANVI", _FakeSCANVI)
    original = _FakeAnnData()
    model_path = tmp_path / "reference-model"

    fitted_query, query, latent, predicted_labels, audit = TRAIN._frozen_query_mapping(
        original,
        reference_model=model_path,
        latent_dim=2,
    )

    assert fitted_query is query_model
    assert "major_annotation" in original.obs
    assert "major_annotation" not in query.obs
    assert latent.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert predicted_labels.tolist() == ["T", "B"]
    assert observed["reference_model"] == str(model_path)
    assert observed["kwargs"] == {
        "inplace_subset_query_vars": True,
        "accelerator": "gpu",
        "device": 0,
    }
    assert query_model.module.evaluation is True
    assert query_model.is_trained is True
    assert all(not parameter.requires_grad for parameter in query_model.module.values)
    assert audit == {
        "labels_removed_before_registry_transfer": True,
        "query_train_called": False,
        "parameters_frozen_before_inference": True,
        "inference_guard_enabled_without_optimization": True,
        "frozen_parameter_count": 8,
        "cells_mapped": 2,
        "label_predictions_generated_without_target_annotation": True,
        "label_prediction_rule": "SCANVI.predict(soft=False)",
        "predicted_label_sha256": TRAIN._ordered_string_sha256(["T", "B"]),
        "predicted_label_counts": {"B": 1, "T": 1},
    }
