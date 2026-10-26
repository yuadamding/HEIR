"""Joint study inventory, information barriers, and real fitting orchestration."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from test_joint_training import sample_fixture

from heir.personalized import joint_workflow as workflow
from heir.personalized.protein_features import save_joint_inputs
from heir.personalized.study import _sha
from heir.personalized.train import make_patient_folds


def _protocol(config_sha):
    return {
        "schema": "heir.joint_patient_study.v1",
        "model_config_sha256": config_sha,
        "principal_arms": list(workflow.PRINCIPAL_ARMS),
        "control_arms": list(workflow.CONTROL_ARMS),
        "seeds": [0, 17, 42],
        "control_seeds": [0],
        "outer_folds": list(range(5)),
    }


def test_fixed_study_has_exactly_100_unique_fits_and_no_seed_pseudoreplication():
    plan = workflow.fit_schedule(_protocol("fixture"))
    assert len(plan) == 100
    assert len({tuple(row.values()) for row in plan}) == 100
    assert sum(row["arm"] in workflow.PRINCIPAL_ARMS for row in plan) == 90
    assert {row["seed"] for row in plan if row["arm"] in workflow.CONTROL_ARMS} == {0}
    with pytest.raises(ValueError, match="seed inventory"):
        workflow.fit_schedule(_protocol("fixture"), [0])


def _report(patient, arm, seed, score):
    return {
        "patient_id": patient,
        "specimen_id": patient + "-s",
        "arm": arm,
        "seed": seed,
        "component": "full",
        workflow._SCORE: score,
        "normalized_log_mse": 2.0,
        "squared_bias": 0.5,
        "centered_mse": 1.5,
        "evaluation_observation_ids": ["a", "b", "c"],
        "evaluation_gene_ids": ["g1", "g2"],
        "normalization_gene_ids": ["g1", "g2"],
        "target_gene_eligibility": [{"g1": True}],
    }


def test_patient_means_average_seeds_first_and_undefined_remains_undefined():
    rows = [
        _report(p, a, s, v)
        for p, values in (("p1", [0, 0.3, 0.6]), ("p2", [0.6, 0.6, 0.6]))
        for a in ("HRP", "HR")
        for s, v in zip((0, 17, 42), values)
    ]
    result = workflow.summarize_joint(rows)
    assert result["arms"]["HRP"]["n_patients"] == 2
    assert result["arms"]["HRP"][workflow._SCORE] == pytest.approx(0.45)
    assert len(result["paired_contrasts"]) == 2
    rows[0][workflow._SCORE] = None
    result = workflow.summarize_joint(rows)
    assert result["arms"]["HRP"][workflow._SCORE] is None
    assert result["paired_summary"]["HRP-minus-HR"]["mean_paired_delta_" + workflow._SCORE] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_observation_ids", ["a", "different", "c"]),
        ("evaluation_gene_ids", ["g2", "g1"]),
        ("target_gene_eligibility", [{"g1": False}]),
    ],
)
def test_paired_contrasts_reject_different_rows_or_target_gene_eligibility(field, value):
    rows = [_report("patient", arm, 0, 0.1) for arm in ("HRP", "HR")]
    rows[1][field] = value
    with pytest.raises(ValueError, match="identical target rows and gene eligibility"):
        workflow.summarize_joint(rows)


def test_generic_pool_preserves_broad_types_and_equal_patient_RNA_mass():
    first, config = sample_fixture("one")
    second, _ = sample_fixture("two")
    second = replace(
        second,
        inputs=replace(
            second.inputs,
            reference=replace(
                second.inputs.reference,
                expression=second.inputs.reference.expression * 3,
            ),
        ),
    )
    pooled, metadata = workflow.typed_training_reference(
        [first.inputs, second.inputs],
        gene_ids=config.gene_ids,
        embedding_gene_ids=config.state_gene_ids,
        max_states=4,
    )
    assert set(pooled.state_types) == {"tumor", "immune"}
    assert metadata["training_patient_ids"] == ["one", "two"]
    assert not metadata["target_expression_used"]
    expected = 2 * np.average(
        first.inputs.reference.expression, weights=first.inputs.reference.proportions, axis=0
    )
    np.testing.assert_allclose(
        np.average(pooled.expression, weights=pooled.proportions, axis=0), expected
    )


def _prepared_fixture(tmp_path, n_patients=1):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    sample, base = sample_fixture("p0")
    base = replace(base, supervised_gene_ids=base.gene_ids)
    config = {
        "schema": "heir.joint_region_workflow.v1",
        "model": base.to_dict(),
        "preparation": {"region_size": 100, "max_states": 4},
        "training": {"epochs": 2, "warmup_epochs": 1, "learning_rate": 0.001},
    }
    config_path = prepared / "config.json"
    config_path.write_text(json.dumps(config))
    records = {}
    for i in range(n_patients):
        biopsy = f"fixture-{i}"
        current, _ = sample_fixture(f"p{i}")
        provenance = {
            "biopsy_id": biopsy,
            "protein": {
                "panel": {
                    "qualified_features": [
                        {"marker_id": "m1", "gene_ids": ["a"]},
                        {"marker_id": "m2", "gene_ids": ["b"]},
                    ]
                }
            },
        }
        save_joint_inputs(current.inputs, prepared / "inputs" / biopsy, provenance)
        records[biopsy] = {
            "patient_id": current.inputs.patient_id,
            "specimen_id": current.inputs.specimen_id,
            "provenance": provenance,
        }
    folds_path = prepared / "folds.csv"
    if n_patients > 5:
        rows = make_patient_folds([f"p{i}" for i in range(n_patients)])
        with folds_path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        folds_path.write_text("pilot fixture\n")
    metadata = {
        "schema": "heir.joint_prepared.v1",
        "root": "not_opened",
        "config_sha256": _sha(config_path),
        "folds_sha256": _sha(folds_path),
        "biopsies": records,
        "spatial_targets_loaded": False,
    }
    (prepared / "prepared.json").write_text(json.dumps(metadata))
    return prepared, sample, base, config_path, folds_path


def test_generic_inputs_do_not_open_query_RNA_and_HP_RP_keep_unused_modalities_absent(tmp_path):
    prepared, sample, _, _, _ = _prepared_fixture(tmp_path)
    directory = prepared / "inputs" / "fixture-0"
    (directory / "reference.npz").unlink()
    generic, _ = workflow._inputs(prepared, "fixture-0", "HRP_generic", sample.inputs.reference)
    assert generic.reference is sample.inputs.reference
    hp, _ = workflow._inputs(prepared, "fixture-0", "HP")
    assert hp.reference is None
    (directory / "image.npz").unlink()
    # RP only needs its RNA/protein files; restore the RNA payload from the original specimen.
    save_joint_inputs(sample.inputs, tmp_path / "replacement", {})
    (directory / "reference.npz").write_bytes((tmp_path / "replacement/reference.npz").read_bytes())
    rp, _ = workflow._inputs(prepared, "fixture-0", "RP")
    assert rp.image_features is None


def test_pilot_runs_seven_real_fit_paths_reload_interventions_and_all_output_tables(
    tmp_path, monkeypatch
):
    prepared, sample, _, config_path, _ = _prepared_fixture(tmp_path)
    calls = []

    def target(*args):
        calls.append("target")
        return sample.targets, sample.observation, sample.observation_groups, {"fixture": True}

    monkeypatch.setattr(workflow, "qualified_targets", target)
    output = tmp_path / "pilot"
    result = workflow.pilot_joint(prepared, ["fixture-0"], config_path, output)
    assert result["fit_count"] == 7
    assert len(list(output.glob("*/*.bundle"))) == 7
    assert calls == ["target"]
    for name in (
        "patient_scores",
        "paired_contrasts",
        "error_decomposition",
        "modality_interventions",
        "assignment_diagnostics",
        "marker_coverage",
    ):
        assert (output / f"{name}.tsv").stat().st_size > 1
    exports = json.loads((output / "exports.json").read_text())
    assert set(workflow.INTERVENTIONS) <= {r["component"] for r in exports}
    for receipt in output.glob("*/*.acceptance.json"):
        acceptance = json.loads(receipt.read_text())
        assert acceptance["save_reload_expression_exact"]
        assert acceptance["test_targets_loaded"]
        assert not acceptance["held_out_test"]


def test_all100_fit_exports_precede_each_folds_test_target_reads(tmp_path, monkeypatch):
    prepared, _, _, _, folds_path = _prepared_fixture(tmp_path, n_patients=13)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(_protocol(_sha(prepared / "config.json"))))
    output = tmp_path / "study"
    planned = []

    def fit_export(*args, **kwargs):
        arm, seed, test, directory, fold = args[3], args[4], args[7], args[11], args[12]
        planned.append((fold, arm, seed))
        exports = []
        for biopsy in test:
            path = directory / f"{biopsy}-{arm}-{seed}.npz"
            path.write_bytes(b"saved prediction before evaluation")
            exports.append(
                {
                    "path": str(path),
                    "biopsy_id": biopsy,
                    "arm": arm,
                    "seed": seed,
                    "component": "full",
                }
            )
        return exports, [], []

    def target(prepared_arg, metadata, biopsy, model, directory):
        fold = int(directory.parent.name.split("-")[-1])
        roles = workflow.validate_folds(folds_path, {f"p{i}" for i in range(13)})[fold]
        if roles[metadata["biopsies"][biopsy]["patient_id"]] == "test":
            assert len([p for p in planned if p[0] == fold]) == 20
            receipt = json.loads((directory.parent / "test_predictions_complete.json").read_text())
            assert not receipt["test_targets_loaded"]
        return (None, None, None, {})

    monkeypatch.setattr(workflow, "_fit_and_export", fit_export)
    monkeypatch.setattr(workflow, "_target", target)
    monkeypatch.setattr(workflow, "_evaluate_exports", lambda *args: ([], [], []))
    result = workflow.study_joint(prepared, protocol_path, folds_path, [0, 17, 42], output)
    assert len(planned) == result["fit_count"] == 100


def test_input_metadata_binding_detects_mutation_without_opening_unused_payload(tmp_path):
    prepared, _, _, _, _ = _prepared_fixture(tmp_path)
    metadata, _, _ = workflow._prepared(prepared)
    inventory = workflow._input_inventory(prepared, metadata)
    path = prepared / "inputs/fixture-0/metadata.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="input metadata changed"):
        workflow._check_input_inventory(prepared, metadata, inventory)


def test_joint_cli_preserves_reconciliation_command():
    import argparse

    from heir.personalized.cohort_cli import add_commands

    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "cohort-joint",
            "study",
            "--prepared",
            "p",
            "--config",
            "c",
            "--folds",
            "f",
            "--output",
            "o",
        ]
    )
    assert args.seeds == [0, 17, 42]
    args = parser.parse_args(
        [
            "cohort-reconcile",
            "--root",
            "r",
            "--pipeline",
            "p",
            "--measurement",
            "m",
            "--config",
            "c",
            "--output",
            "o",
        ]
    )
    assert args.command == "cohort-reconcile"
