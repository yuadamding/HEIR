import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from heir.data import PrototypeSet
from heir.inference import PredictionBundle

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_snpatho_refinement_benchmark.py"
SPEC = importlib.util.spec_from_file_location("snpatho_refinement_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prediction_case(tmp_path, *, control=None, unsupported_source_type=False):
    checkpoint = tmp_path / "heir.pt"
    prototypes = tmp_path / "prototypes.npz"
    histology = tmp_path / "histology.npz"
    ood = tmp_path / "ood.npz"
    for path, value in ((checkpoint, b"checkpoint"), (ood, b"ood")):
        path.write_bytes(value)
    np.savez(histology, nucleus_ids=np.asarray(["n0", "n1"]))
    prototype_types = ["A", "B", *(("unsupported",) if unsupported_source_type else ())]
    prototype_ids = ["p0", "p1", *(("pX",) if unsupported_source_type else ())]
    PrototypeSet(
        prototype_ids=np.asarray(prototype_ids),
        sample_ids=np.asarray(
            ["4411" if control == "wrong_donor" else "4066"] * len(prototype_ids)
        ),
        cell_type_labels=np.asarray(prototype_types),
        means=np.zeros((len(prototype_ids), 2), dtype=np.float32),
        variances=np.ones((len(prototype_ids), 2), dtype=np.float32),
        weights=np.full(len(prototype_ids), 1.0 / len(prototype_ids)),
        donor_id="4411" if control == "wrong_donor" else "4066",
        block_id="source-block",
    ).save_npz(prototypes)

    bundle = PredictionBundle(
        nucleus_ids=np.asarray(["n0", "n1"]),
        coordinates_um=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        type_probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        type_names=np.asarray(["A", "B"]),
        labels=np.asarray([0, 1]),
        prototype_probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        prototype_ids=np.asarray(["p0", "p1"]),
        latent_mean=np.zeros((2, 2), dtype=np.float32),
        latent_variance=np.ones((2, 2), dtype=np.float32),
        expression_mean=np.ones((2, 2), dtype=np.float32),
        expression_lower=np.full((2, 2), 0.5, dtype=np.float32),
        expression_upper=np.full((2, 2), 1.5, dtype=np.float32),
        gene_names=np.asarray(["g0", "g1"]),
        unknown_probability=np.zeros(2, dtype=np.float32),
        abstain_score=np.zeros(2, dtype=np.float32),
        abstain=np.zeros(2, dtype=bool),
        ood_score=np.zeros(2, dtype=np.float32),
        refinement_round=4,
        sample_id="4066",
        donor_id="4066",
        slide_id="4066_FFPE",
        checkpoint_sha256=_sha256(checkpoint),
        prototype_sha256=_sha256(prototypes),
        histology_sha256=_sha256(histology),
        latent_space_id="latent-test",
        expression_space_id="log1p-test",
        model_version="heir.refined_model.v1",
        ood_sha256=_sha256(ood),
        ood_training_donors=np.asarray(["reference"]),
        inference_seed=17,
        latent_samples=20,
        probability_threshold=0.35,
        artifact_threshold=0.50,
    )
    output = tmp_path / "predictions.npz"
    bundle.to_npz(output)
    flags = dict(RUNNER._control_flags(control))
    telemetry = tmp_path / "prediction.telemetry.json"
    telemetry.write_text(
        json.dumps(
            {
                "schema": "heir.inference_telemetry.v1",
                "prediction_path": str(output.resolve()),
                "prediction_sha256": _sha256(output),
                "nuclei": 2,
                "genes": 2,
                "latent_samples": 20,
                "mc_chunk_size": 8,
                "negative_control": {
                    **flags,
                    "prototype_donor_id": "4411" if control == "wrong_donor" else "4066",
                    "seed": 17,
                    "transform": (
                        RUNNER._expected_shuffle_transform(control, 17, histology)
                        if control in {"image_shuffle", "graph_shuffle"}
                        else None
                    ),
                },
            }
        )
    )
    return output, telemetry, checkpoint, prototypes, histology, ood


def test_existing_prediction_validation_rejects_stale_input_and_partial_stage(tmp_path):
    output, telemetry, checkpoint, prototypes, histology, ood = _write_prediction_case(tmp_path)
    arguments = {
        "sample": "4066",
        "seed": 17,
        "checkpoint": checkpoint,
        "prototypes": prototypes,
        "histology": histology,
        "ood_artifact": ood,
        "refinement_round": 4,
    }
    RUNNER._validate_prediction(output, telemetry, **arguments)

    checkpoint.write_bytes(b"stale replacement")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        RUNNER._validate_prediction(output, telemetry, **arguments)

    first = tmp_path / "partial-one"
    first.write_bytes(b"present")
    with pytest.raises(RuntimeError, match="partial stage output"):
        RUNNER._run(
            ("not-executed",),
            (first, tmp_path / "partial-two"),
            False,
            validator=lambda: None,
            repository=tmp_path,
        )


def test_wrong_donor_validation_binds_the_explicit_source_donor(tmp_path):
    output, telemetry, checkpoint, prototypes, histology, ood = _write_prediction_case(
        tmp_path,
        control="wrong_donor",
    )
    arguments = {
        "sample": "4066",
        "seed": 17,
        "checkpoint": checkpoint,
        "prototypes": prototypes,
        "histology": histology,
        "ood_artifact": ood,
        "refinement_round": 4,
        "control": "wrong_donor",
    }
    RUNNER._validate_prediction(
        output,
        telemetry,
        prototype_donor_id="4411",
        **arguments,
    )
    with pytest.raises(ValueError, match="donor/seed provenance"):
        RUNNER._validate_prediction(
            output,
            telemetry,
            prototype_donor_id="4399",
            **arguments,
        )


def test_wrong_donor_validation_requires_filter_audit_for_unsupported_types(tmp_path):
    output, telemetry, checkpoint, prototypes, histology, ood = _write_prediction_case(
        tmp_path,
        control="wrong_donor",
        unsupported_source_type=True,
    )
    with pytest.raises(ValueError, match="lacks the required prototype-filter audit"):
        RUNNER._validate_prediction(
            output,
            telemetry,
            sample="4066",
            seed=17,
            checkpoint=checkpoint,
            prototypes=prototypes,
            histology=histology,
            ood_artifact=ood,
            refinement_round=4,
            control="wrong_donor",
            prototype_donor_id="4411",
        )


@pytest.mark.parametrize("control", ["image_shuffle", "graph_shuffle"])
def test_shuffle_validation_requires_exact_recipe_and_permutation_hashes(tmp_path, control):
    output, telemetry, checkpoint, prototypes, histology, ood = _write_prediction_case(
        tmp_path,
        control=control,
    )
    arguments = {
        "sample": "4066",
        "seed": 17,
        "checkpoint": checkpoint,
        "prototypes": prototypes,
        "histology": histology,
        "ood_artifact": ood,
        "refinement_round": 4,
        "control": control,
    }
    RUNNER._validate_prediction(output, telemetry, **arguments)

    payload = json.loads(telemetry.read_text(encoding="utf-8"))
    payload["negative_control"].pop("transform")
    telemetry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks deterministic transform hashes"):
        RUNNER._validate_prediction(output, telemetry, **arguments)

    payload["negative_control"]["transform"] = RUNNER._expected_shuffle_transform(
        control,
        17,
        histology,
    )
    payload["negative_control"]["transform"]["map_sha256"] = "0" * 64
    telemetry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stale transform map_sha256"):
        RUNNER._validate_prediction(output, telemetry, **arguments)


def _write_batch_identity(path, *, bag_id):
    np.savez(
        path,
        __contract__=np.asarray("heir.training_batch"),
        __version__=np.asarray(5),
        sample_id=np.asarray("4066"),
        bag_id=np.asarray(bag_id),
        donor_id=np.asarray("4066"),
        block_id=np.asarray("4066_FFPE"),
        analysis_role=np.asarray("development_retrospective"),
        source_artifacts=np.asarray(["source.npz"]),
        source_sha256=np.asarray(["a" * 64]),
        source_roles=np.asarray(["sample_assay"]),
    )


def test_checkpoint_history_pair_rejects_mismatched_json_identity(tmp_path):
    train_batch = tmp_path / "train.npz"
    validation_batch = tmp_path / "validation.npz"
    _write_batch_identity(train_batch, bag_id="4066_train")
    _write_batch_identity(validation_batch, bag_id="4066_validation")
    decoder = tmp_path / "decoder.pt"
    geometry = tmp_path / "geometry.npz"
    decoder.write_bytes(b"decoder")
    geometry.write_bytes(b"geometry")
    train_identity = RUNNER._batch_identity(train_batch)
    validation_identity = RUNNER._batch_identity(validation_batch)
    checkpoint = tmp_path / "heir.pt"
    torch.save(
        {
            "metadata": {
                "schema": "heir.trained_model.v1",
                "training_stage": "personalized",
                "seed": 17,
                "training_donors": ["4066"],
                "best_epoch": 3,
                "best_validation_loss": 1.25,
                "rna_vae_sha256": _sha256(decoder),
                "residual_geometry_sha256": _sha256(geometry),
                "training_batches": [dict(train_identity)],
                "validation_batches": [dict(validation_identity)],
            }
        },
        checkpoint,
    )
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "best_epoch": 3,
                "best_validation_loss": 1.25,
                "history": [{"epoch": 3.0, "validation/total": 1.25}],
            }
        )
    )
    kwargs = {
        "sample": "4066",
        "seed": 17,
        "train_batch": train_batch,
        "validation_batch": validation_batch,
        "decoder": decoder,
        "residual_geometry": geometry,
    }
    RUNNER._validate_trained_pair(checkpoint, history, **kwargs)
    history.write_text(
        json.dumps(
            {
                "best_epoch": 2,
                "best_validation_loss": 1.25,
                "history": [{"epoch": 2.0}],
            }
        )
    )
    with pytest.raises(ValueError, match="best_epoch"):
        RUNNER._validate_trained_pair(checkpoint, history, **kwargs)


def test_control_plan_covers_only_the_prespecified_ablation_seeds(tmp_path):
    plan = RUNNER.build_plan(
        tmp_path,
        samples=RUNNER.SAMPLES,
        seeds=RUNNER.SEEDS,
        controls=True,
    )
    assert len(plan) == 138
    cli_stages = [stage for stage in plan if stage.name != "build_views"]
    assert cli_stages
    for stage in cli_stages:
        assert stage.command[0] == str(Path(sys.executable).absolute())
        assert stage.command[1:3] == ("-I", "-c")
        assert str((tmp_path / "src").resolve()) in stage.command[3]
        assert stage.command[0] != "heir"
    for name in RUNNER.PREDICTION_CONTROLS:
        stages = [stage for stage in plan if stage.name == name]
        assert {stage.seed for stage in stages} == set(RUNNER.ABLATION_SEEDS)
        assert {stage.sample for stage in stages} == set(RUNNER.SAMPLES)
        assert len(stages) == len(RUNNER.ABLATION_SEEDS) * len(RUNNER.SAMPLES)
    wrong_donor = [stage for stage in plan if stage.control == "wrong_donor"]
    expected_pairings = set(RUNNER.wrong_donor_pairings(RUNNER.SAMPLES))
    observed_pairings = {(stage.sample, stage.prototype_donor_id) for stage in wrong_donor}
    assert observed_pairings == expected_pairings
    assert {stage.seed for stage in wrong_donor} == set(RUNNER.ABLATION_SEEDS)
    assert len(wrong_donor) == len(expected_pairings) * len(RUNNER.ABLATION_SEEDS)
    for target, source in expected_pairings:
        stages = [
            stage
            for stage in wrong_donor
            if stage.sample == target and stage.prototype_donor_id == source
        ]
        assert {stage.seed for stage in stages} == set(RUNNER.ABLATION_SEEDS)
        assert all(stage.name == "wrong_donor_" + source for stage in stages)


def test_source_bound_cli_executes_with_current_environment_and_repository_source() -> None:
    command = RUNNER._heir_source_command(ROOT, "--help")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "HEIR molecular spatialization" in completed.stdout
    assert command[0] == str(Path(sys.executable).absolute())
    assert command[1:3] == ["-I", "-c"]


def test_unknown_mass_sensitivity_plan_is_isolated_cuda_and_has_no_duplicate_ablations(tmp_path):
    plan = RUNNER.build_plan(
        tmp_path,
        samples=("4066",),
        seeds=(17,),
        unknown_mass_sensitivity=True,
    )
    assert {stage.unknown_mass for stage in plan} == set(RUNNER.UNKNOWN_MASS_SENSITIVITY)
    assert len(plan) == len(RUNNER.UNKNOWN_MASS_SENSITIVITY) * 5
    assert {stage.name for stage in plan} == {
        "train_round0",
        "build_views",
        "refine",
        "predict_round0",
        "predict_refined",
    }
    assert all(path.is_absolute() for stage in plan for path in stage.outputs)
    output_roots = {stage.outputs[0].parent for stage in plan if stage.name == "train_round0"}
    assert len(output_roots) == len(RUNNER.UNKNOWN_MASS_SENSITIVITY)
    for stage in plan:
        assert stage.command[stage.command.index("--device") + 1] == "cuda"
        if stage.name in {"train_round0", "refine"}:
            assert stage.command[stage.command.index("--uot-unknown-mass-mode") + 1] == "fixed"
            assert float(
                stage.command[stage.command.index("--uot-unknown-mass") + 1]
            ) == pytest.approx(stage.unknown_mass)
        if stage.name == "refine":
            assert "--save-round-checkpoints" not in stage.command
            assert len(stage.outputs) == 3
        if stage.name == "predict_refined":
            assert stage.validate.keywords["refinement_round"] is None

    with pytest.raises(ValueError, match="seed 17"):
        RUNNER.build_plan(
            tmp_path,
            samples=("4066",),
            seeds=(41,),
            unknown_mass_sensitivity=True,
        )


def test_unknown_mass_manifest_binds_full_recipe_outputs_and_adoption_status(tmp_path):
    for relative in RUNNER.UNKNOWN_MASS_SOURCE_FILES:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    plan = RUNNER.build_plan(
        tmp_path,
        samples=("4066",),
        seeds=(17,),
        unknown_mass_sensitivity=True,
    )
    for stage in plan:
        for output in stage.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes((stage.name + "\n").encode())
    records = [
        {
            "sample": stage.sample,
            "seed": stage.seed,
            "stage": stage.name,
            "unknown_mass": stage.unknown_mass,
            "status": "skipped_valid",
        }
        for stage in plan
    ]

    manifest = RUNNER.build_unknown_mass_manifest(
        tmp_path,
        plan,
        records,
        samples=("4066",),
    )

    assert manifest["schema"] == RUNNER.UNKNOWN_MASS_MANIFEST_SCHEMA
    assert manifest["execution_mode"] == "all_skipped_valid"
    assert manifest["manifest_role"] == "post_execute_output_adoption_and_validation"
    assert manifest["cli_source_binding"]["schema"] == "heir.source_bound_cli.v1"
    assert len(manifest["cli_source_binding"]["cli_source_sha256"]) == 64
    assert manifest["stage_count"] == 25
    assert len(manifest["plan_sha256"]) == 64
    assert len(manifest["validation_recipe_source_identity"]["aggregate_sha256"]) == 64
    for planned, manifested in zip(plan, manifest["stages"]):
        assert manifested["command"] == list(planned.command)
        assert manifested["status"] == "skipped_valid"
        assert [row["path"] for row in manifested["outputs"]] == [
            str(path.resolve()) for path in planned.outputs
        ]
        assert all(len(row["sha256"]) == 64 for row in manifested["outputs"])


def test_full_run_manifest_marks_posthoc_adoption_as_unverified(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "input.npz"
    source.write_bytes(b"input")
    prediction = tmp_path / "prediction.npz"
    telemetry = tmp_path / "telemetry.json"
    prediction.write_bytes(b"prediction")
    telemetry.write_text("{}", encoding="utf-8")
    stage = RUNNER.PlannedStage(
        sample="4066",
        seed=17,
        name="predict_refined",
        command=("heir", "predict"),
        outputs=(prediction, telemetry),
        validate=lambda: None,
        inputs=(("checkpoint", source),),
        output_roles=("prediction", "telemetry"),
    )
    planned = {
        "stage_index": 0,
        "stage_id": "4066/seed17/predict_refined",
        "sample": "4066",
        "seed": 17,
        "stage": "predict_refined",
        "control": None,
        "command": ["heir", "predict"],
        "command_sha256": RUNNER._canonical_sha256(["heir", "predict"]),
        "inputs": [{"role": "checkpoint", "path": str(source.resolve())}],
        "outputs": [
            {"role": "prediction", "path": str(prediction.resolve())},
            {"role": "telemetry", "path": str(telemetry.resolve())},
        ],
        "deterministic_transform_recipe": None,
    }
    plan = {
        "samples": ["4066"],
        "seeds": [17],
        "control_seeds": [],
        "trajectory_seed": 17,
        "controls": [],
        "wrong_donor_target": "4066",
        "wrong_donor_source": "4411",
        "stage_count": 1,
        "stages": [planned],
    }
    compatibility = {
        "analysis_role": "development",
        "negative_control": False,
        "native_scanvi_manifest_sha256": "a" * 64,
        "latent_space_id": "latent",
        "expression_space_id": "expression",
        "seeds": [17],
        "samples": ["4066"],
        "controls_available": [],
        "wrong_donor_pairings": [],
        "wrong_donor_coverage_complete": False,
    }
    monkeypatch.setattr(RUNNER, "full_matrix_plan_payload", lambda stages: plan)
    monkeypatch.setattr(
        RUNNER,
        "_compatibility_cases",
        lambda repository, stages, manifest_directory: (compatibility, []),
    )
    monkeypatch.setattr(
        RUNNER,
        "refinement_run_source_identity",
        lambda repository: {"schema": "heir.source_identity.v1", "aggregate_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        RUNNER,
        "_heir_source_binding",
        lambda repository, **kwargs: {
            "schema": "heir.source_bound_cli.v1",
            "cli_source_sha256": "c" * 64,
        },
    )

    manifest = RUNNER.build_refinement_run_manifest(
        tmp_path,
        (stage,),
        [{"sample": "4066", "seed": 17, "stage": "predict_refined", "status": "skipped_valid"}],
        manifest_path=tmp_path / "reports" / "manifest.json",
    )

    assert manifest["schema"] == RUNNER.REFINEMENT_RUN_MANIFEST_SCHEMA
    assert manifest["manifest_role"] == "posthoc_output_adoption_not_original_execution_proof"
    assert manifest["execution"]["posthoc_adoption_present"] is True
    assert manifest["execution"]["original_execution_source_verified"] is False
    assert manifest["execution"]["execution_provenance_verified"] is False
    assert manifest["stages"][0]["status"] == ("adopted_existing_output_after_current_validation")


def test_shuffle_recipe_hashes_the_deterministic_expected_map(tmp_path):
    histology = tmp_path / "histology.npz"
    np.savez(histology, nucleus_ids=np.asarray(["n0", "n1", "n2", "n3"]))
    stage = RUNNER.PlannedStage(
        sample="4066",
        seed=17,
        name="image_shuffle",
        command=("heir", "predict", "--image-feature-shuffle"),
        outputs=(),
        validate=lambda: None,
        inputs=(("histology", histology),),
        control="image_shuffle",
    )

    first = RUNNER._control_transform_recipe(stage)
    second = RUNNER._control_transform_recipe(stage)

    assert first == second
    assert first["control"] == "image_shuffle"
    assert len(first["expected_transform_map_sha256"]) == 64
    assert len(first["recipe_sha256"]) == 64
    assert first["map_sha256"] == first["expected_transform_map_sha256"]


def test_full_run_manifest_fails_closed_on_legacy_shuffle_telemetry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, telemetry, checkpoint, prototypes, histology, ood = _write_prediction_case(
        tmp_path,
        control="image_shuffle",
    )
    stage = RUNNER.PlannedStage(
        sample="4066",
        seed=17,
        name="image_shuffle",
        command=("source-bound-python", "predict"),
        outputs=(output, telemetry),
        validate=lambda: None,
        inputs=(
            ("checkpoint", checkpoint),
            ("prototype", prototypes),
            ("histology", histology),
            ("ood", ood),
        ),
        output_roles=("prediction", "telemetry"),
        control="image_shuffle",
        prototype_donor_id="4066",
    )
    recipe = RUNNER._control_transform_recipe(stage)
    planned = {
        "stage_index": 0,
        "stage_id": "4066/seed17/image_shuffle",
        "sample": "4066",
        "seed": 17,
        "stage": "image_shuffle",
        "control": "image_shuffle",
        "prototype_donor_id": "4066",
        "command": list(stage.command),
        "command_sha256": RUNNER._canonical_sha256(list(stage.command)),
        "inputs": [],
        "outputs": [],
        "deterministic_transform_recipe": recipe,
    }
    plan = {
        "samples": ["4066"],
        "seeds": [17],
        "control_seeds": [17],
        "trajectory_seed": 17,
        "controls": ["image_shuffle"],
        "wrong_donor_pairings": [],
        "stage_count": 1,
        "stages": [planned],
    }
    compatibility = {
        "analysis_role": "development",
        "negative_control": False,
        "native_scanvi_manifest_sha256": "a" * 64,
        "latent_space_id": "latent",
        "expression_space_id": "expression",
        "seeds": [17],
        "samples": ["4066"],
        "controls_available": ["image_shuffle"],
        "wrong_donor_pairings": [],
        "wrong_donor_coverage_complete": False,
    }
    monkeypatch.setattr(RUNNER, "full_matrix_plan_payload", lambda stages: plan)
    monkeypatch.setattr(
        RUNNER,
        "_compatibility_cases",
        lambda repository, stages, manifest_directory: (compatibility, []),
    )
    monkeypatch.setattr(
        RUNNER,
        "refinement_run_source_identity",
        lambda repository: {"schema": "heir.source_identity.v1", "aggregate_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        RUNNER,
        "_heir_source_binding",
        lambda repository, **kwargs: {
            "schema": "heir.source_bound_cli.v1",
            "cli_source_sha256": "c" * 64,
        },
    )
    record = [{"sample": "4066", "seed": 17, "stage": "image_shuffle", "status": "completed"}]

    manifest = RUNNER.build_refinement_run_manifest(
        tmp_path,
        (stage,),
        record,
        manifest_path=tmp_path / "reports" / "manifest.json",
    )
    assert manifest["execution"]["execution_transform_hash_verified"] is True
    assert manifest["stages"][0]["telemetry_transform"] == recipe

    payload = json.loads(telemetry.read_text(encoding="utf-8"))
    payload["negative_control"].pop("transform")
    telemetry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or stale transform hashes"):
        RUNNER.build_refinement_run_manifest(
            tmp_path,
            (stage,),
            record,
            manifest_path=tmp_path / "reports" / "manifest.json",
        )
