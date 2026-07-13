import hashlib
import importlib.util
import json
import os
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


def _record(path):
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _execution_record(stage, status="skipped_valid"):
    record = {
        "sample": stage.sample,
        "seed": stage.seed,
        "stage": stage.name,
        "status": status,
        "validated_inputs": RUNNER._absolute_artifact_rows(stage.inputs),
        "validated_outputs": RUNNER._absolute_artifact_rows(RUNNER._stage_output_artifacts(stage)),
    }
    if stage.unknown_mass is not None:
        record["unknown_mass"] = stage.unknown_mass
    return record


def test_source_hash_rechecks_content_when_size_and_mtime_are_unchanged(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"first")
    original_stat = source.stat()
    original_digest = RUNNER._sha256(source)

    source.write_bytes(b"later")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert RUNNER._sha256(source) != original_digest


def test_runtime_environment_identity_is_reobserved_each_time(monkeypatch):
    calls = 0

    def distributions():
        nonlocal calls
        calls += 1
        return ()

    monkeypatch.setattr(RUNNER.importlib.metadata, "distributions", distributions)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    RUNNER._runtime_environment_identity()
    RUNNER._runtime_environment_identity()

    assert calls == 2


def test_stage_output_capture_brackets_validation(tmp_path):
    output = tmp_path / "prediction.npz"
    output.write_bytes(b"before")
    stage = RUNNER.PlannedStage(
        sample="4066",
        seed=17,
        name="predict_refined",
        command=(),
        outputs=(output,),
        validate=lambda: output.write_bytes(b"during"),
        output_roles=("prediction",),
    )

    with pytest.raises(RuntimeError, match="validator mutated output artifacts"):
        RUNNER._validate_and_capture_stage_outputs(stage)


def test_final_artifact_recheck_deduplicates_canonical_paths(tmp_path, monkeypatch):
    source = tmp_path / "shared.npz"
    source.write_bytes(b"shared")
    artifacts = (("input", source),)
    captured = RUNNER._absolute_artifact_rows(artifacts)
    observed = {}
    calls = 0
    original = RUNNER._sha256

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(RUNNER, "_sha256", counted)
    for stage_id in ("first", "second"):
        RUNNER._manifest_rows_from_stage_capture(
            artifacts,
            captured,
            manifest_directory=None,
            stage_id=stage_id,
            boundary="final",
            observed_sha256=observed,
        )

    assert calls == 1


def _write_true_loo_fold(tmp_path, target):
    root = tmp_path / ("fold_" + target)
    root.mkdir(parents=True)
    training = tuple(sample for sample in RUNNER.SAMPLES if sample != target)
    artifacts = {}
    for name in (
        "batch_train",
        "batch_validation",
        "prototypes",
        "residual_geometry",
        "decoder",
    ):
        path = root / (name + (".pt" if name == "decoder" else ".npz"))
        path.write_bytes((target + ":" + name).encode())
        artifacts[name] = path
    latent_space_id = "sha256:" + hashlib.sha256(target.encode()).hexdigest()
    label_ontology = ("B", "T")
    native_path = root / "native_manifest.json"
    native = {
        "schema": "heir.snpatho_scanvi_r2_manifest.v1",
        "status": "native_scanvi_true_leave_one_donor_out",
        "expression_space_id": "log1p-cpm-10000-v1",
        "latent_space_id": latent_space_id,
        "molecular_producer": _record(ROOT / "scripts/train_snpatho_scanvi.py"),
        "training_partition": {
            "mode": "leave_one_donor_out",
            "held_out_sample": target,
            "backbone_training_donors": list(training),
            "decoder_training_donors": list(training),
            "label_ontology": list(label_ontology),
            "label_ontology_sha256": RUNNER._ordered_string_sha256(label_ontology),
            "held_out_mapping": {
                "held_out_annotation_used_for_label_mapping": False,
                "label_mapping_method": "frozen_training_donor_SCANVI_classifier",
                "label_training_donors": list(training),
            },
        },
        "distilled_decoder": {
            "external_path": str(artifacts["decoder"].resolve()),
            "sha256": _sha256(artifacts["decoder"]),
            "contract": {"training_donors": list(training)},
        },
        "specimens": {
            target: {
                "rare_complete_prototypes": str(artifacts["prototypes"].resolve()),
                "rare_complete_prototypes_sha256": _sha256(artifacts["prototypes"]),
                "residual_geometry": str(artifacts["residual_geometry"].resolve()),
                "residual_geometry_sha256": _sha256(artifacts["residual_geometry"]),
            }
        },
    }
    native_path.write_text(json.dumps(native), encoding="utf-8")
    preparation_path = root / "preparation_manifest.json"
    preparation = {
        "schema": "heir.snpatho_refinement_input_preparation.v1",
        "status": "complete",
        "molecular_generation": "r2",
        "latent_space_id": latent_space_id,
        "producer": _record(ROOT / "scripts/prepare_snpatho_refinement_inputs.py"),
        "native_manifest": _record(native_path),
        "outputs": {
            target: {
                "batch_train": _record(artifacts["batch_train"]),
                "batch_validation": _record(artifacts["batch_validation"]),
                "prototypes": _record(artifacts["prototypes"]),
                "residual_geometry": _record(artifacts["residual_geometry"]),
            }
        },
    }
    preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
    return preparation_path, artifacts


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


def test_stage_runner_guards_source_immediately_before_and_after_subprocess(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "output.npz"
    phases = []

    def fake_run(command, *, check, cwd):
        assert command == ("source-bound-command",)
        assert check is True
        assert cwd == tmp_path
        output.write_bytes(b"complete")

    def guard(phase):
        phases.append(phase)
        if phase == "immediately_after_stage_subprocess":
            raise RuntimeError("source changed during stage")

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="source changed during stage"):
        RUNNER._run(
            ("source-bound-command",),
            (output,),
            True,
            validator=lambda: None,
            repository=tmp_path,
            source_guard=guard,
        )

    assert phases == [
        "immediately_before_stage_subprocess",
        "immediately_after_stage_subprocess",
    ]


def test_stage_runner_guards_existing_output_adoption_transition(tmp_path):
    output = tmp_path / "output.npz"
    output.write_bytes(b"existing")
    phases = []

    status = RUNNER._run(
        ("not-executed",),
        (output,),
        True,
        validator=lambda: None,
        repository=tmp_path,
        source_guard=phases.append,
    )

    assert status == "skipped_valid"
    assert phases == [
        "before_existing_output_adoption",
        "after_existing_output_adoption_validation",
    ]


def test_manifest_output_rejects_symlink_and_hardlink_aliases_of_all_protected_files(
    tmp_path,
):
    source = tmp_path / "runner_dependency.py"
    interpreter = tmp_path / "python"
    module_entrypoint = tmp_path / "__main__.py"
    cli_source = tmp_path / "cli.py"
    stage_input = tmp_path / "batch.npz"
    stage_output = tmp_path / "checkpoint.pt"
    for path in (source, interpreter, module_entrypoint, cli_source, stage_input, stage_output):
        path.write_bytes(path.name.encode())
    source_identity = {"files": [{"path": str(source)}]}
    cli_source_binding = {
        "python_executable": str(interpreter),
        "module_entrypoint": str(module_entrypoint),
        "cli_source": str(cli_source),
    }
    stage = RUNNER.PlannedStage(
        sample="4066",
        seed=17,
        name="train_round0",
        command=(),
        outputs=(stage_output,),
        validate=lambda: None,
        inputs=(("train_batch", stage_input),),
    )

    symlink_manifest = tmp_path / "source-symlink.json"
    symlink_manifest.symlink_to(SCRIPT)
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        RUNNER._reject_manifest_output_collisions(
            ROOT,
            symlink_manifest,
            (stage,),
            source_identity=source_identity,
            cli_source_binding=cli_source_binding,
        )

    for index, protected in enumerate(
        (source, interpreter, module_entrypoint, cli_source, stage_input, stage_output)
    ):
        hardlink_manifest = tmp_path / ("hardlink-%d.json" % index)
        os.link(protected, hardlink_manifest)
        with pytest.raises(ValueError, match="output would overwrite a bound input"):
            RUNNER._reject_manifest_output_collisions(
                ROOT,
                hardlink_manifest,
                (stage,),
                source_identity=source_identity,
                cli_source_binding=cli_source_binding,
            )


def test_main_rejects_manifest_alias_before_any_stage_execution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "runner.py"
    interpreter = tmp_path / "python"
    module_entrypoint = tmp_path / "__main__.py"
    cli_source = tmp_path / "cli.py"
    stage_input = tmp_path / "batch.npz"
    for path in (source, interpreter, module_entrypoint, cli_source, stage_input):
        path.write_bytes(path.name.encode())
    manifest = tmp_path / "manifest.json"
    os.link(stage_input, manifest)
    source_identity = {"files": [{"path": str(source)}]}
    cli_binding = {
        "python_executable": str(interpreter),
        "module_entrypoint": str(module_entrypoint),
        "cli_source": str(cli_source),
    }
    stage = RUNNER.PlannedStage(
        sample="4066",
        seed=17,
        name="train_round0",
        command=("must-not-run",),
        outputs=(tmp_path / "checkpoint.pt",),
        validate=lambda: None,
        inputs=(("train_batch", stage_input),),
    )
    monkeypatch.setattr(
        RUNNER, "refinement_run_source_identity", lambda repository: source_identity
    )
    monkeypatch.setattr(RUNNER, "_heir_source_binding", lambda repository, **kwargs: cli_binding)
    monkeypatch.setattr(RUNNER, "_assert_execution_source_unchanged", lambda *args, **kwargs: None)
    monkeypatch.setattr(RUNNER, "build_plan", lambda *args, **kwargs: (stage,))
    monkeypatch.setattr(
        RUNNER,
        "_run",
        lambda *args, **kwargs: pytest.fail(
            "stage execution was reached before collision preflight"
        ),
    )

    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        RUNNER.main(
            [
                "--execute",
                "--controls",
                "--manifest-output",
                str(manifest),
            ]
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
                "direct_training_donors": ["4066"],
                "initialization_validation_status": "uninitialized_negative_control",
                "molecular_e_step_mode": "live_student_negative_control",
                "excluded_from_primary_claims": True,
                "best_epoch": 3,
                "best_validation_loss": 1.25,
                "uot_unknown_mass": 0.05,
                "uot_unknown_mass_mode": "fixed",
                "rna_vae_sha256": _sha256(decoder),
                "residual_geometry_sha256": _sha256(geometry),
                "training_batches": [dict(train_identity)],
                "validation_batches": [dict(validation_identity)],
                "training_batch_artifacts": [
                    {"path": str(train_batch), "sha256": _sha256(train_batch)}
                ],
                "validation_batch_artifacts": [
                    {
                        "path": str(validation_batch),
                        "sha256": _sha256(validation_batch),
                    }
                ],
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
        "expected_unknown_mass": 0.05,
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


@pytest.mark.parametrize(
    "metadata",
    (
        {},
        {"uot_unknown_mass": 0.05},
        {"uot_unknown_mass": 0.05, "uot_unknown_mass_mode": "targets_or_fixed"},
        {"uot_unknown_mass": 0.20, "uot_unknown_mass_mode": "fixed"},
    ),
)
def test_runner_rejects_legacy_or_mismatched_unknown_mass_with_clean_root_guidance(metadata):
    with pytest.raises(ValueError, match="clean output root"):
        RUNNER._validate_fixed_unknown_mass(
            metadata,
            0.05,
            label="round-zero",
        )


def test_existing_legacy_stage_is_not_silently_adopted(tmp_path):
    outputs = (tmp_path / "heir.pt", tmp_path / "history.json")
    for output in outputs:
        output.write_bytes(b"legacy")

    with pytest.raises(RuntimeError, match="existing stage outputs are invalid.*clean output root"):
        RUNNER._run(
            ("not-executed",),
            outputs,
            False,
            validator=lambda: RUNNER._validate_fixed_unknown_mass(
                {},
                0.05,
                label="round-zero",
            ),
            repository=tmp_path,
        )


def test_control_plan_covers_only_the_prespecified_ablation_seeds(tmp_path):
    plan = RUNNER.build_plan(
        tmp_path,
        samples=RUNNER.SAMPLES,
        seeds=RUNNER.SEEDS,
        controls=True,
    )
    assert len(plan) == 147
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
    for stage in plan:
        if stage.name in {"train_round0", "refine"}:
            assert stage.validate.keywords["expected_unknown_mass"] == pytest.approx(0.05)
            assert stage.command[stage.command.index("--uot-unknown-mass-mode") + 1] == "fixed"
            assert float(stage.command[stage.command.index("--uot-unknown-mass") + 1]) == 0.05
    wrong_donor = [stage for stage in plan if stage.control == "wrong_prototype_bank"]
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
        assert all(stage.name == "wrong_prototype_bank_" + source for stage in stages)

    round0_off = [stage for stage in plan if stage.name == "round0_prototype_only"]
    refined_off = [stage for stage in plan if stage.name == "refined_prototype_only"]
    assert len(round0_off) == len(refined_off) == 9
    assert all(stage.validate.keywords["refinement_round"] == 0 for stage in round0_off)
    assert all(stage.validate.keywords["refinement_round"] == 4 for stage in refined_off)
    assert all("--prototype-only" in stage.command for stage in (*round0_off, *refined_off))


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
            assert stage.validate.keywords["expected_unknown_mass"] == pytest.approx(
                stage.unknown_mass
            )
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


def test_clean_unknown_mass_plan_separates_output_root_from_v2_molecular_inputs(tmp_path):
    output_root = tmp_path / "fresh-outputs"
    plan = RUNNER.build_plan(
        ROOT,
        artifact_root=output_root,
        samples=("4066",),
        seeds=(17,),
        unknown_mass_sensitivity=True,
    )

    assert all(
        path.is_relative_to(output_root.resolve()) for stage in plan for path in stage.outputs
    )
    train_stage = next(stage for stage in plan if stage.name == "train_round0")
    inputs = dict(train_stage.inputs)
    assert inputs["residual_geometry"].name == "residual_geometry_rare_complete_v2.npz"
    assert not inputs["residual_geometry"].is_relative_to(output_root.resolve())

    existing = train_stage.outputs[0]
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"legacy")
    with pytest.raises(RuntimeError, match="Choose a clean --artifact-root"):
        RUNNER.main(
            [
                "--sample",
                "4066",
                "--seed",
                "17",
                "--unknown-mass-sensitivity",
                "--execute",
                "--prohibit-adoption",
                "--artifact-root",
                str(output_root),
            ]
        )


def test_r2_plan_routes_all_molecular_inputs_and_decoder_to_r2(tmp_path):
    plan = RUNNER.build_plan(
        ROOT,
        artifact_root=tmp_path / "outputs",
        samples=("4066",),
        seeds=(17,),
        molecular_generation="r2",
    )
    train = next(stage for stage in plan if stage.name == "train_round0")
    inputs = dict(train.inputs)
    assert "/r2_scanvi/4066/" in str(inputs["train_batch"])
    assert "/r2_scanvi/4066/" in str(inputs["residual_geometry"])
    assert str(inputs["rna_decoder"]).endswith(
        "HEIR_assets/pretrained/snpatho_scanvi_r2_preserve_biology_v1_decoder.pt"
    )
    views = next(stage for stage in plan if stage.name == "build_views")
    assert views.command[1:3] == ("-I", "-c")
    assert str((ROOT / "src").resolve()) in views.command[3]


def test_true_loo_three_fold_plan_uses_target_specific_hash_bound_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        RUNNER,
        "_expected_shuffle_transform",
        lambda control, seed, histology: {
            "schema": "synthetic.shuffle.recipe",
            "control": control,
            "seed": seed,
            "histology": str(histology),
        },
    )
    specifications = []
    expected = {}
    for sample in RUNNER.SAMPLES:
        manifest, artifacts = _write_true_loo_fold(tmp_path, sample)
        specifications.append("%s=%s" % (sample, manifest))
        expected[sample] = artifacts
    folds = RUNNER.load_true_loo_molecular_folds(
        ROOT,
        specifications,
        required_samples=RUNNER.SAMPLES,
    )

    plan = RUNNER.build_plan(
        ROOT,
        artifact_root=tmp_path / "outputs",
        samples=RUNNER.SAMPLES,
        seeds=(17,),
        controls=True,
        molecular_generation="r2",
        molecular_folds=folds,
    )
    train_stages = [stage for stage in plan if stage.name == "train_round0"]
    assert {stage.sample for stage in train_stages} == set(RUNNER.SAMPLES)
    assert len({fold.latent_space_id for fold in folds.values()}) == len(RUNNER.SAMPLES)
    assert len({fold.decoder_sha256 for fold in folds.values()}) == len(RUNNER.SAMPLES)
    for stage in train_stages:
        inputs = dict(stage.inputs)
        artifacts = expected[stage.sample]
        assert inputs["train_batch"] == artifacts["batch_train"].resolve()
        assert inputs["validation_batch"] == artifacts["batch_validation"].resolve()
        assert inputs["residual_geometry"] == artifacts["residual_geometry"].resolve()
        assert inputs["rna_decoder"] == artifacts["decoder"].resolve()
        assert (
            inputs["molecular_fold_preparation_manifest"]
            == folds[stage.sample].preparation_manifest
        )
        assert inputs["molecular_fold_native_manifest"] == folds[stage.sample].native_manifest
        assert stage.command[stage.command.index("--rna-vae-checkpoint") + 1] == str(
            artifacts["decoder"].resolve()
        )
    assert not [stage for stage in plan if stage.control == "wrong_prototype_bank"]
    assert {stage.name for stage in plan if stage.control is not None} == set(
        RUNNER.PREDICTION_CONTROLS
    )
    full_plan = RUNNER.build_plan(
        ROOT,
        artifact_root=tmp_path / "outputs",
        samples=RUNNER.SAMPLES,
        seeds=RUNNER.SEEDS,
        controls=True,
        molecular_generation="r2",
        molecular_folds=folds,
    )
    full_payload = RUNNER.full_matrix_plan_payload(
        full_plan,
        molecular_generation="r2",
        molecular_folds=folds,
    )
    assert full_payload["stage_count"] == 129
    assert full_payload["wrong_prototype_bank_pairings"] == []
    assert full_payload["wrong_prototype_bank_coverage_complete"] is False

    prediction_stages = []
    for seed in RUNNER.SEEDS:
        for sample in RUNNER.SAMPLES:
            prediction = tmp_path / "predictions" / sample / ("seed_%d.npz" % seed)
            prediction.parent.mkdir(parents=True, exist_ok=True)
            prediction.write_bytes((sample + ":" + str(seed)).encode())
            prediction_stages.append(
                RUNNER.PlannedStage(
                    sample=sample,
                    seed=seed,
                    name="predict_refined",
                    command=(),
                    outputs=(prediction,),
                    validate=lambda: None,
                )
            )
    compatibility, cases = RUNNER._compatibility_cases(
        ROOT,
        prediction_stages,
        manifest_directory=tmp_path,
        molecular_generation="r2",
        molecular_folds=folds,
    )
    assert compatibility["negative_control"] is True
    assert compatibility["schema"] == RUNNER.TRUE_LOO_FIVE_SEED_MANIFEST_SCHEMA
    assert compatibility["native_scanvi_manifest_sha256"] is None
    assert len(compatibility["native_scanvi_fold_bundle_sha256"]) == 64
    assert compatibility["latent_space_id"] is None
    assert set(compatibility["latent_space_id_by_sample"]) == set(RUNNER.SAMPLES)
    assert set(compatibility["native_scanvi_fold_manifests"]) == set(RUNNER.SAMPLES)
    assert "wrong_prototype_bank" not in compatibility["controls_available"]
    assert compatibility["wrong_prototype_bank_pairings"] == []
    assert compatibility["wrong_prototype_bank_coverage_complete"] is False
    assert compatibility["wrong_prototype_bank_unavailable_reason"]
    assert len(cases) == len(RUNNER.SEEDS) * len(RUNNER.SAMPLES)
    assert all(
        row["molecular_fold_decoder_sha256"] == folds[row["section_id"]].decoder_sha256
        for row in cases
    )


def test_true_loo_fold_loader_rejects_target_in_decoder_scope_and_missing_fold(tmp_path):
    manifest, _ = _write_true_loo_fold(tmp_path, "4066")
    preparation = json.loads(manifest.read_text(encoding="utf-8"))
    native_path = Path(preparation["native_manifest"]["path"])
    native = json.loads(native_path.read_text(encoding="utf-8"))
    native["distilled_decoder"]["contract"]["training_donors"] = list(RUNNER.SAMPLES)
    native_path.write_text(json.dumps(native), encoding="utf-8")
    preparation["native_manifest"] = _record(native_path)
    manifest.write_text(json.dumps(preparation), encoding="utf-8")

    with pytest.raises(ValueError, match="decoder contract does not exclude"):
        RUNNER.load_true_loo_molecular_folds(
            ROOT,
            ["4066=%s" % manifest],
            required_samples=("4066",),
        )

    valid_manifest, _ = _write_true_loo_fold(tmp_path / "valid", "4066")
    with pytest.raises(ValueError, match="exactly cover requested samples"):
        RUNNER.load_true_loo_molecular_folds(
            ROOT,
            ["4066=%s" % valid_manifest],
            required_samples=("4066", "4399"),
        )


def test_true_loo_fold_loader_rejects_reused_decoder_identity(tmp_path):
    first_manifest, first_artifacts = _write_true_loo_fold(tmp_path / "first", "4066")
    second_manifest, _ = _write_true_loo_fold(tmp_path / "second", "4399")
    second_preparation = json.loads(second_manifest.read_text(encoding="utf-8"))
    second_native_path = Path(second_preparation["native_manifest"]["path"])
    second_native = json.loads(second_native_path.read_text(encoding="utf-8"))
    second_native["distilled_decoder"]["external_path"] = str(first_artifacts["decoder"].resolve())
    second_native["distilled_decoder"]["sha256"] = _sha256(first_artifacts["decoder"])
    second_native_path.write_text(json.dumps(second_native), encoding="utf-8")
    second_preparation["native_manifest"] = _record(second_native_path)
    second_manifest.write_text(json.dumps(second_preparation), encoding="utf-8")

    with pytest.raises(ValueError, match="reuse a decoder identity"):
        RUNNER.load_true_loo_molecular_folds(
            ROOT,
            ["4066=%s" % first_manifest, "4399=%s" % second_manifest],
            required_samples=("4066", "4399"),
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
    for stage in plan:
        for _, stage_input in stage.inputs:
            if not stage_input.exists():
                stage_input.parent.mkdir(parents=True, exist_ok=True)
                stage_input.write_bytes(b"synthetic input")
    records = [_execution_record(stage) for stage in plan]

    manifest = RUNNER.build_unknown_mass_manifest(
        tmp_path,
        plan,
        records,
        samples=("4066",),
    )

    assert manifest["schema"] == RUNNER.UNKNOWN_MASS_MANIFEST_SCHEMA
    assert manifest["execution_mode"] == "all_skipped_valid"
    assert manifest["manifest_role"] == "post_execute_output_adoption_and_validation"
    assert manifest["stage_time_artifact_identities_complete"] is True
    assert manifest["cli_source_binding"]["schema"] == "heir.source_bound_cli.v1"
    assert len(manifest["cli_source_binding"]["cli_source_sha256"]) == 64
    assert manifest["stage_count"] == 25
    assert len(manifest["plan_sha256"]) == 64
    assert len(manifest["validation_recipe_source_identity"]["aggregate_sha256"]) == 64
    for planned, manifested in zip(plan, manifest["stages"]):
        assert manifested["command"] == list(planned.command)
        assert manifested["status"] == "skipped_valid"
        assert manifested["artifact_identity_capture"] == (RUNNER.STAGE_ARTIFACT_IDENTITY_CAPTURE)
        assert [row["role"] for row in manifested["inputs"]] == [role for role, _ in planned.inputs]
        assert [row["path"] for row in manifested["outputs"]] == [
            str(path.resolve()) for path in planned.outputs
        ]
        assert all(len(row["sha256"]) == 64 for row in manifested["outputs"])

    first_input = plan[0].inputs[0][1]
    first_input.write_bytes(b"mutated after its stage was validated")
    with pytest.raises(RuntimeError, match="changed after stage validation"):
        RUNNER.build_unknown_mass_manifest(
            tmp_path,
            plan,
            records,
            samples=("4066",),
        )


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
    monkeypatch.setattr(RUNNER, "full_matrix_plan_payload", lambda stages, **kwargs: plan)
    monkeypatch.setattr(
        RUNNER,
        "_compatibility_cases",
        lambda repository, stages, manifest_directory, **kwargs: (compatibility, []),
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
        [_execution_record(stage)],
        manifest_path=tmp_path / "reports" / "manifest.json",
    )

    assert manifest["schema"] == RUNNER.REFINEMENT_RUN_MANIFEST_SCHEMA
    assert manifest["manifest_role"] == "posthoc_output_adoption_not_original_execution_proof"
    assert manifest["execution"]["posthoc_adoption_present"] is True
    assert manifest["execution"]["original_execution_source_verified"] is False
    assert manifest["execution"]["execution_provenance_verified"] is False
    assert manifest["execution"]["stage_time_artifact_identities_complete"] is True
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
    monkeypatch.setattr(RUNNER, "full_matrix_plan_payload", lambda stages, **kwargs: plan)
    monkeypatch.setattr(
        RUNNER,
        "_compatibility_cases",
        lambda repository, stages, manifest_directory, **kwargs: (compatibility, []),
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
    record = [_execution_record(stage, status="completed")]
    execution_source_identity = {
        "schema": "heir.source_identity.v1",
        "aggregate_sha256": "b" * 64,
    }
    execution_cli_source_binding = {
        "schema": "heir.source_bound_cli.v1",
        "cli_source_sha256": "c" * 64,
    }

    final_tree_only = RUNNER.build_refinement_run_manifest(
        tmp_path,
        (stage,),
        record,
        manifest_path=tmp_path / "reports" / "manifest.json",
    )
    assert final_tree_only["execution_source_identity"] is None
    assert final_tree_only["execution"]["original_execution_source_verified"] is False
    assert final_tree_only["manifest_role"] == (
        "current_invocation_outputs_source_identity_unverified"
    )

    manifest = RUNNER.build_refinement_run_manifest(
        tmp_path,
        (stage,),
        record,
        manifest_path=tmp_path / "reports" / "manifest.json",
        execution_source_identity=execution_source_identity,
        execution_cli_source_binding=execution_cli_source_binding,
    )
    assert manifest["execution"]["execution_transform_hash_verified"] is True
    assert manifest["execution"]["execution_source_identity_captured_before_stage_1"] is True
    assert manifest["execution"]["execution_source_identity_unchanged"] is True
    assert manifest["execution"]["execution_cli_source_binding_unchanged"] is True
    assert manifest["execution"]["original_execution_source_verified"] is True
    assert manifest["execution_source_identity"] == execution_source_identity
    assert manifest["stages"][0]["telemetry_transform"] == recipe

    payload = json.loads(telemetry.read_text(encoding="utf-8"))
    payload["negative_control"].pop("transform")
    telemetry.write_text(json.dumps(payload), encoding="utf-8")
    record = [_execution_record(stage, status="completed")]
    with pytest.raises(ValueError, match="missing or stale transform hashes"):
        RUNNER.build_refinement_run_manifest(
            tmp_path,
            (stage,),
            record,
            manifest_path=tmp_path / "reports" / "manifest.json",
            execution_source_identity=execution_source_identity,
            execution_cli_source_binding=execution_cli_source_binding,
        )
