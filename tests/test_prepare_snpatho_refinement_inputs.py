import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_snpatho_refinement_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_snpatho_refinement_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def _samples(tmp_path):
    return {
        sample: PREPARE.SamplePaths(
            sample=sample,
            source=tmp_path / "source" / sample,
            scanvi=tmp_path / "scanvi" / sample,
            scanvi_input=tmp_path / "ffpe" / sample,
        )
        for sample in PREPARE.SAMPLES
    }


def test_command_plan_freezes_rare_types_geometry_and_batch_identity(tmp_path):
    stages = PREPARE.build_stages(samples=_samples(tmp_path), heir_command="heir")

    assert [(stage.sample, stage.name) for stage in stages] == [
        (sample, name)
        for sample in PREPARE.SAMPLES
        for name in ("prototypes", "residual_geometry", "batch_train", "batch_validation")
    ]
    prototype = stages[0].command(stages[0].output)
    geometry = stages[1].command(stages[1].output)
    train = stages[2].command(stages[2].output)
    assert "--include-rare-types" in prototype
    assert prototype[prototype.index("--seed") + 1] == "17"
    assert geometry[geometry.index("--rank") + 1] == "4"
    assert train[train.index("--analysis-role") + 1] == "development_retrospective"
    assert train[train.index("--block-id") + 1] == "4066_FFPE"
    assert train[train.index("--ood-artifact") + 1].endswith("4066/ood_target_calibrated.npz")


def test_dry_run_refuses_untracked_output_without_explicit_adoption(tmp_path):
    source = tmp_path / "input.npz"
    output = tmp_path / "output.npz"
    source.write_bytes(b"input")
    output.write_bytes(b"untracked")
    stage = PREPARE.Stage(
        sample="4066",
        name="synthetic",
        inputs=(("source", source),),
        output=output,
        command=lambda destination: ("never-executed", str(destination)),
        validate=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="untracked stage output"):
        PREPARE._run_stage(
            stage,
            repository=tmp_path,
            receipt_root=tmp_path / "receipts",
            execute=False,
            adopt_existing=False,
        )

    assert not (tmp_path / "receipts").exists()
