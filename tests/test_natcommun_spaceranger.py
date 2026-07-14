from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_natcommun_spaceranger.py"
SPEC = importlib.util.spec_from_file_location("run_natcommun_spaceranger", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_protocol_has_16_sections_14_donors_and_13_primary_donors():
    protocol = runner._load_protocol(runner.DEFAULT_PROTOCOL)
    sections = protocol["sections"]
    assert len(sections) == 16
    assert len({row["donor"] for row in sections}) == 14
    assert set(protocol["primary_donors"]) == {
        "B1", "B3", "B4", "L1", "L2", "L3", "L4",
        "D1", "D2", "D3", "D4", "D5", "D6",
    }
    b2 = [row for row in sections if row["donor"] == "B2"]
    assert len(b2) == 1
    assert b2[0]["primary_eligible"] is False


def test_fastq_discovery_requires_all_dual_index_reads(tmp_path):
    sample = "V4_B1_2_OPHI"
    first = tmp_path / "run1"
    second = tmp_path / "run2"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        for read in ("R1", "R2", "I1", "I2"):
            (directory / f"{sample}_S1_L001_{read}_001.fastq.gz").touch()
    assert runner._fastq_directories(tmp_path, sample) == (
        first.resolve(), second.resolve()
    )
    (second / f"{sample}_S1_L001_I2_001.fastq.gz").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        runner._fastq_directories(tmp_path, sample)


def test_command_pins_registration_and_resource_limits_without_hd_segmentation(
    tmp_path,
):
    protocol = runner._load_protocol(runner.DEFAULT_PROTOCOL)
    row = protocol["sections"][0]
    processed = tmp_path / "arrayexpress/E-MTAB-14560/processed_data"
    raw = tmp_path / "arrayexpress/E-MTAB-14560/ENA_submitted/run"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    for read in ("R1", "R2", "I1", "I2"):
        (raw / f"{row['fastq_sample']}_S1_L001_{read}_001.fastq.gz").touch()
    command, _directories = runner._command(
        row,
        data_root=tmp_path,
        spaceranger=Path("/bin/true"),
        reference=tmp_path / "reference",
        probe_set=tmp_path / "probe.csv",
        output_root=tmp_path / "out",
        localcores=4,
        localmem=24,
        dry=True,
    )
    assert "--image" in command and "--cytaimage" in command
    assert not any(value.startswith("--nucleus-segmentation") for value in command)
    assert "--create-bam=false" in command
    assert command[command.index("--localcores") + 1] == "4"
    assert command[command.index("--localmem") + 1] == "24"
    assert command[command.index("--localvmem") + 1] == "64"
    assert command[-1] == "--dry"


def test_cli_rejects_unsafe_combined_resource_request():
    with pytest.raises(SystemExit):
        runner.parse_args(["--max-workers", "2", "--localcores", "8"])


def test_complete_output_requires_molecule_info_for_exact_umi_halves(tmp_path):
    output = tmp_path / "section"
    spatial = output / "outs" / "spatial"
    spatial.mkdir(parents=True)
    for path in (
        output / "outs" / "filtered_feature_bc_matrix.h5",
        spatial / "tissue_positions.csv",
        spatial / "scalefactors_json.json",
    ):
        path.write_bytes(b"present")
    assert not runner._output_complete(output)
    (output / "outs" / "molecule_info.h5").write_bytes(b"present")
    assert runner._output_complete(output)
