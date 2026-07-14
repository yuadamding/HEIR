from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_natcommun_spaceranger.py"
SPEC = importlib.util.spec_from_file_location("run_natcommun_spaceranger", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _run_args(tmp_path, monkeypatch, *, sections: str):
    protocol = runner._load_protocol(runner.DEFAULT_PROTOCOL)
    selected = {section for section in sections.split(",") if section} or {
        str(row["section"]) for row in protocol["sections"]
    }
    data_root = tmp_path / "data"
    processed = data_root / "arrayexpress/E-MTAB-14560/processed_data"
    processed.mkdir(parents=True)
    for row in protocol["sections"]:
        if str(row["section"]) not in selected:
            continue
        for key in ("h_and_e", "cytassist"):
            (processed / str(row[key])).write_bytes(b"image")
    spaceranger = tmp_path / "spaceranger"
    spaceranger.write_text("#!/bin/sh\n", encoding="utf-8")
    spaceranger.chmod(0o755)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "reference.json").write_text("{}\n", encoding="utf-8")
    probe_set = tmp_path / "probe.csv"
    probe_set.write_text("probe\n", encoding="utf-8")
    output_root = tmp_path / "output"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="spaceranger 4.1.0\n"),
    )
    monkeypatch.setattr(
        runner,
        "_fastq_directories",
        lambda *_args, **_kwargs: (tmp_path.resolve(),),
    )
    args = SimpleNamespace(
        protocol=runner.DEFAULT_PROTOCOL,
        data_root=data_root,
        spaceranger=spaceranger,
        reference=reference,
        probe_set=probe_set,
        output_root=output_root,
        sections=sections,
        max_workers=1,
        localcores=2,
        localmem=16,
        dry=False,
    )
    return args, protocol


def _receipt_for_args(args, protocol, sections):
    identity = runner._receipt_identity(
        protocol_path=args.protocol.resolve(),
        spaceranger=args.spaceranger.resolve(),
        spaceranger_version="spaceranger 4.1.0",
        reference=args.reference.resolve(),
        probe_set=args.probe_set.resolve(),
        output_root=args.output_root.resolve(),
    )
    return {
        **identity,
        "analysis_scope": protocol["analysis_scope"],
        "status": "failed",
        "sections": sections,
    }


def test_protocol_has_16_sections_14_donors_and_13_primary_donors():
    protocol = runner._load_protocol(runner.DEFAULT_PROTOCOL)
    sections = protocol["sections"]
    assert len(sections) == 16
    assert len({row["donor"] for row in sections}) == 14
    assert set(protocol["primary_donors"]) == {
        "B1",
        "B3",
        "B4",
        "L1",
        "L2",
        "L3",
        "L4",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
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
    assert runner._fastq_directories(tmp_path, sample) == (first.resolve(), second.resolve())
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


def test_partial_retry_preserves_unrequested_records_and_restores_full_completion(
    tmp_path, monkeypatch
):
    args, protocol = _run_args(tmp_path, monkeypatch, sections="B3_2")
    records = {
        str(row["section"]): {
            "section": str(row["section"]),
            "donor": str(row["donor"]),
            "status": "failed" if row["section"] == "B3_2" else "complete",
            "original_marker": str(row["section"]),
        }
        for row in protocol["sections"]
    }
    preserved = dict(records["B1_2"])
    receipt = _receipt_for_args(args, protocol, records)
    args.output_root.mkdir()
    runner._write_json(args.output_root / "run_status.json", receipt)

    def successful_retry(row, **_kwargs):
        return {
            "section": str(row["section"]),
            "donor": str(row["donor"]),
            "status": "complete_existing",
            "retry_marker": True,
        }

    monkeypatch.setattr(runner, "_run_section", successful_retry)
    assert runner.run(args) == 0

    merged = json.loads((args.output_root / "run_status.json").read_text())
    assert set(merged["sections"]) == {str(row["section"]) for row in protocol["sections"]}
    assert merged["sections"]["B1_2"] == preserved
    assert merged["sections"]["B3_2"]["status"] == "complete_existing"
    assert merged["sections"]["B3_2"]["retry_marker"] is True
    assert merged["status"] == "complete"
    assert merged["failed_sections"] == 0
    assert "active_partial_sections" not in merged


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "protocol",
        "protocol_sha256",
        "spaceranger",
        "spaceranger_version",
        "reference",
        "reference_metadata_sha256",
        "probe_set",
        "probe_set_sha256",
        "pipestance_root",
    ],
)
def test_partial_retry_fails_closed_on_every_receipt_identity_mismatch(
    tmp_path, monkeypatch, field
):
    args, protocol = _run_args(tmp_path, monkeypatch, sections="B3_2")
    receipt = _receipt_for_args(args, protocol, {})
    receipt[field] = "stale-or-different"
    args.output_root.mkdir()
    runner._write_json(args.output_root / "run_status.json", receipt)

    with pytest.raises(ValueError, match=rf"identity mismatch:.*{field}"):
        runner.run(args)


def test_partial_retry_requires_existing_receipt(tmp_path, monkeypatch):
    args, _protocol = _run_args(tmp_path, monkeypatch, sections="B3_2")
    with pytest.raises(ValueError, match="requires an existing run_status.json"):
        runner.run(args)


def test_partial_success_remains_incomplete_when_exact_16_records_are_absent(tmp_path, monkeypatch):
    args, protocol = _run_args(tmp_path, monkeypatch, sections="B3_2")
    b3_row = next(row for row in protocol["sections"] if row["section"] == "B3_2")
    receipt = _receipt_for_args(
        args,
        protocol,
        {
            "B3_2": {
                "section": "B3_2",
                "donor": str(b3_row["donor"]),
                "status": "failed",
            }
        },
    )
    args.output_root.mkdir()
    runner._write_json(args.output_root / "run_status.json", receipt)
    monkeypatch.setattr(
        runner,
        "_run_section",
        lambda row, **_kwargs: {
            "section": str(row["section"]),
            "donor": str(row["donor"]),
            "status": "complete",
        },
    )

    assert runner.run(args) == 0
    merged = json.loads((args.output_root / "run_status.json").read_text())
    assert merged["status"] == "incomplete"
    assert merged["failed_sections"] == 0
    assert set(merged["sections"]) == {"B3_2"}


def test_overall_complete_requires_exact_sections_and_real_completion_statuses():
    protocol = runner._load_protocol(runner.DEFAULT_PROTOCOL)
    expected = {str(row["section"]) for row in protocol["sections"]}
    records = {section: {"section": section, "status": "complete"} for section in expected}
    records["B3_2"]["status"] = "complete_existing"
    assert runner._overall_receipt_status(records, expected) == ("complete", 0)

    records["B3_2"]["status"] = "dry_complete"
    assert runner._overall_receipt_status(records, expected) == ("incomplete", 0)

    records["B3_2"]["status"] = "complete"
    del records["D6"]
    assert runner._overall_receipt_status(records, expected) == ("incomplete", 0)


def test_partial_retry_rejects_stale_false_complete_receipt(tmp_path, monkeypatch):
    args, protocol = _run_args(tmp_path, monkeypatch, sections="B3_2")
    receipt = _receipt_for_args(args, protocol, {})
    receipt["status"] = "complete"
    args.output_root.mkdir()
    runner._write_json(args.output_root / "run_status.json", receipt)

    with pytest.raises(ValueError, match="claims completion without all 16 sections"):
        runner.run(args)


def test_fresh_full_invocation_still_replaces_prior_receipt(tmp_path, monkeypatch):
    args, protocol = _run_args(tmp_path, monkeypatch, sections="")
    args.output_root.mkdir()
    runner._write_json(
        args.output_root / "run_status.json",
        {"schema": "stale", "sections": {"not-a-section": {}}},
    )
    monkeypatch.setattr(
        runner,
        "_run_section",
        lambda row, **_kwargs: {
            "section": str(row["section"]),
            "donor": str(row["donor"]),
            "status": "complete",
        },
    )

    assert runner.run(args) == 0
    receipt = json.loads((args.output_root / "run_status.json").read_text())
    assert receipt["schema"] == runner.RUN_RECEIPT_SCHEMA
    assert receipt["status"] == "complete"
    assert set(receipt["sections"]) == {str(row["section"]) for row in protocol["sections"]}
