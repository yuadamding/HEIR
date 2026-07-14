#!/usr/bin/env python3
"""Resource-bounded, resumable Space Ranger processing for E-MTAB-14560."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs/natcommun_matched_regional_protocol.json"
DEFAULT_DATA_ROOT = Path("/mnt/seagate/HnE/NatCommun_2025_s41467_025_59005_9")
DEFAULT_SPACERANGER = Path(
    "/storage/hackathon_2026/tools/spaceranger-4.1.0/bin/spaceranger"
)
DEFAULT_REFERENCE = DEFAULT_DATA_ROOT / "refs/refdata-gex-GRCh38-2020-A"
DEFAULT_PROBE_SET = Path(
    "/storage/hackathon_2026/tools/spaceranger-4.1.0/external/"
    "tenx_feature_references/targeted_panels/"
    "Visium_Human_Transcriptome_Probe_Set_v2.0_GRCh38-2020-A.csv"
)
# Martian requires POSIX symlinks, which the exFAT /mnt/seagate volume cannot
# create.  Keep the transient pipestance on ext4; downstream compact sources
# and benchmark reports remain on /mnt/seagate.
DEFAULT_OUTPUT_ROOT = Path("/storage/HEIR_work/natcommun_spaceranger")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_protocol(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "heir.natcommun_matched_regional_protocol.v1"
        or not isinstance(value.get("sections"), list)
        or len(value["sections"]) != 16
    ):
        raise ValueError("NatCommun protocol is malformed")
    sections = value["sections"]
    ids = [str(row["section"]) for row in sections]
    if len(set(ids)) != len(ids):
        raise ValueError("NatCommun section IDs are not unique")
    return value


def _fastq_directories(raw_root: Path, sample: str) -> tuple[Path, ...]:
    matches = sorted(raw_root.rglob(f"{sample}*_R1_001.fastq.gz"))
    directories = tuple(sorted({path.parent.resolve() for path in matches}))
    if not directories:
        raise FileNotFoundError(f"no FASTQ directory found for {sample}")
    for directory in directories:
        files = list(directory.glob(f"{sample}*fastq.gz"))
        read_kinds = {
            part
            for path in files
            for part in ("R1", "R2", "I1", "I2")
            if f"_{part}_" in path.name
        }
        if read_kinds != {"R1", "R2", "I1", "I2"}:
            raise ValueError(f"incomplete paired dual-index FASTQs for {sample}: {directory}")
    return directories


def _output_complete(output: Path) -> bool:
    required = (
        output / "outs/filtered_feature_bc_matrix.h5",
        output / "outs/molecule_info.h5",
        output / "outs/spatial/tissue_positions.csv",
        output / "outs/spatial/scalefactors_json.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _command(
    row: Mapping[str, object],
    *,
    data_root: Path,
    spaceranger: Path,
    reference: Path,
    probe_set: Path,
    output_root: Path,
    localcores: int,
    localmem: int,
    dry: bool,
) -> tuple[list[str], tuple[Path, ...]]:
    processed = data_root / "arrayexpress/E-MTAB-14560/processed_data"
    raw = data_root / "arrayexpress/E-MTAB-14560/ENA_submitted"
    fastq_directories = _fastq_directories(raw, str(row["fastq_sample"]))
    command = [
        str(spaceranger),
        "count",
        "--id",
        str(row["section"]),
        "--description",
        f"E-MTAB-14560 {row['donor']} {row['section']}",
        "--image",
        str((processed / str(row["h_and_e"])).resolve()),
        "--cytaimage",
        str((processed / str(row["cytassist"])).resolve()),
        "--slide",
        str(row["slide"]),
        "--area",
        str(row["area"]),
        "--transcriptome",
        str(reference.resolve()),
        "--probe-set",
        str(probe_set.resolve()),
        "--fastqs",
        ",".join(str(path) for path in fastq_directories),
        "--sample",
        str(row["fastq_sample"]),
        "--output-dir",
        str((output_root / str(row["section"])).resolve()),
        "--create-bam=false",
        "--nosecondary",
        "--disable-cell-annotation",
        "--disable-ui",
        "--localcores",
        str(localcores),
        "--localmem",
        str(localmem),
        "--localvmem",
        # Space Ranger image stages request a 64-GB virtual address space even
        # when their physical RSS stays within --localmem.
        str(max(64, localmem + 8)),
    ]
    if dry:
        command.append("--dry")
    return command, fastq_directories


def _run_section(
    row: Mapping[str, object],
    *,
    data_root: Path,
    spaceranger: Path,
    reference: Path,
    probe_set: Path,
    output_root: Path,
    localcores: int,
    localmem: int,
    dry: bool,
) -> Mapping[str, object]:
    section = str(row["section"])
    output = output_root / section
    log_path = output_root / "logs" / f"{section}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not dry and _output_complete(output):
        return {
            "section": section,
            "donor": str(row["donor"]),
            "status": "complete_existing",
            "output": str(output),
            "log": str(log_path),
            "primary_eligible": bool(row["primary_eligible"]),
        }
    command, fastq_directories = _command(
        row,
        data_root=data_root,
        spaceranger=spaceranger,
        reference=reference,
        probe_set=probe_set,
        output_root=output_root,
        localcores=localcores,
        localmem=localmem,
        dry=dry,
    )
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = str(localcores)
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "started_unix": started}) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=output_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env=environment,
        )
    complete = dry or _output_complete(output)
    status = "dry_complete" if dry and process.returncode == 0 else (
        "complete" if process.returncode == 0 and complete else "failed"
    )
    return {
        "section": section,
        "donor": str(row["donor"]),
        "status": status,
        "returncode": int(process.returncode),
        "elapsed_seconds": float(time.time() - started),
        "output": str(output),
        "log": str(log_path),
        "primary_eligible": bool(row["primary_eligible"]),
        "fastq_directories": [str(path) for path in fastq_directories],
        "command": command,
        "thread_environment": {
            key: environment[key]
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def run(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    data_root = args.data_root.expanduser().resolve()
    spaceranger = args.spaceranger.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    probe_set = args.probe_set.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not spaceranger.is_file() or not os.access(spaceranger, os.X_OK):
        raise FileNotFoundError(f"Space Ranger is not executable: {spaceranger}")
    if not (reference / "reference.json").is_file():
        raise FileNotFoundError(f"10x reference is incomplete: {reference}")
    if not probe_set.is_file():
        raise FileNotFoundError(f"Visium probe set is missing: {probe_set}")
    version = subprocess.run(
        [str(spaceranger), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if version != "spaceranger 4.1.0":
        raise ValueError(f"unexpected local Space Ranger version: {version}")

    requested = set(args.sections.split(",")) if args.sections else None
    rows = [
        row
        for row in protocol["sections"]
        if requested is None or str(row["section"]) in requested
    ]
    if requested is not None and {str(row["section"]) for row in rows} != requested:
        raise ValueError("--sections includes unknown section IDs")
    processed = data_root / "arrayexpress/E-MTAB-14560/processed_data"
    for row in rows:
        for key in ("h_and_e", "cytassist"):
            path = processed / str(row[key])
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing {key} input: {path}")
        _fastq_directories(
            data_root / "arrayexpress/E-MTAB-14560/ENA_submitted",
            str(row["fastq_sample"]),
        )

    receipt = {
        "schema": "heir.natcommun_spaceranger_run.v1",
        "analysis_scope": protocol["analysis_scope"],
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "spaceranger": str(spaceranger),
        "spaceranger_version": version,
        "reference": str(reference),
        "reference_metadata_sha256": _sha256(reference / "reference.json"),
        "probe_set": str(probe_set),
        "probe_set_sha256": _sha256(probe_set),
        "pipestance_root": str(output_root),
        "pipestance_filesystem_requirement": "POSIX_symlink_capable_ext4",
        "compact_downstream_outputs_root": "/mnt/seagate/HEIR_runs",
        "resource_limits": {
            "parallel_sections": args.max_workers,
            "local_cores_per_section": args.localcores,
            "local_memory_gb_per_section": args.localmem,
            "maximum_concurrent_cores": args.max_workers * args.localcores,
            "maximum_concurrent_memory_gb": args.max_workers * args.localmem,
            "virtual_address_space_gb_per_section": max(64, args.localmem + 8),
            "thread_libraries_capped_per_section": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        },
        "published_processing_deviation": (
            "Space Ranger 4.1.0 rerun; the paper used 2.0.1. Reference and v2.0 "
            "probe-set release remain GRCh38-2020-A."
        ),
        "segmentation_policy": (
            "Space Ranger default; nucleus segmentation is not applicable to this "
            "standard Visium v2 spot array and is therefore not forced"
        ),
        "started_unix": time.time(),
        "dry": bool(args.dry),
        "sections": {},
    }
    receipt_path = output_root / "run_status.json"
    receipt_lock = threading.Lock()
    _write_json(receipt_path, receipt)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _run_section,
                row,
                data_root=data_root,
                spaceranger=spaceranger,
                reference=reference,
                probe_set=probe_set,
                output_root=output_root,
                localcores=args.localcores,
                localmem=args.localmem,
                dry=args.dry,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            result = future.result()
            failures += int(result["status"] == "failed")
            with receipt_lock:
                receipt["sections"][result["section"]] = result
                receipt["updated_unix"] = time.time()
                _write_json(receipt_path, receipt)
            print(
                f"Space Ranger {result['section']}: {result['status']}",
                flush=True,
            )
    receipt["completed_unix"] = time.time()
    receipt["status"] = "complete" if failures == 0 else "failed"
    receipt["failed_sections"] = failures
    _write_json(receipt_path, receipt)
    return 0 if failures == 0 else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--spaceranger", type=Path, default=DEFAULT_SPACERANGER)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--probe-set", type=Path, default=DEFAULT_PROBE_SET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sections", default="", help="comma-separated section IDs")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--localcores", type=int, default=4)
    parser.add_argument("--localmem", type=int, default=24)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args(argv)
    if args.max_workers not in {1, 2}:
        parser.error("--max-workers must be 1 or 2")
    if args.localcores < 2 or args.localcores > 8:
        parser.error("--localcores must be between 2 and 8")
    if args.localmem < 16 or args.localmem > 48:
        parser.error("--localmem must be between 16 and 48 GB")
    if args.max_workers * args.localcores > 12 or args.max_workers * args.localmem > 64:
        parser.error("combined Space Ranger resource ceiling exceeded")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
