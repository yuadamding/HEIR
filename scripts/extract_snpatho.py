#!/usr/bin/env python3
"""Selectively extract one snPATHO Visium sample from the 9.5 GB GEO tar."""

import argparse
import gzip
import os
import shutil
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path

SAMPLE_TO_GSM = {
    "4411": "GSM8291069",
    "4066": "GSM8291070",
    "4399": "GSM8291071",
}

# Files required to reproduce the image-to-Visium benchmark.  The GEO archive
# also contains multi-gigabyte Loupe browser bundles and raw matrices; neither
# is an input to HEIR/SIGHT, and unpacking them roughly doubles preparation I/O.
BENCHMARK_SUFFIXES = (
    "_high_res.tif.gz",
    "_CytAssist_image.tif.gz",
    ".json.gz",
    "_filtered_feature_bc_matrix.h5",
    "_scalefactors_json.json.gz",
    "_tissue_hires_image.png.gz",
    "_tissue_lowres_image.png.gz",
    "_tissue_positions.csv.gz",
)


def _find_pigz():
    """Find pigz on PATH or in the local conda package cache."""
    executable = shutil.which("pigz")
    if executable is not None:
        return executable
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        prefix_path = Path(prefix).resolve()
        package_root = (
            prefix_path.parent.parent / "pkgs"
            if prefix_path.parent.name == "envs"
            else prefix_path.parent / "pkgs"
        )
        for candidate in sorted(package_root.glob("pigz-*/bin/pigz"), reverse=True):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _decompress_gzip(stream, target, pigz) -> None:
    if pigz is None:
        with gzip.GzipFile(fileobj=stream, mode="rb") as decompressed:
            shutil.copyfileobj(decompressed, target, length=8 * 1024 * 1024)
        return

    process = subprocess.Popen(
        [pigz, "-d", "-c", "-p", str(min(8, os.cpu_count() or 1))],
        stdin=subprocess.PIPE,
        stdout=target,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    try:
        shutil.copyfileobj(stream, process.stdin, length=8 * 1024 * 1024)
        process.stdin.close()
        error = process.stderr.read()
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if returncode != 0:
        raise RuntimeError(
            "pigz failed with exit code %d: %s"
            % (returncode, error.decode("utf-8", errors="replace").strip())
        )


def _write_member_atomically(stream, name: str, destination: Path, pigz) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle as target:
            if name.endswith(".gz"):
                _decompress_gzip(stream, target, pigz)
            else:
                shutil.copyfileobj(stream, target, length=8 * 1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _is_complete(archive: Path, member: tarfile.TarInfo, destination: Path) -> bool:
    """Check output length without reading the target's validation contents."""
    if not destination.is_file():
        return False
    actual_size = destination.stat().st_size
    if not member.name.endswith(".gz"):
        return actual_size == member.size

    # RFC 1952 stores the uncompressed length modulo 2^32 in the final four
    # bytes. This also handles the 4.36 GB 4066 BigTIFF without re-inflating it.
    with archive.open("rb") as source:
        source.seek(member.offset_data + member.size - 4)
        expected_modulo = struct.unpack("<I", source.read(4))[0]
    return actual_size > 0 and actual_size % (1 << 32) == expected_modulo


def extract(archive: Path, sample: str, output: Path, *, include_bulky: bool = False) -> None:
    prefix = SAMPLE_TO_GSM[sample] + "_"
    output.mkdir(parents=True, exist_ok=True)
    pigz = _find_pigz()
    written = 0
    skipped = 0
    with tarfile.open(archive, "r") as handle:
        for member in handle:
            name = Path(member.name).name
            if not name.startswith(prefix) or not member.isfile():
                continue
            if not include_bulky and not name.endswith(BENCHMARK_SUFFIXES):
                continue
            stream = handle.extractfile(member)
            if stream is None:
                continue
            destination_name = name[:-3] if name.endswith(".gz") else name
            destination = output / destination_name
            if _is_complete(archive, member, destination):
                skipped += 1
                continue
            _write_member_atomically(stream, name, destination, pigz)
            written += 1
    if written + skipped == 0:
        raise RuntimeError("no members matched %s in %s" % (prefix, archive))
    print(
        "prepared %d members in %s (%d written, %d complete and reused)"
        % (written + skipped, output, written, skipped)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample", choices=sorted(SAMPLE_TO_GSM), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-bulky",
        action="store_true",
        help="also unpack Loupe and raw-matrix files not used by the benchmark",
    )
    args = parser.parse_args()
    extract(
        args.archive.expanduser().resolve(),
        args.sample,
        args.output.expanduser().resolve(),
        include_bulky=args.include_bulky,
    )


if __name__ == "__main__":
    main()
