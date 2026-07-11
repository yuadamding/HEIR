"""Geometry-only filtering of segmented nuclei to evaluable Visium disks.

This stage intentionally accepts no expression matrix.  It uses only the
Space Ranger nucleus centroids, tissue positions, and spot diameter so costly
pathology features are extracted only for nuclei that can contribute to the
locked spot-level benchmark.
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np

from heir.image.nuclei import NucleusTable, assign_nuclei_to_visium_spots, load_nuclei
from heir.utils import sha256_file

from .spatial_truth import read_spot_diameter, read_tissue_positions

PathLike = Union[str, os.PathLike]

CAPTURE_FILTER_CONTRACT = "heir.visium_capture_filter"
CAPTURE_FILTER_VERSION = 1
_ID_ALIASES = ("nucleus_id", "cell_id", "object_id", "id")


def _strings(values: Sequence[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype="<U%d" % width)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


@dataclass(frozen=True)
class CaptureAreaAssignment:
    """Assignment of every source nucleus to in-tissue Visium geometry."""

    source_nucleus_ids: np.ndarray
    source_centroids_px: np.ndarray
    source_spot_index: np.ndarray
    source_spot_distance_px: np.ndarray
    retained_source_index: np.ndarray
    spot_ids: np.ndarray
    spot_coordinates_px: np.ndarray
    position_source_index: np.ndarray
    spot_radius_px: float
    coordinate_scale: float

    def __post_init__(self) -> None:
        identifiers = _strings(self.source_nucleus_ids.tolist())
        centroids = np.asarray(self.source_centroids_px, dtype=np.float64)
        spot_index = np.asarray(self.source_spot_index, dtype=np.int64)
        distance = np.asarray(self.source_spot_distance_px, dtype=np.float64)
        retained = np.asarray(self.retained_source_index, dtype=np.int64)
        spot_ids = _strings(self.spot_ids.tolist())
        spot_coordinates = np.asarray(self.spot_coordinates_px, dtype=np.float64)
        position_source_index = np.asarray(self.position_source_index, dtype=np.int64)
        count = len(identifiers)
        if not count or len(set(identifiers.tolist())) != count:
            raise ValueError("source nucleus IDs must be non-empty and unique")
        if centroids.shape != (count, 2) or not np.isfinite(centroids).all():
            raise ValueError("source centroids must be finite with shape (nuclei, 2)")
        if spot_index.shape != (count,) or distance.shape != (count,):
            raise ValueError("source assignment vectors must align to nuclei")
        if not len(spot_ids) or len(set(spot_ids.tolist())) != len(spot_ids):
            raise ValueError("in-tissue spot IDs must be non-empty and unique")
        if spot_coordinates.shape != (len(spot_ids), 2):
            raise ValueError("spot coordinates must align to in-tissue spot IDs")
        if not np.isfinite(spot_coordinates).all():
            raise ValueError("spot coordinates must be finite")
        if position_source_index.shape != (len(spot_ids),):
            raise ValueError("position source indices must align to in-tissue spots")
        if np.any(position_source_index < 0):
            raise ValueError("position source indices cannot be negative")
        if np.any(spot_index < -1) or np.any(spot_index >= len(spot_ids)):
            raise ValueError("source spot assignment contains an invalid index")
        expected_retained = np.flatnonzero(spot_index >= 0)
        if not np.array_equal(retained, expected_retained):
            raise ValueError("retained source indices must exactly match assigned nuclei")
        if not len(retained):
            raise ValueError("no nuclei fall inside an in-tissue Visium disk")
        if not np.isfinite(distance[retained]).all() or np.any(distance[retained] < 0):
            raise ValueError("retained assignment distances must be finite and non-negative")
        if not np.isfinite(self.spot_radius_px) or self.spot_radius_px <= 0:
            raise ValueError("spot_radius_px must be finite and positive")
        if not np.isfinite(self.coordinate_scale) or self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be finite and positive")
        object.__setattr__(self, "source_nucleus_ids", _readonly(identifiers))
        object.__setattr__(self, "source_centroids_px", _readonly(centroids))
        object.__setattr__(self, "source_spot_index", _readonly(spot_index))
        object.__setattr__(self, "source_spot_distance_px", _readonly(distance))
        object.__setattr__(self, "retained_source_index", _readonly(retained))
        object.__setattr__(self, "spot_ids", _readonly(spot_ids))
        object.__setattr__(self, "spot_coordinates_px", _readonly(spot_coordinates))
        object.__setattr__(self, "position_source_index", _readonly(position_source_index))

    @property
    def retained_nucleus_ids(self) -> np.ndarray:
        return self.source_nucleus_ids[self.retained_source_index]

    @property
    def retained_spot_index(self) -> np.ndarray:
        return self.source_spot_index[self.retained_source_index]

    @property
    def retained_spot_distance_px(self) -> np.ndarray:
        return self.source_spot_distance_px[self.retained_source_index]


def assign_nuclei_to_visium_capture_area(
    nuclei: NucleusTable,
    *,
    positions_path: PathLike,
    scalefactors_path: PathLike,
    coordinate_scale: float = 1.0,
) -> CaptureAreaAssignment:
    """Assign nuclei using only in-tissue spot coordinates and disk diameter."""

    scale = float(coordinate_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("coordinate_scale must be finite and positive")
    positions = read_tissue_positions(positions_path, coordinate_scale=scale)
    in_tissue_index = np.flatnonzero(positions.in_tissue).astype(np.int64)
    if not len(in_tissue_index):
        raise ValueError("tissue positions contain no in-tissue spots")
    spot_ids = positions.barcodes[in_tissue_index]
    spot_coordinates = positions.coordinates_px[in_tissue_index]
    spot_radius = (
        read_spot_diameter(
            scalefactors_path,
            coordinate_scale=scale,
        )
        / 2.0
    )
    assignment = assign_nuclei_to_visium_spots(
        nuclei.centroids_px,
        spot_coordinates,
        spot_radius=spot_radius,
        spot_ids=spot_ids,
    )
    retained = np.flatnonzero(assignment.assigned).astype(np.int64)
    return CaptureAreaAssignment(
        source_nucleus_ids=nuclei.source_ids,
        source_centroids_px=nuclei.centroids_px,
        source_spot_index=assignment.spot_index,
        source_spot_distance_px=assignment.distance,
        retained_source_index=retained,
        spot_ids=spot_ids,
        spot_coordinates_px=spot_coordinates,
        position_source_index=in_tissue_index,
        spot_radius_px=spot_radius,
        coordinate_scale=scale,
    )


def _read_nucleus_csv(path: Path) -> Tuple[str, Tuple[str, ...], list]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        if not sample:
            raise ValueError("nucleus CSV is empty")
        if path.suffix.lower() in {".tsv", ".tab"}:
            delimiter = "\t"
        else:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
            except csv.Error:
                delimiter = ","
        handle.seek(0)
        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        raise ValueError("nucleus CSV is empty")
    header = tuple(rows[0])
    if not header or any(not value.strip() for value in header):
        raise ValueError("nucleus CSV header cannot contain blank columns")
    if len(set(header)) != len(header):
        raise ValueError("nucleus CSV header columns must be unique")
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError("nucleus CSV has no data rows")
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != len(header):
            raise ValueError("nucleus CSV row %d does not match the header width" % row_number)
    lookup = {_column_key(value): index for index, value in enumerate(header)}
    id_indices = [lookup[alias] for alias in _ID_ALIASES if alias in lookup]
    if not id_indices:
        raise ValueError("Space Ranger nucleus CSV requires an explicit source ID column")
    return delimiter, header, data_rows


def _temporary_path(destination: Path) -> Tuple[int, Path]:
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    return descriptor, Path(temporary)


def _commit(temporary: Path, destination: Path, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, destination)
    else:
        os.link(temporary, destination)
        temporary.unlink()


def _write_filtered_csv(
    destination: Path,
    *,
    delimiter: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    retained_indices: np.ndarray,
    overwrite: bool,
) -> None:
    descriptor, temporary = _temporary_path(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
            writer.writerow(header)
            for index in retained_indices:
                writer.writerow(rows[int(index)])
            handle.flush()
            os.fsync(handle.fileno())
        _commit(temporary, destination, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def _write_assignment_npz(
    destination: Path,
    assignment: CaptureAreaAssignment,
    *,
    source_hashes: Mapping[str, str],
    filtered_csv_sha256: str,
    overwrite: bool,
) -> None:
    descriptor, temporary = _temporary_path(destination)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(
                handle,
                __contract__=np.asarray(CAPTURE_FILTER_CONTRACT),
                __version__=np.asarray(CAPTURE_FILTER_VERSION, dtype=np.int64),
                geometry_only=np.asarray(True),
                target_expression_accessed=np.asarray(False),
                source_nucleus_ids=assignment.source_nucleus_ids,
                source_centroids_px=assignment.source_centroids_px,
                source_nucleus_spot_index=assignment.source_spot_index,
                source_nucleus_spot_distance_px=assignment.source_spot_distance_px,
                source_retained_mask=assignment.source_spot_index >= 0,
                source_row_index=assignment.retained_source_index,
                nucleus_ids=assignment.retained_nucleus_ids,
                nucleus_spot_index=assignment.retained_spot_index,
                nucleus_spot_distance_px=assignment.retained_spot_distance_px,
                spot_ids=assignment.spot_ids,
                spot_coordinates_px=assignment.spot_coordinates_px,
                position_source_index=assignment.position_source_index,
                spot_radius_px=np.asarray(assignment.spot_radius_px, dtype=np.float64),
                coordinate_scale=np.asarray(assignment.coordinate_scale, dtype=np.float64),
                nuclei_source_sha256=np.asarray(source_hashes["nuclei"]),
                positions_source_sha256=np.asarray(source_hashes["positions"]),
                scalefactors_source_sha256=np.asarray(source_hashes["scalefactors"]),
                filtered_csv_sha256=np.asarray(filtered_csv_sha256),
            )
            handle.flush()
            os.fsync(handle.fileno())
        _commit(temporary, destination, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(
    destination: Path,
    payload: Mapping[str, object],
    *,
    overwrite: bool,
) -> None:
    descriptor, temporary = _temporary_path(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _commit(temporary, destination, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def filter_nucleus_csv_to_visium(
    *,
    nuclei_path: PathLike,
    positions_path: PathLike,
    scalefactors_path: PathLike,
    filtered_csv_path: PathLike,
    assignment_npz_path: PathLike,
    provenance_json_path: PathLike,
    coordinate_scale: float = 1.0,
    overwrite: bool = False,
) -> Tuple[CaptureAreaAssignment, Dict[str, object]]:
    """Filter a Space Ranger CSV while preserving its columns and row values."""

    nuclei_source = Path(nuclei_path).expanduser().resolve()
    positions_source = Path(positions_path).expanduser().resolve()
    scalefactors_source = Path(scalefactors_path).expanduser().resolve()
    destinations = tuple(
        Path(value).expanduser().resolve()
        for value in (filtered_csv_path, assignment_npz_path, provenance_json_path)
    )
    if len(set(destinations)) != len(destinations):
        raise ValueError("filtered CSV, assignment NPZ, and provenance JSON paths must differ")
    if set(destinations) & {nuclei_source, positions_source, scalefactors_source}:
        raise ValueError("capture-filter outputs cannot replace source geometry files")
    filtered_csv, assignment_npz, provenance_json = destinations
    if filtered_csv.suffix.lower() not in {".csv", ".tsv", ".tab"}:
        raise ValueError("filtered nucleus output must use .csv or .tsv")
    if assignment_npz.suffix.lower() != ".npz":
        raise ValueError("capture assignment output must use .npz")
    if provenance_json.suffix.lower() != ".json":
        raise ValueError("capture provenance output must use .json")
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(str(destination))

    delimiter, header, rows = _read_nucleus_csv(nuclei_source)
    nuclei = load_nuclei(nuclei_source)
    if len(rows) != len(nuclei):
        raise RuntimeError("nucleus CSV parser row counts disagree")
    lookup = {_column_key(value): index for index, value in enumerate(header)}
    id_index = next(lookup[alias] for alias in _ID_ALIASES if alias in lookup)
    csv_ids = _strings([row[id_index] for row in rows])
    if not np.array_equal(csv_ids, nuclei.source_ids):
        raise ValueError("nucleus source IDs changed during canonical CSV parsing")

    assignment = assign_nuclei_to_visium_capture_area(
        nuclei,
        positions_path=positions_source,
        scalefactors_path=scalefactors_source,
        coordinate_scale=coordinate_scale,
    )
    source_hashes = {
        "nuclei": sha256_file(nuclei_source),
        "positions": sha256_file(positions_source),
        "scalefactors": sha256_file(scalefactors_source),
    }
    _write_filtered_csv(
        filtered_csv,
        delimiter=delimiter,
        header=header,
        rows=rows,
        retained_indices=assignment.retained_source_index,
        overwrite=overwrite,
    )
    filtered_hash = sha256_file(filtered_csv)
    _write_assignment_npz(
        assignment_npz,
        assignment,
        source_hashes=source_hashes,
        filtered_csv_sha256=filtered_hash,
        overwrite=overwrite,
    )
    assignment_hash = sha256_file(assignment_npz)
    payload: Dict[str, object] = {
        "contract": CAPTURE_FILTER_CONTRACT,
        "version": CAPTURE_FILTER_VERSION,
        "geometry_only": True,
        "target_expression_accessed": False,
        "inputs": {
            "nuclei": {"path": str(nuclei_source), "sha256": source_hashes["nuclei"]},
            "positions": {
                "path": str(positions_source),
                "sha256": source_hashes["positions"],
            },
            "scalefactors": {
                "path": str(scalefactors_source),
                "sha256": source_hashes["scalefactors"],
            },
        },
        "outputs": {
            "filtered_nuclei": {"path": str(filtered_csv), "sha256": filtered_hash},
            "assignment": {"path": str(assignment_npz), "sha256": assignment_hash},
        },
        "nucleus_columns": list(header),
        "source_nuclei": len(assignment.source_nucleus_ids),
        "retained_nuclei": len(assignment.retained_source_index),
        "excluded_nuclei": len(assignment.source_nucleus_ids)
        - len(assignment.retained_source_index),
        "in_tissue_spots": len(assignment.spot_ids),
        "spot_radius_px": assignment.spot_radius_px,
        "coordinate_scale": assignment.coordinate_scale,
    }
    _write_json(provenance_json, payload, overwrite=overwrite)
    return assignment, payload


__all__ = [
    "CAPTURE_FILTER_CONTRACT",
    "CAPTURE_FILTER_VERSION",
    "CaptureAreaAssignment",
    "assign_nuclei_to_visium_capture_area",
    "filter_nucleus_csv_to_visium",
]
