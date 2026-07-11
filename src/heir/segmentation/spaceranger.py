"""Space Ranger 4.x nucleus-segmentation adapter.

This module converts ``spaceranger segment`` GeoJSON into HEIR's canonical
nucleus and feature contracts.  Geometry is derived from the polygon itself;
the Space Ranger centroid is retained only after it is checked against the
polygon bounds.  Exported NPZ files never contain object arrays or pickles.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from heir.image.nuclei import NucleusTable, canonical_nucleus_ids

PathLike = Union[str, Path]

SEGMENTATION_METHOD = "10x-spaceranger-segment"
SEGMENTATION_SCHEMA = "heir-spaceranger-segmentation-v1"
FEATURE_TRANSFORM = "robust-median-mad-z-v1"
MORPHOLOGY_FEATURE_NAMES = (
    "morph_area_px2",
    "morph_perimeter_px",
    "morph_circularity",
    "morph_eccentricity",
    "morph_major_axis_length_px",
    "morph_minor_axis_length_px",
    "morph_orientation_rad",
    "morph_solidity",
    "morph_extent",
    "morph_equivalent_diameter_px",
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SLIDE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VERSION = re.compile(r"(?:^|\s)(4\.\d+(?:\.\d+)?)(?:\s|$)")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SUFFIXES = {".tif", ".tiff", ".btf", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SpaceRangerSegmentation:
    """Validated, immutable Space Ranger segmentation in native pixel space."""

    slide_id: str
    nucleus_ids: np.ndarray
    source_ids: np.ndarray
    centroids_px: np.ndarray
    morphology: np.ndarray
    spaceranger_version: str
    source_name: str
    source_sha256: str
    skipped_features: int = 0
    method: str = SEGMENTATION_METHOD

    def __post_init__(self) -> None:
        if not _SLIDE_ID.fullmatch(self.slide_id):
            raise ValueError("slide_id must contain only letters, digits, dot, underscore, or dash")
        version = normalize_spaceranger_version(self.spaceranger_version)
        if self.method != SEGMENTATION_METHOD:
            raise ValueError("unexpected segmentation method")
        if not self.source_name or Path(self.source_name).name != self.source_name:
            raise ValueError("source_name must be a file name without directories")
        if not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.skipped_features, bool) or int(self.skipped_features) < 0:
            raise ValueError("skipped_features must be a non-negative integer")
        ids = _string_array(self.nucleus_ids)
        source_ids = _string_array(self.source_ids)
        centroids = np.asarray(self.centroids_px, dtype=np.float64)
        morphology = np.asarray(self.morphology, dtype=np.float32)
        count = len(ids)
        if count == 0:
            raise ValueError("Space Ranger segmentation has no valid nuclei")
        if len(set(ids.tolist())) != count:
            raise ValueError("canonical nucleus IDs must be unique")
        if source_ids.shape != (count,) or len(set(source_ids.tolist())) != count:
            raise ValueError("Space Ranger source cell IDs must be unique and aligned")
        if centroids.shape != (count, 2) or not np.isfinite(centroids).all():
            raise ValueError("centroids_px must be finite with shape (nuclei, 2)")
        if morphology.shape != (count, len(MORPHOLOGY_FEATURE_NAMES)):
            raise ValueError("morphology must contain the canonical ten features")
        if not np.isfinite(morphology).all():
            raise ValueError("morphology must contain only finite values")
        object.__setattr__(self, "nucleus_ids", _readonly(ids))
        object.__setattr__(self, "source_ids", _readonly(source_ids))
        object.__setattr__(self, "centroids_px", _readonly(centroids))
        object.__setattr__(self, "morphology", _readonly(morphology))
        object.__setattr__(self, "spaceranger_version", version)
        object.__setattr__(self, "skipped_features", int(self.skipped_features))

    def __len__(self) -> int:
        return int(self.nucleus_ids.shape[0])

    def to_nucleus_table(self) -> NucleusTable:
        """Return the generic HEIR nucleus table without inventing confidence."""

        count = len(self)
        metadata = {
            "source_id": self.source_ids,
            "segmentation_method": _repeated_string(self.method, count),
            "segmentation_version": _repeated_string(self.spaceranger_version, count),
            "segmentation_source_sha256": _repeated_string(self.source_sha256, count),
        }
        return NucleusTable(
            nucleus_ids=self.nucleus_ids,
            centroids_px=self.centroids_px,
            morphology=self.morphology,
            morphology_names=MORPHOLOGY_FEATURE_NAMES,
            confidence=np.full(count, np.nan, dtype=np.float32),
            cell_types=_repeated_string("", count),
            source_ids=self.source_ids,
            metadata=metadata,
        )


@dataclass(frozen=True)
class SpaceRangerSegmentRun:
    """Provenance and output locations for a successful segmentation run."""

    run_id: str
    command: Tuple[str, ...]
    executable: Path
    spaceranger_version: str
    tissue_image: Path
    tissue_image_sha256: str
    run_directory: Path
    geojson_path: Path
    log_path: Path
    cuda_visible_devices: str


def read_spaceranger_geojson(
    path: PathLike,
    *,
    slide_id: str,
    spaceranger_version: str,
    minimum_area_px2: float = 8.0,
) -> SpaceRangerSegmentation:
    """Parse a Space Ranger 4.x ``nucleus_segmentations.geojson`` file.

    Degenerate polygons and polygons below ``minimum_area_px2`` are counted and
    skipped.  Contract violations such as duplicate cell IDs, non-Polygon
    geometry, holes, non-finite values, or an out-of-bounds supplied centroid
    fail closed.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if not _SLIDE_ID.fullmatch(slide_id):
        raise ValueError("slide_id must contain only letters, digits, dot, underscore, or dash")
    version = normalize_spaceranger_version(spaceranger_version)
    minimum_area = float(minimum_area_px2)
    if not np.isfinite(minimum_area) or minimum_area < 0.0:
        raise ValueError("minimum_area_px2 must be finite and non-negative")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("type") != "FeatureCollection":
        raise ValueError("Space Ranger GeoJSON must be a FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Space Ranger GeoJSON has no features")

    source_ids: List[str] = []
    centroids: List[Tuple[float, float]] = []
    morphology: List[Tuple[float, ...]] = []
    skipped = 0
    for feature_index, feature in enumerate(features):
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise ValueError("GeoJSON feature %d is not a Feature mapping" % feature_index)
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("type") != "Polygon":
            raise ValueError("GeoJSON feature %d is not a Polygon" % feature_index)
        coordinate_sets = geometry.get("coordinates")
        if not isinstance(coordinate_sets, list) or len(coordinate_sets) != 1:
            raise ValueError(
                "GeoJSON feature %d must contain one exterior polygon ring" % feature_index
            )
        polygon = _polygon_vertices(coordinate_sets[0], feature_index)
        descriptors = _polygon_descriptors(polygon)
        if descriptors is None or descriptors[0] < minimum_area:
            skipped += 1
            continue
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("GeoJSON feature %d has no properties mapping" % feature_index)
        source_id = str(properties.get("cell_id", "")).strip()
        if not source_id or any(character in source_id for character in "\r\n\t"):
            raise ValueError("GeoJSON feature %d has an invalid cell_id" % feature_index)
        centroid = _feature_centroid(properties, polygon, feature_index)
        source_ids.append(source_id)
        centroids.append(centroid)
        morphology.append(descriptors)

    canonical = canonical_nucleus_ids(source_ids, sample_id=slide_id)
    return SpaceRangerSegmentation(
        slide_id=slide_id,
        nucleus_ids=canonical,
        source_ids=_string_array(source_ids),
        centroids_px=np.asarray(centroids, dtype=np.float64),
        morphology=np.asarray(morphology, dtype=np.float32),
        spaceranger_version=version,
        source_name=source.name,
        source_sha256=_sha256(source),
        skipped_features=skipped,
    )


def export_spaceranger_artifacts(
    segmentation: SpaceRangerSegmentation,
    *,
    csv_path: PathLike,
    npz_path: PathLike,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Atomically export a canonical nucleus CSV and safe feature NPZ.

    The NPZ ``features`` matrix is robust-standardized for direct use by
    ``prepare-histology``; the raw ten-feature matrix is also retained as
    ``morphology``.  Space Ranger does not emit per-nucleus confidence, so the
    confidence vector is NaN instead of a misleading all-ones vector.
    """

    csv_output = Path(csv_path).expanduser().resolve()
    npz_output = Path(npz_path).expanduser().resolve()
    if csv_output == npz_output:
        raise ValueError("CSV and NPZ outputs must be different paths")
    if csv_output.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("nucleus-table output must use .csv or .tsv")
    if npz_output.suffix.lower() != ".npz":
        raise ValueError("feature output must use .npz")
    for output in (csv_output, npz_output):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not overwrite:
            raise FileExistsError(str(output))

    csv_temp = _temporary_path(csv_output)
    npz_temp = _temporary_path(npz_output)
    try:
        _write_segmentation_csv(segmentation, csv_temp, csv_output.suffix.lower())
        _write_segmentation_npz(segmentation, npz_temp)
        _commit_temp(csv_temp, csv_output, overwrite=overwrite)
        _commit_temp(npz_temp, npz_output, overwrite=overwrite)
    finally:
        for temporary in (csv_temp, npz_temp):
            temporary.unlink(missing_ok=True)
    return csv_output, npz_output


def discover_spaceranger_executable(executable: Optional[PathLike] = None) -> Path:
    """Find an executable Space Ranger installation without invoking a shell."""

    if executable is not None:
        return _validated_executable(Path(executable).expanduser())
    configured = os.environ.get("SPACERANGER")
    if configured:
        return _first_valid_installation(
            _installation_candidates(Path(configured).expanduser()), "SPACERANGER"
        )
    home = os.environ.get("SPACERANGER_HOME")
    if home:
        return _first_valid_installation(
            _installation_candidates(Path(home).expanduser()), "SPACERANGER_HOME"
        )
    on_path = shutil.which("spaceranger")
    if on_path:
        return _validated_executable(Path(on_path))
    candidates: List[Path] = []
    for root in (
        Path("/storage/hackathon_2026/tools"),
        Path("/opt"),
        Path("/usr/local"),
        Path.home(),
    ):
        if not root.is_dir():
            continue
        candidates.extend(root.glob("spaceranger-4*/spaceranger"))
        candidates.extend(root.glob("spaceranger-4*/bin/spaceranger"))
    valid = []
    for candidate in candidates:
        try:
            resolved = _validated_executable(candidate)
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        if resolved not in valid:
            valid.append(resolved)
    if not valid:
        raise FileNotFoundError(
            "could not find Space Ranger; set SPACERANGER, SPACERANGER_HOME, or PATH"
        )
    return max(valid, key=_executable_version_key)


def run_spaceranger_segment(
    tissue_image: PathLike,
    *,
    run_id: str,
    output_directory: PathLike,
    executable: Optional[PathLike] = None,
    localcores: int = 8,
    localmem_gb: int = 24,
    max_nucleus_diameter_px: Optional[int] = None,
    cuda_visible_devices: str = "auto",
    timeout_seconds: Optional[float] = None,
) -> SpaceRangerSegmentRun:
    """Run ``spaceranger segment`` safely with GPU selection left automatic.

    ``cuda_visible_devices='auto'`` preserves the parent CUDA environment and
    lets Space Ranger discover an available CUDA device.  An explicit value is
    passed only to the child process.  Existing run directories are never
    removed or overwritten.
    """

    image = Path(tissue_image).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(str(image))
    if image.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError("Space Ranger tissue image must be TIF, TIFF, BTF, JPG, or JPEG")
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9_-]*")
    cores = _positive_integer(localcores, "localcores")
    memory = _positive_integer(localmem_gb, "localmem_gb")
    diameter = None
    if max_nucleus_diameter_px is not None:
        diameter = _positive_integer(max_nucleus_diameter_px, "max_nucleus_diameter_px")
        if diameter > 1024:
            raise ValueError("max_nucleus_diameter_px cannot exceed 1024")
    if cuda_visible_devices != "auto":
        if not cuda_visible_devices or "\x00" in cuda_visible_devices:
            raise ValueError("cuda_visible_devices must be 'auto' or a non-empty value")
    if timeout_seconds is not None and (
        not np.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("timeout_seconds must be finite and positive")

    binary = discover_spaceranger_executable(executable)
    version = spaceranger_executable_version(binary)
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_directory = root / run_id
    if run_directory.exists():
        raise FileExistsError(str(run_directory))
    log_path = root / (run_id + ".spaceranger.log")
    if log_path.exists():
        raise FileExistsError(str(log_path))
    command = [
        str(binary),
        "segment",
        "--id",
        run_id,
        "--tissue-image",
        str(image),
        "--output-dir",
        str(run_directory),
        "--localcores",
        str(cores),
        "--localmem",
        str(memory),
        "--disable-ui",
    ]
    if diameter is not None:
        command.extend(("--max-nucleus-diameter-px", str(diameter)))
    environment = os.environ.copy()
    if cuda_visible_devices != "auto":
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    with log_path.open("x", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=str(root),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=None if timeout_seconds is None else float(timeout_seconds),
        )
    if result.returncode != 0:
        tail = _log_tail(log_path)
        raise RuntimeError(
            "spaceranger segment failed with exit code %d; log=%s%s"
            % (result.returncode, log_path, "\n" + tail if tail else "")
        )
    geojson = run_directory / "outs" / "nucleus_segmentations.geojson"
    if not geojson.is_file():
        raise RuntimeError(
            "spaceranger segment succeeded but did not produce %s; inspect %s" % (geojson, log_path)
        )
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES", "auto")
    return SpaceRangerSegmentRun(
        run_id=run_id,
        command=tuple(command),
        executable=binary,
        spaceranger_version=version,
        tissue_image=image,
        tissue_image_sha256=_sha256(image),
        run_directory=run_directory,
        geojson_path=geojson,
        log_path=log_path,
        cuda_visible_devices=visible_devices,
    )


def spaceranger_executable_version(executable: PathLike) -> str:
    """Query and normalize a Space Ranger 4.x executable version."""

    binary = _validated_executable(Path(executable).expanduser())
    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("could not query Space Ranger version from %s" % binary)
    return normalize_spaceranger_version((result.stdout or "") + " " + (result.stderr or ""))


def normalize_spaceranger_version(value: str) -> str:
    """Extract a supported numeric 4.x version from CLI or user input."""

    match = _VERSION.search(str(value).strip())
    if match is None:
        raise ValueError("Space Ranger version must identify a supported 4.x release")
    return match.group(1)


def _polygon_vertices(raw: object, feature_index: int) -> np.ndarray:
    try:
        polygon = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "GeoJSON feature %d has non-numeric polygon coordinates" % feature_index
        ) from error
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4:
        raise ValueError("GeoJSON feature %d polygon needs at least three vertices" % feature_index)
    if not np.isfinite(polygon).all():
        raise ValueError(
            "GeoJSON feature %d polygon contains non-finite coordinates" % feature_index
        )
    if np.allclose(polygon[0], polygon[-1], rtol=0.0, atol=1.0e-9):
        polygon = polygon[:-1]
    if len(polygon) < 3 or len(np.unique(polygon, axis=0)) < 3:
        raise ValueError(
            "GeoJSON feature %d polygon has fewer than three unique vertices" % feature_index
        )
    return polygon


def _polygon_descriptors(polygon: np.ndarray) -> Optional[Tuple[float, ...]]:
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    signed_area = 0.5 * float(cross.sum())
    area = abs(signed_area)
    perimeter = float(np.linalg.norm(following - polygon, axis=1).sum())
    if area <= 1.0e-12 or perimeter <= 1.0e-12:
        return None
    centroid = _polygon_centroid(polygon, signed_area=signed_area, cross=cross)
    centered = polygon - centroid
    covariance = centered.T @ centered / len(centered)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1.0e-12)
    major = 4.0 * np.sqrt(eigenvalues[1])
    minor = 4.0 * np.sqrt(eigenvalues[0])
    eccentricity = np.sqrt(max(0.0, 1.0 - eigenvalues[0] / eigenvalues[1]))
    mxx, myy, mxy = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    orientation = 0.5 * np.arctan2(2.0 * mxy, mxx - myy)
    circularity = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
    width, height = np.ptp(polygon, axis=0)
    bounding_area = width * height
    extent = min(1.0, area / bounding_area) if bounding_area > 0.0 else 0.0
    hull_area = _convex_hull_area(polygon)
    solidity = min(1.0, area / hull_area) if hull_area > 0.0 else 0.0
    equivalent_diameter = np.sqrt(4.0 * area / np.pi)
    return tuple(
        float(value)
        for value in (
            area,
            perimeter,
            circularity,
            eccentricity,
            major,
            minor,
            orientation,
            solidity,
            extent,
            equivalent_diameter,
        )
    )


def _polygon_centroid(
    polygon: np.ndarray,
    *,
    signed_area: Optional[float] = None,
    cross: Optional[np.ndarray] = None,
) -> np.ndarray:
    following = np.roll(polygon, -1, axis=0)
    if cross is None:
        cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    if signed_area is None:
        signed_area = 0.5 * float(cross.sum())
    if abs(signed_area) <= 1.0e-12:
        return polygon.mean(axis=0)
    return ((polygon + following) * cross[:, None]).sum(axis=0) / (6.0 * signed_area)


def _feature_centroid(
    properties: Mapping[object, object], polygon: np.ndarray, feature_index: int
) -> Tuple[float, float]:
    raw = properties.get("nucleus_centroid")
    if raw is None:
        centroid = _polygon_centroid(polygon)
    else:
        try:
            centroid = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "GeoJSON feature %d has a non-numeric centroid" % feature_index
            ) from error
        if centroid.shape != (2,) or not np.isfinite(centroid).all():
            raise ValueError(
                "GeoJSON feature %d centroid must contain two finite values" % feature_index
            )
    minimum = polygon.min(axis=0) - 1.0e-6
    maximum = polygon.max(axis=0) + 1.0e-6
    if np.any(centroid < minimum) or np.any(centroid > maximum):
        raise ValueError(
            "GeoJSON feature %d centroid lies outside its polygon bounds" % feature_index
        )
    return float(centroid[0]), float(centroid[1])


def _convex_hull_area(points: np.ndarray) -> float:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        return 0.0

    def cross(
        origin: Tuple[float, float], left: Tuple[float, float], right: Tuple[float, float]
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
        )

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    following = np.roll(hull, -1, axis=0)
    return 0.5 * abs(float(np.sum(hull[:, 0] * following[:, 1] - following[:, 0] * hull[:, 1])))


def _write_segmentation_csv(segmentation: SpaceRangerSegmentation, path: Path, suffix: str) -> None:
    delimiter = "\t" if suffix == ".tsv" else ","
    fields = [
        "nucleus_id",
        "source_id",
        "x",
        "y",
        "confidence",
        *MORPHOLOGY_FEATURE_NAMES,
        "segmentation_method",
        "segmentation_version",
        "segmentation_source_name",
        "segmentation_source_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for index in range(len(segmentation)):
            row: Dict[str, object] = {
                "nucleus_id": segmentation.nucleus_ids[index],
                "source_id": segmentation.source_ids[index],
                "x": "%.12g" % segmentation.centroids_px[index, 0],
                "y": "%.12g" % segmentation.centroids_px[index, 1],
                "confidence": "",
                "segmentation_method": segmentation.method,
                "segmentation_version": segmentation.spaceranger_version,
                "segmentation_source_name": segmentation.source_name,
                "segmentation_source_sha256": segmentation.source_sha256,
            }
            row.update(
                {
                    name: "%.12g" % segmentation.morphology[index, column]
                    for column, name in enumerate(MORPHOLOGY_FEATURE_NAMES)
                }
            )
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _write_segmentation_npz(segmentation: SpaceRangerSegmentation, path: Path) -> None:
    median = np.median(segmentation.morphology, axis=0, keepdims=True)
    mad = np.median(np.abs(segmentation.morphology - median), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1.0e-6)
    features = ((segmentation.morphology - median) / scale).astype(np.float32)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(SEGMENTATION_SCHEMA, dtype=np.dtype("U")),
            nucleus_ids=segmentation.nucleus_ids,
            source_ids=segmentation.source_ids,
            features=features,
            morphology=segmentation.morphology,
            feature_names=np.asarray(MORPHOLOGY_FEATURE_NAMES, dtype=np.dtype("U")),
            coordinates=segmentation.centroids_px,
            segmentation_confidence=np.full(len(segmentation), np.nan, dtype=np.float32),
            feature_transform=np.asarray(FEATURE_TRANSFORM, dtype=np.dtype("U")),
            feature_median=median.reshape(-1).astype(np.float32),
            feature_scale=scale.reshape(-1).astype(np.float32),
            segmentation_method=np.asarray(segmentation.method, dtype=np.dtype("U")),
            segmentation_version=np.asarray(segmentation.spaceranger_version, dtype=np.dtype("U")),
            segmentation_source_name=np.asarray(segmentation.source_name, dtype=np.dtype("U")),
            segmentation_source_sha256=np.asarray(segmentation.source_sha256, dtype=np.dtype("U")),
            skipped_features=np.asarray(segmentation.skipped_features, dtype=np.int64),
        )
        handle.flush()
        os.fsync(handle.fileno())


def _installation_candidates(path: Path) -> Sequence[Path]:
    if path.is_dir():
        return (path / "spaceranger", path / "bin" / "spaceranger")
    return (path,)


def _first_valid_installation(candidates: Sequence[Path], setting: str) -> Path:
    for candidate in candidates:
        try:
            return _validated_executable(candidate)
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    raise FileNotFoundError("%s does not identify an executable Space Ranger binary" % setting)


def _validated_executable(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    if not os.access(resolved, os.X_OK):
        raise PermissionError("Space Ranger is not executable: %s" % resolved)
    if resolved.name != "spaceranger":
        raise ValueError("Space Ranger executable must be named spaceranger")
    return resolved


def _executable_version_key(path: Path) -> Tuple[int, ...]:
    match = re.search(r"spaceranger-(4(?:\.\d+){0,2})", str(path))
    if match is None:
        return (4, 0, 0)
    parts = tuple(int(part) for part in match.group(1).split("."))
    return parts + (0,) * (3 - len(parts))


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


def _temporary_path(target: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".%s." % target.name,
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(descriptor)
    return Path(raw_path)


def _commit_temp(temporary: Path, target: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, target)
        return
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise FileExistsError(str(target)) from None
    temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_tail(path: Path, maximum_bytes: int = 4000) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace").strip()


def _string_array(values: Sequence[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype="<U%d" % width)


def _repeated_string(value: str, count: int) -> np.ndarray:
    return np.full(count, value, dtype="<U%d" % max(1, len(value)))


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result
