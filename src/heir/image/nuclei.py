"""Canonical nucleus tables, feature bundles, and Visium spot aggregation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


ID_ALIASES = ("nucleus_id", "cell_id", "object_id", "id")
X_ALIASES = ("x", "x_px", "centroid_x", "center_x", "pixel_x")
Y_ALIASES = ("y", "y_px", "centroid_y", "center_y", "pixel_y")
CONFIDENCE_ALIASES = (
    "confidence",
    "score",
    "probability",
    "cell_type_probability",
    "classification_probability",
)
CELL_TYPE_ALIASES = ("cell_type", "nucleus_type", "class", "label")
KNOWN_MORPHOLOGY = {
    "area",
    "area_px",
    "perimeter",
    "perimeter_px",
    "eccentricity",
    "solidity",
    "circularity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "extent",
    "equiv_diameter",
    "equivalent_diameter",
    "equivalent_diameter_px",
    "mean_intensity",
    "hematoxylin_mean",
    "eosin_mean",
}


def _string_array(values: Iterable[object]) -> np.ndarray:
    sequence = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]
    width = max((len(value) for value in sequence), default=1)
    return np.asarray(sequence, dtype="<U%d" % width)


def canonical_nucleus_ids(
    source_ids: Sequence[object],
    sample_id: Optional[str] = None,
) -> np.ndarray:
    """Create stable, unique nucleus identifiers.

    When ``sample_id`` is supplied, source identifiers are namespaced as
    ``sample_id::source_id`` to prevent collisions after concatenating slides.
    """

    raw = [str(value).strip() for value in source_ids]
    if any(not value for value in raw):
        raise ValueError("source nucleus IDs cannot be empty")
    prefix = None if sample_id is None else str(sample_id).strip()
    if prefix == "":
        raise ValueError("sample_id cannot be empty")
    canonical = raw if prefix is None else ["%s::%s" % (prefix, value) for value in raw]
    if len(set(canonical)) != len(canonical):
        raise ValueError("nucleus IDs must be unique within a sample")
    return _string_array(canonical)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NucleusTable:
    """Validated canonical nucleus records independent of pandas."""

    nucleus_ids: np.ndarray
    centroids_px: np.ndarray
    morphology: np.ndarray
    morphology_names: Tuple[str, ...]
    confidence: np.ndarray
    cell_types: np.ndarray
    source_ids: np.ndarray
    metadata: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = _string_array(self.nucleus_ids)
        source_ids = _string_array(self.source_ids)
        centroids = np.asarray(self.centroids_px, dtype=np.float64)
        morphology = np.asarray(self.morphology, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        cell_types = _string_array(self.cell_types)
        count = len(ids)
        if len(set(ids.tolist())) != count:
            raise ValueError("nucleus_ids must be unique")
        if source_ids.shape != (count,):
            raise ValueError("source_ids must have shape (nuclei,)")
        if centroids.shape != (count, 2) or not np.isfinite(centroids).all():
            raise ValueError("centroids_px must be finite with shape (nuclei, 2)")
        if morphology.ndim != 2 or morphology.shape[0] != count:
            raise ValueError("morphology must have shape (nuclei, features)")
        if morphology.shape[1] != len(self.morphology_names):
            raise ValueError("morphology_names must match morphology columns")
        if morphology.size and not np.isfinite(morphology).all():
            raise ValueError("morphology features must be finite")
        if confidence.shape != (count,):
            raise ValueError("confidence must have shape (nuclei,)")
        finite_confidence = confidence[np.isfinite(confidence)]
        if finite_confidence.size and (
            bool((finite_confidence < 0.0).any()) or bool((finite_confidence > 1.0).any())
        ):
            raise ValueError("finite confidence values must be in [0, 1]")
        if cell_types.shape != (count,):
            raise ValueError("cell_types must have shape (nuclei,)")
        normalized_metadata: Dict[str, np.ndarray] = {}
        for name, values in self.metadata.items():
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != count:
                raise ValueError("metadata column %s must have one value per nucleus" % name)
            normalized_metadata[str(name)] = _readonly(array)
        object.__setattr__(self, "nucleus_ids", _readonly(ids))
        object.__setattr__(self, "source_ids", _readonly(source_ids))
        object.__setattr__(self, "centroids_px", _readonly(centroids))
        object.__setattr__(self, "morphology", _readonly(morphology))
        object.__setattr__(self, "confidence", _readonly(confidence))
        object.__setattr__(self, "cell_types", _readonly(cell_types))
        object.__setattr__(self, "metadata", normalized_metadata)

    def __len__(self) -> int:
        return int(self.nucleus_ids.shape[0])

    @property
    def ids(self) -> np.ndarray:
        return self.nucleus_ids

    @property
    def centroids(self) -> np.ndarray:
        return self.centroids_px

    @property
    def has_confidence(self) -> bool:
        return bool(np.isfinite(self.confidence).any())

    @classmethod
    def from_csv(cls, path: PathLike, **kwargs: Any) -> "NucleusTable":
        """Class-level convenience wrapper around :func:`load_nuclei`."""

        return load_nuclei(path, **kwargs)


def _resolve_column(
    fieldnames: Sequence[str],
    explicit: Optional[str],
    aliases: Sequence[str],
    name: str,
    required: bool,
) -> Optional[str]:
    lookup = {_column_key(column): column for column in fieldnames}
    if explicit is not None:
        key = _column_key(explicit)
        if key not in lookup:
            raise ValueError("%s column %r was not found" % (name, explicit))
        return lookup[key]
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    if required:
        raise ValueError("could not identify the %s column" % name)
    return None


def _delimiter(path: Path, explicit: Optional[str]) -> str:
    if explicit is not None:
        if len(explicit) != 1:
            raise ValueError("delimiter must be one character")
        return explicit
    if path.suffix.lower() in (".tsv", ".tab"):
        return "\t"
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def load_nuclei(
    path: PathLike,
    sample_id: Optional[str] = None,
    delimiter: Optional[str] = None,
    id_column: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    morphology_columns: Optional[Sequence[str]] = None,
    confidence_column: Optional[str] = None,
    cell_type_column: Optional[str] = None,
) -> NucleusTable:
    """Load a CSV/TSV nucleus table into the canonical representation."""

    table_path = Path(path).expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(str(table_path))
    selected_delimiter = _delimiter(table_path, delimiter)
    with table_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=selected_delimiter)
        if not reader.fieldnames:
            raise ValueError("nucleus table has no header")
        fieldnames = tuple(reader.fieldnames)
        id_name = _resolve_column(fieldnames, id_column, ID_ALIASES, "ID", False)
        x_name = _resolve_column(fieldnames, x_column, X_ALIASES, "x centroid", True)
        y_name = _resolve_column(fieldnames, y_column, Y_ALIASES, "y centroid", True)
        confidence_name = _resolve_column(
            fieldnames,
            confidence_column,
            CONFIDENCE_ALIASES,
            "confidence",
            False,
        )
        cell_type_name = _resolve_column(
            fieldnames,
            cell_type_column,
            CELL_TYPE_ALIASES,
            "cell type",
            False,
        )
        lookup = {_column_key(column): column for column in fieldnames}
        if morphology_columns is None:
            morphology_names = tuple(
                column
                for column in fieldnames
                if _column_key(column) in KNOWN_MORPHOLOGY
                or _column_key(column).startswith("morph_")
            )
        else:
            resolved = []
            for requested in morphology_columns:
                key = _column_key(requested)
                if key not in lookup:
                    raise ValueError("morphology column %r was not found" % requested)
                resolved.append(lookup[key])
            morphology_names = tuple(resolved)
        source_ids = []
        centroids = []
        morphology = []
        confidence = []
        cell_types = []
        used = {name for name in (id_name, x_name, y_name, confidence_name, cell_type_name) if name}
        used.update(morphology_names)
        metadata_columns = [column for column in fieldnames if column not in used]
        metadata: Dict[str, list] = {column: [] for column in metadata_columns}
        # Parse the reader as a stream.  This avoids retaining a second copy of
        # every CSV dictionary for slides containing millions of nuclei.
        for row_number, row in enumerate(reader, start=1):
            raw_id = row.get(id_name, "") if id_name else ""
            source_ids.append(str(raw_id).strip() or "nucleus_%08d" % (row_number - 1))
            try:
                x_value = float(row[x_name])  # type: ignore[index]
                y_value = float(row[y_name])  # type: ignore[index]
            except (TypeError, ValueError) as error:
                raise ValueError("row %d has an invalid centroid" % row_number) from error
            centroids.append((x_value, y_value))
            feature_row = []
            for column in morphology_names:
                try:
                    feature_row.append(float(row[column]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "row %d has an invalid morphology value in %s" % (row_number, column)
                    ) from error
            morphology.append(feature_row)
            raw_confidence = row.get(confidence_name, "") if confidence_name else ""
            if raw_confidence is None or not str(raw_confidence).strip():
                confidence.append(np.nan)
            else:
                try:
                    confidence.append(float(raw_confidence))
                except ValueError as error:
                    raise ValueError("row %d has an invalid confidence" % row_number) from error
            cell_types.append(str(row.get(cell_type_name, "") or "") if cell_type_name else "")
            for column in metadata_columns:
                metadata[column].append(row.get(column, ""))

    canonical = canonical_nucleus_ids(source_ids, sample_id=sample_id)
    count = len(source_ids)
    centroid_array = np.asarray(centroids, dtype=np.float64).reshape(count, 2)
    morphology_array = np.asarray(morphology, dtype=np.float32).reshape(
        count, len(morphology_names)
    )
    return NucleusTable(
        nucleus_ids=canonical,
        centroids_px=centroid_array,
        morphology=morphology_array,
        morphology_names=tuple(str(name) for name in morphology_names),
        confidence=np.asarray(confidence, dtype=np.float32),
        cell_types=_string_array(cell_types),
        source_ids=_string_array(source_ids),
        metadata={name: _string_array(values) for name, values in metadata.items()},
    )


load_nucleus_table = load_nuclei


@dataclass(frozen=True)
class FeatureBundle:
    """Precomputed per-nucleus feature matrix loaded from a safe NPZ file."""

    nucleus_ids: np.ndarray
    features: np.ndarray
    feature_names: Tuple[str, ...]
    coordinates: Optional[np.ndarray] = None
    metadata: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = _string_array(self.nucleus_ids)
        features = np.asarray(self.features)
        if features.ndim != 2 or features.shape[0] != len(ids):
            raise ValueError("features must have shape (nuclei, features)")
        if not np.issubdtype(features.dtype, np.number):
            raise TypeError("features must be numeric")
        features = features.astype(np.float32, copy=False)
        if features.size and not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("feature bundle nucleus IDs must be unique")
        if len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names must match the feature width")
        coordinates = None
        if self.coordinates is not None:
            coordinates = np.asarray(self.coordinates, dtype=np.float64)
            if coordinates.shape != (len(ids), 2) or not np.isfinite(coordinates).all():
                raise ValueError("coordinates must be finite with shape (nuclei, 2)")
            coordinates = _readonly(coordinates)
        normalized_metadata = {}
        for name, values in self.metadata.items():
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != len(ids):
                raise ValueError("feature metadata %s must have one row per nucleus" % name)
            normalized_metadata[str(name)] = _readonly(array)
        object.__setattr__(self, "nucleus_ids", _readonly(ids))
        object.__setattr__(self, "features", _readonly(features))
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "metadata", normalized_metadata)

    def __len__(self) -> int:
        return int(self.nucleus_ids.shape[0])

    def align(self, expected_ids: Sequence[object]) -> "FeatureBundle":
        """Reorder the bundle to canonical IDs, rejecting missing or extra rows."""

        expected = _string_array(expected_ids)
        if len(set(expected.tolist())) != len(expected):
            raise ValueError("expected_ids must be unique")
        index = {value: position for position, value in enumerate(self.nucleus_ids.tolist())}
        missing = [value for value in expected.tolist() if value not in index]
        extras = sorted(set(index).difference(expected.tolist()))
        if missing or extras:
            raise ValueError(
                "feature IDs do not match expected IDs (missing=%d, extra=%d)"
                % (len(missing), len(extras))
            )
        order = np.asarray([index[value] for value in expected.tolist()], dtype=np.int64)
        return FeatureBundle(
            nucleus_ids=expected,
            features=self.features[order],
            feature_names=self.feature_names,
            coordinates=None if self.coordinates is None else self.coordinates[order],
            metadata={name: values[order] for name, values in self.metadata.items()},
        )

    @classmethod
    def load(
        cls,
        path: PathLike,
        expected_ids: Optional[Sequence[object]] = None,
    ) -> "FeatureBundle":
        """Class-level convenience wrapper around :func:`load_feature_bundle`."""

        return load_feature_bundle(path, expected_ids=expected_ids)


def _npz_key(archive: Mapping[str, np.ndarray], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in archive:
            return key
    return None


def load_feature_bundle(
    path: PathLike,
    expected_ids: Optional[Sequence[object]] = None,
) -> FeatureBundle:
    """Load and validate a non-pickled NPZ feature bundle.

    Accepted feature keys are ``features``, ``embeddings``, or ``X``; accepted
    identifier keys are ``nucleus_ids``, ``ids``, or ``cell_ids``.
    """

    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(str(bundle_path))
    try:
        with np.load(bundle_path, allow_pickle=False) as archive:
            feature_key = _npz_key(archive, ("features", "embeddings", "X"))
            id_key = _npz_key(archive, ("nucleus_ids", "ids", "cell_ids"))
            if feature_key is None or id_key is None:
                raise ValueError("feature bundle requires feature and nucleus ID arrays")
            features = np.array(archive[feature_key], copy=True)
            ids = np.array(archive[id_key], copy=True)
            if features.ndim != 2:
                raise ValueError("feature bundle features must have shape (nuclei, features)")
            name_key = _npz_key(archive, ("feature_names", "columns"))
            if name_key is None:
                names = tuple("feature_%04d" % index for index in range(features.shape[1]))
            else:
                names = tuple(_string_array(np.asarray(archive[name_key]).tolist()).tolist())
            coordinate_key = _npz_key(archive, ("coordinates", "centroids", "centroids_px"))
            coordinates = (
                None if coordinate_key is None else np.array(archive[coordinate_key], copy=True)
            )
            consumed = {feature_key, id_key, name_key, coordinate_key}
            metadata = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
                if key not in consumed
                and np.asarray(archive[key]).ndim > 0
                and np.asarray(archive[key]).shape[0] == features.shape[0]
            }
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError("feature bundle may not contain pickled object arrays") from error
        raise
    bundle = FeatureBundle(ids, features, names, coordinates, metadata)
    return bundle if expected_ids is None else bundle.align(expected_ids)


def _spatial_tree():
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Visium assignment requires scipy; install the spatial dependencies"
        ) from error
    return cKDTree


def _xy(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("%s must have shape (items, 2)" % name)
    if not np.isfinite(result).all():
        raise ValueError("%s must contain only finite coordinates" % name)
    return result


@dataclass(frozen=True)
class SpotAssignment:
    """Nearest valid Visium spot for every nucleus; ``-1`` means unassigned."""

    spot_index: np.ndarray
    distance: np.ndarray
    spot_ids: np.ndarray

    def __post_init__(self) -> None:
        index = np.asarray(self.spot_index, dtype=np.int64)
        distance = np.asarray(self.distance, dtype=np.float64)
        spot_ids = _string_array(self.spot_ids)
        if index.ndim != 1 or distance.shape != index.shape:
            raise ValueError("spot_index and distance must be aligned vectors")
        if bool((index < -1).any()) or (index.size and int(index.max()) >= len(spot_ids)):
            raise ValueError("spot_index contains an invalid spot")
        assigned = index >= 0
        if assigned.any() and not np.isfinite(distance[assigned]).all():
            raise ValueError("assigned distances must be finite")
        if bool((distance[assigned] < 0.0).any()):
            raise ValueError("assigned distances cannot be negative")
        object.__setattr__(self, "spot_index", _readonly(index))
        object.__setattr__(self, "distance", _readonly(distance))
        object.__setattr__(self, "spot_ids", _readonly(spot_ids))

    @property
    def assigned(self) -> np.ndarray:
        return self.spot_index >= 0

    @property
    def assigned_count(self) -> int:
        return int(self.assigned.sum())

    @property
    def unassigned_count(self) -> int:
        return int((~self.assigned).sum())


def assign_nuclei_to_visium_spots(
    nucleus_coordinates: np.ndarray,
    spot_coordinates: np.ndarray,
    spot_radius: Union[float, Sequence[float]] = 27.5,
    spot_ids: Optional[Sequence[object]] = None,
) -> SpotAssignment:
    """Assign each nucleus to the nearest spot whose physical disk contains it."""

    nuclei = _xy(nucleus_coordinates, "nucleus_coordinates")
    spots = _xy(spot_coordinates, "spot_coordinates")
    identifiers = _string_array(range(len(spots))) if spot_ids is None else _string_array(spot_ids)
    if identifiers.shape != (len(spots),) or len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("spot_ids must be unique with one ID per spot")
    radii = np.asarray(spot_radius, dtype=np.float64)
    scalar_radius = radii.ndim == 0
    if scalar_radius:
        scalar_value = float(radii)
        if not np.isfinite(scalar_value) or scalar_value <= 0.0:
            raise ValueError("spot radii must be finite and positive")
        radii = np.full(len(spots), scalar_value, dtype=np.float64)
    if radii.shape != (len(spots),):
        raise ValueError("spot_radius must be scalar or have one value per spot")
    if not np.isfinite(radii).all() or bool((radii <= 0.0).any()):
        raise ValueError("spot radii must be finite and positive")
    assignment = np.full(len(nuclei), -1, dtype=np.int64)
    distance = np.full(len(nuclei), np.inf, dtype=np.float64)
    if len(nuclei) == 0 or len(spots) == 0:
        return SpotAssignment(assignment, distance, identifiers)
    tree = _spatial_tree()(spots)
    if scalar_radius:
        nearest_distance, nearest = tree.query(
            nuclei,
            k=1,
            distance_upper_bound=float(radii[0]),
        )
        valid = np.asarray(nearest) < len(spots)
        assignment[valid] = np.asarray(nearest, dtype=np.int64)[valid]
        distance[valid] = np.asarray(nearest_distance, dtype=np.float64)[valid]
    else:
        candidate_lists = tree.query_ball_point(nuclei, r=float(radii.max()))
        for nucleus_index, candidates in enumerate(candidate_lists):
            eligible = []
            for candidate in candidates:
                candidate = int(candidate)
                candidate_distance = float(np.linalg.norm(nuclei[nucleus_index] - spots[candidate]))
                if candidate_distance <= radii[candidate]:
                    eligible.append((candidate_distance, candidate))
            if eligible:
                selected_distance, selected = min(eligible, key=lambda item: (item[0], item[1]))
                assignment[nucleus_index] = selected
                distance[nucleus_index] = selected_distance
    return SpotAssignment(assignment, distance, identifiers)


assign_nuclei_to_spots = assign_nuclei_to_visium_spots


class ConservationError(RuntimeError):
    """Raised when spot aggregation does not conserve assigned input mass."""


@dataclass(frozen=True)
class SpotAggregation:
    values: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    weight_sums: np.ndarray
    assigned_count: int
    unassigned_count: int
    spot_ids: np.ndarray
    reduction: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        sums = np.asarray(self.sums)
        counts = np.asarray(self.counts, dtype=np.int64)
        weight_sums = np.asarray(self.weight_sums, dtype=np.float64)
        identifiers = _string_array(self.spot_ids)
        spot_count = len(identifiers)
        if values.shape[0] != spot_count or sums.shape != values.shape:
            raise ValueError("spot values and sums must align to spot_ids")
        if counts.shape != (spot_count,) or weight_sums.shape != (spot_count,):
            raise ValueError("spot counts and weights must align to spot_ids")
        if self.assigned_count < 0 or self.unassigned_count < 0:
            raise ValueError("assigned and unassigned counts cannot be negative")
        if int(counts.sum()) != self.assigned_count:
            raise ConservationError("spot counts do not match assigned_count")
        if len(set(identifiers.tolist())) != spot_count:
            raise ValueError("spot_ids must be unique")
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "sums", _readonly(sums))
        object.__setattr__(self, "counts", _readonly(counts))
        object.__setattr__(self, "weight_sums", _readonly(weight_sums))
        object.__setattr__(self, "spot_ids", _readonly(identifiers))


def check_spot_conservation(
    assigned_input_sum: np.ndarray,
    aggregated_sum: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Raise :class:`ConservationError` when feature mass changes."""

    expected = np.asarray(assigned_input_sum, dtype=np.float64)
    observed = np.asarray(aggregated_sum, dtype=np.float64)
    if expected.shape != observed.shape or not np.allclose(
        expected, observed, rtol=rtol, atol=atol, equal_nan=False
    ):
        maximum_error = np.inf
        if expected.shape == observed.shape and expected.size:
            maximum_error = float(np.max(np.abs(expected - observed)))
        raise ConservationError(
            "spot aggregation failed conservation (max error=%s)" % maximum_error
        )


def aggregate_nuclei_to_spots(
    values: np.ndarray,
    assignment: Union[SpotAssignment, Sequence[int], np.ndarray],
    num_spots: Optional[int] = None,
    reduction: str = "sum",
    weights: Optional[Sequence[float]] = None,
    spot_ids: Optional[Sequence[object]] = None,
    check_conservation: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> SpotAggregation:
    """Aggregate per-nucleus values with cell-count and mass conservation checks."""

    data = np.asarray(values)
    squeeze = data.ndim == 1
    if squeeze:
        data = data[:, None]
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("values must be a numeric vector or matrix")
    data = data.astype(np.float64, copy=False)
    if not np.isfinite(data).all():
        raise ValueError("values must contain only finite values")
    if isinstance(assignment, SpotAssignment):
        indices = assignment.spot_index
        inferred_ids = assignment.spot_ids
        inferred_spots = len(inferred_ids)
    else:
        raw_indices = np.asarray(assignment)
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise TypeError("assignment indices must be integers")
        indices = raw_indices.astype(np.int64, copy=False)
        inferred_ids = None
        inferred_spots = int(indices[indices >= 0].max()) + 1 if bool((indices >= 0).any()) else 0
    if indices.shape != (data.shape[0],):
        raise ValueError("assignment must contain one index per nucleus")
    if bool((indices < -1).any()):
        raise ValueError("assignment indices cannot be below -1")
    if num_spots is not None and (int(num_spots) != num_spots):
        raise ValueError("num_spots must be an integer")
    total_spots = inferred_spots if num_spots is None else int(num_spots)
    if total_spots < inferred_spots or total_spots < 0:
        raise ValueError("num_spots is smaller than an assigned spot index")
    if spot_ids is not None:
        identifiers = _string_array(spot_ids)
    elif inferred_ids is not None and len(inferred_ids) == total_spots:
        identifiers = inferred_ids
    else:
        identifiers = _string_array(range(total_spots))
    if identifiers.shape != (total_spots,):
        raise ValueError("spot_ids must contain one ID per output spot")
    if len(set(identifiers.tolist())) != total_spots:
        raise ValueError("spot_ids must be unique")
    selected_reduction = reduction.strip().lower()
    if selected_reduction not in ("sum", "mean", "weighted_mean"):
        raise ValueError("reduction must be 'sum', 'mean', or 'weighted_mean'")
    if weights is None:
        cell_weights = np.ones(data.shape[0], dtype=np.float64)
    else:
        cell_weights = np.asarray(weights, dtype=np.float64)
        if cell_weights.shape != (data.shape[0],):
            raise ValueError("weights must contain one value per nucleus")
        if not np.isfinite(cell_weights).all() or bool((cell_weights < 0.0).any()):
            raise ValueError("weights must be finite and nonnegative")
    assigned = indices >= 0
    counts = np.bincount(indices[assigned], minlength=total_spots).astype(np.int64)
    if int(counts.sum()) != int(assigned.sum()):
        raise ConservationError("spot counts do not conserve assigned nuclei")
    weight_sums = np.zeros(total_spots, dtype=np.float64)
    np.add.at(weight_sums, indices[assigned], cell_weights[assigned])
    sums = np.zeros((total_spots, data.shape[1]), dtype=np.float64)
    effective = data * cell_weights[:, None]
    np.add.at(sums, indices[assigned], effective[assigned])
    if check_conservation:
        check_spot_conservation(
            effective[assigned].sum(axis=0),
            sums.sum(axis=0),
            rtol=rtol,
            atol=atol,
        )
    if selected_reduction == "sum":
        aggregated = sums.copy()
    else:
        aggregated = np.zeros_like(sums)
        valid = weight_sums > 0.0
        aggregated[valid] = sums[valid] / weight_sums[valid, None]
    if squeeze:
        aggregated = aggregated[:, 0]
        sums = sums[:, 0]
    return SpotAggregation(
        values=aggregated,
        sums=sums,
        counts=counts,
        weight_sums=weight_sums,
        assigned_count=int(assigned.sum()),
        unassigned_count=int((~assigned).sum()),
        spot_ids=identifiers,
        reduction=selected_reduction,
    )


aggregate_to_spots = aggregate_nuclei_to_spots


__all__ = [
    "NucleusTable",
    "canonical_nucleus_ids",
    "load_nuclei",
    "load_nucleus_table",
    "FeatureBundle",
    "load_feature_bundle",
    "SpotAssignment",
    "assign_nuclei_to_visium_spots",
    "assign_nuclei_to_spots",
    "ConservationError",
    "SpotAggregation",
    "check_spot_conservation",
    "aggregate_nuclei_to_spots",
    "aggregate_to_spots",
]
