"""Immutable multimodal cohort manifests and scientific leakage checks.

The manifest is the boundary between public files and HEIR experiments.  A
record describes one assay or one section and carries enough provenance to
decide whether two records are genuinely matched.  Loading is deliberately
strict: misspelled columns, donor/block leakage, and spatial data assigned to
training roles are errors rather than warnings.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, os.PathLike]


# The first 33 fields follow the blueprint verbatim.  Python uses ``he_file``
# internally because ``H&E_file`` is not a valid identifier.
MANIFEST_COLUMNS: Tuple[str, ...] = (
    "cohort_id",
    "donor_id",
    "specimen_id",
    "block_id",
    "section_id",
    "section_order",
    "distance_from_reference_section_um",
    "modality",
    "assay_platform",
    "preservation",
    "tissue",
    "disease",
    "anatomic_region",
    "H&E_file",
    "count_matrix_file",
    "spatial_coordinate_file",
    "annotation_file",
    "scanner",
    "magnification",
    "native_mpp",
    "stain_batch",
    "sequencing_chemistry",
    "genome_build",
    "gene_annotation_version",
    "matching_tier",
    "matching_notes",
    "public_accession",
    "license",
    "checksum",
    "analysis_role",
    "outer_fold",
    "inner_fold",
    # HEIR extensions needed to represent composite public sections without
    # overloading the snRNA count matrix with held-out spatial expression.
    "spatial_count_matrix_file",
    "included",
    "exclusion_reason",
    "donor_filter_key",
    "donor_filter_value",
    "sample_filter_key",
    "sample_filter_value",
)


PATH_FIELDS: Tuple[str, ...] = (
    "he_file",
    "count_matrix_file",
    "spatial_count_matrix_file",
    "spatial_coordinate_file",
    "annotation_file",
)

MATCHING_TIERS = frozenset({"tier_1", "tier_2", "tier_3", "tier_4"})
ANALYSIS_ROLES = frozenset(
    {
        "train",
        "training",
        "development",
        "pretraining",
        "personalized_input",
        "validation",
        "spatial_validation",
        "external_validation",
        "test",
        "locked_test",
        "locked_validation",
        "excluded",
    }
)
SPATIAL_VALIDATION_ROLES = frozenset(
    {
        "validation",
        "spatial_validation",
        "external_validation",
        "test",
        "locked_test",
        "locked_validation",
        "excluded",
    }
)
# Public spatial expression may supervise only the explicitly decontaminated
# generic-pretraining stage. It remains prohibited from development,
# personalized, and refinement roles.
SPATIAL_ALLOWED_ROLES = SPATIAL_VALIDATION_ROLES | frozenset({"pretraining"})


class ManifestValidationError(ValueError):
    """Raised when a manifest could invalidate a scientific evaluation."""


def _clean_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


_ALIASES: Dict[str, str] = {_clean_key(column): column for column in MANIFEST_COLUMNS}
_ALIASES.update(
    {
        "h_e_file": "H&E_file",
        "he_file": "H&E_file",
        "h_and_e_file": "H&E_file",
        "histology_file": "H&E_file",
        "role": "analysis_role",
        "match_tier": "matching_tier",
        "notes": "matching_notes",
        "spatial_expression_file": "spatial_count_matrix_file",
    }
)


def _canonical_column(value: str) -> Optional[str]:
    return _ALIASES.get(_clean_key(value))


def _parse_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ManifestValidationError("%s must be an integer" % field_name) from exc


def _parse_optional_float(value: Any, field_name: str) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ManifestValidationError("%s must be numeric" % field_name) from exc


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "include", "included"}:
        return True
    if normalized in {"0", "false", "no", "n", "exclude", "excluded"}:
        return False
    raise ManifestValidationError("%s must be a boolean" % field_name)


def _parse_matching_tier(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"1", "2", "3", "4"}:
        normalized = "tier_" + normalized
    if normalized.startswith("tier") and "_" not in normalized:
        normalized = normalized[:4] + "_" + normalized[4:]
    return normalized


def _parse_role(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _string(value: Any) -> str:
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class ManifestRecord:
    """One immutable public assay/section ledger entry."""

    cohort_id: str = ""
    donor_id: str = ""
    specimen_id: str = ""
    block_id: str = ""
    section_id: str = ""
    section_order: Optional[int] = None
    distance_from_reference_section_um: Optional[float] = None
    modality: str = ""
    assay_platform: str = ""
    preservation: str = ""
    tissue: str = ""
    disease: str = ""
    anatomic_region: str = ""
    he_file: str = ""
    count_matrix_file: str = ""
    spatial_coordinate_file: str = ""
    annotation_file: str = ""
    scanner: str = ""
    magnification: Optional[float] = None
    native_mpp: Optional[float] = None
    stain_batch: str = ""
    sequencing_chemistry: str = ""
    genome_build: str = ""
    gene_annotation_version: str = ""
    matching_tier: str = ""
    matching_notes: str = ""
    public_accession: str = ""
    license: str = ""
    checksum: str = ""
    analysis_role: str = ""
    outer_fold: str = ""
    inner_fold: str = ""
    spatial_count_matrix_file: str = ""
    included: bool = True
    exclusion_reason: str = ""
    donor_filter_key: str = ""
    donor_filter_value: str = ""
    sample_filter_key: str = ""
    sample_filter_value: str = ""

    @property
    def record_id(self) -> str:
        """Stable identity for duplicate detection and provenance logs."""

        return "%s::%s::%s::%s" % (
            self.cohort_id,
            self.section_id,
            self.modality,
            self.assay_platform,
        )

    @property
    def h_and_e_file(self) -> str:
        """Readable alias for the serialized ``H&E_file`` column."""

        return self.he_file

    @property
    def histology_file(self) -> str:
        return self.he_file

    @property
    def primary_file(self) -> str:
        """Return the file to which the row-level checksum applies."""

        for value in (
            self.he_file,
            self.count_matrix_file,
            self.spatial_count_matrix_file,
            self.annotation_file,
        ):
            if value:
                return value
        return ""

    @property
    def is_spatial(self) -> bool:
        if self.spatial_count_matrix_file or self.spatial_coordinate_file:
            return True
        value = (self.modality + " " + self.assay_platform).lower()
        tokens = ("spatial", "visium", "xenium", "cosmx", "merfish", "merscope")
        return any(token in value for token in tokens)

    def validate(self) -> None:
        required = {
            "cohort_id": self.cohort_id,
            "donor_id": self.donor_id,
            "specimen_id": self.specimen_id,
            "block_id": self.block_id,
            "section_id": self.section_id,
            "modality": self.modality,
            "assay_platform": self.assay_platform,
            "preservation": self.preservation,
            "tissue": self.tissue,
            "matching_tier": self.matching_tier,
            "analysis_role": self.analysis_role,
        }
        missing = sorted(name for name, value in required.items() if not value.strip())
        if missing:
            raise ManifestValidationError(
                "%s is missing required fields: %s" % (self.record_id, ", ".join(missing))
            )
        if self.matching_tier not in MATCHING_TIERS:
            raise ManifestValidationError(
                "%s has invalid matching_tier %r" % (self.record_id, self.matching_tier)
            )
        if self.analysis_role not in ANALYSIS_ROLES:
            raise ManifestValidationError(
                "%s has invalid analysis_role %r" % (self.record_id, self.analysis_role)
            )
        if self.section_order is not None and self.section_order < 0:
            raise ManifestValidationError("section_order must be non-negative")
        if (
            self.distance_from_reference_section_um is not None
            and self.distance_from_reference_section_um < 0
        ):
            raise ManifestValidationError("distance_from_reference_section_um must be non-negative")
        if self.magnification is not None and self.magnification <= 0:
            raise ManifestValidationError("magnification must be positive")
        if self.native_mpp is not None and self.native_mpp <= 0:
            raise ManifestValidationError("native_mpp must be positive")
        if not self.included and not self.exclusion_reason:
            raise ManifestValidationError(
                "%s is excluded without an exclusion_reason" % self.record_id
            )
        if not self.included and self.analysis_role != "excluded":
            raise ManifestValidationError(
                "%s must use analysis_role=excluded when included=false" % self.record_id
            )
        if self.included and self.analysis_role == "excluded":
            raise ManifestValidationError(
                "%s uses analysis_role=excluded while included=true" % self.record_id
            )
        if bool(self.donor_filter_key) != bool(self.donor_filter_value):
            raise ManifestValidationError(
                "%s donor filter key and value must be specified together" % self.record_id
            )
        if bool(self.sample_filter_key) != bool(self.sample_filter_value):
            raise ManifestValidationError(
                "%s sample filter key and value must be specified together" % self.record_id
            )
        if self.checksum:
            value = self.checksum.lower()
            if value.startswith("sha256:"):
                value = value.split(":", 1)[1]
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ManifestValidationError(
                    "%s checksum must be a SHA-256 hex digest" % self.record_id
                )

    def to_mapping(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for column in MANIFEST_COLUMNS:
            attribute = "he_file" if column == "H&E_file" else column
            value = getattr(self, attribute)
            if isinstance(value, bool):
                result[column] = "true" if value else "false"
            elif value is None:
                result[column] = ""
            else:
                result[column] = value
        return result

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        base_dir: Optional[Path] = None,
        resolve_paths: bool = True,
    ) -> "ManifestRecord":
        canonical: Dict[str, Any] = {}
        unknown: List[str] = []
        for raw_key, value in values.items():
            column = _canonical_column(str(raw_key))
            if column is None:
                unknown.append(str(raw_key))
            elif column in canonical:
                raise ManifestValidationError("duplicate manifest column %s" % column)
            else:
                canonical[column] = value
        if unknown:
            raise ManifestValidationError(
                "unknown manifest columns: %s" % ", ".join(sorted(unknown))
            )

        data: Dict[str, Any] = {}
        valid_fields = {item.name for item in fields(cls)}
        for column in MANIFEST_COLUMNS:
            attribute = "he_file" if column == "H&E_file" else column
            if attribute not in valid_fields:
                raise RuntimeError("manifest column %s has no dataclass field" % column)
            raw = canonical.get(column, "")
            if attribute == "section_order":
                data[attribute] = _parse_optional_int(raw, attribute)
            elif attribute in {
                "distance_from_reference_section_um",
                "magnification",
                "native_mpp",
            }:
                data[attribute] = _parse_optional_float(raw, attribute)
            elif attribute == "included":
                data[attribute] = (
                    True if raw is None or str(raw).strip() == "" else _parse_bool(raw, attribute)
                )
            elif attribute == "matching_tier":
                data[attribute] = _parse_matching_tier(raw)
            elif attribute == "analysis_role":
                data[attribute] = _parse_role(raw)
            else:
                data[attribute] = _string(raw)

        if resolve_paths and base_dir is not None:
            for attribute in PATH_FIELDS:
                data[attribute] = resolve_manifest_path(data[attribute], base_dir)
        record = cls(**data)
        record.validate()
        return record


@dataclass(frozen=True)
class Manifest(Sequence[ManifestRecord]):
    """A tuple-backed immutable manifest."""

    records: Tuple[ManifestRecord, ...]
    source: str = ""

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not all(isinstance(record, ManifestRecord) for record in records):
            raise TypeError("Manifest records must be ManifestRecord instances")
        object.__setattr__(self, "records", records)

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: Any) -> Any:
        return self.records[index]

    @property
    def included_records(self) -> Tuple[ManifestRecord, ...]:
        return tuple(record for record in self.records if record.included)

    @property
    def donors(self) -> Tuple[str, ...]:
        return tuple(sorted({record.donor_id for record in self.included_records}))

    def by_role(self, role: str) -> Tuple[ManifestRecord, ...]:
        normalized = _parse_role(role)
        return tuple(record for record in self.records if record.analysis_role == normalized)

    def validate(self, require_folds: bool = False) -> None:
        validate_manifest(self.records, require_folds=require_folds)


def _expand_environment(value: str) -> str:
    variables = re.findall(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))", value)
    missing = sorted(
        {first or second for first, second in variables if (first or second) not in os.environ}
    )
    if missing:
        raise ManifestValidationError(
            "undefined environment variables in manifest path: %s" % ", ".join(missing)
        )
    return os.path.expandvars(value)


def resolve_manifest_path(value: str, base_dir: Path) -> str:
    """Resolve local, environment-based, and ``archive::member`` paths."""

    raw = _string(value)
    if not raw:
        return ""
    raw = _expand_environment(raw)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw):
        return raw
    member = ""
    if "::" in raw:
        raw, member = raw.split("::", 1)
        if not raw or not member:
            raise ManifestValidationError("archive paths require archive::member")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    resolved = str(path.resolve())
    return resolved + ("::" + member if member else "")


def split_filter_values(value: str) -> Tuple[str, ...]:
    """Decode a manifest filter value; ``|`` represents an OR selection."""

    return tuple(part.strip() for part in value.split("|") if part.strip())


def _role_partition(role: str) -> str:
    if role in {"train", "training", "development", "pretraining"}:
        return "train"
    if role in {"validation", "spatial_validation"}:
        return "validation"
    if role in {"external_validation", "test", "locked_test", "locked_validation"}:
        return "test"
    return ""


def _validate_group_consistency(
    records: Sequence[ManifestRecord],
    key_name: str,
    fold_name: str,
) -> None:
    grouped: Dict[Tuple[str, str], List[ManifestRecord]] = {}
    for record in records:
        if not record.included:
            continue
        key = getattr(record, key_name)
        grouped.setdefault((record.cohort_id, key), []).append(record)
    for group_key, members in grouped.items():
        folds = {getattr(record, fold_name) for record in members if getattr(record, fold_name)}
        if len(folds) > 1:
            raise ManifestValidationError(
                "%s %s crosses %s assignments: %s"
                % (key_name, group_key, fold_name, ", ".join(sorted(folds)))
            )
        partitions = {
            _role_partition(record.analysis_role)
            for record in members
            if _role_partition(record.analysis_role)
        }
        if len(partitions) > 1:
            raise ManifestValidationError(
                "%s %s crosses analysis partitions: %s"
                % (key_name, group_key, ", ".join(sorted(partitions)))
            )


def validate_no_donor_leakage(records: Sequence[ManifestRecord]) -> None:
    _validate_group_consistency(records, "donor_id", "outer_fold")
    _validate_group_consistency(records, "donor_id", "inner_fold")


def validate_no_block_leakage(records: Sequence[ManifestRecord]) -> None:
    _validate_group_consistency(records, "block_id", "outer_fold")
    _validate_group_consistency(records, "block_id", "inner_fold")


def validate_spatial_validation_isolation(records: Sequence[ManifestRecord]) -> None:
    """Permit decontaminated pretraining ST, but isolate target spatial truth."""

    spatial_files: Dict[str, str] = {}
    for record in records:
        if not record.included or not record.is_spatial:
            continue
        if record.analysis_role not in SPATIAL_ALLOWED_ROLES:
            raise ManifestValidationError(
                "%s contains spatial data outside pretraining/validation role %s"
                % (record.record_id, record.analysis_role)
            )
        for path in (record.spatial_count_matrix_file, record.spatial_coordinate_file):
            if not path:
                continue
            previous = spatial_files.setdefault(path, record.analysis_role)
            if previous != record.analysis_role:
                raise ManifestValidationError(
                    "spatial file %s is reused across roles %s and %s"
                    % (path, previous, record.analysis_role)
                )


def validate_pretraining_overlap(records: Sequence[ManifestRecord]) -> None:
    """Reject accession/file/checksum overlap between pretraining and locked data.

    Cohort names are not a safe boundary: the same public slide can appear in
    HEST, CELLxGENE, GEO, and an author archive under different names. The
    blueprint therefore requires exclusion at accession/artifact identity.
    """

    identity_roles: Dict[str, set] = {}
    for record in records:
        if not record.included:
            continue
        partition = (
            "pretraining"
            if record.analysis_role == "pretraining"
            else (
                "locked" if record.analysis_role in SPATIAL_VALIDATION_ROLES - {"excluded"} else ""
            )
        )
        if not partition:
            continue
        accessions = [
            value.strip().lower()
            for value in re.split(r"[;,|]", record.public_accession)
            if value.strip()
        ]
        paths = [
            value.split("::", 1)[0]
            for value in (
                record.he_file,
                record.count_matrix_file,
                record.spatial_count_matrix_file,
                record.spatial_coordinate_file,
            )
            if value
        ]
        identities = ["accession:" + value for value in accessions]
        identities.extend("path:" + value for value in paths)
        if record.checksum:
            identities.append("checksum:" + record.checksum.lower().removeprefix("sha256:"))
        for identity in identities:
            identity_roles.setdefault(identity, set()).add(partition)
    overlap = sorted(
        identity for identity, roles in identity_roles.items() if roles == {"pretraining", "locked"}
    )
    if overlap:
        preview = ", ".join(overlap[:5])
        suffix = " ..." if len(overlap) > 5 else ""
        raise ManifestValidationError(
            "pretraining overlaps locked evaluation identities: %s%s" % (preview, suffix)
        )


# Backwards-readable alias used in experiment guards.
validate_spatial_isolation = validate_spatial_validation_isolation


def validate_manifest(
    records: Sequence[ManifestRecord],
    require_folds: bool = False,
) -> None:
    if not records:
        raise ManifestValidationError("manifest contains no records")
    seen = set()
    for record in records:
        record.validate()
        if record.record_id in seen:
            raise ManifestValidationError("duplicate manifest record %s" % record.record_id)
        seen.add(record.record_id)
        if require_folds and record.included and not record.outer_fold:
            raise ManifestValidationError("%s has no outer_fold" % record.record_id)
    validate_no_donor_leakage(records)
    validate_no_block_leakage(records)
    validate_spatial_validation_isolation(records)
    validate_pretraining_overlap(records)


def _load_tsv(source: Path) -> List[Mapping[str, Any]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ManifestValidationError("TSV manifest has no header")
        rows: List[Mapping[str, Any]] = []
        for row in reader:
            if row.get(None):
                raise ManifestValidationError("TSV row has more values than header columns")
            if any(_string(value) for value in row.values()):
                rows.append(row)
        return rows


def _load_yaml(source: Path) -> Tuple[List[Mapping[str, Any]], Path]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency, defensive for lean installs
        raise ImportError("PyYAML is required to load YAML manifests") from exc
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    root = source.parent
    if isinstance(payload, Mapping):
        raw_root = payload.get("path_root")
        if raw_root:
            root = Path(resolve_manifest_path(str(raw_root), source.parent))
        rows = payload.get("records", payload.get("samples"))
    else:
        rows = payload
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ManifestValidationError("YAML manifest must be a list or contain a records list")
    return list(rows), root


def load_manifest(
    path: PathLike,
    resolve_paths: bool = True,
    validate: bool = True,
    require_folds: bool = False,
) -> Manifest:
    """Load a TSV or YAML manifest with paths relative to the manifest file."""

    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        rows, base_dir = _load_yaml(source)
    elif suffix in {".tsv", ".txt"}:
        rows = _load_tsv(source)
        base_dir = source.parent
    else:
        raise ManifestValidationError("manifest must be TSV, YAML, or YML")
    records = tuple(
        ManifestRecord.from_mapping(row, base_dir=base_dir, resolve_paths=resolve_paths)
        for row in rows
    )
    manifest = Manifest(records=records, source=str(source))
    if validate:
        manifest.validate(require_folds=require_folds)
    return manifest


read_manifest = load_manifest


def write_manifest_tsv(manifest: Sequence[ManifestRecord], path: PathLike) -> None:
    """Write the canonical column order without mutating records."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        for record in manifest:
            writer.writerow(record.to_mapping())


def verify_checksums(
    records: Iterable[ManifestRecord],
    chunk_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Stream row-level SHA-256 checksums for records that provide one."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    for record in records:
        if not record.checksum:
            continue
        path_value = record.primary_file
        if not path_value:
            raise ManifestValidationError("%s has a checksum but no file" % record.record_id)
        if "::" in path_value or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", path_value):
            raise ManifestValidationError(
                "%s checksum verification requires a local extracted file" % record.record_id
            )
        digest = hashlib.sha256()
        with Path(path_value).open("rb") as handle:
            while True:
                block = handle.read(chunk_bytes)
                if not block:
                    break
                digest.update(block)
        expected = record.checksum.lower().removeprefix("sha256:")
        if digest.hexdigest() != expected:
            raise ManifestValidationError("checksum mismatch for %s" % record.record_id)
