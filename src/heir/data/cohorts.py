"""Audited local public-cohort definitions for the HEIR MVP."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from .manifest import Manifest, ManifestRecord, load_manifest, split_filter_values

DEFAULT_MANIFEST_DIR = resources.files("heir.resources.manifests")

NATCOMM_ROOT = Path("/mnt/seagate/HnE/NatCommun_2025_s41467_025_59005_9")
SNPATHO_ROOT = Path("/mnt/seagate/HnE/snPATHO_seq")


@dataclass(frozen=True)
class ReferenceMapping:
    """Exact public-H5AD mapping for a histology specimen."""

    donor_values: Tuple[str, ...]
    sample_values: Tuple[str, ...]
    expected_nuclei: int


NATCOMM_REFERENCE_MAPPINGS: Mapping[str, ReferenceMapping] = {
    "B1": ReferenceMapping(("7",), ("6",), 3937),
    "B2": ReferenceMapping(("2",), ("1",), 50),
    "B3": ReferenceMapping(("0",), ("0", "8"), 4729),
    "B4": ReferenceMapping(("1",), ("7", "9"), 1973),
    "L1": ReferenceMapping(("5",), ("4",), 802),
    "L2": ReferenceMapping(("4",), ("3",), 1756),
    "L3": ReferenceMapping(("3",), ("2",), 17804),
    "L4": ReferenceMapping(("6",), ("5",), 15592),
    "D1": ReferenceMapping(("D1",), ("DLBCL_1",), 8936),
    "D2": ReferenceMapping(("D2",), ("DLBCL_2",), 7097),
    "D3": ReferenceMapping(("D3",), ("DLBCL_3",), 14375),
    "D4": ReferenceMapping(("D4",), ("DLBCL_4",), 5335),
    "D5": ReferenceMapping(("D5",), ("DLBCL_5",), 2485),
    "D6": ReferenceMapping(("D6",), ("DLBCL_6",), 1485),
}

NATCOMM_SECTION_IDS: Tuple[str, ...] = (
    "B1_2",
    "B1_4",
    "B2_2",
    "B3_2",
    "B4_2",
    "L1_2",
    "L1_4",
    "L2_2",
    "L3_2",
    "L4_2",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
)

SNPATHO_SAMPLE_IDS: Tuple[str, ...] = ("4066", "4399", "4411")
SNPATHO_REFERENCE_NUCLEI: Mapping[str, int] = {
    "4066": 20472,
    "4399": 23080,
    "4411": 27311,
}
SNPATHO_VISIUM_SPOTS: Mapping[str, int] = {
    "4066": 4769,
    "4399": 4560,
    "4411": 2812,
}


def default_manifest_path(cohort: str) -> Path:
    normalized = cohort.strip().lower().replace("-", "_")
    aliases = {
        "natcommun": "natcommun.tsv",
        "mosaic": "natcommun.tsv",
        "mosaic_natcommun_2025": "natcommun.tsv",
        "snpatho": "snpatho.tsv",
        "snpatho_seq": "snpatho.tsv",
    }
    if normalized not in aliases:
        raise KeyError("unknown built-in cohort %s" % cohort)
    return Path(str(DEFAULT_MANIFEST_DIR.joinpath(aliases[normalized])))


def load_cohort_manifest(
    cohort: str,
    manifest_path: Optional[Path] = None,
    resolve_paths: bool = True,
) -> Manifest:
    source = manifest_path or default_manifest_path(cohort)
    manifest = load_manifest(source, resolve_paths=resolve_paths, require_folds=True)
    validate_builtin_cohort(cohort, manifest)
    return manifest


def load_natcommun_manifest(
    manifest_path: Optional[Path] = None,
    resolve_paths: bool = True,
) -> Manifest:
    return load_cohort_manifest("natcommun", manifest_path, resolve_paths)


def load_snpatho_manifest(
    manifest_path: Optional[Path] = None,
    resolve_paths: bool = True,
) -> Manifest:
    return load_cohort_manifest("snpatho", manifest_path, resolve_paths)


def h5ad_filters(record: ManifestRecord) -> Dict[str, Tuple[str, ...]]:
    """Return exact observation filters encoded in a manifest record."""

    result: Dict[str, Tuple[str, ...]] = {}
    if record.donor_filter_key:
        result[record.donor_filter_key] = split_filter_values(record.donor_filter_value)
    if record.sample_filter_key:
        result[record.sample_filter_key] = split_filter_values(record.sample_filter_value)
    return result


def validate_builtin_cohort(cohort: str, manifest: Manifest) -> None:
    """Guard audited sample identities from accidental manifest drift."""

    normalized = cohort.strip().lower().replace("-", "_")
    if normalized in {"natcommun", "mosaic", "mosaic_natcommun_2025"}:
        sections = tuple(record.section_id for record in manifest)
        if sections != NATCOMM_SECTION_IDS:
            raise ValueError("NatCommun manifest no longer contains the audited 16-section order")
        b2_records = tuple(record for record in manifest if record.specimen_id == "B2")
        if len(b2_records) != 1 or b2_records[0].included:
            raise ValueError("NatCommun B2 must remain explicitly excluded")
        for record in manifest:
            mapping = NATCOMM_REFERENCE_MAPPINGS[record.specimen_id]
            if split_filter_values(record.donor_filter_value) != mapping.donor_values:
                raise ValueError("incorrect donor filter for %s" % record.section_id)
            if split_filter_values(record.sample_filter_value) != mapping.sample_values:
                raise ValueError("incorrect sample filter for %s" % record.section_id)
    elif normalized in {"snpatho", "snpatho_seq"}:
        samples = tuple(record.specimen_id for record in manifest)
        if samples != SNPATHO_SAMPLE_IDS:
            raise ValueError("snPATHO manifest must contain exactly 4066, 4399, and 4411")
        if any(record.analysis_role != "locked_validation" for record in manifest):
            raise ValueError("snPATHO spatial measurements must remain locked validation data")
