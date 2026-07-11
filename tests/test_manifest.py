"""Scientific-integrity tests for immutable cohort manifests."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heir.data.cohorts import (  # noqa: E402
    NATCOMM_REFERENCE_MAPPINGS,
    load_natcommun_manifest,
    load_snpatho_manifest,
)
from heir.data.manifest import (  # noqa: E402
    MANIFEST_COLUMNS,
    Manifest,
    ManifestRecord,
    ManifestValidationError,
    load_manifest,
    validate_manifest,
)


def record(**updates):
    values = {
        "cohort_id": "cohort",
        "donor_id": "donor-1",
        "specimen_id": "specimen-1",
        "block_id": "block-1",
        "section_id": "section-1",
        "modality": "histology+snrna",
        "assay_platform": "H&E + 10x",
        "preservation": "FFPE",
        "tissue": "breast",
        "matching_tier": "tier_1",
        "matching_notes": "same block, non-registered nuclei",
        "analysis_role": "development",
        "outer_fold": "fold_0",
        "inner_fold": "inner_0",
    }
    values.update(updates)
    return ManifestRecord(**values)


class ManifestTests(unittest.TestCase):
    def test_record_is_frozen(self):
        value = record()
        with self.assertRaises(FrozenInstanceError):
            value.donor_id = "changed"  # type: ignore[misc]

    def test_manifest_defensively_freezes_record_sequence(self):
        source = [record()]
        manifest = Manifest(source)  # type: ignore[arg-type]
        source.clear()
        self.assertEqual(len(manifest), 1)
        self.assertIsInstance(manifest.records, tuple)

    def test_tsv_expands_environment_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ["HEIR_TEST_DATA"] = str(root / "public")
            try:
                values = record(
                    he_file="${HEIR_TEST_DATA}/slide.tif",
                    count_matrix_file="relative/reference.h5ad",
                ).to_mapping()
                path = root / "manifest.tsv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
                    writer.writeheader()
                    writer.writerow(values)
                manifest = load_manifest(path)
                self.assertEqual(manifest[0].he_file, str(root / "public" / "slide.tif"))
                self.assertEqual(
                    manifest[0].count_matrix_file,
                    str(root / "relative" / "reference.h5ad"),
                )
            finally:
                os.environ.pop("HEIR_TEST_DATA", None)

    def test_yaml_records_and_archive_member_resolution(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = record(he_file="archives/data.tar::slide.tif").to_mapping()
            path = root / "manifest.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump({"records": [values]}, handle)
            manifest = load_manifest(path)
            self.assertEqual(
                manifest[0].he_file,
                str(root / "archives" / "data.tar") + "::slide.tif",
            )

    def test_undefined_environment_variable_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = record(he_file="${DEFINITELY_MISSING_HEIR_VAR}/slide.tif").to_mapping()
            path = root / "manifest.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
                writer.writeheader()
                writer.writerow(values)
            with self.assertRaises(ManifestValidationError):
                load_manifest(path)

    def test_donor_fold_leakage_is_rejected(self):
        records = (
            record(section_id="section-a", outer_fold="fold_0"),
            record(section_id="section-b", outer_fold="fold_1"),
        )
        with self.assertRaisesRegex(ManifestValidationError, "donor_id"):
            validate_manifest(records)

    def test_block_partition_leakage_is_rejected(self):
        records = (
            record(
                donor_id="donor-a",
                section_id="section-a",
                analysis_role="development",
            ),
            record(
                donor_id="donor-b",
                section_id="section-b",
                block_id="block-1",
                analysis_role="locked_test",
                outer_fold="fold_1",
                inner_fold="inner_1",
            ),
        )
        with self.assertRaisesRegex(ManifestValidationError, "block_id"):
            validate_manifest(records)

    def test_spatial_measurements_cannot_be_training_data(self):
        spatial = record(
            modality="histology+visium",
            spatial_count_matrix_file="held-out-visium.h5ad",
            spatial_coordinate_file="positions.csv",
            analysis_role="development",
        )
        with self.assertRaisesRegex(ManifestValidationError, "spatial data"):
            validate_manifest((spatial,))

    def test_spatial_locked_validation_is_allowed(self):
        spatial = record(
            modality="histology+visium",
            spatial_count_matrix_file="held-out-visium.h5ad",
            spatial_coordinate_file="positions.csv",
            analysis_role="locked_validation",
        )
        validate_manifest((spatial,))

    def test_builtin_natcommun_is_exact_and_b2_excluded(self):
        manifest = load_natcommun_manifest(resolve_paths=False)
        self.assertEqual(len(manifest), 16)
        self.assertEqual(len(manifest.included_records), 15)
        self.assertTrue(all(item.matching_notes for item in manifest))
        b2 = [item for item in manifest if item.specimen_id == "B2"]
        self.assertEqual(len(b2), 1)
        self.assertFalse(b2[0].included)
        self.assertIn("50 nuclei", b2[0].exclusion_reason)
        b1 = [item for item in manifest if item.specimen_id == "B1"]
        self.assertEqual({item.outer_fold for item in b1}, {"fold_0"})
        self.assertEqual(NATCOMM_REFERENCE_MAPPINGS["B1"].expected_nuclei, 3937)

    def test_builtin_snpatho_spatial_data_is_locked(self):
        manifest = load_snpatho_manifest(resolve_paths=False)
        self.assertEqual(tuple(item.specimen_id for item in manifest), ("4066", "4399", "4411"))
        self.assertTrue(all(item.is_spatial for item in manifest))
        self.assertTrue(all(item.analysis_role == "locked_validation" for item in manifest))
        self.assertTrue(all("::" in item.he_file for item in manifest))


if __name__ == "__main__":
    unittest.main()
