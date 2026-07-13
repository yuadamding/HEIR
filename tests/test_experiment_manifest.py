from __future__ import annotations

import json
from pathlib import Path

import pytest

from heir.data import (
    EXPERIMENT_MANIFEST_SCHEMA,
    ExperimentManifest,
    canonical_sha256,
    ordered_ids_sha256,
)
from heir.utils import sha256_file


def _content(protocol: Path, source: Path) -> dict[str, object]:
    return {
        "schema": EXPERIMENT_MANIFEST_SCHEMA,
        "experiment_role": "primary_hest_uni2h",
        "scientific_scope": "registered_cell_local_context_association",
        "code_commit_sha": "1" * 40,
        "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
        "source": {
            "schema": "source.v1",
            "schema_sha256": canonical_sha256(["gene_ids", "type_names"]),
            "observations_sha256": sha256_file(source),
            "cohort_id": "HEST",
            "cohort_release": "GSE250346",
            "assay": "Xenium",
            "observation_level": "cell",
            "donor_sections": {
                "d1": ["s1"],
                "d2": ["s2"],
                "d3": ["s3"],
                "d4": ["s4"],
            },
        },
        "partitions": {
            "development_donors": ["d1", "d2"],
            "locked_test_donors": ["d3", "d4"],
        },
        "encoder": {
            "repository": "MahmoodLab/UNI2-h",
            "revision": "revision",
            "checkpoint_sha256": "2" * 64,
            "feature_width": 1536,
        },
        "preprocessing": {
            "implementation": "heir.hest_crop.v1",
            "implementation_sha256": "3" * 64,
            "crop_role": "nucleus_centered",
            "crop_diameter_um": 112.0,
            "source_mpp": 0.2125,
            "model_mpp": 0.5,
            "model_input_pixels": 224,
            "mask_mode": "none",
        },
        "target": {
            "construction": "nucleus_overlapping_xenium_transcripts",
            "schema": "ordered_gene_counts.v1",
            "gene_ids": ["G1", "G2"],
            "gene_panel_sha256": ordered_ids_sha256(["G1", "G2"]),
        },
        "labels": {
            "procedure": "RNA annotation",
            "source_sha256": "4" * 64,
            "type_names": ["A", "B"],
            "marker_gene_ids": ["M1"],
            "conditioning_levels": ["broad_lineage", "fine_final_CT"],
        },
        "reference_pool": {
            "construction": "spatially_disjoint_donor_type_mean",
            "spatially_disjoint": True,
            "minimum_per_donor_type": 10,
            "observation_manifest_sha256": "5" * 64,
        },
        "nuisance_covariates": ["log1p_library_size", "section"],
        "gate": {
            "ranks": [2, 4],
            "ridge_penalties": [0.1, 1.0],
            "permutation_seeds": [17, 29, 41],
            "permutations_per_seed": 100,
            "minimum_support": 10,
            "minimum_development_donors": 2,
            "minimum_locked_donors": 2,
            "minimum_coverage_fraction": 0.8,
            "minimum_shuffled_fraction": 0.5,
            "nulls": ["within_roi_derangement", "spatial_block_reassignment"],
            "thresholds": {"minimum_donor_effect": 0.01},
        },
    }


def test_experiment_manifest_binds_protocol_source_and_ordered_schema(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    source = tmp_path / "source.npz"
    manifest_path = tmp_path / "manifest.json"
    protocol.write_text("{}\n", encoding="utf-8")
    source.write_bytes(b"source")
    manifest_path.write_text(json.dumps(_content(protocol, source)), encoding="utf-8")

    manifest = ExperimentManifest.load(manifest_path)
    manifest.validate_source(source, "source.v1")
    assert manifest.experiment_role == "primary_hest_uni2h"
    assert manifest.gene_ids == ("G1", "G2")
    assert manifest.sha256 == sha256_file(manifest_path)


def test_experiment_manifest_rejects_gene_relabel_and_protocol_drift(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    source = tmp_path / "source.npz"
    manifest_path = tmp_path / "manifest.json"
    protocol.write_text("{}\n", encoding="utf-8")
    source.write_bytes(b"source")
    content = _content(protocol, source)
    content["target"]["gene_ids"] = ["G2", "G1"]
    manifest_path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered gene-panel hash"):
        ExperimentManifest.load(manifest_path)

    content = _content(protocol, source)
    manifest_path.write_text(json.dumps(content), encoding="utf-8")
    protocol.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="protocol file or SHA"):
        ExperimentManifest.load(manifest_path)
