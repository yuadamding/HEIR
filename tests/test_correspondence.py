"""Hash-bound source evidence never promotes design labels to exact pairing."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_personalized_cohort import published_cohort as published_cohort

from heir.personalized.cohort import discover_cohort, load_measurement_substrate
from heir.personalized.correspondence import build_source_correspondence, load_source_correspondence
from heir.personalized.measurement_workflow import _linkage_fields


@pytest.fixture
def crosswalk(published_cohort, tmp_path):
    pipeline = tmp_path / "pipeline"
    spatial = pipeline / "HTAPP_MBC/03_spatial_analysis"
    processing = spatial / "00.1_processing"
    processing.mkdir(parents=True)
    samples = discover_cohort(published_cohort)
    notebook = {
        "cells": [
            {
                "source": ["# codex source section- filenames"],
                "outputs": [
                    {
                        "text": [
                            "HTAPP-313-SMP-932_1\n",
                            "/original/HTAPP-313-section-3-backgroundsubtraction-reg001_comp.csv\n",
                        ]
                    }
                ],
            }
        ]
    }
    (spatial / "00_matchSpatial.ipynb").write_text(json.dumps(notebook))
    for biopsy in samples:
        path = processing / f"HTAPP-{biopsy.replace('-', '-SMP-')}_processing.ipynb"
        path.write_text(
            json.dumps({"cells": [{"source": ['aligner(img_idx="HE 1",img_idx2="HE 2")']}]})
        )
    document = build_source_correspondence(published_cohort, pipeline)
    return published_cohort, document, samples["313-932"], tmp_path / "crosswalk.json"


def _load(case, document=None):
    root, original, entry, path = case
    path.write_text(json.dumps(original if document is None else document))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_source_correspondence(
        path,
        expected_sha256=digest,
        biopsy_id="313-932",
        patient_id=entry["patient_id"],
        specimen_id=entry["specimen_id"],
        sources=entry["role_sources"],
    )
    return result, path, digest


def test_builder_preserves_known_source_section_label_and_explicit_gaps(crosswalk):
    bound, _, _ = _load(crosswalk)
    rows = bound["records"]
    protein = next(r for r in rows if r["source_role"] == "protein")
    assert protein["assay_replicate_label"] == "1"
    assert protein["section_id"] == "3"
    assert protein["section_order"] is None
    assert protein["section_identity_basis"] == "author_source_filename"
    assert protein["study_design_relationship"] == "serial_sections_spatial"
    assert (
        next(r for r in rows if r["source_role"] == "rna")["study_design_relationship"]
        == "separate_core_independent_rna"
    )
    assert all(r["relationship"] == "unresolved" and r["missing_fields"] for r in rows)
    assert bound["target_linkage"] == "unknown"
    assert not bound["exact_relationship_qualified"]
    assert bound["target_section_ids"] == {}
    assert crosswalk[1]["tracking_table"]["status"] == "not_recovered"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("patient", "identity differs"),
        ("source", "identity differs"),
        ("frame", "identity differs"),
        ("duplicate", "exactly once"),
        ("missing", "exactly once"),
        ("assay_as_section", "independent of the assay"),
        ("design_as_exact", "study-design evidence"),
        ("filename_order", "order cannot be inferred"),
        ("relabel_study_evidence", "study-level evidence cannot qualify"),
        ("core_from_design", "core/piece identity needs assay-specific"),
    ],
)
def test_inconsistent_or_overclaimed_crosswalks_are_rejected(crosswalk, mutation, match):
    document = deepcopy(crosswalk[1])
    row = next(
        r
        for r in document["records"]
        if r["biopsy_id"] == "313-932" and r["source_role"] == "protein"
    )
    if mutation == "patient":
        row["patient_id"] = "other"
    elif mutation == "source":
        row["source"]["sha256"] = "0" * 64
    elif mutation == "frame":
        row["associated_he_frame"]["path"] = "other.png"
    elif mutation == "duplicate":
        document["records"].append(deepcopy(row))
    elif mutation == "missing":
        document["records"].remove(row)
    elif mutation == "assay_as_section":
        row.update(section_id="1", section_identity_basis="assay_replicate_label")
    elif mutation == "design_as_exact":
        row.update(
            relationship="serial_section",
            relationship_scope="study_design",
            relationship_evidence_ids=["publication"],
        )
    elif mutation == "filename_order":
        row["section_order"] = 3
    elif mutation == "relabel_study_evidence":
        row.update(
            relationship="serial_section",
            relationship_scope="documented_assay_crosswalk",
            relationship_evidence_ids=["publication"],
        )
    elif mutation == "core_from_design":
        row["core_id"] = "inferred-spatial-core"
    with pytest.raises(ValueError, match=match):
        _load(crosswalk, document)


def test_checksum_binding_and_optional_loader_preserve_unknown_defaults(crosswalk):
    root, _, _, _ = crosswalk
    bound, path, digest = _load(crosswalk)
    measured = load_measurement_substrate(
        root, "313-932", base_region_size=10, correspondence_path=path, correspondence_sha256=digest
    )
    assert measured.provenance["source_correspondence"]["sha256"] == digest
    assert measured.provenance["replicate_linkage"] == "unknown"
    assert measured.provenance["section_identity"] == "unknown"
    assert measured.provenance["matching_scope"] == "published_biopsy"
    output_fields = _linkage_fields(measured.provenance)
    assert output_fields["linkage"] == "unknown"
    assert output_fields["source_crosswalk_sha256"] == digest
    assert "serial_sections_spatial" in output_fields["study_design_relationships"]
    assert "section_to_he_image_crosswalk" in output_fields["missing_source_fields"]
    assert output_fields["exact_section_ids"] == "{}"
    assert _linkage_fields({}) == {}
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        load_measurement_substrate(
            root, "313-932", correspondence_path=path, correspondence_sha256=digest
        )
    with pytest.raises(ValueError, match="both path and SHA"):
        load_measurement_substrate(root, "313-932", correspondence_path=path)


def test_only_documented_target_crosswalk_updates_measurement_linkage(crosswalk):
    document = deepcopy(crosswalk[1])
    document["evidence"]["acquisition"] = {
        "source": "synthetic acquisition crosswalk, not real data",
        "scope": "assay_specific",
    }
    for row in document["records"]:
        if row["biopsy_id"] == "313-932" and row["source_role"] == "targets":
            row.update(
                core_id="documented-core-A",
                section_id="documented-section-" + row["assay_replicate_label"],
                section_identity_basis="acquisition_crosswalk",
                section_evidence_ids=["acquisition"],
                relationship="serial_section",
                relationship_scope="documented_assay_crosswalk",
                relationship_evidence_ids=["acquisition"],
            )
    bound, path, digest = _load(crosswalk, document)
    assert bound["exact_relationship_qualified"]
    assert bound["target_linkage"] == "serial_section"
    measured = load_measurement_substrate(
        crosswalk[0],
        "313-932",
        base_region_size=10,
        correspondence_path=path,
        correspondence_sha256=digest,
    )
    assert measured.provenance["replicate_linkage"] == "serial_section"
    assert measured.provenance["matching_scope"] == "documented_target_core_and_section_crosswalk"
    assert measured.provenance["section_identity"] == "documented_target_section_ids"
    assert measured.provenance["core_identity"] == "documented-core-A"
    output_fields = _linkage_fields(measured.provenance)
    assert output_fields["linkage"] == "serial_section"
    assert json.loads(output_fields["exact_section_ids"]) == bound["target_section_ids"]
    assert output_fields["core_relationship"] == "documented-core-A"
    targets = [
        r
        for r in document["records"]
        if r["biopsy_id"] == "313-932" and r["source_role"] == "targets"
    ]
    targets[1]["section_id"] = targets[0]["section_id"]
    with pytest.raises(ValueError, match="distinct target section"):
        _load(crosswalk, document)
