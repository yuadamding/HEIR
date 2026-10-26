"""Small source crosswalk for the assays actually retained by the regional workflow.

Study-design serial sections and separate RNA cores do not identify individual
sections. Author filename section labels are retained without inventing order or
same-cell correspondence. This module never estimates registration or reads counts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .cohort import _table, discover_cohort

_SCHEMA = "heir.source_correspondence.v1"
_RELATIONSHIPS = {"unresolved", "documented_same_section_repeat", "serial_section", "separate_core"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(source):
    return {"path": source["working_path"], "sha256": source["sha256"]}


def _notebook_cells(path):
    return json.loads(Path(path).read_text())["cells"]


def _codex_sections(cells):
    """Use printed input filenames, not the replicate-to-section search pattern."""
    result = {}
    for index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        if "section-" not in source or "codex" not in source:
            continue
        current = None
        for output in cell.get("outputs", []):
            for line in "".join(output.get("text", [])).splitlines():
                sample = re.fullmatch(r"HTAPP-(\d+)-SMP-(\d+)_(\d+)", line.strip())
                if sample:
                    current = (f"{sample[1]}-{sample[2]}", sample[3])
                filename = re.search(
                    r"HTAPP-(\d+)-section-(\d+)-backgroundsubtraction-reg\d+_comp.csv", line
                )
                if filename and current:
                    if filename[1] != current[0].split("-")[0]:
                        raise ValueError("notebook source filename differs from its sample context")
                    record = result.setdefault(
                        current,
                        {"cell_index": index, "section_labels": set(), "source_filenames": []},
                    )
                    record["section_labels"].add(filename[2])
                    record["source_filenames"].append(Path(line.strip()).name)
    for record in result.values():
        if len(record["section_labels"]) != 1:
            raise ValueError("CODEX source filenames do not identify one section label")
        record["section_label"] = record.pop("section_labels").pop()
    return result


def build_source_correspondence(root, pipeline_root):
    """Regenerate explicit records from local manifests and pinned notebook evidence."""
    root, pipeline = Path(root).resolve(), Path(pipeline_root).resolve()
    cohort = discover_cohort(root)
    manifest_path = root / "metadata/selected_13_verified_file_manifest.tsv"
    manifest = _table(manifest_path)
    notebook = pipeline / "HTAPP_MBC/03_spatial_analysis/00_matchSpatial.ipynb"
    sections = _codex_sections(_notebook_cells(notebook))
    evidence = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha(manifest_path),
            "scope": "source_identity_only",
        },
        "match_spatial": {
            "path": str(notebook),
            "sha256": _sha(notebook),
            "scope": "author_source_labels_and_processing",
            "locators": (
                "cells4(tracking-table head),23(H&E discover"
                "y),25(Slide-seq source),57(CODEX filenames)"
            ),
        },
        "publication": {
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11564109/",
            "locators": "Figure1; Single-cell and spatial expression profiling; Sectioning methods",
            "scope": "study_design_only",
            "finding": (
                "Spatial assays/H&E use serial sections from a second OCT core; independent RNA "
                "uses a distinct fresh/frozen core. Exact file-to-section order is not specified."
            ),
        },
        "portal": {
            "url": "https://singlecell.broadinstitute.org/single_cell/study/SCP2702/htapp-mbc",
            "scope": "study_design_only",
            "finding": "Bundles are before/after registration of serial sections.",
        },
    }
    for name, filename in (
        ("assay_table", "selected_Table_S2_assay_rows.tsv"),
        ("library_table", "selected_Table_S1_library_rows.tsv"),
    ):
        path = root / "metadata" / filename
        if path.exists():
            evidence[name] = {"path": str(path), "sha256": _sha(path)}
    records, images = [], []
    for biopsy, entry in sorted(cohort.items()):
        sources = entry["role_sources"]
        processing = (
            pipeline
            / "HTAPP_MBC/03_spatial_analysis/00.1_processing"
            / f"HTAPP-{biopsy.replace('-', '-SMP-')}_processing.ipynb"
        )
        cells = _notebook_cells(processing)
        alignment = [
            i for i, cell in enumerate(cells) if "aligner(" in "".join(cell.get("source", []))
        ]
        evidence_key = f"processing:{biopsy}"
        evidence[evidence_key] = {
            "path": str(processing),
            "sha256": _sha(processing),
            "aligner_cell_indices": alignment,
        }
        for image in [
            row
            for row in manifest
            if row["biopsy_id"] == biopsy
            and row["modality"] == "HE"
            and "/registered/" in row["working_path"]
        ]:
            native = [
                row
                for row in manifest
                if row["biopsy_id"] == biopsy
                and row["modality"] == "HE"
                and row["replicate_id"] == image["replicate_id"]
                and "/unregistered/" in row["working_path"]
                and "highres" in row["working_path"]
            ]
            images.append(
                {
                    "biopsy_id": biopsy,
                    "author_image_label": f"HE {image['replicate_id']}",
                    "registered_image": _source(image),
                    "native_highres_images": [_source(r) for r in native],
                    "selected_heir_grid": image["working_path"] == sources["he"]["working_path"],
                    "biological_section_identity": "unresolved",
                    "evidence_ids": ["manifest", evidence_key],
                }
            )
        for role, source in sorted(sources.items()):
            for replicate in source["replicate_id"].split(";"):
                known = sections.get((biopsy, replicate)) if role == "protein" else None
                native = [
                    row
                    for row in manifest
                    if row["biopsy_id"] == biopsy
                    and row["modality"] == source["modality"]
                    and "/unregistered/" in row["working_path"]
                    and row["working_path"].endswith(".h5ad")
                    and (role == "rna" or replicate in row["replicate_id"].split(";"))
                ]
                missing = ["core_id", "tissue_piece_id", "assay_specific_relationship"]
                if role != "rna":
                    missing += (["section_id"] if known is None else []) + [
                        "section_order",
                        "section_to_he_image_crosswalk",
                    ]
                else:
                    missing += ["rna_core_to_spatial_core_crosswalk"]
                records.append(
                    {
                        "patient_id": entry["patient_id"],
                        "biopsy_id": biopsy,
                        "specimen_id": entry["specimen_id"],
                        "source_role": role,
                        "source": _source(source),
                        "native_measurement_sources": [
                            {**_source(row), "native_assay_labels": row["replicate_id"].split(";")}
                            for row in native
                        ],
                        "assay_replicate_label": replicate,
                        "core_id": None,
                        "tissue_piece_id": None,
                        "study_design_core_role": "independent_rna_core"
                        if role == "rna"
                        else "oct_spatial_core",
                        "study_design_relationship": "separate_core_independent_rna"
                        if role == "rna"
                        else "serial_sections_spatial",
                        "section_id": known["section_label"] if known else None,
                        "section_order": None,
                        "section_identity_basis": "author_source_filename"
                        if known
                        else "not_applicable_dissociated_reference"
                        if role == "rna"
                        else "unresolved",
                        "section_evidence_ids": ["match_spatial"] if known else [],
                        "author_source_filenames": known["source_filenames"] if known else [],
                        "relationship": "unresolved",
                        "relationship_scope": "assay_specific",
                        "relationship_evidence_ids": [],
                        "associated_he_frame": {
                            **_source(sources["he"]),
                            "basis": (
                                "selected_registered_HEIR_coordinate_frame_not_biological_pairing"
                            ),
                        },
                        "evidence_ids": [
                            "manifest",
                            "match_spatial",
                            "publication",
                            "portal",
                            evidence_key,
                        ],
                        "missing_fields": missing,
                    }
                )
    tracking_head = []
    for index, cell in enumerate(_notebook_cells(notebook)):
        if "sampletracking_HTAPP_MBC_spatial.csv" not in "".join(cell.get("source", [])):
            continue
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", [])) + "".join(
                output.get("data", {}).get("text/plain", [])
            )
            for line in text.splitlines():
                match = re.fullmatch(
                    r"\s*\d+\s+HTAPP-(\d+)-SMP-(\d+)\s+(\d+)\s+([\d.]+|NaN)\s+([\d.]+|NaN)\s+(\S+)\s*",
                    line,
                )
                if match and f"{match[1]}-{match[2]}" in cohort:
                    tracking_head.append(
                        {
                            "biopsy_id": f"{match[1]}-{match[2]}",
                            "assay_replicate_label": match[3],
                            "batch_label": match[4],
                            "puck_label": match[5],
                            "slide_seq_source_id": match[6],
                            "notebook_cell_index": index,
                            "biological_section_identity": "unresolved",
                        }
                    )
    tracking = list(pipeline.rglob("sampletracking_HTAPP_MBC_spatial.csv"))
    return {
        "schema": _SCHEMA,
        "scope": (
            "selected_HEIR_HE1_independent_RNA_CODEX_Slide_seq_sources_and_retained_assay_labels"
        ),
        "records": records,
        "supplemental_he_image_chains": images,
        "evidence": evidence,
        "tracking_table": {
            "name": "sampletracking_HTAPP_MBC_spatial.csv",
            "local_paths": list(map(str, tracking)),
            "selected_rows_recovered_from_displayed_head": tracking_head,
            "status": "available_requires_review" if tracking else "not_recovered",
            "retained_notebook_head_fields": [
                "sample",
                "replicate",
                "batch",
                "puck",
                "slide_seq_ID",
            ],
            "limitation": (
                "The displayed head is not the full file and supplies no section/core crosswalk."
            ),
        },
        "pairing_qualification": (
            "unresolved_exact_crosswalk; study_design_serial_sections_is_separate_evidence"
        ),
    }


def load_source_correspondence(
    path, *, expected_sha256, biopsy_id, patient_id, specimen_id, sources
):
    """Load a hash-bound crosswalk, validating exact coverage and source identities."""
    path = Path(path)
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("source correspondence requires an explicit SHA-256")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("source correspondence checksum differs")
    document = json.loads(payload)
    if document.get("schema") != _SCHEMA:
        raise ValueError("unsupported source correspondence schema")
    records = [r for r in document["records"] if r["biopsy_id"] == biopsy_id]
    expected = {
        (role, rep) for role, source in sources.items() for rep in source["replicate_id"].split(";")
    }
    keys = [(r["source_role"], r["assay_replicate_label"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            "source correspondence must cover every retained role/replicate exactly once"
        )
    evidence = document["evidence"]
    for record in records:
        role = record["source_role"]
        if (
            record["patient_id"] != patient_id
            or record["specimen_id"] != specimen_id
            or record["source"] != _source(sources[role])
            or {k: record["associated_he_frame"][k] for k in ("path", "sha256")}
            != _source(sources["he"])
        ):
            raise ValueError(
                "source correspondence patient, specimen or source/frame identity differs"
            )
        referenced = (
            record["evidence_ids"]
            + record["section_evidence_ids"]
            + record["relationship_evidence_ids"]
        )
        if not record["evidence_ids"] or not set(referenced) <= set(evidence):
            raise ValueError("source correspondence requires resolvable evidence identifiers")
        if record["relationship"] not in _RELATIONSHIPS:
            raise ValueError("unknown biological relationship category")
        for field in ("core_id", "tissue_piece_id", "section_id"):
            value = record[field]
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or value.lower() in {"unknown", "unresolved", "none", "nan"}
            ):
                raise ValueError(
                    "unresolved biological identities must be null, not placeholder IDs"
                )
        if record["section_id"] is not None and (
            not record["section_evidence_ids"]
            or record["section_identity_basis"]
            not in {"author_source_filename", "acquisition_crosswalk"}
        ):
            raise ValueError(
                "section identity needs evidence independent of the assay replicate label"
            )
        if (
            record["section_order"] is not None
            and record["section_identity_basis"] != "acquisition_crosswalk"
        ):
            raise ValueError("biological section order cannot be inferred from a filename label")
        if record["relationship"] != "unresolved" and (
            record["relationship_scope"] != "documented_assay_crosswalk"
            or not record["relationship_evidence_ids"]
        ):
            raise ValueError("study-design evidence cannot establish an exact assay relationship")
        if record["section_identity_basis"] == "acquisition_crosswalk" and any(
            evidence[key].get("scope") != "assay_specific" for key in record["section_evidence_ids"]
        ):
            raise ValueError("study-level evidence cannot supply an acquisition section crosswalk")
        if record["relationship"] != "unresolved" and any(
            evidence[key].get("scope") != "assay_specific"
            for key in record["relationship_evidence_ids"]
        ):
            raise ValueError("study-level evidence cannot qualify an exact assay relationship")
        if any(record[field] is not None for field in ("core_id", "tissue_piece_id")) and not any(
            evidence[key].get("scope") == "assay_specific" for key in referenced
        ):
            raise ValueError("core/piece identity needs assay-specific evidence")
        if record["relationship"] == "unresolved" and not record["missing_fields"]:
            raise ValueError("unresolved correspondence must retain specific missing fields")
    targets = [r for r in records if r["source_role"] == "targets"]
    documented = all(
        r["section_identity_basis"] == "acquisition_crosswalk" and r["section_id"] for r in targets
    )
    relations = {r["relationship"] for r in targets}
    relationship = (
        next(iter(relations))
        if len(relations) == 1 and "unresolved" not in relations
        else "unknown"
    )
    core_ids = {r["core_id"] for r in targets}
    if relationship != "unknown" and (not documented or None in core_ids):
        raise ValueError("documented target linkage requires exact section and core fields")
    if relationship in {"serial_section", "documented_same_section_repeat"} and len(core_ids) != 1:
        raise ValueError("same-core relationship conflicts with target core identities")
    if (
        relationship == "documented_same_section_repeat"
        and len({r["section_id"] for r in targets}) != 1
    ):
        raise ValueError("same-section relationship conflicts with target section identities")
    if (
        relationship == "serial_section"
        and len(targets) > 1
        and len({r["section_id"] for r in targets}) != len(targets)
    ):
        raise ValueError("serial-section relationship requires distinct target section identities")
    if relationship == "separate_core" and len(targets) > 1 and len(core_ids) != len(targets):
        raise ValueError("separate-core relationship requires distinct target core identities")
    return {
        "schema": _SCHEMA,
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "scope": document["scope"],
        "records": records,
        "evidence": evidence,
        "supplemental_he_image_chains": [
            r
            for r in document.get("supplemental_he_image_chains", [])
            if r["biopsy_id"] == biopsy_id
        ],
        "target_linkage": relationship,
        "target_section_ids": {r["assay_replicate_label"]: r["section_id"] for r in targets}
        if documented
        else {},
        "target_core_id": next(iter(core_ids))
        if len(core_ids) == 1 and None not in core_ids
        else None,
        "study_design_relationships": sorted({r["study_design_relationship"] for r in records}),
        "exact_relationship_qualified": relationship != "unknown",
    }
