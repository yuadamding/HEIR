"""Bounded source correspondence and author-coordinate replay, without new fitting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .author_replay import replay_slide_seq
from .capability import _sha, _write
from .codex_replay import replay_codex
from .cohort import discover_cohort
from .correspondence import build_source_correspondence, load_source_correspondence


def _table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def linkage_rows(document):
    """Flatten source-scoped relationships without overwriting the study-design evidence."""
    return [
        {
            **{
                key: record[key]
                for key in (
                    "patient_id",
                    "biopsy_id",
                    "specimen_id",
                    "source_role",
                    "assay_replicate_label",
                    "core_id",
                    "tissue_piece_id",
                    "study_design_core_role",
                    "study_design_relationship",
                    "section_id",
                    "section_order",
                    "section_identity_basis",
                    "relationship",
                    "relationship_scope",
                    "missing_fields",
                    "author_source_filenames",
                    "evidence_ids",
                )
            },
            "source_path": record["source"]["path"],
            "source_sha256": record["source"]["sha256"],
            "associated_he_path": record["associated_he_frame"]["path"],
            "associated_he_sha256": record["associated_he_frame"]["sha256"],
            "associated_he_basis": record["associated_he_frame"]["basis"],
        }
        for record in document["records"]
    ]


def _request(document):
    missing = sorted(
        {field for record in document["records"] for field in record["missing_fields"]}
    )
    return "\n".join(
        [
            "# Source-information request (prepared, not sent)",
            "",
            "For the 13 selected HTAPP MBC biopsies and the exact source files/assay labels in "
            "`linkage_review.tsv`, please provide:",
            "",
            "1. The full `sampletracking_HTAPP_MBC_spatial.csv` and "
            "the crosswalk linking biopsy/core "
            "and tissue-piece identity, assay labels, biological section IDs/order, native files, "
            "and associated H&E images. The notebook's retained head is not the full crosswalk.",
            "2. The authoritative registration version, including H&E thumbnail/crop/resampling "
            "history, fragment-specific transforms, overlap exclusions and pre-registration QC "
            "populations. The retained coordinate chains can be checked against these records.",
            "3. Clarification of the executed Slide-seq 0.6 coordinate factor versus the reported "
            "0.65 micrometres/native pixel, plus acquisition/scanner calibration supporting the "
            "final image frame. A global post-hoc rescaling is not assumed.",
            "4. Independent registration evidence, such as paired images/landmarks and residuals "
            "with the fitting and validation landmarks distinguished, or acquisition evidence "
            "establishing which tissue corresponds at the intended 100/200-pixel regional scale.",
            "",
            "Exact currently unresolved record fields: " + ", ".join(missing) + ".",
            "",
            "Study-level serial sections and a separate independent-RNA core "
            "are already documented. "
            "The request concerns file-level linkage and alignment adequacy, "
            "not proof of identical cells.",
            "",
        ]
    )


def _report(output, summary, document, reports):
    lines = [
        "# HTAPP source correspondence and coordinate replay (v0.9)",
        "",
        "The v0.8 measurement screen remains complete. This follow-up reconciles source "
        "identities and coordinate provenance; it does not fit models, select a transform "
        "from expression agreement, or alter the completed studies.",
        "",
        "## Source correspondence",
        "",
        f"The source table contains {len(document['records'])} retained assay-label records and "
        f"{len(document['supplemental_he_image_chains'])} H&E image-chain records. Study-design "
        "serial sections are distinct from exact section identifiers. CODEX filename labels "
        "are retained as source evidence, while missing section/core/image crosswalk fields "
        "remain explicit. Assay replicate numbers are not converted to biological section IDs.",
        "",
        "The publication describes spatial assays and H&E from serial sections of a second "
        "OCT core, distinct from the independent-RNA core. This supports regional serial-section "
        "interpretation without establishing every file's section order. "
        f"[Publication]({summary['publication_url']}); [portal]({summary['portal_url']}).",
        "",
        "## Coordinate replay",
        "",
        "Replay residuals measure agreement with the published coordinate files. They are "
        "not landmark errors or evidence that different sections "
        "represent identical molecular fields.",
        "",
        "| Assay | Replicate records | Retained observations | "
        "Maximum replay error (registered pixels) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for assay, value in summary["assays"].items():
        lines.append(
            f"| {assay} | {value['replicate_records']} | {value['retained_observations']:,} | "
            f"{value['maximum_replay_error_pixels']:.6g} |"
        )
    lines += [
        "",
        "Native-to-analysis scaling, origin changes, preprocessing populations, "
        "rotations/reflections, translations and piecewise transforms are recorded in "
        "`coordinate_chains.json`. Observation IDs join native to registered records; no "
        "transform parameter is estimated from the registered coordinates or gene correlations.",
        "",
        "The executed Slide-seq factor is 0.6. Reproducing that source chain does not resolve "
        "the publication's 0.65 micrometres/native-pixel statement into a calibrated final H&E "
        "frame. Rescaling already registered coordinates alone would not reproduce the full "
        "origin/rotation/translation/image chain. No corrected geometry is published here.",
        "",
        "H&E image records retain thumbnail, rotation, crop/pad, resampling and display "
        "conventions. Image-chain descriptions and file dimensions are not pixel-replay or "
        "independent landmark validation. The final coordinate units remain registered pixels.",
        "",
        "## Terminal decision",
        "",
        summary["decision"],
        "",
        "Unknown physical calibration alone does not prohibit exploratory pixel-frame modeling. "
        "Likewise, an undefined CD8A correlation does not invalidate other measured genes. "
        "The unresolved issue for a new qualification claim is independent evidence of tissue "
        "correspondence and alignment adequacy. The 8,396-gene denominator screen still measured "
        "only nine outcomes; it was not a broader-outcome reproducibility experiment.",
        "",
        "The earlier 100/200-pixel results remain available in the completed measurement report. "
        "They are not relabeled as a comparison of original versus "
        "independently qualified geometry. "
        "`source_request.md` specifies the missing evidence and has not been sent to anyone.",
        "",
        "## Artifacts",
        "",
        "- `source_correspondence.json`, `linkage_review.tsv`: exact source identities, scoped "
        "relationships, evidence references and missing fields.",
        "- `coordinate_replay.tsv`, `coordinate_chains.json`, `coordinate_replay.npz`: replay "
        "statistics, declared operations and per-observation identities/residuals.",
        "- `summary.json`, `identity.json`, `config.json`: terminal decision and source bindings.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def run_reconciliation(root, pipeline, measurement, config_path, output):
    root, pipeline, measurement, output = map(Path, (root, pipeline, measurement, output))
    config = json.loads(Path(config_path).read_text())
    if (
        config.get("schema") != "heir.source_reconciliation.v1"
        or config.get("assays") != ["slide_seq", "codex"]
        or config.get("new_geometry_view") is not False
        or config.get("new_model_fitting") is not False
        or config.get("new_measurement_sweep") is not False
        or config.get("reference_region_widths_pixels") != [100, 200]
        or config.get("native_slide_seq_factor_executed") != 0.6
        or config.get("native_slide_seq_micrometres_per_pixel_reported") != 0.65
        or config.get("output_frame") != "registered_pixel"
    ):
        raise ValueError("unsupported source reconciliation scope")
    tolerance = config.get("replay_tolerance_pixels")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not (np.isfinite(tolerance) and tolerance > 0)
    ):
        raise ValueError("source replay tolerance must be finite positive registered pixels")
    if any(
        output.resolve().is_relative_to(p.resolve())
        for p in (root, pipeline, measurement, Path(__file__).parents[3])
    ):
        raise ValueError("reconciliation output must be outside immutable inputs and source")
    for name, digest in config["pipeline_files_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or _sha(pipeline / name) != digest:
            raise ValueError("pinned author source differs from reconciliation configuration")
    if _sha(root / "metadata/selected_13_verified_file_manifest.tsv") != config["manifest_sha256"]:
        raise ValueError("source manifest differs from reconciliation configuration")
    if _sha(measurement / "summary.json") != config["completed_measurement_summary_sha256"]:
        raise ValueError("completed measurement screen differs from its frozen binding")
    cohort = discover_cohort(root)
    document = build_source_correspondence(root, pipeline)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(Path(config_path).read_bytes())
    _write(output / "source_correspondence.json", document)
    crosswalk_hash = _sha(output / "source_correspondence.json")
    package = Path(__file__).resolve().parents[1]
    _write(
        output / "identity.json",
        {
            "config_sha256": _sha(config_path),
            "source_correspondence_sha256": crosswalk_hash,
            "completed_measurement": str(measurement.resolve()),
            "source_file_sha256": {
                str(p.relative_to(package.parent)): _sha(p) for p in sorted(package.rglob("*.py"))
            },
            "pipeline_files_sha256": config["pipeline_files_sha256"],
        },
    )
    reports, chains, arrays, verification = [], [], {}, {}
    for biopsy, entry in sorted(cohort.items()):
        print(f"reconciliation: {biopsy}, source identity and author transforms", flush=True)
        load_source_correspondence(
            output / "source_correspondence.json",
            expected_sha256=crosswalk_hash,
            biopsy_id=biopsy,
            patient_id=entry["patient_id"],
            specimen_id=entry["specimen_id"],
            sources=entry["role_sources"],
        )
        for replay in (replay_slide_seq, replay_codex):
            rows, operations, values = replay(
                root, biopsy, entry, pipeline_root=pipeline, verification_cache=verification
            )
            if set(arrays) & set(values):
                raise ValueError("replay observation arrays have conflicting identities")
            reports.extend(rows)
            chains.extend(operations)
            arrays.update(values)
    assays = {}
    for assay in config["assays"]:
        rows = [r for r in reports if r["assay"] == assay and r["n_registered"] > 0]
        assays[assay] = {
            "replicate_records": len(rows),
            "retained_observations": sum(r["n_registered"] for r in rows),
            "maximum_replay_error_pixels": max(r["max_replay_error_pixels"] for r in rows),
            "source_ids_complete": all(r["source_ids_complete"] for r in rows),
        }
    reproduced = all(
        v["source_ids_complete"]
        and v["maximum_replay_error_pixels"] <= config["replay_tolerance_pixels"]
        for v in assays.values()
    )
    summary = {
        "schema": "heir.source_reconciliation_result.v1",
        "assays": assays,
        "coordinate_chain_reproduced": reproduced,
        "native_only_qc_contributors": [
            {k: r[k] for k in ("biopsy_id", "assay", "replicate", "n_native")}
            for r in reports
            if r["n_registered"] == 0
        ],
        "replay_tolerance_pixels": config["replay_tolerance_pixels"],
        "independent_biological_alignment_validated": False,
        "physical_calibration": "unresolved",
        "new_geometry_view": False,
        "new_model_fitting": False,
        "new_measurement_sweep": False,
        "focused_100_200_validation": "not_run_no_independently_qualified_geometry_available",
        "qualification_branch": "closed_with_specific_external_evidence_gap",
        "decision": (
            "Retain HTAPP as exploratory registered-pixel serial-section evidence. "
            "The available source reconciliation is complete; replay alone does not qualify "
            "biological alignment. No independently validated geometry or exact source crosswalk "
            "was recovered, so the conditional original-versus-qualified 100/200-pixel comparison "
            "is not launched. Preserve v0.8 and pursue the specific source information listed "
            "in source_request.md or better-documented measurements. This closes the current "
            "qualification branch. The separately specified joint-model experiment may proceed "
            "on the historical 100-pixel grid as exploratory development; it does not qualify "
            "that geometry or authorize a 200-pixel modeling view."
        ),
        "publication_url": config["publication_url"],
        "portal_url": config["portal_url"],
    }
    _table(output / "linkage_review.tsv", linkage_rows(document))
    _table(output / "coordinate_replay.tsv", reports)
    _write(output / "coordinate_chains.json", chains)
    with (output / "coordinate_replay.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    _write(output / "summary.json", summary)
    (output / "source_request.md").write_text(_request(document))
    _report(output, summary, document, reports)
    print("reconciliation: completed all thirteen biopsies", flush=True)
    return summary
