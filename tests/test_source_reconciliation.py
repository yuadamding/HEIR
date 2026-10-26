"""The source replay receipt preserves a failed numerical result and qualification gaps."""

import hashlib
import json

import numpy as np
import pytest

from heir.personalized import source_reconciliation as workflow


@pytest.mark.parametrize("residual,reproduced", [(0.0, True), (0.01, False)])
def test_replay_receipt_distinguishes_provenance_from_alignment(
    tmp_path, monkeypatch, residual, reproduced
):
    root, pipeline, measurement = [tmp_path / name for name in ("data", "pipeline", "old")]
    (root / "metadata").mkdir(parents=True)
    pipeline.mkdir()
    measurement.mkdir()
    manifest = root / "metadata/selected_13_verified_file_manifest.tsv"
    manifest.write_text("fixture\n")
    previous = measurement / "summary.json"
    previous.write_text("{}\n")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = {
        "schema": "heir.source_reconciliation.v1",
        "assays": ["slide_seq", "codex"],
        "new_geometry_view": False,
        "new_model_fitting": False,
        "new_measurement_sweep": False,
        "reference_region_widths_pixels": [100, 200],
        "native_slide_seq_factor_executed": 0.6,
        "native_slide_seq_micrometres_per_pixel_reported": 0.65,
        "output_frame": "registered_pixel",
        "replay_tolerance_pixels": 1e-6,
        "pipeline_files_sha256": {},
        "manifest_sha256": sha(manifest),
        "completed_measurement_summary_sha256": sha(previous),
        "publication_url": "https://example.org/publication",
        "portal_url": "https://example.org/portal",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        workflow,
        "discover_cohort",
        lambda root: {
            "specimen": {"patient_id": "patient", "specimen_id": "specimen", "role_sources": {}}
        },
    )
    monkeypatch.setattr(
        workflow,
        "build_source_correspondence",
        lambda *args: {"records": [], "supplemental_he_image_chains": []},
    )
    monkeypatch.setattr(workflow, "load_source_correspondence", lambda *args, **kwargs: {})

    def replay(assay):
        def run(*args, **kwargs):
            rows = [
                {
                    "assay": assay,
                    "n_registered": 2,
                    "source_ids_complete": True,
                    "max_replay_error_pixels": residual,
                }
            ]
            if assay == "slide_seq":
                rows.append(
                    {
                        "assay": assay,
                        "biopsy_id": "specimen",
                        "replicate": "2",
                        "n_native": 5,
                        "n_registered": 0,
                    }
                )
            return rows, [], {assay: np.array([0.0, residual])}

        return run

    monkeypatch.setattr(workflow, "replay_slide_seq", replay("slide_seq"))
    monkeypatch.setattr(workflow, "replay_codex", replay("codex"))
    output = tmp_path / "output"
    result = workflow.run_reconciliation(root, pipeline, measurement, path, output)
    assert result["coordinate_chain_reproduced"] is reproduced
    assert result["independent_biological_alignment_validated"] is False
    assert result["new_geometry_view"] is False
    assert result["assays"]["slide_seq"]["replicate_records"] == 1
    assert result["native_only_qc_contributors"][0]["n_native"] == 5
    assert "not sent" in (output / "source_request.md").read_text()
    with np.load(output / "coordinate_replay.npz", allow_pickle=False) as arrays:
        assert arrays["slide_seq"][1] == residual
