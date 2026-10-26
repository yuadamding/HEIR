"""Offline inspection preserves image orientation and exact regional exclusion reasons."""

import base64
import csv
import hashlib
import io
import json
import re

import numpy as np
import pytest
from test_personalized_cohort import _tsv
from test_personalized_cohort import published_cohort as published_cohort

from heir.personalized import cohort, inspection
from heir.personalized.regions import build_region_grid


def _refresh_manifest(root, path):
    manifest = root / "metadata/selected_13_verified_file_manifest.tsv"
    rows = cohort._table(manifest)
    for row in rows:
        if row["working_path"] == str(path.relative_to(root)):
            row.update(
                bytes=str(path.stat().st_size), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
    _tsv(manifest, rows)


def _write(root, output, **kwargs):
    return inspection.write_pilot_inspection(
        root,
        "313-932",
        output,
        preparation={"region_size": 10, "max_states": 2},
        gene_ids=["EPCAM", "KRT8"],
        normalization_gene_ids=["EPCAM", "KRT8"],
        **kwargs,
    )


def test_report_exact_coverage_and_lower_origin_after_target_free_grid(
    published_cohort, monkeypatch
):
    import anndata
    from PIL import Image

    root = published_cohort
    directory = root / "working/registered/313-932"
    image = np.empty((20, 20, 3), dtype=np.uint8)
    image[:10] = [180, 40, 100]
    image[10:] = [80, 40, 180]
    Image.fromarray(image).save(directory / "HE.jpg", format="PNG")
    _refresh_manifest(root, directory / "HE.jpg")
    target_path = directory / "slide_seq.h5ad"
    target = anndata.read_h5ad(target_path)
    target.obsm["counts"][2] = 0
    target.write_h5ad(target_path)
    _refresh_manifest(root, target_path)
    order = []
    original_prepare, original_targets = (
        inspection.prepare_cohort_inputs,
        inspection.load_cohort_targets,
    )

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        order.append("grid_complete")
        return result

    def load_targets(*args, **kwargs):
        assert order == ["grid_complete"]
        order.append("target_read")
        return original_targets(*args, **kwargs)

    monkeypatch.setattr(inspection, "prepare_cohort_inputs", prepare)
    monkeypatch.setattr(inspection, "load_cohort_targets", load_targets)
    output = root.parent / (root.name + "_report")
    summary = _write(root, output)
    assert order == ["grid_complete", "target_read"]
    assert summary["n_selected_regions"] == 4
    assert len(summary["retained_target_observation_ids"]) == 3
    assert summary["excluded_target_observations"] == [
        {
            "observation_id": "slide_seq:1:he_region:1:0",
            "reason": "zero_normalization_exposure",
        }
    ]
    assert summary["uncovered_regions"]["codex:1"] == ["he_region:0:1"]
    panels = {row["key"]: row for row in summary["panels"]}
    assert panels["codex:1"]["status_counts"]["protein_partial"] == 1
    assert panels["slide_seq:2"]["outside_image"] == 1
    with (output / "regions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 12
    zero = next(row for row in rows if row["observation_id"] == "slide_seq:1:he_region:1:0")
    assert zero["target_exposure"] == "0.0"
    assert json.loads(zero["normalization_counts"]) == {"EPCAM": 0.0, "KRT8": 0.0}
    uncovered = next(row for row in rows if row["status"] == "uncovered")
    assert uncovered["target_exposure"] == ""  # Missing coverage is not an observed zero.
    page = (output / "index.html").read_text()
    assert 'transform="translate(0 20) scale(1 -1)"' in page
    assert 'id="lower-origin"' in page
    assert 'src="https:' not in page
    encoded = re.search(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", page).group(1)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as thumbnail:
        pixels = np.asarray(thumbnail)
    # Pixels remain unflipped; SVG displays row 0 at y=0 along the bottom.
    assert pixels[2, 2, 0] > pixels[2, 2, 2]
    assert pixels[-3, 2, 2] > pixels[-3, 2, 0]
    assert not summary["scientific_qualification"]


def test_prepared_grid_reuse_keeps_selection_when_target_exposure_changes(
    published_cohort, monkeypatch
):
    import anndata

    root = published_cohort
    prepared = cohort.prepare_cohort_inputs(root, "313-932", region_size=10, max_states=2)

    def no_reprepare(*args, **kwargs):
        pytest.fail("prepared feature arm must be reused")

    monkeypatch.setattr(inspection, "prepare_cohort_inputs", no_reprepare)
    first = _write(root, root.parent / (root.name + "_before"), prepared=prepared)
    path = root / "working/registered/313-932/slide_seq.h5ad"
    target = anndata.read_h5ad(path)
    target.obsm["counts"] = target.obsm["counts"].multiply(0).tocsr()
    target.write_h5ad(path)
    _refresh_manifest(root, path)
    second = _write(root, root.parent / (root.name + "_after"), prepared=prepared)
    assert first["n_selected_regions"] == second["n_selected_regions"] == 4
    assert first["grid_settings"] == second["grid_settings"]
    assert second["retained_target_observation_ids"] == []
    assert len(second["excluded_target_observations"]) == 4
    assert first["uncovered_regions"] == second["uncovered_regions"]


def test_report_cannot_write_into_source_or_replace_existing_output(published_cohort):
    root = published_cohort
    with pytest.raises(ValueError, match="immutable dataset root"):
        _write(root, root / "inspection_new")
    with pytest.raises(FileExistsError):
        _write(root, root.parent)


@pytest.mark.parametrize("legacy", [False, True])
def test_coordinate_reader_supports_published_legacy_category_references(tmp_path, legacy):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "coordinates.h5ad"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs["x"], obs["y"] = [1.0, 11.0], [1.0, 1.0]
        if legacy:
            categories = obs.create_group("__categories").create_dataset(
                "replicate", data=np.array(["1", "2"], dtype=h5py.string_dtype())
            )
            obs["replicate"] = [0, 1]
            obs["replicate"].attrs["categories"] = categories.ref
        else:
            group = obs.create_group("replicate")
            group["codes"] = [0, 1]
            group["categories"] = np.array(["1", "2"], dtype=h5py.string_dtype())
    grid = build_region_grid(
        np.full((10, 20, 3), [180, 60, 100], dtype=np.uint8), bin_size_pixels=10
    )
    _, replicates, mapped, inside = inspection._coordinates(path, {"replicate_id": "1;2"}, grid)
    assert list(replicates) == ["1", "2"]
    np.testing.assert_array_equal(mapped, [0, 1])
    assert inside.all()
