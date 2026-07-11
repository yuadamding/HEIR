import csv
import json

import numpy as np
import pytest

from heir.cli import main
from heir.data import (
    assign_nuclei_to_visium_capture_area,
    filter_nucleus_csv_to_visium,
)
from heir.image import load_nuclei


def _write_geometry(tmp_path):
    nuclei = tmp_path / "nuclei.csv"
    with nuclei.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["nucleus_id", "x", "y", "area", "review_note"])
        writer.writerow(["cell-001", "10", "10", "40.5", "inside, reviewed"])
        writer.writerow(["cell-002", "14.9", "10", "42", "disk boundary"])
        writer.writerow(["cell-003", "50", "50", "43", "out-tissue spot"])
        writer.writerow(["cell-004", "100", "100", "44", "outside"])
    positions = tmp_path / "tissue_positions.csv"
    positions.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "spot-in,1,0,0,10,10\n"
        "spot-out,0,0,1,50,50\n",
        encoding="utf-8",
    )
    scales = tmp_path / "scalefactors_json.json"
    scales.write_text(json.dumps({"spot_diameter_fullres": 10.0}), encoding="utf-8")
    return nuclei, positions, scales


def test_filter_cli_preserves_csv_and_records_geometry_only_assignment(tmp_path):
    nuclei, positions, scales = _write_geometry(tmp_path)
    filtered = tmp_path / "filtered.csv"
    assignment = tmp_path / "capture_assignment.npz"
    provenance = tmp_path / "capture_provenance.json"

    assert (
        main(
            [
                "filter-nuclei-to-visium",
                "--nuclei",
                str(nuclei),
                "--positions",
                str(positions),
                "--scalefactors",
                str(scales),
                "--output",
                str(filtered),
                "--assignment-output",
                str(assignment),
                "--provenance-output",
                str(provenance),
            ]
        )
        == 0
    )

    with nuclei.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.reader(handle))
    with filtered.open(newline="", encoding="utf-8") as handle:
        filtered_rows = list(csv.reader(handle))
    assert filtered_rows == [source_rows[0], source_rows[1], source_rows[2]]
    assert filtered_rows[0] == ["nucleus_id", "x", "y", "area", "review_note"]
    assert filtered_rows[1][0] == "cell-001"
    assert filtered_rows[1][-1] == "inside, reviewed"

    with np.load(assignment, allow_pickle=False) as archive:
        assert str(archive["__contract__"].item()) == "heir.visium_capture_filter"
        assert int(archive["__version__"].item()) == 1
        assert bool(archive["geometry_only"].item())
        assert not bool(archive["target_expression_accessed"].item())
        np.testing.assert_array_equal(
            archive["source_nucleus_ids"],
            ["cell-001", "cell-002", "cell-003", "cell-004"],
        )
        np.testing.assert_array_equal(archive["source_nucleus_spot_index"], [0, 0, -1, -1])
        np.testing.assert_array_equal(archive["source_row_index"], [0, 1])
        np.testing.assert_array_equal(archive["nucleus_ids"], ["cell-001", "cell-002"])
        np.testing.assert_array_equal(archive["spot_ids"], ["spot-in"])
        np.testing.assert_allclose(archive["spot_coordinates_px"], [[10.0, 10.0]])
        assert all(value.dtype != object for value in archive.values())

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert payload["geometry_only"] is True
    assert payload["target_expression_accessed"] is False
    assert payload["source_nuclei"] == 4
    assert payload["retained_nuclei"] == 2
    assert payload["excluded_nuclei"] == 2
    assert payload["in_tissue_spots"] == 1
    assert payload["nucleus_columns"] == source_rows[0]
    assert set(payload["inputs"]) == {"nuclei", "positions", "scalefactors"}
    assert all(len(item["sha256"]) == 64 for item in payload["inputs"].values())


def test_capture_assignment_scales_only_visium_geometry(tmp_path):
    nuclei_path = tmp_path / "nuclei.csv"
    nuclei_path.write_text("nucleus_id,x,y\nn0,23.9,20\nn1,25,20\n", encoding="utf-8")
    positions = tmp_path / "positions.csv"
    positions.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "spot,1,0,0,10,10\n",
        encoding="utf-8",
    )
    scales = tmp_path / "scales.json"
    scales.write_text(json.dumps({"spot_diameter_fullres": 4.0}), encoding="utf-8")
    result = assign_nuclei_to_visium_capture_area(
        load_nuclei(nuclei_path),
        positions_path=positions,
        scalefactors_path=scales,
        coordinate_scale=2.0,
    )
    np.testing.assert_allclose(result.spot_coordinates_px, [[20.0, 20.0]])
    assert result.spot_radius_px == 4.0
    np.testing.assert_array_equal(result.source_spot_index, [0, -1])
    np.testing.assert_array_equal(result.retained_source_index, [0])


def test_capture_filter_fails_closed_and_does_not_overwrite(tmp_path):
    nuclei, positions, scales = _write_geometry(tmp_path)
    filtered = tmp_path / "filtered.csv"
    assignment = tmp_path / "assignment.npz"
    provenance = tmp_path / "provenance.json"
    filter_nucleus_csv_to_visium(
        nuclei_path=nuclei,
        positions_path=positions,
        scalefactors_path=scales,
        filtered_csv_path=filtered,
        assignment_npz_path=assignment,
        provenance_json_path=provenance,
    )
    with pytest.raises(FileExistsError):
        filter_nucleus_csv_to_visium(
            nuclei_path=nuclei,
            positions_path=positions,
            scalefactors_path=scales,
            filtered_csv_path=filtered,
            assignment_npz_path=assignment,
            provenance_json_path=provenance,
        )
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="cannot replace source"):
        filter_nucleus_csv_to_visium(
            nuclei_path=nuclei,
            positions_path=positions,
            scalefactors_path=scales,
            filtered_csv_path=nuclei,
            assignment_npz_path=tmp_path / "different_assignment.npz",
            provenance_json_path=tmp_path / "different_provenance.json",
            overwrite=True,
        )

    no_tissue = tmp_path / "no_tissue.csv"
    no_tissue.write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "spot,0,0,0,10,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no in-tissue spots"):
        assign_nuclei_to_visium_capture_area(
            load_nuclei(nuclei),
            positions_path=no_tissue,
            scalefactors_path=scales,
        )
