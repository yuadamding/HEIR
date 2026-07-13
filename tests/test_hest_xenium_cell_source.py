from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_hest_xenium_cell_source.py"
    spec = importlib.util.spec_from_file_location("build_hest_xenium_cell_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _wkb_square(x: float, y: float, half: float = 2.0) -> bytes:
    points = (
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
        (x - half, y - half),
    )
    return b"".join(
        (b"\x01", struct.pack("<I", 3), struct.pack("<I", 1), struct.pack("<I", len(points)))
        + tuple(struct.pack("<dd", *point) for point in points)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role_blocks(sample_id: str, salt: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for block_x in range(8):
        role = builder._block_role(sample_id, block_x, 0, salt)
        result.setdefault(role, block_x)
    assert set(result) == {"reference", "evaluation"}
    return result


class _FixtureEncoder:
    output_width = builder.FEATURE_WIDTH

    def encode(self, patches: np.ndarray) -> np.ndarray:
        mean = patches.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return np.repeat(mean[:, None], self.output_width, axis=1)


def _write_sample(root: Path, sample_id: str, salt: str, image_value: int) -> dict[str, object]:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    tifffile = pytest.importorskip("tifffile")
    blocks = _role_blocks(sample_id, salt)
    cell_ids = []
    cell_geometries = []
    nucleus_geometries = []
    transcript_cell_ids = []
    transcript_genes = []
    transcript_ids = []
    qvs = []
    overlaps_nucleus = []
    cellvit_geometries = []
    cellvit_classes = []
    next_transcript = 0
    for role in ("reference", "evaluation"):
        x = blocks[role] * 32 + 16
        for type_index, y in enumerate((12, 20)):
            cell_id = "%s_%d" % (role, type_index)
            cell_ids.append(cell_id)
            cell_geometries.append(_wkb_square(x, y, half=3.0))
            nucleus_geometries.append(_wkb_square(x, y, half=2.0))
            cellvit_geometries.append(_wkb_square(x, y, half=1.0))
            cellvit_classes.append("Epithelial" if type_index == 0 else "Inflammatory")
            genes = (["TYPE_A"] if type_index == 0 else ["TYPE_B"]) * 8 + ["G1"] * 4 + ["G2"] * 4
            for gene in genes:
                transcript_cell_ids.append(cell_id)
                transcript_genes.append(gene)
                transcript_ids.append(next_transcript)
                qvs.append(40.0)
                overlaps_nucleus.append(1)
                next_transcript += 1
            for gene, qv, overlaps in (
                ("G1", 40.0, 0),
                ("G2", 10.0, 1),
                ("NegControlProbe_1", 40.0, 1),
            ):
                transcript_cell_ids.append(cell_id)
                transcript_genes.append(gene)
                transcript_ids.append(next_transcript)
                qvs.append(qv)
                overlaps_nucleus.append(overlaps)
                next_transcript += 1

    wsi = root / "wsis" / (sample_id + ".tif")
    transcripts = root / "transcripts" / (sample_id + "_transcripts.parquet")
    cell_seg = root / "xenium_seg" / (sample_id + "_xenium_cell_seg.parquet")
    nucleus_seg = root / "xenium_seg" / (sample_id + "_xenium_nucleus_seg.parquet")
    cellvit_seg = root / "cellvit_seg" / (sample_id + "_cellvit_seg.parquet")
    for path in (wsi, transcripts, cell_seg, nucleus_seg, cellvit_seg):
        path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 256, 3), image_value, dtype=np.uint8)
    tifffile.imwrite(wsi, image, photometric="rgb")
    parquet.write_table(
        pa.table({"geometry": cell_geometries, "__index_level_0__": cell_ids}), cell_seg
    )
    parquet.write_table(
        pa.table({"geometry": nucleus_geometries, "__index_level_0__": cell_ids}), nucleus_seg
    )
    parquet.write_table(
        pa.table(
            {
                "transcript_id": transcript_ids,
                "cell_id": transcript_cell_ids,
                "feature_name": transcript_genes,
                "qv": qvs,
                "overlaps_nucleus": overlaps_nucleus,
            }
        ),
        transcripts,
    )
    parquet.write_table(
        pa.table(
            {
                "geometry": cellvit_geometries,
                "class": cellvit_classes,
                "cell_id": list(range(len(cell_ids))),
            }
        ),
        cellvit_seg,
    )

    def declaration(path: Path) -> dict[str, str]:
        return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}

    return {
        "sample_id": sample_id,
        "pixel_size_um": 1.0,
        "wsi": declaration(wsi),
        "transcripts": declaration(transcripts),
        "cell_seg": declaration(cell_seg),
        "nucleus_seg": declaration(nucleus_seg),
        "cellvit_seg": declaration(cellvit_seg),
    }


def _protocol(samples: list[dict[str, object]], development: list[str], locked: list[str]):
    return {
        "schema": builder.PROTOCOL_SCHEMA,
        "scientific_scope": "nucleus_centered_morphology_confirmation",
        "dataset_repo": builder.DATASET_REPO,
        "dataset_revision": builder.DATASET_REVISION,
        "model_repo": builder.MODEL_REPO,
        "model_revision": builder.MODEL_REVISION,
        "model_checkpoint_sha256": builder.MODEL_CHECKPOINT_SHA256,
        "model_input_pixels": 224,
        "model_feature_width": builder.FEATURE_WIDTH,
        "model_mpp": 0.5,
        "model_mean": [0.707223, 0.578729, 0.703617],
        "model_std": [0.211883, 0.230117, 0.177517],
        "normalization": "log1p_cpm_10000",
        "assay": "Xenium",
        "observation_level": "cell",
        "target_construction": "registered_cell_expression",
        "registration_method": "native_xenium_cell_id_join",
        "development_donors": development,
        "locked_test_donors": locked,
        "type_names": list(builder.TYPE_NAMES),
        "type_markers": {
            "Endothelial": ["TYPE_C"],
            "Epithelial": ["TYPE_A"],
            "Immune": ["TYPE_B"],
            "Mesenchymal": ["TYPE_D"],
        },
        "gene_ids": ["G1", "G2"],
        "minimum_transcripts_per_cell": 10,
        "minimum_transcript_qv": 20.0,
        "excluded_feature_prefixes": list(builder.CONTROL_PREFIXES),
        "spatial_block_um": 32.0,
        "spatial_roi_um": 8.0,
        "opposite_pool_guard_um": 4.0,
        "cellvit_sensitivity_radius_um": 5.0,
        "cellvit_class_names": ["Epithelial", "Inflammatory"],
        "pool_assignment_salt": "synthetic-frozen-v1",
        "samples": samples,
    }


def _write_annotations(path: Path) -> None:
    columns = (
        "hest_id",
        "sample",
        "patient",
        "cell_id",
        "final_CT",
        "final_lineage",
        "x_centroid",
        "y_centroid",
        "nCount_RNA",
    )
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(columns) + "\n")
        for sample_id, (donor_id, source_sample) in sorted(builder.SECTION_IDENTITIES.items()):
            blocks = _role_blocks(sample_id, "synthetic-frozen-v1")
            for role in ("reference", "evaluation"):
                x = blocks[role] * 32 + 16
                for type_index, y in enumerate((12, 20)):
                    lineage = "Epithelial" if type_index == 0 else "Immune"
                    values = (
                        sample_id,
                        source_sample,
                        donor_id,
                        "%s_%d" % (role, type_index),
                        "synthetic_subtype",
                        lineage,
                        str(x),
                        str(y),
                        "16",
                    )
                    handle.write("\t".join(values) + "\n")


def test_wkb_centroid_and_spatial_pool_are_deterministic() -> None:
    np.testing.assert_allclose(builder._polygon_centroid(_wkb_square(17.0, 23.0)), (17.0, 23.0))
    first = builder._spatial_identity(
        "NCBI1",
        (48.0, 16.0),
        1.0,
        block_um=32.0,
        roi_um=8.0,
        guard_um=4.0,
        salt="frozen",
    )
    second = builder._spatial_identity(
        "NCBI1",
        (48.0, 16.0),
        1.0,
        block_um=32.0,
        roi_um=8.0,
        guard_um=4.0,
        salt="frozen",
    )
    assert first == second
    assert first.guard_pass is True


def test_builder_creates_registered_cell_source_and_isolates_cellvit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("zarr")
    development = list(builder.DEVELOPMENT_DONORS)
    locked = list(builder.LOCKED_TEST_DONORS)
    samples = []
    for index, (sample_id, (donor, _)) in enumerate(sorted(builder.SECTION_IDENTITIES.items())):
        sample = _write_sample(tmp_path, sample_id, "synthetic-frozen-v1", 20 + index)
        sample["donor_id"] = donor
        samples.append(sample)
    annotation_path = tmp_path / "GSE250346" / "annotations.tsv.gz"
    annotation_path.parent.mkdir(parents=True)
    _write_annotations(annotation_path)
    monkeypatch.setattr(builder, "ANNOTATION_SHA256", _sha256(annotation_path))
    monkeypatch.setattr(builder, "ANNOTATION_ROWS", 80)
    protocol = _protocol(samples, development, locked)
    protocol["annotation_export"] = {
        "path": str(annotation_path.relative_to(tmp_path)),
        "sha256": builder.ANNOTATION_SHA256,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "source.npz"
    builder.build_source(
        protocol_path,
        tmp_path,
        tmp_path / "unused-model",
        output,
        device="cpu",
        batch_size=3,
        encoder=_FixtureEncoder(),
    )
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["schema_version"]) == builder.SOURCE_SCHEMA
        assert len(archive["observation_ids"]) == 80
        assert len(set(archive["observation_ids"].astype(str))) == 80
        assert set(archive["split_ids"].astype(str)) == {"development", "locked_test"}
        assert set(archive["pool_roles"].astype(str)) == {"reference", "evaluation"}
        assert archive["frozen_features"].shape == (80, builder.FEATURE_WIDTH)
        assert archive["molecular_targets"].shape == (80, 2)
        np.testing.assert_allclose(archive["molecular_targets"], np.log1p(2500.0))
        assert archive["coordinate_features"].shape == (80, 5)
        assert archive["stain_features"].shape == (80, 0)
        assert archive["stain_feature_names"].shape == (0,)
        assert archive["composition_features"].shape == (80, 0)
        assert archive["composition_feature_names"].shape == (0,)
        assert set(archive["type_labels"].tolist()) == {1, 2}
        assert archive["registration_is_one_to_one"].all()
        assert archive["cellvit_sensitivity_features"].shape == (80, 2)
        assert set(archive["cellvit_sensitivity_feature_names"].astype(str)) == {
            "cellvit_log1p_count_Epithelial",
            "cellvit_log1p_count_Inflammatory",
        }
        provenance = json.loads(str(archive["provenance_json"]))
        assert provenance["native_xenium_registration_only"] is True
        assert provenance["cellvit_target_registration"] is False


def test_protocol_and_input_hashes_fail_closed(tmp_path: Path) -> None:
    development = list(builder.DEVELOPMENT_DONORS)
    locked = list(builder.LOCKED_TEST_DONORS)
    samples = []
    for index, (sample_id, (donor, _)) in enumerate(sorted(builder.SECTION_IDENTITIES.items())):
        sample = {
            "sample_id": sample_id,
            "donor_id": donor,
            "pixel_size_um": 0.2125,
            "wsi": {"path": "w%d.tif" % index, "sha256": "1" * 64},
            "transcripts": {"path": "t%d.parquet" % index, "sha256": "2" * 64},
            "cell_seg": {"path": "c%d.parquet" % index, "sha256": "3" * 64},
            "nucleus_seg": {"path": "n%d.parquet" % index, "sha256": "4" * 64},
        }
        samples.append(sample)
    protocol = _protocol(samples, development, locked)
    protocol["annotation_export"] = {
        "path": "GSE250346/annotations.tsv.gz",
        "sha256": builder.ANNOTATION_SHA256,
    }
    builder._validate_protocol(protocol)
    with pytest.raises(ValueError, match="dataset_revision differs"):
        builder._validate_protocol({**protocol, "dataset_revision": "main"})
    wrong_samples = [dict(value) for value in samples]
    wrong_samples[0]["donor_id"] = "Patient 1"
    with pytest.raises(ValueError, match="frozen partitions|corrected true-donor identity"):
        builder._validate_protocol({**protocol, "samples": wrong_samples})
    missing = builder.InputFile("missing.tif", "1" * 64)
    with pytest.raises(ValueError, match="missing or differs"):
        builder._resolve_input(tmp_path.resolve(), missing)
