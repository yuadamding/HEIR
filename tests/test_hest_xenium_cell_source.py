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

from heir.data.study_manifest import current_git_commit, freeze_manifest_content


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_hest_xenium_cell_source.py"
    spec = importlib.util.spec_from_file_location("build_hest_xenium_cell_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()
REPOSITORY_ROOT = Path(__file__).parents[1]
ENCODER_MANIFEST_PATH = REPOSITORY_ROOT / "manifests" / "encoders" / "uni2h.json"
CROP_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "crops" / "hest_crop_ladder.json"


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
    feature_width = 1536

    def __init__(self, manifest_sha256: str):
        self.manifest_sha256 = manifest_sha256

    def encode(self, patches: np.ndarray) -> np.ndarray:
        mean = patches.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return np.repeat(mean[:, None], self.feature_width, axis=1)


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
    x_locations = []
    y_locations = []
    he_x_locations = []
    he_y_locations = []
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
                x_locations.append(float(x) * builder.SOURCE_MPP)
                y_locations.append(float(y) * builder.SOURCE_MPP)
                he_x_locations.append(float(x))
                he_y_locations.append(float(y))
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
                x_locations.append(float(x) * builder.SOURCE_MPP)
                y_locations.append(float(y) * builder.SOURCE_MPP)
                he_x_locations.append(float(x))
                he_y_locations.append(float(y))
                next_transcript += 1
            for control_index in range(10):
                transcript_cell_ids.append(cell_id)
                transcript_genes.append("NegControlCodeword_%04d" % control_index)
                transcript_ids.append(next_transcript)
                qvs.append(40.0)
                overlaps_nucleus.append(1)
                x_locations.append((float(x) + control_index * 0.001) * builder.SOURCE_MPP)
                y_locations.append((float(y) + control_index * 0.001) * builder.SOURCE_MPP)
                he_x_locations.append(float(x) + control_index * 0.001)
                he_y_locations.append(float(y) + control_index * 0.001)
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
                "x_location": x_locations,
                "y_location": y_locations,
                "he_x": he_x_locations,
                "he_y": he_y_locations,
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
        "pixel_size_um": builder.SOURCE_MPP,
        "wsi": declaration(wsi),
        "transcripts": declaration(transcripts),
        "cell_seg": declaration(cell_seg),
        "nucleus_seg": declaration(nucleus_seg),
        "cellvit_seg": declaration(cellvit_seg),
    }


def _protocol(samples: list[dict[str, object]], development: list[str], locked: list[str]):
    encoder_manifest_sha256 = _sha256(ENCODER_MANIFEST_PATH)
    crop_manifest_sha256 = _sha256(CROP_MANIFEST_PATH)
    return {
        "schema": builder.PROTOCOL_SCHEMA,
        "scientific_scope": "registered_cell_local_context_112um_association",
        "g2_claim_scope": "registered_cell_local_context_112um",
        "authorizes_nucleus_intrinsic_claim": False,
        "dataset_repo": builder.DATASET_REPO,
        "dataset_revision": builder.DATASET_REVISION,
        "encoder_manifest_sha256": encoder_manifest_sha256,
        "crop_manifest_sha256": crop_manifest_sha256,
        "normalization": "log1p_cpm_10000",
        "assay": "Xenium",
        "observation_level": "cell",
        "target_construction": "nucleus_overlapping_xenium_transcripts",
        "registration_method": "native_xenium_cell_id_join",
        "disease_estimands": ["disease_inclusive", "disease_adjusted"],
        "nuisance_fields": [
            "log1p_library_size",
            "section_id",
            "disease_status",
            "site_id",
            "batch_id",
            "stain_quality",
            "nuclear_morphology",
            "cell_morphology",
            "local_density",
            "boundary_position",
            "smooth_spatial_basis",
        ],
        "development_donors": development,
        "locked_test_donors": locked,
        "broad_type_names": list(builder.TYPE_NAMES),
        "type_markers": {
            "Endothelial": ["TYPE_C"],
            "Epithelial": ["TYPE_A"],
            "Immune": ["TYPE_B"],
            "Mesenchymal": ["TYPE_D"],
        },
        "fine_type_marker_gene_ids": ["TYPE_FINE"],
        "fine_type_annotation_provenance": {
            "source_field": "final_CT",
            "source_export": "GSE250346_corrected_Seurat_metadata",
            "exact_annotation_marker_panel_available": False,
            "exclusion_policy": "conservative_predeclared_proxy",
            "establishes_full_target_independence": False,
            "limitation": "synthetic fixture",
        },
        "gene_ids": ["G1", "G2"],
        "minimum_transcripts_per_cell": 10,
        "minimum_transcript_qv": 20.0,
        "minimum_reference_cells_per_donor_type": 1,
        "minimum_evaluation_cells_per_donor_type": 1,
        "minimum_development_donors_per_fine_type": 1,
        "minimum_locked_donors_per_fine_type": 1,
        "excluded_feature_prefixes": list(builder.CONTROL_PREFIXES),
        "spatial_block_um": 32.0 * builder.SOURCE_MPP,
        "spatial_roi_um": 8.0 * builder.SOURCE_MPP,
        "opposite_pool_guard_um": 4.0 * builder.SOURCE_MPP,
        "cellvit_sensitivity_radius_um": 1.0,
        "cellvit_class_names": ["Epithelial", "Inflammatory"],
        "maximum_affine_registration_residual_p95_um": 0.000001,
        "maximum_annotation_nucleus_distance_p95_um": 0.000001,
        "maximum_registration_outlier_fraction": 0.01,
        "maximum_crop_padding_fraction": 0.99,
        "pool_assignment_salt": "synthetic-frozen-v1",
        "reference_splits": {
            "primary_split_id": "primary",
            "selection_unit": "spatial_block",
            "primary_evaluation_rows_fixed": True,
            "alternate_splits": [
                {
                    "split_id": "reference_hash_fold_0",
                    "salt": "synthetic-reference-fold-0",
                    "initial_reference_retention_fraction": 0.8,
                },
                {
                    "split_id": "reference_hash_fold_1",
                    "salt": "synthetic-reference-fold-1",
                    "initial_reference_retention_fraction": 0.8,
                },
            ],
        },
        "transcript_split_salt": "synthetic-split-v1",
        "target_programs": {"synthetic_program": ["G1", "G2"]},
        "samples": samples,
    }


def _write_annotations(path: Path) -> None:
    columns = (
        "hest_id",
        "sample",
        "patient",
        "cell_id",
        "sample_type",
        "sample_affect",
        "disease_status",
        "tma",
        "run",
        "final_CT",
        "final_lineage",
        "x_centroid",
        "y_centroid",
        "nCount_RNA",
        "nFeature_RNA",
        "perc_negcontrolorunassigned",
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
                        "synthetic_site",
                        "Less Affected",
                        "Disease",
                        "TMA1",
                        "Run1",
                        "Synthetic epithelial" if type_index == 0 else "Synthetic immune",
                        lineage,
                        str(x * builder.SOURCE_MPP),
                        str(y * builder.SOURCE_MPP),
                        "16",
                        "4",
                        "0.5",
                    )
                    handle.write("\t".join(values) + "\n")


def test_wkb_centroid_and_spatial_pool_are_deterministic() -> None:
    np.testing.assert_allclose(builder._polygon_centroid(_wkb_square(17.0, 23.0)), (17.0, 23.0))
    first = builder._spatial_identity(
        "NCBI1",
        (48.0, 16.0),
        builder.SOURCE_MPP,
        block_um=32.0 * builder.SOURCE_MPP,
        roi_um=8.0 * builder.SOURCE_MPP,
        guard_um=4.0 * builder.SOURCE_MPP,
        salt="frozen",
    )
    second = builder._spatial_identity(
        "NCBI1",
        (48.0, 16.0),
        builder.SOURCE_MPP,
        block_um=32.0 * builder.SOURCE_MPP,
        roi_um=8.0 * builder.SOURCE_MPP,
        guard_um=4.0 * builder.SOURCE_MPP,
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
    draft = json.loads(
        (
            REPOSITORY_ROOT
            / "manifests"
            / "studies"
            / "hest_lung_measurement_development.draft.json"
        ).read_text(encoding="utf-8")
    )
    draft["analysis_plan_sha256"] = _sha256(protocol_path)
    draft["candidate_target_gene_panel_sha256"] = builder._canonical_sha256(["G1", "G2"])
    marker_sha256 = builder._canonical_sha256(["TYPE_FINE"])
    draft["type_marker_panel_sha256"] = marker_sha256
    draft["label_target_independence"]["marker_panel_sha256"] = marker_sha256
    locked_manifest = freeze_manifest_content(
        draft,
        git_commit=current_git_commit(REPOSITORY_ROOT),
        container_digest="sha256:" + "1" * 64,
    )
    study_manifest_path = tmp_path / "measurement-study.locked.json"
    study_manifest_path.write_text(json.dumps(locked_manifest), encoding="utf-8")
    for sample in samples:
        if sample["donor_id"] in locked:
            for field in ("wsi", "transcripts", "cell_seg", "nucleus_seg", "cellvit_seg"):
                (tmp_path / sample[field]["path"]).unlink()
    output = tmp_path / "source.npz"
    plan_output = tmp_path / "preparation-plan.json"
    qc_output = tmp_path / "qc.json"
    encoder_manifest = builder._load_encoder_manifest(ENCODER_MANIFEST_PATH)
    builder.build_source(
        protocol_path,
        study_manifest_path,
        ENCODER_MANIFEST_PATH,
        CROP_MANIFEST_PATH,
        tmp_path,
        tmp_path / "unused-model",
        output,
        plan_output,
        qc_output,
        device="cpu",
        batch_size=3,
        encoder=_FixtureEncoder(encoder_manifest.sha256),
    )
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["schema_version"]) == builder.SOURCE_SCHEMA
        assert len(archive["observation_ids"]) == 48
        assert len(set(archive["observation_ids"].astype(str))) == 48
        assert set(archive["split_ids"].astype(str)) == {"development"}
        assert str(archive["study_stage"]) == "measurement_development"
        assert str(archive["source_scope"]) == "development_donors_only"
        assert not bool(archive["locked_donor_outcomes_materialized"])
        assert str(archive["study_manifest_sha256"]) == _sha256(study_manifest_path)
        assert set(archive["pool_roles"].astype(str)) == {"reference", "evaluation"}
        assert tuple(archive["reference_split_ids"].astype(str)) == (
            "primary",
            "reference_hash_fold_0",
            "reference_hash_fold_1",
        )
        assert archive["pool_roles_by_split"].shape == (48, 3)
        primary_evaluation = archive["pool_roles"].astype(str) == "evaluation"
        assert np.all(
            archive["pool_roles_by_split"].astype(str)[primary_evaluation] == "evaluation"
        )
        assert archive["frozen_features"].shape == (48, encoder_manifest.feature_width)
        assert archive["image_features"].shape == (48, 18, encoder_manifest.feature_width)
        assert tuple(archive["crop_ids"].astype(str)) == (
            "crop_112um",
            "nucleus_mask_only",
            "nucleus_mask_mean_fill_112um",
            "nucleus_mask_blurred_112um",
            "nucleus_shape_random_location_mean_fill_112um",
            "cell_mask_only",
            "cell_mask_mean_fill_112um",
            "cell_mask_blurred_112um",
            "cell_shape_random_location_mean_fill_112um",
            "context_ring_32_to_112um",
            "context_ring_64_to_112um",
            "target_cell_removed_112um",
            "target_cell_removed_mean_fill_112um",
            "target_cell_removed_blurred_112um",
            "random_location_cell_removed_mean_fill_112um",
            "crop_32um",
            "crop_64um",
            "blank_patch",
        )
        assert archive["crop_padding_fractions"].shape == (48, 18)
        assert archive["crop_mask_fractions"].shape == (48, 18)
        assert archive["crop_roles"].shape == (18,)
        assert archive["crop_comparison_families"].shape == (18,)
        common = np.isin(
            archive["crop_comparison_families"].astype(str),
            ["intrinsic_common_canvas", "mask_artifact_control", "context_control"],
        )
        np.testing.assert_array_equal(archive["crop_diameters_um"][common], 112.0)
        np.testing.assert_allclose(archive["crop_effective_mpp"][common], 0.5)
        assert set(
            archive["crop_ids"][
                archive["crop_comparison_families"].astype(str) == "resolution_sensitivity"
            ].astype(str)
        ) == {"crop_32um", "crop_64um"}
        crop_metadata = {
            crop_id: (mask_mode, fill_mode)
            for crop_id, mask_mode, fill_mode in zip(
                archive["crop_ids"].astype(str),
                archive["crop_mask_modes"].astype(str),
                archive["crop_fill_modes"].astype(str),
            )
        }
        assert crop_metadata["nucleus_mask_mean_fill_112um"] == (
            "keep_nucleus",
            "mean_color",
        )
        assert crop_metadata["nucleus_mask_blurred_112um"] == (
            "keep_nucleus",
            "blurred",
        )
        assert crop_metadata["cell_shape_random_location_mean_fill_112um"] == (
            "random_keep_cell",
            "mean_color",
        )
        assert crop_metadata["random_location_cell_removed_mean_fill_112um"] == (
            "random_remove_cell",
            "mean_color",
        )
        assert archive["molecular_targets"].shape == (48, 2)
        np.testing.assert_allclose(archive["molecular_targets"], np.log1p(2500.0))
        np.testing.assert_allclose(
            archive["whole_cell_molecular_targets"][:, 0], np.log1p(10_000.0 * 5.0 / 17.0)
        )
        np.testing.assert_allclose(
            archive["whole_cell_molecular_targets"][:, 1], np.log1p(10_000.0 * 4.0 / 17.0)
        )
        np.testing.assert_array_equal(archive["nucleus_library_sizes"], 16)
        np.testing.assert_array_equal(archive["whole_cell_library_sizes"], 17)
        np.testing.assert_array_equal(
            archive["nucleus_library_size_half_a"] + archive["nucleus_library_size_half_b"],
            archive["nucleus_library_sizes"],
        )
        np.testing.assert_array_equal(
            archive["whole_cell_library_size_half_a"] + archive["whole_cell_library_size_half_b"],
            archive["whole_cell_library_sizes"],
        )
        np.testing.assert_array_equal(
            archive["nucleus_target_counts_half_a"] + archive["nucleus_target_counts_half_b"],
            archive["nucleus_target_counts"],
        )
        np.testing.assert_array_equal(
            archive["whole_cell_target_counts_half_a"] + archive["whole_cell_target_counts_half_b"],
            archive["whole_cell_target_counts"],
        )
        assert str(archive["transcript_split_method"]) == "sha256-final-byte-lsb-v1"
        assert (
            str(archive["transcript_split_salt_sha256"])
            == hashlib.sha256(b"synthetic-split-v1").hexdigest()
        )
        assert int(archive["eligible_target_transcripts"]) == 48 * 9
        assert tuple(archive["program_names"].astype(str)) == ("synthetic_program",)
        np.testing.assert_array_equal(archive["program_gene_membership"], [[True, True]])
        assert len(archive["planned_stratum_ids"]) == 24
        assert str(archive["planned_stratum_manifest_sha256"]) == builder._canonical_sha256(
            list(archive["planned_stratum_ids"].astype(str))
        )
        assert archive["coordinate_features"].shape == (48, 24)
        assert archive["frozen_feature_names"].shape == (encoder_manifest.feature_width,)
        assert archive["coordinate_feature_names"].shape == (24,)
        assert archive["stain_features"].shape == (48, 70)
        assert archive["stain_feature_names"].shape == (70,)
        assert archive["composition_features"].shape == (48, 10)
        assert archive["composition_feature_names"].shape == (10,)
        assert archive["nuclear_morphometric_features"].shape == (48, 34)
        assert archive["cell_morphometric_features"].shape == (48, 33)
        assert archive["classical_morphology_features"].shape == (48, 67)
        assert np.isfinite(archive["stain_features"]).all()
        assert np.isfinite(archive["classical_morphology_features"]).all()
        assert tuple(archive["fine_type_marker_gene_ids"].astype(str)) == ("TYPE_FINE",)
        assert str(archive["fine_type_marker_panel_sha256"]) == builder._canonical_sha256(
            ["TYPE_FINE"]
        )
        assert set(archive["disease_estimands"].astype(str)) == {
            "disease_inclusive",
            "disease_adjusted",
        }
        assert set(archive["type_labels"].tolist()) == {0, 1}
        assert set(archive["type_names"].astype(str)) == {
            "Synthetic epithelial",
            "Synthetic immune",
        }
        assert set(archive["broad_type_labels"].tolist()) == {1, 2}
        assert tuple(archive["broad_type_names"].astype(str)) == builder.TYPE_NAMES
        assert archive["registration_qc_features"].shape == (48, 23)
        assert archive["registration_qc_feature_names"].shape == (23,)
        assert set(archive["section_ids"].astype(str)) == {
            sample_id
            for sample_id, (donor_id, _) in builder.SECTION_IDENTITIES.items()
            if donor_id in development
        }
        assert set(archive["disease_statuses"].astype(str)) == {"Disease"}
        assert set(archive["site_ids"].astype(str)) == {"synthetic_site"}
        assert set(archive["batch_ids"].astype(str)) == {"TMA1:Run1"}
        assert str(archive["encoder_name"]) == encoder_manifest.repository
        assert str(archive["target_construction"]) == ("nucleus_overlapping_xenium_transcripts")
        assert str(archive["primary_crop_id"]) == "crop_112um"
        assert str(archive["crop_role"]) == "registered_cell_local_context_112um"
        assert float(archive["crop_diameter_um"]) == 112.0
        assert str(archive["mask_mode"]) == "none"
        assert not bool(archive["authorizes_nucleus_intrinsic_claim"])
        assert not bool(archive["nucleus_hypothesis_tested"])
        assert not bool(archive["cell_intrinsic_hypothesis_tested"])
        assert str(archive["g2_claim_scope"]) == "registered_cell_local_context_112um"
        assert archive["registration_qc_pass"].all()
        np.testing.assert_array_equal(archive["registration_cardinality"], 1)
        assert archive["target_qc_pass"].all()
        assert archive["crop_qc_pass"].all()
        assert archive["cellvit_sensitivity_features"].shape == (48, 2)
        assert archive["cellvit_context_features"].shape == (48, 2)
        assert set(archive["cellvit_sensitivity_feature_names"].astype(str)) == {
            "cellvit_log1p_count_Epithelial",
            "cellvit_log1p_count_Inflammatory",
        }
        provenance = json.loads(str(archive["provenance_json"]))
        assert provenance["native_xenium_registration_only"] is True
        assert provenance["cellvit_target_registration"] is False
        assert provenance["authorizes_nucleus_intrinsic_claim"] is False
        required_row_fields = {
            "observation_id",
            "donor_id",
            "patient_id",
            "section_id",
            "source_sample_id",
            "cell_id",
            "broad_type_label",
            "fine_type_label",
            "disease_state",
            "site_id",
            "batch_id",
            "block_id",
            "roi_id",
            "pool_role",
            "x_coordinate_um",
            "y_coordinate_um",
            "cell_centroid_x_um",
            "cell_centroid_y_um",
            "nucleus_centroid_x_um",
            "nucleus_centroid_y_um",
            "annotation_centroid_x_um",
            "annotation_centroid_y_um",
            "registration_distance_um",
            "cell_area_um2",
            "nucleus_area_um2",
            "library_size",
            "detected_target_genes",
            "transcript_qv_summary",
            "target_qc_pass",
            "registration_qc_pass",
        }
        assert required_row_fields <= set(archive.files)
        required_matrices = {
            "nucleus_target_counts",
            "whole_cell_target_counts",
            "normalized_nucleus_targets",
            "normalized_whole_cell_targets",
            "coordinate_features",
            "stain_features",
            "nuclear_morphometric_features",
            "cell_morphometric_features",
            "cellvit_context_features",
            "image_features_by_crop_and_encoder",
            "pool_roles_by_split",
        }
        assert required_matrices <= set(archive.files)
        assert "registration_is_one_to_one" not in archive.files
        assert "study_manifest_sha256" in archive.files
    plan = json.loads(plan_output.read_text(encoding="utf-8"))
    assert plan["source_observations_sha256"] == _sha256(output)
    assert plan["source_schema"] == builder.SOURCE_SCHEMA
    assert plan["study_stage"] == "measurement_development"
    assert plan["source_scope"] == "development_donors_only"
    assert plan["locked_donor_outcomes_materialized"] is False
    assert plan["encoder_name"] == encoder_manifest.repository
    assert plan["target_construction"] == "nucleus_overlapping_xenium_transcripts"
    assert plan["type_names"] == ["Synthetic epithelial", "Synthetic immune"]
    assert plan["broad_type_names"] == list(builder.TYPE_NAMES)
    assert len(plan["frozen_feature_names"]) == encoder_manifest.feature_width
    assert plan["coordinate_feature_names"] == list(builder.SPATIAL_FEATURE_NAMES)
    assert plan["crop_metadata"]["primary_crop_id"] == "crop_112um"
    assert plan["crop_metadata"]["crop_role"] == ("registered_cell_local_context_112um")
    assert plan["crop_metadata"]["crop_diameter_um"] == 112.0
    assert plan["crop_metadata"]["source_mpp"] == builder.SOURCE_MPP
    assert plan["crop_metadata"]["mask_mode"] == "none"
    assert plan["crop_metadata"]["fill_mode"] == "none"
    assert len(plan["crop_metadata"]["variants"]) == 18
    assert plan["target"]["primary"] == "nucleus_overlapping_xenium_transcripts"
    assert plan["target"]["secondary"] == "whole_cell_xenium_transcripts"
    assert plan["crop_ids"] == list(np.load(output, allow_pickle=False)["crop_ids"].astype(str))
    assert plan["gene_ids"] == list(np.load(output, allow_pickle=False)["gene_ids"].astype(str))
    assert plan["type_names"] == list(np.load(output, allow_pickle=False)["type_names"].astype(str))
    assert plan["authorizes_nucleus_intrinsic_claim"] is False
    assert plan["nucleus_hypothesis_tested"] is False
    assert plan["cell_intrinsic_hypothesis_tested"] is False
    assert plan["fine_type_marker_panel_sha256"] == builder._canonical_sha256(["TYPE_FINE"])
    assert plan["reference_splits"] == {
        "primary_split_id": "primary",
        "split_ids": ["primary", "reference_hash_fold_0", "reference_hash_fold_1"],
        "primary_evaluation_rows_fixed": True,
        "selection_unit": "spatial_block",
    }
    qc = json.loads(qc_output.read_text(encoding="utf-8"))
    assert qc["pass"] is True
    assert qc["study_stage"] == "measurement_development"
    assert qc["source_scope"] == "development_donors_only"
    assert qc["locked_donor_outcomes_materialized"] is False
    assert qc["source_observations"]["sha256"] == _sha256(output)
    assert qc["preparation_plan"]["sha256"] == _sha256(plan_output)
    assert qc["crops"]["g2_claim_scope"] == "registered_cell_local_context_112um"
    assert qc["crops"]["inpainting_substitute"] == "blurred_replacement"
    assert qc["feature_families"]["stain_quality_columns"] == 70
    assert qc["feature_families"]["nuclear_morphology_columns"] == 34
    assert qc["feature_families"]["cell_morphology_columns"] == 33
    assert set(qc["reference_evaluation_balance"]) == set(builder.DEVELOPMENT_DONORS)
    assert (
        qc["targets"]["whole_cell_eligible_transcripts"]
        > qc["targets"]["nucleus_eligible_transcripts"]
    )


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
    encoder_manifest = builder._load_encoder_manifest(ENCODER_MANIFEST_PATH)
    crop_manifest = builder._load_crop_manifest(CROP_MANIFEST_PATH)
    builder._validate_protocol(protocol, encoder_manifest, crop_manifest)
    with pytest.raises(ValueError, match="dataset_revision differs"):
        builder._validate_protocol(
            {**protocol, "dataset_revision": "main"}, encoder_manifest, crop_manifest
        )
    wrong_samples = [dict(value) for value in samples]
    wrong_samples[0]["donor_id"] = "Patient 1"
    with pytest.raises(ValueError, match="frozen partitions|corrected true-donor identity"):
        builder._validate_protocol(
            {**protocol, "samples": wrong_samples}, encoder_manifest, crop_manifest
        )
    with pytest.raises(ValueError, match="broad/fine marker and evaluation genes"):
        builder._validate_protocol(
            {**protocol, "fine_type_marker_gene_ids": ["G1"]},
            encoder_manifest,
            crop_manifest,
        )
    missing = builder.InputFile("missing.tif", "1" * 64)
    with pytest.raises(ValueError, match="missing or differs"):
        builder._resolve_input(tmp_path.resolve(), missing)


def test_builder_cannot_open_confirmatory_rows_from_a_draft_manifest(tmp_path: Path) -> None:
    draft = REPOSITORY_ROOT / "manifests" / "studies" / "hest_lung_cell_association.draft.json"
    with pytest.raises(ValueError, match="must have status locked"):
        builder.build_source(
            tmp_path / "unread-protocol.json",
            draft,
            ENCODER_MANIFEST_PATH,
            CROP_MANIFEST_PATH,
            tmp_path / "unread-data",
            tmp_path / "unread-model",
            tmp_path / "source.npz",
            tmp_path / "plan.json",
            tmp_path / "qc.json",
            device="cpu",
        )


def test_expression_aggregation_rejects_duplicate_transcript_ids(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "duplicate-transcripts.parquet"
    parquet.write_table(
        pa.table(
            {
                "transcript_id": [7, 7],
                "cell_id": ["cell-1", "cell-1"],
                "feature_name": ["G1", "G1"],
                "qv": [40.0, 40.0],
                "overlaps_nucleus": [1, 1],
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="duplicate transcript_id"):
        builder._aggregate_expression(
            path,
            ["cell-1"],
            ["cell-1"],
            ["G1"],
            ["G1"],
            minimum_qv=20.0,
            excluded_prefixes=builder.CONTROL_PREFIXES,
            split_salt="synthetic-split-v1",
        )


def test_encoder_manifests_record_access_status_and_handcrafted_control() -> None:
    from heir.features import HandcraftedPatchEncoder, load_encoder_manifest

    uni2h = load_encoder_manifest(ENCODER_MANIFEST_PATH)
    assert uni2h.available
    assert uni2h.repository == "MahmoodLab/UNI2-h"
    inaccessible = REPOSITORY_ROOT / "manifests" / "encoders" / "hoptimus1.inaccessible.json"
    status = load_encoder_manifest(inaccessible, require_available=False)
    assert not status.available
    assert status.checkpoint_sha256 == "0" * 64
    with pytest.raises(ValueError, match="inaccessible"):
        load_encoder_manifest(inaccessible)
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    features = HandcraftedPatchEncoder().encode(patches)
    assert features.shape == (2, 12)
    assert np.isfinite(features).all()
