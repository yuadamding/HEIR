from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_hescape_regional_source.py"
    spec = importlib.util.spec_from_file_location("build_hescape_regional_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _minimal_source_payload() -> dict[str, object]:
    rows = 2
    coverage = []
    exclusions = []
    for section_id in sorted(
        section
        for section, donor in builder.SECTION_TO_TRUE_DONOR.items()
        if donor in set(builder.DEVELOPMENT_DONORS)
    ):
        identity = {
            "donor_id": builder.SECTION_TO_TRUE_DONOR[section_id],
            "section_id": section_id,
            "disease_state": "healthy",
            "site_id": "lung",
            "batch_id": section_id,
        }
        for type_index in range(4):
            retained = type_index == 0
            coverage.append(
                {
                    **identity,
                    "type_index": type_index,
                    "labeled_before_guard": 2 if retained else 0,
                    "guard_excluded": 0,
                    "unsupported_stratum_excluded": 0,
                    "retained_reference": int(retained),
                    "retained_evaluation": int(retained),
                }
            )
        exclusions.append(
            {
                **identity,
                "release": 2,
                "zero_library_excluded": 0,
                "ambiguous_niche_excluded": 0,
                "guard_excluded": 0,
                "unsupported_stratum_excluded": 0,
                "retained": 2,
            }
        )
    crop_role = "target_matched_55um_common_mpp"
    crop = builder.CROP_PROTOCOLS[crop_role]
    payload: dict[str, object] = {
        "schema_version": np.asarray(builder.SOURCE_SCHEMA),
        "analysis_scope": np.asarray(
            "development_donors_only_reserved_outcomes_previously_materialized"
        ),
        "reserved_hest_locked_donors": np.asarray(builder.RESERVED_HEST_LOCKED_DONORS),
        "reserved_donor_outcomes_loaded": np.asarray(False),
        "observation_ids": np.asarray(["A", "B"]),
        "donor_ids": np.asarray(["VUILD91", "VUILD91"]),
        "section_ids": np.asarray(["NCBI858", "NCBI858"]),
        "disease_states": np.asarray(["healthy", "healthy"]),
        "site_ids": np.asarray(["lung", "lung"]),
        "batch_ids": np.asarray(["NCBI858", "NCBI858"]),
        "hescape_patient_ids": np.asarray(["Patient 1", "Patient 1"]),
        "block_ids": np.asarray(["B1", "B2"]),
        "roi_ids": np.asarray(["R1", "R2"]),
        "pool_roles": np.asarray(["reference", "evaluation"]),
        "type_labels": np.asarray([0, 1]),
        "gene_ids": np.asarray(["G0", "G1", "G2"]),
        "type_names": np.asarray(builder.TYPE_NAMES),
        "type_marker_gene_ids": np.asarray(["M0", "M1"]),
        "frozen_features": np.vstack(
            (np.zeros(builder.FEATURE_WIDTH), np.ones(builder.FEATURE_WIDTH))
        ),
        "frozen_feature_names": np.asarray(builder.FROZEN_FEATURE_NAMES),
        "stain_features": np.vstack(
            (
                np.zeros(len(builder.STAIN_FEATURE_NAMES)),
                np.ones(len(builder.STAIN_FEATURE_NAMES)),
            )
        ),
        "stain_feature_names": np.asarray(builder.STAIN_FEATURE_NAMES),
        "composition_features": np.ones((rows, len(builder.COMPOSITION_FEATURE_NAMES))),
        "composition_feature_names": np.asarray(builder.COMPOSITION_FEATURE_NAMES),
        "molecular_targets": np.ones((rows, 3)),
        "coordinate_features": np.ones((rows, len(builder.COORDINATE_FEATURE_NAMES))),
        "coordinate_feature_names": np.asarray(builder.COORDINATE_FEATURE_NAMES),
        "technical_covariates": np.ones((rows, len(builder.TECHNICAL_COVARIATE_NAMES))),
        "technical_covariate_names": np.asarray(builder.TECHNICAL_COVARIATE_NAMES),
        "registration_is_one_to_one": np.ones(rows, dtype=np.bool_),
        "crop_role": np.asarray(crop_role),
        "crop_structure": np.asarray(crop["structure"]),
        "crop_source_pixels": np.asarray(crop["source_pixels"]),
        "crop_retained_center_source_pixels": np.asarray(crop["retained_center_source_pixels"]),
        "crop_window_offset_source_pixels": np.asarray(crop["window_offset_source_pixels"]),
        "crop_inner_mask_source_pixels": np.asarray(crop["inner_mask_source_pixels"]),
        "crop_masked_center_fill": np.asarray(crop["masked_center_fill"]),
        "crop_stain_inclusion_mask": np.asarray(crop["stain_inclusion_mask"]),
        "crop_physical_width_um": np.asarray(crop["physical_width_um"]),
        "crop_signal_width_um": np.asarray(crop["signal_width_um"]),
        "crop_resize_pixels": np.asarray(crop["resize_pixels"]),
        "crop_protocol_sha256": np.asarray(builder._canonical_sha256(crop)),
        "ordered_input_gene_schema_sha256": np.asarray(
            builder._ordered_schema_sha256("hescape_input_genes", ("G0", "G1", "G2"))
        ),
        "ordered_target_gene_schema_sha256": np.asarray(
            builder._ordered_schema_sha256("hescape_target_genes", ("G0", "G1", "G2"))
        ),
        "ordered_frozen_feature_schema_sha256": np.asarray(
            builder._ordered_schema_sha256("uni2h_direct_features", builder.FROZEN_FEATURE_NAMES)
        ),
        "ordered_coordinate_schema_sha256": np.asarray(
            builder._ordered_schema_sha256(
                "hescape_coordinate_features", builder.COORDINATE_FEATURE_NAMES
            )
        ),
        "ordered_stain_schema_sha256": np.asarray(
            builder._ordered_schema_sha256("hescape_stain_features", builder.STAIN_FEATURE_NAMES)
        ),
        "ordered_composition_schema_sha256": np.asarray(
            builder._ordered_schema_sha256(
                "hescape_composition_features", builder.COMPOSITION_FEATURE_NAMES
            )
        ),
        "ordered_technical_schema_sha256": np.asarray(
            builder._ordered_schema_sha256(
                "hescape_technical_covariates", builder.TECHNICAL_COVARIATE_NAMES
            )
        ),
        "ordered_metadata_schema_sha256": np.asarray(
            builder._ordered_schema_sha256(
                "hescape_row_metadata",
                builder.ROW_METADATA_NAMES,
            )
        ),
        **builder._coverage_source_arrays(coverage, exclusions),
    }
    payload["source_schema_field_order_sha256"] = np.asarray(
        builder._ordered_schema_sha256(
            "hescape_source_fields",
            tuple(payload) + ("source_schema_field_order_sha256",),
        )
    )
    return payload


def test_rna_only_niche_fit_uses_development_statistics_and_rejects_ambiguity() -> None:
    genes = ("E", "I", "S", "V", "STATE")
    markers = {"epithelial": ("E",), "immune": ("I",), "stromal": ("S",), "endothelial": ("V",)}
    expression = np.asarray(
        [
            [8, 0, 0, 0, 1],
            [7, 1, 0, 0, 2],
            [0, 8, 0, 0, 3],
            [1, 7, 0, 0, 4],
            [0, 0, 8, 0, 5],
            [0, 0, 7, 1, 6],
            [0, 0, 0, 8, 7],
            [0, 0, 1, 7, 8],
            [9, 0, 0, 0, 9],
            [4, 4, 0, 0, 10],
        ],
        dtype=np.float64,
    )
    development = np.asarray([True] * 8 + [False, False])
    fit = builder._fit_dominant_niches(
        expression,
        genes,
        development,
        markers,
        minimum_score=0.5,
        minimum_margin=0.25,
    )
    assert fit.labels[-2] == 0
    assert fit.labels[-1] == -1
    assert fit.marker_gene_ids == ("E", "I", "S", "V")
    evaluation = tuple(gene for gene in genes if gene not in set(fit.marker_gene_ids))
    assert evaluation == ("STATE",)


def test_niche_fit_fails_closed_on_missing_or_unfittable_markers() -> None:
    values = np.arange(18, dtype=np.float64).reshape(6, 3)
    development = np.asarray([True, True, True, False, False, False])
    with pytest.raises(ValueError, match="must occur once"):
        builder._fit_dominant_niches(
            values,
            ("A", "B", "C"),
            development,
            {"one": ("A",), "two": ("MISSING",)},
            minimum_score=0.0,
            minimum_margin=0.1,
        )
    with pytest.raises(ValueError, match="no development-donor variation"):
        builder._fit_dominant_niches(
            np.ones((6, 2)),
            ("A", "B"),
            development,
            {"one": ("A",), "two": ("B",)},
            minimum_score=0.0,
            minimum_margin=0.1,
        )


def test_spatial_pools_are_deterministic_and_enforce_opposite_pool_guard() -> None:
    opposite = next(
        (x, y)
        for x in range(5)
        for y in range(5)
        if builder._pool_for_block("S", x, y) != builder._pool_for_block("S", x + 1, y)
    )
    x, y = opposite
    diagonal_overlap = np.asarray(
        [[x * 200 + 160, y * 200 + 10], [(x + 1) * 200 + 40, y * 200 + 90]]
    )
    assert np.linalg.norm(diagonal_overlap[0] - diagonal_overlap[1]) > 90
    assert np.max(np.abs(diagonal_overlap[0] - diagonal_overlap[1])) < 90
    grid = np.concatenate(
        (
            np.asarray([(x, y) for x in range(0, 1200, 100) for y in range(0, 1200, 100)]),
            diagonal_overlap,
        )
    )
    sections = np.asarray(["S"] * len(grid))
    donors = np.asarray(["D"] * len(grid))
    first = builder._spatial_pools(
        sections,
        donors,
        grid,
        block_size=200,
        roi_size=100,
        guard=90,
    )
    second = builder._spatial_pools(
        sections,
        donors,
        grid,
        block_size=200,
        roi_size=100,
        guard=90,
    )
    np.testing.assert_array_equal(first.roles, second.roles)
    np.testing.assert_array_equal(first.guard_pass, second.guard_pass)
    assert not first.guard_pass[-2:].any()
    retained = np.flatnonzero(first.guard_pass)
    for left in retained:
        for right in retained:
            if first.roles[left] != first.roles[right]:
                assert np.max(np.abs(grid[left] - grid[right])) >= 90
    for block in set(first.block_ids.tolist()):
        assert len(set(first.roles[first.block_ids == block].tolist())) == 1


def test_density_and_coordinate_controls_do_not_cross_sections() -> None:
    sections = np.asarray(["A", "A", "B", "B"])
    coordinates = np.asarray([[0, 0], [3, 4], [0, 0], [100, 100]], dtype=np.float64)
    density = builder._local_density(sections, coordinates, radius=5)
    np.testing.assert_array_equal(density, np.asarray([1, 1, 0, 0]))
    controls = builder._coordinate_features(sections, coordinates, density)
    assert controls.shape == (4, 7)
    assert np.isfinite(controls).all()
    np.testing.assert_allclose(controls[:2, :2], [[0, 0], [1, 1]])


def test_support_filter_drops_only_underpowered_donor_niche_strata() -> None:
    donors = np.asarray(["A"] * 9 + ["B"] * 4)
    labels = np.asarray([0] * 6 + [1] * 3 + [0] * 4)
    roles = np.asarray(
        ["reference"] * 3
        + ["evaluation"] * 3
        + ["reference"] * 2
        + ["evaluation"]
        + ["reference"] * 2
        + ["evaluation"] * 2
    )
    retained = builder._supported_strata_mask(
        donors,
        labels,
        roles,
        np.ones(len(donors), dtype=np.bool_),
        minimum_reference=2,
        minimum_evaluation=2,
    )
    np.testing.assert_array_equal(retained, labels == 0)


def test_protocol_pins_true_donors_and_never_splits_paired_sections() -> None:
    protocol_path = Path(__file__).parents[1] / "configs" / "hescape_lung_regional_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    builder._validate_protocol(protocol)
    assert protocol["authorization_ceiling"] == (
        "regional_pseudospot_only_no_cell_or_nucleus_claims"
    )
    mapping = protocol["section_to_true_donor"]
    assert len(set(mapping.values())) == 15
    assert tuple(protocol["development_donors"]) == builder.DEVELOPMENT_DONORS
    assert tuple(protocol["reserved_hest_locked_donors"]) == (builder.RESERVED_HEST_LOCKED_DONORS)
    assert "locked_test_donors" not in protocol
    roles = {donor: "development" for donor in builder.DEVELOPMENT_DONORS} | {
        donor: "excluded_previously_materialized" for donor in builder.RESERVED_HEST_LOCKED_DONORS
    }
    for left, right in (
        ("NCBI856", "NCBI857"),
        ("NCBI858", "NCBI859"),
        ("NCBI860", "NCBI861"),
        ("NCBI875", "NCBI876"),
        ("NCBI881", "NCBI882"),
    ):
        assert mapping[left] == mapping[right]
        assert roles[mapping[left]] == roles[mapping[right]]


def test_protocol_makes_55um_target_match_primary_and_prespecifies_context() -> None:
    protocol_path = Path(__file__).parents[1] / "configs" / "hescape_lung_regional_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    primary = builder._resolve_crop_protocol(protocol, "target_matched_55um_common_mpp")
    high_resolution = builder._resolve_crop_protocol(
        protocol, "target_matched_55um_high_resolution_sensitivity"
    )
    context = builder._resolve_crop_protocol(protocol, "context_108um")
    annulus = builder._resolve_crop_protocol(protocol, "context_annulus_55_to_109um")
    assert protocol["primary_crop_role"] == "target_matched_55um_common_mpp"
    assert primary["source_pixels"] == 512
    assert primary["retained_center_source_pixels"] == 256
    assert primary["signal_width_um"] == pytest.approx(256 * protocol["source_pixel_size_um"])
    assert primary["effective_model_mpp"] == context["effective_model_mpp"]
    assert high_resolution["effective_model_mpp"] != context["effective_model_mpp"]
    assert context["source_pixels"] == 512
    assert context["physical_width_um"] == pytest.approx(512 * protocol["source_pixel_size_um"])
    assert context["crop_scale"] == "context_108um_sensitivity"
    assert annulus["structure"] == "center_annulus"
    assert annulus["inner_mask_source_pixels"] == primary["retained_center_source_pixels"]
    assert annulus["masked_center_fill"] == "rounded_imagenet_mean_rgb"
    assert annulus["stain_inclusion_mask"] == "strict_outer_annulus_after_bilinear_resize"
    with pytest.raises(ValueError, match="crop role is not prespecified"):
        builder._resolve_crop_protocol(protocol, "post_hoc_crop")
    drifted = json.loads(json.dumps(protocol))
    drifted["crop_protocols"]["target_matched_55um_common_mpp"]["retained_center_source_pixels"] = (
        128
    )
    with pytest.raises(ValueError, match="crop_protocols differs"):
        builder._validate_protocol(drifted)


def test_protocol_rejects_nucleus_claim_revision_or_true_donor_drift() -> None:
    protocol_path = Path(__file__).parents[1] / "configs" / "hescape_lung_regional_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="prohibit nucleus claims"):
        builder._validate_protocol({**protocol, "authorizes_nucleus_claim": True})
    with pytest.raises(ValueError, match="dataset_revision differs"):
        builder._validate_protocol({**protocol, "dataset_revision": "unpinned"})
    wrong_mapping = {**protocol["section_to_true_donor"], "NCBI856": "wrong"}
    with pytest.raises(ValueError, match="section-to-GSE250346-donor"):
        builder._validate_protocol({**protocol, "section_to_true_donor": wrong_mapping})


def test_stain_statistics_are_fixed_finite_and_low_capacity() -> None:
    image = np.zeros((16, 16, 3), dtype=np.float64)
    image[..., 0] = 0.25
    image[..., 1] = 0.50
    image[..., 2] = 0.75
    statistics = builder._stain_statistics(image)
    assert statistics.shape == (len(builder.STAIN_FEATURE_NAMES),)
    np.testing.assert_allclose(statistics[:3], [0.25, 0.50, 0.75])
    np.testing.assert_allclose(statistics[3:6], 0.0)
    assert np.isfinite(statistics).all()
    assert statistics[-2] == 0.0
    assert statistics[-1] == 0.0


def test_annulus_stain_statistics_exclude_the_target_matched_center() -> None:
    image = np.full((16, 16, 3), 0.5, dtype=np.float64)
    image[4:12, 4:12] = np.asarray([1.0, 0.0, 0.0])
    annulus = np.ones((16, 16), dtype=np.bool_)
    annulus[4:12, 4:12] = False
    masked = builder._stain_statistics(image, annulus)
    gray_baseline = builder._stain_statistics(np.full_like(image, 0.5), annulus)
    unmasked = builder._stain_statistics(image)
    np.testing.assert_allclose(masked, gray_baseline)
    assert not np.allclose(masked, unmasked)


def test_preprocessing_uses_primary_center_and_context_only_annulus() -> None:
    image_module = pytest.importorskip("PIL.Image")
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    image[384:640, 384:640] = np.asarray([255, 0, 0], dtype=np.uint8)
    buffer = io.BytesIO()
    image_module.fromarray(image, mode="RGB").save(buffer, format="PNG")
    primary_tensor, primary_stain = builder._preprocess_image(
        buffer.getvalue(),
        crop_protocol=builder.CROP_PROTOCOLS["target_matched_55um_common_mpp"],
    )
    context_tensor, context_stain = builder._preprocess_image(
        buffer.getvalue(), crop_protocol=builder.CROP_PROTOCOLS["context_108um"]
    )
    annulus_tensor, annulus_stain = builder._preprocess_image(
        buffer.getvalue(),
        crop_protocol=builder.CROP_PROTOCOLS["context_annulus_55_to_109um"],
    )
    assert tuple(primary_tensor.shape) == (3, 224, 224)
    assert tuple(context_tensor.shape) == tuple(annulus_tensor.shape) == (3, 224, 224)
    np.testing.assert_allclose(primary_stain[:3], [1.0, 0.0, 0.0], atol=1e-6)
    assert not np.allclose(context_stain, annulus_stain)
    np.testing.assert_allclose(annulus_stain[:3], np.asarray([128 / 255] * 3), atol=1e-6)
    assert float(annulus_tensor[:, 64:160, 64:160].abs().max()) < 0.01


def test_coverage_tables_bind_section_metadata_and_partition_exclusions() -> None:
    sections = np.asarray(["A"] * 6 + ["B"] * 2)
    pools = builder.SpatialPools(
        block_ids=np.asarray(["block"] * 8),
        roi_ids=np.asarray(["roi"] * 8),
        roles=np.asarray(
            [
                "reference",
                "evaluation",
                "reference",
                "evaluation",
                "reference",
                "evaluation",
                "reference",
                "evaluation",
            ]
        ),
        guard_pass=np.asarray([True, True, True, True, False, True, True, True]),
    )
    coverage, exclusions = builder._coverage_tables(
        np.asarray(["DA"] * 6 + ["DB"] * 2),
        sections,
        np.asarray(["healthy"] * 6 + ["diseased"] * 2),
        np.asarray(["lung"] * 8),
        sections,
        np.asarray([0, 0, 0, -1, 1, 1, 1, 1]),
        np.asarray([True, True, False, True, True, True, True, True]),
        pools,
        np.asarray([True, True, False, False, False, False, True, True]),
        num_niches=2,
    )
    assert len(coverage) == 4
    a_niche_1 = next(row for row in coverage if row["section_id"] == "A" and row["type_index"] == 1)
    assert a_niche_1 == {
        "donor_id": "DA",
        "disease_state": "healthy",
        "site_id": "lung",
        "batch_id": "A",
        "section_id": "A",
        "type_index": 1,
        "labeled_before_guard": 2,
        "guard_excluded": 1,
        "unsupported_stratum_excluded": 1,
        "retained_reference": 0,
        "retained_evaluation": 0,
    }
    assert exclusions[0] == {
        "donor_id": "DA",
        "disease_state": "healthy",
        "site_id": "lung",
        "batch_id": "A",
        "section_id": "A",
        "release": 6,
        "zero_library_excluded": 1,
        "ambiguous_niche_excluded": 1,
        "guard_excluded": 1,
        "unsupported_stratum_excluded": 1,
        "retained": 2,
    }
    assert exclusions[0]["release"] == sum(
        value
        for name, value in exclusions[0].items()
        if name.endswith("excluded") or name == "retained"
    )


def test_ordered_schema_hash_is_order_sensitive_and_rejects_duplicates() -> None:
    forward = builder._ordered_schema_sha256("example", ("a", "b"))
    reverse = builder._ordered_schema_sha256("example", ("b", "a"))
    assert forward != reverse
    with pytest.raises(ValueError, match="must be unique"):
        builder._ordered_schema_sha256("example", ("a", "a"))


def test_source_schema_keeps_composition_stain_and_coordinate_controls_separate() -> None:
    payload = _minimal_source_payload()
    builder._validate_source_payload(payload)
    assert payload["reserved_donor_outcomes_loaded"].item() is False
    assert set(payload["donor_ids"]) <= set(builder.DEVELOPMENT_DONORS)
    assert tuple(payload["site_ids"]) == ("lung", "lung")
    assert tuple(payload["batch_ids"]) == ("NCBI858", "NCBI858")
    assert tuple(payload["gene_ids"]) == ("G0", "G1", "G2")
    assert tuple(payload["frozen_feature_names"]) == builder.FROZEN_FEATURE_NAMES
    assert tuple(payload["composition_feature_names"]) == (
        "composition_epithelial",
        "composition_immune",
        "composition_stromal",
        "composition_endothelial",
    )
    with pytest.raises(ValueError, match="feature names differ"):
        builder._validate_source_payload(
            {**payload, "composition_feature_names": np.asarray(["wrong"] * 4)}
        )
    with pytest.raises(ValueError, match="crop fields differ"):
        builder._validate_source_payload({**payload, "crop_source_pixels": np.asarray(256)})
    with pytest.raises(ValueError, match="ordered schema differs"):
        builder._validate_source_payload(
            {**payload, "ordered_stain_schema_sha256": np.asarray("a" * 64)}
        )
    changed_coverage = np.asarray(payload["coverage_labeled_before_guard"]).copy()
    changed_coverage[0] += 1
    with pytest.raises(ValueError, match="coverage counts do not partition"):
        builder._validate_source_payload(
            {**payload, "coverage_labeled_before_guard": changed_coverage}
        )
    with pytest.raises(ValueError, match="field order is not hash-bound"):
        builder._validate_source_payload(dict(reversed(tuple(payload.items()))))


def test_huggingface_content_address_avoids_rehash_but_plain_files_are_hashed(
    tmp_path: Path,
) -> None:
    import hashlib

    payload = b"pinned shard bytes"
    digest = hashlib.sha256(payload).hexdigest()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / digest
    blob.write_bytes(payload)
    snapshot = tmp_path / "snapshot.parquet"
    snapshot.symlink_to(blob)
    assert builder._verified_content_sha256(snapshot) == (
        digest,
        "huggingface_lfs_content_address",
    )
    plain = tmp_path / "plain.parquet"
    plain.write_bytes(payload)
    assert builder._verified_content_sha256(plain) == (digest, "byte_sha256")
