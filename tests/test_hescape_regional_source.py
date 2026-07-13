from __future__ import annotations

import importlib.util
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
    mapping = protocol["section_to_true_donor"]
    assert len(set(mapping.values())) == 15
    assert tuple(protocol["development_donors"]) == builder.DEVELOPMENT_DONORS
    assert tuple(protocol["locked_test_donors"]) == builder.LOCKED_DONORS
    roles = {donor: "development" for donor in builder.DEVELOPMENT_DONORS} | {
        donor: "locked_test" for donor in builder.LOCKED_DONORS
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


def test_source_schema_keeps_composition_stain_and_coordinate_controls_separate() -> None:
    rows = 2
    payload = {
        "schema_version": np.asarray(builder.SOURCE_SCHEMA),
        "observation_ids": np.asarray(["A", "B"]),
        "donor_ids": np.asarray(["D", "D"]),
        "section_ids": np.asarray(["S", "S"]),
        "disease_states": np.asarray(["healthy", "healthy"]),
        "block_ids": np.asarray(["B1", "B2"]),
        "roi_ids": np.asarray(["R1", "R2"]),
        "pool_roles": np.asarray(["reference", "evaluation"]),
        "type_labels": np.asarray([0, 1]),
        "frozen_features": np.vstack(
            (np.zeros(builder.FEATURE_WIDTH), np.ones(builder.FEATURE_WIDTH))
        ),
        "stain_features": np.vstack(
            (np.zeros(len(builder.STAIN_FEATURE_NAMES)), np.ones(len(builder.STAIN_FEATURE_NAMES)))
        ),
        "stain_feature_names": np.asarray(builder.STAIN_FEATURE_NAMES),
        "composition_features": np.ones((rows, len(builder.COMPOSITION_FEATURE_NAMES))),
        "composition_feature_names": np.asarray(builder.COMPOSITION_FEATURE_NAMES),
        "molecular_targets": np.ones((rows, 3)),
        "coordinate_features": np.ones((rows, 7)),
        "technical_covariates": np.ones((rows, 1)),
        "registration_is_one_to_one": np.ones(rows, dtype=np.bool_),
    }
    builder._validate_source_payload(payload)
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
