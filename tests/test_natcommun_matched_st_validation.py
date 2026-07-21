from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_validator():
    path = Path(__file__).parents[1] / "scripts/validate_natcommun_matched_st.py"
    spec = importlib.util.spec_from_file_location("validate_natcommun_matched_st", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()
runner = validator._load_runner()
core = runner._import_core()


def _public(*, same_section: bool = False) -> dict[str, np.ndarray]:
    rows, genes = 4, 3
    reference_section = "section" if same_section else "reference"
    query_section = "section" if same_section else "query"
    return {
        "schema": np.asarray(validator.PREPARED_SCHEMA),
        "direction_id": np.asarray("direction"),
        "donor": np.asarray("D"),
        "indication": np.asarray("tissue"),
        "design_family": np.asarray(
            "same_section_upper_bound" if same_section else "adjacent_section_primary"
        ),
        "evidence_label": np.asarray("diagnostic"),
        "guard_total_width_mm": np.asarray(1.0 if same_section else -1.0),
        "reference_section": np.asarray(reference_section),
        "query_section": np.asarray(query_section),
        "reference_spot_ids": np.asarray(["r1", "r2", "r3"]),
        "reference_st_counts": np.ones((3, genes), dtype=np.int32),
        "reference_st_library": np.full(3, genes, dtype=np.float32),
        "query_spot_ids": np.asarray([f"q{index}" for index in range(rows)]),
        "query_section_ids": np.repeat(query_section, rows),
        "query_indication_ids": np.repeat("tissue", rows),
        "query_image": np.ones((rows, 5), dtype=np.float32),
        "gene_ids": np.asarray([f"g{index}" for index in range(genes)]),
        "matched_sc_donor_ids": np.repeat("D", 4),
        **{
            f"baseline_rate_{arm}": np.ones((rows, genes), dtype=np.float32)
            for arm in validator.BASELINE_ARMS
        },
    }


def test_protocol_freezes_reciprocal_and_same_section_designs() -> None:
    protocol = validator._load_protocol(validator.DEFAULT_PROTOCOL)
    directions = validator._direction_map(protocol)

    assert len(directions) == 4
    assert set(value["donor"] for value in directions.values()) == {"B1", "L1"}
    assert protocol["same_section_upper_bound"]["empty_guard_total_width_mm"] == [
        1.0,
        2.0,
    ]
    assert protocol["evidence_boundary"]["measurement_floor_claim"] is False


def test_signature_proxy_matches_frozen_training_helper() -> None:
    counts = np.asarray([[5, 0, 1], [4, 1, 0], [0, 4, 2], [1, 5, 1]])
    labels = np.asarray(["A", "A", "B", "B"])
    query = np.asarray([[4, 1, 1], [1, 5, 1], [2, 2, 2]])
    signatures = validator.type_signatures(counts, labels, ("A", "B"))
    observed = validator.composition_proxy_from_signatures(query, signatures)
    expected, names = runner.training_only_composition_proxy(query, counts, labels)

    assert names == ("A", "B")
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1.0e-7)
    np.testing.assert_allclose(observed.sum(axis=1), 1.0)


def test_weighted_components_are_deterministic_and_noncollapsed() -> None:
    latent = np.arange(120, dtype=np.float64).reshape(20, 6) / 10
    identifiers = np.asarray([f"spot-{index:02d}" for index in range(20)])
    weights = np.linspace(0.1, 1.0, len(latent))
    first = validator.weighted_soft_components(
        latent,
        identifiers,
        weights,
        components=3,
        seed=17,
    )
    second = validator.weighted_soft_components(
        latent,
        identifiers,
        weights,
        components=3,
        seed=17,
    )

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert first[0].shape == (3, 6)
    assert np.all(first[1] > 0)
    np.testing.assert_allclose(first[2].sum(), 1.0)


def test_spatial_mixture_uses_soft_type_mass_and_rejects_query_overlap() -> None:
    latent = np.column_stack((np.linspace(-1, 1, 12), np.linspace(1, -1, 12)))
    composition = np.column_stack(
        (np.linspace(0.9, 0.1, 12), np.linspace(0.1, 0.9, 12))
    )
    identifiers = np.asarray([f"r{index}" for index in range(12)])
    mixture, weights, diagnostics = validator.build_spatial_reference_mixture(
        core,
        latent,
        composition,
        ("A", "B"),
        identifiers,
        "D",
        seed=17,
        minimum_mass=3,
        minimum_effective_sample_size=3,
    )

    assert mixture.source_modality == "spatial_st_adjacent_section"
    assert len(mixture.means) == 6
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert diagnostics["supported_type_count"] == 2
    mixture.assert_no_outcome_overlap(["query"])
    with pytest.raises(ValueError, match="spatial outcomes"):
        mixture.assert_no_outcome_overlap(["r0"])


def test_coordinate_pc1_tail_split_enforces_physical_guard() -> None:
    grid = np.column_stack((np.zeros(300), 2 * np.arange(300)))
    coordinates = validator.physical_array_coordinates_mm(grid)
    np.testing.assert_allclose(np.diff(coordinates[:2], axis=0), [[0.1, 0.0]])

    primary = validator.coordinate_pc1_tail_split(
        coordinates,
        guard_mm=1.0,
        minimum_spots=100,
    )
    sensitivity = validator.coordinate_pc1_tail_split(
        coordinates,
        guard_mm=2.0,
        minimum_spots=100,
    )
    for split, guard in ((primary, 1.0), (sensitivity, 2.0)):
        reference = split["reference_mask"]
        query = split["query_mask"]
        assert not np.any(reference & query)
        assert int(reference.sum()) >= 100
        assert int(query.sum()) >= 100
        assert split["minimum_observed_tail_separation_mm"] >= guard
    assert sensitivity["guard_mask"].sum() > primary["guard_mask"].sum()


def test_public_boundary_allows_disjoint_same_section_blocks_but_no_query_st() -> None:
    adjacent = _public()
    validator.validate_direction_public(adjacent)
    same = _public(same_section=True)
    validator.validate_direction_public(same)

    leaking = dict(adjacent)
    leaking["query_st_counts"] = np.ones((4, 3))
    with pytest.raises(ValueError, match="query ST leaked"):
        validator.validate_direction_public(leaking)

    overlapping = dict(same)
    overlapping["reference_spot_ids"] = np.asarray(["q0", "r2", "r3"])
    with pytest.raises(ValueError, match="disjoint"):
        validator.validate_direction_public(overlapping)


def test_reciprocal_directions_are_averaged_before_donor_inference() -> None:
    reports = {}
    for donor, shifts in (("B1", (1.0, 3.0)), ("L1", (2.0, 4.0))):
        for index, shift in enumerate(shifts):
            losses = {arm: 10.0 for arm in validator.ARMS}
            losses["S1"] = 10.0 + shift
            losses["S3"] = 10.0
            reports[f"{donor}-{index}"] = {
                "donor": donor,
                "design_family": "adjacent_section_primary",
                "guard_total_width_mm": -1.0,
                "mean_nb_deviance": losses,
            }
    aggregate = validator._aggregate_family(
        core,
        reports,
        design_family="adjacent_section_primary",
        guard_mm=None,
    )

    effect = aggregate["comparisons"]["S3_vs_S1_H_and_E_conditional_value"]
    assert aggregate["donor_count"] == 2
    assert aggregate["directions_per_donor"] == {"B1": 2, "L1": 2}
    assert effect["donor_effect"] == [2.0, 3.0]
    assert effect["exact_sign_flip"]["p_value"] == 0.25


def _score_payloads() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
]:
    public = _public()
    predictions = {
        key: np.asarray(public[key])
        for key in (
            "query_spot_ids",
            "query_section_ids",
            "query_indication_ids",
            "gene_ids",
        )
    }
    secret = {
        "schema": np.asarray(validator.PREPARED_SCHEMA),
        "direction_id": np.asarray("direction"),
        "donor": np.asarray("D"),
        "indication": np.asarray("tissue"),
        "query_section": np.asarray("query"),
        "design_family": np.asarray("adjacent_section_primary"),
        "evidence_label": np.asarray("diagnostic"),
        "guard_total_width_mm": np.asarray(-1.0),
        "query_spot_ids": np.asarray(public["query_spot_ids"]),
        "query_section_ids": np.asarray(public["query_section_ids"]),
        "query_indication_ids": np.asarray(public["query_indication_ids"]),
        "gene_ids": np.asarray(public["gene_ids"]),
        "query_st_counts": np.ones((4, 3), dtype=np.int32),
        "query_st_library": np.full(4, 5.0, dtype=np.float32),
        "primary_score_eligible": np.ones(4, dtype=bool),
    }
    direction: dict[str, object] = {
        "direction_id": "direction",
        "donor": "D",
        "indication": "tissue",
        "query_section": "query",
        "design_family": "adjacent_section_primary",
        "evidence_label": "diagnostic",
        "guard_total_width_mm": -1.0,
    }
    return secret, predictions, direction


def test_score_target_validation_fails_closed_on_identity_and_numeric_corruption() -> None:
    public = _public()
    secret, predictions, direction = _score_payloads()
    validator._validate_score_target(secret, public, predictions, direction)

    corruptions: list[tuple[str, np.ndarray]] = [
        ("schema", np.asarray("wrong")),
        ("donor", np.asarray("wrong")),
        ("indication", np.asarray("wrong")),
        ("query_section", np.asarray("wrong")),
        ("design_family", np.asarray("wrong")),
        ("evidence_label", np.asarray("wrong")),
        ("guard_total_width_mm", np.asarray(2.0)),
        ("gene_ids", np.asarray(["g1", "g0", "g2"])),
        ("query_spot_ids", np.asarray(["q1", "q0", "q2", "q3"])),
        ("query_section_ids", np.asarray(["wrong"] * 4)),
        ("query_indication_ids", np.asarray(["wrong"] * 4)),
        ("query_st_counts", np.ones((4, 2), dtype=np.int32)),
        ("query_st_counts", np.full((4, 3), -1, dtype=np.int32)),
        ("query_st_counts", np.full((4, 3), 0.5, dtype=np.float32)),
        ("query_st_library", np.full(4, -1.0, dtype=np.float32)),
        ("query_st_library", np.full(4, 2.0, dtype=np.float32)),
        ("primary_score_eligible", np.ones(4, dtype=np.int8)),
        ("primary_score_eligible", np.zeros(4, dtype=bool)),
    ]
    for key, value in corruptions:
        mutant = {name: np.asarray(array).copy() for name, array in secret.items()}
        mutant[key] = value
        with pytest.raises((ValueError, KeyError), match="score target"):
            validator._validate_score_target(mutant, public, predictions, direction)


def test_canonical_paths_and_recursive_target_free_boundary(tmp_path: Path) -> None:
    expected = (tmp_path / "artifact.npz").resolve()
    assert validator._require_canonical_path(str(expected), expected, "artifact") == expected
    with pytest.raises(ValueError, match="canonical"):
        validator._require_canonical_path("artifact.npz", expected, "artifact")
    with pytest.raises(ValueError, match="canonical"):
        validator._require_canonical_path(
            str(tmp_path / "nested" / ".." / "artifact.npz"),
            expected,
            "artifact",
        )
    assert validator._mapping_key_contains({"nested": {"score_target_path": "x"}}, "score_target")


def test_manifest_self_identity_detects_semantic_mutation() -> None:
    payload: dict[str, object] = {"schema": validator.PREPARED_SCHEMA, "directions": {}}
    payload["prepared_identity"] = validator._manifest_semantic_identity(payload)
    assert payload["prepared_identity"] == validator._manifest_semantic_identity(payload)
    payload["directions"] = {"unexpected": {}}
    assert payload["prepared_identity"] != validator._manifest_semantic_identity(payload)


def test_prediction_resume_receipt_rejects_stale_or_noncanonical_binding(
    tmp_path: Path,
) -> None:
    output = tmp_path.resolve()
    direction_id = "direction"
    directory = output / "directions" / direction_id
    directory.mkdir(parents=True)
    prediction_path = directory / "predictions.npz"
    prediction_path.write_bytes(b"placeholder")
    receipt_path = directory / "predict_receipt.json"
    direction = {
        "direction_id": direction_id,
        "donor": "D",
        "indication": "tissue",
        "reference_section": "reference",
        "query_section": "query",
        "design_family": "adjacent_section_primary",
        "evidence_label": "diagnostic",
        "guard_total_width_mm": -1.0,
        "predict_input_semantic_sha256": "input-sha",
    }
    model_receipt = {"model_identity": "model", "checkpoint_sha256": "checkpoint"}
    receipt: dict[str, object] = {
        "schema": validator.PREDICTION_SCHEMA,
        **{key: direction[key] for key in direction if key != "guard_total_width_mm"},
        "guard_total_width_mm": -1.0,
        "prediction_identity": "prediction",
        "prepared_identity": "prepared",
        "prediction_path": str(prediction_path),
        "predict_receipt_path": str(receipt_path),
        "prediction_semantic_sha256": "semantic",
        "model_identity": "model",
        "checkpoint_sha256": "checkpoint",
        "query_ST_opened": False,
        "reference_ST_opened": True,
        "directions_predicted_in_process": [direction_id],
        "process_isolation_rule_satisfied": True,
        "process_swap_kib_at_completion": 0,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validator._validate_prediction_receipt(
        receipt,
        output=output,
        direction=direction,
        expected_identity="prediction",
        prepared_identity="prepared",
        model_receipt=model_receipt,
    ) == prediction_path

    stale = dict(receipt)
    stale["prepared_identity"] = "stale"
    receipt_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction receipt"):
        validator._validate_prediction_receipt(
            stale,
            output=output,
            direction=direction,
            expected_identity="prediction",
            prepared_identity="prepared",
            model_receipt=model_receipt,
        )


def test_zero_process_swap_guard(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\ttest\nVmSwap:\t0 kB\n", encoding="utf-8")
    assert validator._assert_zero_process_swap(status) == 0
    status.write_text("Name:\ttest\nVmSwap:\t1 kB\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero process swap"):
        validator._assert_zero_process_swap(status)


def test_expected_upper_direction_design_is_exact() -> None:
    directions = validator._expected_upper_directions()
    assert len(directions) == 30
    assert len(set(directions)) == 30
    assert {value.rsplit("__", 1)[1] for value in directions} == {
        "guard_1mm",
        "guard_2mm",
    }
