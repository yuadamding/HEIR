"""Explicit frame transformations and source-derived author pivot/QC replay."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from heir.personalized.author_replay import (
    author_slide_seq_qc,
    density_supported_mask,
    registration_parameters,
    replay_slide_seq,
    source_identity,
)
from heir.personalized.transform_replay import (
    TransformStep,
    author_rigid_step,
    observation_join,
    replay_coordinates,
    replay_piecewise,
)


def test_declared_units_and_order_prevent_blind_registered_coordinate_rescaling():
    xy = np.array([[10.0, 20.0], [30.0, 50.0]])
    original = xy.copy()
    origin = TransformStep(
        "subtract_origin",
        np.array([[1, 0, -10], [0, 1, -20], [0, 0, 1]]),
        "native_pixel",
        "native_pixel",
    )
    scale = TransformStep("native_scale", np.diag([0.6, 0.6, 1]), "native_pixel", "analysis")
    rigid = author_rigid_step(
        "rotate_shift",
        rotation_origin=[4, 5],
        degrees=90,
        shift=[100, 200],
        input_units="analysis",
        output_units="registered_pixel",
    )
    result = replay_coordinates(xy, [origin, scale, rigid], input_units="native_pixel")
    centered = (xy - [10, 20]) * 0.6 - [4, 5]
    expected = centered @ np.array([[0, 1], [-1, 0]]) + [4, 5] + [100, 200]
    np.testing.assert_allclose(result, expected)
    changed = replace(scale, matrix=np.diag([0.65, 0.65, 1]))
    changed_inside_chain = replay_coordinates(
        xy, [origin, changed, rigid], input_units="native_pixel"
    )
    assert not np.allclose(changed_inside_chain, result * (0.65 / 0.6))
    np.testing.assert_array_equal(xy, original)
    with pytest.raises(ValueError, match="units"):
        replay_coordinates(xy, [scale, origin], input_units="native_pixel")


def test_author_reflection_uses_preflip_origin_and_rejects_undeclared_domain_extension():
    xy = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 1.0]])
    origin, shift = np.array([2.0, 0.5]), np.array([10.0, -3.0])
    step = author_rigid_step(
        "author",
        rotation_origin=origin,
        degrees=90,
        shift=shift,
        flip="v",
        reflection_maximum=4,
        input_units="analysis",
        output_units="registered_pixel",
    )
    flipped = xy.copy()
    flipped[:, 0] = abs(flipped[:, 0] - 4)
    expected = (flipped - origin) @ np.array([[0, 1], [-1, 0]]) + origin + shift
    np.testing.assert_allclose(replay_coordinates(xy, [step], input_units="analysis"), expected)
    with pytest.raises(ValueError, match="reflection domain"):
        replay_coordinates([[5, 0]], [step], input_units="analysis")
    with pytest.raises(ValueError, match="maximum"):
        author_rigid_step(
            "missing",
            rotation_origin=origin,
            degrees=90,
            shift=shift,
            flip="h",
            input_units="analysis",
            output_units="registered_pixel",
        )


def test_piecewise_origins_and_row_identity_remain_explicit():
    xy = np.array([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [30.0, 40.0]])
    chains = {
        "fragment-A": [
            author_rigid_step(
                "A",
                rotation_origin=[2, 3],
                degrees=90,
                shift=[1, 0],
                input_units="analysis",
                output_units="registered_pixel",
            )
        ],
        "fragment-B": [
            author_rigid_step(
                "B",
                rotation_origin=[20, 30],
                degrees=-90,
                shift=[0, 5],
                input_units="analysis",
                output_units="registered_pixel",
            )
        ],
    }
    pieces = ("fragment-A", "fragment-B", "fragment-A", "fragment-B")
    result = replay_piecewise(xy, pieces, chains, input_units="analysis")
    np.testing.assert_allclose(
        result[[0, 2]], replay_coordinates(xy[[0, 2]], chains["fragment-A"], input_units="analysis")
    )
    np.testing.assert_allclose(
        result[[1, 3]], replay_coordinates(xy[[1, 3]], chains["fragment-B"], input_units="analysis")
    )
    with pytest.raises(ValueError, match="declared"):
        replay_piecewise(xy, ("missing", *pieces[1:]), chains, input_units="analysis")
    inconsistent = {**chains, "fragment-B": [replace(chains["fragment-B"][0], output_units="um")]}
    with pytest.raises(ValueError, match="output unit"):
        replay_piecewise(xy, pieces, inconsistent, input_units="analysis")
    json.dumps(
        {key: [step.to_dict() for step in steps] for key, steps in chains.items()}, allow_nan=False
    )


def test_identifier_join_only_removes_declared_suffix_and_rejects_ambiguous_axes():
    assert observation_join(
        ("bead-1", "bead-2"), ("bead-2-2", "bead-1-2"), target_suffix="-2"
    ).tolist() == [1, 0]
    with pytest.raises(ValueError, match="duplicate"):
        observation_join(("bead", "bead"), ("bead-1",), target_suffix="-1")
    with pytest.raises(ValueError, match="suffix"):
        observation_join(("bead",), ("bead-2",), target_suffix="-1")
    with pytest.raises(ValueError, match="absent"):
        observation_join(("bead",), ("other-1",), target_suffix="-1")


def test_density_support_matches_dbscan_core_border_rule_including_radius_boundary():
    points = [[0, 0], [0, 0.1], [0.1, 0], [0.1, 0.1], [1.1, 0.1], [3, 3]]
    assert density_supported_mask(points, radius=1, min_samples=4).tolist() == [True] * 5 + [False]
    assert not density_supported_mask(points, radius=0.001, min_samples=2).any()


def test_author_qc_uses_all_native_replicates_and_preserves_strict_cutoff_loop():
    # Replicate1 alone has20low/20high rows: cutoff becomes40, whose equal-count rows remain.
    values = np.concatenate((np.ones((20, 40)), np.full((20, 40), 3)))
    low = {
        "counts": csr_matrix(values),
        "gene_ids": tuple(f"g{i}" for i in range(40)),
        "density_keep": np.ones(40, bool),
    }
    selected, initial, settings = author_slide_seq_qc({"1": low})
    assert settings["adaptive_count_cutoff"] == 40
    assert selected["1"].all() and initial["1"].all()
    # A native replicate later absent fromregistereddata still changes the original sharedQC.
    high = {**low, "counts": csr_matrix(np.full((40, 40), 3))}
    _, _, together = author_slide_seq_qc({"1": low, "2": high})
    assert together["adaptive_count_cutoff"] == 30
    assert together["native_replicate_ids_in_shared_qc"] == ["1", "2"]
    with pytest.raises(ValueError, match="high-count population"):
        author_slide_seq_qc({"only_low": {**low, "counts": csr_matrix(np.ones((40, 40)))}})


def _write_h5(path, ids, coordinates, *, counts=None, registered=False, original=None):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs["_index"] = np.array(ids, dtype=h5py.string_dtype())
        obs["x" if registered else "xcoord"] = coordinates[:, 0]
        obs["y" if registered else "ycoord"] = coordinates[:, 1]
        if registered:
            category = obs.create_group("__categories").create_dataset(
                "replicate", data=np.array(["1"], dtype=h5py.string_dtype())
            )
            obs["replicate"] = np.zeros(len(ids), dtype=int)
            obs["replicate"].attrs["categories"] = category.ref
            obs["x_orig"], obs["y_orig"] = original[:, 0], original[:, 1]
        if counts is not None:
            sparse = csr_matrix(counts)
            group = handle.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = sparse.shape
            for key in ("data", "indices", "indptr"):
                group[key] = getattr(sparse, key)
            handle.create_group("var")["_index"] = np.array(
                [f"g{i}" for i in range(sparse.shape[1])], dtype=h5py.string_dtype()
            )


def test_source_replay_recovers_prefilter_pivot_from_native_inputs_and_retains_excluded_replicate(
    tmp_path,
):
    root = tmp_path / "data"
    pipeline = tmp_path / "pipeline"
    (pipeline / "00.1_processing").mkdir(parents=True)
    (pipeline / "spatial_utils.py").write_text("# synthetic source binding; no execution\n")
    code = "\n".join(
        f'registration.loc["slide_seq {r}","{field}"]={value}'
        for r in (1, 2)
        for field, value in (("degrees", 37), ("shift_x", 100), ("shift_y", 200))
    )
    notebook = pipeline / "00.1_processing/HTAPP-313-SMP-932_processing.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": code}]}))
    xy = np.array(
        [[1000 + x * 10, 2000 + y * 10] for y in range(5) for x in range(5)] + [[0, 0]], dtype=float
    )
    values = np.full((26, 40), 3)
    values[0] = 0
    ids = tuple(f"bead{i}" for i in range(26))
    sources = []
    for label in ("1", "2"):
        path = root / f"working/unregistered/synthetic/HTAPP-313-SMP-932_{label}_slide_seq.h5ad"
        _write_h5(path, ids, xy, counts=values)
        sources.append(
            {
                "working_path": str(path.relative_to(root)),
                "modality": "slide_seq",
                "replicate_id": label,
                **{k: v for k, v in source_identity(path).items() if k != "path"},
            }
        )
    original = xy / (5000 / 3000) - [600, 1200]
    origin = original[1:25].mean(0)  # Before later retained-subset filtering.
    selected = np.arange(2, 17, 2)
    angle = np.deg2rad(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    registered = (original[selected] - origin) @ rotation.T + origin + [100, 200]
    path = root / "working/registered/synthetic/slide_seq.h5ad"
    _write_h5(
        path,
        [ids[i] + "-1" for i in selected],
        registered,
        registered=True,
        original=original[selected],
    )
    source = {
        "working_path": str(path.relative_to(root)),
        **{k: v for k, v in source_identity(path).items() if k != "path"},
    }
    entry = {
        "patient_id": "P",
        "specimen_id": "S",
        "sources": sources,
        "role_sources": {"targets": source},
    }
    reports, chains, arrays = replay_slide_seq(root, "313-932", entry, pipeline_root=pipeline)
    assert len(reports) == len(chains) == 2
    assert reports[0]["max_coordinate_error_pixels"] < 1e-10
    assert reports[0]["intermediate_residuals"]["max_euclidean"] == 0
    assert reports[1]["status"] == "native_replicate_not_retained_in_registered_bundle"
    np.testing.assert_allclose(chains[0]["rotation_origin"], origin)
    assert not np.allclose(origin, original[selected].mean(0))
    assert chains[0]["qc"]["native_replicate_ids_in_shared_qc"] == ["1", "2"]
    assert not chains[0]["physical_calibration_verified"]
    np.testing.assert_array_equal(arrays["313-932__slide_seq__1__joined_native_rows"], selected)
    json.dumps(reports, allow_nan=False)
    json.dumps(chains, allow_nan=False)


def test_notebook_parser_does_not_execute_unrelated_code_or_dynamic_parameters(tmp_path):
    marker = tmp_path / "must_not_exist"
    code = (
        f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
        'registration.loc["codex 1","shift_x"]=(3, -4, 12/2)\n'
        'registration.loc["HE 1 annot","scale"]=external_value'
    )
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": code}]}))
    parameters, evidence = registration_parameters(path, "codex")
    assert parameters == {"codex 1": {"shift_x": (3, -4, 6)}}
    assert evidence["source_cell_indices_zero_based"] == [0]
    assert not marker.exists()
