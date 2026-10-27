"""Measurement eligibility stays separate from prediction and spatial qualification."""

import json
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from heir.personalized.regional_endpoint import (
    RegionalCounts,
    _array_prefixes,
    aggregate_retained_counts,
    freeze_regional_endpoint,
    qualify_measurements,
)
from heir.personalized.regions import build_region_grid

POOL = ("A", "B", "C")
QC = dict(
    min_observations=5,
    min_pool_counts=100,
    min_regions=20,
    min_gene_counts=50,
    min_positive_regions=5,
    min_positive_fraction=0.1,
    min_log_quantile_range=0.25,
)


def grid():
    image = np.full((500, 500, 3), [180, 60, 100], dtype=np.uint8)
    return build_region_grid(image, compute_features=False)


def measurements():
    counts = np.column_stack([np.arange(25) + 1, np.full(25, 100), np.zeros(25)])
    return [
        RegionalCounts(
            label,
            counts.astype(float),
            np.full(25, 5),
            np.arange(125),
            np.repeat(np.arange(25), 5),
        )
        for label in ("merfish:1", "merfish:2", "exseq:1")
    ]


@pytest.mark.parametrize("sparse", [False, True])
def test_sum_retained_counts_preserves_replicates_and_observed_zero(sparse):
    from scipy.sparse import csr_matrix

    current = grid()
    xy = np.array([[50, 50], [50, 50], [150, 50], [-1, 50], [50, 50]])
    values = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [99, 99, 99], [7, 8, 9]])
    result = aggregate_retained_counts(
        csr_matrix(values) if sparse else values,
        POOL,
        xy,
        ["1", "1", "1", "1", "2"],
        current,
        ("B", "A"),
    )
    assert [r.replicate for r in result] == ["1", "2"]
    np.testing.assert_array_equal(result[0].counts[0], [7, 5])
    np.testing.assert_array_equal(result[1].counts[0], [8, 7])
    assert result[0].observation_counts[1] == 1  # measured all-zero is present
    assert result[1].observation_counts[1] == 0  # absent is not measured zero
    assert result[0].region_assignments.tolist() == [0, 0, 1, -1]
    assert result[0].counts.sum() == 12


@pytest.mark.parametrize("bad", [-1, 0.5, np.nan, np.inf])
def test_rejects_noncount_measurements(bad):
    values = np.array([[bad, 1, 2]])
    with pytest.raises(ValueError, match="integer counts"):
        aggregate_retained_counts(values, POOL, [[1, 1]], ["1"], grid(), POOL)


def test_duplicate_genes_missing_pool_and_nonfinite_coordinates_fail():
    with pytest.raises(ValueError, match="unique"):
        aggregate_retained_counts(np.ones((1, 3)), ["A", "A", "C"], [[1, 1]], ["1"], grid(), POOL)
    with pytest.raises(ValueError, match="absent"):
        aggregate_retained_counts(np.ones((1, 2)), ["A", "B"], [[1, 1]], ["1"], grid(), POOL)
    with pytest.raises(ValueError, match="finite"):
        aggregate_retained_counts(np.ones((1, 3)), POOL, [[np.nan, 1]], ["1"], grid(), POOL)


def test_common_geometry_then_shared_qc_uses_every_retained_replicate():
    reps = measurements()
    altered = reps[1].observation_counts.copy()
    altered[:2] = [0, 4]
    reps[1] = replace(reps[1], observation_counts=altered)
    counts = reps[2].counts.copy()
    counts[2] = [1, 10, 0]
    reps[2] = replace(reps[2], counts=counts)
    proteins = np.ones(25, bool)
    proteins[3] = False
    result = qualify_measurements(reps, proteins, POOL, ("A", "C"), ("B",), QC)
    assert result["geometric_support"].sum() == 23
    assert result["eligible_regions"].sum() == 21
    assert not result["eligible_regions"][:4].any()
    assert result["measurement_qualified"]
    assert result["panels"]["primary"] == ["A"]
    assert all(
        r["failure_reasons"]
        == "insufficient_gene_counts;insufficient_detection;insufficient_dynamic_range"
        for r in result["gene_qc"]
        if r["gene"] == "C"
    )


def test_pool_never_shrinks_after_gene_qc_and_same_denominator_for_secondary():
    reps = measurements()
    for i, rep in enumerate(reps):
        counts = rep.counts.copy()
        counts[:, 2] = 50  # abundant but insufficiently variable; still in exposure
        reps[i] = replace(rep, counts=counts)
    result = qualify_measurements(reps, np.ones(25, bool), POOL, ("A", "C"), ("B",), QC)
    expected = np.log1p(10000 * reps[0].counts / reps[0].counts.sum(1, keepdims=True))
    np.testing.assert_allclose(result["normalized"]["merfish:1"], expected)
    assert result["panels"]["primary"] == ["A"]
    assert result["normalized"]["merfish:1"].shape == (25, 3)
    assert expected[0, 0] != pytest.approx(np.log1p(10000))
    assert not np.allclose(
        expected[:, 0],
        np.log1p(10000 * reps[0].counts[:, 0] / reps[0].counts[:, :2].sum(1)),
    )


def test_a_gene_failing_one_replicate_is_excluded_from_common_panel():
    reps = measurements()
    counts = reps[1].counts.copy()
    counts[:, 0] = 0
    reps[1] = replace(reps[1], counts=counts)
    result = qualify_measurements(reps, np.ones(25, bool), POOL, ("A", "C"), ("B",), QC)
    assert result["panels"]["primary"] == []
    assert not result["measurement_qualified"]
    passing = [r for r in result["gene_qc"] if r["gene"] == "A" and r["eligible"]]
    assert len(passing) == 2


def test_empty_or_too_small_common_support_fails_without_relaxing_rules():
    for n in (0, 19):
        present = np.arange(25) < n
        result = qualify_measurements(measurements(), present, POOL, ("A", "C"), ("B",), QC)
        assert not result["measurement_qualified"]
        assert result["panels"]["primary"] == []
        assert all("insufficient_regions" in r["failure_reasons"] for r in result["gene_qc"])
        assert result["normalized"]["merfish:1"].shape == (n, 3)


def endpoint_fixture(tmp_path, monkeypatch):
    from PIL import Image

    import heir.personalized.regional_endpoint as module

    root = tmp_path / "cohort"
    root.mkdir()
    metadata = root / "metadata"
    metadata.mkdir()
    for name in ("selected_13_biopsies.tsv", "selected_13_verified_file_manifest.tsv"):
        (metadata / name).write_text(name)
    Image.fromarray(np.full((500, 500, 3), [180, 60, 100], dtype=np.uint8)).save(root / "he.png")
    roles = {}
    for role in ("he", "protein", "merfish", "exseq"):
        path = root / ("he.png" if role == "he" else f"registered/{role}.h5ad")
        if role != "he":
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(role.encode())
        roles[role] = dict(
            biopsy_id="case",
            working_path=str(path.relative_to(root)),
            bytes=path.stat().st_size,
            sha256=sha256(path.read_bytes()).hexdigest(),
            modality="codex" if role == "protein" else role,
            replicate_id="1;2" if role == "merfish" else "1",
            measurement_key="obsm/counts",
            feature_key="uns/counts_var",
        )
    entry = dict(
        patient_id="patient",
        specimen_id="specimen",
        role_sources=roles,
        sources=list(roles.values()),
    )
    monkeypatch.setattr(module, "discover_cohort", lambda _: {"case": entry})
    import pandas as pd

    coords = np.repeat(grid().coordinates, 5, axis=0)
    counts = np.repeat(
        np.column_stack([np.arange(25) + 1, np.full(25, 100), np.zeros(25)]), 5, axis=0
    )

    @contextmanager
    def read(path):
        assay = path.stem
        copies = 2 if assay == "merfish" else 1
        n = len(coords) * copies
        yield SimpleNamespace(
            X=np.full((n, 1), -999),  # processed X cannot be the outcome
            n_obs=n,
            obs=pd.DataFrame(
                dict(
                    x=np.tile(coords[:, 0], copies),
                    y=np.tile(coords[:, 1], copies),
                    replicate=np.repeat([str(i + 1) for i in range(copies)], len(coords)),
                )
            ),
            obs_names=[f"obs{i}" for i in range(n)],
            uns={"counts_var": POOL},
            obsm={"counts": np.tile(counts, (copies, 1))},
        )

    monkeypatch.setattr(module, "_read_h5ad", read)
    config = dict(
        schema="heir.regional_endpoint.v1",
        biopsy_ids=["case"],
        normalization_gene_ids=list(POOL),
        primary_gene_ids=["A", "C"],
        secondary_gene_ids=["B"],
        region_size_pixels=100,
        min_tissue_fraction=0.1,
        qc=QC,
        spatial_status="conditional_author_registered_pixel",
        expected_sources={r["working_path"]: r["sha256"] for r in roles.values()},
    )
    config_path = tmp_path / "config.json"
    config["expected_sources"].update(
        {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in metadata.iterdir()}
    )
    sidecar_path = tmp_path / "spatial.json"
    sidecar_path.write_text('{"spatial_status":"conditional"}')
    config["spatial_evidence"] = dict(
        path=sidecar_path.name, sha256=sha256(sidecar_path.read_bytes()).hexdigest()
    )
    config_path.write_text(json.dumps(config))
    out = tmp_path / "endpoint"
    return root, config_path, out, module


def test_runner_binds_sources_and_keeps_measurement_and_spatial_qualification_separate(
    tmp_path, monkeypatch
):
    root, config_path, out, module = endpoint_fixture(tmp_path, monkeypatch)
    result = freeze_regional_endpoint(root, config_path, out)
    assert result["measurement_qualified"]
    assert not result["biological_spatial_qualification"]
    assert not result["prediction_payloads_read"]
    assert result["biopsies"][0]["replicates"] == ["merfish:1", "merfish:2", "exseq:1"]
    arrays = np.load(out / "case.npz", allow_pickle=False)
    assert arrays["merfish__1__counts"][0].tolist() == [5, 500, 0]
    assert set(json.loads((out / "artifact_sha256.json").read_text())) == {
        "protocol.json",
        "summary.json",
        "region_qc.tsv",
        "gene_qc.tsv",
        "case.npz",
    }
    with pytest.raises(FileExistsError):
        freeze_regional_endpoint(root, config_path, out)
    (root / "registered/exseq.h5ad").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        freeze_regional_endpoint(root, config_path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("missing_binding", [False, True])
def test_metadata_binding_checked_before_discovery(tmp_path, monkeypatch, missing_binding):
    root, config_path, out, module = endpoint_fixture(tmp_path, monkeypatch)

    def forbidden(_):
        raise AssertionError("discovery must wait for verified metadata")

    monkeypatch.setattr(module, "discover_cohort", forbidden)
    key = "metadata/selected_13_biopsies.tsv"
    if missing_binding:
        config = json.loads(config_path.read_text())
        del config["expected_sources"][key]
        config_path.write_text(json.dumps(config))
    else:
        (root / key).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identit"):
        freeze_regional_endpoint(root, config_path, out)
    assert not out.exists()


@pytest.mark.parametrize("which", ["payload", "metadata", "config", "sidecar"])
def test_mutation_during_extraction_cannot_publish_summary(tmp_path, monkeypatch, which):
    root, config_path, out, module = endpoint_fixture(tmp_path, monkeypatch)
    target = {
        "payload": root / "registered/exseq.h5ad",
        "metadata": root / "metadata/selected_13_biopsies.tsv",
        "config": config_path,
        "sidecar": tmp_path / "spatial.json",
    }[which]
    original = module._read_h5ad

    @contextmanager
    def mutating(path):
        with original(path) as adata:
            yield adata
        if path.stem == "exseq":
            target.write_bytes(b"changed after extraction")

    monkeypatch.setattr(module, "_read_h5ad", mutating)
    with pytest.raises(ValueError, match="differs|changed"):
        freeze_regional_endpoint(root, config_path, out)
    assert (out / "protocol.json").is_file()
    assert not (out / "summary.json").exists()
    assert not (out / "artifact_sha256.json").exists()


def test_output_cannot_modify_input_root_even_through_symlink(tmp_path, monkeypatch):
    root, config_path, _, _ = endpoint_fixture(tmp_path, monkeypatch)
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)
    for out in (root, root / "output", linked / "output"):
        with pytest.raises(ValueError, match="outside the input"):
            freeze_regional_endpoint(root, config_path, out)
    assert not (root / "output").exists()


def test_serialized_replicate_name_collisions_fail_instead_of_overwriting():
    reps = measurements()[:2]
    reps = [replace(rep, replicate=label) for rep, label in zip(reps, ["1:2", "1__2"])]
    with pytest.raises(ValueError, match="collide"):
        _array_prefixes(reps)
