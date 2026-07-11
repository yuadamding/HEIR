from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from heir.data import gene_panel
from heir.data.gene_panel import (
    candidate_exclusion_reason,
    log1p_cpm,
    read_h5ad_gene_names_only,
    resolve_development_record,
    select_gene_panel,
)
from heir.data.manifest import ManifestRecord


def test_log1p_cpm_uses_full_library_denominator() -> None:
    counts = sparse.csr_matrix([[1.0, 1.0, 8.0], [0.0, 5.0, 5.0]])
    normalized = log1p_cpm(counts, scale=100.0).toarray()
    np.testing.assert_allclose(normalized[0], np.log1p([10.0, 10.0, 80.0]))
    np.testing.assert_allclose(normalized[1], np.log1p([0.0, 50.0, 50.0]))


def test_select_gene_panel_retains_curated_balances_markers_and_excludes_noise() -> None:
    genes = (
        "CUR1",
        "CUR2",
        "TYPEA1",
        "TYPEA2",
        "TYPEA3",
        "TYPEB1",
        "TYPEB2",
        "TYPEB3",
        "HVG1",
        "HVG2",
        "HVG3",
        "HVG4",
        "HVG5",
        "HVG6",
        "MS4A2",
        "MT-ND1",
        "RPL3",
        "LOC12345",
    )
    labels = np.asarray(["A"] * 5 + ["B"] * 5)
    counts = np.ones((10, len(genes)), dtype=np.float64)
    counts[:5, 2:5] = np.asarray([20.0, 16.0, 12.0])
    counts[5:, 2:5] = 0.0
    counts[:5, 5:8] = 0.0
    counts[5:, 5:8] = np.asarray([20.0, 16.0, 12.0])
    for row in range(10):
        counts[row, 8:14] = np.asarray(
            [1 + row, 1 + row % 3, 1 + (row * 2) % 5, 2 + row % 4, 1 + row % 2, 3]
        )
    counts[:, 14] = np.asarray([1, 40, 1, 40, 1, 40, 1, 40, 1, 40])

    result = select_gene_panel(
        sparse.csr_matrix(counts),
        genes=genes,
        labels=labels,
        curated_genes=("CUR1", "CUR2"),
        available_genes=tuple(gene for gene in genes if gene != "MS4A2"),
        ranking_genes=genes,
        panel_size=10,
        markers_per_type=2,
        minimum_detection=0.0,
        minimum_type_detection=0.2,
    )

    assert len(result.genes) == 10
    assert [item.gene for item in result.genes[:2]] == ["CUR1", "CUR2"]
    assert result.type_counts == {"A": 2, "B": 2}
    assert sum(item.selection_category == "hvg" for item in result.genes) == 4
    assert not {"MS4A2", "MT-ND1", "RPL3", "LOC12345"}.intersection(
        item.gene for item in result.genes
    )
    assert result.candidate_counts["noise_excluded"] == 3


def test_select_gene_panel_requires_curated_genes_in_every_inventory() -> None:
    genes = ("CUR1", "CUR2", "A1", "A2", "B1", "B2", "H1", "H2")
    counts = sparse.csr_matrix(
        [
            [1, 1, 20, 10, 0, 0, 1, 4],
            [1, 1, 15, 8, 0, 0, 4, 1],
            [1, 1, 0, 0, 20, 10, 1, 4],
            [1, 1, 0, 0, 15, 8, 4, 1],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="curated genes are not available.*CUR2"):
        select_gene_panel(
            counts,
            genes=genes,
            labels=("A", "A", "B", "B"),
            curated_genes=("CUR1", "CUR2"),
            available_genes=tuple(gene for gene in genes if gene != "CUR2"),
            panel_size=6,
            markers_per_type=1,
            minimum_detection=0.0,
            minimum_type_detection=0.0,
        )


def test_builder_requires_three_unique_visium_metadata_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly three unique"):
        gene_panel.build_snpatho_panel(
            manifest_path=tmp_path / "manifest.tsv",
            curated_path=tmp_path / "curated.tsv",
            availability_h5ads=(tmp_path / "reference.h5ad",),
            evaluation_h5ads=(tmp_path / "4066.h5ad", tmp_path / "4399.h5ad"),
            output_path=tmp_path / "panel.tsv",
            provenance_path=tmp_path / "panel.json",
        )


def test_availability_h5ad_reader_never_requests_expression(monkeypatch, tmp_path: Path) -> None:
    import h5py

    source = tmp_path / "availability.h5ad"
    with h5py.File(source, "w") as handle:
        variables = handle.create_group("var")
        variables.create_dataset("feature_name", data=np.asarray([b"A", b"B"]))
        # If the implementation regresses to AnnData it would load this
        # malformed expression object while parsing the file.
        handle.create_dataset("X", data=np.asarray([b"forbidden-expression"]))

    def reject_anndata():  # pragma: no cover - failure guard
        raise AssertionError("availability H5AD was opened through AnnData")

    monkeypatch.setattr(gene_panel, "_require_anndata", reject_anndata)
    genes = read_h5ad_gene_names_only(source, "feature_name")
    assert genes == ("A", "B")


def test_development_record_rejects_nondevelopment_role(monkeypatch, tmp_path: Path) -> None:
    record = ManifestRecord(
        cohort_id="mosaic_natcommun_2025",
        donor_id="B1",
        specimen_id="B1",
        block_id="B1",
        section_id="B1_4",
        modality="histology+snrna",
        assay_platform="test",
        preservation="FFPE",
        tissue="breast",
        matching_tier="tier_1",
        analysis_role="locked_validation",
        count_matrix_file=str(tmp_path / "development.h5ad"),
        included=True,
    )
    monkeypatch.setattr(gene_panel, "load_manifest", lambda *args, **kwargs: (record,))
    with pytest.raises(ValueError, match="included development"):
        resolve_development_record(tmp_path / "manifest.tsv")


@pytest.mark.parametrize(
    ("gene", "reason"),
    [
        ("MT-ND1", "mitochondrial"),
        ("RPS18", "ribosomal"),
        ("RPL23AP1", "ribosomal"),
        ("LOC12345", "obvious_pseudogene"),
        ("AC123456.1", "uncharacterized_locus"),
        ("EPCAM", None),
    ],
)
def test_candidate_exclusion_reason(gene: str, reason: str) -> None:
    assert candidate_exclusion_reason(gene) == reason
