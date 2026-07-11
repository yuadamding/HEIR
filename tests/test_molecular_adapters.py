"""Tests for portable frozen-molecular-teacher artifacts."""

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from heir.cli import _log_normalize
from heir.data import RNAReference, normalize_panel_counts
from heir.prior import SCGPTTeacherArtifact, SCVIAdapter


def test_scvi_panel_is_reordered_without_selected_panel_renormalization() -> None:
    class FakeModel:
        def get_normalized_expression(self, *args, **kwargs):
            assert kwargs["transform_batch"] == ["reference-batch"]
            assert kwargs["library_size"] == 10_000.0
            return pd.DataFrame([[1.0, 3.0]], columns=["B", "A"])

    adapter = SCVIAdapter(latent_dim=2)
    adapter.model = FakeModel()
    observed = adapter.normalized_expression(
        gene_list=["A", "B"],
        transform_batch=["reference-batch"],
    )
    expected = np.log1p(np.asarray([[3.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(observed, expected)


def test_scvi_accepts_zero_mass_in_selected_panel() -> None:
    class FakeModel:
        def get_normalized_expression(self, *args, **kwargs):
            return pd.DataFrame([[0.0, 0.0]], columns=["A", "B"])

    adapter = SCVIAdapter(latent_dim=2)
    adapter.model = FakeModel()
    observed = adapter.normalized_expression(
        gene_list=["A", "B"],
        transform_batch=["reference-batch"],
    )

    # A cell can have positive full-library mass entirely outside the panel.
    np.testing.assert_array_equal(observed, np.zeros((1, 2), dtype=np.float32))


def test_scvi_matches_reference_and_spatial_full_library_normalization() -> None:
    full_counts = np.asarray(
        [
            [3.0, 7.0, 90.0],
            [10.0, 10.0, 20.0],
        ],
        dtype=np.float32,
    )
    full_library_sizes = full_counts.sum(axis=1)
    panel_counts = sparse.csr_matrix(full_counts[:, [1, 0]])
    panel_genes = ["B", "A"]
    scvi_linear = full_counts * (10_000.0 / full_library_sizes[:, None])

    class FakeModel:
        def get_normalized_expression(self, *args, **kwargs):
            assert kwargs["gene_list"] == panel_genes
            assert kwargs["library_size"] == 10_000.0
            assert kwargs["transform_batch"] == ["reference-batch"]
            return pd.DataFrame(scvi_linear[:, [1, 0]], columns=panel_genes)

    adapter = SCVIAdapter(latent_dim=2)
    adapter.model = FakeModel()
    scvi_expression = adapter.normalized_expression(
        gene_list=panel_genes,
        transform_batch=["reference-batch"],
    )

    reference = RNAReference(
        sample_id="synthetic",
        cell_ids=np.asarray(["cell-1", "cell-2"]),
        gene_ids=np.asarray(panel_genes),
        counts=panel_counts,
        library_sizes=full_library_sizes,
    )
    reference_expression = _log_normalize(
        reference.counts,
        library_sizes=reference.library_sizes,
    ).toarray()
    spatial_expression = normalize_panel_counts(panel_counts, full_library_sizes)

    np.testing.assert_allclose(scvi_expression, reference_expression, rtol=1.0e-6)
    np.testing.assert_allclose(scvi_expression, spatial_expression, rtol=1.0e-6)
    # The selected panel retains its true 10%/50% share of each full library.
    np.testing.assert_allclose(
        np.expm1(scvi_expression).sum(axis=1),
        [1_000.0, 5_000.0],
        rtol=1.0e-6,
    )


def test_scgpt_artifact_exports_type_moments(tmp_path) -> None:
    artifact = SCGPTTeacherArtifact(
        cell_ids=np.asarray(["a", "b", "c", "d"]),
        embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]),
        type_names=np.asarray(["epithelial", "immune"]),
        type_indices=np.asarray([0, 0, 1, 1]),
        gene_vocabulary=np.asarray(["G1", "G2"]),
        checkpoint_id="whole-human+lora-test",
        training_donors=np.asarray(["d1", "d2"]),
    )
    path = tmp_path / "teacher.npz"
    artifact.to_npz(path)
    restored = SCGPTTeacherArtifact.from_npz(path)
    assert restored.type_prototypes().shape == (2, 2)
    assert restored.type_variances().shape == (2, 2)
    assert np.all(restored.type_variances() >= 0)


class _FakeSCVI(SCVIAdapter):
    def latent(self, adata=None):
        return np.asarray(
            [[-1.0, 0.0], [-0.5, 0.1], [0.0, 0.2], [0.5, 0.3], [1.0, 0.4], [1.5, 0.5]],
            dtype=np.float32,
        )

    def normalized_expression(self, adata=None, gene_list=None, transform_batch=None):
        assert transform_batch == ["reference"] or transform_batch == ("reference",)
        latent = self.latent(adata)
        return np.stack((latent[:, 0] + 2.0, latent[:, 1] + 1.0), axis=1)


def test_scvi_decoder_distillation_exports_heir_compatible_decoder(tmp_path) -> None:
    adapter = _FakeSCVI(latent_dim=2)
    adapter.model = object()
    distilled = adapter.distill_transferable_decoder(
        object(),
        ["G1", "G2"],
        validation_mask=np.asarray([False, False, False, False, True, True]),
        decoder_hidden_dims=(4,),
        max_epochs=2,
        patience=2,
        batch_size=2,
        device="cpu",
        transform_batch=["reference"],
    )
    assert distilled.config.input_dim == 2
    assert distilled.config.latent_dim == 2
    assert all(not parameter.requires_grad for parameter in distilled.decoder.parameters())
    with torch.no_grad():
        assert distilled.decoder(torch.zeros(1, 2)).shape == (1, 2)
    output = tmp_path / "scvi_decoder.pt"
    adapter.export_transferable_decoder_checkpoint(
        str(output),
        object(),
        ["G1", "G2"],
        np.asarray([False, False, False, False, True, True]),
        training_donors=["d1", "d2"],
        latent_space_id="scvi:test",
        transform_batch=["reference"],
        decoder_hidden_dims=(4,),
        max_epochs=1,
        patience=1,
        batch_size=2,
        device="cpu",
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["decoder_only"]
    assert checkpoint["metadata"]["schema"] == "heir.scvi_distilled_decoder.v2"
    assert checkpoint["metadata"]["latent_space_id"] == "scvi:test"
    assert checkpoint["metadata"]["expression_space_id"] == "log1p-cpm-10000-v1"
    assert (
        checkpoint["metadata"]["expression_normalization_contract"]
        == "full_library_10000_then_panel_log1p_v2"
    )
    assert checkpoint["metadata"]["transform_batch"] == ["reference"]
    assert checkpoint["metadata"]["gene_names"] == ["G1", "G2"]
    assert checkpoint["metadata"]["expression_normalization"] == {
        "method": "scvi.get_normalized_expression",
        "library_size": 10_000.0,
        "library_basis": "full-transcriptome",
        "gene_selection": "after-library-normalization",
        "transform": "log1p",
        "version": 2,
    }
