"""Unit tests for NPZ array contracts and sparse/backed H5AD selection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heir.data.arrays import HistologyBag, PrototypeSet, RNAReference  # noqa: E402
from heir.data.h5ad import prepare_h5ad, selection_to_rna_reference  # noqa: E402


class ArrayContractTests(unittest.TestCase):
    def test_histology_bag_round_trip_and_immutability(self):
        bag = HistologyBag(
            slide_id="slide-1",
            nucleus_ids=np.asarray(["n1", "n2"]),
            features=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float16),
            coordinates_um=np.asarray([[10.0, 20.0], [30.0, 40.0]]),
            morphology=np.asarray([[0.1], [0.2]]),
            segmentation_confidence=np.asarray([0.9, 0.8]),
            artifact_probability=np.asarray([0.0, 0.1]),
            edge_index=np.asarray([[0, 1], [1, 0]]),
            edge_weight=np.asarray([0.25, 0.75], dtype=np.float32),
            sample_id="sample-1",
            donor_id="donor-1",
            block_id="block-1",
            feature_space_id="encoder-v1",
            histology_source_sha256="a" * 64,
            nuclei_source_sha256="b" * 64,
            feature_source_sha256="c" * 64,
        )
        self.assertFalse(bag.features.flags.writeable)
        self.assertEqual(bag.n_nuclei, 2)
        with self.assertRaises(ValueError):
            bag.features[0, 0] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bag.npz"
            bag.save_npz(path)
            loaded = HistologyBag.load_npz(path)
        np.testing.assert_array_equal(loaded.nucleus_ids, bag.nucleus_ids)
        np.testing.assert_allclose(loaded.features, bag.features)
        np.testing.assert_array_equal(loaded.edge_index, bag.edge_index)
        np.testing.assert_allclose(loaded.edge_weight, bag.edge_weight)
        self.assertEqual(loaded.donor_id, "donor-1")
        self.assertEqual(loaded.feature_space_id, "encoder-v1")
        self.assertEqual(loaded.histology_source_sha256, "a" * 64)

    def test_histology_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "coordinates_um"):
            HistologyBag(
                slide_id="slide",
                nucleus_ids=np.asarray(["n1", "n2"]),
                features=np.ones((2, 3)),
                coordinates_um=np.ones((2, 3)),
            )

    def test_sparse_rna_reference_round_trip(self):
        counts = sparse.csr_matrix(np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
        reference = RNAReference(
            sample_id="sample-1",
            cell_ids=np.asarray(["c1", "c2"]),
            gene_ids=np.asarray(["g1", "g2", "g3"]),
            counts=counts,
            library_sizes=np.asarray([10.0, 12.0]),
            cell_type_labels=np.asarray(["T", "B"]),
            donor_ids=np.asarray(["d1", "d1"]),
            latent_space_id="sha256:test-latent",
            latent_training_donors=("atlas-1", "atlas-2"),
            latent_transform_sha256="a" * 64,
        )
        self.assertTrue(sparse.isspmatrix_csr(reference.counts))
        self.assertFalse(reference.counts.data.flags.writeable)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rna.npz"
            reference.save_npz(path)
            loaded = RNAReference.load_npz(path)
        np.testing.assert_array_equal(loaded.counts.toarray(), counts.toarray())
        np.testing.assert_allclose(loaded.library_sizes, np.asarray([10.0, 12.0]))
        np.testing.assert_array_equal(loaded.cell_type_labels, np.asarray(["T", "B"]))
        self.assertEqual(loaded.latent_space_id, "sha256:test-latent")
        self.assertEqual(loaded.latent_training_donors, ("atlas-1", "atlas-2"))
        self.assertEqual(loaded.latent_transform_sha256, "a" * 64)

    def test_rna_reference_rejects_negative_counts(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RNAReference(
                sample_id="sample",
                cell_ids=np.asarray(["c1"]),
                gene_ids=np.asarray(["g1"]),
                counts=sparse.csr_matrix([[-1.0]]),
            )

    def test_rna_reference_defaults_are_not_truncated(self):
        reference = RNAReference(
            sample_id="sample-long",
            cell_ids=np.asarray(["c1"]),
            gene_ids=np.asarray(["g1"]),
            counts=sparse.csr_matrix([[1.0]]),
        )
        self.assertEqual(reference.cell_type_labels.tolist(), ["unknown"])
        self.assertEqual(reference.donor_ids.tolist(), ["sample-long"])
        self.assertEqual(reference.sample_ids.tolist(), ["sample-long"])

    def test_prototype_round_trip_and_weight_normalization(self):
        prototypes = PrototypeSet(
            prototype_ids=np.asarray(["p1", "p2", "p3"]),
            sample_ids=np.asarray(["s1", "s1", "s2"]),
            cell_type_labels=np.asarray(["T", "B", "T"]),
            means=np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]),
            variances=np.ones((3, 2)),
            weights=np.asarray([0.4, 0.6, 1.0]),
            n_cells=np.asarray([80, 120, 75]),
            latent_space_id="sha256:test-latent",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prototypes.npz"
            prototypes.save_npz(path)
            loaded = PrototypeSet.load_npz(path)
        np.testing.assert_allclose(loaded.means, prototypes.means)
        np.testing.assert_allclose(loaded.weights, prototypes.weights)
        self.assertEqual(loaded.latent_space_id, "sha256:test-latent")
        with self.assertRaisesRegex(ValueError, "sum to one"):
            PrototypeSet(
                prototype_ids=np.asarray(["p1", "p2"]),
                sample_ids=np.asarray(["s1", "s1"]),
                cell_type_labels=np.asarray(["T", "B"]),
                means=np.ones((2, 2)),
                variances=np.ones((2, 2)),
                weights=np.asarray([0.2, 0.2]),
            )


class H5ADTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import anndata  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("anndata/pandas optional test dependencies are unavailable")

    def test_backed_filtering_preserves_sparse_matrix(self):
        import anndata
        import pandas as pd

        matrix = sparse.csr_matrix(
            np.asarray(
                [
                    [1, 0, 0, 2, 0],
                    [0, 3, 0, 0, 0],
                    [4, 0, 5, 0, 0],
                    [0, 0, 0, 0, 6],
                ],
                dtype=np.float32,
            )
        )
        obs = pd.DataFrame(
            {
                "donor_id": ["d1", "d2", "d1", "d1"],
                "sample_id": ["s1", "s1", "s1", "s2"],
                "cell_type": ["T", "B", "T", "myeloid"],
            },
            index=pd.Index(["c1", "c2", "c3", "c4"], dtype=object),
            dtype=object,
        )
        var = pd.DataFrame(
            {"feature_name": ["g1", "g2", "g3", "g4", "g5"]},
            index=pd.Index(["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"], dtype=object),
            dtype=object,
        )
        adata = anndata.AnnData(X=matrix, obs=obs, var=var)
        adata.layers["SoupX"] = matrix.astype(np.float64) * 0.9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.h5ad"
            adata.write_h5ad(path)
            selection = prepare_h5ad(
                path,
                donor_filter="d1",
                sample_filter="s1",
                genes=["g4", "g2"],
                gene_key="feature_name",
                layer="SoupX",
                sample_id="d1-s1",
            )
            self.assertEqual(selection.shape, (2, 2))
            np.testing.assert_array_equal(selection.gene_ids, np.asarray(["g4", "g2"]))
            reference = selection_to_rna_reference(selection, chunk_size=1)
        self.assertTrue(sparse.isspmatrix_csr(reference.counts))
        np.testing.assert_allclose(
            reference.counts.toarray(),
            np.asarray([[1.8, 0.0], [0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(reference.cell_ids, np.asarray(["c1", "c3"]))
        np.testing.assert_allclose(reference.library_sizes, np.asarray([2.7, 8.1]))

    def test_empty_h5ad_filter_is_rejected(self):
        import anndata
        import pandas as pd

        adata = anndata.AnnData(
            X=sparse.csr_matrix([[1.0]]),
            obs=pd.DataFrame(
                {"donor_id": ["d1"], "sample_id": ["s1"], "cell_type": ["T"]},
                index=pd.Index(["c1"], dtype=object),
                dtype=object,
            ),
            var=pd.DataFrame(index=pd.Index(["g1"], dtype=object)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.h5ad"
            adata.write_h5ad(path)
            with self.assertRaisesRegex(ValueError, "selected no cells"):
                prepare_h5ad(path, donor_filter="missing")


if __name__ == "__main__":
    unittest.main()
