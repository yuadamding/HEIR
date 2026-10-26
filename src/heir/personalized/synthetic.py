"""Tiny synthetic files for engineering checks; never HTAPP qualification evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def create_fixture(directory: str | Path) -> list[Path]:
    import anndata as ad
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.sparse import csr_matrix

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    rng = np.random.default_rng(4)
    manifests = []
    for p in range(3):
        patient = f"SYNTHETIC_P{p}"
        folder = root / patient
        folder.mkdir()
        matching = {
            "patient_id": patient,
            "sampling_episode_id": "episode",
            "lesion_id": "lesion",
            "represented_region_id": "region",
            "core_id": "core",
            "section_id": "he_section",
            "specimen_id": patient + "_specimen",
        }
        protein_identity = {
            **matching,
            "section_id": "protein_serial_section",
            "specimen_id": patient + "_protein_section",
        }
        cell_ids = [f"{patient}_nucleus_{i}" for i in range(12)]
        coordinates = np.column_stack((np.repeat([20, 120, 220, 320], 3), np.tile([20, 40, 60], 4)))
        pq.write_table(
            pa.table(
                {
                    "cell_id": cell_ids,
                    "x_um": coordinates[:, 0],
                    "y_um": coordinates[:, 1],
                    "segmentation_quality": np.ones(12),
                    "tissue_component_id": ["tissue"] * 12,
                }
            ),
            folder / "cells.parquet",
        )
        features = rng.normal(size=(12, 4)).astype(np.float32)
        features[:, 0] += np.repeat(np.arange(4), 3)
        np.savez(
            folder / "image_features.npz", cell_ids=np.array(cell_ids), image_features=features
        )
        counts = np.vstack([rng.poisson([12, 2, 3], (6, 3)), rng.poisson([2, 12, 4], (6, 3))])
        rna = ad.AnnData(
            X=np.zeros((12, 2), dtype=np.float32),
            obs=pd.DataFrame(
                {
                    "broad_type": ["type_A"] * 6 + ["type_B"] * 6,
                    "patient_id": [patient] * 12,
                    "sampling_episode_id": ["episode"] * 12,
                    "lesion_id": ["lesion"] * 12,
                    "represented_region_id": ["region"] * 12,
                },
                index=[f"rna_{i}" for i in range(12)],
            ),
            var=pd.DataFrame(index=["processed_1", "processed_2"]),
        )
        rna.obsm["counts"] = csr_matrix(counts)
        rna.uns["counts_var"] = pd.DataFrame(index=genes)
        rna.write_h5ad(folder / "rna.h5ad")
        proteins = ad.AnnData(
            X=np.column_stack((np.linspace(1, 9, 12), np.linspace(9, 1, 12))).astype(np.float32),
            obs=pd.DataFrame(
                {
                    "tissue_component_id": ["tissue"] * 12,
                    **{field: [value] * 12 for field, value in protein_identity.items()},
                },
                index=[f"protein_{i}" for i in range(12)],
            ),
            var=pd.DataFrame(index=["MARKER_A", "MARKER_B"]),
        )
        proteins.obsm["spatial"] = coordinates.astype(float)
        proteins.layers["observed"] = np.ones((12, 2), dtype=bool)
        proteins.write_h5ad(folder / "proteins.h5ad")
        registration = {
            "source_frame": "protein",
            "target_frame": "he",
            "source_units": "um",
            "target_units": "um",
            "direction": "source_to_target",
            "affine": np.eye(3).tolist(),
            "correspondence": "serial_section",
            "uncertainty_um": 5.0,
            "source_identity": protein_identity,
            "target_identity": matching,
        }
        (folder / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
        observations = [f"bin:{i}:0" for i in range(4)]
        targets = ad.AnnData(
            X=np.array([[12, 2, 4], [9, 5, 3], [5, 9, 5], [2, 12, 4]], dtype=np.float32),
            obs=pd.DataFrame(
                {"patient_id": [patient] * 4, "specimen_id": [patient + "_specimen"] * 4},
                index=observations,
            ),
            var=pd.DataFrame(index=genes),
        )
        targets.obsm["spatial"] = np.array([[50, 50], [150, 50], [250, 50], [350, 50]])
        targets.layers["observed"] = np.ones((4, 3), dtype=bool)
        targets.write_h5ad(folder / "spatial_targets.h5ad")
        np.savez(
            folder / "observation.npz",
            cell_ids=np.array(cell_ids),
            observation_ids=np.array(observations),
            rows=np.repeat(np.arange(4), 3),
            columns=np.arange(12),
            weights=np.ones(12),
        )
        manifest = {
            "schema_version": 2,
            "synthetic": True,
            "patient_id": patient,
            "specimen_id": patient + "_specimen",
            "matching": {
                "query": {**matching, "assay": "H&E"},
                "reference": {**matching, "assay": "snRNA-seq", "section_id": "rna_aliquot"},
                "protein": {**protein_identity, "assay": "multiplex_IF"},
                "reference_independent_of_spatial_targets": True,
                "evidence": "synthetic integration fixture; no biological matching assertion",
            },
            "paths": {
                "cells": "cells.parquet",
                "image_features": "image_features.npz",
                "rna": "rna.h5ad",
                "proteins": "proteins.h5ad",
                "registration": "registration.json",
                "spatial_targets": "spatial_targets.h5ad",
                "observation": "observation.npz",
            },
            "feature_provenance": {
                "encoder_name": "synthetic_random_features",
                "checkpoint_sha256": hashlib.sha256(b"synthetic-no-checkpoint").hexdigest(),
                "frozen": True,
                "contains_spatial_rna_derived_features": False,
                "crop_scales_um": [32, 112],
            },
            "reference": {
                "counts_source": "obsm",
                "broad_type_column": "broad_type",
                "annotation_provenance": "matched_dissociated_rna",
                "embedding_gene_ids": genes,
                "max_states": 4,
                "library_size": 10000,
            },
            "proteins": {
                "bin_size_um": 100,
                "origin_um": [0, 0],
                "marker_table": [
                    {
                        "marker_id": "MARKER_A",
                        "gene_id": "GENE_A",
                        "interpretation": "identity",
                        "intensity_half_saturation": 5,
                        "rna_half_saturation": 3000,
                    },
                    {
                        "marker_id": "MARKER_B",
                        "gene_id": "GENE_B",
                        "interpretation": "total_receptor",
                        "intensity_half_saturation": 5,
                        "rna_half_saturation": 3000,
                    },
                ],
            },
            "graph": {"radius_um": 120, "max_neighbors": 8},
            "targets": {"coordinate_units": "um", "expression_scale": "nonnegative_linear"},
        }
        path = folder / "specimen.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(path)
    model = {
        "variant": "he_rna_protein",
        "feature_dim": 4,
        "state_dim": 3,
        "state_gene_ids": genes,
        "gene_ids": genes,
        "supervised_gene_ids": genes[:2],
        "normalization_gene_ids": genes,
        "hidden_dim": 8,
        "residual_rank": 2,
    }
    (root / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (root / "genes.json").write_text(json.dumps(genes) + "\n")
    with (root / "folds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outer_fold", "patient_id", "role"])
        writer.writeheader()
        writer.writerows(
            {"outer_fold": 0, "patient_id": f"SYNTHETIC_P{i}", "role": role}
            for i, role in enumerate(("train", "validation", "test"))
        )
    (root / "README.txt").write_text(
        "Synthetic engineering fixture. No HTAPP data, encoder checkpoint, scientific "
        "qualification, or biological conclusion. Validation patients are reserved; "
        "this example performs no tuning or model selection.\n"
    )
    return manifests
