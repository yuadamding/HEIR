"""Published HTAPP biopsy adapter for explicitly regional, pixel-frame experiments.

This adapter uses the publication's biopsy grouping without manufacturing section,
lesion, or same-cell identities. Spatial RNA is read only by the target loader.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from .graph import build_graph
from .observation import ObservationOperator
from .reference import build_reference, extract_raw_counts
from .regions import RegionGrid, build_region_grid, point_to_grid
from .schema import PredictionInputs, ProteinEvidence, ProteinObservationOperator, SpatialTargets

COHORT_BIOPSIES = (
    "213-6752",
    "313-932",
    "330-1082",
    "364-1321",
    "514-6760",
    "783-4081",
    "812-8239",
    "853-4381",
    "878-7149",
    "880-7179",
    "917-4531",
    "982-7629",
    "997-7789",
)
DEFAULT_EMBEDDING_GENES = (
    "EPCAM",
    "KRT8",
    "KRT18",
    "KRT19",
    "VIM",
    "COL1A1",
    "COL1A2",
    "DCN",
    "PECAM1",
    "VWF",
    "PTPRC",
    "CD3D",
    "CD4",
    "CD8A",
    "NKG7",
    "LYZ",
    "CD68",
    "MKI67",
)
# Explicit one-gene pairs checked against the retained original Table S5.
# CD25/ISG20, isoform-specific CD45RA, and multigene antibodies are excluded.
DEFAULT_PROTEIN_MARKERS = (
    ("CD4", "CD4"),
    ("CD8", "CD8A"),
    ("CD45", "PTPRC"),
    ("CD31", "PECAM1"),
    ("Ki67", "MKI67"),
)


def _table(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _source_path(root, source, *, check_size=True):
    root = Path(root).resolve()
    path = (root / source["working_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("cohort source path escapes the dataset root")
    if check_size and (not path.is_file() or path.stat().st_size != int(source["bytes"])):
        raise ValueError(f"cohort source missing or size differs: {source['working_path']}")
    return path


def _verified_source_path(root, source, *, verification_cache=None):
    path = _source_path(root, source)
    before = path.stat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    expected = (source["sha256"], identity)
    if verification_cache is not None and str(path) in verification_cache:
        if verification_cache[str(path)] != expected:
            raise ValueError("consumed source changed after verification in this run")
        return path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("consumed source changed during checksum verification")
    if digest.hexdigest() != source["sha256"]:
        raise ValueError(f"cohort source checksum differs: {source['working_path']}")
    if verification_cache is not None:
        verification_cache[str(path)] = expected
    return path


def discover_cohort(root):
    """Check all selected manifest paths/sizes and preserve published source hashes.

    This is structural discovery, not a fresh payload checksum verification.
    Missing or size-mismatched payloads are reported, not rejected globally:
    prediction never requires target availability. Each loader verifies the
    complete SHA-256 of only the payloads it consumes.
    """
    root = Path(root)
    metadata = _table(root / "metadata/selected_13_biopsies.tsv")
    if len(metadata) != 13 or {r["biopsy_id"] for r in metadata} != set(COHORT_BIOPSIES):
        raise ValueError("selected cohort metadata must contain the exact 13 unique biopsies")
    result = {
        row["biopsy_id"]: {
            "patient_id": row["participant_group_id"],
            "specimen_id": row["htan_biopsy_id"],
            "metadata": row,
            "sources": [],
            "role_sources": {},
            "file_checks": {},
        }
        for row in metadata
    }
    paths = set()
    for source in _table(root / "metadata/selected_13_verified_file_manifest.tsv"):
        biopsy = source["biopsy_id"]
        if biopsy not in result:
            raise ValueError(f"unexpected biopsy in selected source manifest: {biopsy}")
        entry = result[biopsy]
        if (
            source["participant_id"] != entry["patient_id"]
            or source["htan_biopsy_id"] != entry["specimen_id"]
            or source["selected_cohort"].lower() != "true"
            or source["included_in_working_subset"].lower() != "true"
        ):
            raise ValueError(f"cohort identity or selection mismatch for {biopsy}")
        if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
            raise ValueError("source manifest requires SHA-256 identifiers")
        path = _source_path(root, source, check_size=False)
        if path in paths:
            raise ValueError("duplicate working path in cohort source manifest")
        paths.add(path)
        entry["sources"].append(source)
        entry["file_checks"][source["working_path"]] = (
            "missing"
            if not path.is_file()
            else "size_matches"
            if path.stat().st_size == int(source["bytes"])
            else "size_differs"
        )
        if "registered" not in Path(source["working_path"]).parts:
            continue
        role = {"scRNAseq": "rna", "codex": "protein", "slide_seq": "targets"}.get(
            source["modality"]
        )
        if source["modality"] == "HE" and source["replicate_id"] == "1":
            role = "he"
        if role is not None:
            if role in entry["role_sources"]:
                raise ValueError(f"multiple registered sources for {biopsy}/{role}")
            if role != "he" and (
                source["measurement_key"] != "obsm/counts"
                or source["feature_key"] != "uns/counts_var"
            ):
                raise ValueError("registered measurements require the retained count axes")
            entry["role_sources"][role] = source
    for biopsy, entry in result.items():
        if set(entry["role_sources"]) != {"he", "rna", "protein", "targets"}:
            raise ValueError(f"missing registered H&E/RNA/CODEX/Slide-seq source for {biopsy}")
        entry["assay_replicates"] = {
            role: source["replicate_id"].split(";")
            for role, source in entry["role_sources"].items()
        }
    return result


@contextmanager
def _read_h5ad(path):
    import anndata

    # Backed X avoids loading the unused dense, processed 5,000-HVG matrix.
    adata = anndata.read_h5ad(path, backed="r")
    try:
        yield adata
    finally:
        adata.file.close()


def _axis(values, name):
    result = tuple(str(v) for v in values)
    if (
        not result
        or len(set(result)) != len(result)
        or any(not v.strip() or v.lower() in {"nan", "none"} for v in result)
    ):
        raise ValueError(f"{name} requires unique, nonempty identifiers")
    return result


def _axis_provenance(ids):
    return {
        "key": "obs/_index",
        "count": len(ids),
        "ordered_ids_sha256": hashlib.sha256(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "digest_encoding": "UTF-8_JSON_array_compact",
    }


def _counts_by_label(values):
    labels, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def _replicates(adata, source):
    values = np.asarray(adata.obs["replicate"].astype(str))
    if set(values) != set(source["replicate_id"].split(";")):
        raise ValueError("actual assay replicate labels differ from the verified manifest")
    return values


def _dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _groups(grid, xy, replicates):
    mapped = point_to_grid(grid, xy)
    keys = sorted({(str(rep), int(index)) for rep, index in zip(replicates, mapped) if index >= 0})
    if not keys:
        raise ValueError("assay has no observations inside the selected H&E tissue regions")
    lookup = {key: i for i, key in enumerate(keys)}
    rows = np.asarray(
        [lookup.get((str(rep), int(index)), -1) for rep, index in zip(replicates, mapped)],
        dtype=np.int64,
    )
    return keys, rows


def _protein_evidence(root, source, reference, grid, protein_cofactor, rna_cofactor):
    panel = _table(Path(root) / "metadata/Table_S5_CODEX_panel_original.tsv")
    for marker, gene in DEFAULT_PROTEIN_MARKERS:
        matching = [r["Matching gene"] for r in panel if r["Target"] == marker]
        if matching != [gene]:
            raise ValueError(f"original antibody panel does not confirm {marker}/{gene}")
    markers, genes = zip(*DEFAULT_PROTEIN_MARKERS)
    with _read_h5ad(_verified_source_path(root, source)) as adata:
        axis = _axis(adata.uns["counts_var"], "CODEX measurement features")
        counts = adata.obsm["counts"]
        if counts.shape != (adata.n_obs, len(axis)):
            raise ValueError("CODEX retained intensity axes differ")
        if not set(genes) <= set(axis) or not set(genes) <= set(reference.gene_ids):
            raise ValueError("fixed protein panel genes must exist in CODEX and independent RNA")
        values = _dense(counts[:, [axis.index(g) for g in genes]]).astype(np.float64)
        observed = np.isfinite(values) & (values >= 0)
        transformed = np.where(observed, values, 0)
        transformed = transformed / (transformed + protein_cofactor)
        original_ids = _axis(adata.obs_names, "CODEX observation IDs")
        replicates = _replicates(adata, source)
        keys, rows = _groups(grid, adata.obs[["x", "y"]].to_numpy(), replicates)
    means = np.zeros((len(keys), len(genes)), dtype=np.float32)
    complete = np.zeros(means.shape, dtype=bool)
    for row in range(len(keys)):
        selected = rows == row
        complete[row] = observed[selected].all(0)
        means[row] = np.where(complete[row], transformed[selected].mean(0), 0)
    state_values = reference.expression[:, [reference.gene_ids.index(g) for g in genes]]
    state_values = state_values / (state_values + rna_cofactor)
    ids = tuple(f"codex:{rep}:{grid.region_ids[index]}" for rep, index in keys)
    operator = ProteinObservationOperator(
        cell_ids=grid.region_ids,
        observation_ids=ids,
        rows=np.arange(len(keys)),
        columns=np.asarray([index for _, index in keys]),
        weights=np.ones(len(keys)),
    )
    coverage = {
        "covered_region_replicates": len(keys),
        "covered_he_regions": len(set(operator.columns.tolist())),
        "excluded_source_observations": int((rows < 0).sum()),
        "unobserved_region_markers": int((~complete).sum()),
    }
    evidence = ProteinEvidence(
        marker_ids=markers,
        values=means,
        observed=complete,
        state_values=state_values,
        correspondence="region",
        operator=operator,
        quality=complete.astype(np.float32),
        coverage=coverage,
    )
    return evidence, {
        "source_observation_axis": _axis_provenance(original_ids),
        "source_replicates": _counts_by_label(replicates),
        "marker_gene_pairs": list(map(list, DEFAULT_PROTEIN_MARKERS)),
        "transform": "x/(x+cofactor), transform_each_cell_then_uniform_region_mean",
        "protein_cofactor": protein_cofactor,
        "rna_cofactor": rna_cofactor,
        "quality_policy": "complete_finite_coverage_indicator_not_registration_confidence",
        "protein_rna_calibration": "unvalidated_fixed_saturating_proxy",
        "coverage": coverage,
    }


def prepare_cohort_inputs(
    root,
    biopsy_id,
    *,
    region_size=100,
    min_tissue_fraction=0.1,
    embedding_gene_ids=None,
    gene_ids=None,
    max_states=32,
    protein_cofactor=100.0,
    rna_cofactor=10.0,
    graph_radius=150.0,
    graph_max_neighbors=8,
    feature_cache_dir=None,
    feature_encoder_manifest_sha256=None,
):
    """Prepare H&E regions, independent RNA, and CODEX without opening spatial RNA.

    ``gene_ids`` optionally limits stored state expression after full-library RNA
    normalization; embedding and protein genes are additionally retained.
    """
    if any(not np.isfinite(v) or v <= 0 for v in (protein_cofactor, rna_cofactor)):
        raise ValueError("protein and RNA saturation cofactors must be finite and positive")
    entry = discover_cohort(root)[biopsy_id]
    sources = entry["role_sources"]
    grid = build_region_grid(
        _verified_source_path(root, sources["he"]),
        bin_size_pixels=region_size,
        min_tissue_fraction=min_tissue_fraction,
    )
    feature_provenance = {
        "kind": "rgb_control",
        "encoder_name": grid.settings["encoder_name"],
        "encoder_manifest_sha256": grid.settings["encoder_manifest_sha256"],
        "feature_dim": int(grid.image_features.shape[1]),
        "contains_spatial_rna_derived_features": False,
    }
    if (feature_cache_dir is None) != (feature_encoder_manifest_sha256 is None):
        raise ValueError("frozen features require both cache directory and encoder manifest hash")
    if feature_cache_dir is not None:
        from .region_features import load_feature_cache

        cache = Path(feature_cache_dir) / f"{biopsy_id}.npz"
        features, metadata = load_feature_cache(
            cache,
            grid,
            expected_image_sha256=sources["he"]["sha256"],
            expected_encoder_manifest_sha256=feature_encoder_manifest_sha256,
        )
        grid = replace(
            grid,
            image_features=features,
            settings={
                **grid.settings,
                "encoder_name": metadata["encoder"]["encoder_id"],
                "encoder_manifest_sha256": feature_encoder_manifest_sha256,
            },
        )
        feature_provenance = {
            "kind": "frozen_histology",
            "cache_path": str(cache.resolve()),
            "metadata": metadata,
            "feature_dim": int(features.shape[1]),
            "encoder_manifest_sha256": feature_encoder_manifest_sha256,
            "contains_spatial_rna_derived_features": False,
        }
    panel = _axis(
        DEFAULT_EMBEDDING_GENES if embedding_gene_ids is None else embedding_gene_ids,
        "embedding gene panel",
    )
    with _read_h5ad(_verified_source_path(root, sources["rna"])) as adata:
        counts, genes = extract_raw_counts(adata)
        original_ids = _axis(adata.obs_names, "RNA observation IDs")
        replicates = _replicates(adata, sources["rna"])
        reference = build_reference(
            counts,
            genes,
            adata.obs["cell_type"].astype(str).to_numpy(),
            embedding_gene_ids=panel,
            max_states=max_states,
        )
    if gene_ids is not None:
        requested = _axis(gene_ids, "requested RNA gene panel")
        retained = tuple(
            dict.fromkeys((*requested, *panel, *(g for _, g in DEFAULT_PROTEIN_MARKERS)))
        )
        if not set(retained) <= set(reference.gene_ids):
            raise ValueError("requested genes are absent from the independent RNA count axis")
        columns = [reference.gene_ids.index(g) for g in retained]
        reference = replace(
            reference,
            gene_ids=retained,
            expression=reference.expression[:, columns],
            variance=reference.variance[:, columns],
        )
    proteins, protein_provenance = _protein_evidence(
        root,
        sources["protein"],
        reference,
        grid,
        protein_cofactor,
        rna_cofactor,
    )
    selected = set(map(tuple, grid.bin_keys))

    def adjacent(a, b):
        x, y = grid.bin_keys[a]
        u, v = grid.bin_keys[b]
        return max(abs(x - u), abs(y - v)) <= 1 and (
            x == u or y == v or ((x, v) in selected and (u, y) in selected)
        )

    graph = build_graph(
        grid.coordinates,
        graph_radius,
        tissue_component_ids=grid.component_ids,
        max_neighbors=graph_max_neighbors,
        edge_allowed=adjacent,
        coordinate_units="registered_pixel",
    )
    inputs = PredictionInputs(
        specimen_id=entry["specimen_id"],
        patient_id=entry["patient_id"],
        cell_ids=grid.region_ids,
        coordinates_um=grid.coordinates,
        image_features=grid.image_features,
        reference=reference,
        proteins=proteins,
        graph=graph,
        prediction_unit="tissue_region",
        coordinate_units="registered_pixel",
    )
    provenance = {
        "schema": "heir.htapp_biopsy_regions.v1",
        "biopsy_id": biopsy_id,
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "matching_scope": "published_biopsy",
        "section_identity": "unknown",
        "patient_grouping_basis": entry["metadata"]["participant_group_basis"],
        "prediction_unit": "tissue_region",
        "coordinate_units": "registered_pixel",
        "source_integrity": "consumed_input_payload_sizes_and_sha256_verified",
        "image_features": feature_provenance,
        "sources": {key: sources[key] for key in ("he", "rna", "protein")},
        "grid": {
            "settings": grid.settings,
            "region_ids": list(grid.region_ids),
            "coordinates": grid.coordinates.tolist(),
            "bin_keys": grid.bin_keys.tolist(),
            "tissue_fraction": grid.tissue_fraction.tolist(),
            "component_ids": list(grid.component_ids),
        },
        "rna": {
            "source_observation_axis": _axis_provenance(original_ids),
            "source_replicates": _counts_by_label(replicates),
            "annotation_column": "cell_type",
            "annotation_policy": "published_cohort_curated_rna_only",
            "annotation_fit_scope": "published_cohort_not_training_fold_fitted",
            "measurement": "retained_cellbender_qc_counts",
            "normalization": "full_measured_rna_library_10000",
            "embedding_gene_ids": list(panel),
            "gene_ids": list(reference.gene_ids),
            "max_states": max_states,
        },
        "protein": protein_provenance,
        "graph": {
            "radius": graph_radius,
            "max_neighbors": graph_max_neighbors,
            "coordinate_units": "registered_pixel",
            "distance_units": "registered_pixel",
            "radius_units": "registered_pixel",
            "support": "adjacent_selected_bins_no_corner_gaps",
        },
        "registration": {
            "status": "published_coarse_registration_unvalidated",
            "physical_calibration": "unresolved",
            "same_cell_correspondence": False,
        },
    }
    return inputs, provenance


def _restore_grid(prediction, provenance):
    record = provenance["grid"]
    settings = record["settings"]
    ids = tuple(record["region_ids"])
    keys = np.asarray(record["bin_keys"])
    xy = np.asarray(record["coordinates"], dtype=np.float32)
    shape = settings.get("image_shape", [])
    size = settings.get("bin_size_pixels")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or len(shape) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape)
        or settings.get("origin_pixels") != [0, 0]
        or settings.get("coordinate_units") != "registered_pixel"
        or settings.get("prediction_unit") != "tissue_region"
        or settings.get("array_row_is_y") is not True
        or settings.get("display_origin") != "lower"
        or keys.shape != (len(ids), 2)
        or keys.dtype.kind not in "iu"
        or np.any(keys < 0)
    ):
        raise ValueError("prediction and saved H&E grid provenance differ")
    lower = keys * size
    image_extent = np.asarray(shape[::-1])
    upper = np.minimum(lower + size, image_extent)
    centers = ((lower + upper) / 2).astype(np.float32)
    fractions = np.asarray(record["tissue_fraction"])
    actual_xy = prediction.coordinates
    if hasattr(actual_xy, "detach"):
        actual_xy = actual_xy.detach().cpu().numpy()
    if (
        provenance.get("schema") != "heir.htapp_biopsy_regions.v1"
        or prediction.prediction_unit != "tissue_region"
        or prediction.coordinate_units != "registered_pixel"
        or prediction.patient_id != provenance["patient_id"]
        or prediction.specimen_id != provenance["specimen_id"]
        or prediction.cell_ids != ids
        or keys.shape != (len(ids), 2)
        or xy.shape != (len(ids), 2)
        or not np.array_equal(actual_xy, xy)
        or settings.get("schema") != "heir.he_region_grid.v1"
        or ids != tuple(f"he_region:{x}:{y}" for x, y in keys)
        or len(set(map(tuple, keys))) != len(keys)
        or np.any(lower >= image_extent)
        or not np.array_equal(centers, xy)
        or fractions.shape != (len(ids),)
        or not np.isfinite(fractions).all()
        or np.any((fractions < 0) | (fractions > 1))
        or len(record["component_ids"]) != len(ids)
    ):
        raise ValueError("prediction and saved H&E grid provenance differ")
    return RegionGrid(
        region_ids=ids,
        coordinates=xy,
        image_features=np.empty((len(ids), 0)),
        bin_keys=keys,
        tissue_fraction=fractions,
        component_ids=tuple(record["component_ids"]),
        settings=settings,
    )


def load_cohort_targets(root, biopsy_id, prediction, provenance, *, gene_ids):
    """Sum measured Slide-seq counts within existing regions, separately per replicate.

    Accepts PredictionInputs or an exported/reloaded Prediction. RNA, protein, and
    H&E payloads are not reopened. Repeated regions retain the same patient ID.
    """
    from scipy.sparse import csr_matrix

    entry = discover_cohort(root)[biopsy_id]
    if provenance["biopsy_id"] != biopsy_id or any(
        provenance["sources"][role]["sha256"] != entry["role_sources"][role]["sha256"]
        for role in ("he", "rna", "protein")
    ):
        raise ValueError("saved input sources do not match the selected published biopsy")
    grid = _restore_grid(prediction, provenance)
    requested = _axis(gene_ids, "target gene panel")
    source = entry["role_sources"]["targets"]
    with _read_h5ad(_verified_source_path(root, source)) as adata:
        counts, genes = extract_raw_counts(adata)
        if not set(requested) <= set(genes):
            raise ValueError("requested target genes are absent from the retained Slide-seq axis")
        original_ids = _axis(adata.obs_names, "Slide-seq observation IDs")
        replicates = _replicates(adata, source)
        keys, rows = _groups(grid, adata.obs[["x", "y"]].to_numpy(), replicates)
        keep = np.flatnonzero(rows >= 0)
        pooling = csr_matrix(
            (np.ones(len(keep)), (rows[keep], keep)), shape=(len(keys), adata.n_obs)
        )
        values = _dense(pooling @ counts[:, [genes.index(g) for g in requested]]).astype(np.float32)
    observation_ids = tuple(f"slide_seq:{rep}:{grid.region_ids[index]}" for rep, index in keys)
    targets = SpatialTargets(
        observation_ids=observation_ids,
        gene_ids=requested,
        expression=values,
        observed=np.ones(values.shape, dtype=bool),
        patient_id=entry["patient_id"],
        specimen_id=entry["specimen_id"],
    )
    operator = ObservationOperator(
        cell_ids=grid.region_ids,
        observation_ids=observation_ids,
        rows=np.arange(len(keys)),
        columns=np.asarray([index for _, index in keys]),
        weights=np.ones(len(keys)),
    )
    return (
        targets,
        operator,
        {
            "source": source,
            "aggregation": "sum_counts_per_he_region_per_assay_replicate",
            "source_integrity": "consumed_target_payload_size_and_sha256_verified",
            "source_observation_axis": _axis_provenance(original_ids),
            "source_replicates": _counts_by_label(replicates),
            "observation_replicates": [r for r, _ in keys],
            "excluded_source_observations": int((rows < 0).sum()),
            "region_bead_counts": np.bincount(rows[rows >= 0], minlength=len(keys)).tolist(),
            "gene_ids": list(requested),
            "coordinate_units": "registered_pixel",
            "patient_id": entry["patient_id"],
            "specimen_id": entry["specimen_id"],
            "matching_scope": "published_biopsy",
            "section_identity": "unknown",
        },
    )


def _retained_gene_axis(path):
    """Read only the actual retained count axis, never processed .var or expression."""
    import h5py

    with h5py.File(path, "r") as handle:
        axis = handle["uns/counts_var"]
        if not isinstance(axis, h5py.Dataset) or axis.ndim != 1 or axis.dtype.kind not in "OS":
            raise ValueError("measurement preflight requires a retained string gene-ID vector")
        genes = _axis(axis.asstr()[:], "retained measured genes")
        counts = handle["obsm/counts"]
        shape = counts.shape if isinstance(counts, h5py.Dataset) else counts.attrs.get("shape")
        if shape is None or len(shape) != 2 or shape[1] != len(genes):
            raise ValueError("retained gene axis differs from the count matrix dimensions")
    return genes


def load_measurement_gene_axes(root, biopsy_id, *, verification_cache=None):
    """Predeclare spatial/RNA gene availability without loading or fitting a reference."""
    entry = discover_cohort(root)[biopsy_id]
    sources = {role: entry["role_sources"][role] for role in ("targets", "rna")}
    axes = {
        role: _retained_gene_axis(
            _verified_source_path(root, source, verification_cache=verification_cache)
        )
        for role, source in sources.items()
    }
    return {
        "spatial_gene_ids": axes["targets"],
        "independent_rna_gene_ids": axes["rna"],
        "sources": sources,
    }


def load_measurement_substrate(
    root, biopsy_id, *, base_region_size=100, min_tissue_fraction=0.1, verification_cache=None
):
    """Load raw measurement support without model, reference fitting, or embeddings."""
    from scipy.sparse import csr_matrix

    from .measurement import MeasurementSubstrate

    verification_cache = {} if verification_cache is None else verification_cache
    entry = discover_cohort(root)[biopsy_id]
    sources = {role: entry["role_sources"][role] for role in ("he", "targets", "rna")}
    axes = load_measurement_gene_axes(root, biopsy_id, verification_cache=verification_cache)
    grid = build_region_grid(
        _verified_source_path(root, sources["he"], verification_cache=verification_cache),
        bin_size_pixels=base_region_size,
        min_tissue_fraction=min_tissue_fraction,
        compute_features=False,
    )
    with _read_h5ad(
        _verified_source_path(root, sources["targets"], verification_cache=verification_cache)
    ) as adata:
        counts, genes = extract_raw_counts(adata)
        if genes != axes["spatial_gene_ids"]:
            raise ValueError("spatial gene axis changed since measurement preflight")
        counts = csr_matrix(counts, copy=True)
        if counts.data.size and counts.data.max() > np.iinfo(np.int64).max:
            raise ValueError("raw counts exceed signed integer aggregation capacity")
        counts.data = np.rint(counts.data).astype(np.int64)
        bead_ids = _axis(adata.obs_names, "Slide-seq source bead IDs")
        coordinates = adata.obs[["x", "y"]].to_numpy(dtype=np.float64)
        replicates = tuple(_replicates(adata, sources["targets"]).tolist())
    mapping = point_to_grid(grid, coordinates)
    provenance = {
        "schema": "heir.raw_measurement_substrate.v1",
        "biopsy_id": biopsy_id,
        "patient_id": entry["patient_id"],
        "specimen_id": entry["specimen_id"],
        "sources": sources,
        "source_observation_axis": _axis_provenance(bead_ids),
        "source_replicates": _counts_by_label(replicates),
        "metadata": entry["metadata"],
        "matching_scope": "published_biopsy",
        "section_identity": "unknown",
        "core_identity": "unknown",
        "replicate_linkage": "unknown",
        "linkage_basis": "no documented section/core linkage in consumed acquisition metadata",
        "registration": "published_coarse_registration_unvalidated",
        "physical_calibration": "unknown",
        "coordinate_units": "registered_pixel",
        "grid_selection": "H&E_only_fixed_base_grid",
        "support_limitation": (
            "shared bead-occupied base bins; same sampled tissue within a bin is unknown"
        ),
        "total_exposure_basis": "all_retained_spatial_measured_gene_ids",
        "outside_selected_he_regions": int((mapping < 0).sum()),
    }
    return MeasurementSubstrate(
        biopsy_id,
        entry["patient_id"],
        entry["specimen_id"],
        grid,
        tuple(genes),
        axes["independent_rna_gene_ids"],
        bead_ids,
        coordinates,
        replicates,
        counts,
        mapping,
        provenance,
    )
