"""Local HTAPP adapter with separate prediction and spatial-target readers.

The manifest binds biological matching, physical frames, count store and gene
axis, cached feature provenance, and protein calibration. It does not assert
that the proposed thirteen participants are eligible. No download is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import (
    mask_component_ids,
    prepare_geometry,
    regional_protein_evidence,
    tissue_mask_edge_filter,
)
from .reference import build_reference, extract_raw_counts
from .schema import PredictionInputs, ProteinEvidence, ProteinObservationOperator, SpatialTargets

_MATCH_FIELDS = ("patient_id", "sampling_episode_id", "lesion_id", "represented_region_id")
_IDENTITY_FIELDS = (*_MATCH_FIELDS, "core_id", "section_id", "specimen_id")
_BAD_ID = {"", "unknown", "unresolved", "none", "nan", "na", "n/a", "tbd"}
_CORRESPONDENCE = {"same_cell", "same_section_soft", "serial_section", "region"}


def _read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError(
            "HTAPP manifest schema_version must be 2; "
            "migrate protein identities and region settings"
        )
    return manifest, path.parent


def _path(manifest, root, key):
    value = manifest.get("paths", {}).get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing paths.{key}")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _identifier(value, label):
    if not isinstance(value, str) or value.strip().lower() in _BAD_ID:
        raise ValueError(f"{label} is unresolved")
    return value


def validate_matching(manifest):
    """Require documented biological match; different cores remain explicit."""
    patient = _identifier(manifest.get("patient_id"), "patient_id")
    _identifier(manifest.get("specimen_id"), "specimen_id")
    matching = manifest.get("matching", {})
    query, reference = matching.get("query", {}), matching.get("reference", {})
    for field in _MATCH_FIELDS:
        left = _identifier(query.get(field), f"matching.query.{field}")
        right = _identifier(reference.get(field), f"matching.reference.{field}")
        if left != right:
            raise ValueError(f"query/reference {field} differs")
    if query["patient_id"] != patient:
        raise ValueError("manifest patient differs from matched patient")
    query_specimen = _identifier(query.get("specimen_id"), "matching.query.specimen_id")
    if query_specimen != manifest["specimen_id"]:
        raise ValueError("matching.query.specimen_id differs from manifest specimen_id")
    for side, item in (("query", query), ("reference", reference)):
        for field in ("core_id", "section_id", "assay"):
            _identifier(item.get(field), f"matching.{side}.{field}")
    if reference["assay"] not in {"scRNA-seq", "snRNA-seq", "scFFPE-seq"}:
        raise ValueError("reference assay must identify independent dissociated sc/snRNA")
    if matching.get("reference_independent_of_spatial_targets") is not True:
        raise ValueError("matched reference must be independent of spatial RNA targets")
    _identifier(matching.get("evidence"), "matching.evidence")
    if manifest.get("paths", {}).get("proteins"):
        protein = matching.get("protein", {})
        for field in (*_IDENTITY_FIELDS, "assay"):
            _identifier(protein.get(field), f"matching.protein.{field}")
        for field in _MATCH_FIELDS:
            if protein[field] != query[field]:
                raise ValueError(f"query/protein {field} differs")


def _validate_protein_identity(manifest, adata, registration):
    """Bind actual protein rows and both registration frames to biological identities."""
    query, protein = (manifest["matching"][side] for side in ("query", "protein"))
    for field in _IDENTITY_FIELDS:
        values = _column(adata.obs, field).astype(str)
        if set(values) != {protein[field]}:
            raise ValueError(f"protein payload {field} differs from matching.protein")
    # Legacy unscoped identities cannot distinguish adjacent source/target sections.
    for field in _IDENTITY_FIELDS:
        prefixes = ("", "source_", "target_", "protein_", "he_")
        if any(prefix + field in registration for prefix in prefixes):
            raise ValueError(
                f"ambiguous registration {field}; use source_identity and target_identity"
            )
    for side, expected in (("source", protein), ("target", query)):
        identity = registration.get(f"{side}_identity")
        if not isinstance(identity, dict):
            raise ValueError(f"registration requires {side}_identity")
        for field in _IDENTITY_FIELDS:
            value = _identifier(identity.get(field), f"registration.{side}_identity.{field}")
            if value != expected[field]:
                raise ValueError(
                    f"registration {side}_identity {field} differs from {side} payload"
                )
    correspondence = registration.get("correspondence")
    same_core = protein["core_id"] == query["core_id"]
    same_section = protein["section_id"] == query["section_id"]
    if correspondence in {"same_cell", "same_section_soft"} and not (same_core and same_section):
        raise ValueError("same-section protein correspondence requires matching core and section")
    if correspondence == "serial_section" and (not same_core or same_section):
        raise ValueError(
            "serial-section protein correspondence requires one core and distinct sections"
        )


def _read_h5ad(path):
    try:
        import anndata
    except ImportError as exc:
        raise ImportError("HTAPP .h5ad loading requires pip install 'heir-spatial[htapp]'") from exc
    return anndata.read_h5ad(path)


def _column(table, name):
    try:
        value = table[name]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"required column {name!r} is missing") from exc
    return np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value)


def _read_cells(path):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError("HTAPP cells require pip install 'heir-spatial[htapp]'") from exc
    table = parquet.read_table(path)
    names = ("cell_id", "x_um", "y_um", "segmentation_quality", "tissue_component_id")
    if not set(names).issubset(table.column_names):
        raise ValueError(f"cells.parquet requires columns {names}")
    columns = {name: np.asarray(table[name].to_pylist()) for name in names}
    quality = np.asarray(columns["segmentation_quality"], dtype=float)
    if not np.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
        raise ValueError("segmentation quality must lie in [0, 1]")
    for value in columns["tissue_component_id"]:
        _identifier(value, "tissue_component_id")
    return columns


def _read_features(path, cell_ids):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["cell_ids"]).astype(str)
            values = np.asarray(archive["image_features"], dtype=np.float32)
    else:
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("Zarr feature caches require heir-spatial[htapp]") from exc
        group = zarr.open_group(str(path), mode="r")
        ids = np.asarray(group["cell_ids"]).astype(str)
        values = np.asarray(group["image_features"], dtype=np.float32)
    if len(set(ids)) != len(ids) or set(ids) != set(cell_ids):
        raise ValueError("cached image feature IDs do not exactly match H&E nucleus IDs")
    if values.ndim != 2 or len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError("cached image embeddings must be finite and keyed by nucleus ID")
    lookup = {name: i for i, name in enumerate(ids)}
    return values[[lookup[name] for name in cell_ids]]


def _feature_provenance(manifest):
    provenance = manifest.get("feature_provenance", {})
    _identifier(provenance.get("encoder_name"), "feature_provenance.encoder_name")
    digest = provenance.get("checkpoint_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("feature_provenance.checkpoint_sha256 must be a SHA-256")
    if provenance.get("frozen") is not True:
        raise ValueError("initial image features must come from a frozen encoder")
    if provenance.get("contains_spatial_rna_derived_features") is not False:
        raise ValueError("image feature provenance must exclude spatial-RNA-derived features")


def _mask_filter(manifest, root, coordinates):
    if not manifest.get("paths", {}).get("tissue_mask"):
        return None
    with np.load(_path(manifest, root, "tissue_mask"), allow_pickle=False) as raster:
        return tissue_mask_edge_filter(
            coordinates,
            raster["mask"],
            origin_um=raster["origin_um"],
            pixel_size_um=raster["pixel_size_um"],
        )


def _proteins(manifest, root, reference, cell_ids, coordinates, components):
    settings = manifest.get("proteins", {})
    marker_table = settings.get("marker_table")
    if not isinstance(marker_table, list) or not marker_table:
        raise ValueError("protein evidence requires an explicit marker interpretation table")
    adata = _read_h5ad(_path(manifest, root, "proteins"))
    measurement_ids = tuple(map(str, adata.obs_names))
    if len(set(measurement_ids)) != len(measurement_ids):
        raise ValueError("protein measurement IDs must be unique")
    for measurement_id in measurement_ids:
        _identifier(measurement_id, "protein measurement ID")
    registration = json.loads(_path(manifest, root, "registration").read_text())
    correspondence = registration.get("correspondence")
    if correspondence not in _CORRESPONDENCE:
        raise ValueError("registration correspondence type must be explicit")
    _validate_protein_identity(manifest, adata, registration)
    if "spatial" not in adata.obsm or "observed" not in adata.layers:
        raise ValueError("protein measurements require coordinates and an observed mask")
    source_coordinates = prepare_geometry(adata.obsm["spatial"], registration)
    protein_components = _column(adata.obs, "tissue_component_id").astype(str)
    for component in protein_components:
        _identifier(component, "protein tissue_component_id")
    source_ids = tuple(str(x) for x in adata.var_names)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("protein marker IDs must be unique")
    marker_ids = tuple(row["marker_id"] for row in marker_table)
    if len(set(marker_ids)) != len(marker_ids) or not set(marker_ids).issubset(source_ids):
        raise ValueError("marker table does not align with measured marker IDs")
    columns = [source_ids.index(marker) for marker in marker_ids]
    raw = adata.X[:, columns]
    raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    observed = adata.layers["observed"][:, columns]
    observed = observed.toarray() if hasattr(observed, "toarray") else np.asarray(observed)
    if observed.dtype != bool:
        raise ValueError("protein observed mask must be stored as boolean")
    if "quality" in adata.layers:
        quality = adata.layers["quality"][:, columns]
        quality = quality.toarray() if hasattr(quality, "toarray") else np.asarray(quality)
    else:
        quality = np.ones(raw.shape)
    if quality.shape != raw.shape or not np.isfinite(quality[observed]).all():
        raise ValueError("protein quality must be finite at observed measurements")
    if ((quality[observed] < 0) | (quality[observed] > 1)).any():
        raise ValueError("protein quality must lie in [0, 1]")
    observed = observed & (quality > 0)
    quality = np.where(observed, quality, 0)
    values = np.zeros(raw.shape, dtype=np.float32)
    state_values = np.zeros((len(reference.state_ids), len(marker_ids)), dtype=np.float32)
    for j, marker in enumerate(marker_table):
        if marker.get("interpretation") not in {"identity", "total_receptor"}:
            raise ValueError(
                "initial RNA/protein proxies support identity and total_receptor only; "
                "functional markers require separately qualified constraints"
            )
        gene = marker.get("gene_id")
        if gene not in reference.gene_ids:
            raise ValueError(f"marker {marker_ids[j]} has no mapped reference gene")
        for key in ("intensity_half_saturation", "rna_half_saturation"):
            value = marker.get(key)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"marker {marker_ids[j]} requires positive {key}")
        # Fixed saturating transforms preserve masks and expose scale assumptions.
        column = np.where(observed[:, j], raw[:, j], 0)
        if not np.isfinite(column).all() or (column < 0).any():
            raise ValueError("observed protein values must be finite and nonnegative")
        values[:, j] = column / (column + marker["intensity_half_saturation"])
        expression = reference.expression[:, reference.gene_ids.index(gene)]
        state_values[:, j] = expression / (expression + marker["rna_half_saturation"])
    if correspondence == "same_cell":
        if registration.get("same_cell_verified") is not True:
            raise ValueError("same-cell protein evidence requires verified correspondence")
        pair_ids = tuple(str(x) for x in _column(adata.obs, "paired_he_cell_id"))
        if len(set(pair_ids)) != len(pair_ids) or not set(pair_ids).issubset(cell_ids):
            raise ValueError("protein paired cell IDs must be unique known H&E nucleus IDs")
        lookup = {name: i for i, name in enumerate(cell_ids)}
        for source, name in enumerate(pair_ids):
            target = lookup[name]
            if protein_components[source] != str(components[target]):
                raise ValueError("same-cell pairing crosses tissue components")
        operator = ProteinObservationOperator(
            cell_ids=cell_ids,
            observation_ids=tuple(f"cell:{name}" for name in pair_ids),
            rows=np.arange(len(pair_ids), dtype=np.int64),
            columns=np.asarray([lookup[name] for name in pair_ids], dtype=np.int64),
            weights=np.ones(len(pair_ids), dtype=np.float32),
        )
        coverage = {
            "shared_regions": len(pair_ids),
            "excluded_he_regions": len(cell_ids) - len(pair_ids),
            "excluded_protein_regions": 0,
            "unobserved_region_markers": int((~observed).sum()),
        }
    else:
        if "radius_um" in settings:
            raise ValueError("protein radius_um is obsolete; declare bin_size_um and origin_um")
        if manifest.get("paths", {}).get("tissue_mask"):
            with np.load(_path(manifest, root, "tissue_mask"), allow_pickle=False) as raster:
                split = mask_component_ids(
                    np.concatenate((coordinates, source_coordinates)),
                    np.concatenate((components, protein_components)),
                    raster["mask"],
                    origin_um=raster["origin_um"],
                    pixel_size_um=raster["pixel_size_um"],
                )
            components, protein_components = split[: len(coordinates)], split[len(coordinates) :]
        values, observed, quality, operator, coverage = regional_protein_evidence(
            cell_ids,
            coordinates,
            source_coordinates,
            values,
            observed,
            bin_size_um=settings.get("bin_size_um", 100.0),
            origin_um=settings.get("origin_um", (0.0, 0.0)),
            quality=quality,
            he_component_ids=components,
            protein_component_ids=protein_components,
        )
    return ProteinEvidence(
        marker_ids=marker_ids,
        values=values,
        observed=observed,
        state_values=state_values,
        correspondence=correspondence,
        operator=operator,
        quality=quality,
        coverage=coverage,
    )


def prepare_sample(manifest_path) -> PredictionInputs:
    """Read only allowed prediction inputs; never open spatial target files.

    All imported RNA annotations must originate in the matched dissociated
    reference. RCTD/OT and other spatially transferred labels are not accepted.
    """
    manifest, root = _read_manifest(manifest_path)
    validate_matching(manifest)
    _feature_provenance(manifest)
    cells = _read_cells(_path(manifest, root, "cells"))
    cell_ids = tuple(str(x) for x in cells["cell_id"])
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("H&E nucleus IDs must be unique")
    coordinates = np.column_stack((cells["x_um"], cells["y_um"])).astype(np.float32)
    features = _read_features(_path(manifest, root, "image_features"), cell_ids)
    settings = manifest.get("reference", {})
    if settings.get("annotation_provenance") != "matched_dissociated_rna":
        raise ValueError("RNA state labels require matched_dissociated_rna provenance")
    type_column = settings.get("broad_type_column")
    _identifier(type_column, "reference.broad_type_column")
    state_column = settings.get("state_column")
    for name in (type_column, state_column):
        forbidden = ("rctd", "transferred", "ot_")
        if name is not None and any(part in name.lower() for part in forbidden):
            raise ValueError("spatial-RNA-derived annotation columns are excluded")
        if name is not None and name.lower() == "ot":
            raise ValueError("spatial-RNA-derived annotation columns are excluded")
    rna = _read_h5ad(_path(manifest, root, "rna"))
    patients = _column(rna.obs, settings.get("patient_id_column", "patient_id")).astype(str)
    if set(patients) != {manifest["patient_id"]}:
        raise ValueError("RNA payload patient identities differ from the matched specimen")
    matching_columns = settings.get("matching_columns", {})
    for field in _MATCH_FIELDS[1:]:
        values = _column(rna.obs, matching_columns.get(field, field)).astype(str)
        if set(values) != {manifest["matching"]["reference"][field]}:
            raise ValueError(f"RNA payload {field} differs from the matching metadata")
    if settings.get("assay_column") is not None:
        assays = _column(rna.obs, settings["assay_column"]).astype(str)
        if set(assays) != {manifest["matching"]["reference"]["assay"]}:
            raise ValueError("RNA payload assay differs from the matching metadata")
    counts, genes = extract_raw_counts(
        rna,
        source=settings.get("counts_source", "obsm"),
        gene_id_column=settings.get("gene_id_column"),
        raw_declared=settings.get("raw_counts_declared") is True,
    )
    if len(set(map(str, rna.obs_names))) != rna.n_obs:
        raise ValueError("RNA-cell IDs must be unique")
    reference = build_reference(
        counts,
        genes,
        _column(rna.obs, type_column),
        embedding_gene_ids=settings.get("embedding_gene_ids", ()),
        state_labels=None if state_column is None else _column(rna.obs, state_column),
        max_states=settings.get("max_states", 64),
        library_size=settings.get("library_size", 10000),
    )
    from .graph import build_graph

    graph_settings = manifest.get("graph", {})
    graph = build_graph(
        coordinates,
        radius_um=graph_settings.get("radius_um", 100.0),
        tissue_component_ids=cells["tissue_component_id"],
        max_neighbors=graph_settings.get("max_neighbors", 32),
        edge_allowed=_mask_filter(manifest, root, coordinates),
    )
    proteins = None
    if manifest.get("paths", {}).get("proteins"):
        proteins = _proteins(
            manifest,
            root,
            reference,
            cell_ids,
            coordinates,
            cells["tissue_component_id"],
        )
    # Registration sensitivity is not calibrated by an affine uncertainty scalar.
    # Leave it absent until explicit perturbation experiments have been run.
    return PredictionInputs(
        specimen_id=manifest["specimen_id"],
        patient_id=manifest["patient_id"],
        cell_ids=cell_ids,
        coordinates_um=coordinates,
        image_features=features,
        reference=reference,
        proteins=proteins,
        graph=graph,
        registration_sensitivity=None,
    )


def load_spatial_targets(manifest_path) -> SpatialTargets:
    """Explicit training/evaluation reader, separate from prepare_sample."""
    manifest, root = _read_manifest(manifest_path)
    settings = manifest.get("targets", {})
    if settings.get("coordinate_units") != "um":
        raise ValueError("spatial target coordinates must explicitly use um")
    if settings.get("expression_scale") != "nonnegative_linear":
        raise ValueError("spatial targets must declare nonnegative_linear expression")
    adata = _read_h5ad(_path(manifest, root, "spatial_targets"))
    for field in ("patient_id", "specimen_id"):
        values = _column(adata.obs, settings.get(f"{field}_column", field)).astype(str)
        if set(values) != {manifest[field]}:
            raise ValueError(f"spatial target payload {field} differs from manifest")
    if "spatial" not in adata.obsm or "observed" not in adata.layers:
        raise ValueError("spatial targets require physical coordinates and gene availability")
    coordinates = np.asarray(adata.obsm["spatial"])
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("spatial target coordinates must be finite N-by-2 values")
    source = settings.get("expression_source", "X")
    if source == "X":
        values = adata.X
    elif source.startswith("layer:"):
        values = adata.layers[source.split(":", 1)[1]]
    else:
        raise ValueError("target expression_source must be X or layer:<name>")
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    observed = adata.layers["observed"]
    observed = observed.toarray() if hasattr(observed, "toarray") else np.asarray(observed)
    gene_column = settings.get("gene_id_column")
    gene_axis = adata.var_names if gene_column is None else _column(adata.var, gene_column)
    genes = tuple(map(str, gene_axis))
    return SpatialTargets(
        observation_ids=tuple(map(str, adata.obs_names)),
        gene_ids=genes,
        expression=values,
        observed=observed,
        patient_id=manifest["patient_id"],
        specimen_id=manifest["specimen_id"],
    )


def specimen_eligibility(manifest_paths, *, validate_payload=True):
    """Derive per-specimen eligibility from local metadata and actual files.

    ``prediction_eligible`` includes payload validation by default. Evaluation
    file presence is reported separately and does not claim scientific or
    registration qualification. No spatial RNA target is read here.
    """
    rows = []
    for path in manifest_paths:
        row = {
            "manifest_path": str(path),
            "patient_id": None,
            "specimen_id": None,
            "prediction_eligible": False,
            "payload_checked": False,
            "evaluation_files_present": False,
            "reasons": [],
        }
        try:
            manifest, root = _read_manifest(path)
            row.update(
                patient_id=manifest.get("patient_id"), specimen_id=manifest.get("specimen_id")
            )
            validate_matching(manifest)
            for key in ("cells", "image_features", "rna"):
                if not _path(manifest, root, key).exists():
                    row["reasons"].append(f"missing input file: {key}")
            if manifest.get("paths", {}).get("proteins"):
                for key in ("proteins", "registration"):
                    if not _path(manifest, root, key).is_file():
                        row["reasons"].append(f"missing input file: {key}")
            row["evaluation_files_present"] = all(
                bool(manifest.get("paths", {}).get(key)) and _path(manifest, root, key).is_file()
                for key in ("spatial_targets", "observation")
            )
            if not row["reasons"] and validate_payload:
                inputs = prepare_sample(path)
                row["payload_checked"] = True
                row["n_rna_genes"] = len(inputs.reference.gene_ids)
                row["n_he_nuclei"] = len(inputs.cell_ids)
                row["n_reference_states"] = len(inputs.reference.state_ids)
                if inputs.proteins is not None:
                    row["protein_coverage"] = inputs.proteins.coverage
                row["prediction_eligible"] = True
        except (ValueError, OSError, KeyError, ImportError, TypeError) as exc:
            row["reasons"].append(str(exc))
        rows.append(row)
    return rows
