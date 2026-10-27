"""Freeze retained-count regional outcomes without reading predictions or fitting models.

The published pixel coordinates and centroid occupancy support a conditional regional
measurement diagnostic. They do not certify tissue identity, physical calibration,
independent registration, actual capture boundaries, or cell-type correspondence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cohort import (
    _axis,
    _axis_provenance,
    _read_h5ad,
    _replicates,
    _verified_source_path,
    discover_cohort,
)
from .reference import extract_raw_counts
from .regions import build_region_grid, point_to_grid

_METADATA_SOURCES = (
    "metadata/selected_13_biopsies.tsv",
    "metadata/selected_13_verified_file_manifest.tsv",
)


@dataclass(frozen=True)
class RegionalCounts:
    replicate: str
    counts: np.ndarray
    observation_counts: np.ndarray
    source_rows: np.ndarray
    region_assignments: np.ndarray


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _table(path, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _verify_bindings(root, bindings, *, cache=None):
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError("source bindings require paths within the cohort root")
        try:
            _verified_source_path(
                root,
                dict(working_path=relative, sha256=expected, bytes=path.stat().st_size),
                verification_cache=cache,
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"frozen source identity differs: {relative}") from exc


def _array_prefixes(replicates):
    prefixes = [rep.replicate.replace(":", "__") for rep in replicates]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("replicate labels collide in the serialized array names")
    return prefixes


def aggregate_retained_counts(counts, gene_ids, coordinates, replicates, grid, pool):
    """Sum each retained observation once; never merge source assay replicates."""
    from scipy.sparse import csr_matrix, issparse

    genes, pool = _axis(gene_ids, "retained count genes"), _axis(pool, "normalization genes")
    labels = np.asarray(replicates, dtype=str)
    coordinates = np.asarray(coordinates, dtype=float)
    if labels.ndim != 1 or any(not s.strip() or s.lower() in {"nan", "none"} for s in labels):
        raise ValueError("retained observations require nonempty replicate labels")
    if counts.shape != (len(labels), len(genes)) or coordinates.shape != (len(labels), 2):
        raise ValueError("retained measurement axes differ")
    if not set(pool) <= set(genes):
        raise ValueError("fixed normalization pool is absent from retained counts")
    values = counts.data if issparse(counts) else np.asarray(counts)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("retained RNA measurements must be finite nonnegative integer counts")
    mapped = point_to_grid(grid, coordinates)
    chosen = counts[:, [genes.index(gene) for gene in pool]]
    result = []
    for label in sorted(set(labels)):
        source_rows = np.flatnonzero(labels == label)
        assignment = mapped[source_rows]
        present = assignment >= 0
        operator = csr_matrix(
            (np.ones(present.sum()), (assignment[present], source_rows[present])),
            shape=(len(grid.region_ids), len(labels)),
        )
        summed = operator @ chosen
        summed = summed.toarray() if issparse(summed) else np.asarray(summed)
        result.append(
            RegionalCounts(
                replicate=label,
                counts=np.asarray(summed, dtype=np.float64),
                observation_counts=np.bincount(assignment[present], minlength=len(grid.region_ids)),
                source_rows=source_rows,
                region_assignments=assignment,
            )
        )
    if not result:
        raise ValueError("retained count source has no replicates")
    return result


def _validate_qc(qc):
    integer_keys = ("min_observations", "min_regions", "min_positive_regions")
    positive_keys = ("min_pool_counts", "min_gene_counts", "min_log_quantile_range")
    expected = set(integer_keys + positive_keys + ("min_positive_fraction",))
    if set(qc) != expected:
        raise ValueError("QC requires the complete fixed threshold set")
    for key in integer_keys:
        if isinstance(qc[key], bool) or not isinstance(qc[key], int) or qc[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in positive_keys + ("min_positive_fraction",):
        if isinstance(qc[key], bool) or not np.isfinite(qc[key]) or qc[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if qc["min_positive_fraction"] > 1:
        raise ValueError("min_positive_fraction must be at most one")


def qualify_measurements(replicates, protein_present, pool, primary, secondary, qc):
    """Shared geometry, then identical count QC, then a common gene eligibility set.

    The 10th--90th percentile range uses NumPy's linear interpolation. Exposure
    always sums the full frozen pool, including genes that fail outcome QC.
    """
    _validate_qc(qc)
    pool = _axis(pool, "normalization genes")
    primary, secondary = _axis(primary, "primary genes"), _axis(secondary, "secondary genes")
    if not set(primary + secondary) <= set(pool):
        raise ValueError("every outcome must be within the fixed normalization pool")
    if not replicates or len({r.replicate for r in replicates}) != len(replicates):
        raise ValueError("assay/replicate identities must be unique and nonempty")
    present = np.asarray(protein_present)
    shape = (len(present), len(pool))
    if present.ndim != 1 or present.dtype.kind != "b":
        raise ValueError("protein footprint must be a boolean region vector")
    for rep in replicates:
        if rep.counts.shape != shape or rep.observation_counts.shape != (len(present),):
            raise ValueError("all replicate measurements must share the same regional axes")
        for values in (rep.counts, rep.observation_counts):
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError("regional counts must be finite nonnegative integers")
    geometric = present.copy()
    for rep in replicates:
        geometric &= rep.observation_counts > 0
    exposure = np.stack([rep.counts.sum(1) for rep in replicates])
    individual_qc = np.stack(
        [
            (rep.observation_counts >= qc["min_observations"]) & (row >= qc["min_pool_counts"])
            for rep, row in zip(replicates, exposure)
        ]
    )
    eligible = geometric & individual_qc.all(0)
    n = int(eligible.sum())
    reports, normalized, passes = [], {}, {}
    for rep, row in zip(replicates, exposure):
        selected = rep.counts[eligible]
        transformed = np.log1p(10000.0 * selected / row[eligible, None])
        normalized[rep.replicate] = transformed
        for gene in dict.fromkeys(primary + secondary):
            index = pool.index(gene)
            count = float(selected[:, index].sum())
            positive = int((selected[:, index] > 0).sum())
            fraction = positive / n if n else 0.0
            spread = (
                float(np.diff(np.quantile(transformed[:, index], [0.1, 0.9]))[0]) if n else None
            )
            failures = []
            if n < qc["min_regions"]:
                failures.append("insufficient_regions")
            if count < qc["min_gene_counts"]:
                failures.append("insufficient_gene_counts")
            if positive < qc["min_positive_regions"] or fraction < qc["min_positive_fraction"]:
                failures.append("insufficient_detection")
            if spread is None or spread < qc["min_log_quantile_range"]:
                failures.append("insufficient_dynamic_range")
            passes[rep.replicate, gene] = not failures
            reports.append(
                dict(
                    replicate=rep.replicate,
                    gene=gene,
                    role="primary" if gene in primary else "secondary",
                    eligible_regions=n,
                    total_gene_counts=count,
                    positive_regions=positive,
                    positive_fraction=fraction,
                    normalized_log_q90_minus_q10=spread,
                    eligible=not failures,
                    failure_reasons=";".join(failures),
                )
            )
    panels = {
        name: [gene for gene in genes if all(passes[r.replicate, gene] for r in replicates)]
        for name, genes in (("primary", primary), ("secondary", secondary))
    }
    return dict(
        geometric_support=geometric,
        region_qc_by_replicate=individual_qc,
        eligible_regions=eligible,
        exposure=exposure,
        normalized=normalized,
        gene_qc=reports,
        panels=panels,
        measurement_qualified=bool(n >= qc["min_regions"] and panels["primary"]),
    )


def _validate_config(config):
    if config["schema"] != "heir.regional_endpoint.v1":
        raise ValueError("unsupported regional endpoint schema")
    _axis(config["biopsy_ids"], "biopsies")
    pool = _axis(config["normalization_gene_ids"], "normalization genes")
    for key in ("primary_gene_ids", "secondary_gene_ids"):
        if not set(_axis(config[key], key)) <= set(pool):
            raise ValueError("outcome genes must be contained in the fixed pool")
    if set(config["primary_gene_ids"]) & set(config["secondary_gene_ids"]):
        raise ValueError("primary and secondary gene panels must be disjoint")
    if config["spatial_status"] != "conditional_author_registered_pixel":
        raise ValueError("this workflow does not certify independent spatial registration")
    size, fraction = config["region_size_pixels"], config["min_tissue_fraction"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("region_size_pixels must be a positive integer")
    if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("min_tissue_fraction must be finite in (0, 1]")
    if not set(_METADATA_SOURCES) <= set(config["expected_sources"]):
        raise ValueError("both cohort metadata files require frozen source identities")
    _validate_qc(config["qc"])


def freeze_regional_endpoint(root, config_path, output):
    """Publish a source-bound measurement endpoint; never choose models or score them."""
    from dataclasses import replace

    from . import cohort, reference, regions

    root, config_path, output = (
        Path(root).resolve(),
        Path(config_path).resolve(),
        Path(output).resolve(),
    )
    if output.is_relative_to(root):
        raise ValueError("endpoint output must be outside the input cohort root")
    config_bytes = config_path.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    _validate_config(config)
    cache = {}
    _verify_bindings(
        root,
        {key: config["expected_sources"][key] for key in _METADATA_SOURCES},
        cache=cache,
    )
    entries = discover_cohort(root)
    sources = []
    selected = {}
    for biopsy in config["biopsy_ids"]:
        entry = entries[biopsy]
        roles = {key: entry["role_sources"][key] for key in ("he", "protein")}
        for assay in ("merfish", "exseq"):
            matching = [
                s
                for s in entry["sources"]
                if s["modality"] == assay and "registered" in Path(s["working_path"]).parts
            ]
            if len(matching) != 1:
                raise ValueError(f"exactly one registered {assay} count source is required")
            roles[assay] = matching[0]
        for role, source in roles.items():
            if config["expected_sources"].get(source["working_path"]) != source["sha256"]:
                raise ValueError(
                    "consumed source is absent or differs from frozen source identities"
                )
            if role in ("merfish", "exseq") and (
                source["measurement_key"] != "obsm/counts"
                or source["feature_key"] != "uns/counts_var"
            ):
                raise ValueError("endpoint requires retained counts and their original gene axis")
            sources.append({**source, "biopsy_id": biopsy, "role": role})
        selected[biopsy] = roles
    _verify_bindings(root, config["expected_sources"], cache=cache)
    sidecar = config.get("spatial_evidence")
    if sidecar:
        sidecar_path = (config_path.parent / sidecar["path"]).resolve()
        if _sha(sidecar_path) != sidecar["sha256"]:
            raise ValueError("spatial evidence sidecar identity differs")
    code_paths = (__file__, cohort.__file__, reference.__file__, regions.__file__)
    code_hashes = {Path(path).name: _sha(path) for path in code_paths}
    output.mkdir(parents=True, exist_ok=False)
    _write(
        output / "protocol.json",
        dict(
            schema="heir.frozen_regional_endpoint_protocol.v1",
            config=config,
            config_sha256=config_sha,
            source_code_sha256=code_hashes,
            sources=sources,
            coordinate_units="registered_pixel",
            support=(
                "H&E tissue bins intersect all retained RNA replicate and CODEX centroid occupancy"
            ),
            footprint_limitation="empirical occupied bins; not certified capture boundaries",
            normalization="log1p(10000 * pooled_gene_count / fixed_pool_total); pool never shrinks",
            aggregation=(
                "score within each replicate; equal replicates within each assay/specimen, "
                "equal specimens within patient, equal patients; assays reported separately"
            ),
            spatial_status=config["spatial_status"],
            independent_registration_qualified=False,
            physical_calibration_qualified=False,
            independent_compartments_qualified=False,
            cross_assay_absolute_scale_qualified=False,
            within_cell_type_endpoint=False,
            prediction_payloads_read=False,
        ),
    )
    summaries, region_rows, gene_rows = [], [], []
    pool = config["normalization_gene_ids"]
    for biopsy, roles in selected.items():
        entry = entries[biopsy]
        grid = build_region_grid(
            _verified_source_path(root, roles["he"], verification_cache=cache),
            bin_size_pixels=config["region_size_pixels"],
            min_tissue_fraction=config["min_tissue_fraction"],
            compute_features=False,
        )
        with _read_h5ad(
            _verified_source_path(root, roles["protein"], verification_cache=cache)
        ) as adata:
            protein_map = point_to_grid(grid, adata.obs[["x", "y"]].to_numpy())
        protein_count = np.bincount(protein_map[protein_map >= 0], minlength=len(grid.region_ids))
        reps, axis_provenance = [], {}
        for assay in ("merfish", "exseq"):
            with _read_h5ad(
                _verified_source_path(root, roles[assay], verification_cache=cache)
            ) as adata:
                counts, genes = extract_raw_counts(adata)
                labels = _replicates(adata, roles[assay])
                axis_provenance[assay] = _axis_provenance(
                    _axis(adata.obs_names, f"{assay} observation IDs")
                )
                grouped = aggregate_retained_counts(
                    counts, genes, adata.obs[["x", "y"]].to_numpy(), labels, grid, pool
                )
                reps.extend(replace(r, replicate=f"{assay}:{r.replicate}") for r in grouped)
        result = qualify_measurements(
            reps,
            protein_count > 0,
            pool,
            config["primary_gene_ids"],
            config["secondary_gene_ids"],
            config["qc"],
        )
        arrays = dict(
            region_ids=np.asarray(grid.region_ids),
            gene_ids=np.asarray(pool),
            coordinates=grid.coordinates,
            tissue_fraction=grid.tissue_fraction,
            protein_observation_counts=protein_count,
            geometric_support=result["geometric_support"],
            eligible_regions=result["eligible_regions"],
        )
        for i, (rep, prefix) in enumerate(zip(reps, _array_prefixes(reps))):
            arrays.update(
                {
                    prefix + "__counts": rep.counts,
                    prefix + "__observation_counts": rep.observation_counts,
                    prefix + "__source_rows": rep.source_rows,
                    prefix + "__region_assignments": rep.region_assignments,
                    prefix + "__region_qc": result["region_qc_by_replicate"][i],
                    prefix + "__normalized_eligible": result["normalized"][rep.replicate],
                }
            )
            for j, region in enumerate(grid.region_ids):
                region_rows.append(
                    dict(
                        biopsy_id=biopsy,
                        replicate=rep.replicate,
                        region_id=region,
                        retained_observations=int(rep.observation_counts[j]),
                        pool_counts=float(result["exposure"][i, j]),
                        codex_observations=int(protein_count[j]),
                        common_geometry=bool(result["geometric_support"][j]),
                        replicate_count_qc=bool(result["region_qc_by_replicate"][i, j]),
                        common_eligible=bool(result["eligible_regions"][j]),
                    )
                )
        np.savez_compressed(output / f"{biopsy}.npz", **arrays)
        gene_rows.extend(dict(biopsy_id=biopsy, **row) for row in result["gene_qc"])
        summaries.append(
            dict(
                biopsy_id=biopsy,
                patient_id=entry["patient_id"],
                specimen_id=entry["specimen_id"],
                source_observation_axes=axis_provenance,
                grid_settings=grid.settings,
                n_he_regions=len(grid.region_ids),
                n_codex_occupied_regions=int((protein_count > 0).sum()),
                n_common_geometry_regions=int(result["geometric_support"].sum()),
                n_common_eligible_regions=int(result["eligible_regions"].sum()),
                replicates=[r.replicate for r in reps],
                primary_gene_ids=result["panels"]["primary"],
                secondary_gene_ids=result["panels"]["secondary"],
                measurement_qualified=result["measurement_qualified"],
                spatial_status=config["spatial_status"],
                fully_qualified_biological_endpoint=False,
            )
        )
    _table(output / "region_qc.tsv", region_rows)
    _table(output / "gene_qc.tsv", gene_rows)
    # Rehash independently of the initial stat cache before publishing completion.
    _verify_bindings(root, config["expected_sources"])
    if _sha(config_path) != config_sha:
        raise ValueError("endpoint config changed during extraction")
    if sidecar and _sha(sidecar_path) != sidecar["sha256"]:
        raise ValueError("spatial evidence sidecar changed during extraction")
    if any(_sha(path) != code_hashes[Path(path).name] for path in code_paths):
        raise ValueError("endpoint source code changed during extraction")
    summary = dict(
        schema="heir.frozen_regional_measurement.v1",
        biopsies=summaries,
        measurement_qualified=all(s["measurement_qualified"] for s in summaries),
        biological_spatial_qualification=False,
        prediction_payloads_read=False,
        model_comparisons=0,
        new_fits=0,
    )
    _write(output / "summary.json", summary)
    _write(
        output / "artifact_sha256.json",
        {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    return summary
