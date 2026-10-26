"""Raw-cell regional proteins, training-only scaling and separate joint inputs.

Protein features describe measured antibody channels. Their presence never
implies quantitative equality with RNA or a validated section correspondence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .regions import point_to_grid
from .schema import JointInputs, RegionalProteinFeatures, RNAReference, identifiers

_FIVE = (("CD4", "CD4"), ("CD8", "CD8A"), ("CD45", "PTPRC"), ("CD31", "PECAM1"), ("Ki67", "MKI67"))
# These published aliases do not establish a complete biological target mapping.
_UNRESOLVED_RNA = {
    "CD15",
    "CD25",
    "CD45RA",
    "CD57",
    "collagen IV",
    "cytokeratin",
    "HLA-ABC",
    "HLA-DR",
    "TCRyd",
}
_PROXY_MAPPING_NOTES = {
    "CD15": "HI98 glycan epitope is not direct FUT4 abundance",
    "CD57": "HCD57 carbohydrate epitope is not direct B3GAT1 abundance",
    "collagen IV": "polyclonal collagen IV family; chain-specific abundance unresolved",
    "HLA-DR": "Tu36 HLA-DR complex; complete chain mapping unresolved",
    "TCRyd": "B1 gamma/delta TCR complex; complete chain mapping unresolved",
}
# Manufacturer mappings qualify only a conservative overlap screen, not an
# abundance equality. Keep the retained measurement aliases untouched.
_DIAGNOSTIC_MAPPINGS = {
    "cytokeratin": (
        ["KRT4", "KRT5", "KRT6A", "KRT6B", "KRT6C", "KRT8", "KRT10", "KRT13", "KRT18"],
        "https://www.sigmaaldrich.com/US/en/product/sigma/c2931",
        "C-11 CK6 family subtype unresolved; "
        "KRT19 retained alias conflicts with documented targets",
    ),
    "CD25": (
        ["IL2RA"],
        "https://bioxcell.com/invivomab-anti-human-cd25-il-2r-alpha-be0014",
        "7G7B6 target differs from retained ISG20 alias",
    ),
    "CD3": (
        ["CD3E"],
        "https://www.bdbiosciences.com/en-us/products/reagents/flow-cytometry-reagents/research-reagents/single-color-antibodies-ruo/purified-mouse-anti-human-cd3.550368",
        "UCHT1 target differs from retained CD3D alias",
    ),
    "HLA-ABC": (
        ["HLA-A", "HLA-B", "HLA-C"],
        "https://www.bdbiosciences.com/content/bdb/paths/generate-tds-document.cn.565292.pdf",
        "EMR8-5 multiple class I targets",
    ),
    "CD66": (
        ["CEACAM1", "CEACAM3", "CEACAM5", "CEACAM6"],
        "https://www.bdbiosciences.com/content/dam/bdb/product_assets/product_pdf/singcolorpureantibody/pe/pdf_0/551480.pdf",
        "B1.1 multigene overlap screen",
    ),
    "CD45RA": (
        ["PTPRC"],
        "https://stjohnslabs.com/content/PDF25/STJA0000349_datasheet.pdf",
        "HI100 isoform-specific; PTPRC is overlap screen only",
    ),
}
_REPLICATE_FIELDS = (
    "replicate_mean",
    "replicate_q90",
    "replicate_observed",
    "replicate_valid_counts",
    "replicate_disagreement",
)


def qualify_marker_panel(
    panel_path, measured_gene_ids, marker_panel="five", *, declared_marker_ids=None
):
    """Select identifiable original antibody channels, separate from RNA aliases.

    An explicit panel is a list of dictionaries with ``marker_id``,
    ``measurement_id``, ``biological_feature_qualified=True``, ``evidence``, and
    optional ``gene_ids`` plus ``gene_mapping_evidence``. Explicit RNA mappings
    require their own evidence; they are never used as an RNA/protein equality.
    """
    path = Path(panel_path)
    with path.open(newline="") as handle:
        original = list(csv.DictReader(handle, delimiter="\t"))
    if len({row["Target"] for row in original}) != len(original):
        raise ValueError("original protein panel has duplicate antibody targets")
    by_marker = {row["Target"]: row for row in original}
    measured = tuple(measured_gene_ids)
    identifiers(measured, "measured protein axis")
    if declared_marker_ids is not None:
        declared_marker_ids = tuple(declared_marker_ids)
        identifiers(declared_marker_ids, "declared protein feature axis")
    selected, excluded = [], []
    if isinstance(marker_panel, str):
        if marker_panel not in {"five", "expanded"}:
            raise ValueError("marker_panel must be five, expanded or explicitly qualified entries")
        candidates = [m for m, _ in _FIVE] if marker_panel == "five" else list(by_marker)
        if declared_marker_ids is not None:
            if not set(declared_marker_ids) <= set(candidates):
                raise ValueError("declared protein markers are outside the qualified panel")
            candidates = list(declared_marker_ids)
        for marker in candidates:
            if marker not in by_marker:
                raise ValueError(f"protein marker absent from original panel: {marker}")
            row = by_marker[marker]
            alias = row["Matching gene"].strip()
            if marker_panel == "five" and alias != dict(_FIVE)[marker]:
                raise ValueError("initial protein marker mapping differs from original Table S5")
            # Table S5 documents the biological antibody identity and clone;
            # arbitrary retained channels outside this table are never admitted.
            if not alias or (marker_panel == "expanded" and not row.get("Clone", "").strip()):
                excluded.append({"marker_id": marker, "reason": "unresolved_antibody_identity"})
                continue
            if alias not in measured and declared_marker_ids is None:
                if marker_panel == "five":
                    raise ValueError(f"protein marker measurement absent: {marker}/{alias}")
                excluded.append({"marker_id": marker, "reason": "not_in_retained_measurement_axis"})
                continue
            unresolved = marker in _UNRESOLVED_RNA or "|" in alias
            selected.append(
                {
                    "marker_id": marker,
                    "measurement_id": alias,
                    "measurement_available": alias in measured,
                    "biological_feature_qualified": True,
                    "evidence": "original_Table_S5_antibody_identity_and_clone",
                    "clone": row.get("Clone"),
                    "source_excel_row": row.get("source_excel_row"),
                    "published_gene_alias": alias,
                    "gene_ids": [] if unresolved else [alias],
                    "gene_mapping_status": "unresolved_biological_mapping"
                    if unresolved
                    else "published_single_gene_alias",
                    "gene_mapping_evidence": None if unresolved else "original_Table_S5",
                    "possible_direct_gene_ids": [alias] if unresolved else [],
                    "possible_gene_mapping_exhaustive": not unresolved,
                }
            )
            if marker in _PROXY_MAPPING_NOTES:
                selected[-1]["gene_mapping_note"] = _PROXY_MAPPING_NOTES[marker]
            if marker == "collagen IV":
                selected[-1]["possible_direct_gene_ids"] = [f"COL4A{i}" for i in range(1, 7)]
            if marker in _DIAGNOSTIC_MAPPINGS:
                genes, evidence, note = _DIAGNOSTIC_MAPPINGS[marker]
                selected[-1].update(
                    gene_ids=genes,
                    gene_mapping_status="manufacturer_conservative_overlap_screen",
                    gene_mapping_evidence=evidence,
                    gene_mapping_note=note,
                    possible_direct_gene_ids=[alias] if alias not in genes else [],
                )
    else:
        for entry in marker_panel:
            record = dict(entry)
            marker, alias = record.get("marker_id"), record.get("measurement_id")
            if marker not in by_marker or alias != by_marker[marker]["Matching gene"].strip():
                raise ValueError(
                    "qualified protein panel must identify an original measured antibody channel"
                )
            if alias not in measured and declared_marker_ids is None:
                raise ValueError(f"qualified protein measurement absent: {alias}")
            if record.get("biological_feature_qualified") is not True or not record.get("evidence"):
                raise ValueError(
                    "protein feature qualification requires explicit supporting evidence"
                )
            genes = tuple(record.get("gene_ids", ()))
            if genes:
                identifiers(genes, "protein diagnostic gene mapping")
                if not record.get("gene_mapping_evidence"):
                    raise ValueError(
                        "explicit diagnostic gene mappings require supporting evidence"
                    )
            record.update(
                gene_ids=list(genes),
                measurement_available=alias in measured,
                published_gene_alias=alias,
                gene_mapping_status="explicitly_qualified"
                if genes
                else "unresolved_biological_mapping",
            )
            selected.append(record)
        if declared_marker_ids is not None:
            if set(declared_marker_ids) != {record["marker_id"] for record in selected}:
                raise ValueError("declared protein marker axis differs from explicit qualification")
            by_id = {record["marker_id"]: record for record in selected}
            selected = [by_id[marker] for marker in declared_marker_ids]
    identifiers(tuple(entry["marker_id"] for entry in selected), "qualified protein markers")
    if len({entry["measurement_id"] for entry in selected}) != len(selected):
        raise ValueError("retained measurement axis cannot distinguish duplicate antibody aliases")
    return selected, {
        "selection": marker_panel
        if isinstance(marker_panel, str)
        else "explicit_qualified_extension",
        "source_path": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "qualified_features": selected,
        "excluded_channels": excluded,
        "unmatched_retained_channels": sorted(
            set(measured) - {row["Matching gene"].strip() for row in original}
        ),
        "rna_mapping_use": "diagnostic_overlap_only_never_quantitative_equality",
    }


def _consensus(mean, q90, observed, counts):
    """Equal observed-replicate mean statistics; counts sum, disagreement is SD."""
    number = observed.sum(axis=0)
    denominator = np.maximum(number, 1)
    mean = np.where(observed, mean, 0)
    quantile = np.where(observed, q90, 0).sum(axis=0) / denominator
    consensus = mean.sum(axis=0) / denominator
    variance = np.where(observed, (mean - consensus[None]) ** 2, 0).sum(axis=0) / denominator
    return (
        consensus,
        quantile,
        number > 0,
        np.where(observed, counts, 0).sum(axis=0),
        np.sqrt(variance),
    )


def aggregate_protein_features(
    grid,
    values,
    coordinates,
    replicate_labels,
    marker_ids,
    *,
    cofactor=100.0,
    metadata=None,
):
    """Transform each valid raw cell then pool directly in the requested grid.

    Finite intensities are available, including signed measurements and zero. Every
    marker has its own valid-cell count. Means and linear-interpolated q90s are
    computed separately in each replicate; the consensus is an equal-replicate
    average of those summaries (its q90 is not a pooled-cell quantile).
    """
    if isinstance(cofactor, bool) or not np.isfinite(cofactor) or cofactor <= 0:
        raise ValueError("protein asinh cofactor must be finite and positive")
    marker_ids = tuple(marker_ids)
    identifiers(marker_ids, "protein markers")
    values = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(coordinates)
    labels = np.asarray(replicate_labels)
    if values.ndim != 2 or values.shape[1] != len(marker_ids):
        raise ValueError("raw protein values must follow source cells by named markers")
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError("protein coordinates must be finite and follow source cells")
    if labels.shape != (len(values),) or not len(labels) or any(not str(v).strip() for v in labels):
        raise ValueError("protein replicate labels must identify every source cell")
    labels = labels.astype(str)
    replicates = tuple(sorted(set(labels)))
    shape = (len(replicates), len(grid.region_ids), len(marker_ids))
    means, quantiles = np.zeros(shape), np.zeros(shape)
    counts = np.zeros(shape, dtype=np.int64)
    mapped = point_to_grid(grid, coordinates)
    valid = np.isfinite(values)
    transformed = np.arcsinh(np.where(valid, values, 0) / cofactor)
    for r, replicate in enumerate(replicates):
        selected = np.flatnonzero((labels == replicate) & (mapped >= 0))
        selected = selected[np.argsort(mapped[selected], kind="stable")]
        boundaries = np.r_[0, np.flatnonzero(np.diff(mapped[selected])) + 1, len(selected)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop <= start:
                continue
            rows = selected[start:stop]
            region = mapped[rows[0]]
            for m in range(len(marker_ids)):
                x = transformed[rows[valid[rows, m]], m]
                if len(x):
                    counts[r, region, m] = len(x)
                    means[r, region, m] = x.mean()
                    quantiles[r, region, m] = np.quantile(x, 0.9, method="linear")
    observed = counts > 0
    mean, q90, available, total, disagreement = _consensus(means, quantiles, observed, counts)
    record = {
        **(metadata or {}),
        "transform": "asinh(raw_cell_intensity/cofactor)",
        "cofactor": float(cofactor),
        "aggregation": "raw_cells_directly_to_requested_grid",
        "replicate_consensus": "equal_weight_observed_replicate_means_and_q90s",
        "replicate_disagreement": "population_sd_of_observed_replicate_means",
        "valid_cell_rule": "finite_intensity_per_marker_including_negative_and_measured_zero",
        "coverage_not_registration_confidence": True,
        "source_cells": len(values),
        "excluded_source_cells": int((mapped < 0).sum()),
    }
    return RegionalProteinFeatures(
        region_ids=grid.region_ids,
        marker_ids=marker_ids,
        mean=mean,
        q90=q90,
        observed=available,
        valid_counts=total,
        replicate_labels=replicates,
        replicate_mean=means,
        replicate_q90=quantiles,
        replicate_observed=observed,
        replicate_valid_counts=counts,
        replicate_disagreement=disagreement,
        metadata=record,
    )


@dataclass(frozen=True)
class ProteinFeatureScaler:
    """Observed training-region median and IQR, separately for mean and q90."""

    marker_ids: tuple[str, ...]
    mean_center: tuple[float, ...]
    mean_scale: tuple[float, ...]
    q90_center: tuple[float, ...]
    q90_scale: tuple[float, ...]
    training_patient_ids: tuple[str, ...]
    observed_training_regions: tuple[int, ...]

    def __post_init__(self):
        identifiers(self.marker_ids, "scaler marker IDs")
        identifiers(self.training_patient_ids, "scaler training patient IDs")
        n = len(self.marker_ids)
        for name in ("mean_center", "mean_scale", "q90_center", "q90_scale"):
            values = np.asarray(getattr(self, name))
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError("protein scaler parameters must be finite on its marker axis")
            if name.endswith("scale") and np.any(values <= 0):
                raise ValueError("protein scaler scales must be positive")
        counts = np.asarray(self.observed_training_regions)
        if counts.shape != (n,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("protein scaler training counts must be nonnegative integers")

    @classmethod
    def fit(cls, samples: Sequence[JointInputs]):
        samples = tuple(samples)
        if not samples or any(not isinstance(s, JointInputs) for s in samples):
            raise ValueError("protein scaler requires explicitly supplied training JointInputs")
        available = [s for s in samples if s.protein_features is not None]
        if not available:
            raise ValueError("protein scaler needs training protein features")
        markers = tuple(sorted(available[0].protein_features.marker_ids))
        if any(set(s.protein_features.marker_ids) != set(markers) for s in available):
            raise ValueError("training protein marker sets must agree")
        statistics, counts = {}, []
        for name in ("mean", "q90"):
            centers, scales = [], []
            for marker in markers:
                entries = []
                for sample in available:
                    features = sample.protein_features
                    column = features.marker_ids.index(marker)
                    entries.append(
                        np.asarray(getattr(features, name))[features.observed[:, column], column]
                    )
                values = np.concatenate(entries)
                centers.append(float(np.median(values)) if len(values) else 0.0)
                scales.append(
                    max(float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 1e-6)
                    if len(values)
                    else 1.0
                )
                if name == "mean":
                    counts.append(len(values))
            statistics[f"{name}_center"] = tuple(centers)
            statistics[f"{name}_scale"] = tuple(scales)
        return cls(
            marker_ids=markers,
            **statistics,
            training_patient_ids=tuple(sorted({s.patient_id for s in samples})),
            observed_training_regions=tuple(counts),
        )

    def transform(self, features: RegionalProteinFeatures):
        if set(features.marker_ids) != set(self.marker_ids):
            raise ValueError("protein feature marker set differs from fitted scaler")
        columns = [self.marker_ids.index(m) for m in features.marker_ids]
        updates = {}
        for name in ("mean", "q90"):
            center = np.asarray(getattr(self, f"{name}_center"))[columns]
            scale = np.asarray(getattr(self, f"{name}_scale"))[columns]
            safe = np.where(features.observed, getattr(features, name), center)
            updates[name] = np.where(features.observed, (safe - center) / scale, 0)
            rep = getattr(features, f"replicate_{name}")
            if rep is not None:
                safe = np.where(features.replicate_observed, rep, center)
                updates[f"replicate_{name}"] = np.where(
                    features.replicate_observed, (safe - center) / scale, 0
                )
        updates["valid_counts"] = np.where(features.observed, features.valid_counts, 0)
        if features.replicate_valid_counts is not None:
            updates["replicate_valid_counts"] = np.where(
                features.replicate_observed, features.replicate_valid_counts, 0
            )
        if features.replicate_disagreement is not None:
            scale = np.asarray(self.mean_scale)[columns]
            updates["replicate_disagreement"] = (
                np.where(features.observed, features.replicate_disagreement, 0) / scale
            )
        updates["metadata"] = {**features.metadata, "robust_scaling": self.to_dict()}
        return replace(features, **updates)

    def to_dict(self):
        return {
            "schema": "heir.protein_feature_scaler.v1",
            **asdict(self),
            "fit_scope": "supplied_training_patients_only",
            "scale_floor": 1e-6,
            "weighting": "equal_observed_regions_pooled_across_training_patients",
        }

    @classmethod
    def from_dict(cls, data):
        if (
            data.get("schema") != "heir.protein_feature_scaler.v1"
            or data.get("fit_scope") != "supplied_training_patients_only"
        ):
            raise ValueError("unsupported protein scaler metadata")
        return cls(**{field.name: tuple(data[field.name]) for field in fields(cls)})


def permute_protein_features(features: RegionalProteinFeatures, seed=0):
    """Fixed-model intervention: shuffle full marker/count rows within availability.

    Source replicate identities and masks stay fixed. All protein-valued paths,
    including source counts and replicate diagnostics, are updated consistently.
    This is a prediction intervention, not a refitted shuffled-protein model.
    """
    if not features.replicate_labels:
        mean, q90 = features.mean[None], features.q90[None]
        observed, counts = features.observed[None], features.valid_counts[None]
        labels = ("consensus_only",)
    else:
        mean, q90 = features.replicate_mean, features.replicate_q90
        observed, counts = features.replicate_observed, features.replicate_valid_counts
        labels = features.replicate_labels
    rng = np.random.default_rng(seed)
    output_mean, output_q90, output_counts = mean.copy(), q90.copy(), counts.copy()
    permutations, movable = {}, 0
    for r, label in enumerate(labels):
        order = np.arange(len(features.region_ids))
        patterns = {}
        for row, pattern in enumerate(observed[r]):
            patterns.setdefault(tuple(pattern), []).append(row)
        for pattern, rows in sorted(patterns.items()):
            if any(pattern) and len(rows) > 1:
                order[rows] = rng.permutation(rows)
                movable += len(rows)
        output_mean[r], output_q90[r], output_counts[r] = (
            mean[r, order],
            q90[r, order],
            counts[r, order],
        )
        permutations[label] = order.tolist()
    consensus = _consensus(output_mean, output_q90, observed, output_counts)
    details = {
        "seed": int(seed),
        "scope": "within_replicate_and_marker_availability_pattern",
        "kind": "fixed_model_prediction_intervention",
        "movable_replicate_region_rows": movable,
        "permutations": permutations,
        "counts_permuted_with_intensities": True,
    }
    updates = dict(
        zip(("mean", "q90", "observed", "valid_counts", "replicate_disagreement"), consensus)
    )
    if features.replicate_labels:
        updates.update(
            replicate_mean=output_mean,
            replicate_q90=output_q90,
            replicate_valid_counts=output_counts,
        )
    updates["metadata"] = {**features.metadata, "permutation": details}
    return replace(features, **updates), details


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz(directory, name, arrays, payloads):
    path = directory / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    payloads[name] = {"file": path.name, "sha256": _sha(path)}


def save_joint_inputs(inputs: JointInputs, directory, provenance):
    """Create a target-free bundle with independent, hash-bound modality files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payloads = {}
    _write_npz(directory, "geometry", {"coordinates": inputs.coordinates}, payloads)
    if inputs.image_features is not None:
        _write_npz(directory, "image", {"image_features": inputs.image_features}, payloads)
    if inputs.reference is not None:
        _write_npz(
            directory,
            "reference",
            {
                field.name: np.asarray(getattr(inputs.reference, field.name))
                for field in fields(RNAReference)
            },
            payloads,
        )
    protein_metadata = None
    if inputs.protein_features is not None:
        features = inputs.protein_features
        arrays = {
            name: np.asarray(getattr(features, name))
            for name in ("mean", "q90", "observed", "valid_counts", *_REPLICATE_FIELDS)
            if getattr(features, name) is not None
        }
        _write_npz(directory, "protein", arrays, payloads)
        protein_metadata = {
            "marker_ids": list(features.marker_ids),
            "replicate_labels": list(features.replicate_labels),
            "metadata": features.metadata,
        }
    record = {
        "schema": "heir.joint_inputs.v1",
        "image_encoder_manifest_sha256": inputs.image_encoder_manifest_sha256,
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "cell_ids": list(inputs.cell_ids),
        "prediction_unit": inputs.prediction_unit,
        "coordinate_units": inputs.coordinate_units,
        "payloads": payloads,
        "protein": protein_metadata,
        "provenance": provenance,
    }
    (directory / "metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


def load_joint_inputs(directory, *, modalities=("H", "R", "P")):
    """Load only requested modality files; an absent requested protein is explicit."""
    directory = Path(directory)
    modalities = tuple(modalities)
    if (
        not modalities
        or len(set(modalities)) != len(modalities)
        or not set(modalities) <= {"H", "R", "P"}
    ):
        raise ValueError("joint modalities must be unique H/R/P labels")
    record = json.loads((directory / "metadata.json").read_text())
    if record.get("schema") != "heir.joint_inputs.v1":
        raise ValueError("unsupported joint input schema")

    def read(name):
        entry = record["payloads"].get(name)
        if entry is None:
            raise ValueError(f"joint input bundle has no requested {name} payload")
        path = directory / entry["file"]
        if path.name != entry["file"] or not path.is_file() or _sha(path) != entry["sha256"]:
            raise ValueError(f"joint {name} payload missing or checksum differs")
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    image, reference, protein, encoder_identity = None, None, None, None
    if "H" in modalities:
        image = read("image")["image_features"]
        provenance_identity = (record["provenance"].get("image_features") or {}).get(
            "encoder_manifest_sha256"
        )
        # Historical v1 preparations already bind this identity in provenance.
        encoder_identity = record.get("image_encoder_manifest_sha256", provenance_identity)
        if provenance_identity is not None and encoder_identity != provenance_identity:
            raise ValueError("prepared image encoder identity disagrees with cache provenance")
        cache_identity = (
            (record["provenance"].get("image_features") or {}).get("metadata") or {}
        ).get("encoder_manifest_sha256")
        if cache_identity is not None and cache_identity != encoder_identity:
            raise ValueError(
                "prepared image encoder identity disagrees with embedded cache identity"
            )
    if "R" in modalities:
        values = read("reference")
        for name in ("state_ids", "gene_ids", "state_types", "embedding_gene_ids"):
            values[name] = tuple(map(str, values[name]))
        reference = RNAReference(**values)
    if "P" in modalities and "protein" in record["payloads"]:
        values = read("protein")
        info = record["protein"]
        protein = RegionalProteinFeatures(
            region_ids=tuple(record["cell_ids"]),
            marker_ids=tuple(info["marker_ids"]),
            replicate_labels=tuple(info["replicate_labels"]),
            metadata=info["metadata"],
            **values,
        )
    inputs = JointInputs(
        patient_id=record["patient_id"],
        specimen_id=record["specimen_id"],
        cell_ids=tuple(record["cell_ids"]),
        coordinates_um=read("geometry")["coordinates"],
        image_features=image,
        image_encoder_manifest_sha256=encoder_identity,
        reference=reference,
        protein_features=protein,
        prediction_unit=record["prediction_unit"],
        coordinate_units=record["coordinate_units"],
    )
    return inputs, record["provenance"]
