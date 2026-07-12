"""Command-line entry points for auditable HEIR artifact stages."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import tarfile
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from . import __version__
from .config import LossWeightConfig, OptimizationConfig, RefinementConfig
from .data import (
    BARCODE_SUFFIX_POLICIES,
    LOCKED_TARGET_ROLES,
    HistologyBag,
    PrototypeSet,
    RNAReference,
    SpatialTruthArtifact,
    build_spatial_truth,
    default_manifest_path,
    filter_nucleus_csv_to_visium,
    h5ad_filters,
    load_h5ad_reference,
    load_manifest,
    read_spot_diameter,
    read_tissue_positions,
    read_visium_counts,
    verify_checksums,
)
from .evaluation import cell_type_metrics, composition_metrics, expression_metrics
from .expression import EXPRESSION_MAX, EXPRESSION_SPACE_ID, EXPRESSION_TARGET_SUM
from .image import (
    PixelMicronTransform,
    build_spatial_graph,
    canonical_nucleus_ids,
    extract_nucleus_pathology_features,
    load_feature_bundle,
    load_imagenet_resnet50_encoder,
    load_nuclei,
    load_omiclip_visual_encoder,
    open_slide,
    save_pathology_feature_npz,
    with_peak_memory,
)
from .inference import PredictionBundle, predict_cells
from .models.heir import HEIRConfig, HEIRModel
from .models.rna import RNAVAE
from .prior import (
    GenePrograms,
    RNAResidualGeometry,
    SCGPTTeacherArtifact,
    fit_rna_residual_geometry,
)
from .prior.prototypes import build_sample_prototypes
from .refinement import IterativeRefiner
from .segmentation import (
    export_spaceranger_artifacts,
    read_spaceranger_geojson,
    run_spaceranger_segment,
)
from .training import HEIRTrainer, HEIRTrainingBatch, TrainingStage
from .uncertainty import MahalanobisOOD
from .utils import atomic_json_dump, resolve_device, set_seed


def _json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _path_exists(value: str) -> bool:
    if not value:
        return True
    if "://" in value:
        return True
    base = value.split("::", 1)[0]
    return Path(base).exists()


def command_doctor(args: argparse.Namespace) -> int:
    manifests = [Path(value).expanduser().resolve() for value in args.manifest]
    report: Dict[str, object] = {
        "heir": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "numpy",
                "scipy",
                "scikit-learn",
                "anndata",
                "scanpy",
                "scvi-tools",
            )
        },
        "manifests": {},
    }
    failed = False
    for source in manifests:
        try:
            manifest = load_manifest(source, require_folds=True)
            missing = []
            for record in manifest:
                for value in (
                    record.he_file,
                    record.count_matrix_file,
                    record.spatial_coordinate_file,
                    record.spatial_count_matrix_file,
                ):
                    if value and not _path_exists(value):
                        missing.append(value)
            report["manifests"][str(source)] = {  # type: ignore[index]
                "valid": True,
                "records": len(manifest),
                "included": len(manifest.included_records),
                "donors": len(manifest.donors),
                "missing_files": sorted(set(missing)),
            }
            failed = failed or bool(missing and args.require_files)
        except Exception as error:
            report["manifests"][str(source)] = {"valid": False, "error": str(error)}  # type: ignore[index]
            failed = True
    _json(report)
    return 1 if failed else 0


def command_validate_manifest(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest, require_folds=args.require_folds)
    if args.checksums:
        verify_checksums(manifest)
    missing = []
    if args.require_files:
        for record in manifest:
            for value in (
                record.he_file,
                record.count_matrix_file,
                record.spatial_count_matrix_file,
            ):
                if value and not _path_exists(value):
                    missing.append(value)
        if missing:
            raise FileNotFoundError(
                "manifest files are absent: %s" % ", ".join(sorted(set(missing)))
            )
    _json({"valid": True, "records": len(manifest), "donors": manifest.donors})
    return 0


def _record(manifest_path: str, section_id: str):
    manifest = load_manifest(manifest_path, require_folds=True)
    matches = [record for record in manifest if record.section_id == section_id]
    if len(matches) != 1:
        raise ValueError("section_id %s matched %d manifest rows" % (section_id, len(matches)))
    return matches[0]


def _gene_list(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        values = [
            line.strip().split("\t")[0]
            for line in handle
            if line.strip() and not line.startswith("#")
        ]
    if not values:
        raise ValueError("gene panel is empty")
    return values


def _sha256(path: str) -> str:
    raw_path, separator, member_name = path.partition("::")
    source = Path(raw_path).expanduser().resolve()
    digest = hashlib.sha256()
    if not separator:
        handle = source.open("rb")
        close_stack = [handle]
    else:
        archive = tarfile.open(source, "r:*")
        matches = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (member.name == member_name or Path(member.name).name == Path(member_name).name)
        ]
        if len(matches) != 1:
            archive.close()
            raise ValueError(
                "archive member %s matched %d files in %s" % (member_name, len(matches), source)
            )
        extracted = archive.extractfile(matches[0])
        if extracted is None:
            archive.close()
            raise ValueError("could not open archive member %s" % member_name)
        if member_name.lower().endswith(".gz"):
            handle = gzip.GzipFile(fileobj=extracted, mode="rb")
            close_stack = [handle, extracted, archive]
        else:
            handle = extracted
            close_stack = [handle, archive]
    try:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    finally:
        for resource in close_stack:
            resource.close()
    return digest.hexdigest()


def _artifact_sha256(path: str) -> str:
    """Hash a file/archive member or a 10x matrix directory deterministically."""

    if "::" in path:
        return _sha256(path)
    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        return _sha256(path)
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ValueError("artifact directory contains no files: %s" % source)
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(str(item))))
    return digest.hexdigest()


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".pt.tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_checkpoint(path: str) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - only torch versions below the project floor
        raise RuntimeError("this command requires torch.load(..., weights_only=True)") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return checkpoint


_LATENT_TRANSFORM_CONTRACT = "heir.truncated_svd_transform"
_LATENT_TRANSFORM_VERSION = 2


def _save_latent_transform(
    path: str,
    gene_ids: np.ndarray,
    components: np.ndarray,
    target_sum: float,
    provenance: Mapping[str, object],
) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".npz.tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                __contract__=np.asarray(_LATENT_TRANSFORM_CONTRACT, dtype=np.dtype("U")),
                __version__=np.asarray(_LATENT_TRANSFORM_VERSION, dtype=np.int64),
                gene_ids=np.asarray(gene_ids, dtype=np.dtype("U")),
                components=np.asarray(components, dtype=np.float32),
                target_sum=np.asarray(target_sum, dtype=np.float64),
                training_donors=np.asarray(provenance["training_donors"], dtype=np.dtype("U")),
                source_reference_sha256=np.asarray(
                    provenance["source_reference_sha256"], dtype=np.dtype("U")
                ),
                manifest_sha256=np.asarray(provenance["manifest_sha256"], dtype=np.dtype("U")),
                analysis_role=np.asarray(provenance["analysis_role"], dtype=np.dtype("U")),
                cohort_id=np.asarray(provenance["cohort_id"], dtype=np.dtype("U")),
                section_id=np.asarray(provenance["section_id"], dtype=np.dtype("U")),
                outer_fold=np.asarray(provenance["outer_fold"], dtype=np.dtype("U")),
                inner_fold=np.asarray(provenance["inner_fold"], dtype=np.dtype("U")),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_latent_transform(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"__contract__", "__version__", "gene_ids", "components", "target_sum"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("latent-transform artifact is missing: %s" % ", ".join(missing))
        if str(np.asarray(archive["__contract__"]).item()) != _LATENT_TRANSFORM_CONTRACT:
            raise ValueError("artifact is not a HEIR latent transform")
        version = int(np.asarray(archive["__version__"]).item())
        if version not in {1, _LATENT_TRANSFORM_VERSION}:
            raise ValueError("unsupported HEIR latent-transform version")
        genes = np.asarray(archive["gene_ids"], dtype=np.dtype("U")).copy()
        components = np.asarray(archive["components"], dtype=np.float32).copy()
        target_sum = float(np.asarray(archive["target_sum"]).item())
        if version == 1:
            provenance: Dict[str, object] = {"training_donors": ()}
        else:
            provenance_fields = {
                "training_donors",
                "source_reference_sha256",
                "manifest_sha256",
                "analysis_role",
                "cohort_id",
                "section_id",
                "outer_fold",
                "inner_fold",
            }
            missing_provenance = sorted(provenance_fields - set(archive.files))
            if missing_provenance:
                raise ValueError(
                    "latent transform lacks provenance: %s" % ", ".join(missing_provenance)
                )
            provenance = {
                "training_donors": tuple(
                    str(value) for value in archive["training_donors"].tolist()
                ),
                **{
                    name: str(np.asarray(archive[name]).item())
                    for name in provenance_fields - {"training_donors"}
                },
            }
    if genes.ndim != 1 or components.ndim != 2 or components.shape[1] != len(genes):
        raise ValueError("latent-transform genes and components are misaligned")
    if len(set(genes.tolist())) != len(genes) or not np.isfinite(components).all():
        raise ValueError("latent-transform gene IDs/components are invalid")
    if target_sum <= 0:
        raise ValueError("latent-transform target_sum must be positive")
    return genes, components, target_sum, provenance


def command_prepare_reference(args: argparse.Namespace) -> int:
    record = _record(args.manifest, args.section_id)
    input_path = args.input or record.count_matrix_file
    if (
        args.input
        and Path(args.input).expanduser().resolve()
        != Path(record.count_matrix_file).expanduser().resolve()
    ):
        if not args.conversion_provenance:
            raise ValueError(
                "--input derivatives require --conversion-provenance from export_seurat.R"
            )
        with Path(args.conversion_provenance).expanduser().open("r", encoding="utf-8") as handle:
            conversion = json.load(handle)
        required_conversion = {
            "source_path",
            "source_sha256",
            "derivative_path",
            "derivative_sha256",
        }
        if not isinstance(conversion, dict) or not required_conversion.issubset(conversion):
            raise ValueError("conversion provenance sidecar is incomplete")
        if (
            Path(str(conversion["source_path"])).expanduser().resolve()
            != Path(record.count_matrix_file).expanduser().resolve()
        ):
            raise ValueError("converted H5AD sidecar names a different manifest RNA source")
        if (
            Path(str(conversion["derivative_path"])).expanduser().resolve()
            != Path(input_path).expanduser().resolve()
        ):
            raise ValueError("conversion sidecar names a different H5AD derivative")
        if str(conversion["source_sha256"]) != _sha256(record.count_matrix_file):
            raise ValueError("conversion sidecar source hash differs from the manifest RNA file")
        if str(conversion["derivative_sha256"]) != _sha256(input_path):
            raise ValueError("conversion sidecar derivative hash differs from --input")
    if not input_path.lower().endswith(".h5ad"):
        raise ValueError(
            "prepare-reference accepts H5AD; convert RDS with scripts/export_seurat.R "
            "and pass the result via --input"
        )
    filters = h5ad_filters(record)
    reference = load_h5ad_reference(
        input_path,
        filters=filters,
        cell_type_key=args.cell_type_key,
        genes=_gene_list(args.genes),
        gene_key=args.gene_key,
        layer=args.layer,
        sample_id=record.specimen_id,
        chunk_size=args.chunk_size,
    )
    # Public H5AD donor/sample columns often use accession-internal codes.  The
    # manifest IDs are the experiment-level identities used by all downstream
    # leakage guards, so normalize them after the exact filters have run.
    reference = replace(
        reference,
        donor_ids=np.full(
            reference.shape[0],
            record.donor_id,
            dtype=np.dtype("U%d" % max(1, len(record.donor_id))),
        ),
        sample_ids=np.full(
            reference.shape[0],
            record.specimen_id,
            dtype=np.dtype("U%d" % max(1, len(record.specimen_id))),
        ),
        block_id=record.block_id,
        source_count_sha256=_sha256(input_path),
    )
    reference.save_npz(args.output)
    _json(
        {
            "output": str(Path(args.output).resolve()),
            "sample_id": reference.sample_id,
            "cells": reference.shape[0],
            "genes": reference.shape[1],
            "cell_types": sorted(set(reference.cell_type_labels.tolist())),
            "source_count_sha256": reference.source_count_sha256,
        }
    )
    return 0


def command_prepare_spatial_truth(args: argparse.Namespace) -> int:
    """Create a locked, provenance-bound Visium evaluation artifact."""

    record = _record(args.manifest, args.section_id)
    if not record.included:
        raise ValueError("cannot prepare spatial truth from an excluded manifest row")
    if record.analysis_role not in LOCKED_TARGET_ROLES:
        raise ValueError(
            "prepare-spatial-truth requires a locked target manifest role, found %s"
            % record.analysis_role
        )
    if not record.spatial_count_matrix_file:
        raise ValueError("selected manifest row has no spatial_count_matrix_file")

    counts_path = str(Path(args.counts).expanduser().resolve())
    manifest_spatial_source = record.spatial_count_matrix_file
    direct_manifest_source = False
    if "::" not in manifest_spatial_source:
        direct_manifest_source = Path(manifest_spatial_source).expanduser().resolve() == Path(
            counts_path
        )
    conversion_path = ""
    if not direct_manifest_source:
        if not args.conversion_provenance:
            raise ValueError(
                "derived spatial counts require --conversion-provenance bound to the "
                "manifest spatial source"
            )
        conversion_path = str(Path(args.conversion_provenance).expanduser().resolve())
        with Path(conversion_path).open("r", encoding="utf-8") as handle:
            conversion = json.load(handle)
        required_conversion = {
            "source_path",
            "source_sha256",
            "derivative_path",
            "derivative_sha256",
        }
        if not isinstance(conversion, dict) or not required_conversion.issubset(conversion):
            raise ValueError("spatial conversion provenance sidecar is incomplete")
        if "::" in manifest_spatial_source:
            source_matches = str(conversion["source_path"]) == manifest_spatial_source
        else:
            source_matches = (
                Path(str(conversion["source_path"])).expanduser().resolve()
                == Path(manifest_spatial_source).expanduser().resolve()
            )
        if not source_matches:
            raise ValueError("spatial conversion sidecar names a different manifest source")
        if Path(str(conversion["derivative_path"])).expanduser().resolve() != Path(counts_path):
            raise ValueError("spatial conversion sidecar names a different counts derivative")
        if str(conversion["source_sha256"]) != _artifact_sha256(manifest_spatial_source):
            raise ValueError("spatial conversion source hash differs from the manifest source")
        if str(conversion["derivative_sha256"]) != _artifact_sha256(counts_path):
            raise ValueError("spatial conversion derivative hash differs from --counts")

    genes = _gene_list(args.genes)
    assert genes is not None
    counts = read_visium_counts(
        counts_path,
        genes=genes,
        layer=args.layer,
        gene_key=args.gene_key,
        chunk_size=args.chunk_size,
    )
    positions = read_tissue_positions(
        args.positions,
        coordinate_scale=args.coordinate_scale,
    )
    spot_diameter = read_spot_diameter(
        args.scalefactors,
        coordinate_scale=args.coordinate_scale,
    )
    nuclei = load_nuclei(args.nuclei)
    # Use exactly the same raw-ID namespacing operation as prepare-histology so
    # evaluate-spatial can enforce row identity against PredictionBundle.
    nucleus_ids = canonical_nucleus_ids(nuclei.source_ids, sample_id=record.specimen_id)

    source_artifacts = [
        counts_path,
        str(Path(args.positions).expanduser().resolve()),
        str(Path(args.scalefactors).expanduser().resolve()),
        str(Path(args.nuclei).expanduser().resolve()),
        str(Path(args.genes).expanduser().resolve()),
        str(Path(args.manifest).expanduser().resolve()),
    ]
    source_roles = [
        "locked_spatial_counts",
        "locked_spatial_coordinates",
        "locked_spatial_scalefactors",
        "sample_segmentation",
        "canonical_gene_panel",
        "shared_manifest",
    ]
    if conversion_path:
        source_artifacts.extend((conversion_path, manifest_spatial_source))
        source_roles.extend(("conversion_provenance", "manifest_spatial_source"))
    source_hashes = [_artifact_sha256(path) for path in source_artifacts]
    artifact = build_spatial_truth(
        counts=counts,
        positions=positions,
        nucleus_ids=nucleus_ids,
        nucleus_coordinates_px=nuclei.centroids_px,
        spot_radius_px=spot_diameter / 2.0,
        barcode_suffix_policy=args.barcode_suffix_policy,
        metadata={
            "analysis_role": record.analysis_role,
            "cohort_id": record.cohort_id,
            "donor_id": record.donor_id,
            "specimen_id": record.specimen_id,
            "block_id": record.block_id,
            "section_id": record.section_id,
            "outer_fold": record.outer_fold,
            "inner_fold": record.inner_fold,
        },
        source_artifacts=source_artifacts,
        source_sha256=source_hashes,
        source_roles=source_roles,
    )
    artifact.save_npz(args.output)
    _json(
        {
            "output": str(Path(args.output).expanduser().resolve()),
            "contract": "heir.spatial_truth",
            "version": 1,
            "analysis_role": artifact.analysis_role,
            "section_id": artifact.section_id,
            "spots": len(artifact.spot_ids),
            "genes": len(artifact.gene_names),
            "nuclei": len(artifact.nucleus_ids),
            "assigned_nuclei": artifact.assigned_nuclei,
            "evaluable_spots": artifact.evaluable_spots,
            "barcode_suffix_policy": artifact.barcode_suffix_policy,
            "spot_radius_px": artifact.spot_radius_px,
            "expression_space_id": artifact.expression_space_id,
            "source_sha256": dict(zip(source_artifacts, source_hashes)),
        }
    )
    return 0


def command_filter_nuclei_to_visium(args: argparse.Namespace) -> int:
    """Filter Space Ranger nuclei using capture geometry without target counts."""

    assignment, provenance = filter_nucleus_csv_to_visium(
        nuclei_path=args.nuclei,
        positions_path=args.positions,
        scalefactors_path=args.scalefactors,
        filtered_csv_path=args.output,
        assignment_npz_path=args.assignment_output,
        provenance_json_path=args.provenance_output,
        coordinate_scale=args.coordinate_scale,
        overwrite=args.overwrite,
    )
    _json(
        {
            "contract": provenance["contract"],
            "version": provenance["version"],
            "geometry_only": True,
            "target_expression_accessed": False,
            "source_nuclei": len(assignment.source_nucleus_ids),
            "retained_nuclei": len(assignment.retained_source_index),
            "excluded_nuclei": len(assignment.source_nucleus_ids)
            - len(assignment.retained_source_index),
            "in_tissue_spots": len(assignment.spot_ids),
            "spot_radius_px": assignment.spot_radius_px,
            "output": str(Path(args.output).expanduser().resolve()),
            "assignment_output": str(Path(args.assignment_output).expanduser().resolve()),
            "provenance_output": str(Path(args.provenance_output).expanduser().resolve()),
        }
    )
    return 0


def command_prepare_histology(args: argparse.Namespace) -> int:
    """Join canonical nuclei and cached features, calibrate coordinates, and build a graph."""

    if bool(args.manifest) != bool(args.section_id):
        raise ValueError("--manifest and --section-id must be supplied together")
    record = _record(args.manifest, args.section_id) if args.manifest else None
    sample_id = args.sample_id or (record.specimen_id if record is not None else "")
    donor_id = args.donor_id or (record.donor_id if record is not None else "")
    block_id = args.block_id or (record.block_id if record is not None else "")
    slide_id = args.slide_id or (record.section_id if record is not None else "")
    histology_source = args.histology_source or (record.he_file if record is not None else "")
    for name, value in (
        ("sample-id", sample_id),
        ("donor-id", donor_id),
        ("block-id", block_id),
        ("slide-id", slide_id),
        ("histology-source", histology_source),
        ("feature-space-id", args.feature_space_id),
    ):
        if not value or not value.strip():
            raise ValueError("prepare-histology requires --%s or manifest provenance" % name)
    if record is not None:
        for name, supplied, expected in (
            ("sample-id", args.sample_id, record.specimen_id),
            ("donor-id", args.donor_id, record.donor_id),
            ("block-id", args.block_id, record.block_id),
            ("slide-id", args.slide_id, record.section_id),
        ):
            if supplied and supplied != expected:
                raise ValueError("--%s conflicts with the selected manifest row" % name)
    histology_source_sha256 = _sha256(histology_source)
    if record is not None and args.histology_source:
        manifest_histology_sha256 = _sha256(record.he_file)
        if histology_source_sha256 != manifest_histology_sha256:
            raise ValueError("--histology-source content differs from the manifest H&E")

    # Align cached features to the raw segmentation IDs first.  Namespacing is
    # applied only to the emitted contract, otherwise a normal raw-ID feature
    # bundle cannot be joined when --sample-id is supplied.
    nuclei = load_nuclei(args.nuclei)
    features = load_feature_bundle(args.features, expected_ids=nuclei.source_ids)
    nucleus_ids = canonical_nucleus_ids(nuclei.source_ids, sample_id=sample_id)
    if args.coordinates_are_microns:
        coordinates_um = nuclei.centroids_px
    else:
        mpp = args.mpp or (record.native_mpp if record is not None else None)
        if mpp is None:
            raise ValueError("prepare-histology requires --mpp or --coordinates-are-microns")
        coordinates_um = PixelMicronTransform(mpp).native_to_microns(nuclei.centroids_px)
    confidence = np.asarray(nuclei.confidence, dtype=np.float32)
    confidence = np.where(
        np.isfinite(confidence), confidence, args.default_segmentation_confidence
    ).astype(np.float32)
    if not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError("resolved segmentation confidence must lie in [0, 1]")
    artifact = np.zeros(len(nuclei), dtype=np.float32)
    if args.artifact_key:
        if args.artifact_key not in features.metadata:
            raise ValueError("feature bundle lacks artifact key %s" % args.artifact_key)
        artifact = np.asarray(features.metadata[args.artifact_key], dtype=np.float32)
    boundary = None
    if args.boundary_weight_key:
        if args.boundary_weight_key not in features.metadata:
            raise ValueError("feature bundle lacks boundary key %s" % args.boundary_weight_key)
        boundary = np.asarray(features.metadata[args.boundary_weight_key], dtype=np.float32)
    graph = build_spatial_graph(
        coordinates_um,
        k=args.graph_k,
        radius=args.graph_radius_um,
        max_degree=args.graph_max_degree,
        boundary_weights=boundary,
    )
    combined_features = features.features
    if not args.exclude_morphology and nuclei.morphology.shape[1]:
        morphology = nuclei.morphology.astype(np.float32)
        mean = morphology.mean(axis=0, keepdims=True)
        std = morphology.std(axis=0, keepdims=True)
        standardized = (morphology - mean) / np.maximum(std, 1.0e-6)
        combined_features = np.concatenate((combined_features, standardized), axis=1)
    bag = HistologyBag(
        slide_id=slide_id,
        nucleus_ids=nucleus_ids,
        features=combined_features,
        coordinates_um=coordinates_um,
        morphology=nuclei.morphology,
        segmentation_confidence=confidence,
        artifact_probability=artifact,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        sample_id=sample_id,
        donor_id=donor_id,
        block_id=block_id,
        feature_space_id=args.feature_space_id,
        histology_source_sha256=histology_source_sha256,
        nuclei_source_sha256=_sha256(args.nuclei),
        feature_source_sha256=_sha256(args.features),
    )
    bag.save_npz(args.output)
    _json(
        {
            "output": str(Path(args.output).expanduser().resolve()),
            "slide_id": bag.slide_id,
            "nuclei": bag.n_nuclei,
            "feature_width": int(bag.features.shape[1]),
            "edges": int(graph.num_edges),
            "nuclei_sha256": _sha256(args.nuclei),
            "features_sha256": _sha256(args.features),
            "histology_source_sha256": bag.histology_source_sha256,
            "feature_space_id": bag.feature_space_id,
        }
    )
    return 0


def command_segment_histology(args: argparse.Namespace) -> int:
    """Run/import the default Space Ranger nucleus segmentation stage."""

    if bool(args.image) == bool(args.geojson):
        raise ValueError("supply exactly one of --image or --geojson")
    if args.image:
        run = run_spaceranger_segment(
            args.image,
            run_id=args.run_id or args.slide_id,
            output_directory=args.output_directory,
            executable=args.spaceranger,
            localcores=args.localcores,
            localmem_gb=args.localmem_gb,
            max_nucleus_diameter_px=args.max_nucleus_diameter_px,
            cuda_visible_devices=args.cuda_visible_devices,
            timeout_seconds=args.timeout_seconds,
        )
        geojson = run.geojson_path
        version = run.spaceranger_version
    else:
        geojson = Path(args.geojson).expanduser().resolve()
        version = args.spaceranger_version
    segmentation = read_spaceranger_geojson(
        geojson,
        slide_id=args.slide_id,
        spaceranger_version=version,
        minimum_area_px2=args.minimum_area_px2,
    )
    csv_path, npz_path = export_spaceranger_artifacts(
        segmentation,
        csv_path=args.nuclei_output,
        npz_path=args.features_output,
        overwrite=args.overwrite,
    )
    _json(
        {
            "method": segmentation.method,
            "spaceranger_version": segmentation.spaceranger_version,
            "slide_id": segmentation.slide_id,
            "nuclei": len(segmentation),
            "skipped_features": segmentation.skipped_features,
            "geojson": str(Path(geojson).resolve()),
            "geojson_sha256": segmentation.source_sha256,
            "nuclei_output": str(csv_path),
            "features_output": str(npz_path),
            "cuda_visible_devices": args.cuda_visible_devices,
        }
    )
    return 0


def _positive_float_tuple(value: str, name: str) -> Tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("%s must contain comma-separated numbers" % name) from error
    if not values or any(not np.isfinite(item) or item <= 0 for item in values):
        raise ValueError("%s must contain finite positive numbers" % name)
    if len(set(values)) != len(values):
        raise ValueError("%s values must be unique" % name)
    return values


def command_extract_pathology_features(args: argparse.Namespace) -> int:
    """Extract frozen multi-scale image features for Space Ranger nuclei."""

    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    nuclei = load_nuclei(args.nuclei)
    stop = len(nuclei) if args.limit is None else min(len(nuclei), args.offset + args.limit)
    if args.offset >= stop:
        raise ValueError("selected nucleus range is empty")
    selection = slice(args.offset, stop)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" if args.mixed_precision is None else args.mixed_precision
    if use_amp and device.type != "cuda":
        raise ValueError("mixed precision is currently supported only on CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_start = time.perf_counter()
    if args.encoder == "omiclip-loki":
        if not args.checkpoint:
            raise ValueError("OmiCLIP/Loki extraction requires --checkpoint")
        encoder, descriptor = load_omiclip_visual_encoder(
            args.checkpoint,
            device=device,
            input_size=args.input_size,
            trust_checkpoint=args.trust_checkpoint,
        )
    elif args.encoder == "resnet50-imagenet":
        if args.checkpoint:
            raise ValueError("the torchvision ImageNet baseline does not accept --checkpoint")
        encoder, descriptor = load_imagenet_resnet50_encoder(
            device=device,
            input_size=args.input_size,
        )
    else:  # pragma: no cover - argparse enforces this
        raise ValueError("unknown pathology encoder")
    model_load_seconds = time.perf_counter() - model_start
    scales = _positive_float_tuple(args.patch_diameters_um, "patch-diameters-um")
    with open_slide(args.image, native_mpp=args.mpp, backend=args.backend) as slide:
        result = extract_nucleus_pathology_features(
            slide,
            nuclei.source_ids[selection],
            nuclei.centroids_px[selection],
            encoder,
            descriptor,
            patch_diameters_um=scales,
            batch_size=args.batch_size,
            device=device,
            mixed_precision=use_amp,
            model_load_seconds=model_load_seconds,
        )
    if device.type == "cuda":
        result = with_peak_memory(result, torch.cuda.max_memory_allocated(device))
    slide_sha256 = _sha256(args.image)
    nuclei_sha256 = _sha256(args.nuclei)
    output = save_pathology_feature_npz(
        result,
        args.output,
        slide_sha256=slide_sha256,
        nuclei_sha256=nuclei_sha256,
        overwrite=args.overwrite,
    )
    report = {
        "output": str(output),
        "contract": "heir.nucleus_pathology_features.v1",
        "encoder": result.descriptor.name,
        "scientific_role": result.descriptor.scientific_role,
        "checkpoint_sha256": result.descriptor.checkpoint_sha256,
        "feature_space_id": result.feature_space_id,
        "feature_width": int(result.features.shape[1]),
        "nuclei": len(result.nucleus_ids),
        "selection_offset": args.offset,
        "selection_stop": stop,
        "patch_diameters_um": list(result.patch_diameters_um),
        "native_mpp": list(result.native_mpp),
        "slide_backend": result.slide_backend,
        "slide_sha256": slide_sha256,
        "nuclei_sha256": nuclei_sha256,
        "telemetry": asdict(result.telemetry),
    }
    if args.telemetry_output:
        atomic_json_dump(report, args.telemetry_output)
    _json(report)
    return 0


def _log_normalize(
    counts: sparse.csr_matrix,
    library_sizes: Optional[np.ndarray] = None,
    target_sum: float = EXPRESSION_TARGET_SUM,
) -> sparse.csr_matrix:
    matrix = counts.astype(np.float32, copy=True)
    library = (
        np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        if library_sizes is None
        else np.asarray(library_sizes, dtype=np.float64).reshape(-1)
    )
    if library.shape != (matrix.shape[0],):
        raise ValueError("library_sizes must have one value per count-matrix row")
    if not np.isfinite(library).all() or np.any(library < 0):
        raise ValueError("library_sizes must be finite and non-negative")
    scale = target_sum / np.maximum(library, 1.0)
    matrix = sparse.diags(scale).dot(matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def command_build_prototypes(args: argparse.Namespace) -> int:
    reference = RNAReference.load_npz(args.reference)
    if bool(args.manifest) != bool(args.section_id):
        raise ValueError("--manifest and --section-id must be supplied together")
    manifest_record = _record(args.manifest, args.section_id) if args.manifest else None
    latent_training_donors: Tuple[str, ...] = ()
    latent_transform_sha256 = ""
    emitted_reference_path = ""
    transform_requested = bool(args.fit_latent_transform or args.latent_transform)
    if reference.latent.shape[1] > 0 and not args.recompute_latent and not transform_requested:
        latent = reference.latent
        latent_method = "reference"
        latent_space_id = args.latent_space_id or reference.latent_space_id
        if (
            args.latent_space_id
            and reference.latent_space_id
            and (args.latent_space_id != reference.latent_space_id)
        ):
            raise ValueError("manual latent-space ID conflicts with the RNA reference")
    else:
        if args.fit_latent_transform and args.latent_transform:
            raise ValueError("fit and consume latent-transform options are mutually exclusive")
        if not transform_requested:
            raise ValueError(
                "reference has no shared latent; pass --fit-latent-transform once on a "
                "development reference or consume it with --latent-transform"
            )
        if args.latent_transform:
            genes, components, target_sum, transform_provenance = _load_latent_transform(
                args.latent_transform
            )
            latent_training_donors = tuple(
                str(value) for value in transform_provenance["training_donors"]
            )
            if not latent_training_donors and not args.unsafe_allow_legacy_latent_transform:
                raise ValueError(
                    "latent transform lacks training-donor provenance; rebuild it from a "
                    "manifest-bound development reference"
                )
            if not np.array_equal(reference.gene_ids, genes):
                raise ValueError("RNA reference gene order differs from the latent transform")
            normalized = _log_normalize(
                reference.counts,
                library_sizes=reference.library_sizes,
                target_sum=target_sum,
            )
            latent = np.asarray(normalized.dot(components.T), dtype=np.float32)
            latent_method = "shared_truncated_svd_transform"
            latent_transform_sha256 = _sha256(args.latent_transform)
            latent_space_id = "sha256:%s" % latent_transform_sha256
        else:
            if manifest_record is None:
                raise ValueError(
                    "fitting a shared latent transform requires --manifest and --section-id"
                )
            if manifest_record.analysis_role not in {
                "train",
                "training",
                "development",
                "pretraining",
            }:
                raise ValueError("shared latent transforms cannot be fitted on a locked role")
            reference_donor = _single_value(reference.donor_ids, "RNA donor IDs")
            if (
                reference_donor != manifest_record.donor_id
                or reference.sample_id != manifest_record.specimen_id
                or reference.block_id != manifest_record.block_id
            ):
                raise ValueError("RNAReference identity differs from the latent-fit manifest row")
            if args.latent_dim >= min(reference.shape):
                raise ValueError("latent_dim must be smaller than cells and genes")
            normalized = _log_normalize(
                reference.counts,
                library_sizes=reference.library_sizes,
            )
            reducer = TruncatedSVD(n_components=args.latent_dim, random_state=args.seed)
            latent = reducer.fit_transform(normalized).astype(np.float32)
            assert args.fit_latent_transform
            _save_latent_transform(
                args.fit_latent_transform,
                reference.gene_ids,
                reducer.components_,
                EXPRESSION_TARGET_SUM,
                {
                    "training_donors": (reference_donor,),
                    "source_reference_sha256": _sha256(args.reference),
                    "manifest_sha256": _sha256(args.manifest),
                    "analysis_role": manifest_record.analysis_role,
                    "cohort_id": manifest_record.cohort_id,
                    "section_id": manifest_record.section_id,
                    "outer_fold": manifest_record.outer_fold,
                    "inner_fold": manifest_record.inner_fold,
                },
            )
            latent_training_donors = (reference_donor,)
            latent_method = "fitted_shared_truncated_svd"
            latent_transform_sha256 = _sha256(args.fit_latent_transform)
            latent_space_id = "sha256:%s" % latent_transform_sha256
        if args.latent_space_id and args.latent_space_id != latent_space_id:
            raise ValueError("manual latent-space ID conflicts with the shared transform")
        if args.reference_with_latent:
            enriched = replace(
                reference,
                latent=latent,
                latent_space_id=latent_space_id,
            )
            enriched.save_npz(args.reference_with_latent)
            emitted_reference_path = str(Path(args.reference_with_latent).expanduser().resolve())
    if not latent_space_id:
        raise ValueError(
            "latent identity is missing; supply --latent-space-id or rebuild "
            "with a shared transform"
        )
    # When an identified latent reference is emitted, it is the only
    # RNAReference artifact that can be consumed safely by assemble-batch.
    # Bind the prototype provenance to that exact derivative instead of the
    # latent-free input; otherwise no truthful build -> assemble artifact pair
    # can satisfy both the latent-space and source-reference checks.
    prototype_reference_path = emitted_reference_path or str(
        Path(args.reference).expanduser().resolve()
    )
    prototypes = build_sample_prototypes(
        latent,
        reference.cell_type_labels,
        sample_id=reference.sample_id,
        max_prototypes_per_type=args.max_per_type,
        minimum_cells=args.minimum_cells,
        shrinkage_kappa=args.shrinkage_kappa,
        seed=args.seed,
        include_rare_types=args.include_rare_types,
        latent_space_id=latent_space_id,
        donor_id=_single_value(reference.donor_ids, "RNA donor IDs"),
        block_id=reference.block_id,
        source_reference_sha256=_sha256(prototype_reference_path),
        latent_training_donors=latent_training_donors,
        latent_transform_sha256=latent_transform_sha256,
    )
    prototypes.save_npz(args.output)
    _json(
        {
            "output": str(Path(args.output).resolve()),
            "sample_id": reference.sample_id,
            "latent_method": latent_method,
            "latent_transform": args.fit_latent_transform or args.latent_transform,
            "latent_space_id": latent_space_id,
            "prototype_reference": prototype_reference_path,
            "prototype_reference_sha256": prototypes.source_reference_sha256,
            "prototypes": len(prototypes.prototype_ids),
            "cell_types": sorted(set(prototypes.cell_type_labels.tolist())),
        }
    )
    return 0


def command_fit_residual_geometry(args: argparse.Namespace) -> int:
    """Freeze within-type RNA PCA directions and molecularly scaled bounds."""

    reference = RNAReference.load_npz(args.reference)
    if reference.latent.shape[1] == 0 or not reference.latent_space_id:
        raise ValueError("residual geometry requires an identified RNA latent representation")
    prototypes = None if args.prototypes is None else PrototypeSet.load_npz(args.prototypes)
    if prototypes is not None and prototypes.latent_space_id != reference.latent_space_id:
        raise ValueError("prototype and reference latent-space identities differ")
    geometry = fit_rna_residual_geometry(
        reference.latent,
        reference.cell_type_labels,
        args.rank,
        type_names=args.type_name,
        prototype_means=None if prototypes is None else prototypes.means,
        prototype_labels=None if prototypes is None else prototypes.cell_type_labels,
        prototype_variances=None if prototypes is None else prototypes.variances,
        calibration_quantile=args.calibration_quantile,
        bound_fraction=args.bound_fraction,
        minimum_bound=args.minimum_bound,
        maximum_bound=args.maximum_bound,
        minimum_calibration_cells=args.minimum_calibration_cells,
        latent_space_id=reference.latent_space_id,
        source_reference_sha256=_sha256(args.reference),
        training_donors=tuple(sorted(set(reference.donor_ids.tolist()))),
        latent_transform_sha256=("" if prototypes is None else prototypes.latent_transform_sha256),
    )
    geometry.to_npz(args.output)
    _json(
        {
            "output": str(Path(args.output).expanduser().resolve()),
            "schema": geometry.SCHEMA,
            "latent_space_id": geometry.latent_space_id,
            "rank": geometry.rank,
            "type_bounds": {
                str(name): float(bound)
                for name, bound in zip(
                    geometry.type_names.tolist(),
                    geometry.residual_type_max_norm.tolist(),
                )
            },
            "scale_sources": {
                str(name): str(source)
                for name, source in zip(
                    geometry.type_names.tolist(),
                    geometry.scale_sources.tolist(),
                )
            },
        }
    )
    return 0


def command_fit_ood(args: argparse.Namespace) -> int:
    """Fit pathology-feature OOD only on declared development donors."""

    bags = [HistologyBag.load_npz(path) for path in args.histology]
    widths = {bag.features.shape[1] for bag in bags}
    if len(widths) != 1:
        raise ValueError("OOD training histology bags use different feature widths")
    if any(not bag.donor_id or not bag.feature_space_id for bag in bags):
        raise ValueError("fit-ood requires donor and feature-space provenance in every bag")
    feature_spaces = {bag.feature_space_id for bag in bags}
    if len(feature_spaces) != 1:
        raise ValueError("OOD training histology bags use different feature spaces")
    feature_space_id = next(iter(feature_spaces))
    donors = tuple(sorted({bag.donor_id for bag in bags}))
    if args.training_donor and set(args.training_donor) != set(donors):
        raise ValueError("--training-donor must exactly match HistologyBag donor provenance")
    detector = MahalanobisOOD().fit(
        np.concatenate([bag.features for bag in bags], axis=0),
        analysis_role=args.analysis_role,
        quantile=args.quantile,
        training_donors=donors,
        feature_space_id=feature_space_id,
    )
    detector.source_sha256 = tuple(_sha256(path) for path in args.histology)
    detector.to_npz(args.output)
    _json(
        {
            "output": str(Path(args.output).expanduser().resolve()),
            "training_donors": list(donors),
            "feature_width": next(iter(widths)),
            "feature_space_id": feature_space_id,
            "threshold": detector.threshold,
            "quantile": detector.quantile,
            "source_sha256": list(detector.source_sha256),
        }
    )
    return 0


_LOCKED_ROLES = {
    "validation",
    "spatial_validation",
    "external_validation",
    "test",
    "locked_test",
    "locked_validation",
}


def _single_value(values: np.ndarray, name: str, fallback: str = "") -> str:
    unique = sorted(set(str(value) for value in values.tolist()))
    if len(unique) == 1:
        return unique[0]
    if fallback:
        return fallback
    raise ValueError("%s contains multiple values; specify the intended value explicitly" % name)


def _ranked_marker_mask(centroids: np.ndarray, markers_per_type: int) -> np.ndarray:
    if markers_per_type <= 0:
        raise ValueError("markers-per-type must be positive")
    types, genes = centroids.shape
    selected = min(markers_per_type, max(1, genes // 2))
    result = np.zeros((types, genes), dtype=bool)
    for index in range(types):
        others = np.delete(centroids, index, axis=0).mean(axis=0)
        # Centroids are in log-normalized expression space, so this difference
        # is a conservative log-fold-change ranking.
        score = centroids[index] - others
        order = np.argsort(-score, kind="stable")
        positive = order[score[order] > 0]
        chosen = positive[:selected] if len(positive) else order[:1]
        result[index, chosen] = True
    return result


def command_assemble_batch(args: argparse.Namespace) -> int:
    """Assemble unregistered H&E + RNA into sample-level weak supervision."""

    if not 0.0 <= args.artifact_threshold <= 1.0:
        raise ValueError("artifact-threshold must lie in [0, 1]")
    bag = HistologyBag.load_npz(args.histology)
    prototypes = PrototypeSet.load_npz(args.prototypes)
    reference = RNAReference.load_npz(args.reference)
    artifact_latent_ids = {
        value for value in (reference.latent_space_id, prototypes.latent_space_id) if value
    }
    if len(artifact_latent_ids) > 1:
        raise ValueError("RNA reference and prototypes use different latent-space identities")
    if args.latent_space_id:
        if artifact_latent_ids and args.latent_space_id not in artifact_latent_ids:
            raise ValueError("manual latent-space ID conflicts with RNA/prototype artifacts")
        latent_space_id = args.latent_space_id
    elif reference.latent_space_id and prototypes.latent_space_id:
        latent_space_id = reference.latent_space_id
    elif args.unsafe_allow_unspecified_latent_space:
        latent_space_id = next(iter(artifact_latent_ids), "unspecified")
    else:
        raise ValueError(
            "RNA reference and prototypes must carry the same latent_space_id; "
            "use --latent-space-id only to migrate checked legacy artifacts"
        )
    manifest_record = None
    if bool(args.manifest) != bool(args.section_id):
        raise ValueError("--manifest and --section-id must be supplied together")
    if args.manifest:
        manifest_record = _record(args.manifest, args.section_id)

    prototype_sample = _single_value(prototypes.sample_ids, "prototype sample IDs")
    if prototype_sample != reference.sample_id:
        raise ValueError(
            "prototype sample %s does not match RNA reference %s"
            % (prototype_sample, reference.sample_id)
        )
    if args.sample_id and args.sample_id != reference.sample_id:
        raise ValueError("--sample-id differs from the RNAReference sample")
    sample_id = args.sample_id or reference.sample_id
    reference_donor = _single_value(reference.donor_ids, "RNA donor IDs", args.donor_id or "")
    donor_id = args.donor_id or (
        manifest_record.donor_id if manifest_record is not None else reference_donor
    )
    block_id = args.block_id or (manifest_record.block_id if manifest_record is not None else "")
    analysis_role = args.analysis_role or (
        manifest_record.analysis_role if manifest_record is not None else "train"
    )
    bag_id = args.bag_id or bag.slide_id
    if not donor_id or not block_id:
        raise ValueError("assemble-batch requires donor and block provenance")
    if reference_donor != donor_id:
        raise ValueError("RNAReference donor differs from the selected donor")
    if reference.block_id and reference.block_id != block_id:
        raise ValueError("RNAReference block differs from the selected block")
    missing_bag_provenance = [
        name
        for name, value in (
            ("sample_id", bag.sample_id),
            ("donor_id", bag.donor_id),
            ("block_id", bag.block_id),
            ("feature_space_id", bag.feature_space_id),
            ("histology_source_sha256", bag.histology_source_sha256),
            ("nuclei_source_sha256", bag.nuclei_source_sha256),
            ("feature_source_sha256", bag.feature_source_sha256),
        )
        if not value
    ]
    if missing_bag_provenance and not args.unsafe_allow_missing_histology_provenance:
        raise ValueError(
            "HistologyBag lacks required provenance: %s" % ", ".join(missing_bag_provenance)
        )
    if bag.sample_id and bag.sample_id != sample_id:
        raise ValueError("HistologyBag sample differs from the RNAReference")
    if bag.donor_id and bag.donor_id != donor_id:
        raise ValueError("HistologyBag donor differs from the RNAReference")
    if bag.block_id and bag.block_id != block_id:
        raise ValueError("HistologyBag block differs from the selected RNA block")
    if not prototypes.donor_id or not prototypes.block_id or not prototypes.source_reference_sha256:
        if not args.unsafe_allow_missing_prototype_provenance:
            raise ValueError(
                "PrototypeSet lacks donor/reference provenance; rebuild it or use the unsafe "
                "legacy migration override"
            )
    else:
        if prototypes.donor_id != donor_id:
            raise ValueError("prototype donor differs from the selected RNA/H&E donor")
        if prototypes.block_id != block_id:
            raise ValueError("prototype block differs from the selected RNA/H&E block")
        if prototypes.source_reference_sha256 != _sha256(args.reference):
            raise ValueError("PrototypeSet was built from a different RNAReference artifact")
    if analysis_role in _LOCKED_ROLES and donor_id in set(prototypes.latent_training_donors):
        raise ValueError("prototype latent transform was fitted on the locked target donor")
    if manifest_record is not None:
        if args.donor_id and args.donor_id != manifest_record.donor_id:
            raise ValueError("--donor-id conflicts with the selected manifest row")
        if args.block_id and args.block_id != manifest_record.block_id:
            raise ValueError("--block-id conflicts with the selected manifest row")
        if args.analysis_role and args.analysis_role != manifest_record.analysis_role:
            raise ValueError("--analysis-role conflicts with the selected manifest row")
        if bag.slide_id != manifest_record.section_id:
            raise ValueError("HistologyBag slide differs from the selected manifest section")
        if bag.histology_source_sha256 and (
            bag.histology_source_sha256 != _sha256(manifest_record.he_file)
        ):
            raise ValueError("HistologyBag was prepared from a different H&E source")

    type_names = tuple(sorted(set(str(value) for value in prototypes.cell_type_labels.tolist())))
    if len(type_names) < 2:
        raise ValueError("HEIR needs prototypes for at least two cell types")
    type_lookup = {name: index for index, name in enumerate(type_names)}
    prototype_types = np.asarray(
        [type_lookup[str(value)] for value in prototypes.cell_type_labels],
        dtype=np.int64,
    )
    labels = np.asarray(reference.cell_type_labels, dtype=np.dtype("U"))
    supported = np.asarray([value in type_lookup for value in labels], dtype=bool)
    if not supported.any():
        raise ValueError("RNA labels do not overlap the prototype cell types")
    composition = np.asarray(
        [(labels[supported] == name).sum() for name in type_names],
        dtype=np.float32,
    )
    composition /= composition.sum()

    normalized = _log_normalize(
        reference.counts,
        library_sizes=reference.library_sizes,
    )
    supported_expression = normalized[supported]
    supported_linear = supported_expression.copy()
    supported_linear.data = np.expm1(supported_linear.data)
    pseudobulk = np.log1p(np.asarray(supported_linear.mean(axis=0), dtype=np.float32).reshape(-1))
    marker_centroids = np.stack(
        [
            np.asarray(normalized[labels == name].mean(axis=0), dtype=np.float32).reshape(-1)
            for name in type_names
        ]
    )
    marker_mask = _ranked_marker_mask(marker_centroids, args.markers_per_type)
    confidence = np.asarray(bag.segmentation_confidence, dtype=np.float32)
    artifact_probability = np.asarray(bag.artifact_probability, dtype=np.float32)
    cell_weights = confidence * (1.0 - artifact_probability)
    cell_weights[artifact_probability >= args.artifact_threshold] = 0.0
    if not np.any(cell_weights > 0):
        raise ValueError("segmentation/artifact weights leave no usable image cells")
    ood_mask = None
    ood_detector = None
    if args.ood_artifact:
        ood_detector = MahalanobisOOD.from_npz(args.ood_artifact)
        if bag.feature_space_id and ood_detector.feature_space_id != bag.feature_space_id:
            raise ValueError("OOD detector feature space differs from the HistologyBag")
        if ood_detector.threshold is None:
            raise ValueError("OOD artifact has no fitted threshold")
        if analysis_role in _LOCKED_ROLES and donor_id in set(ood_detector.training_donors):
            raise ValueError("OOD detector was fitted on locked target donor %s" % donor_id)
        ood_mask = ood_detector.score(bag.features) > ood_detector.threshold
    unknown_targets = None
    if args.unknown_targets:
        if analysis_role in _LOCKED_ROLES:
            raise ValueError("unknown calibration targets cannot come from a locked role")
        with np.load(args.unknown_targets, allow_pickle=False) as archive:
            if "nucleus_ids" not in archive or "unknown_targets" not in archive:
                raise ValueError("unknown-target artifact needs nucleus_ids and unknown_targets")
            target_ids = [str(value) for value in archive["nucleus_ids"].tolist()]
            target_values = np.asarray(archive["unknown_targets"], dtype=np.float32)
        if len(set(target_ids)) != len(target_ids) or target_values.shape != (len(target_ids),):
            raise ValueError("unknown-target artifact rows are invalid")
        lookup = {value: index for index, value in enumerate(target_ids)}
        if set(lookup) != set(str(value) for value in bag.nucleus_ids.tolist()):
            raise ValueError("unknown-target nuclei differ from the HistologyBag")
        unknown_targets = target_values[
            np.asarray([lookup[str(value)] for value in bag.nucleus_ids.tolist()])
        ]
        if (
            not np.isfinite(unknown_targets).all()
            or np.any(unknown_targets < 0)
            or np.any(unknown_targets > 1)
        ):
            raise ValueError("unknown_targets must be finite and lie in [0, 1]")
    elif ood_mask is not None:
        # A calibrated pathology OOD mask is a target-H&E-only supervision
        # signal for the model's explicit unknown head.  This keeps the weak
        # UOT dustbin and per-nucleus p_unknown aligned without accepting any
        # target-expression input.  An explicit development-only target
        # artifact still takes precedence above.
        unknown_targets = ood_mask.astype(np.float32)
    domain_labels = (
        None
        if args.domain_label is None
        else np.full(bag.n_nuclei, args.domain_label, dtype=np.int64)
    )
    if args.domain_label is not None and args.domain_label < 0:
        raise ValueError("domain-label must be non-negative")
    edges = np.asarray(bag.edge_index, dtype=np.int64)
    spot_assignment = None
    target_spatial_expression = None
    spot_ids: Tuple[str, ...] = ()
    if args.spatial_pretraining_truth:
        if analysis_role != "pretraining":
            raise ValueError("spatial pretraining truth requires analysis_role=pretraining")
        with np.load(args.spatial_pretraining_truth, allow_pickle=False) as archive:
            required = {
                "nucleus_ids",
                "spot_assignment",
                "observed_expression",
                "gene_names",
                "spot_ids",
                "expression_space_id",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("spatial pretraining artifact is missing: %s" % ", ".join(missing))
            spatial_ids = np.asarray(archive["nucleus_ids"], dtype=np.dtype("U"))
            spatial_genes = np.asarray(archive["gene_names"], dtype=np.dtype("U"))
            spot_assignment = np.asarray(archive["spot_assignment"], dtype=np.float32)
            target_spatial_expression = np.asarray(archive["observed_expression"], dtype=np.float32)
            spot_ids = tuple(str(value) for value in archive["spot_ids"].tolist())
            spatial_expression_space_id = str(np.asarray(archive["expression_space_id"]).item())
        if spatial_expression_space_id != EXPRESSION_SPACE_ID:
            raise ValueError(
                "spatial pretraining expression_space_id must be %s" % EXPRESSION_SPACE_ID
            )
        if not np.array_equal(spatial_ids, bag.nucleus_ids.astype(str)):
            raise ValueError("spatial pretraining nucleus_ids must exactly match HistologyBag")
        if not np.array_equal(spatial_genes, reference.gene_ids.astype(str)):
            raise ValueError("spatial pretraining genes must exactly match RNA reference")
        if spot_assignment.ndim != 2 or spot_assignment.shape[1] != bag.n_nuclei:
            raise ValueError("spot_assignment must have shape (spots, nuclei)")
        if target_spatial_expression.shape != (
            spot_assignment.shape[0],
            reference.shape[1],
        ):
            raise ValueError("observed_expression must have shape (spots, genes)")
        if len(spot_ids) != spot_assignment.shape[0] or len(set(spot_ids)) != len(spot_ids):
            raise ValueError("spot_ids must be unique and align to spot_assignment")
        if any(not value.strip() for value in spot_ids):
            raise ValueError("spot_ids cannot contain empty values")
        if (
            not np.isfinite(spot_assignment).all()
            or np.any(spot_assignment < 0)
            or not np.isfinite(target_spatial_expression).all()
            or np.any(target_spatial_expression < 0)
            or np.any(target_spatial_expression > EXPRESSION_MAX + 1.0e-5)
        ):
            raise ValueError(
                "spatial pretraining requires non-negative assignments and finite "
                "log1p-CPM expression within the canonical range"
            )
        spot_mass = spot_assignment.sum(axis=1)
        if np.any(spot_mass <= 0):
            raise ValueError("spatial pretraining spots must each contain assigned nucleus mass")
        pseudobulk = np.log1p(
            np.average(
                np.expm1(target_spatial_expression),
                axis=0,
                weights=spot_mass,
            )
        ).astype(np.float32)

    scgpt_prototypes = None
    scgpt_variances = None
    scgpt_space_id = ""
    program_matrix = None
    target_program_scores = None
    molecular_training_donors = set(prototypes.latent_training_donors)
    sources = [args.histology, args.prototypes, args.reference]
    source_roles = ["sample_assay", "sample_assay", "sample_assay"]
    if args.manifest:
        sources.append(args.manifest)
        source_roles.append("shared_manifest")
    if args.ood_artifact:
        assert ood_detector is not None
        molecular_training_donors.update(ood_detector.training_donors)
        sources.append(args.ood_artifact)
        source_roles.append("shared_teacher")
    if args.unknown_targets:
        sources.append(args.unknown_targets)
        source_roles.append("sample_assay")
    if args.spatial_pretraining_truth:
        sources.append(args.spatial_pretraining_truth)
        source_roles.append("sample_assay")
    if args.scgpt_artifact:
        teacher = SCGPTTeacherArtifact.from_npz(args.scgpt_artifact)
        teacher_names = [str(value) for value in teacher.type_names.tolist()]
        if len(set(teacher_names)) != len(teacher_names):
            raise ValueError("scGPT artifact contains duplicate type names")
        teacher_lookup = {name: index for index, name in enumerate(teacher_names)}
        missing = sorted(set(type_names) - set(teacher_lookup))
        if missing:
            raise ValueError("scGPT artifact lacks batch types: %s" % ", ".join(missing))
        if analysis_role in _LOCKED_ROLES and donor_id in {
            str(value) for value in teacher.training_donors.tolist()
        }:
            raise ValueError(
                "scGPT teacher training donors overlap locked target donor %s" % donor_id
            )
        order = np.asarray([teacher_lookup[name] for name in type_names], dtype=np.int64)
        scgpt_prototypes = teacher.type_prototypes()[order]
        scgpt_variances = teacher.type_variances()[order]
        scgpt_space_id = teacher.checkpoint_id
        molecular_training_donors.update(str(value) for value in teacher.training_donors.tolist())
        sources.append(args.scgpt_artifact)
        source_roles.append("shared_teacher")
    if args.program_artifact:
        programs = GenePrograms.from_npz(args.program_artifact)
        if not np.array_equal(programs.gene_names.astype(str), reference.gene_ids.astype(str)):
            raise ValueError("gene-program artifact order differs from the RNA reference")
        if analysis_role in _LOCKED_ROLES and donor_id in {
            str(value) for value in programs.training_donors.tolist()
        }:
            raise ValueError(
                "gene-program training donors overlap locked target donor %s" % donor_id
            )
        program_matrix = np.asarray(programs.loadings, dtype=np.float32)
        target_program_scores = np.asarray(
            marker_centroids @ program_matrix,
            dtype=np.float32,
        )
        molecular_training_donors.update(str(value) for value in programs.training_donors.tolist())
        sources.append(args.program_artifact)
        source_roles.append("shared_teacher")

    resolved_sources = tuple(str(Path(value).expanduser().resolve()) for value in sources)
    source_hashes = tuple(_sha256(value) for value in resolved_sources)
    training_batch = HEIRTrainingBatch(
        morphology=torch.from_numpy(np.array(bag.features, dtype=np.float32, copy=True)),
        edge_index=torch.from_numpy(np.array(edges, copy=True)).long(),
        edge_weight=torch.from_numpy(np.array(bag.edge_weight, dtype=np.float32, copy=True)),
        prototype_means=torch.from_numpy(np.array(prototypes.means, copy=True)),
        prototype_variances=torch.from_numpy(np.array(prototypes.variances, copy=True)),
        prototype_types=torch.from_numpy(prototype_types).long(),
        prototype_weights=torch.from_numpy(
            np.array(prototypes.weights, dtype=np.float32, copy=True)
        ),
        target_composition=torch.from_numpy(composition),
        target_pseudobulk=torch.from_numpy(pseudobulk),
        cell_weights=torch.from_numpy(cell_weights),
        marker_centroids=torch.from_numpy(marker_centroids),
        marker_mask=torch.from_numpy(marker_mask),
        program_matrix=(None if program_matrix is None else torch.from_numpy(program_matrix)),
        target_program_scores=(
            None if target_program_scores is None else torch.from_numpy(target_program_scores)
        ),
        segmentation_confidence=torch.from_numpy(confidence.copy()),
        ood_mask=None if ood_mask is None else torch.from_numpy(ood_mask),
        unknown_targets=(None if unknown_targets is None else torch.from_numpy(unknown_targets)),
        domain_labels=(None if domain_labels is None else torch.from_numpy(domain_labels)),
        spot_assignment=(None if spot_assignment is None else torch.from_numpy(spot_assignment)),
        target_spatial_expression=(
            None
            if target_spatial_expression is None
            else torch.from_numpy(target_spatial_expression)
        ),
        scgpt_type_prototypes=(
            None if scgpt_prototypes is None else torch.from_numpy(scgpt_prototypes)
        ),
        scgpt_type_variances=(
            None if scgpt_variances is None else torch.from_numpy(scgpt_variances)
        ),
        sample_id=sample_id,
        bag_id=bag_id,
        donor_id=donor_id,
        block_id=block_id,
        analysis_role=analysis_role,
        latent_space_id=latent_space_id,
        feature_space_id=bag.feature_space_id or "unspecified",
        expression_space_id=EXPRESSION_SPACE_ID,
        scgpt_space_id=scgpt_space_id,
        nucleus_ids=tuple(str(value) for value in bag.nucleus_ids.tolist()),
        type_names=type_names,
        gene_names=tuple(str(value) for value in reference.gene_ids.tolist()),
        prototype_ids=tuple(str(value) for value in prototypes.prototype_ids.tolist()),
        spot_ids=spot_ids,
        source_artifacts=resolved_sources,
        source_sha256=source_hashes,
        source_roles=tuple(source_roles),
        molecular_training_donors=tuple(sorted(molecular_training_donors)),
    )
    stage = (
        TrainingStage.GENERIC_SPATIAL_PRETRAINING
        if target_spatial_expression is not None
        else TrainingStage.PERSONALIZED
    )
    training_batch.validate(stage)
    training_batch.save_npz(args.output)
    excluded = labels[~supported]
    _json(
        {
            "output": str(Path(args.output).expanduser().resolve()),
            "sample_id": sample_id,
            "bag_id": bag_id,
            "cells": int(bag.n_nuclei),
            "types": list(type_names),
            "genes": int(reference.shape[1]),
            "prototypes": int(len(prototypes.prototype_ids)),
            "unregistered": True,
            "excluded_rna_cells": int((~supported).sum()),
            "excluded_rna_types": sorted(set(excluded.tolist())),
            "scgpt_supervision": scgpt_prototypes is not None,
            "latent_space_id": latent_space_id,
            "program_supervision": program_matrix is not None,
            "spatial_pretraining": target_spatial_expression is not None,
            "markers_per_type": int(marker_mask.sum(axis=1).min()),
            "source_sha256": dict(zip(resolved_sources, source_hashes)),
        }
    )
    return 0


def _parse_widths(value: str, name: str) -> Tuple[int, ...]:
    try:
        widths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("%s must be comma-separated integers" % name) from error
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("%s must contain positive widths" % name)
    return widths


def _load_training_batches(paths: Sequence[str]) -> List[HEIRTrainingBatch]:
    batches = [HEIRTrainingBatch.load_npz(path) for path in paths]
    identifiers = [(batch.donor_id, batch.sample_id, batch.bag_id) for batch in batches]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("bag_id must be unique within each sample")
    return batches


def _batch_ontology(
    batches: Sequence[HEIRTrainingBatch],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    first = batches[0]
    type_names = first.type_names or tuple(
        "type_%d" % index for index in range(first.target_composition.shape[0])
    )
    gene_names = first.gene_names or tuple(
        "gene_%d" % index for index in range(first.target_pseudobulk.shape[0])
    )
    signature = (
        first.morphology.shape[1],
        first.prototype_means.shape[1],
        first.target_composition.shape[0],
        first.target_pseudobulk.shape[0],
    )
    for batch in batches:
        current = (
            batch.morphology.shape[1],
            batch.prototype_means.shape[1],
            batch.target_composition.shape[0],
            batch.target_pseudobulk.shape[0],
        )
        if current != signature:
            raise ValueError("training batches use incompatible model dimensions")
        if batch.type_names and batch.type_names != type_names:
            raise ValueError("training batches use different cell-type ontology orders")
        if batch.gene_names and batch.gene_names != gene_names:
            raise ValueError("training batches use different gene orders")
    return type_names, gene_names


def _npz_contract(path: Path) -> Optional[str]:
    """Return a framework-light artifact contract, or ``None`` when unavailable."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if "__contract__" not in archive:
                return None
            return str(np.asarray(archive["__contract__"]).item())
    except (OSError, ValueError, EOFError):
        return None


def _validate_residual_geometry_training_provenance(
    geometry: RNAResidualGeometry,
    batches: Sequence[HEIRTrainingBatch],
    type_names: Sequence[str],
    *,
    unsafe_allow_legacy: bool,
) -> str:
    """Bind RNA residual geometry to the exact molecular inputs in every batch.

    A shared latent-space label and compatible rank are insufficient: the
    geometry must name the exact RNAReference hash used by each PrototypeSet,
    and its latent transform/source identity must agree with those artifacts.
    The unsafe override only permits provenance that is genuinely unavailable;
    an observed mismatch always fails.
    """

    expected_types = tuple(str(value) for value in type_names)
    geometry_types = tuple(str(value) for value in geometry.type_names.tolist())
    if geometry_types != expected_types:
        raise ValueError("RNA residual geometry cell-type order differs from the training batches")

    legacy_gaps: List[str] = []

    def unavailable(message: str) -> None:
        if not unsafe_allow_legacy:
            raise ValueError(
                message + "; rebuild the molecular artifacts or use "
                "--unsafe-allow-legacy-residual-geometry-provenance only for an audited "
                "legacy migration"
            )
        legacy_gaps.append(message)

    if not geometry.source_reference_sha256:
        unavailable("RNA residual geometry lacks source_reference_sha256")
    if geometry.latent_transform_sha256:
        expected_latent_id = "sha256:%s" % geometry.latent_transform_sha256
        if geometry.latent_space_id != expected_latent_id:
            raise ValueError(
                "RNA residual geometry latent_space_id is not bound to its latent transform"
            )
    elif not (
        geometry.latent_space_id.startswith("sha256:")
        and len(geometry.latent_space_id) == len("sha256:") + 64
        and all(
            character in "0123456789abcdef"
            for character in geometry.latent_space_id[len("sha256:") :]
        )
    ):
        unavailable(
            "RNA residual geometry lacks both latent_transform_sha256 and a hash-bound "
            "latent_space_id"
        )

    for batch in batches:
        label = "%s/%s" % (batch.sample_id, batch.bag_id)
        if batch.type_names and tuple(batch.type_names) != expected_types:
            raise ValueError("training batch %s has a different cell-type order" % label)
        if geometry.latent_space_id != batch.latent_space_id:
            raise ValueError(
                "RNA residual geometry latent_space_id differs from training batch %s" % label
            )

        prototype_sources: List[Tuple[Path, str]] = []
        reference_sources: List[Tuple[Path, str]] = []
        for raw_path, recorded_sha256 in zip(batch.source_artifacts, batch.source_sha256):
            source = Path(raw_path).expanduser().resolve()
            contract = _npz_contract(source) if source.is_file() else None
            if contract not in {PrototypeSet.CONTRACT, RNAReference.CONTRACT}:
                continue
            actual_sha256 = _sha256(str(source))
            if actual_sha256 != recorded_sha256:
                raise ValueError(
                    "training batch %s molecular source hash no longer matches %s" % (label, source)
                )
            target = prototype_sources if contract == PrototypeSet.CONTRACT else reference_sources
            target.append((source, actual_sha256))
        if len(prototype_sources) > 1 or len(reference_sources) > 1:
            raise ValueError(
                "training batch %s has ambiguous prototype/reference source provenance" % label
            )
        if not prototype_sources:
            unavailable("training batch %s has no accessible hash-bound PrototypeSet" % label)
        if not reference_sources:
            unavailable("training batch %s has no accessible hash-bound RNAReference" % label)

        prototype = None
        if prototype_sources:
            prototype_path, _ = prototype_sources[0]
            prototype = PrototypeSet.load_npz(prototype_path)
            if prototype.latent_space_id != batch.latent_space_id:
                raise ValueError(
                    "PrototypeSet latent_space_id differs from training batch %s" % label
                )
            if not prototype.source_reference_sha256:
                unavailable("PrototypeSet for training batch %s lacks reference provenance" % label)
            elif (
                geometry.source_reference_sha256
                and prototype.source_reference_sha256 != geometry.source_reference_sha256
            ):
                raise ValueError(
                    "RNA residual geometry source reference differs from PrototypeSet for %s"
                    % label
                )
            if prototype.latent_transform_sha256 != geometry.latent_transform_sha256:
                raise ValueError(
                    "RNA residual geometry latent transform differs from PrototypeSet for %s"
                    % label
                )
            prototype_type_order = tuple(
                dict.fromkeys(str(value) for value in prototype.cell_type_labels.tolist())
            )
            if prototype_type_order != expected_types:
                raise ValueError("PrototypeSet cell-type order differs for %s" % label)
            if tuple(str(value) for value in prototype.prototype_ids.tolist()) != tuple(
                batch.prototype_ids
            ):
                raise ValueError("PrototypeSet identifiers differ from training batch %s" % label)
            type_lookup = {name: index for index, name in enumerate(expected_types)}
            expected_prototype_types = np.asarray(
                [type_lookup[str(value)] for value in prototype.cell_type_labels.tolist()],
                dtype=np.int64,
            )
            prototype_payload_matches = (
                np.array_equal(
                    batch.prototype_means.detach().cpu().numpy(),
                    np.asarray(prototype.means, dtype=np.float32),
                )
                and np.array_equal(
                    batch.prototype_variances.detach().cpu().numpy(),
                    np.asarray(prototype.variances, dtype=np.float32),
                )
                and np.array_equal(
                    batch.prototype_weights.detach().cpu().numpy(),
                    np.asarray(prototype.weights, dtype=np.float32),
                )
                and np.array_equal(
                    batch.prototype_types.detach().cpu().numpy(),
                    expected_prototype_types,
                )
            )
            if not prototype_payload_matches:
                raise ValueError(
                    "training batch %s prototype payload differs from its hash-bound source" % label
                )

        if reference_sources:
            reference_path, reference_sha256 = reference_sources[0]
            reference = RNAReference.load_npz(reference_path)
            if (
                geometry.source_reference_sha256
                and geometry.source_reference_sha256 != reference_sha256
            ):
                raise ValueError(
                    "RNA residual geometry source_reference_sha256 differs from the "
                    "RNAReference used by training batch %s" % label
                )
            if prototype is not None and (
                prototype.source_reference_sha256
                and prototype.source_reference_sha256 != reference_sha256
            ):
                raise ValueError(
                    "PrototypeSet for training batch %s names a different RNAReference" % label
                )
            if reference.latent_space_id != batch.latent_space_id:
                raise ValueError(
                    "RNAReference latent_space_id differs from training batch %s" % label
                )
            if batch.gene_names and tuple(
                str(value) for value in reference.gene_ids.tolist()
            ) != tuple(batch.gene_names):
                raise ValueError("RNAReference gene order differs for training batch %s" % label)

    return (
        "unsafe_legacy_missing_provenance"
        if legacy_gaps
        else "strict_hash_bound_prototype_reference_and_latent_source"
    )


def _load_hierarchy(
    path: Optional[str],
    type_names: Sequence[str],
) -> Tuple[Optional[Tuple[int, ...]], Tuple[str, ...]]:
    if path is None:
        return None, ()
    assignments: Dict[str, str] = {}
    parent_names: List[str] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 2 or not all(value.strip() for value in columns):
                raise ValueError("ontology line %d must be type<TAB>parent" % line_number)
            cell_type, parent = (value.strip() for value in columns)
            if cell_type in assignments:
                raise ValueError("ontology repeats cell type %s" % cell_type)
            assignments[cell_type] = parent
            if parent not in parent_names:
                parent_names.append(parent)
    missing = sorted(set(type_names) - set(assignments))
    extra = sorted(set(assignments) - set(type_names))
    if missing or extra:
        raise ValueError("ontology mismatch missing=%s extra=%s" % (missing, extra))
    parent_lookup = {name: index for index, name in enumerate(parent_names)}
    return tuple(parent_lookup[assignments[name]] for name in type_names), tuple(parent_names)


def _checkpoint_training_donors(
    metadata: object,
    label: str,
    unsafe_allow_missing: bool,
) -> Tuple[str, ...]:
    values: object = None
    if isinstance(metadata, Mapping):
        values = metadata.get("training_donors")
        if values is None and isinstance(metadata.get("training_batches"), list):
            values = [
                row.get("donor_id")
                for row in metadata["training_batches"]
                if isinstance(row, Mapping) and row.get("donor_id")
            ]
    if values is None:
        if unsafe_allow_missing:
            return ()
        raise ValueError("%s lacks required training_donors provenance" % label)
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("%s training_donors provenance is malformed" % label)
    donors = tuple(sorted(set(str(value).strip() for value in values if str(value).strip())))
    if not donors and not unsafe_allow_missing:
        raise ValueError("%s training_donors provenance is empty" % label)
    return donors


def command_train(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    training_batches = _load_training_batches(args.train_batch)
    validation_batches = _load_training_batches(args.validation_batch)
    all_batches = training_batches + validation_batches
    type_names, gene_names = _batch_ontology(all_batches)
    latent_space_ids = {batch.latent_space_id for batch in all_batches}
    if len(latent_space_ids) != 1:
        raise ValueError("training batches use different latent-space identities")
    latent_space_id = next(iter(latent_space_ids))
    if latent_space_id == "unspecified" and not args.unsafe_allow_latent_space_mismatch:
        raise ValueError("training batches lack a verifiable latent_space_id")
    feature_space_ids = {batch.feature_space_id for batch in all_batches}
    if len(feature_space_ids) != 1:
        raise ValueError("training batches use different pathology feature spaces")
    feature_space_id = next(iter(feature_space_ids))
    if feature_space_id == "unspecified" and not args.unsafe_allow_feature_space_mismatch:
        raise ValueError("training batches lack a verifiable feature_space_id")
    expression_space_ids = {batch.expression_space_id for batch in all_batches}
    if len(expression_space_ids) != 1:
        raise ValueError("training batches use different expression normalization spaces")
    expression_space_id = next(iter(expression_space_ids))
    if expression_space_id == "unspecified" and not args.unsafe_allow_expression_space_mismatch:
        raise ValueError("training batches lack a verifiable expression_space_id")
    if args.allow_negative_expression and expression_space_id == EXPRESSION_SPACE_ID:
        raise ValueError("canonical log1p-CPM expression requires a non-negative decoder")
    validation_donors = {batch.donor_id or batch.sample_id for batch in validation_batches}
    molecular_batch_donors = {
        donor for batch in all_batches for donor in batch.molecular_training_donors
    }
    molecular_overlap = sorted(validation_donors & molecular_batch_donors)
    if molecular_overlap and not args.unsafe_allow_molecular_validation_overlap:
        raise ValueError(
            "molecular artifacts were trained on validation donors: %s"
            % ", ".join(molecular_overlap)
        )
    if not args.allow_split_overlap:
        training_sources = {
            digest
            for batch in training_batches
            for digest, role in zip(batch.source_sha256, batch.source_roles)
            if role == "sample_assay"
        }
        validation_sources = {
            digest
            for batch in validation_batches
            for digest, role in zip(batch.source_sha256, batch.source_roles)
            if role == "sample_assay"
        }
        overlap = sorted(training_sources & validation_sources)
        if overlap:
            raise ValueError(
                "training/validation source artifacts overlap SHA-256: %s" % ", ".join(overlap[:3])
            )
        train_ids = {(batch.donor_id, batch.sample_id, batch.bag_id) for batch in training_batches}
        validation_ids = {
            (batch.donor_id, batch.sample_id, batch.bag_id) for batch in validation_batches
        }
        if train_ids & validation_ids:
            raise ValueError("training/validation repeat a sample bag")

    first = all_batches[0]
    num_domains = 0
    training_domain_tensors = [batch.domain_labels for batch in training_batches]
    if any(value is not None for value in training_domain_tensors):
        if any(value is None for value in training_domain_tensors):
            raise ValueError("domain labels must cover every training batch")
        training_domains = sorted(
            {
                int(value)
                for tensor in training_domain_tensors
                if tensor is not None
                for value in tensor[tensor >= 0].unique().tolist()
            }
        )
        if len(training_domains) < 2:
            raise ValueError("domain-adversarial training requires at least two training domains")
        if training_domains != list(range(len(training_domains))):
            raise ValueError("training domain labels must be contiguous from zero")
        validation_domains = {
            int(value)
            for batch in validation_batches
            if batch.domain_labels is not None
            for value in batch.domain_labels[batch.domain_labels >= 0].unique().tolist()
        }
        if not validation_domains.issubset(training_domains):
            raise ValueError("validation domain labels are absent from training")
        num_domains = len(training_domains)
    scgpt_dims = {
        int(batch.scgpt_type_prototypes.shape[1])
        for batch in all_batches
        if batch.scgpt_type_prototypes is not None
    }
    if len(scgpt_dims) > 1:
        raise ValueError("training batches use different scGPT embedding widths")
    scgpt_dim = next(iter(scgpt_dims), 0)
    if scgpt_dim and any(batch.scgpt_type_prototypes is None for batch in all_batches):
        raise ValueError("scGPT supervision must be present in every train/validation batch")
    scgpt_space_ids = {
        batch.scgpt_space_id for batch in all_batches if batch.scgpt_type_prototypes is not None
    }
    if len(scgpt_space_ids) > 1:
        raise ValueError("training batches use different scGPT checkpoint spaces")
    scgpt_space_id = next(iter(scgpt_space_ids), "")

    fine_to_parent, parent_names = _load_hierarchy(args.ontology, type_names)
    rna_vae = None
    rna_encoder = None
    rna_checkpoint_sha256 = None
    rna_training_donors: Tuple[str, ...] = ()
    if args.rna_vae_checkpoint:
        raw_rna_checkpoint = _load_checkpoint(args.rna_vae_checkpoint)
        rna_vae = RNAVAE.from_checkpoint(raw_rna_checkpoint)
        if rna_vae.config.input_dim != len(gene_names):
            raise ValueError("RNA VAE gene width differs from the training batches")
        if rna_vae.config.latent_dim != first.prototype_means.shape[1]:
            raise ValueError("RNA VAE latent width differs from the prototype banks")
        if expression_space_id == EXPRESSION_SPACE_ID and not rna_vae.config.nonnegative_output:
            raise ValueError("canonical log1p-CPM targets require a non-negative RNA decoder")
        rna_metadata = raw_rna_checkpoint.get("metadata")
        rna_training_donors = _checkpoint_training_donors(
            rna_metadata,
            "RNA VAE checkpoint",
            args.unsafe_allow_molecular_validation_overlap,
        )
        overlap = sorted(validation_donors & set(rna_training_donors))
        if overlap and not args.unsafe_allow_molecular_validation_overlap:
            raise ValueError("RNA VAE was trained on validation donors: %s" % ", ".join(overlap))
        if isinstance(rna_metadata, Mapping) and "gene_names" in rna_metadata:
            rna_genes = tuple(str(value) for value in rna_metadata["gene_names"])
            if rna_genes != gene_names:
                raise ValueError("RNA VAE gene order differs from the training batches")
        rna_expression_space_id = (
            str(rna_metadata.get("expression_space_id", ""))
            if isinstance(rna_metadata, Mapping)
            else ""
        )
        if (
            rna_expression_space_id != expression_space_id
            and not args.unsafe_allow_expression_space_mismatch
        ):
            raise ValueError("RNA VAE expression_space_id differs from the weak targets")
        rna_latent_space_id = (
            str(rna_metadata.get("latent_space_id", ""))
            if isinstance(rna_metadata, Mapping)
            else ""
        )
        if rna_latent_space_id != latent_space_id and not args.unsafe_allow_latent_space_mismatch:
            raise ValueError("RNA VAE latent_space_id differs from the prototype batches")
        decoder_widths = rna_vae.config.decoder_hidden_dims or tuple(
            reversed(rna_vae.config.hidden_dims)
        )
        decoder_dropout = rna_vae.config.dropout
        nonnegative_expression = rna_vae.config.nonnegative_output
        rna_checkpoint_sha256 = _sha256(args.rna_vae_checkpoint)
        decoder_only = bool(
            isinstance(rna_metadata, Mapping) and rna_metadata.get("decoder_only", False)
        )
        rna_encoder = None if decoder_only else rna_vae
    else:
        if not args.allow_random_decoder:
            raise ValueError(
                "real training requires --rna-vae-checkpoint; use --allow-random-decoder "
                "only for architecture smoke tests"
            )
        decoder_widths = _parse_widths(args.decoder_hidden_dims, "decoder-hidden-dims")
        decoder_dropout = args.dropout
        nonnegative_expression = not args.allow_negative_expression

    model_config = HEIRConfig(
        morphology_dim=int(first.morphology.shape[1]),
        num_cell_types=int(first.target_composition.shape[0]),
        expression_dim=int(first.target_pseudobulk.shape[0]),
        latent_dim=int(first.prototype_means.shape[1]),
        graph_hidden_dim=args.graph_hidden_dim,
        graph_output_dim=args.graph_output_dim,
        graph_layers=args.graph_layers,
        trunk_hidden_dims=_parse_widths(args.trunk_hidden_dims, "trunk-hidden-dims"),
        decoder_hidden_dims=decoder_widths,
        dropout=decoder_dropout,
        fine_to_parent=fine_to_parent,
        hard_type_routing=args.hard_type_routing,
        abstain_threshold=args.abstain_threshold,
        nonnegative_expression=nonnegative_expression,
        num_domains=num_domains,
        residual_rank=args.residual_rank,
        residual_max_norm=args.residual_max_norm,
        scgpt_embedding_dim=scgpt_dim,
    )
    initial_checkpoint_sha256 = None
    initial_training_donors: Tuple[str, ...] = ()
    if args.initial_heir_checkpoint:
        raw_initial_checkpoint = _load_checkpoint(args.initial_heir_checkpoint)
        model = HEIRModel.from_checkpoint(raw_initial_checkpoint)
        if model.config.to_dict() != model_config.to_dict():
            raise ValueError("initial HEIR checkpoint architecture differs from inferred config")
        initial_metadata = raw_initial_checkpoint.get("metadata")
        if not isinstance(initial_metadata, Mapping):
            raise ValueError("initial HEIR checkpoint lacks ontology/provenance metadata")
        if tuple(str(value) for value in initial_metadata.get("type_names", ())) != type_names:
            raise ValueError("initial HEIR checkpoint cell-type ontology differs")
        if tuple(str(value) for value in initial_metadata.get("gene_names", ())) != gene_names:
            raise ValueError("initial HEIR checkpoint gene order differs")
        if (
            str(initial_metadata.get("latent_space_id", "")) != latent_space_id
            and not args.unsafe_allow_latent_space_mismatch
        ):
            raise ValueError("initial HEIR checkpoint latent_space_id differs")
        if (
            str(initial_metadata.get("feature_space_id", "")) != feature_space_id
            and not args.unsafe_allow_feature_space_mismatch
        ):
            raise ValueError("initial HEIR checkpoint feature_space_id differs")
        if (
            str(initial_metadata.get("expression_space_id", "")) != expression_space_id
            and not args.unsafe_allow_expression_space_mismatch
        ):
            raise ValueError("initial HEIR checkpoint expression_space_id differs")
        initial_training_donors = _checkpoint_training_donors(
            initial_metadata,
            "initial HEIR checkpoint",
            args.unsafe_allow_molecular_validation_overlap,
        )
        overlap = sorted(validation_donors & set(initial_training_donors))
        if overlap and not args.unsafe_allow_molecular_validation_overlap:
            raise ValueError(
                "initial HEIR checkpoint was trained on validation donors: %s" % ", ".join(overlap)
            )
        initial_checkpoint_sha256 = _sha256(args.initial_heir_checkpoint)
    else:
        model = HEIRModel(model_config)
    if rna_vae is not None:
        model.load_rna_decoder(rna_vae, freeze=not args.finetune_rna_decoder)
    residual_geometry_sha256 = None
    residual_geometry_provenance_status = None
    residual_geometry_source_reference_sha256 = None
    residual_geometry_latent_transform_sha256 = None
    if args.residual_geometry is not None:
        geometry_path = Path(args.residual_geometry).expanduser().resolve()
        geometry = RNAResidualGeometry.from_npz(geometry_path)
        if geometry.latent_space_id != latent_space_id:
            raise ValueError("RNA residual geometry latent_space_id differs from training batches")
        residual_geometry_provenance_status = _validate_residual_geometry_training_provenance(
            geometry,
            all_batches,
            type_names,
            unsafe_allow_legacy=args.unsafe_allow_legacy_residual_geometry_provenance,
        )
        geometry_overlap = sorted(validation_donors & set(geometry.training_donors))
        if geometry_overlap and not (
            args.allow_split_overlap or args.unsafe_allow_molecular_validation_overlap
        ):
            raise ValueError(
                "RNA residual geometry was fitted on validation donors: %s"
                % ", ".join(geometry_overlap)
            )
        if geometry.residual_type_basis.shape[2] != model.config.residual_rank:
            raise ValueError("RNA residual geometry rank differs from the HEIR residual rank")
        basis, maximums = geometry.model_parameters(type_names)
        model.configure_residual_geometry(
            torch.from_numpy(basis),
            torch.from_numpy(maximums),
            freeze_basis=not args.finetune_residual_basis,
        )
        residual_geometry_sha256 = _sha256(str(geometry_path))
        residual_geometry_source_reference_sha256 = geometry.source_reference_sha256
        residual_geometry_latent_transform_sha256 = geometry.latent_transform_sha256
    optimization = OptimizationConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        adapter_learning_rate=args.adapter_learning_rate,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        gradient_clip_norm=args.gradient_clip_norm,
        bag_size=args.bag_size or max(len(batch.morphology) for batch in all_batches),
        reference_batch_size=args.reference_batch_size
        or max(len(batch.prototype_means) for batch in all_batches),
        maximum_sample_cells=args.maximum_sample_cells,
        early_stopping_patience=args.early_stopping_patience,
        mixed_precision=(
            resolve_device(args.device).type == "cuda"
            if args.mixed_precision is None
            else args.mixed_precision
        ),
    )
    stage = TrainingStage(args.stage)
    trainer = HEIRTrainer(
        model,
        stage,
        optimization,
        LossWeightConfig(),
        rna_encoder=rna_encoder,
        seed=args.seed,
        device=args.device,
        allow_split_overlap=args.allow_split_overlap,
        uot_unknown_mass=args.uot_unknown_mass,
        uot_unknown_mass_mode=args.uot_unknown_mass_mode,
    )
    result = trainer.fit(training_batches, validation_batches)
    destination = Path(args.output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint()
    checkpoint["metadata"] = {
        "schema": "heir.trained_model.v1",
        "type_names": list(type_names),
        "parent_type_names": list(parent_names),
        "gene_names": list(gene_names),
        "training_stage": stage.value,
        "uot_unknown_mass": float(args.uot_unknown_mass),
        "uot_unknown_mass_mode": str(args.uot_unknown_mass_mode),
        "latent_space_id": latent_space_id,
        "feature_space_id": feature_space_id,
        "expression_space_id": expression_space_id,
        "scgpt_space_id": scgpt_space_id,
        "training_donors": sorted(
            {batch.donor_id or batch.sample_id for batch in training_batches}
        ),
        "seed": args.seed,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "rna_vae_checkpoint": (
            None
            if args.rna_vae_checkpoint is None
            else str(Path(args.rna_vae_checkpoint).expanduser().resolve())
        ),
        "rna_vae_sha256": rna_checkpoint_sha256,
        "rna_vae_training_donors": list(rna_training_donors),
        "initial_heir_checkpoint": (
            None
            if args.initial_heir_checkpoint is None
            else str(Path(args.initial_heir_checkpoint).expanduser().resolve())
        ),
        "initial_heir_sha256": initial_checkpoint_sha256,
        "initial_heir_training_donors": list(initial_training_donors),
        "residual_geometry": (
            None
            if args.residual_geometry is None
            else str(Path(args.residual_geometry).expanduser().resolve())
        ),
        "residual_geometry_sha256": residual_geometry_sha256,
        "residual_geometry_provenance_status": residual_geometry_provenance_status,
        "residual_geometry_source_reference_sha256": (residual_geometry_source_reference_sha256),
        "residual_geometry_latent_transform_sha256": (residual_geometry_latent_transform_sha256),
        "residual_basis_trainable": bool(args.finetune_residual_basis),
        "training_batches": [
            {
                "sample_id": batch.sample_id,
                "bag_id": batch.bag_id,
                "donor_id": batch.donor_id,
                "block_id": batch.block_id,
                "analysis_role": batch.analysis_role,
                "source_artifacts": list(batch.source_artifacts),
                "source_sha256": list(batch.source_sha256),
                "source_roles": list(batch.source_roles),
            }
            for batch in training_batches
        ],
        "validation_batches": [
            {
                "sample_id": batch.sample_id,
                "bag_id": batch.bag_id,
                "donor_id": batch.donor_id,
                "block_id": batch.block_id,
                "analysis_role": batch.analysis_role,
                "source_artifacts": list(batch.source_artifacts),
                "source_sha256": list(batch.source_sha256),
                "source_roles": list(batch.source_roles),
            }
            for batch in validation_batches
        ],
    }
    checkpoint_path = destination / "heir.pt"
    history_path = destination / "history.json"
    _atomic_torch_save(checkpoint, checkpoint_path)
    atomic_json_dump(
        {
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "history": list(result.history),
        },
        history_path,
    )
    _json(
        {
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
        }
    )
    return 0


_REFINEMENT_VIEW_SCHEMA = "heir.refinement_views.v2"
_LEGACY_REFINEMENT_VIEW_SCHEMA = "heir.refinement_views.v1"
_REFINEMENT_VIEW_CONSTRUCTION = "one_encoder_scale_block_plus_shared_explicit_morphology"


def _ordered_identity_sha256(values: Sequence[object]) -> str:
    encoded = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _refinement_view_metadata(
    archive: np.lib.npyio.NpzFile,
    *,
    checkpoint_sha256: str,
    batch_sha256: str,
    batch: HEIRTrainingBatch,
) -> Mapping[str, object]:
    if "metadata_json" not in archive:
        raise ValueError("view artifact lacks versioned metadata_json provenance")
    raw_metadata = np.asarray(archive["metadata_json"])
    if raw_metadata.ndim != 0:
        raise ValueError("view artifact metadata_json must be a scalar")
    try:
        metadata = json.loads(str(raw_metadata.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("view artifact metadata_json is malformed") from error
    if not isinstance(metadata, Mapping):
        raise ValueError("view artifact metadata_json must contain an object")
    schema = str(metadata.get("schema", ""))
    if schema not in {_REFINEMENT_VIEW_SCHEMA, _LEGACY_REFINEMENT_VIEW_SCHEMA}:
        raise ValueError("view artifact has an unsupported provenance schema")

    # v1 already committed the complete checkpoint and batch bytes. Accept it
    # only through this explicit migration branch after both digests match;
    # the bound batch bytes supply the identities added as first-class v2
    # fields. No unversioned or partially versioned artifact is migrated.
    for field, expected in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("batch_sha256", batch_sha256),
    ):
        if str(metadata.get(field, "")) != expected:
            raise ValueError("view artifact %s differs from the refinement input" % field)
    if schema == _LEGACY_REFINEMENT_VIEW_SCHEMA:
        return metadata

    expected_metadata: Mapping[str, object] = {
        "batch_contract": batch.CONTRACT,
        "batch_contract_version": batch.CONTRACT_VERSION,
        "batch_source_sha256": list(batch.source_sha256),
        "batch_source_roles": list(batch.source_roles),
        "sample_id": batch.sample_id,
        "donor_id": batch.donor_id,
        "bag_id": batch.bag_id,
        "block_id": batch.block_id,
        "feature_space_id": batch.feature_space_id,
        "latent_space_id": batch.latent_space_id,
        "type_names": list(batch.type_names),
        "type_ontology_sha256": _ordered_identity_sha256(batch.type_names),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError("view artifact %s differs from the refinement batch" % field)
    return metadata


def _load_refinement_views(
    specifications: Sequence[str],
    batches: Sequence[HEIRTrainingBatch],
    *,
    checkpoint_path: str,
    batch_paths: Sequence[str],
) -> Dict[str, np.ndarray]:
    if len(batch_paths) != len(batches):
        raise ValueError("refinement batch paths must align to loaded training batches")
    checkpoint_sha256 = _sha256(checkpoint_path)
    expected = {
        "%s::%s::%s" % (batch.donor_id, batch.sample_id, batch.bag_id): (
            batch,
            _sha256(path),
        )
        for batch, path in zip(batches, batch_paths)
    }
    result: Dict[str, np.ndarray] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("view predictions must use DONOR::SAMPLE::BAG=artifact.npz")
        key, raw_path = specification.split("=", 1)
        if key not in expected:
            raise ValueError("view prediction key is not a training batch: %s" % key)
        if key in result:
            raise ValueError("view prediction key is repeated: %s" % key)
        batch, batch_sha256 = expected[key]
        if not batch.nucleus_ids:
            raise ValueError("view alignment requires nucleus_ids in every training batch")
        if not batch.source_sha256:
            raise ValueError("view alignment requires source SHA-256 provenance in every batch")
        with np.load(raw_path, allow_pickle=False) as archive:
            required = {
                "nucleus_ids",
                "view_predictions",
                "view_ids",
                "view_source_sha256",
                "metadata_json",
            }
            missing_keys = sorted(required - set(archive.files))
            if missing_keys:
                raise ValueError("view artifact is missing: %s" % ", ".join(missing_keys))
            view_metadata = _refinement_view_metadata(
                archive,
                checkpoint_sha256=checkpoint_sha256,
                batch_sha256=batch_sha256,
                batch=batch,
            )
            nucleus_ids = [str(value) for value in archive["nucleus_ids"].tolist()]
            predictions = np.array(archive["view_predictions"], copy=True)
            view_ids = [str(value) for value in archive["view_ids"].tolist()]
            view_hashes = [str(value) for value in archive["view_source_sha256"].tolist()]
        if len(set(nucleus_ids)) != len(nucleus_ids):
            raise ValueError("view artifact nucleus_ids must be unique")
        lookup = {value: index for index, value in enumerate(nucleus_ids)}
        missing = [value for value in batch.nucleus_ids if value not in lookup]
        extras = sorted(set(nucleus_ids) - set(batch.nucleus_ids))
        if missing or extras:
            raise ValueError(
                "view artifact nuclei differ for %s (missing=%d, extra=%d)"
                % (key, len(missing), len(extras))
            )
        if predictions.ndim not in (2, 3) or predictions.shape[1] != len(nucleus_ids):
            raise ValueError("view_predictions must have shape (views, nuclei[, types])")
        if predictions.ndim == 3 and predictions.shape[2] != len(batch.type_names):
            raise ValueError("view_predictions type axis differs from the bound ontology")
        if not np.issubdtype(predictions.dtype, np.number) or not np.isfinite(predictions).all():
            raise ValueError("view_predictions must be finite and numeric")
        if (
            len(view_ids) != predictions.shape[0]
            or len(set(view_ids)) != len(view_ids)
            or any(not value.strip() for value in view_ids)
        ):
            raise ValueError("view_ids must be unique and align to independent views")
        if len(view_hashes) != predictions.shape[0] or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in view_hashes
        ):
            raise ValueError("view_source_sha256 must align and contain SHA-256 digests")
        if view_metadata.get("view_construction") != _REFINEMENT_VIEW_CONSTRUCTION:
            raise ValueError("view artifact uses an unsupported view construction")
        try:
            blocks = int(view_metadata["encoder_blocks"])
            block_width = int(view_metadata["encoder_block_width"])
            tail = int(view_metadata["shared_tail_features"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("view artifact construction metadata is incomplete") from error
        width = int(batch.morphology.shape[1])
        if (
            blocks < 2
            or blocks != predictions.shape[0]
            or tail < 0
            or tail >= width
            or block_width <= 0
            or blocks * block_width + tail != width
        ):
            raise ValueError("view artifact construction metadata differs from the batch")
        expected_view_ids = ["encoder_scale_%d" % index for index in range(blocks)]
        if view_ids != expected_view_ids:
            raise ValueError("view_ids differ from the bound scale-held-out construction")
        expected_hashes = []
        for block_index in range(blocks):
            start = block_index * block_width
            stop = start + block_width
            digest = hashlib.sha256()
            digest.update(checkpoint_sha256.encode("ascii"))
            digest.update(batch_sha256.encode("ascii"))
            digest.update(("encoder_block_%d:%d:%d" % (block_index, start, stop)).encode("ascii"))
            expected_hashes.append(digest.hexdigest())
        if view_hashes != expected_hashes:
            raise ValueError("view_source_sha256 does not bind the selected checkpoint and batch")
        for left in range(predictions.shape[0]):
            for right in range(left + 1, predictions.shape[0]):
                if np.array_equal(predictions[left], predictions[right]):
                    raise ValueError("independent view predictions cannot be exact duplicates")
        order = np.asarray([lookup[value] for value in batch.nucleus_ids], dtype=np.int64)
        result[key] = predictions[:, order]
    return result


def command_refine(args: argparse.Namespace) -> int:
    """Run constrained broad-to-fine generalized EM with an EMA teacher."""

    set_seed(args.seed)
    raw_checkpoint = _load_checkpoint(args.checkpoint)
    model = HEIRModel.from_checkpoint(raw_checkpoint)
    if args.maximum_rounds <= 0:
        raise ValueError("refine requires maximum-rounds > 0")
    if args.broad_refinement_rounds > 0 and model.config.fine_to_parent is None:
        raise ValueError(
            "broad refinement requires a hierarchical checkpoint; set "
            "--broad-refinement-rounds 0 only for an explicit fine-only ablation"
        )
    metadata = raw_checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("HEIR checkpoint lacks ontology/provenance metadata")
    training_batches = _load_training_batches(args.train_batch)
    validation_batches = _load_training_batches(args.validation_batch)
    all_batches = training_batches + validation_batches
    type_names, gene_names = _batch_ontology(all_batches)
    if tuple(str(value) for value in metadata.get("type_names", ())) != type_names:
        raise ValueError("checkpoint and refinement batches use different cell types")
    if tuple(str(value) for value in metadata.get("gene_names", ())) != gene_names:
        raise ValueError("checkpoint and refinement batches use different genes")
    latent_ids = {batch.latent_space_id for batch in all_batches}
    if latent_ids != {str(metadata.get("latent_space_id", ""))}:
        raise ValueError("checkpoint and refinement batches use different latent spaces")
    feature_ids = {batch.feature_space_id for batch in all_batches}
    if feature_ids != {str(metadata.get("feature_space_id", ""))}:
        raise ValueError("checkpoint and refinement batches use different pathology feature spaces")
    expression_ids = {batch.expression_space_id for batch in all_batches}
    if expression_ids != {str(metadata.get("expression_space_id", ""))}:
        raise ValueError("checkpoint and refinement batches use different expression spaces")
    scgpt_ids = {batch.scgpt_space_id for batch in all_batches}
    if scgpt_ids != {str(metadata.get("scgpt_space_id", ""))}:
        raise ValueError("checkpoint and refinement batches use different scGPT spaces")
    first = all_batches[0]
    expected_dimensions = (
        int(first.morphology.shape[1]),
        int(first.target_composition.shape[0]),
        int(first.target_pseudobulk.shape[0]),
        int(first.prototype_means.shape[1]),
    )
    model_dimensions = (
        model.config.morphology_dim,
        model.config.num_cell_types,
        model.config.expression_dim,
        model.config.latent_dim,
    )
    if expected_dimensions != model_dimensions:
        raise ValueError("checkpoint architecture differs from refinement batches")
    validation_donors = {batch.donor_id for batch in validation_batches}
    molecular_batch_donors = {
        donor for batch in all_batches for donor in batch.molecular_training_donors
    }
    molecular_overlap = sorted(validation_donors & molecular_batch_donors)
    if molecular_overlap and not args.unsafe_allow_molecular_validation_overlap:
        raise ValueError(
            "molecular artifacts were trained on refinement-validation donors: %s"
            % ", ".join(molecular_overlap)
        )
    if not args.allow_split_overlap:
        training_sources = {
            digest
            for batch in training_batches
            for digest, role in zip(batch.source_sha256, batch.source_roles)
            if role == "sample_assay"
        }
        validation_sources = {
            digest
            for batch in validation_batches
            for digest, role in zip(batch.source_sha256, batch.source_roles)
            if role == "sample_assay"
        }
        overlap = sorted(training_sources & validation_sources)
        if overlap:
            raise ValueError(
                "refinement train/validation source artifacts overlap SHA-256: %s"
                % ", ".join(overlap[:3])
            )
    checkpoint_donors = set(_checkpoint_training_donors(metadata, "HEIR checkpoint", False))
    overlap = sorted(validation_donors & checkpoint_donors)
    if overlap and not args.allow_split_overlap:
        raise ValueError(
            "refinement checkpoint was trained on validation donors: %s" % ", ".join(overlap)
        )

    rna_encoder = None
    if args.rna_vae_checkpoint:
        rna_checkpoint = _load_checkpoint(args.rna_vae_checkpoint)
        rna_model = RNAVAE.from_checkpoint(rna_checkpoint)
        if (
            rna_model.config.input_dim != model.config.expression_dim
            or rna_model.config.latent_dim != model.config.latent_dim
        ):
            raise ValueError("RNA VAE is incompatible with the refined HEIR model")
        rna_metadata = rna_checkpoint.get("metadata")
        if not isinstance(rna_metadata, Mapping):
            raise ValueError("RNA VAE checkpoint lacks provenance metadata")
        if str(rna_metadata.get("latent_space_id", "")) != next(iter(latent_ids)):
            raise ValueError("RNA VAE and refinement batches use different latent spaces")
        if str(rna_metadata.get("expression_space_id", "")) != next(iter(expression_ids)):
            raise ValueError("RNA VAE and refinement batches use different expression spaces")
        rna_donors = set(_checkpoint_training_donors(rna_metadata, "RNA VAE checkpoint", False))
        rna_overlap = sorted(validation_donors & rna_donors)
        if rna_overlap and not args.unsafe_allow_molecular_validation_overlap:
            raise ValueError(
                "RNA VAE was trained on refinement-validation donors: %s" % ", ".join(rna_overlap)
            )
        if not bool(rna_metadata.get("decoder_only", False)):
            rna_encoder = rna_model

    optimization = OptimizationConfig(
        epochs=args.epochs_per_round,
        learning_rate=args.learning_rate,
        adapter_learning_rate=args.adapter_learning_rate,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        gradient_clip_norm=args.gradient_clip_norm,
        bag_size=args.bag_size or max(len(batch.morphology) for batch in all_batches),
        reference_batch_size=args.reference_batch_size
        or max(len(batch.prototype_means) for batch in all_batches),
        maximum_sample_cells=args.maximum_sample_cells,
        early_stopping_patience=args.early_stopping_patience,
        mixed_precision=(
            resolve_device(args.device).type == "cuda"
            if args.mixed_precision is None
            else args.mixed_precision
        ),
    )
    refinement_config = RefinementConfig(
        maximum_rounds=args.maximum_rounds,
        min_probability=args.min_probability,
        max_normalized_entropy=args.max_normalized_entropy,
        teacher_ema=args.teacher_ema,
        prior_old_weight=args.prior_old_weight,
        minimum_segmentation_confidence=args.minimum_segmentation_confidence,
        require_view_agreement=not args.allow_no_view_agreement,
        maximum_prior_total_variation=args.maximum_prior_total_variation,
        max_anchors_per_class=args.max_anchors_per_class,
        stable_rounds_required=args.stable_rounds_required,
        maximum_validation_loss_degradation=args.maximum_validation_loss_degradation,
        objective_relative_stability_tolerance=(args.objective_relative_stability_tolerance),
        objective_stability_tolerance=args.objective_stability_tolerance,
        round_selection_mode=args.round_selection_mode,
        maximum_spatial_score_degradation=args.maximum_spatial_score_degradation,
        broad_refinement_rounds=args.broad_refinement_rounds,
    )
    views = _load_refinement_views(
        args.view_predictions or [],
        training_batches,
        checkpoint_path=args.checkpoint,
        batch_paths=args.train_batch,
    )

    def trainer_factory() -> HEIRTrainer:
        return HEIRTrainer(
            model,
            TrainingStage.REFINEMENT,
            optimization,
            LossWeightConfig(),
            rna_encoder=rna_encoder,
            seed=args.seed,
            device=args.device,
            allow_split_overlap=args.allow_split_overlap,
            uot_unknown_mass=args.uot_unknown_mass,
            uot_unknown_mass_mode=args.uot_unknown_mass_mode,
        )

    result = IterativeRefiner(
        trainer_factory,
        refinement_config,
        device=args.device,
    ).fit(
        training_batches,
        validation_batches,
        view_probabilities=views or None,
    )
    destination = Path(args.output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    round_rows = []
    for item in result.rounds:
        row = asdict(item)
        if not np.isfinite(row["objective_relative_change"]):
            row["objective_relative_change"] = None
        round_rows.append(row)
    refined_metadata = dict(metadata)
    refinement_training_donors = sorted(
        {batch.donor_id for batch in training_batches if batch.donor_id}
    )
    all_training_donors = sorted(
        set(str(value) for value in metadata.get("training_donors", ()))
        | set(refinement_training_donors)
    )
    view_artifacts = []
    for specification in args.view_predictions or []:
        key, raw_path = specification.split("=", 1)
        view_artifacts.append(
            {
                "key": key,
                "path": str(Path(raw_path).expanduser().resolve()),
                "sha256": _sha256(raw_path),
            }
        )
    refined_metadata.update(
        {
            "schema": "heir.refined_model.v1",
            "uot_unknown_mass": float(args.uot_unknown_mass),
            "uot_unknown_mass_mode": str(args.uot_unknown_mass_mode),
            "parent_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "parent_checkpoint_sha256": _sha256(args.checkpoint),
            "refinement_round": result.selected_round,
            "refinement_rounds_executed": len(result.rounds),
            "refinement_round_zero_validation_loss": result.round_zero_validation_loss,
            "refinement_stopped_reason": result.stopped_reason,
            "refinement_rounds": round_rows,
            "training_donors": all_training_donors,
            "refinement_training_donors": refinement_training_donors,
            "refinement_training_batches": [
                {
                    "sample_id": batch.sample_id,
                    "bag_id": batch.bag_id,
                    "donor_id": batch.donor_id,
                    "block_id": batch.block_id,
                    "source_sha256": list(batch.source_sha256),
                    "source_roles": list(batch.source_roles),
                }
                for batch in training_batches
            ],
            "refinement_validation_batches": [
                {
                    "sample_id": batch.sample_id,
                    "bag_id": batch.bag_id,
                    "donor_id": batch.donor_id,
                    "block_id": batch.block_id,
                    "source_sha256": list(batch.source_sha256),
                    "source_roles": list(batch.source_roles),
                }
                for batch in validation_batches
            ],
            "refinement_view_artifacts": view_artifacts,
            "refinement_rna_vae_checkpoint": (
                None
                if args.rna_vae_checkpoint is None
                else str(Path(args.rna_vae_checkpoint).expanduser().resolve())
            ),
            "refinement_rna_vae_sha256": (
                None if args.rna_vae_checkpoint is None else _sha256(args.rna_vae_checkpoint)
            ),
        }
    )
    refined_checkpoint = model.checkpoint()
    refined_checkpoint["metadata"] = refined_metadata
    checkpoint_path = destination / "heir_refined.pt"
    _atomic_torch_save(refined_checkpoint, checkpoint_path)
    round_checkpoint_outputs: Dict[str, str] = {}
    if args.save_round_checkpoints:
        selected_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        for item, state in zip(result.rounds, result.round_state_dicts):
            model.load_state_dict(state)
            round_metadata = dict(refined_metadata)
            round_metadata.update(
                {
                    "schema": "heir.refined_round_model.v1",
                    "refinement_round": item.round_id,
                    "refinement_round_committed": item.committed,
                    "selected_by_parent_run": item.round_id == result.selected_round,
                }
            )
            round_checkpoint = model.checkpoint()
            round_checkpoint["metadata"] = round_metadata
            round_directory = destination / ("round_%d" % item.round_id)
            round_directory.mkdir(parents=True, exist_ok=True)
            round_path = round_directory / "heir_refined.pt"
            _atomic_torch_save(round_checkpoint, round_path)
            round_checkpoint_outputs[str(item.round_id)] = str(round_path)
        model.load_state_dict(selected_state)

    prototype_directory = destination / "prototypes"
    prototype_directory.mkdir(parents=True, exist_ok=True)
    prototype_outputs: Dict[str, str] = {}
    seen = set()
    for batch in training_batches:
        key = "%s::%s" % (batch.donor_id, batch.sample_id)
        if key in seen:
            continue
        seen.add(key)
        if key not in result.sample_prototype_weights:
            raise RuntimeError("refinement did not return prototype weights for %s" % key)
        type_labels = np.asarray(batch.type_names, dtype=np.dtype("U"))[
            batch.prototype_types.detach().cpu().numpy()
        ]
        refined_prototypes = PrototypeSet(
            prototype_ids=np.asarray(batch.prototype_ids, dtype=np.dtype("U")),
            sample_ids=np.full(
                len(batch.prototype_ids),
                batch.sample_id,
                dtype=np.dtype("U%d" % max(1, len(str(batch.sample_id)))),
            ),
            cell_type_labels=type_labels,
            means=batch.prototype_means.detach().cpu().numpy(),
            variances=batch.prototype_variances.detach().cpu().numpy(),
            weights=result.sample_prototype_weights[key],
            latent_space_id=batch.latent_space_id,
            donor_id=batch.donor_id,
            block_id=batch.block_id,
            latent_training_donors=batch.molecular_training_donors,
        )
        safe_key = key.replace("/", "_").replace("::", "__")
        prototype_path = prototype_directory / (safe_key + ".npz")
        refined_prototypes.save_npz(prototype_path)
        prototype_outputs[key] = str(prototype_path)
    audit_path = destination / "refinement.json"
    atomic_json_dump(
        {
            "rounds": round_rows,
            "round_zero_validation_loss": result.round_zero_validation_loss,
            "selected_round": result.selected_round,
            "stopped_reason": result.stopped_reason,
            "prototype_artifacts": prototype_outputs,
            "round_checkpoints": round_checkpoint_outputs,
        },
        audit_path,
    )
    _json(
        {
            "checkpoint": str(checkpoint_path),
            "audit": str(audit_path),
            "rounds": len(result.rounds),
            "round_zero_validation_loss": result.round_zero_validation_loss,
            "selected_round": result.selected_round,
            "stopped_reason": result.stopped_reason,
            "prototype_artifacts": prototype_outputs,
            "round_checkpoints": round_checkpoint_outputs,
        }
    )
    return 0


def _wrong_donor_ontology_intersection(
    prototypes: PrototypeSet,
    target_type_names: Sequence[str],
) -> Tuple[PrototypeSet, Dict[str, object]]:
    """Filter a wrong-donor bank to the checkpoint ontology in memory.

    This helper is called only from the explicitly requested wrong-donor
    branch in :func:`command_predict`.  The returned bank keeps source row
    order and artifact metadata; the caller remains responsible for binding
    the prediction to the hash of the original, unfiltered PrototypeSet.
    """

    if len(np.unique(prototypes.sample_ids)) != 1:
        raise ValueError(
            "wrong-donor ontology intersection requires one sample-specific prototype bank"
        )
    checkpoint_types = tuple(str(value) for value in target_type_names)
    checkpoint_type_set = set(checkpoint_types)
    original_types = tuple(
        dict.fromkeys(str(value) for value in prototypes.cell_type_labels.tolist())
    )
    original_type_set = set(original_types)
    retained_types = tuple(value for value in checkpoint_types if value in original_type_set)
    omitted_types = tuple(value for value in original_types if value not in checkpoint_type_set)
    selected = np.asarray(
        [str(value) in checkpoint_type_set for value in prototypes.cell_type_labels.tolist()],
        dtype=bool,
    )
    retained_prototype_count = int(selected.sum())
    if len(retained_types) < 2 or retained_prototype_count < 2:
        raise ValueError(
            "wrong-donor ontology intersection must retain at least two cell types "
            "and two prototypes"
        )
    weights = np.asarray(prototypes.weights[selected], dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("wrong-donor ontology intersection has no positive prototype mass")
    weights /= total
    filtered = PrototypeSet(
        prototype_ids=np.asarray(prototypes.prototype_ids[selected]),
        sample_ids=np.asarray(prototypes.sample_ids[selected]),
        cell_type_labels=np.asarray(prototypes.cell_type_labels[selected]),
        means=np.asarray(prototypes.means[selected]),
        variances=np.asarray(prototypes.variances[selected]),
        weights=weights,
        n_cells=np.asarray(prototypes.n_cells[selected]),
        latent_space_id=prototypes.latent_space_id,
        donor_id=prototypes.donor_id,
        block_id=prototypes.block_id,
        source_reference_sha256=prototypes.source_reference_sha256,
        latent_training_donors=prototypes.latent_training_donors,
        latent_transform_sha256=prototypes.latent_transform_sha256,
    )
    return filtered, {
        "policy": "target_checkpoint_ontology_intersection_v1",
        "minimum_retained_type_count": 2,
        "minimum_retained_prototype_count": 2,
        "original_prototype_count": int(len(prototypes.prototype_ids)),
        "retained_prototype_count": int(len(filtered.prototype_ids)),
        "omitted_prototype_count": int(len(prototypes.prototype_ids) - len(filtered.prototype_ids)),
        "original_type_count": len(original_types),
        "retained_type_count": len(retained_types),
        "omitted_type_count": len(omitted_types),
        "original_type_names": list(original_types),
        "retained_type_names": list(retained_types),
        "omitted_type_names": list(omitted_types),
        "weights_renormalized": True,
    }


def _canonical_payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shuffle_control_transform(
    control: str,
    seed: int,
    nuclei: int,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Return the exact deterministic shuffle map and its auditable recipe."""

    algorithms = {
        "image_shuffle": (
            "apply default_rng(seed).permutation(n_nuclei) to histology feature rows"
        ),
        "graph_shuffle": (
            "apply default_rng(seed).permutation(n_nuclei) to graph edge endpoint indices"
        ),
    }
    if control not in algorithms:
        raise ValueError("unsupported shuffle control %s" % control)
    permutation = np.asarray(
        np.random.default_rng(seed).permutation(nuclei),
        dtype="<i8",
    )
    map_sha256 = hashlib.sha256(permutation.tobytes(order="C")).hexdigest()
    recipe: Dict[str, object] = {
        "schema": "heir.inference_control_transform.v1",
        "control": control,
        "seed": int(seed),
        "random_generator": "numpy.random.default_rng",
        "algorithm": algorithms[control],
        "nuclei": int(nuclei),
        "map_encoding": "little-endian-int64-c-order",
        "expected_transform_map_sha256": map_sha256,
    }
    return permutation, {
        **recipe,
        "recipe_sha256": _canonical_payload_sha256(recipe),
        "map_sha256": map_sha256,
    }


def command_predict(args: argparse.Namespace) -> int:
    if not 0.0 <= args.artifact_threshold <= 1.0:
        raise ValueError("artifact-threshold must lie in [0, 1]")
    if not 0.0 <= args.probability_threshold <= 1.0:
        raise ValueError("probability-threshold must lie in [0, 1]")
    if args.telemetry_output:
        telemetry_destination = Path(args.telemetry_output).expanduser().resolve()
        reserved_paths = {
            Path(value).expanduser().resolve()
            for value in (
                args.output,
                args.checkpoint,
                args.histology,
                args.prototypes,
                args.genes,
                args.program_artifact,
                args.ood_artifact,
            )
            if value
        }
        if telemetry_destination in reserved_paths:
            raise ValueError("--telemetry-output must not overwrite a prediction input/output")
    checkpoint = _load_checkpoint(args.checkpoint)
    model = HEIRModel.from_checkpoint(checkpoint)
    if args.prototype_only:
        if model.config.legacy_unrestricted_residual or model.residual_gate_head is None:
            raise ValueError("prototype-only control requires a restricted-residual checkpoint")
        with torch.no_grad():
            model.residual_gate_head.weight.zero_()
            model.residual_gate_head.bias.fill_(-100.0)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint lacks HEIR ontology metadata")
    raw_type_names = metadata.get("type_names")
    raw_gene_names = metadata.get("gene_names")
    if not isinstance(raw_type_names, (list, tuple)) or not isinstance(
        raw_gene_names, (list, tuple)
    ):
        raise ValueError("checkpoint ontology metadata is malformed")
    type_names = tuple(str(value) for value in raw_type_names)
    checkpoint_genes = tuple(str(value) for value in raw_gene_names)
    checkpoint_latent_space_id = str(metadata.get("latent_space_id", ""))
    checkpoint_feature_space_id = str(metadata.get("feature_space_id", ""))
    checkpoint_expression_space_id = str(metadata.get("expression_space_id", ""))
    if not checkpoint_expression_space_id:
        raise ValueError("checkpoint lacks expression_space_id provenance")
    requested_genes = _gene_list(args.genes)
    assert requested_genes is not None
    gene_names = tuple(requested_genes)
    if gene_names != checkpoint_genes:
        raise ValueError("gene list/order does not match the trained checkpoint")
    if len(type_names) != model.config.num_cell_types:
        raise ValueError("checkpoint type ontology does not match its model config")
    if len(gene_names) != model.config.expression_dim:
        raise ValueError("checkpoint gene ontology does not match its model config")

    bag = HistologyBag.load_npz(args.histology)
    prototypes = PrototypeSet.load_npz(args.prototypes)
    original_prototype_sha256 = _sha256(args.prototypes)
    inference_prototypes = prototypes
    prototype_filter: Optional[Dict[str, object]] = None
    missing_bag_provenance = [
        name
        for name, value in (
            ("sample_id", bag.sample_id),
            ("donor_id", bag.donor_id),
            ("block_id", bag.block_id),
            ("feature_space_id", bag.feature_space_id),
            ("histology_source_sha256", bag.histology_source_sha256),
            ("nuclei_source_sha256", bag.nuclei_source_sha256),
            ("feature_source_sha256", bag.feature_source_sha256),
        )
        if not value
    ]
    if missing_bag_provenance and not args.unsafe_allow_missing_histology_provenance:
        raise ValueError(
            "HistologyBag lacks required provenance: %s" % ", ".join(missing_bag_provenance)
        )
    if bag.donor_id and bag.donor_id != args.donor_id:
        raise ValueError("HistologyBag donor differs from --donor-id")
    if args.sample_id and bag.sample_id and args.sample_id != bag.sample_id:
        raise ValueError("--sample-id differs from HistologyBag sample provenance")
    resolved_sample_id = args.sample_id or bag.sample_id or None
    if (
        not checkpoint_feature_space_id
        or not bag.feature_space_id
        or bag.feature_space_id != checkpoint_feature_space_id
    ) and not args.unsafe_allow_feature_space_mismatch:
        raise ValueError("HistologyBag feature_space_id does not match the trained checkpoint")
    if args.wrong_donor_control:
        if not prototypes.donor_id:
            raise ValueError("wrong-donor control requires PrototypeSet donor provenance")
        if prototypes.donor_id == args.donor_id:
            raise ValueError("wrong-donor control requires a non-matched PrototypeSet donor")
    if not prototypes.donor_id or not prototypes.block_id:
        if not args.unsafe_allow_missing_prototype_provenance:
            raise ValueError("PrototypeSet lacks donor/block provenance")
    elif prototypes.donor_id != args.donor_id and not args.wrong_donor_control:
        raise ValueError("PrototypeSet donor differs from --donor-id")
    elif bag.block_id and prototypes.block_id != bag.block_id and not args.wrong_donor_control:
        raise ValueError("PrototypeSet block differs from the HistologyBag")
    if args.wrong_donor_control:
        inference_prototypes, prototype_filter = _wrong_donor_ontology_intersection(
            prototypes,
            type_names,
        )
        prototype_filter["original_source_prototype_sha256"] = original_prototype_sha256
    if (
        not checkpoint_latent_space_id
        or not prototypes.latent_space_id
        or prototypes.latent_space_id != checkpoint_latent_space_id
    ) and not args.unsafe_allow_latent_space_mismatch:
        raise ValueError("prototype latent_space_id does not match the trained checkpoint")
    parent_type_names = metadata.get("parent_type_names") or None
    if parent_type_names is not None and not isinstance(parent_type_names, (list, tuple)):
        raise ValueError("checkpoint parent ontology metadata is malformed")
    programs = None
    if args.program_artifact:
        programs = GenePrograms.from_npz(args.program_artifact)
        if tuple(str(value) for value in programs.gene_names.tolist()) != gene_names:
            raise ValueError("gene-program artifact order differs from the checkpoint")
        if (
            args.donor_id
            and args.donor_id in {str(value) for value in programs.training_donors.tolist()}
            and not args.unsafe_allow_molecular_validation_overlap
        ):
            raise ValueError("gene programs were fitted on the prediction donor")
    ood_detector = None
    ood_score = None
    ood_threshold = None
    if args.ood_artifact:
        ood_detector = MahalanobisOOD.from_npz(args.ood_artifact)
        if bag.feature_space_id != ood_detector.feature_space_id:
            raise ValueError("OOD detector feature space differs from the HistologyBag")
        if args.donor_id in set(ood_detector.training_donors) and not (
            args.unsafe_allow_molecular_validation_overlap
        ):
            raise ValueError("OOD detector was fitted on the prediction donor")
        if ood_detector.threshold is None:
            raise ValueError("OOD artifact has no fitted threshold")
        ood_score = ood_detector.score(bag.features)
        ood_threshold = ood_detector.threshold
    refinement_round = int(metadata.get("refinement_round", 0))
    effective_confidence = np.asarray(
        bag.segmentation_confidence * (1.0 - bag.artifact_probability),
        dtype=np.float32,
    )
    effective_confidence[bag.artifact_probability >= args.artifact_threshold] = 0.0
    set_seed(args.seed)
    if args.image_feature_shuffle and args.graph_node_shuffle:
        raise ValueError("image-feature and graph-node shuffle controls must be run separately")
    control_transform: Optional[Mapping[str, object]] = None
    inference_features = np.asarray(bag.features)
    if args.image_feature_shuffle:
        image_permutation, control_transform = _shuffle_control_transform(
            "image_shuffle",
            args.seed,
            len(inference_features),
        )
        inference_features = inference_features[image_permutation]
    inference_edge_index = bag.edge_index
    inference_edge_weight = bag.edge_weight
    if args.graph_node_shuffle:
        node_permutation, control_transform = _shuffle_control_transform(
            "graph_shuffle",
            args.seed,
            len(bag.nucleus_ids),
        )
        inference_edge_index = node_permutation[np.asarray(bag.edge_index, dtype=np.int64)]
    elif args.no_graph:
        inference_edge_index = np.empty((2, 0), dtype=np.int64)
        inference_edge_weight = np.empty(0, dtype=np.float32)
    inference_device = resolve_device(args.device)
    use_mixed_precision = (
        inference_device.type == "cuda"
        if args.mixed_precision is None
        else bool(args.mixed_precision)
    )
    if inference_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(inference_device)
        torch.cuda.synchronize(inference_device)
    inference_started = time.perf_counter()
    prediction = predict_cells(
        model,
        inference_features,
        bag.coordinates_um,
        bag.nucleus_ids,
        inference_prototypes,
        type_names,
        gene_names,
        edge_index=inference_edge_index,
        edge_weight=inference_edge_weight,
        latent_samples=args.latent_samples,
        probability_threshold=args.probability_threshold,
        segmentation_confidence=effective_confidence,
        ood_score=ood_score,
        ood_threshold=ood_threshold,
        refinement_round=refinement_round,
        device=args.device,
        sample_id=resolved_sample_id,
        donor_id=args.donor_id,
        slide_id=bag.slide_id,
        checkpoint_sha256=_sha256(args.checkpoint),
        prototype_sha256=original_prototype_sha256,
        histology_sha256=_sha256(args.histology),
        latent_space_id=checkpoint_latent_space_id,
        model_version=str(metadata.get("schema", __version__)),
        parent_type_names=parent_type_names,
        program_matrix=None if programs is None else programs.loadings,
        program_names=None if programs is None else programs.names,
        program_sha256="" if programs is None else _sha256(args.program_artifact),
        program_training_donors=(None if programs is None else programs.training_donors),
        ood_sha256="" if ood_detector is None else _sha256(args.ood_artifact),
        ood_training_donors=(None if ood_detector is None else ood_detector.training_donors),
        inference_seed=args.seed,
        artifact_threshold=args.artifact_threshold,
        expression_space_id=checkpoint_expression_space_id,
        mixed_precision=args.mixed_precision,
        mc_chunk_size=args.mc_chunk_size,
        use_model_abstain=args.use_model_abstain,
        allow_prototype_sample_mismatch=args.wrong_donor_control,
    )
    if inference_device.type == "cuda":
        torch.cuda.synchronize(inference_device)
    inference_wall_seconds = time.perf_counter() - inference_started
    prediction.to_npz(args.output)
    report = {
        "output": str(Path(args.output).expanduser().resolve()),
        "cells": len(prediction.nucleus_ids),
        "genes": len(prediction.gene_names),
        "abstained": int(prediction.abstain.sum()),
        "refinement_round": prediction.refinement_round,
        "negative_control": {
            "prototype_only": bool(args.prototype_only),
            "image_feature_shuffle": bool(args.image_feature_shuffle),
            "graph_node_shuffle": bool(args.graph_node_shuffle),
            "no_graph": bool(args.no_graph),
            "wrong_donor": bool(args.wrong_donor_control),
            "prototype_donor_id": prototypes.donor_id,
            "prototype_filter": prototype_filter,
            "transform": control_transform,
        },
    }
    if args.telemetry_output:
        telemetry = {
            "schema": "heir.inference_telemetry.v1",
            "prediction_path": str(Path(args.output).expanduser().resolve()),
            "prediction_sha256": _sha256(args.output),
            "wall_seconds": inference_wall_seconds,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(inference_device))
                if inference_device.type == "cuda"
                else 0
            ),
            "device_type": inference_device.type,
            "device_name": (
                torch.cuda.get_device_name(inference_device)
                if inference_device.type == "cuda"
                else "CPU"
            ),
            "mixed_precision": use_mixed_precision,
            "nuclei": len(prediction.nucleus_ids),
            "genes": len(prediction.gene_names),
            "latent_samples": args.latent_samples,
            "mc_chunk_size": args.mc_chunk_size,
            "negative_control": {
                "prototype_only": bool(args.prototype_only),
                "image_feature_shuffle": bool(args.image_feature_shuffle),
                "graph_node_shuffle": bool(args.graph_node_shuffle),
                "no_graph": bool(args.no_graph),
                "wrong_donor": bool(args.wrong_donor_control),
                "prototype_donor_id": prototypes.donor_id,
                "prototype_filter": prototype_filter,
                "seed": args.seed,
                "transform": control_transform,
            },
        }
        atomic_json_dump(telemetry, args.telemetry_output)
        report["telemetry_output"] = str(Path(args.telemetry_output).expanduser().resolve())
    _json(report)
    return 0


def _synthetic_batch(seed: int = 17):
    rng = np.random.default_rng(seed)
    cells, feature_dim, types, latent_dim, genes = 72, 12, 3, 4, 8
    prototypes_per_type = 2
    prototype_types = np.repeat(np.arange(types), prototypes_per_type)
    prototype_means = rng.normal(size=(len(prototype_types), latent_dim)).astype(np.float32)
    prototype_variances = np.full_like(prototype_means, 0.25)
    labels = np.tile(np.arange(types), cells // types)
    selected_prototypes = labels * prototypes_per_type + rng.integers(0, prototypes_per_type, cells)
    latent = (
        prototype_means[selected_prototypes] + rng.normal(0, 0.15, size=(cells, latent_dim))
    ).astype(np.float32)
    projection = rng.normal(size=(latent_dim, feature_dim))
    features = (latent @ projection + rng.normal(0, 0.1, size=(cells, feature_dim))).astype(
        np.float32
    )
    coordinates = np.column_stack((np.arange(cells) % 12, np.arange(cells) // 12)).astype(
        np.float32
    )
    source = []
    target = []
    for index in range(cells - 1):
        source.extend((index, index + 1))
        target.extend((index + 1, index))
    edges = np.asarray((source, target), dtype=np.int64)
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=feature_dim,
            num_cell_types=types,
            expression_dim=genes,
            latent_dim=latent_dim,
            graph_hidden_dim=24,
            graph_output_dim=16,
            trunk_hidden_dims=(32, 16),
            decoder_hidden_dims=(16,),
            graph_layers=1,
            dropout=0.0,
            nonnegative_expression=True,
            hard_type_routing=False,
        )
    )
    with torch.no_grad():
        expression = model.expression_decoder(torch.from_numpy(latent)).numpy()
    composition = np.bincount(labels, minlength=types).astype(np.float32)
    composition /= composition.sum()
    type_centroids = np.stack([expression[labels == index].mean(axis=0) for index in range(types)])
    batch = HEIRTrainingBatch(
        morphology=torch.from_numpy(features),
        edge_index=torch.from_numpy(edges),
        edge_weight=torch.ones(edges.shape[1]),
        prototype_means=torch.from_numpy(prototype_means),
        prototype_variances=torch.from_numpy(prototype_variances),
        prototype_types=torch.from_numpy(prototype_types).long(),
        prototype_weights=torch.full((len(prototype_types),), 1.0 / len(prototype_types)),
        target_composition=torch.from_numpy(composition),
        target_pseudobulk=torch.from_numpy(
            np.log1p(np.expm1(expression).mean(axis=0)).astype(np.float32)
        ),
        anchor_labels=torch.from_numpy(labels).long(),
        anchor_weights=torch.ones(cells),
        marker_centroids=torch.from_numpy(type_centroids),
        sample_id="synthetic",
        analysis_role="development",
    )
    return model, batch, coordinates


def command_demo(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    model, batch, _ = _synthetic_batch(args.seed)
    optimization = OptimizationConfig(
        epochs=args.epochs,
        learning_rate=5.0e-4,
        bag_size=len(batch.morphology),
        reference_batch_size=len(batch.morphology),
        early_stopping_patience=max(2, args.epochs),
        mixed_precision=False,
    )
    trainer = HEIRTrainer(
        model,
        TrainingStage.PERSONALIZED,
        optimization,
        LossWeightConfig(),
        seed=args.seed,
        device=args.device,
        allow_split_overlap=True,
    )
    result = trainer.fit([batch], [batch])
    destination = Path(args.output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint(), destination / "heir_demo.pt")
    atomic_json_dump(
        {
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "history": list(result.history),
        },
        destination / "metrics.json",
    )
    _json(
        {
            "checkpoint": str(destination / "heir_demo.pt"),
            "metrics": str(destination / "metrics.json"),
            "best_validation_loss": result.best_validation_loss,
        }
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    prediction = PredictionBundle.from_npz(args.predictions)
    with np.load(args.truth, allow_pickle=False) as truth:
        if "nucleus_ids" not in truth:
            raise ValueError("truth NPZ requires nucleus_ids for row integrity")
        truth_ids = np.asarray(truth["nucleus_ids"], dtype=np.dtype("U"))
        if not np.array_equal(truth_ids, prediction.nucleus_ids.astype(str)):
            raise ValueError("truth nucleus_ids must exactly match prediction row order")
        result: Dict[str, object] = {}
        if "true_labels" in truth:
            if "type_names" not in truth or not np.array_equal(
                np.asarray(truth["type_names"], dtype=np.dtype("U")),
                prediction.type_names.astype(str),
            ):
                raise ValueError("truth type_names must exactly match prediction ontology")
            result["cell_type"] = cell_type_metrics(
                truth["true_labels"], prediction.type_probabilities
            )
        if "observed_expression" in truth:
            if "gene_names" not in truth or "expression_space_id" not in truth:
                raise ValueError("expression truth requires gene_names and expression_space_id")
            truth_expression_space_id = str(np.asarray(truth["expression_space_id"]).item())
            if truth_expression_space_id != prediction.expression_space_id:
                raise ValueError("truth and prediction use different expression spaces")
            truth_genes = np.asarray(truth["gene_names"], dtype=np.dtype("U"))
            if truth_genes.ndim != 1 or len(set(truth_genes.tolist())) != len(truth_genes):
                raise ValueError("truth gene_names must be a unique vector")
            lookup = {name: index for index, name in enumerate(prediction.gene_names.astype(str))}
            missing = sorted(set(truth_genes.tolist()) - set(lookup))
            if missing:
                raise ValueError("prediction is missing truth genes: %s" % ", ".join(missing))
            order = np.asarray([lookup[name] for name in truth_genes], dtype=np.int64)
            observed_expression = np.asarray(truth["observed_expression"])
            if observed_expression.shape != (len(prediction.nucleus_ids), len(truth_genes)):
                raise ValueError(
                    "observed_expression must have shape (prediction cells, truth genes)"
                )
            public_expression = prediction.public_cell_expression_mean[:, order]
            available = np.asarray(prediction.expression_mean_available, dtype=bool)
            if available.shape != (len(prediction.nucleus_ids),):
                raise ValueError("prediction expression_mean_available is not cell-aligned")
            if not np.array_equal(available, np.isfinite(public_expression).all(axis=1)):
                raise ValueError("public cell expression and expression_mean_available disagree")
            if not available.any():
                raise ValueError("prediction has no available public cell-level expression means")
            expression_result = expression_metrics(
                public_expression[available],
                observed_expression[available],
            )
            expression_result.update(
                {
                    "cells_total": int(len(available)),
                    "cells_evaluated": int(available.sum()),
                    "cells_unavailable_excluded": int((~available).sum()),
                    "availability_policy": "prediction.expression_mean_available",
                }
            )
            result["expression"] = expression_result
    if not result:
        raise ValueError("truth NPZ needs true_labels and/or observed_expression")
    if args.output:
        atomic_json_dump(result, args.output)
    _json(result)
    return 0


def _aggregate_spatial_values(
    values: np.ndarray,
    num_spots: int,
    dense_assignment: Optional[np.ndarray],
    spot_index: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("cell values for spatial aggregation must be finite and two-dimensional")
    if dense_assignment is not None:
        assignment = np.asarray(dense_assignment, dtype=np.float64)
        if assignment.shape != (num_spots, data.shape[0]):
            raise ValueError("spot_assignment must have shape (spots, prediction cells)")
        if np.any(assignment < 0) or not np.isfinite(assignment).all():
            raise ValueError("spot_assignment must be finite and non-negative")
        mass = assignment.sum(axis=1)
        sums = assignment @ data
    else:
        assert spot_index is not None
        raw_index = np.asarray(spot_index)
        if raw_index.shape != (data.shape[0],) or not np.issubdtype(raw_index.dtype, np.integer):
            raise ValueError("nucleus_spot_index must be an integer vector aligned to cells")
        indices = raw_index.astype(np.int64, copy=False)
        if np.any(indices < -1) or np.any(indices >= num_spots):
            raise ValueError("nucleus_spot_index contains an unavailable spot")
        assigned = indices >= 0
        mass = np.bincount(indices[assigned], minlength=num_spots).astype(np.float64)
        sums = np.zeros((num_spots, data.shape[1]), dtype=np.float64)
        np.add.at(sums, indices[assigned], data[assigned])
    means = sums / np.maximum(mass[:, None], 1.0e-12)
    return means.astype(np.float32), mass


def command_evaluate_spatial(args: argparse.Namespace) -> int:
    """Evaluate locked spot expression without exposing it to training."""

    prediction = PredictionBundle.from_npz(args.predictions)
    with np.load(args.truth, allow_pickle=False) as truth:
        if "__contract__" in truth or "__version__" in truth:
            # Enforce locked role, version, identities, and provenance before
            # scoring artifacts created by prepare-spatial-truth.
            SpatialTruthArtifact.from_npz(args.truth)
        required = {
            "observed_expression",
            "gene_names",
            "spot_ids",
            "nucleus_ids",
            "expression_space_id",
        }
        missing = sorted(required - set(truth.files))
        if missing:
            raise ValueError("spatial truth artifact is missing: %s" % ", ".join(missing))
        observed_expression = np.asarray(truth["observed_expression"], dtype=np.float32).copy()
        truth_genes = np.asarray(truth["gene_names"], dtype=np.dtype("U")).copy()
        spot_ids = np.asarray(truth["spot_ids"], dtype=np.dtype("U")).copy()
        truth_nucleus_ids = np.asarray(
            truth["nucleus_ids"],
            dtype=np.dtype("U"),
        ).copy()
        truth_expression_space_id = str(np.asarray(truth["expression_space_id"]).item())
        has_dense = "spot_assignment" in truth
        has_index = "nucleus_spot_index" in truth
        if has_dense == has_index:
            raise ValueError(
                "spatial truth needs exactly one of spot_assignment or nucleus_spot_index"
            )
        dense_assignment = (
            np.asarray(truth["spot_assignment"], dtype=np.float64).copy() if has_dense else None
        )
        spot_index = np.asarray(truth["nucleus_spot_index"]).copy() if has_index else None
        has_composition = "observed_composition" in truth or "type_names" in truth
        if has_composition and not {
            "observed_composition",
            "type_names",
        }.issubset(truth.files):
            raise ValueError("observed_composition and type_names must be supplied together")
        observed_composition = (
            np.asarray(truth["observed_composition"], dtype=np.float32).copy()
            if has_composition
            else None
        )
        truth_types = (
            np.asarray(truth["type_names"], dtype=np.dtype("U")).copy() if has_composition else None
        )

    if truth_genes.ndim != 1 or spot_ids.ndim != 1:
        raise ValueError("spatial gene_names and spot_ids must be vectors")
    if truth_expression_space_id != prediction.expression_space_id:
        raise ValueError("spatial truth and prediction use different expression spaces")
    if truth_nucleus_ids.ndim != 1 or not np.array_equal(
        truth_nucleus_ids,
        prediction.nucleus_ids.astype(str),
    ):
        raise ValueError("spatial truth nucleus_ids must exactly match prediction row order")
    if len(set(truth_genes.tolist())) != len(truth_genes):
        raise ValueError("spatial truth contains duplicate gene names")
    if len(set(spot_ids.tolist())) != len(spot_ids):
        raise ValueError("spatial truth contains duplicate spot IDs")
    if observed_expression.shape != (len(spot_ids), len(truth_genes)):
        raise ValueError("observed_expression must have shape (spots, genes)")
    predicted_genes = [str(value) for value in prediction.gene_names.tolist()]
    if len(set(predicted_genes)) != len(predicted_genes):
        raise ValueError("prediction contains duplicate gene names")
    predicted_gene_lookup = {name: index for index, name in enumerate(predicted_genes)}
    missing_genes = sorted(set(truth_genes.tolist()) - set(predicted_gene_lookup))
    if missing_genes:
        raise ValueError("prediction is missing truth genes: %s" % ", ".join(missing_genes))
    gene_order = np.asarray(
        [predicted_gene_lookup[str(name)] for name in truth_genes],
        dtype=np.int64,
    )
    spot_expression, mass = _aggregate_spatial_values(
        np.expm1(prediction.internal_aggregate_expression_mean[:, gene_order]),
        len(spot_ids),
        dense_assignment,
        spot_index,
    )
    spot_expression = np.log1p(spot_expression)
    spot_probabilities, probability_mass = _aggregate_spatial_values(
        prediction.type_probabilities,
        len(spot_ids),
        dense_assignment,
        spot_index,
    )
    if not np.allclose(mass, probability_mass):
        raise RuntimeError("expression/type aggregation produced inconsistent spot mass")
    evaluable = mass > 0
    if not evaluable.any():
        raise ValueError("spatial assignment contains no evaluable spots")
    result: Dict[str, object] = {
        "spots_total": int(len(spot_ids)),
        "spots_evaluated": int(evaluable.sum()),
        "empty_spots_ignored": int((~evaluable).sum()),
        "gene_names": truth_genes.tolist(),
        "expression": expression_metrics(
            spot_expression[evaluable],
            observed_expression[evaluable],
        ),
    }
    aligned_spot_probabilities = spot_probabilities
    if observed_composition is not None:
        assert truth_types is not None
        if truth_types.ndim != 1 or len(set(truth_types.tolist())) != len(truth_types):
            raise ValueError("spatial truth type_names must be a unique vector")
        if observed_composition.shape != (len(spot_ids), len(truth_types)):
            raise ValueError("observed_composition must have shape (spots, types)")
        predicted_types = [str(value) for value in prediction.type_names.tolist()]
        if len(set(predicted_types)) != len(predicted_types):
            raise ValueError("prediction contains duplicate type names")
        predicted_type_lookup = {name: index for index, name in enumerate(predicted_types)}
        missing_types = sorted(set(truth_types.tolist()) - set(predicted_type_lookup))
        if missing_types:
            raise ValueError("prediction is missing truth types: %s" % ", ".join(missing_types))
        type_order = np.asarray(
            [predicted_type_lookup[str(name)] for name in truth_types],
            dtype=np.int64,
        )
        aligned_spot_probabilities = spot_probabilities[:, type_order]
        if np.any(observed_composition[evaluable].sum(axis=1) <= 0):
            raise ValueError("observed composition needs positive mass in evaluable spots")
        result["type_names"] = truth_types.tolist()
        result["composition"] = composition_metrics(
            aligned_spot_probabilities[evaluable],
            observed_composition[evaluable],
        )
    else:
        result["composition"] = {
            "scored": False,
            "reason": "truth artifact has no explicit observed_composition",
            "predicted_mean_by_type": spot_probabilities[evaluable].mean(axis=0).tolist(),
            "predicted_type_names": prediction.type_names.tolist(),
        }
    if args.aggregates_output:
        destination = Path(args.aggregates_output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            spot_ids=spot_ids[evaluable],
            gene_names=truth_genes,
            predicted_expression=spot_expression[evaluable],
            observed_expression=observed_expression[evaluable],
            predicted_composition=aligned_spot_probabilities[evaluable],
            spot_mass=mass[evaluable],
        )
        result["aggregates_output"] = str(destination)
    if args.output:
        atomic_json_dump(result, args.output)
    _json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heir", description="HEIR molecular spatialization")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="audit runtime and local manifests")
    doctor.add_argument(
        "--manifest",
        action="append",
        default=[str(default_manifest_path("natcommun")), str(default_manifest_path("snpatho"))],
    )
    doctor.add_argument("--require-files", action="store_true")
    doctor.set_defaults(func=command_doctor)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("manifest")
    validate.add_argument("--require-folds", action="store_true")
    validate.add_argument("--require-files", action="store_true")
    validate.add_argument("--checksums", action="store_true")
    validate.set_defaults(func=command_validate_manifest)

    prepare = subparsers.add_parser("prepare-reference")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--section-id", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--input",
        help="audited H5AD derivative (for example an RDS export) bound to the manifest row",
    )
    prepare.add_argument(
        "--conversion-provenance",
        help="JSON lineage sidecar emitted by scripts/export_seurat.R",
    )
    prepare.add_argument("--genes")
    prepare.add_argument("--cell-type-key", default="Level1")
    prepare.add_argument("--gene-key", default="feature_name")
    prepare.add_argument("--layer", default="SoupX")
    prepare.add_argument("--chunk-size", type=int, default=512)
    prepare.set_defaults(func=command_prepare_reference)

    prepare_spatial = subparsers.add_parser(
        "prepare-spatial-truth",
        help="build a locked Visium expression/spot-assignment artifact",
    )
    prepare_spatial.add_argument("--manifest", required=True)
    prepare_spatial.add_argument("--section-id", required=True)
    prepare_spatial.add_argument(
        "--counts",
        required=True,
        help="filtered/QC H5AD, 10x HDF5, or 10x Matrix Market directory",
    )
    prepare_spatial.add_argument(
        "--conversion-provenance",
        help="JSON lineage sidecar binding derived counts to the manifest spatial source",
    )
    prepare_spatial.add_argument("--positions", required=True)
    prepare_spatial.add_argument("--scalefactors", required=True)
    prepare_spatial.add_argument("--nuclei", required=True)
    prepare_spatial.add_argument("--genes", required=True)
    prepare_spatial.add_argument("--output", required=True)
    prepare_spatial.add_argument("--layer")
    prepare_spatial.add_argument("--gene-key")
    prepare_spatial.add_argument("--chunk-size", type=int, default=512)
    prepare_spatial.add_argument("--coordinate-scale", type=float, default=1.0)
    prepare_spatial.add_argument(
        "--barcode-suffix-policy",
        choices=BARCODE_SUFFIX_POLICIES,
        default="auto",
    )
    prepare_spatial.set_defaults(func=command_prepare_spatial_truth)

    filter_capture = subparsers.add_parser(
        "filter-nuclei-to-visium",
        help="retain nuclei inside in-tissue Visium disks using geometry only",
    )
    filter_capture.add_argument("--nuclei", required=True)
    filter_capture.add_argument("--positions", required=True)
    filter_capture.add_argument("--scalefactors", required=True)
    filter_capture.add_argument(
        "--output",
        required=True,
        help="filtered nucleus CSV consumed by feature extraction and prepare-histology",
    )
    filter_capture.add_argument("--assignment-output", required=True)
    filter_capture.add_argument("--provenance-output", required=True)
    filter_capture.add_argument("--coordinate-scale", type=float, default=1.0)
    filter_capture.add_argument("--overwrite", action="store_true")
    filter_capture.set_defaults(func=command_filter_nuclei_to_visium)

    segment_histology = subparsers.add_parser(
        "segment-histology",
        help="run Space Ranger 4.x (default) or import its nucleus GeoJSON",
    )
    segmentation_source = segment_histology.add_mutually_exclusive_group(required=True)
    segmentation_source.add_argument("--image")
    segmentation_source.add_argument("--geojson")
    segment_histology.add_argument("--slide-id", required=True)
    segment_histology.add_argument("--run-id")
    segment_histology.add_argument("--output-directory", default="artifacts/segmentation_runs")
    segment_histology.add_argument("--nuclei-output", required=True)
    segment_histology.add_argument("--features-output", required=True)
    segment_histology.add_argument("--spaceranger")
    segment_histology.add_argument("--spaceranger-version", default="4.1.0")
    segment_histology.add_argument("--localcores", type=int, default=8)
    segment_histology.add_argument("--localmem-gb", type=int, default=24)
    segment_histology.add_argument("--max-nucleus-diameter-px", type=int)
    segment_histology.add_argument("--minimum-area-px2", type=float, default=8.0)
    segment_histology.add_argument("--cuda-visible-devices", default="auto")
    segment_histology.add_argument("--timeout-seconds", type=float)
    segment_histology.add_argument("--overwrite", action="store_true")
    segment_histology.set_defaults(func=command_segment_histology)

    pathology_features = subparsers.add_parser(
        "extract-pathology-features",
        help="extract frozen multi-scale CUDA features for Space Ranger nuclei",
    )
    pathology_features.add_argument("--image", required=True)
    pathology_features.add_argument("--nuclei", required=True)
    pathology_features.add_argument("--output", required=True)
    pathology_features.add_argument(
        "--encoder",
        choices=("omiclip-loki", "resnet50-imagenet"),
        default="omiclip-loki",
    )
    pathology_features.add_argument(
        "--checkpoint",
        help="published Loki checkpoint (required for the default OmiCLIP encoder)",
    )
    pathology_features.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="permit pickle loading for a verified raw Loki training archive",
    )
    pathology_features.add_argument("--mpp", type=float, required=True)
    pathology_features.add_argument(
        "--backend", choices=("auto", "openslide", "pil"), default="auto"
    )
    pathology_features.add_argument("--patch-diameters-um", default="32,128")
    pathology_features.add_argument("--input-size", type=int, default=224)
    pathology_features.add_argument("--batch-size", type=int, default=64)
    pathology_features.add_argument("--device", default="auto")
    pathology_features.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    pathology_features.add_argument("--offset", type=int, default=0)
    pathology_features.add_argument("--limit", type=int)
    pathology_features.add_argument("--telemetry-output")
    pathology_features.add_argument("--overwrite", action="store_true")
    pathology_features.set_defaults(func=command_extract_pathology_features)

    prepare_histology = subparsers.add_parser(
        "prepare-histology",
        help="join a nucleus table and cached features into a calibrated graph bag",
    )
    prepare_histology.add_argument("--nuclei", required=True)
    prepare_histology.add_argument("--features", required=True)
    prepare_histology.add_argument("--manifest")
    prepare_histology.add_argument("--section-id")
    prepare_histology.add_argument("--histology-source")
    prepare_histology.add_argument("--feature-space-id", required=True)
    prepare_histology.add_argument("--slide-id")
    prepare_histology.add_argument("--sample-id")
    prepare_histology.add_argument("--donor-id")
    prepare_histology.add_argument("--block-id")
    prepare_histology.add_argument("--output", required=True)
    coordinate_group = prepare_histology.add_mutually_exclusive_group()
    coordinate_group.add_argument("--mpp", type=float)
    coordinate_group.add_argument("--coordinates-are-microns", action="store_true")
    prepare_histology.add_argument("--default-segmentation-confidence", type=float, default=1.0)
    prepare_histology.add_argument("--artifact-key")
    prepare_histology.add_argument("--boundary-weight-key")
    prepare_histology.add_argument("--exclude-morphology", action="store_true")
    prepare_histology.add_argument("--graph-k", type=int, default=12)
    prepare_histology.add_argument("--graph-radius-um", type=float, default=50.0)
    prepare_histology.add_argument("--graph-max-degree", type=int, default=24)
    prepare_histology.set_defaults(func=command_prepare_histology)

    prototypes = subparsers.add_parser("build-prototypes")
    prototypes.add_argument("--reference", required=True)
    prototypes.add_argument("--output", required=True)
    prototypes.add_argument("--manifest")
    prototypes.add_argument("--section-id")
    prototypes.add_argument("--reference-with-latent")
    prototypes.add_argument(
        "--latent-space-id",
        help="checked identity for a pre-existing external latent representation",
    )
    prototypes.add_argument(
        "--fit-latent-transform",
        help="fit and persist the shared development-reference SVD transform",
    )
    prototypes.add_argument(
        "--latent-transform",
        help="consume a previously fitted shared SVD transform without refitting",
    )
    prototypes.add_argument("--latent-dim", type=int, default=32)
    prototypes.add_argument("--max-per-type", type=int, default=10)
    prototypes.add_argument("--minimum-cells", type=int, default=50)
    prototypes.add_argument("--shrinkage-kappa", type=float, default=50.0)
    prototypes.add_argument("--recompute-latent", action="store_true")
    prototypes.add_argument("--unsafe-allow-legacy-latent-transform", action="store_true")
    prototypes.add_argument(
        "--include-rare-types",
        action="store_true",
        help="retain types below minimum-cells as unresolved single prototypes",
    )
    prototypes.add_argument("--seed", type=int, default=17)
    prototypes.set_defaults(func=command_build_prototypes)

    residual_geometry = subparsers.add_parser(
        "fit-residual-geometry",
        help="fit frozen within-type RNA PCA bases and type-calibrated residual bounds",
    )
    residual_geometry.add_argument("--reference", required=True)
    residual_geometry.add_argument("--prototypes")
    residual_geometry.add_argument("--output", required=True)
    residual_geometry.add_argument("--rank", type=int, default=4)
    residual_geometry.add_argument(
        "--type-name",
        action="append",
        help="authoritative model type order; repeat once per type",
    )
    residual_geometry.add_argument("--calibration-quantile", type=float, default=0.90)
    residual_geometry.add_argument("--bound-fraction", type=float, default=0.50)
    residual_geometry.add_argument("--minimum-bound", type=float, default=1.0e-3)
    residual_geometry.add_argument("--maximum-bound", type=float)
    residual_geometry.add_argument("--minimum-calibration-cells", type=int, default=3)
    residual_geometry.set_defaults(func=command_fit_residual_geometry)

    fit_ood = subparsers.add_parser(
        "fit-ood",
        help="fit a shrinkage-Mahalanobis pathology-feature OOD detector",
    )
    fit_ood.add_argument("--histology", action="append", required=True)
    fit_ood.add_argument(
        "--training-donor",
        action="append",
        help="optional assertion; must exactly match donor IDs embedded in the bags",
    )
    fit_ood.add_argument("--analysis-role", default="train")
    fit_ood.add_argument("--quantile", type=float, default=0.95)
    fit_ood.add_argument("--output", required=True)
    fit_ood.set_defaults(func=command_fit_ood)

    assemble = subparsers.add_parser(
        "assemble-batch",
        help="build sample-level weak targets without assuming cell correspondence",
    )
    assemble.add_argument("--histology", required=True, help="HistologyBag NPZ")
    assemble.add_argument("--prototypes", required=True, help="PrototypeSet NPZ")
    assemble.add_argument("--reference", required=True, help="matched RNAReference NPZ")
    assemble.add_argument("--output", required=True, help="HEIRTrainingBatch NPZ")
    assemble.add_argument("--manifest")
    assemble.add_argument("--section-id")
    assemble.add_argument("--scgpt-artifact")
    assemble.add_argument("--program-artifact")
    assemble.add_argument("--ood-artifact")
    assemble.add_argument("--unknown-targets")
    assemble.add_argument("--domain-label", type=int)
    assemble.add_argument("--spatial-pretraining-truth")
    assemble.add_argument("--sample-id")
    assemble.add_argument("--bag-id")
    assemble.add_argument(
        "--latent-space-id",
        help="stable ID/SHA-256 for the encoder/transform defining prototype axes",
    )
    assemble.add_argument("--unsafe-allow-unspecified-latent-space", action="store_true")
    assemble.add_argument("--unsafe-allow-missing-prototype-provenance", action="store_true")
    assemble.add_argument("--unsafe-allow-missing-histology-provenance", action="store_true")
    assemble.add_argument("--donor-id")
    assemble.add_argument("--block-id")
    assemble.add_argument("--analysis-role")
    assemble.add_argument("--artifact-threshold", type=float, default=0.50)
    assemble.add_argument("--markers-per-type", type=int, default=25)
    assemble.set_defaults(func=command_assemble_batch)

    train = subparsers.add_parser("train", help="fit HEIR from versioned training-batch artifacts")
    train.add_argument("--train-batch", action="append", required=True)
    train.add_argument("--validation-batch", action="append", required=True)
    train.add_argument("--output", required=True)
    train.add_argument(
        "--stage",
        choices=(
            TrainingStage.PERSONALIZED.value,
            TrainingStage.GENERIC_SPATIAL_PRETRAINING.value,
        ),
        default=TrainingStage.PERSONALIZED.value,
    )
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1.0e-4)
    train.add_argument("--adapter-learning-rate", type=float, default=1.0e-5)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--warmup-fraction", type=float, default=0.05)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument("--bag-size", type=int)
    train.add_argument("--reference-batch-size", type=int)
    train.add_argument("--maximum-sample-cells", type=int, default=16384)
    train.add_argument("--early-stopping-patience", type=int, default=15)
    train.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA FP16 tensor cores (default: enabled automatically on CUDA)",
    )
    train.add_argument("--graph-hidden-dim", type=int, default=128)
    train.add_argument("--graph-output-dim", type=int, default=128)
    train.add_argument("--graph-layers", type=int, default=2)
    train.add_argument("--trunk-hidden-dims", default="256,128")
    train.add_argument("--decoder-hidden-dims", default="128,256")
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--hard-type-routing", action="store_true")
    train.add_argument("--abstain-threshold", type=float, default=0.60)
    train.add_argument("--uot-unknown-mass", type=float, default=0.05)
    train.add_argument(
        "--uot-unknown-mass-mode",
        choices=("fixed", "targets_or_fixed", "model_estimate"),
        default="fixed",
    )
    train.add_argument(
        "--residual-rank",
        type=int,
        default=0,
        help="low-rank residual width (default: min(4, latent dimension))",
    )
    train.add_argument(
        "--residual-max-norm",
        type=float,
        default=0.5,
        help="hard upper bound on each morphology residual's latent L2 norm",
    )
    train.add_argument(
        "--residual-geometry",
        help="validated RNAResidualGeometry NPZ with type bases and calibrated bounds",
    )
    train.add_argument(
        "--finetune-residual-basis",
        action="store_true",
        help="allow an RNA-initialized residual basis to move (default: frozen)",
    )
    train.add_argument(
        "--unsafe-allow-legacy-residual-geometry-provenance",
        action="store_true",
        help=(
            "migration-only escape hatch for batches whose prototype/reference source files "
            "or geometry identities are unavailable; observed identity mismatches still fail"
        ),
    )
    train.add_argument("--allow-negative-expression", action="store_true")
    train.add_argument("--rna-vae-checkpoint")
    train.add_argument("--initial-heir-checkpoint")
    train.add_argument("--allow-random-decoder", action="store_true")
    train.add_argument("--finetune-rna-decoder", action="store_true")
    train.add_argument(
        "--unsafe-allow-molecular-validation-overlap",
        action="store_true",
        help="smoke-test escape hatch; invalidates donor-held-out molecular evaluation",
    )
    train.add_argument("--unsafe-allow-latent-space-mismatch", action="store_true")
    train.add_argument("--unsafe-allow-feature-space-mismatch", action="store_true")
    train.add_argument("--unsafe-allow-expression-space-mismatch", action="store_true")
    train.add_argument("--ontology", help="type-to-parent TSV covering the exact batch ontology")
    train.add_argument("--allow-split-overlap", action="store_true")
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", default="auto")
    train.set_defaults(func=command_train)

    refine = subparsers.add_parser(
        "refine",
        help="run constrained broad-to-fine generalized EM with an EMA teacher",
    )
    refine.add_argument("--checkpoint", required=True)
    refine.add_argument("--train-batch", action="append", required=True)
    refine.add_argument("--validation-batch", action="append", required=True)
    refine.add_argument("--output", required=True)
    refine.add_argument(
        "--save-round-checkpoints",
        action="store_true",
        help="persist auditable candidate checkpoints for every refinement trajectory round",
    )
    refine.add_argument(
        "--view-predictions",
        action="append",
        help="DONOR::SAMPLE::BAG=NPZ with nucleus_ids and view_predictions",
    )
    refine.add_argument("--rna-vae-checkpoint")
    refine.add_argument(
        "--maximum-rounds",
        type=int,
        default=4,
        help="maximum candidates; default leaves two parent-gated and two fine rounds",
    )
    refine.add_argument(
        "--broad-refinement-rounds",
        type=int,
        default=2,
        help="parent-gated rounds; use 0 for fine-only or at least 2 for anchor trust",
    )
    refine.add_argument("--epochs-per-round", type=int, default=25)
    refine.add_argument("--min-probability", type=float, default=0.90)
    refine.add_argument("--max-normalized-entropy", type=float, default=0.20)
    refine.add_argument("--teacher-ema", type=float, default=0.99)
    refine.add_argument(
        "--prior-old-weight",
        type=float,
        default=1.0,
        help="measured-prior weight (default 1.0 fixes it; lower values are sensitivities)",
    )
    refine.add_argument("--minimum-segmentation-confidence", type=float, default=0.50)
    refine.add_argument("--uot-unknown-mass", type=float, default=0.05)
    refine.add_argument(
        "--uot-unknown-mass-mode",
        choices=("fixed", "targets_or_fixed", "model_estimate"),
        default="fixed",
    )
    refine.add_argument("--maximum-prior-total-variation", type=float, default=0.10)
    refine.add_argument("--max-anchors-per-class", type=int, default=10000)
    refine.add_argument("--stable-rounds-required", type=int, default=1)
    refine.add_argument(
        "--maximum-validation-loss-degradation",
        type=float,
        default=0.01,
        help="absolute candidate-round loss degradation allowed before rollback",
    )
    refine.add_argument(
        "--objective-relative-stability-tolerance",
        type=float,
        default=0.01,
        help="relative weak-objective change treated as stable",
    )
    refine.add_argument(
        "--objective-stability-tolerance",
        type=float,
        default=None,
        help="deprecated alias that overrides both refinement tolerances",
    )
    refine.add_argument(
        "--round-selection-mode",
        choices=("fixed", "weak"),
        default="fixed",
        help="fixed uses a development-locked round count; weak is a legacy sensitivity",
    )
    refine.add_argument(
        "--maximum-spatial-score-degradation",
        type=float,
        default=0.0,
        help="programmatic spatial-selection tolerance (CLI fixed/weak modes only)",
    )
    refine.add_argument("--allow-no-view-agreement", action="store_true")
    refine.add_argument("--learning-rate", type=float, default=1.0e-4)
    refine.add_argument("--adapter-learning-rate", type=float, default=1.0e-5)
    refine.add_argument("--weight-decay", type=float, default=1.0e-4)
    refine.add_argument("--warmup-fraction", type=float, default=0.05)
    refine.add_argument("--gradient-clip-norm", type=float, default=1.0)
    refine.add_argument("--bag-size", type=int)
    refine.add_argument("--reference-batch-size", type=int)
    refine.add_argument("--maximum-sample-cells", type=int, default=16384)
    refine.add_argument("--early-stopping-patience", type=int, default=10)
    refine.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA FP16 tensor cores (default: enabled automatically on CUDA)",
    )
    refine.add_argument("--allow-split-overlap", action="store_true")
    refine.add_argument(
        "--unsafe-allow-molecular-validation-overlap",
        action="store_true",
    )
    refine.add_argument("--seed", type=int, default=17)
    refine.add_argument("--device", default="auto")
    refine.set_defaults(func=command_refine)

    predict = subparsers.add_parser("predict", help="run a trained HEIR model on one graph bag")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--histology", required=True)
    predict.add_argument("--prototypes", required=True)
    predict.add_argument("--genes", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--telemetry-output")
    predict.add_argument("--latent-samples", type=int, default=20)
    predict.add_argument("--seed", type=int, default=17)
    predict.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA FP16 tensor cores (default: enabled automatically on CUDA)",
    )
    predict.add_argument("--mc-chunk-size", type=int)
    predict.add_argument("--probability-threshold", type=float, default=0.60)
    predict.add_argument("--artifact-threshold", type=float, default=0.50)
    predict.add_argument(
        "--prototype-only",
        action="store_true",
        help="set the learned morphology residual gate to zero for a prototype-only control",
    )
    predict.add_argument(
        "--image-feature-shuffle",
        action="store_true",
        help="permute complete morphology records across nuclei with --seed",
    )
    predict.add_argument(
        "--wrong-donor-control",
        action="store_true",
        help="permit a mismatched donor/block prototype bank and record the negative control",
    )
    graph_control = predict.add_mutually_exclusive_group()
    graph_control.add_argument(
        "--graph-node-shuffle",
        action="store_true",
        help="degree-preserving random relabeling of graph nodes with --seed",
    )
    graph_control.add_argument(
        "--no-graph",
        action="store_true",
        help="remove every graph edge for the no-context ablation",
    )
    predict.add_argument(
        "--use-model-abstain",
        action="store_true",
        help="also apply the model's composite uncertainty abstention decision",
    )
    predict.add_argument("--sample-id")
    predict.add_argument("--donor-id", required=True)
    predict.add_argument("--program-artifact")
    predict.add_argument("--ood-artifact")
    predict.add_argument(
        "--unsafe-allow-molecular-validation-overlap",
        action="store_true",
    )
    predict.add_argument("--unsafe-allow-latent-space-mismatch", action="store_true")
    predict.add_argument("--unsafe-allow-feature-space-mismatch", action="store_true")
    predict.add_argument("--unsafe-allow-missing-prototype-provenance", action="store_true")
    predict.add_argument("--unsafe-allow-missing-histology-provenance", action="store_true")
    predict.add_argument("--device", default="auto")
    predict.set_defaults(func=command_predict)

    demo = subparsers.add_parser("demo", help="run a finite synthetic end-to-end smoke test")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--device", default="cpu")
    demo.set_defaults(func=command_demo)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=command_evaluate)

    evaluate_spatial = subparsers.add_parser(
        "evaluate-spatial",
        help="aggregate predictions and score a locked spot-level truth artifact",
    )
    evaluate_spatial.add_argument("--predictions", required=True)
    evaluate_spatial.add_argument("--truth", required=True)
    evaluate_spatial.add_argument("--output")
    evaluate_spatial.add_argument("--aggregates-output")
    evaluate_spatial.set_defaults(func=command_evaluate_spatial)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError, KeyError, FileNotFoundError, ImportError, RuntimeError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
