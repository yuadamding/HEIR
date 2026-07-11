#!/usr/bin/env python3
"""Safely orchestrate the frozen, locked snPATHO SIGHT benchmark.

The default mode is a non-mutating command plan.  Pass ``--execute`` to run
missing stages.  A stage is skipped only after its complete output contract is
loaded and checked; partial or invalid output is never deleted or overwritten.

Target Visium expression is deliberately phase-locked.  The three predictions
are frozen and validated before ``prepare-spatial-truth`` can be invoked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from heir.data import (
    HistologyBag,
    PrototypeSet,
    RNAReference,
    SpatialTruthArtifact,
    load_manifest,
    read_spot_diameter,
)
from heir.evaluation import BENCHMARK_METHODS, InferenceTelemetry, load_snpatho_plan
from heir.expression import EXPRESSION_SPACE_ID
from heir.image import load_feature_bundle, load_nuclei
from heir.inference import PredictionBundle
from heir.models import HEIRModel
from heir.training import HEIRTrainingBatch, TrainingStage
from heir.uncertainty import MahalanobisOOD

SAMPLES = ("4066", "4399", "4411")
PREDICTION_PHASE = (
    "segmentation",
    "capture_filter",
    "pathology_features",
    "prepare_histology",
    "calibrate_ood",
    "split_histology",
    "prepare_reference",
    "build_prototypes",
    "assemble_batches",
    "train",
    "predict",
)
LOCKED_PHASE = ("prepare_locked_truth", "freeze_plan", "benchmark")
ALL_STAGES = PREDICTION_PHASE + LOCKED_PHASE
EXPECTED_PANEL_SHA256 = "22ddb91188b3b124d5cf3ec0f7ae81017399d141e39647b0dce80675119fe927"
_SHA256_CACHE: Dict[Tuple[str, int, int], str] = {}
_ARTIFACT_SHA256_CACHE: Dict[Tuple[str, int, int], str] = {}


class PipelineError(RuntimeError):
    """Fail-closed orchestration error with a user-actionable message."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    if key in _SHA256_CACHE:
        return _SHA256_CACHE[key]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _SHA256_CACHE[key] = value
    return value


def _artifact_sha256(value: str) -> str:
    """Hash a file, directory, or manifest ``archive::member`` artifact."""

    raw_path, separator, member_name = str(value).partition("::")
    source = Path(raw_path).expanduser().resolve()
    if not separator:
        if not source.is_dir():
            return _sha256(source)
        stat = source.stat()
        key = (str(source), int(stat.st_size), int(stat.st_mtime_ns))
        if key in _ARTIFACT_SHA256_CACHE:
            return _ARTIFACT_SHA256_CACHE[key]
        files = sorted(path for path in source.rglob("*") if path.is_file())
        if not files:
            raise ValueError("artifact directory contains no files: %s" % source)
        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(source).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, byteorder="big"))
            digest.update(relative)
            digest.update(bytes.fromhex(_sha256(path)))
        result = digest.hexdigest()
        _ARTIFACT_SHA256_CACHE[key] = result
        return result

    stat = source.stat()
    cache_name = str(source) + "::" + member_name
    key = (cache_name, int(stat.st_size), int(stat.st_mtime_ns))
    if key in _ARTIFACT_SHA256_CACHE:
        return _ARTIFACT_SHA256_CACHE[key]
    digest = hashlib.sha256()
    with tarfile.open(source, "r:*") as archive:
        matches = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (member.name == member_name or Path(member.name).name == Path(member_name).name)
        ]
        if len(matches) != 1:
            raise ValueError(
                "archive member %s matched %d files in %s" % (member_name, len(matches), source)
            )
        extracted = archive.extractfile(matches[0])
        if extracted is None:
            raise ValueError("could not open archive member %s" % member_name)
        with extracted:
            if member_name.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=extracted, mode="rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            else:
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(block)
    result = digest.hexdigest()
    _ARTIFACT_SHA256_CACHE[key] = result
    return result


def _manifest_record(settings: "Settings", section_id: str):
    manifest = load_manifest(settings.manifest, require_folds=True)
    matches = [record for record in manifest if record.section_id == section_id]
    if len(matches) != 1:
        raise ValueError("section_id %s matched %d manifest rows" % (section_id, len(matches)))
    return matches[0]


def _validate_source_binding(
    *,
    label: str,
    artifacts: Sequence[str],
    hashes: Sequence[str],
    roles: Sequence[str],
    expected: Sequence[Tuple[str, str, str]],
) -> None:
    observed = tuple(zip(artifacts, hashes, roles))
    if observed != tuple(expected):
        raise ValueError(
            "%s source artifact/hash/role provenance differs from current inputs" % label
        )


def _metadata_batch(batch: HEIRTrainingBatch) -> Dict[str, object]:
    return {
        "sample_id": batch.sample_id,
        "bag_id": batch.bag_id,
        "donor_id": batch.donor_id,
        "block_id": batch.block_id,
        "analysis_role": batch.analysis_role,
        "source_artifacts": list(batch.source_artifacts),
        "source_sha256": list(batch.source_sha256),
        "source_roles": list(batch.source_roles),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strings(path: Path) -> Tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        values = tuple(
            line.strip().split("\t", 1)[0]
            for line in handle
            if line.strip() and not line.startswith("#")
        )
    if not values or len(set(values)) != len(values):
        raise ValueError("gene panel must contain unique, non-empty genes")
    return values


def _resolve_path(config_path: Path, value: object) -> Path:
    candidate = Path(str(value)).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (config_path.parent / candidate).resolve()
    )


def _nested(config: Mapping[str, object], *keys: str, default: object = None) -> object:
    current: object = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise PipelineError(
            "%s requires exactly one %s (%s), found %d" % (directory, label, pattern, len(matches))
        )
    return matches[0].resolve()


def _image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as error:
        raise PipelineError("Pillow is required to verify the H&E coordinate frame") from error
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _safe_torch_load(path: Path) -> Mapping[str, object]:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root is not a mapping")
    return payload


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(str(path), str(root))
    except ValueError:
        return str(path)


def _active_producers(outputs: Sequence[Path]) -> Tuple[Tuple[int, str], ...]:
    """Find live commands that name a stage output, including relative paths."""

    expected = {path.expanduser().resolve() for path in outputs}
    output_flags = {
        "--output",
        "--output-directory",
        "--nuclei-output",
        "--features-output",
        "--assignment-output",
        "--provenance-output",
        "--telemetry-output",
        "--train-output",
        "--validation-output",
        "--summary",
        "--tsv",
        "--metrics",
        "--reference-with-latent",
        "--fit-latent-transform",
    }
    found: List[Tuple[int, str]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return ()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        tokens = [value.decode("utf-8", errors="replace") for value in raw.split(b"\0") if value]
        matches = False
        for index, token in enumerate(tokens):
            if index == 0 or tokens[index - 1] not in output_flags:
                continue
            candidate = Path(token).expanduser()
            if not candidate.name or candidate.name not in {path.name for path in expected}:
                continue
            resolved = (
                candidate.resolve() if candidate.is_absolute() else (cwd / candidate).resolve()
            )
            if resolved in expected:
                matches = True
                break
        if matches:
            found.append((int(entry.name), shlex.join(tokens)))
    return tuple(sorted(found))


@dataclass(frozen=True)
class Settings:
    repository: Path
    config_path: Path
    manifest: Path
    panel: Path
    panel_sha256: str
    artifact_root: Path
    spaceranger: Path
    omiclip_checkpoint: Path
    omiclip_checkpoint_sha256: str
    feature_space_id: str
    latent_transform: Path
    latent_transform_sha256: str
    rna_decoder: Path
    rna_decoder_sha256: str
    ood_artifact: Path
    ood_artifact_sha256: str
    ood_calibration_quantile: float
    seed: int
    localcores: int
    localmem_gb: int
    segmentation_timeout_seconds: float
    feature_batch_size: int
    pathology_backend: str
    feature_scales: Tuple[float, ...]
    latent_dim: int
    maximum_prototypes: int
    minimum_cells: int
    include_rare_types: bool
    model_graph_hidden_dim: int
    model_graph_output_dim: int
    model_graph_layers: int
    model_trunk_hidden_dims: Tuple[int, ...]
    model_decoder_hidden_dims: Tuple[int, ...]
    model_dropout: float
    hard_type_routing: bool
    nonnegative_expression: bool
    epochs: int
    learning_rate: float
    adapter_learning_rate: float
    weight_decay: float
    warmup_fraction: float
    gradient_clip_norm: float
    bag_size: int
    reference_batch_size: int
    maximum_sample_cells: int
    early_stopping_patience: int
    graph_k: int
    graph_radius_um: float
    graph_max_degree: int
    block_size_um: float
    maximum_train_cells: int
    maximum_validation_cells: int
    latent_samples: int
    mc_chunk_size: int
    probability_threshold: float
    artifact_threshold: float
    bootstrap_resamples: int
    mpp: Mapping[str, float]
    source_histology_sha256: Mapping[str, str]

    @classmethod
    def load(
        cls,
        *,
        repository: Path,
        config_path: Path,
        artifact_root: Optional[Path],
        panel_override: Optional[Path],
        spaceranger_override: Optional[Path],
        omiclip_checkpoint_override: Optional[Path],
    ) -> "Settings":
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise PipelineError("experiment config root must be a mapping")
        manifest = _resolve_path(config_path, raw["manifest"])
        raw_panel = _nested(raw, "molecular_prior", "gene_panel")
        if raw_panel is None:
            raw_panel = _nested(raw, "targets", "gene_panel")
        if raw_panel is None and panel_override is None:
            raise PipelineError("experiment config has no molecular gene panel")
        panel = (
            panel_override.expanduser().resolve()
            if panel_override is not None
            else _resolve_path(config_path, raw_panel)
        )
        configured_panel_sha = str(_nested(raw, "molecular_prior", "gene_panel_sha256", default=""))
        expected_panel_sha = configured_panel_sha or EXPECTED_PANEL_SHA256
        spaceranger = (
            spaceranger_override.expanduser().resolve()
            if spaceranger_override is not None
            else Path(
                str(
                    _nested(
                        raw,
                        "segmentation",
                        "executable",
                        default="/storage/hackathon_2026/tools/spaceranger-4.1.0/bin/spaceranger",
                    )
                )
            )
            .expanduser()
            .resolve()
        )
        prior_root = repository / "artifacts" / "snpatho" / "prior500"
        latent_transform = _resolve_path(
            config_path,
            _nested(
                raw,
                "molecular_prior",
                "latent_transform",
                default=prior_root / "shared_svd.npz",
            ),
        )
        rna_decoder = _resolve_path(
            config_path,
            _nested(
                raw,
                "molecular_prior",
                "rna_decoder",
                default=prior_root / "rna_decoder.pt",
            ),
        )
        ood_artifact = _resolve_path(
            config_path,
            _nested(
                raw,
                "uncertainty",
                "ood_artifact",
                default=repository / "artifacts" / "snpatho" / "prior" / "omiclip_ood.npz",
            ),
        )
        feature_scales = tuple(
            float(value)
            for value in _nested(raw, "pathology_features", "patch_diameters_um", default=(32, 128))
        )
        mpp_raw = raw.get("physical_calibration_um_per_px", {})
        if not isinstance(mpp_raw, Mapping):
            raise PipelineError("physical calibration must be a sample mapping")
        mpp = {str(key): float(value) for key, value in mpp_raw.items()}
        checkpoint_environment = os.environ.get("HEIR_OMICLIP_CHECKPOINT", "").strip()
        pretrained_root = os.environ.get("HEIR_PRETRAINED_DIR", "").strip()
        if omiclip_checkpoint_override is not None:
            omiclip_checkpoint = omiclip_checkpoint_override.expanduser().resolve()
        elif checkpoint_environment:
            omiclip_checkpoint = Path(checkpoint_environment).expanduser().resolve()
        elif pretrained_root:
            omiclip_checkpoint = (
                Path(pretrained_root).expanduser() / "omiclip_loki" / "checkpoint.pt"
            ).resolve()
        else:
            omiclip_checkpoint = _resolve_path(
                config_path, _nested(raw, "pathology_features", "checkpoint")
            )
        return cls(
            repository=repository,
            config_path=config_path,
            manifest=manifest,
            panel=panel,
            panel_sha256=expected_panel_sha,
            artifact_root=(
                artifact_root.expanduser().resolve()
                if artifact_root is not None
                else (repository / "artifacts" / "snpatho").resolve()
            ),
            spaceranger=spaceranger,
            omiclip_checkpoint=omiclip_checkpoint,
            omiclip_checkpoint_sha256=str(
                _nested(raw, "pathology_features", "checkpoint_sha256", default="")
            ),
            feature_space_id=str(_nested(raw, "pathology_features", "feature_space_id")),
            latent_transform=latent_transform,
            latent_transform_sha256=str(
                _nested(raw, "molecular_prior", "latent_transform_sha256", default="")
            ),
            rna_decoder=rna_decoder,
            rna_decoder_sha256=str(
                _nested(raw, "molecular_prior", "rna_decoder_sha256", default="")
            ),
            ood_artifact=ood_artifact,
            ood_artifact_sha256=str(_nested(raw, "uncertainty", "ood_artifact_sha256", default="")),
            ood_calibration_quantile=float(
                _nested(raw, "uncertainty", "target_histology_calibration_quantile")
            ),
            seed=int(raw.get("seed", 17)),
            localcores=int(_nested(raw, "segmentation", "localcores", default=8)),
            localmem_gb=int(_nested(raw, "segmentation", "localmem_gb", default=24)),
            segmentation_timeout_seconds=float(
                _nested(raw, "segmentation", "timeout_seconds", default=7200)
            ),
            feature_batch_size=int(_nested(raw, "pathology_features", "batch_size", default=64)),
            pathology_backend=str(
                _nested(raw, "pathology_features", "backend", default="openslide")
            ),
            feature_scales=feature_scales,
            latent_dim=int(_nested(raw, "molecular_prior", "latent_dim", default=32)),
            maximum_prototypes=int(
                _nested(raw, "molecular_prior", "maximum_prototypes_per_type", default=10)
            ),
            minimum_cells=int(_nested(raw, "molecular_prior", "minimum_cells", default=30)),
            include_rare_types=bool(
                _nested(raw, "molecular_prior", "retain_rare_types", default=True)
            ),
            model_graph_hidden_dim=int(_nested(raw, "model", "graph_hidden_dim", default=256)),
            model_graph_output_dim=int(_nested(raw, "model", "graph_output_dim", default=256)),
            model_graph_layers=int(_nested(raw, "model", "graph_layers", default=3)),
            model_trunk_hidden_dims=tuple(
                int(value)
                for value in _nested(raw, "model", "trunk_hidden_dims", default=(512, 256))
            ),
            model_decoder_hidden_dims=tuple(
                int(value)
                for value in _nested(raw, "model", "decoder_hidden_dims", default=(128, 256))
            ),
            model_dropout=float(_nested(raw, "model", "dropout", default=0.05)),
            hard_type_routing=bool(_nested(raw, "model", "hard_type_routing", default=False)),
            nonnegative_expression=bool(
                _nested(raw, "model", "nonnegative_expression", default=True)
            ),
            epochs=int(_nested(raw, "optimization", "epochs", default=100)),
            learning_rate=float(_nested(raw, "optimization", "learning_rate", default=1.0e-4)),
            adapter_learning_rate=float(
                _nested(raw, "optimization", "adapter_learning_rate", default=1.0e-5)
            ),
            weight_decay=float(_nested(raw, "optimization", "weight_decay", default=1.0e-4)),
            warmup_fraction=float(_nested(raw, "optimization", "warmup_fraction", default=0.05)),
            gradient_clip_norm=float(
                _nested(raw, "optimization", "gradient_clip_norm", default=1.0)
            ),
            bag_size=int(_nested(raw, "optimization", "bag_size", default=2048)),
            reference_batch_size=int(
                _nested(raw, "optimization", "reference_batch_size", default=2048)
            ),
            maximum_sample_cells=int(
                _nested(raw, "optimization", "maximum_sample_cells", default=16384)
            ),
            early_stopping_patience=int(
                _nested(raw, "optimization", "early_stopping_patience", default=15)
            ),
            graph_k=int(_nested(raw, "graph", "knn", default=12)),
            graph_radius_um=float(_nested(raw, "graph", "radius_um", default=50)),
            graph_max_degree=int(_nested(raw, "graph", "maximum_degree", default=24)),
            block_size_um=float(_nested(raw, "optimization", "spatial_block_size_um", default=512)),
            maximum_train_cells=int(
                _nested(raw, "optimization", "maximum_train_cells", default=15000)
            ),
            maximum_validation_cells=int(
                _nested(raw, "optimization", "maximum_validation_cells", default=5000)
            ),
            latent_samples=int(_nested(raw, "inference", "latent_samples", default=20)),
            mc_chunk_size=int(_nested(raw, "inference", "mc_chunk_size", default=8)),
            probability_threshold=float(
                _nested(raw, "inference", "unknown_probability_threshold", default=0.60)
            ),
            artifact_threshold=float(
                _nested(raw, "inference", "artifact_probability_threshold", default=0.50)
            ),
            bootstrap_resamples=int(
                _nested(raw, "evaluation", "bootstrap_resamples", default=2000)
            ),
            mpp=mpp,
            source_histology_sha256={
                str(key): str(value)
                for key, value in dict(raw.get("full_resolution_histology_sha256", {})).items()
            },
        )

    def validate_fixed_inputs(self, *, require_runtime: bool) -> List[str]:
        problems: List[str] = []
        if self.bag_size < self.maximum_train_cells:
            problems.append(
                "optimization.bag_size (%d) is smaller than maximum_train_cells (%d)"
                % (self.bag_size, self.maximum_train_cells)
            )
        if self.maximum_sample_cells < self.maximum_train_cells:
            problems.append(
                "optimization.maximum_sample_cells (%d) is smaller than "
                "maximum_train_cells (%d)" % (self.maximum_sample_cells, self.maximum_train_cells)
            )
        model_widths = (
            self.model_graph_hidden_dim,
            self.model_graph_output_dim,
            self.model_graph_layers,
            *self.model_trunk_hidden_dims,
            *self.model_decoder_hidden_dims,
        )
        if any(value <= 0 for value in model_widths):
            problems.append("model architecture widths/layers must be positive")
        if not 0.0 <= self.model_dropout < 1.0:
            problems.append("model.dropout must lie in [0, 1)")
        if not self.nonnegative_expression:
            problems.append("canonical log1p-CPM prediction requires nonnegative_expression")
        if not 0.0 < self.ood_calibration_quantile < 1.0:
            problems.append("uncertainty.target_histology_calibration_quantile must lie in (0, 1)")
        for name, path in (
            ("config", self.config_path),
            ("manifest", self.manifest),
            ("gene panel", self.panel),
            ("OmiCLIP checkpoint", self.omiclip_checkpoint),
            ("B1 latent transform", self.latent_transform),
            ("B1 RNA decoder", self.rna_decoder),
            ("B1 OOD artifact", self.ood_artifact),
        ):
            if not path.is_file():
                problems.append("missing %s: %s" % (name, path))
        for name, path, expected in (
            (
                "OmiCLIP checkpoint",
                self.omiclip_checkpoint,
                self.omiclip_checkpoint_sha256,
            ),
            ("B1 latent transform", self.latent_transform, self.latent_transform_sha256),
            ("B1 RNA decoder", self.rna_decoder, self.rna_decoder_sha256),
            ("B1 OOD artifact", self.ood_artifact, self.ood_artifact_sha256),
        ):
            if name == "OmiCLIP checkpoint" and not require_runtime:
                # The published archive is 7.6 GB. A dry command plan checks
                # its path; execution performs the full configured hash once
                # and caches it for all three feature validators.
                continue
            if expected and path.is_file() and _sha256(path) != expected:
                problems.append("%s SHA-256 differs from the frozen config" % name)
        if self.panel.is_file():
            genes = _strings(self.panel)
            if len(genes) != 500:
                problems.append("frozen panel contains %d genes, expected 500" % len(genes))
            observed = _sha256(self.panel)
            if observed != self.panel_sha256:
                problems.append(
                    "frozen panel SHA-256 differs: expected %s, found %s"
                    % (self.panel_sha256, observed)
                )
        if self.latent_transform.is_file() and self.panel.is_file():
            try:
                _validate_latent_transform(
                    self.latent_transform,
                    _strings(self.panel),
                    latent_dim=self.latent_dim,
                )
            except Exception as error:
                problems.append("invalid B1 latent transform: %s" % error)
        if self.rna_decoder.is_file() and self.panel.is_file():
            try:
                _validate_rna_decoder(
                    self.rna_decoder,
                    _strings(self.panel),
                    self.latent_transform,
                    latent_dim=self.latent_dim,
                    decoder_hidden_dims=self.model_decoder_hidden_dims,
                    dropout=self.model_dropout,
                    nonnegative_output=self.nonnegative_expression,
                )
            except Exception as error:
                problems.append("invalid B1 RNA decoder: %s" % error)
        if self.ood_artifact.is_file():
            try:
                detector = MahalanobisOOD.from_npz(self.ood_artifact)
                if detector.feature_space_id != self.feature_space_id:
                    raise ValueError("feature space differs from frozen OmiCLIP config")
                if detector.training_donors != ("B1",):
                    raise ValueError("OOD training donors must be exactly B1")
            except Exception as error:
                problems.append("invalid B1 OOD artifact: %s" % error)
        if require_runtime:
            if not self.spaceranger.is_file() or not os.access(self.spaceranger, os.X_OK):
                problems.append("Space Ranger executable is unavailable: %s" % self.spaceranger)
            if not torch.cuda.is_available():
                problems.append("CUDA is unavailable; the frozen pipeline requires CUDA")
        return problems


@dataclass(frozen=True)
class SamplePaths:
    sample: str
    root: Path
    source_histology: Path
    pathology_image: Path
    segmentation_image: Path
    positions: Path
    scalefactors: Path
    reference_h5ad: Path
    reference_provenance: Path
    visium_h5ad: Path
    visium_provenance: Path
    raw_nuclei: Path
    segmentation_features: Path
    spaceranger_root: Path
    spaceranger_run: Path
    spaceranger_geojson: Path
    filtered_nuclei: Path
    capture_assignment: Path
    capture_provenance: Path
    pathology_features: Path
    pathology_telemetry: Path
    histology: Path
    calibrated_ood: Path
    calibrated_ood_provenance: Path
    train_histology: Path
    validation_histology: Path
    split_summary: Path
    reference: Path
    reference_latent: Path
    prototypes: Path
    train_batch: Path
    validation_batch: Path
    model_directory: Path
    checkpoint: Path
    training_history: Path
    training_telemetry: Path
    predictions: Path
    prediction_telemetry: Path
    truth: Path

    @classmethod
    def discover(cls, settings: Settings, sample: str) -> "SamplePaths":
        root = (settings.artifact_root / sample).resolve()
        histology = root / "histology"
        source_histology = _one(histology, "*high_res.tif", "full-resolution H&E")
        segmentation_image = (histology / (sample + "_pyramidal.tif")).resolve()
        if segmentation_image.is_file() and _image_size(segmentation_image) != _image_size(
            source_histology
        ):
            raise PipelineError(
                "Space Ranger pyramid and pathology H&E have different pixel dimensions for %s"
                % sample
            )
        positions = _one(histology, "*_tissue_positions.csv", "Visium tissue positions")
        scalefactors = _one(histology, "*_scalefactors_json.json", "Visium scalefactors")
        spaceranger_root = root / "spaceranger"
        spaceranger_run = spaceranger_root / ("snpatho_" + sample)
        model = root / ("model_v0_2_seed%d" % settings.seed)
        prediction = root / ("prediction_v0_2_seed%d" % settings.seed)
        locked = root / "locked_v0_2"
        return cls(
            sample=sample,
            root=root,
            source_histology=source_histology,
            pathology_image=segmentation_image,
            segmentation_image=segmentation_image,
            positions=positions,
            scalefactors=scalefactors,
            reference_h5ad=(root / "reference.h5ad").resolve(),
            reference_provenance=(root / "reference.h5ad.provenance.json").resolve(),
            visium_h5ad=(root / "visium_truth.h5ad").resolve(),
            visium_provenance=(root / "visium_truth.h5ad.provenance.json").resolve(),
            raw_nuclei=root / "nuclei_all.csv",
            segmentation_features=root / "geometry_all.npz",
            spaceranger_root=spaceranger_root,
            spaceranger_run=spaceranger_run,
            spaceranger_geojson=spaceranger_run / "outs" / "nucleus_segmentations.geojson",
            filtered_nuclei=root / "nuclei_capture.csv",
            capture_assignment=root / "capture_assignment.npz",
            capture_provenance=root / "capture_filter.json",
            pathology_features=root / "features_omiclip.npz",
            pathology_telemetry=root / "features_omiclip.telemetry.json",
            histology=root / "histology_full.npz",
            calibrated_ood=root / "ood_target_calibrated.npz",
            calibrated_ood_provenance=root / "ood_target_calibrated.provenance.json",
            train_histology=root / "histology_train.npz",
            validation_histology=root / "histology_validation.npz",
            split_summary=root / "histology_split.json",
            reference=root / "reference500.npz",
            reference_latent=root / "reference500_latent.npz",
            prototypes=root / "prototypes500.npz",
            train_batch=root / "batch_train.npz",
            validation_batch=root / "batch_validation.npz",
            model_directory=model,
            checkpoint=model / "heir.pt",
            training_history=model / "history.json",
            training_telemetry=model / "training.telemetry.json",
            predictions=prediction / "predictions.npz",
            prediction_telemetry=prediction / "inference.telemetry.json",
            truth=locked / "spatial_truth.locked.npz",
        )


@dataclass(frozen=True)
class Stage:
    name: str
    sample: str
    outputs: Tuple[Path, ...]
    requires: Tuple[Path, ...]
    command: Callable[[], Sequence[str]]
    validate: Callable[[], Mapping[str, object]]
    cuda: bool = False
    locked_target: bool = False
    telemetry_output: Optional[Path] = None
    guarded_directory: Optional[Path] = None

    @property
    def key(self) -> str:
        return "%s:%s" % (self.sample, self.name) if self.sample else self.name


class PipelineRunner:
    def __init__(
        self,
        *,
        repository: Path,
        execute: bool,
        status_path: Path,
        events_path: Path,
        logs_directory: Path,
        stop_after: Optional[str] = None,
    ) -> None:
        self.repository = repository
        self.execute = execute
        self.status_path = status_path
        self.events_path = events_path
        self.logs_directory = logs_directory
        self.stop_after = stop_after
        self.records: List[Dict[str, object]] = []
        self.locked_targets_enabled = False
        self.started_at = _utc_now()

    def unlock_locked_targets(self) -> None:
        self.locked_targets_enabled = True

    def should_stop(self, stage: str) -> bool:
        return self.stop_after == stage

    def _record(self, row: Dict[str, object]) -> None:
        row = {"timestamp": _utc_now(), **row}
        self.records.append(row)
        _append_jsonl(self.events_path, row)
        self.write_status()
        print(json.dumps(row, sort_keys=True), flush=True)

    def write_status(self) -> None:
        _atomic_json(
            self.status_path,
            {
                "schema": "heir.snpatho_orchestration_status.v1",
                "mode": "execute" if self.execute else "dry_run",
                "started_at": self.started_at,
                "updated_at": _utc_now(),
                "locked_targets_enabled": self.locked_targets_enabled,
                "records": self.records,
            },
        )

    def run(self, stage: Stage) -> str:
        if stage.locked_target and self.execute and not self.locked_targets_enabled:
            raise PipelineError("locked target stage requested before predictions were frozen")
        active_targets = stage.outputs + (
            () if stage.guarded_directory is None else (stage.guarded_directory,)
        )
        active = _active_producers(active_targets)
        if active:
            message = "stage outputs are being produced by live process(es): %s" % "; ".join(
                "pid=%d %s" % value for value in active
            )
            self._record(
                {
                    "stage": stage.name,
                    "sample": stage.sample,
                    "status": "blocked_active",
                    "message": message,
                }
            )
            if self.execute:
                raise PipelineError(message)
            return "blocked_active"
        existing = tuple(output.exists() for output in stage.outputs)
        if any(existing):
            if not all(existing):
                message = "partial outputs exist; move them aside before resuming: %s" % ", ".join(
                    str(path) for path, present in zip(stage.outputs, existing) if present
                )
                self._record(
                    {
                        "stage": stage.name,
                        "sample": stage.sample,
                        "status": "blocked_partial",
                        "message": message,
                    }
                )
                if self.execute:
                    raise PipelineError(message)
                return "blocked_partial"
            try:
                details = dict(stage.validate())
            except Exception as error:
                message = "existing outputs failed validation: %s" % error
                self._record(
                    {
                        "stage": stage.name,
                        "sample": stage.sample,
                        "status": "blocked_invalid",
                        "message": message,
                    }
                )
                if self.execute:
                    raise PipelineError(message) from error
                return "blocked_invalid"
            self._record(
                {
                    "stage": stage.name,
                    "sample": stage.sample,
                    "status": "skipped_valid",
                    "outputs": [str(path) for path in stage.outputs],
                    "validation": details,
                }
            )
            return "skipped_valid"

        if stage.guarded_directory is not None and stage.guarded_directory.exists():
            if any(stage.guarded_directory.iterdir()):
                message = "non-empty stage directory exists without valid outputs: %s" % (
                    stage.guarded_directory
                )
                self._record(
                    {
                        "stage": stage.name,
                        "sample": stage.sample,
                        "status": "blocked_partial",
                        "message": message,
                    }
                )
                if self.execute:
                    raise PipelineError(message)
                return "blocked_partial"

        try:
            command = tuple(str(value) for value in stage.command())
        except Exception as error:
            self._record(
                {
                    "stage": stage.name,
                    "sample": stage.sample,
                    "status": "blocked_precondition",
                    "message": str(error),
                }
            )
            if self.execute:
                raise PipelineError(str(error)) from error
            return "blocked_precondition"
        if not self.execute:
            self._record(
                {
                    "stage": stage.name,
                    "sample": stage.sample,
                    "status": "planned",
                    "command": shlex.join(command),
                    "outputs": [str(path) for path in stage.outputs],
                    "locked_target": stage.locked_target,
                }
            )
            return "planned"

        missing = [path for path in stage.requires if not path.exists()]
        if missing:
            raise PipelineError(
                "%s prerequisites are missing: %s"
                % (stage.key, ", ".join(str(path) for path in missing))
            )
        for output in stage.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        self.logs_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = self.logs_directory / ("%s.%s.log" % (stage.key.replace(":", "."), stamp))
        started = time.perf_counter()
        peak_gpu_mib: List[int] = []
        stop_monitor = threading.Event()
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(self.repository),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            monitor = None
            if stage.cuda:
                monitor = threading.Thread(
                    target=_monitor_gpu_memory,
                    args=(process.pid, stop_monitor, peak_gpu_mib),
                    daemon=True,
                )
                monitor.start()
            returncode = process.wait()
            stop_monitor.set()
            if monitor is not None:
                monitor.join(timeout=2.0)
        elapsed = time.perf_counter() - started
        if returncode != 0:
            self._record(
                {
                    "stage": stage.name,
                    "sample": stage.sample,
                    "status": "failed",
                    "returncode": returncode,
                    "wall_seconds": elapsed,
                    "log": str(log_path),
                    "command": shlex.join(command),
                }
            )
            raise PipelineError("%s failed; inspect %s" % (stage.key, log_path))
        if stage.telemetry_output is not None:
            _atomic_json(
                stage.telemetry_output,
                {
                    "schema": "heir.training_telemetry.v1",
                    "stage": stage.name,
                    "sample": stage.sample,
                    "wall_seconds": elapsed,
                    "cuda_requested": stage.cuda,
                    "process_peak_cuda_memory_mib": max(peak_gpu_mib, default=None),
                    "command": list(command),
                    "completed_at": _utc_now(),
                },
            )
        try:
            details = dict(stage.validate())
        except Exception as error:
            self._record(
                {
                    "stage": stage.name,
                    "sample": stage.sample,
                    "status": "failed_validation",
                    "wall_seconds": elapsed,
                    "log": str(log_path),
                    "message": str(error),
                }
            )
            raise PipelineError("%s outputs failed validation: %s" % (stage.key, error)) from error
        self._record(
            {
                "stage": stage.name,
                "sample": stage.sample,
                "status": "completed",
                "wall_seconds": elapsed,
                "process_peak_cuda_memory_mib": max(peak_gpu_mib, default=None),
                "log": str(log_path),
                "command": shlex.join(command),
                "outputs": [str(path) for path in stage.outputs],
                "validation": details,
            }
        )
        return "completed"


def _monitor_gpu_memory(pid: int, stop: threading.Event, measurements: List[int]) -> None:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return
    while not stop.wait(0.5):
        result = subprocess.run(
            [
                binary,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) == 2 and fields[0] == str(pid):
                try:
                    measurements.append(int(fields[1]))
                except ValueError:
                    pass


def _validate_latent_transform(path: Path, genes: Sequence[str], *, latent_dim: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if str(np.asarray(archive["__contract__"]).item()) != "heir.truncated_svd_transform":
            raise ValueError("unexpected latent-transform contract")
        if int(np.asarray(archive["__version__"]).item()) != 2:
            raise ValueError("latent transform must use provenance contract v2")
        observed_genes = tuple(str(value) for value in archive["gene_ids"].tolist())
        components = np.asarray(archive["components"], dtype=np.float32)
        donors = tuple(str(value) for value in archive["training_donors"].tolist())
        role = str(np.asarray(archive["analysis_role"]).item())
    if observed_genes != tuple(genes):
        raise ValueError("latent-transform gene order differs from the frozen panel")
    if components.shape != (latent_dim, len(genes)) or not np.isfinite(components).all():
        raise ValueError("latent-transform component shape/content is invalid")
    if donors != ("B1",) or role not in {"train", "training", "development", "pretraining"}:
        raise ValueError("latent transform must be fitted on development donor B1")


def _validate_rna_decoder(
    path: Path,
    genes: Sequence[str],
    latent_transform: Path,
    *,
    latent_dim: int,
    decoder_hidden_dims: Sequence[int],
    dropout: float,
    nonnegative_output: bool,
) -> None:
    payload = _safe_torch_load(path)
    metadata = payload.get("metadata")
    config = payload.get("config")
    if not isinstance(metadata, Mapping) or not isinstance(config, Mapping):
        raise ValueError("RNA decoder lacks config/metadata")
    if tuple(str(value) for value in metadata.get("gene_names", ())) != tuple(genes):
        raise ValueError("RNA decoder genes differ from the frozen panel")
    if tuple(str(value) for value in metadata.get("training_donors", ())) != ("B1",):
        raise ValueError("RNA decoder training donor must be B1")
    if str(metadata.get("latent_transform_sha256", "")) != _sha256(latent_transform):
        raise ValueError("RNA decoder was trained with a different latent transform")
    if (
        int(config.get("input_dim", -1)) != len(genes)
        or int(config.get("latent_dim", -1)) != latent_dim
    ):
        raise ValueError("RNA decoder dimensions differ from the frozen experiment")
    if tuple(int(value) for value in config.get("decoder_hidden_dims", ())) != tuple(
        decoder_hidden_dims
    ):
        raise ValueError("RNA decoder hidden widths differ from the frozen experiment")
    if abs(float(config.get("dropout", -1.0)) - dropout) > 1.0e-12:
        raise ValueError("RNA decoder dropout differs from the frozen experiment")
    if bool(config.get("nonnegative_output", False)) != nonnegative_output:
        raise ValueError("RNA decoder output support differs from the frozen experiment")


def _validate_segmentation(paths: SamplePaths) -> Mapping[str, object]:
    nuclei = load_nuclei(paths.raw_nuclei)
    features = load_feature_bundle(paths.segmentation_features, expected_ids=nuclei.source_ids)
    with np.load(paths.segmentation_features, allow_pickle=False) as archive:
        method = str(np.asarray(archive["segmentation_method"]).item())
        version = str(np.asarray(archive["segmentation_version"]).item())
        source_sha256 = str(np.asarray(archive["segmentation_source_sha256"]).item())
    if method != "10x-spaceranger-segment" or not version.startswith("4."):
        raise ValueError("segmentation is not a Space Ranger 4.x artifact")
    if len(nuclei) < 2 or features.features.shape[1] != 10:
        raise ValueError("segmentation contains too few nuclei or wrong morphology width")
    if paths.spaceranger_geojson.is_file() and source_sha256 != _sha256(paths.spaceranger_geojson):
        raise ValueError("segmentation is bound to a different Space Ranger GeoJSON")
    return {"nuclei": len(nuclei), "method": method, "version": version}


def _validate_capture(paths: SamplePaths) -> Mapping[str, object]:
    nuclei = load_nuclei(paths.filtered_nuclei)
    with np.load(paths.capture_assignment, allow_pickle=False) as archive:
        if str(np.asarray(archive["__contract__"]).item()) != "heir.visium_capture_filter":
            raise ValueError("capture assignment contract is invalid")
        if not bool(np.asarray(archive["geometry_only"]).item()) or bool(
            np.asarray(archive["target_expression_accessed"]).item()
        ):
            raise ValueError("capture filter is not target-expression-free")
        ids = tuple(str(value) for value in archive["nucleus_ids"].tolist())
        if ids != tuple(nuclei.source_ids.tolist()):
            raise ValueError("capture assignment and filtered CSV IDs differ")
        if str(np.asarray(archive["filtered_csv_sha256"]).item()) != _sha256(paths.filtered_nuclei):
            raise ValueError("capture assignment is bound to a different filtered CSV")
    with paths.capture_provenance.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not provenance.get("geometry_only") or provenance.get("target_expression_accessed"):
        raise ValueError("capture-filter provenance does not preserve target isolation")
    if provenance["outputs"]["assignment"]["sha256"] != _sha256(paths.capture_assignment):
        raise ValueError("capture-filter provenance hash differs from assignment")
    for name, source in (
        ("nuclei", paths.raw_nuclei),
        ("positions", paths.positions),
        ("scalefactors", paths.scalefactors),
    ):
        if provenance["inputs"][name]["sha256"] != _sha256(source):
            raise ValueError("capture-filter provenance differs from %s input" % name)
    return {
        "retained_nuclei": len(nuclei),
        "source_nuclei": int(provenance["source_nuclei"]),
        "geometry_only": True,
    }


def _validate_pathology(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    nuclei = load_nuclei(paths.filtered_nuclei)
    bundle = load_feature_bundle(paths.pathology_features, expected_ids=nuclei.source_ids)
    with np.load(paths.pathology_features, allow_pickle=False) as archive:
        if str(np.asarray(archive["__contract__"]).item()) != ("heir.nucleus_pathology_features"):
            raise ValueError("pathology feature contract is invalid")
        feature_space = str(np.asarray(archive["feature_space_id"]).item())
        if feature_space != settings.feature_space_id:
            raise ValueError("pathology feature space differs from the frozen config")
        if str(np.asarray(archive["slide_sha256"]).item()) != _sha256(paths.pathology_image):
            raise ValueError("pathology features are bound to a different H&E")
        if str(np.asarray(archive["nuclei_sha256"]).item()) != _sha256(paths.filtered_nuclei):
            raise ValueError("pathology features are bound to a different nucleus table")
        descriptor = json.loads(str(np.asarray(archive["encoder_descriptor_json"]).item()))
    if descriptor.get("name") != "omiclip-loki-coca-vit-l-14":
        raise ValueError("default pathology encoder must be OmiCLIP/Loki")
    expected_checkpoint = settings.omiclip_checkpoint_sha256 or _sha256(settings.omiclip_checkpoint)
    if descriptor.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("pathology features use a different OmiCLIP checkpoint")
    with paths.pathology_telemetry.open("r", encoding="utf-8") as handle:
        telemetry = json.load(handle)
    if telemetry.get("feature_space_id") != settings.feature_space_id:
        raise ValueError("pathology telemetry and features disagree")
    return {
        "nuclei": len(bundle),
        "feature_width": int(bundle.features.shape[1]),
        "feature_space_id": feature_space,
        "device": telemetry.get("telemetry", {}).get("device_name"),
    }


def _validate_histology(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    bag = HistologyBag.load_npz(paths.histology)
    if (bag.slide_id, bag.sample_id, bag.donor_id) != (paths.sample,) * 3:
        raise ValueError("HistologyBag identity differs from the selected sample")
    if bag.feature_space_id != settings.feature_space_id:
        raise ValueError("HistologyBag feature space differs from config")
    if bag.nuclei_source_sha256 != _sha256(paths.filtered_nuclei):
        raise ValueError("HistologyBag is bound to a different nucleus table")
    if bag.feature_source_sha256 != _sha256(paths.pathology_features):
        raise ValueError("HistologyBag is bound to different pathology features")
    observed_histology_sha256 = _sha256(paths.source_histology)
    expected_histology_sha256 = settings.source_histology_sha256.get(paths.sample, "")
    if not expected_histology_sha256:
        raise ValueError("frozen full-resolution H&E SHA-256 is missing")
    if observed_histology_sha256 != expected_histology_sha256:
        raise ValueError("local full-resolution H&E differs from the frozen SHA-256")
    if bag.histology_source_sha256 != observed_histology_sha256:
        raise ValueError("HistologyBag is bound to a different H&E")
    detector = MahalanobisOOD.from_npz(settings.ood_artifact)
    if bag.features.shape[1] != detector.mean.shape[0]:
        raise ValueError("HistologyBag width differs from the B1-trained OOD detector")
    return {
        "nuclei": bag.n_nuclei,
        "feature_width": int(bag.features.shape[1]),
        "edges": int(bag.edge_index.shape[1]),
    }


def _validate_calibrated_ood(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    base = MahalanobisOOD.from_npz(settings.ood_artifact)
    calibrated = MahalanobisOOD.from_npz(paths.calibrated_ood)
    bag = HistologyBag.load_npz(paths.histology)
    if base.mean is None or base.precision is None:
        raise ValueError("B1 OOD artifact lacks fitted Mahalanobis parameters")
    if calibrated.mean is None or calibrated.precision is None:
        raise ValueError("target-calibrated OOD artifact lacks Mahalanobis parameters")
    if not np.array_equal(calibrated.mean, base.mean) or not np.array_equal(
        calibrated.precision, base.precision
    ):
        raise ValueError("target OOD calibration changed the B1 mean or precision")
    if (
        calibrated.training_donors != base.training_donors
        or calibrated.source_sha256 != base.source_sha256
        or calibrated.feature_space_id != base.feature_space_id
    ):
        raise ValueError("target OOD calibration changed B1 training provenance")
    if abs(calibrated.quantile - settings.ood_calibration_quantile) > 1.0e-12:
        raise ValueError("target OOD quantile differs from the frozen config")
    if bag.sample_id != paths.sample or bag.feature_space_id != calibrated.feature_space_id:
        raise ValueError("target OOD calibration and HistologyBag identity differ")
    if bag.features.shape[1] != calibrated.mean.shape[0]:
        raise ValueError("target OOD calibration width differs from HistologyBag")

    required_npz = {
        "calibration_contract",
        "calibration_version",
        "base_ood_sha256",
        "histology_sha256",
        "sample_id",
        "target_expression_accessed",
        "score_count",
        "score_minimum",
        "score_maximum",
        "score_mean",
        "score_standard_deviation",
        "score_median",
    }
    with np.load(paths.calibrated_ood, allow_pickle=False) as archive:
        missing = sorted(required_npz - set(archive.files))
        if missing:
            raise ValueError("target OOD artifact is missing: %s" % ", ".join(missing))
        if (
            str(np.asarray(archive["calibration_contract"]).item())
            != ("heir.target_histology_ood_calibration")
            or int(np.asarray(archive["calibration_version"]).item()) != 1
        ):
            raise ValueError("target OOD calibration contract is invalid")
        if bool(np.asarray(archive["target_expression_accessed"]).item()):
            raise ValueError("target OOD calibration accessed target expression")
        if str(np.asarray(archive["sample_id"]).item()) != paths.sample:
            raise ValueError("target OOD calibration sample identity differs")
        if str(np.asarray(archive["base_ood_sha256"]).item()) != _sha256(settings.ood_artifact):
            raise ValueError("target OOD calibration is bound to another B1 detector")
        if str(np.asarray(archive["histology_sha256"]).item()) != _sha256(paths.histology):
            raise ValueError("target OOD calibration is bound to another HistologyBag")
        if int(np.asarray(archive["score_count"]).item()) != bag.n_nuclei:
            raise ValueError("target OOD score count differs from HistologyBag")

    with paths.calibrated_ood_provenance.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("schema") != "heir.target_histology_ood_calibration.v1":
        raise ValueError("target OOD provenance schema is invalid")
    if (
        provenance.get("sample_id") != paths.sample
        or provenance.get("target_expression_accessed") is not False
    ):
        raise ValueError("target OOD provenance violates sample/expression isolation")
    if provenance.get("calibration_input_modality") != "target_histology_features_only":
        raise ValueError("target OOD provenance names an invalid calibration modality")
    if (
        provenance.get("quantile") != settings.ood_calibration_quantile
        or provenance.get("threshold") != calibrated.threshold
    ):
        raise ValueError("target OOD provenance threshold differs from the detector")
    inputs = provenance.get("inputs")
    output = provenance.get("output")
    copied = provenance.get("copied_training_provenance")
    score_stats = provenance.get("score_stats")
    if not all(isinstance(value, Mapping) for value in (inputs, output, copied, score_stats)):
        raise ValueError("target OOD provenance sections are malformed")
    assert isinstance(inputs, Mapping)
    base_input = inputs.get("base_ood")
    histology_input = inputs.get("histology")
    if not isinstance(base_input, Mapping) or base_input.get("sha256") != _sha256(
        settings.ood_artifact
    ):
        raise ValueError("target OOD provenance differs from the B1 detector")
    if not isinstance(histology_input, Mapping) or histology_input.get("sha256") != _sha256(
        paths.histology
    ):
        raise ValueError("target OOD provenance differs from HistologyBag")
    assert isinstance(output, Mapping)
    if output.get("sha256") != _sha256(paths.calibrated_ood):
        raise ValueError("target OOD provenance output hash differs")
    assert isinstance(copied, Mapping)
    if (
        tuple(copied.get("training_donors", ())) != base.training_donors
        or tuple(copied.get("source_sha256", ())) != base.source_sha256
        or copied.get("feature_space_id") != base.feature_space_id
    ):
        raise ValueError("target OOD provenance does not preserve B1 training provenance")
    assert isinstance(score_stats, Mapping)
    if (
        score_stats.get("count") != bag.n_nuclei
        or score_stats.get("calibration_quantile_value") != calibrated.threshold
    ):
        raise ValueError("target OOD provenance score statistics differ")
    numeric_stats = (
        score_stats.get("minimum"),
        score_stats.get("maximum"),
        score_stats.get("mean"),
        score_stats.get("standard_deviation"),
        score_stats.get("median"),
    )
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in numeric_stats):
        raise ValueError("target OOD score statistics must be finite")
    return {
        "nuclei": bag.n_nuclei,
        "quantile": calibrated.quantile,
        "threshold": calibrated.threshold,
        "base_ood_sha256": _sha256(settings.ood_artifact),
        "calibrated_ood_sha256": _sha256(paths.calibrated_ood),
        "target_expression_accessed": False,
    }


def _validate_split(paths: SamplePaths) -> Mapping[str, object]:
    full = HistologyBag.load_npz(paths.histology)
    train = HistologyBag.load_npz(paths.train_histology)
    validation = HistologyBag.load_npz(paths.validation_histology)
    train_ids = set(train.nucleus_ids.tolist())
    validation_ids = set(validation.nucleus_ids.tolist())
    full_ids = set(full.nucleus_ids.tolist())
    if train_ids & validation_ids:
        raise ValueError("spatial train/validation bags overlap nuclei")
    if not train_ids or not validation_ids or not (train_ids | validation_ids).issubset(full_ids):
        raise ValueError("spatial split is empty or contains foreign nuclei")
    with paths.split_summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if int(summary.get("nucleus_overlap", -1)) != 0:
        raise ValueError("spatial split summary reports overlap")
    return {"train_nuclei": len(train_ids), "validation_nuclei": len(validation_ids)}


def _validate_reference(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    reference = RNAReference.load_npz(paths.reference)
    genes = _strings(settings.panel)
    if tuple(reference.gene_ids.tolist()) != genes:
        raise ValueError("RNAReference gene order differs from frozen panel")
    if set(reference.donor_ids.tolist()) != {paths.sample} or set(
        reference.sample_ids.tolist()
    ) != {paths.sample}:
        raise ValueError("RNAReference identity differs from selected snPATHO sample")
    if reference.source_count_sha256 != _sha256(paths.reference_h5ad):
        raise ValueError("RNAReference is bound to a different H5AD")
    return {
        "cells": int(reference.shape[0]),
        "genes": int(reference.shape[1]),
        "types": len(set(reference.cell_type_labels.tolist())),
    }


def _validate_prototypes(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    reference = RNAReference.load_npz(paths.reference_latent)
    prototypes = PrototypeSet.load_npz(paths.prototypes)
    latent_sha = _sha256(settings.latent_transform)
    if tuple(reference.gene_ids.tolist()) != _strings(settings.panel):
        raise ValueError("latent RNAReference gene order differs from panel")
    if reference.latent.shape[1] != settings.latent_dim:
        raise ValueError("latent RNAReference has the wrong latent width")
    if prototypes.donor_id != paths.sample or prototypes.block_id != (paths.sample + "_FFPE"):
        raise ValueError("PrototypeSet donor/block identity is invalid")
    if prototypes.source_reference_sha256 != _sha256(paths.reference_latent):
        raise ValueError("PrototypeSet was built from a different RNAReference")
    if prototypes.latent_transform_sha256 != latent_sha or prototypes.latent_training_donors != (
        "B1",
    ):
        raise ValueError("PrototypeSet does not use the frozen B1 latent transform")
    if prototypes.latent_space_id != "sha256:" + latent_sha:
        raise ValueError("PrototypeSet latent identity is invalid")
    return {
        "prototypes": len(prototypes.prototype_ids),
        "types": len(set(prototypes.cell_type_labels.tolist())),
        "latent_dim": int(prototypes.means.shape[1]),
    }


def _validate_batches(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    train = HEIRTrainingBatch.load_npz(paths.train_batch)
    validation = HEIRTrainingBatch.load_npz(paths.validation_batch)
    record = _manifest_record(settings, paths.sample)
    reference = RNAReference.load_npz(paths.reference_latent)
    prototypes = PrototypeSet.load_npz(paths.prototypes)
    detector = MahalanobisOOD.from_npz(paths.calibrated_ood)
    expected_types = tuple(sorted(set(str(value) for value in prototypes.cell_type_labels)))
    expected_molecular_donors = tuple(
        sorted(set(prototypes.latent_training_donors) | set(detector.training_donors))
    )
    for role, batch, bag_path in (
        ("train", train, paths.train_histology),
        ("validation", validation, paths.validation_histology),
    ):
        batch.validate(TrainingStage.PERSONALIZED)
        bag = HistologyBag.load_npz(bag_path)
        if tuple(batch.nucleus_ids) != tuple(bag.nucleus_ids.tolist()):
            raise ValueError("training batch and spatial bag nucleus IDs differ")
        if batch.morphology.shape != bag.features.shape:
            raise ValueError("training batch morphology shape differs from its spatial bag")
        if tuple(batch.gene_names) != _strings(settings.panel):
            raise ValueError("training batch gene order differs from panel")
        if tuple(batch.type_names) != expected_types:
            raise ValueError("training batch type ontology differs from current prototypes")
        if tuple(batch.prototype_ids) != tuple(str(value) for value in prototypes.prototype_ids):
            raise ValueError("training batch prototype identities differ from current prototypes")
        expected_identity = (
            record.specimen_id,
            "%s_%s" % (paths.sample, role),
            record.donor_id,
            record.block_id,
            record.analysis_role,
        )
        observed_identity = (
            batch.sample_id,
            batch.bag_id,
            batch.donor_id,
            batch.block_id,
            batch.analysis_role,
        )
        if observed_identity != expected_identity:
            raise ValueError("training batch sample/bag/donor/block/role identity is invalid")
        if batch.latent_space_id != reference.latent_space_id or (
            batch.latent_space_id != prototypes.latent_space_id
        ):
            raise ValueError("training batch latent identity differs from RNA/prototypes")
        if batch.feature_space_id != settings.feature_space_id:
            raise ValueError("training batch pathology feature space differs from config")
        if batch.expression_space_id != EXPRESSION_SPACE_ID:
            raise ValueError("training batch expression space is not canonical")
        if tuple(batch.molecular_training_donors) != expected_molecular_donors:
            raise ValueError("training batch molecular teacher donors differ from B1 artifacts")
        if paths.sample in set(batch.molecular_training_donors):
            raise ValueError("development molecular artifacts overlap the target donor")
        expected_sources = (
            (str(bag_path.resolve()), _sha256(bag_path), "sample_assay"),
            (str(paths.prototypes.resolve()), _sha256(paths.prototypes), "sample_assay"),
            (
                str(paths.reference_latent.resolve()),
                _sha256(paths.reference_latent),
                "sample_assay",
            ),
            (
                str(paths.calibrated_ood.resolve()),
                _sha256(paths.calibrated_ood),
                "shared_teacher",
            ),
        )
        _validate_source_binding(
            label="%s training batch" % role,
            artifacts=batch.source_artifacts,
            hashes=batch.source_sha256,
            roles=batch.source_roles,
            expected=expected_sources,
        )
        if batch.ood_mask is None:
            raise ValueError("training batch lacks the target-calibrated OOD mask")
        expected_ood = detector.score(bag.features) > detector.threshold
        if not np.array_equal(batch.ood_mask.detach().cpu().numpy(), expected_ood):
            raise ValueError("training batch OOD mask differs from the calibrated detector")
        if batch.unknown_targets is None or not np.array_equal(
            batch.unknown_targets.detach().cpu().numpy(), expected_ood.astype(np.float32)
        ):
            raise ValueError("training batch unknown targets differ from its OOD mask")
    return {
        "train_nuclei": int(train.morphology.shape[0]),
        "validation_nuclei": int(validation.morphology.shape[0]),
        "genes": len(train.gene_names),
    }


def _validate_model(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    payload = _safe_torch_load(paths.checkpoint)
    model = HEIRModel.from_checkpoint(payload)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("HEIR checkpoint lacks metadata")
    train = HEIRTrainingBatch.load_npz(paths.train_batch)
    validation = HEIRTrainingBatch.load_npz(paths.validation_batch)
    if metadata.get("schema") != "heir.trained_model.v1":
        raise ValueError("HEIR checkpoint schema is invalid")
    genes = _strings(settings.panel)
    if tuple(str(value) for value in metadata.get("gene_names", ())) != genes:
        raise ValueError("HEIR checkpoint genes differ from panel")
    if tuple(str(value) for value in metadata.get("type_names", ())) != tuple(train.type_names):
        raise ValueError("HEIR checkpoint type ontology differs from its training batch")
    if tuple(str(value) for value in metadata.get("parent_type_names", ())) != ():
        raise ValueError("snPATHO checkpoint unexpectedly contains a parent ontology")
    if metadata.get("training_stage") != TrainingStage.PERSONALIZED.value:
        raise ValueError("HEIR checkpoint training stage is not personalized")
    if metadata.get("latent_space_id") != train.latent_space_id:
        raise ValueError("HEIR checkpoint latent space differs from its training batch")
    if metadata.get("feature_space_id") != settings.feature_space_id:
        raise ValueError("HEIR checkpoint feature space differs from config")
    if metadata.get("expression_space_id") != EXPRESSION_SPACE_ID:
        raise ValueError("HEIR checkpoint expression space is not canonical")
    if str(metadata.get("scgpt_space_id", "")):
        raise ValueError("snPATHO checkpoint unexpectedly uses scGPT supervision")
    if metadata.get("rna_vae_sha256") != _sha256(settings.rna_decoder):
        raise ValueError("HEIR checkpoint uses a different B1 RNA decoder")
    if metadata.get("rna_vae_checkpoint") != str(settings.rna_decoder.resolve()):
        raise ValueError("HEIR checkpoint names a different B1 RNA decoder path")
    if tuple(str(value) for value in metadata.get("rna_vae_training_donors", ())) != ("B1",):
        raise ValueError("HEIR checkpoint RNA decoder donor provenance is invalid")
    if (
        metadata.get("initial_heir_checkpoint") is not None
        or metadata.get("initial_heir_sha256") is not None
    ):
        raise ValueError("snPATHO checkpoint unexpectedly descends from another HEIR model")
    if tuple(metadata.get("initial_heir_training_donors", ())) != ():
        raise ValueError("snPATHO checkpoint has unexpected initial-model donors")
    if tuple(str(value) for value in metadata.get("training_donors", ())) != (paths.sample,):
        raise ValueError("HEIR checkpoint training donor is invalid")
    if int(metadata.get("seed", -1)) != settings.seed:
        raise ValueError("HEIR checkpoint seed differs from the frozen experiment")
    if metadata.get("training_batches") != [_metadata_batch(train)]:
        raise ValueError("HEIR checkpoint training-batch provenance differs from current batch")
    if metadata.get("validation_batches") != [_metadata_batch(validation)]:
        raise ValueError("HEIR checkpoint validation-batch provenance differs from current batch")
    config = model.config
    if (
        config.morphology_dim != int(train.morphology.shape[1])
        or config.num_cell_types != len(train.type_names)
        or config.expression_dim != len(genes)
        or config.latent_dim != settings.latent_dim
        or config.graph_hidden_dim != settings.model_graph_hidden_dim
        or config.graph_output_dim != settings.model_graph_output_dim
        or config.graph_layers != settings.model_graph_layers
        or tuple(config.trunk_hidden_dims) != settings.model_trunk_hidden_dims
        or tuple(config.decoder_hidden_dims) != settings.model_decoder_hidden_dims
        or abs(config.dropout - settings.model_dropout) > 1.0e-12
        or config.hard_type_routing != settings.hard_type_routing
        or abs(config.abstain_threshold - settings.probability_threshold) > 1.0e-12
        or config.nonnegative_expression != settings.nonnegative_expression
    ):
        raise ValueError("HEIR checkpoint architecture differs from frozen batch dimensions")
    with paths.training_history.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, Mapping) or not isinstance(history.get("history"), list):
        raise ValueError("training history is malformed")
    if history.get("best_epoch") != metadata.get("best_epoch") or history.get(
        "best_validation_loss"
    ) != metadata.get("best_validation_loss"):
        raise ValueError("training history and checkpoint best model disagree")
    with paths.training_telemetry.open("r", encoding="utf-8") as handle:
        telemetry = json.load(handle)
    if telemetry.get("schema") != "heir.training_telemetry.v1":
        raise ValueError("training telemetry schema is invalid")
    if telemetry.get("stage") != "train" or telemetry.get("sample") != paths.sample:
        raise ValueError("training telemetry stage/sample identity is invalid")
    if telemetry.get("cuda_requested") is not True:
        raise ValueError("training telemetry does not record the frozen CUDA path")
    wall_seconds = telemetry.get("wall_seconds")
    if (
        not isinstance(wall_seconds, (int, float))
        or not np.isfinite(wall_seconds)
        or (wall_seconds <= 0)
    ):
        raise ValueError("training telemetry wall time is invalid")
    expected_command = list(_stage(settings, paths, "train").command())
    if telemetry.get("command") != expected_command:
        raise ValueError("training telemetry command differs from the frozen experiment")
    return {
        "checkpoint_sha256": _sha256(paths.checkpoint),
        "best_epoch": history.get("best_epoch"),
        "best_validation_loss": history.get("best_validation_loss"),
        "wall_seconds": wall_seconds,
    }


def _validate_prediction(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    prediction = PredictionBundle.from_npz(paths.predictions)
    prediction.validate(require_provenance=True)
    bag = HistologyBag.load_npz(paths.histology)
    prototypes = PrototypeSet.load_npz(paths.prototypes)
    detector = MahalanobisOOD.from_npz(paths.calibrated_ood)
    checkpoint = _safe_torch_load(paths.checkpoint)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("prediction checkpoint lacks provenance metadata")
    if not np.array_equal(prediction.nucleus_ids.astype(str), bag.nucleus_ids.astype(str)):
        raise ValueError("prediction and full HistologyBag nucleus order differ")
    if not np.array_equal(prediction.coordinates_um, bag.coordinates_um.astype(np.float32)):
        raise ValueError("prediction coordinates differ from the full HistologyBag")
    if tuple(prediction.gene_names.tolist()) != _strings(settings.panel):
        raise ValueError("prediction genes differ from frozen panel")
    if tuple(prediction.type_names.tolist()) != tuple(
        str(value) for value in metadata.get("type_names", ())
    ):
        raise ValueError("prediction type ontology differs from the HEIR checkpoint")
    if tuple(prediction.prototype_ids.tolist()) != tuple(
        str(value) for value in prototypes.prototype_ids
    ):
        raise ValueError("prediction prototype identities differ from the current prototype bank")
    if prediction.checkpoint_sha256 != _sha256(paths.checkpoint):
        raise ValueError("prediction is bound to a different HEIR checkpoint")
    if prediction.prototype_sha256 != _sha256(paths.prototypes):
        raise ValueError("prediction is bound to a different prototype bank")
    if prediction.histology_sha256 != _sha256(paths.histology):
        raise ValueError("prediction is bound to a different full HistologyBag")
    if prediction.latent_space_id != prototypes.latent_space_id or (
        prediction.latent_space_id != metadata.get("latent_space_id")
    ):
        raise ValueError("prediction latent identity differs from prototypes/checkpoint")
    if prediction.model_version != metadata.get("schema"):
        raise ValueError("prediction model version differs from the checkpoint schema")
    if prediction.expression_space_id != EXPRESSION_SPACE_ID:
        raise ValueError("prediction expression space is not canonical")
    if (prediction.donor_id, prediction.sample_id, prediction.slide_id) != (
        paths.sample,
        paths.sample,
        paths.sample,
    ):
        raise ValueError("prediction donor/sample/slide identity is invalid")
    if prediction.ood_sha256 != _sha256(paths.calibrated_ood):
        raise ValueError("prediction is bound to a different target-calibrated OOD detector")
    if prediction.ood_training_donors is None or tuple(
        str(value) for value in prediction.ood_training_donors
    ) != tuple(detector.training_donors):
        raise ValueError("prediction OOD training-donor provenance is invalid")
    if prediction.program_scores is not None or prediction.program_sha256:
        raise ValueError("snPATHO prediction unexpectedly uses a gene-program artifact")
    if (
        prediction.inference_seed != settings.seed
        or prediction.latent_samples != settings.latent_samples
        or prediction.probability_threshold != settings.probability_threshold
        or prediction.artifact_threshold != settings.artifact_threshold
    ):
        raise ValueError("prediction inference decisions differ from the frozen config")
    parsed_telemetry = InferenceTelemetry.from_json(
        paths.prediction_telemetry,
        paths.predictions,
    )
    with paths.prediction_telemetry.open("r", encoding="utf-8") as handle:
        telemetry = json.load(handle)
    if Path(str(telemetry.get("prediction_path", ""))).expanduser().resolve() != (
        paths.predictions.resolve()
    ):
        raise ValueError("prediction telemetry names a different prediction path")
    if parsed_telemetry.device_type != "cuda" or parsed_telemetry.mixed_precision is not True:
        raise ValueError("prediction telemetry does not record the frozen CUDA AMP path")
    if parsed_telemetry.nuclei != len(prediction.nucleus_ids):
        raise ValueError("prediction telemetry nucleus count differs from prediction")
    if int(telemetry.get("genes", -1)) != len(prediction.gene_names):
        raise ValueError("prediction telemetry gene count differs from prediction")
    if (
        int(telemetry.get("latent_samples", -1)) != settings.latent_samples
        or int(telemetry.get("mc_chunk_size", -1)) != settings.mc_chunk_size
    ):
        raise ValueError("prediction telemetry Monte Carlo settings differ from config")
    return {
        "nuclei": len(prediction.nucleus_ids),
        "genes": len(prediction.gene_names),
        "abstained": int(prediction.abstain.sum()),
        "checkpoint_sha256": prediction.checkpoint_sha256,
        "prediction_sha256": _sha256(paths.predictions),
        "telemetry_sha256": _sha256(paths.prediction_telemetry),
        "wall_seconds": parsed_telemetry.wall_seconds,
        "peak_cuda_memory_bytes": parsed_telemetry.peak_cuda_memory_bytes,
    }


def _validate_truth(settings: Settings, paths: SamplePaths) -> Mapping[str, object]:
    truth = SpatialTruthArtifact.from_npz(paths.truth)
    prediction = PredictionBundle.from_npz(paths.predictions)
    record = _manifest_record(settings, paths.sample)
    expected_identity = (
        record.analysis_role,
        record.cohort_id,
        record.donor_id,
        record.specimen_id,
        record.block_id,
        record.section_id,
        record.outer_fold,
        record.inner_fold,
    )
    observed_identity = (
        truth.analysis_role,
        truth.cohort_id,
        truth.donor_id,
        truth.specimen_id,
        truth.block_id,
        truth.section_id,
        truth.outer_fold,
        truth.inner_fold,
    )
    if observed_identity != expected_identity or truth.analysis_role != "locked_validation":
        raise ValueError("locked truth manifest identity/role/folds are invalid")
    if tuple(truth.gene_names.tolist()) != _strings(settings.panel):
        raise ValueError("locked truth genes differ from frozen panel")
    if truth.expression_space_id != EXPRESSION_SPACE_ID:
        raise ValueError("locked truth expression space is not canonical")
    if not np.array_equal(truth.nucleus_ids.astype(str), prediction.nucleus_ids.astype(str)):
        raise ValueError("locked truth and prediction nucleus order differ")
    expected_sources: List[Tuple[str, str, str]] = [
        (str(paths.visium_h5ad.resolve()), _sha256(paths.visium_h5ad), "locked_spatial_counts"),
        (str(paths.positions.resolve()), _sha256(paths.positions), "locked_spatial_coordinates"),
        (
            str(paths.scalefactors.resolve()),
            _sha256(paths.scalefactors),
            "locked_spatial_scalefactors",
        ),
        (
            str(paths.filtered_nuclei.resolve()),
            _sha256(paths.filtered_nuclei),
            "sample_segmentation",
        ),
        (str(settings.panel.resolve()), _sha256(settings.panel), "canonical_gene_panel"),
        (str(settings.manifest.resolve()), _sha256(settings.manifest), "shared_manifest"),
    ]
    manifest_spatial_source = record.spatial_count_matrix_file
    direct_source = (
        "::" not in manifest_spatial_source
        and Path(manifest_spatial_source).expanduser().resolve() == paths.visium_h5ad.resolve()
    )
    if not direct_source:
        with paths.visium_provenance.open("r", encoding="utf-8") as handle:
            conversion = json.load(handle)
        required = {
            "source_path",
            "source_sha256",
            "derivative_path",
            "derivative_sha256",
        }
        if not isinstance(conversion, Mapping) or not required.issubset(conversion):
            raise ValueError("spatial conversion provenance sidecar is incomplete")
        if str(conversion["source_path"]) != manifest_spatial_source:
            raise ValueError("spatial conversion provenance names a different manifest source")
        if Path(str(conversion["derivative_path"])).expanduser().resolve() != (
            paths.visium_h5ad.resolve()
        ):
            raise ValueError("spatial conversion provenance names a different derivative")
        source_sha256 = _artifact_sha256(manifest_spatial_source)
        if str(conversion["source_sha256"]) != source_sha256:
            raise ValueError("spatial conversion provenance source hash is stale")
        if str(conversion["derivative_sha256"]) != _sha256(paths.visium_h5ad):
            raise ValueError("spatial conversion provenance derivative hash is stale")
        expected_sources.extend(
            (
                (
                    str(paths.visium_provenance.resolve()),
                    _sha256(paths.visium_provenance),
                    "conversion_provenance",
                ),
                (manifest_spatial_source, source_sha256, "manifest_spatial_source"),
            )
        )
    _validate_source_binding(
        label="locked truth",
        artifacts=truth.source_artifacts.tolist(),
        hashes=truth.source_sha256.tolist(),
        roles=truth.source_roles.tolist(),
        expected=expected_sources,
    )
    expected_radius = read_spot_diameter(paths.scalefactors) / 2.0
    if not np.isclose(truth.spot_radius_px, expected_radius, rtol=0.0, atol=1.0e-9):
        raise ValueError("locked truth spot radius differs from current scalefactors")
    locked_hashes = {
        digest
        for digest, role in zip(truth.source_sha256.tolist(), truth.source_roles.tolist())
        if role.startswith("locked_spatial") or role == "manifest_spatial_source"
    }
    prediction_inputs = {
        prediction.checkpoint_sha256,
        prediction.prototype_sha256,
        prediction.histology_sha256,
        prediction.ood_sha256,
        prediction.program_sha256,
    } - {""}
    if locked_hashes & prediction_inputs:
        raise ValueError("locked truth overlaps a frozen prediction input")
    return {
        "spots": len(truth.spot_ids),
        "evaluable_spots": truth.evaluable_spots,
        "nuclei": len(truth.nucleus_ids),
        "assigned_nuclei": truth.assigned_nuclei,
        "truth_sha256": _sha256(paths.truth),
    }


def _stage(settings: Settings, paths: SamplePaths, name: str) -> Stage:
    heir = (sys.executable, "-m", "heir")
    common_manifest = ("--manifest", str(settings.manifest), "--section-id", paths.sample)
    if name == "segmentation":

        def segmentation_command() -> Sequence[str]:
            base = list(heir) + ["segment-histology"]
            if paths.spaceranger_geojson.is_file():
                base += [
                    "--geojson",
                    str(paths.spaceranger_geojson),
                    "--spaceranger-version",
                    "4.1.0",
                ]
            elif paths.spaceranger_run.exists():
                raise PipelineError(
                    "Space Ranger run directory exists without a completed GeoJSON: %s; "
                    "resume/inspect that run rather than deleting or launching a duplicate"
                    % paths.spaceranger_run
                )
            else:
                base += [
                    "--image",
                    str(paths.segmentation_image),
                    "--run-id",
                    "snpatho_" + paths.sample,
                    "--output-directory",
                    str(paths.spaceranger_root),
                    "--spaceranger",
                    str(settings.spaceranger),
                    "--localcores",
                    str(settings.localcores),
                    "--localmem-gb",
                    str(settings.localmem_gb),
                    "--cuda-visible-devices",
                    "auto",
                    "--timeout-seconds",
                    str(settings.segmentation_timeout_seconds),
                ]
            base += [
                "--slide-id",
                paths.sample,
                "--nuclei-output",
                str(paths.raw_nuclei),
                "--features-output",
                str(paths.segmentation_features),
            ]
            return base

        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.raw_nuclei, paths.segmentation_features),
            requires=(paths.segmentation_image,),
            command=segmentation_command,
            validate=lambda: _validate_segmentation(paths),
            cuda=True,
        )
    if name == "capture_filter":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(
                paths.filtered_nuclei,
                paths.capture_assignment,
                paths.capture_provenance,
            ),
            requires=(paths.raw_nuclei, paths.positions, paths.scalefactors),
            command=lambda: (
                heir
                + (
                    "filter-nuclei-to-visium",
                    "--nuclei",
                    str(paths.raw_nuclei),
                    "--positions",
                    str(paths.positions),
                    "--scalefactors",
                    str(paths.scalefactors),
                    "--output",
                    str(paths.filtered_nuclei),
                    "--assignment-output",
                    str(paths.capture_assignment),
                    "--provenance-output",
                    str(paths.capture_provenance),
                )
            ),
            validate=lambda: _validate_capture(paths),
        )
    if name == "pathology_features":
        mpp = settings.mpp.get(paths.sample)
        if mpp is None:
            raise PipelineError("missing physical calibration for sample %s" % paths.sample)
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.pathology_features, paths.pathology_telemetry),
            requires=(paths.pathology_image, paths.filtered_nuclei, settings.omiclip_checkpoint),
            command=lambda: (
                heir
                + (
                    "extract-pathology-features",
                    "--image",
                    str(paths.pathology_image),
                    "--nuclei",
                    str(paths.filtered_nuclei),
                    "--output",
                    str(paths.pathology_features),
                    "--encoder",
                    "omiclip-loki",
                    "--checkpoint",
                    str(settings.omiclip_checkpoint),
                    "--trust-checkpoint",
                    "--mpp",
                    str(mpp),
                    "--backend",
                    settings.pathology_backend,
                    "--patch-diameters-um",
                    ",".join("%g" % value for value in settings.feature_scales),
                    "--input-size",
                    "224",
                    "--batch-size",
                    str(settings.feature_batch_size),
                    "--device",
                    "cuda",
                    "--mixed-precision",
                    "--telemetry-output",
                    str(paths.pathology_telemetry),
                )
            ),
            validate=lambda: _validate_pathology(settings, paths),
            cuda=True,
        )
    if name == "prepare_histology":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.histology,),
            requires=(
                paths.source_histology,
                paths.filtered_nuclei,
                paths.pathology_features,
                settings.ood_artifact,
            ),
            command=lambda: (
                heir
                + (
                    "prepare-histology",
                    "--nuclei",
                    str(paths.filtered_nuclei),
                    "--features",
                    str(paths.pathology_features),
                    "--histology-source",
                    str(paths.source_histology),
                    "--slide-id",
                    paths.sample,
                    "--sample-id",
                    paths.sample,
                    "--donor-id",
                    paths.sample,
                    "--block-id",
                    paths.sample + "_FFPE",
                    "--feature-space-id",
                    settings.feature_space_id,
                    "--mpp",
                    str(settings.mpp[paths.sample]),
                    "--graph-k",
                    str(settings.graph_k),
                    "--graph-radius-um",
                    str(settings.graph_radius_um),
                    "--graph-max-degree",
                    str(settings.graph_max_degree),
                    "--output",
                    str(paths.histology),
                )
            ),
            validate=lambda: _validate_histology(settings, paths),
        )
    if name == "calibrate_ood":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.calibrated_ood, paths.calibrated_ood_provenance),
            requires=(paths.histology, settings.ood_artifact),
            command=lambda: (
                sys.executable,
                str(settings.repository / "scripts" / "calibrate_target_ood.py"),
                "--base-ood",
                str(settings.ood_artifact),
                "--histology",
                str(paths.histology),
                "--sample-id",
                paths.sample,
                "--quantile",
                str(settings.ood_calibration_quantile),
                "--output",
                str(paths.calibrated_ood),
                "--provenance-output",
                str(paths.calibrated_ood_provenance),
            ),
            validate=lambda: _validate_calibrated_ood(settings, paths),
        )
    if name == "split_histology":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.train_histology, paths.validation_histology, paths.split_summary),
            requires=(paths.histology,),
            command=lambda: (
                sys.executable,
                str(settings.repository / "scripts" / "split_histology_bag.py"),
                "--input",
                str(paths.histology),
                "--train-output",
                str(paths.train_histology),
                "--validation-output",
                str(paths.validation_histology),
                "--summary",
                str(paths.split_summary),
                "--validation-fraction",
                "0.2",
                "--block-size-um",
                str(settings.block_size_um),
                "--maximum-train-cells",
                str(settings.maximum_train_cells),
                "--maximum-validation-cells",
                str(settings.maximum_validation_cells),
                "--seed",
                str(settings.seed),
            ),
            validate=lambda: _validate_split(paths),
        )
    if name == "prepare_reference":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.reference,),
            requires=(
                paths.reference_h5ad,
                paths.reference_provenance,
                settings.manifest,
                settings.panel,
            ),
            command=lambda: (
                heir
                + (
                    "prepare-reference",
                    *common_manifest,
                    "--input",
                    str(paths.reference_h5ad),
                    "--conversion-provenance",
                    str(paths.reference_provenance),
                    "--genes",
                    str(settings.panel),
                    "--cell-type-key",
                    "major_annotation",
                    "--gene-key",
                    "feature_name",
                    "--layer",
                    "X",
                    "--output",
                    str(paths.reference),
                )
            ),
            validate=lambda: _validate_reference(settings, paths),
        )
    if name == "build_prototypes":
        rare = ("--include-rare-types",) if settings.include_rare_types else ()
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.reference_latent, paths.prototypes),
            requires=(paths.reference, settings.latent_transform),
            command=lambda: (
                heir
                + (
                    "build-prototypes",
                    "--reference",
                    str(paths.reference),
                    "--reference-with-latent",
                    str(paths.reference_latent),
                    "--latent-transform",
                    str(settings.latent_transform),
                    "--latent-dim",
                    str(settings.latent_dim),
                    "--max-per-type",
                    str(settings.maximum_prototypes),
                    "--minimum-cells",
                    str(settings.minimum_cells),
                    *rare,
                    "--seed",
                    str(settings.seed),
                    "--output",
                    str(paths.prototypes),
                )
            ),
            validate=lambda: _validate_prototypes(settings, paths),
        )
    if name == "assemble_batches":

        def assemble_command() -> Sequence[str]:
            script = settings.repository / "scripts" / "assemble_snpatho_batches.py"
            if not script.is_file():
                raise PipelineError(
                    "missing safe two-bag assembler %s; this helper must assemble train and "
                    "validation without target Visium expression" % script
                )
            return (
                sys.executable,
                str(script),
                "--sample-id",
                paths.sample,
                "--donor-id",
                paths.sample,
                "--block-id",
                paths.sample + "_FFPE",
                "--analysis-role",
                "locked_validation",
                "--train-histology",
                str(paths.train_histology),
                "--validation-histology",
                str(paths.validation_histology),
                "--prototypes",
                str(paths.prototypes),
                "--reference",
                str(paths.reference_latent),
                "--ood-artifact",
                str(paths.calibrated_ood),
                "--train-output",
                str(paths.train_batch),
                "--validation-output",
                str(paths.validation_batch),
                "--artifact-threshold",
                str(settings.artifact_threshold),
            )

        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.train_batch, paths.validation_batch),
            requires=(
                paths.train_histology,
                paths.validation_histology,
                paths.prototypes,
                paths.reference_latent,
                paths.calibrated_ood,
            ),
            command=assemble_command,
            validate=lambda: _validate_batches(settings, paths),
        )
    if name == "train":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.checkpoint, paths.training_history, paths.training_telemetry),
            requires=(paths.train_batch, paths.validation_batch, settings.rna_decoder),
            command=lambda: (
                heir
                + (
                    "train",
                    "--train-batch",
                    str(paths.train_batch),
                    "--validation-batch",
                    str(paths.validation_batch),
                    "--output",
                    str(paths.model_directory),
                    "--stage",
                    "personalized",
                    "--epochs",
                    str(settings.epochs),
                    "--learning-rate",
                    str(settings.learning_rate),
                    "--adapter-learning-rate",
                    str(settings.adapter_learning_rate),
                    "--weight-decay",
                    str(settings.weight_decay),
                    "--warmup-fraction",
                    str(settings.warmup_fraction),
                    "--gradient-clip-norm",
                    str(settings.gradient_clip_norm),
                    "--bag-size",
                    str(settings.bag_size),
                    "--reference-batch-size",
                    str(settings.reference_batch_size),
                    "--maximum-sample-cells",
                    str(settings.maximum_sample_cells),
                    "--early-stopping-patience",
                    str(settings.early_stopping_patience),
                    "--graph-hidden-dim",
                    str(settings.model_graph_hidden_dim),
                    "--graph-output-dim",
                    str(settings.model_graph_output_dim),
                    "--graph-layers",
                    str(settings.model_graph_layers),
                    "--trunk-hidden-dims",
                    ",".join(str(value) for value in settings.model_trunk_hidden_dims),
                    "--decoder-hidden-dims",
                    ",".join(str(value) for value in settings.model_decoder_hidden_dims),
                    "--dropout",
                    str(settings.model_dropout),
                    *(("--hard-type-routing",) if settings.hard_type_routing else ()),
                    "--abstain-threshold",
                    str(settings.probability_threshold),
                    *(
                        ("--allow-negative-expression",)
                        if not settings.nonnegative_expression
                        else ()
                    ),
                    "--rna-vae-checkpoint",
                    str(settings.rna_decoder),
                    "--allow-split-overlap",
                    "--mixed-precision",
                    "--seed",
                    str(settings.seed),
                    "--device",
                    "cuda",
                )
            ),
            validate=lambda: _validate_model(settings, paths),
            cuda=True,
            telemetry_output=paths.training_telemetry,
            guarded_directory=paths.model_directory,
        )
    if name == "predict":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.predictions, paths.prediction_telemetry),
            requires=(
                paths.checkpoint,
                paths.histology,
                paths.prototypes,
                settings.panel,
                paths.calibrated_ood,
            ),
            command=lambda: (
                heir
                + (
                    "predict",
                    "--checkpoint",
                    str(paths.checkpoint),
                    "--histology",
                    str(paths.histology),
                    "--prototypes",
                    str(paths.prototypes),
                    "--genes",
                    str(settings.panel),
                    "--output",
                    str(paths.predictions),
                    "--telemetry-output",
                    str(paths.prediction_telemetry),
                    "--latent-samples",
                    str(settings.latent_samples),
                    "--mc-chunk-size",
                    str(settings.mc_chunk_size),
                    "--probability-threshold",
                    str(settings.probability_threshold),
                    "--artifact-threshold",
                    str(settings.artifact_threshold),
                    "--sample-id",
                    paths.sample,
                    "--donor-id",
                    paths.sample,
                    "--ood-artifact",
                    str(paths.calibrated_ood),
                    "--mixed-precision",
                    "--seed",
                    str(settings.seed),
                    "--device",
                    "cuda",
                )
            ),
            validate=lambda: _validate_prediction(settings, paths),
            cuda=True,
        )
    if name == "prepare_locked_truth":
        return Stage(
            name=name,
            sample=paths.sample,
            outputs=(paths.truth,),
            requires=(
                paths.predictions,
                paths.visium_h5ad,
                paths.visium_provenance,
                paths.positions,
                paths.scalefactors,
                paths.filtered_nuclei,
                settings.panel,
                settings.manifest,
            ),
            command=lambda: (
                heir
                + (
                    "prepare-spatial-truth",
                    *common_manifest,
                    "--counts",
                    str(paths.visium_h5ad),
                    "--conversion-provenance",
                    str(paths.visium_provenance),
                    "--positions",
                    str(paths.positions),
                    "--scalefactors",
                    str(paths.scalefactors),
                    "--nuclei",
                    str(paths.filtered_nuclei),
                    "--genes",
                    str(settings.panel),
                    "--gene-key",
                    "feature_name",
                    "--layer",
                    "X",
                    "--output",
                    str(paths.truth),
                )
            ),
            validate=lambda: _validate_truth(settings, paths),
            locked_target=True,
        )
    raise KeyError(name)


def _write_assemble_helper(repository: Path) -> None:
    """The helper is repository source, so fail if installation is incomplete."""

    helper = repository / "scripts" / "assemble_snpatho_batches.py"
    if not helper.is_file():
        raise PipelineError("required helper is absent: %s" % helper)


def _expected_plan(
    settings: Settings,
    paths_by_sample: Mapping[str, SamplePaths],
    selected: Sequence[str],
    plan_path: Path,
) -> Dict[str, object]:
    return {
        "schema_version": "heir.snpatho_benchmark_plan.v1",
        "checkpoint_sha256": "",
        "gene_panel": _relative_or_absolute(settings.panel, plan_path.parent),
        "frozen_model_version": "heir.trained_model.v1",
        "pipeline_config": _relative_or_absolute(settings.config_path, plan_path.parent),
        "pipeline_config_sha256": _sha256(settings.config_path),
        "gene_panel_sha256": _sha256(settings.panel),
        "cases": [
            {
                "section_id": sample,
                "checkpoint_sha256": _sha256(paths_by_sample[sample].checkpoint),
                "predictions": _relative_or_absolute(
                    paths_by_sample[sample].predictions, plan_path.parent
                ),
                "predictions_sha256": _sha256(paths_by_sample[sample].predictions),
                "truth": _relative_or_absolute(paths_by_sample[sample].truth, plan_path.parent),
                "truth_sha256": _sha256(paths_by_sample[sample].truth),
                "matched_reference": _relative_or_absolute(
                    paths_by_sample[sample].reference, plan_path.parent
                ),
                "matched_reference_sha256": _sha256(paths_by_sample[sample].reference),
                "telemetry": _relative_or_absolute(
                    paths_by_sample[sample].prediction_telemetry, plan_path.parent
                ),
                "telemetry_sha256": _sha256(paths_by_sample[sample].prediction_telemetry),
            }
            for sample in selected
        ],
    }


def _freeze_plan_stage(
    runner: PipelineRunner,
    settings: Settings,
    paths_by_sample: Mapping[str, SamplePaths],
    selected: Sequence[str],
    plan_path: Path,
) -> None:
    if not runner.execute:
        runner._record(
            {
                "stage": "freeze_plan",
                "sample": "",
                "status": "planned",
                "command": "internal:freeze hash-bound snPATHO benchmark plan",
                "output": str(plan_path),
            }
        )
        return
    expected = _expected_plan(settings, paths_by_sample, selected, plan_path)
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            observed = json.load(handle)
        if observed != expected:
            raise PipelineError(
                "existing frozen plan differs; move it aside instead of replacing it"
            )
        load_snpatho_plan(plan_path)
        status = "skipped_valid"
    else:
        _atomic_json(plan_path, expected)
        load_snpatho_plan(plan_path)
        status = "completed"
    runner._record(
        {
            "stage": "freeze_plan",
            "sample": "",
            "status": status,
            "output": str(plan_path),
            "sha256": _sha256(plan_path),
            "cases": list(selected),
        }
    )


_BENCHMARK_TSV_FIELDS = (
    "record_type",
    "cohort_id",
    "donor_id",
    "method",
    "baseline_method",
    "metric",
    "value",
    "estimate",
    "mean_difference",
    "mean_improvement",
    "ci_lower",
    "ci_upper",
    "probability_better",
    "higher_is_better",
    "n_observations",
    "n_donors",
    "n_paired_donors",
    "n_method_donors",
    "n_baseline_donors",
    "n_missing",
    "n_data_limited",
    "status",
    "reason",
)


def _tsv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _validate_benchmark_tsv(report: Mapping[str, object], path: Path) -> int:
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("benchmark report aggregate is malformed")
    expected: List[Dict[str, str]] = []
    for key, record_type in (
        ("donor_metrics", "donor_metric"),
        ("summaries", "summary"),
        ("comparisons", "comparison"),
    ):
        records = aggregate.get(key)
        if not isinstance(records, list) or not all(isinstance(row, Mapping) for row in records):
            raise ValueError("benchmark report aggregate %s is malformed" % key)
        for row in records:
            expected.append(
                {
                    field: _tsv_value(record_type if field == "record_type" else row.get(field))
                    for field in _BENCHMARK_TSV_FIELDS
                }
            )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _BENCHMARK_TSV_FIELDS:
            raise ValueError("benchmark TSV header differs from the normalized report contract")
        observed = list(reader)
    if observed != expected:
        raise ValueError("benchmark JSON aggregate and TSV records disagree")
    return len(observed)


def _validate_benchmark_report(
    settings: Settings,
    plan_path: Path,
    output: Path,
    tsv: Path,
    selected: Sequence[str],
) -> Mapping[str, object]:
    plan = load_snpatho_plan(plan_path)
    with output.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, Mapping) or report.get("schema_version") != (
        "heir.snpatho_benchmark.v1"
    ):
        raise ValueError("benchmark report schema is invalid")
    isolation = report.get("isolation")
    if not isinstance(isolation, Mapping):
        raise ValueError("benchmark report isolation block is malformed")
    expected_checkpoints = {
        case.section_id: case.checkpoint_sha256
        for case in sorted(plan.cases, key=lambda value: value.section_id)
    }
    expected_isolation = {
        "target_spatial_truth_role": "locked_validation",
        "target_spatial_expression_used_for_training": False,
        "target_histology_used_for_training": True,
        "target_spatial_metadata_used_for_capture_filtering": True,
        "checkpoint_sha256_by_section": expected_checkpoints,
        "gene_panel_sha256": _sha256(settings.panel),
        "frozen_model_version": plan.frozen_model_version,
        "plan_sha256": _sha256(plan_path),
    }
    if dict(isolation) != expected_isolation:
        raise ValueError("benchmark report isolation provenance differs from the frozen plan")
    if report.get("seed") != settings.seed:
        raise ValueError("benchmark report seed differs from the frozen config")
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list) or not all(isinstance(case, Mapping) for case in raw_cases):
        raise ValueError("benchmark report cases are malformed")
    cases = tuple(str(case.get("section_id", "")) for case in raw_cases)
    if cases != tuple(sorted(selected)):
        raise ValueError("benchmark report cases differ from selected samples")
    plan_cases = {case.section_id: case for case in plan.cases}
    for case in raw_cases:
        section_id = str(case["section_id"])
        frozen = plan_cases.get(section_id)
        if frozen is None:
            raise ValueError("benchmark report contains a case absent from the frozen plan")
        expected_provenance = {
            "predictions": frozen.predictions_sha256,
            "truth": frozen.truth_sha256,
            "matched_reference": frozen.matched_reference_sha256,
            "telemetry": frozen.telemetry_sha256,
        }
        if case.get("provenance") != expected_provenance:
            raise ValueError("benchmark case provenance differs from the frozen plan")
        telemetry = case.get("telemetry")
        if not isinstance(telemetry, Mapping) or telemetry.get("available") is not True:
            raise ValueError("benchmark case lacks frozen inference telemetry")
        if telemetry.get("prediction_sha256") != frozen.predictions_sha256:
            raise ValueError("benchmark case telemetry differs from its frozen prediction")
        if case.get("donor_id") != section_id:
            raise ValueError("benchmark case donor identity differs from its section")
        if set(case.get("methods", {})) != set(BENCHMARK_METHODS):
            raise ValueError("benchmark case method set is incomplete")
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, Mapping) or aggregate.get("schema_version") != "heir-benchmark-v1":
        raise ValueError("benchmark aggregate schema is invalid")
    expected_settings = {
        "confidence": 0.95,
        "iterations": settings.bootstrap_resamples,
        "minimum_donors": 2,
        "seed": settings.seed,
    }
    if aggregate.get("settings") != expected_settings:
        raise ValueError("benchmark aggregate settings differ from the frozen experiment")
    records = _validate_benchmark_tsv(report, tsv)
    return {
        "cases": list(cases),
        "plan_sha256": isolation["plan_sha256"],
        "tsv_records": records,
    }


def _benchmark_stage(
    settings: Settings,
    plan_path: Path,
    output: Path,
    tsv: Path,
    selected: Sequence[str],
) -> Stage:
    def validate() -> Mapping[str, object]:
        return _validate_benchmark_report(settings, plan_path, output, tsv, selected)

    partial = ("--allow-partial",) if set(selected) != set(SAMPLES) else ()
    return Stage(
        name="benchmark",
        sample="",
        outputs=(output, tsv),
        requires=(plan_path,),
        command=lambda: (
            sys.executable,
            str(settings.repository / "scripts" / "benchmark_snpatho.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(output),
            "--tsv",
            str(tsv),
            "--iterations",
            str(settings.bootstrap_resamples),
            "--seed",
            str(settings.seed),
            *partial,
        ),
        validate=validate,
        locked_target=True,
    )


def _parser(repository: Path) -> argparse.ArgumentParser:
    default_config = repository / "configs" / "experiments" / "snpatho_v0_2.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="append",
        choices=("all",) + SAMPLES,
        default=[],
        help="repeat for selected specimens; default/all runs 4066, 4399, and 4411",
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--gene-panel", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--spaceranger", type=Path)
    parser.add_argument(
        "--omiclip-checkpoint",
        type=Path,
        help=(
            "external pretrained visual checkpoint; overrides HEIR_OMICLIP_CHECKPOINT, "
            "HEIR_PRETRAINED_DIR, and the frozen config path"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run missing stages; without this flag only a dry-run plan is emitted",
    )
    parser.add_argument("--stop-after", choices=ALL_STAGES)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--events-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    args = _parser(repository).parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise PipelineError("experiment config is absent: %s" % config_path)
    settings = Settings.load(
        repository=repository,
        config_path=config_path,
        artifact_root=args.artifact_root,
        panel_override=args.gene_panel,
        spaceranger_override=args.spaceranger,
        omiclip_checkpoint_override=args.omiclip_checkpoint,
    )
    requested = args.sample or ["all"]
    selected = SAMPLES if "all" in requested else tuple(dict.fromkeys(requested))
    orchestration = settings.artifact_root / "orchestration_v0_2"
    status_path = (
        args.status_output.expanduser().resolve()
        if args.status_output
        else orchestration / "status.json"
    )
    events_path = (
        args.events_output.expanduser().resolve()
        if args.events_output
        else orchestration / "events.jsonl"
    )
    runner = PipelineRunner(
        repository=repository,
        execute=args.execute,
        status_path=status_path,
        events_path=events_path,
        logs_directory=orchestration / "logs",
        stop_after=args.stop_after,
    )
    problems = settings.validate_fixed_inputs(require_runtime=args.execute)
    if problems:
        runner._record(
            {
                "stage": "preflight",
                "sample": "",
                "status": "blocked" if args.execute else "dry_run_missing_prerequisites",
                "problems": problems,
            }
        )
        if args.execute:
            raise PipelineError("preflight failed: " + "; ".join(problems))
    else:
        runner._record(
            {
                "stage": "preflight",
                "sample": "",
                "status": "validated",
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "panel": str(settings.panel),
                "panel_sha256": _sha256(settings.panel),
                "samples": list(selected),
                "cuda": torch.cuda.is_available(),
            }
        )
    _write_assemble_helper(repository)
    paths_by_sample = {sample: SamplePaths.discover(settings, sample) for sample in selected}

    for stage_name in PREDICTION_PHASE:
        for sample in selected:
            runner.run(_stage(settings, paths_by_sample[sample], stage_name))
        if runner.should_stop(stage_name):
            runner.write_status()
            return 0

    if args.execute:
        for sample in selected:
            _validate_prediction(settings, paths_by_sample[sample])
    runner.unlock_locked_targets()
    runner.write_status()

    for sample in selected:
        runner.run(_stage(settings, paths_by_sample[sample], "prepare_locked_truth"))
    if runner.should_stop("prepare_locked_truth"):
        return 0

    plan_path = orchestration / (
        "benchmark_plan.all.json"
        if set(selected) == set(SAMPLES)
        else "benchmark_plan.%s.json" % "-".join(selected)
    )
    _freeze_plan_stage(runner, settings, paths_by_sample, selected, plan_path)
    if runner.should_stop("freeze_plan"):
        return 0

    report_suffix = "all" if set(selected) == set(SAMPLES) else "-".join(selected)
    runner.run(
        _benchmark_stage(
            settings,
            plan_path,
            orchestration / ("benchmark.%s.json" % report_suffix),
            orchestration / ("benchmark.%s.tsv" % report_suffix),
            selected,
        )
    )
    runner.write_status()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, ValueError, KeyError, FileNotFoundError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        raise SystemExit(2)
