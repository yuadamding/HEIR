#!/usr/bin/env python3
"""Prepare hash-bound native-scANVI inputs for the snPATHO refiner.

The script owns only the four derived artifacts consumed by the refinement
benchmark.  It validates the native scANVI provenance, frozen spatial split,
and histology-only OOD calibration before invoking the public ``heir`` CLI.
Per-stage receipts make interrupted runs resumable; untracked, partial, stale,
or hash-mismatched outputs are rejected instead of overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from heir.data import HistologyBag, PrototypeSet, RNAReference
from heir.prior import RNAResidualGeometry
from heir.training import HEIRTrainingBatch, TrainingStage
from heir.uncertainty import MahalanobisOOD

SAMPLES = ("4066", "4399", "4411")
SCHEMA = "heir.snpatho_refinement_input_preparation.v1"
RECEIPT_SCHEMA = "heir.snpatho_refinement_input_stage.v1"
RECIPE = {
    "prototype": {
        "include_rare_types": True,
        "max_per_type": 10,
        "minimum_cells": 50,
        "seed": 17,
        "shrinkage_kappa": 50.0,
    },
    "residual_geometry": {
        "bound_fraction": 0.5,
        "calibration_quantile": 0.9,
        "minimum_bound": 0.001,
        "minimum_calibration_cells": 3,
        "rank": 4,
    },
    "batch": {
        "analysis_role": "development_retrospective",
        "artifact_threshold": 0.5,
        "markers_per_type": 25,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted(item for item in path.rglob("*") if item.is_file())
    if not sources:
        raise ValueError("native scANVI checkpoint directory is empty: %s" % path)
    for source in sources:
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("JSON root must be an object: %s" % path)
    return payload


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
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_name(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def _record(path: Path, repository: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _path_name(path, repository), "sha256": _sha256(path)}


def _declared_file(record: object, expected: Path, label: str) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError("%s provenance record is missing" % label)
    declared_path = record.get("path")
    declared_hash = record.get("sha256")
    if not isinstance(declared_path, str) or Path(declared_path).expanduser().resolve() != expected:
        raise ValueError("%s provenance path differs from the expected artifact" % label)
    if declared_hash != _sha256(expected):
        raise ValueError("%s provenance SHA-256 is stale" % label)
    return record


@dataclass(frozen=True)
class SamplePaths:
    sample: str
    source: Path
    scanvi: Path
    scanvi_input: Path

    @property
    def reference(self) -> Path:
        return self.scanvi / "reference500_scanvi.npz"

    @property
    def source_h5ad(self) -> Path:
        return self.scanvi_input / "reference.h5ad"

    @property
    def prototypes(self) -> Path:
        return self.scanvi / "prototypes_rare_complete.npz"

    @property
    def geometry(self) -> Path:
        return self.scanvi / "residual_geometry_rare_complete.npz"

    def histology(self, role: str) -> Path:
        return self.source / ("histology_%s.npz" % role)

    @property
    def split(self) -> Path:
        return self.source / "histology_split.json"

    @property
    def ood(self) -> Path:
        return self.source / "ood_target_calibrated.npz"

    @property
    def ood_provenance(self) -> Path:
        return self.source / "ood_target_calibrated.provenance.json"

    def batch(self, role: str) -> Path:
        return self.scanvi / ("batch_%s_rare_complete.npz" % role)


@dataclass(frozen=True)
class Stage:
    sample: str
    name: str
    inputs: Tuple[Tuple[str, Path], ...]
    output: Path
    command: Callable[[Path], Tuple[str, ...]]
    validate: Callable[[Path], None]


def _validate_upstream(
    *,
    repository: Path,
    source_root: Path,
    scanvi_root: Path,
    scanvi_input_root: Path,
    provenance_path: Path,
) -> Tuple[Mapping[str, object], Dict[str, SamplePaths], Dict[str, object]]:
    provenance = _json(provenance_path)
    if provenance.get("schema") != "heir.snpatho_scanvi_r1.v1":
        raise ValueError("native scANVI provenance schema is invalid")
    if tuple(provenance.get("samples", ())) != SAMPLES:
        raise ValueError("native scANVI provenance must contain the three frozen samples")
    if provenance.get("seed") != 17 or provenance.get("cuda") is not True:
        raise ValueError("native scANVI provenance differs from the frozen CUDA/seed recipe")
    if provenance.get("latent_dim") != 32:
        raise ValueError("native scANVI provenance latent width is not 32")

    panel = (repository / "manifests" / "gene_panel_snpatho_500.tsv").resolve()
    if provenance.get("gene_panel_sha256") != _sha256(panel):
        raise ValueError("native scANVI provenance uses a different gene panel")
    native_model = Path(str(provenance.get("native_model", ""))).expanduser().resolve()
    decoder = Path(str(provenance.get("decoder", ""))).expanduser().resolve()
    model_hash = _directory_sha256(native_model)
    if provenance.get("native_model_sha256") != model_hash:
        raise ValueError("native scANVI checkpoint hash differs from its provenance")
    if provenance.get("latent_space_id") != "sha256:" + model_hash:
        raise ValueError("native scANVI latent identity differs from the checkpoint hash")
    if provenance.get("decoder_sha256") != _sha256(decoder):
        raise ValueError("distilled native scANVI decoder hash differs from its provenance")

    latent_outputs = provenance.get("latent_outputs")
    h5ad_hashes = provenance.get("input_h5ad_sha256")
    if not isinstance(latent_outputs, Mapping) or not isinstance(h5ad_hashes, Mapping):
        raise ValueError("native scANVI provenance lacks reference hashes")

    samples: Dict[str, SamplePaths] = {}
    input_records: Dict[str, object] = {
        "scanvi_provenance": _record(provenance_path, repository),
        "gene_panel": _record(panel, repository),
        "native_model": {
            "path": _path_name(native_model, repository),
            "sha256": model_hash,
        },
        "decoder": _record(decoder, repository),
    }
    genes = tuple(
        line.split("\t", 1)[0].strip()
        for line in panel.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if len(genes) != len(set(genes)) or not genes:
        raise ValueError("frozen gene panel is empty or contains duplicates")

    for sample in SAMPLES:
        paths = SamplePaths(
            sample=sample,
            source=(source_root / sample).resolve(),
            scanvi=(scanvi_root / sample).resolve(),
            scanvi_input=(scanvi_input_root / sample).resolve(),
        )
        samples[sample] = paths
        _declared_file(latent_outputs.get(sample), paths.reference, "%s latent reference" % sample)
        if h5ad_hashes.get(sample) != _sha256(paths.source_h5ad):
            raise ValueError("%s source H5AD hash differs from scANVI provenance" % sample)

        reference = RNAReference.load_npz(paths.reference)
        if (
            reference.sample_id != sample
            or set(reference.donor_ids.tolist()) != {sample}
            or reference.block_id != sample + "_FFPE"
        ):
            raise ValueError("%s native RNA reference identity is invalid" % sample)
        if reference.latent.shape != (reference.shape[0], 32):
            raise ValueError("%s native RNA reference has the wrong latent shape" % sample)
        if reference.latent_space_id != provenance["latent_space_id"]:
            raise ValueError("%s native RNA reference has a foreign latent identity" % sample)
        if reference.source_count_sha256 != h5ad_hashes[sample]:
            raise ValueError("%s native RNA reference is bound to a different H5AD" % sample)
        if tuple(reference.gene_ids.tolist()) != genes:
            raise ValueError("%s native RNA reference differs from the frozen gene panel" % sample)

        full = HistologyBag.load_npz(paths.histology("full"))
        train = HistologyBag.load_npz(paths.histology("train"))
        validation = HistologyBag.load_npz(paths.histology("validation"))
        for role, bag in (("full", full), ("train", train), ("validation", validation)):
            if (bag.sample_id, bag.donor_id, bag.block_id) != (
                sample,
                sample,
                sample + "_FFPE",
            ):
                raise ValueError("%s %s HistologyBag identity is invalid" % (sample, role))
        if len({full.feature_space_id, train.feature_space_id, validation.feature_space_id}) != 1:
            raise ValueError("%s spatial split uses different feature spaces" % sample)
        train_ids = set(train.nucleus_ids.tolist())
        validation_ids = set(validation.nucleus_ids.tolist())
        full_ids = set(full.nucleus_ids.tolist())
        if train_ids & validation_ids or not (train_ids | validation_ids).issubset(full_ids):
            raise ValueError("%s spatial train/validation split is not disjoint" % sample)

        split = _json(paths.split)
        declared_split_paths = {
            "input": paths.histology("full"),
            "train_output": paths.histology("train"),
            "validation_output": paths.histology("validation"),
        }
        if any(
            Path(str(split.get(key, ""))).expanduser().resolve() != expected
            for key, expected in declared_split_paths.items()
        ):
            raise ValueError("%s split summary names different HistologyBags" % sample)
        if (
            split.get("seed") != 17
            or split.get("nucleus_overlap") != 0
            or split.get("total_nuclei") != full.n_nuclei
            or split.get("train_nuclei") != train.n_nuclei
            or split.get("validation_nuclei") != validation.n_nuclei
        ):
            raise ValueError("%s spatial split summary is stale" % sample)

        detector = MahalanobisOOD.from_npz(paths.ood)
        ood_provenance = _json(paths.ood_provenance)
        if (
            ood_provenance.get("schema") != "heir.target_histology_ood_calibration.v1"
            or ood_provenance.get("sample_id") != sample
            or ood_provenance.get("target_expression_accessed") is not False
            or ood_provenance.get("calibration_input_modality") != "target_histology_features_only"
        ):
            raise ValueError("%s OOD calibration provenance is invalid" % sample)
        _declared_file(ood_provenance.get("output"), paths.ood, "%s calibrated OOD" % sample)
        ood_inputs = ood_provenance.get("inputs")
        if not isinstance(ood_inputs, Mapping):
            raise ValueError("%s OOD calibration input provenance is missing" % sample)
        _declared_file(
            ood_inputs.get("histology"), paths.histology("full"), "%s OOD histology" % sample
        )
        base_record = ood_inputs.get("base_ood")
        if not isinstance(base_record, Mapping) or not isinstance(base_record.get("path"), str):
            raise ValueError("%s base OOD provenance is missing" % sample)
        base_path = Path(str(base_record["path"])).expanduser().resolve()
        _declared_file(base_record, base_path, "%s base OOD" % sample)
        copied = ood_provenance.get("copied_training_provenance")
        if not isinstance(copied, Mapping):
            raise ValueError("%s OOD training provenance is missing" % sample)
        if (
            detector.feature_space_id != full.feature_space_id
            or detector.training_donors != ("B1",)
            or tuple(copied.get("training_donors", ())) != detector.training_donors
            or tuple(copied.get("source_sha256", ())) != detector.source_sha256
            or copied.get("feature_space_id") != detector.feature_space_id
            or ood_provenance.get("threshold") != detector.threshold
            or ood_provenance.get("quantile") != detector.quantile
        ):
            raise ValueError("%s calibrated OOD detector provenance is stale" % sample)

        input_records[sample] = {
            "source_h5ad": _record(paths.source_h5ad, repository),
            "reference": _record(paths.reference, repository),
            "histology_full": _record(paths.histology("full"), repository),
            "histology_train": _record(paths.histology("train"), repository),
            "histology_validation": _record(paths.histology("validation"), repository),
            "histology_split": _record(paths.split, repository),
            "calibrated_ood": _record(paths.ood, repository),
            "calibrated_ood_provenance": _record(paths.ood_provenance, repository),
            "base_ood": _record(base_path, repository),
        }
    return provenance, samples, input_records


def _validate_prototypes(output: Path, paths: SamplePaths) -> None:
    reference = RNAReference.load_npz(paths.reference)
    prototypes = PrototypeSet.load_npz(output)
    expected_types = set(reference.cell_type_labels.tolist())
    if set(prototypes.cell_type_labels.tolist()) != expected_types:
        raise ValueError("rare-complete prototypes omit or add an RNA cell type")
    if (
        set(prototypes.sample_ids.tolist()) != {paths.sample}
        or prototypes.donor_id != paths.sample
        or prototypes.block_id != paths.sample + "_FFPE"
        or prototypes.source_reference_sha256 != _sha256(paths.reference)
        or prototypes.latent_space_id != reference.latent_space_id
        or prototypes.latent_training_donors
        or prototypes.latent_transform_sha256
    ):
        raise ValueError("prototype provenance differs from the native RNA reference")


def _validate_geometry(output: Path, paths: SamplePaths) -> None:
    reference = RNAReference.load_npz(paths.reference)
    geometry = RNAResidualGeometry.from_npz(output)
    expected_types = tuple(sorted(set(reference.cell_type_labels.tolist())))
    if (
        tuple(geometry.type_names.tolist()) != expected_types
        or geometry.rank != RECIPE["residual_geometry"]["rank"]
        or geometry.latent_space_id != reference.latent_space_id
        or geometry.source_reference_sha256 != _sha256(paths.reference)
        or geometry.training_donors != (paths.sample,)
        or geometry.latent_transform_sha256
    ):
        raise ValueError("RNA residual geometry provenance or ontology is invalid")


def _validate_batch(output: Path, paths: SamplePaths, role: str) -> None:
    reference = RNAReference.load_npz(paths.reference)
    prototypes = PrototypeSet.load_npz(paths.prototypes)
    histology = HistologyBag.load_npz(paths.histology(role))
    detector = MahalanobisOOD.from_npz(paths.ood)
    batch = HEIRTrainingBatch.load_npz(output)
    batch.validate(TrainingStage.PERSONALIZED)
    expected_sources = (
        paths.histology(role),
        paths.prototypes,
        paths.reference,
        paths.ood,
    )
    if tuple(batch.source_artifacts) != tuple(str(path.resolve()) for path in expected_sources):
        raise ValueError("training batch names different source artifacts")
    if tuple(batch.source_sha256) != tuple(_sha256(path) for path in expected_sources):
        raise ValueError("training batch source hashes are stale")
    if batch.source_roles != (
        "sample_assay",
        "sample_assay",
        "sample_assay",
        "shared_teacher",
    ):
        raise ValueError("training batch source roles are invalid")
    if (
        (batch.sample_id, batch.bag_id, batch.donor_id, batch.block_id)
        != (paths.sample, "%s_%s" % (paths.sample, role), paths.sample, paths.sample + "_FFPE")
        or batch.analysis_role != RECIPE["batch"]["analysis_role"]
        or batch.latent_space_id != reference.latent_space_id
        or batch.feature_space_id != histology.feature_space_id
        or tuple(batch.nucleus_ids) != tuple(histology.nucleus_ids.tolist())
        or tuple(batch.type_names) != tuple(sorted(set(reference.cell_type_labels.tolist())))
        or tuple(batch.gene_names) != tuple(reference.gene_ids.tolist())
        or tuple(batch.prototype_ids) != tuple(prototypes.prototype_ids.tolist())
        or batch.molecular_training_donors != detector.training_donors
    ):
        raise ValueError("training batch identity, ontology, or feature provenance is invalid")


def build_stages(
    *,
    samples: Mapping[str, SamplePaths],
    heir_command: str,
) -> Tuple[Stage, ...]:
    stages = []
    for sample in SAMPLES:
        paths = samples[sample]

        def prototype_command(output: Path, p: SamplePaths = paths) -> Tuple[str, ...]:
            return (
                heir_command,
                "build-prototypes",
                "--reference",
                str(p.reference),
                "--output",
                str(output),
                "--max-per-type",
                "10",
                "--minimum-cells",
                "50",
                "--shrinkage-kappa",
                "50",
                "--include-rare-types",
                "--seed",
                "17",
            )

        stages.append(
            Stage(
                sample,
                "prototypes",
                (("reference", paths.reference),),
                paths.prototypes,
                prototype_command,
                lambda output, p=paths: _validate_prototypes(output, p),
            )
        )

        def geometry_command(output: Path, p: SamplePaths = paths) -> Tuple[str, ...]:
            return (
                heir_command,
                "fit-residual-geometry",
                "--reference",
                str(p.reference),
                "--prototypes",
                str(p.prototypes),
                "--output",
                str(output),
                "--rank",
                "4",
                "--calibration-quantile",
                "0.90",
                "--bound-fraction",
                "0.50",
                "--minimum-bound",
                "0.001",
                "--minimum-calibration-cells",
                "3",
            )

        stages.append(
            Stage(
                sample,
                "residual_geometry",
                (("reference", paths.reference), ("prototypes", paths.prototypes)),
                paths.geometry,
                geometry_command,
                lambda output, p=paths: _validate_geometry(output, p),
            )
        )
        for role in ("train", "validation"):

            def batch_command(
                output: Path, p: SamplePaths = paths, r: str = role
            ) -> Tuple[str, ...]:
                return (
                    heir_command,
                    "assemble-batch",
                    "--histology",
                    str(p.histology(r)),
                    "--prototypes",
                    str(p.prototypes),
                    "--reference",
                    str(p.reference),
                    "--ood-artifact",
                    str(p.ood),
                    "--output",
                    str(output),
                    "--sample-id",
                    p.sample,
                    "--bag-id",
                    "%s_%s" % (p.sample, r),
                    "--donor-id",
                    p.sample,
                    "--block-id",
                    p.sample + "_FFPE",
                    "--analysis-role",
                    str(RECIPE["batch"]["analysis_role"]),
                    "--artifact-threshold",
                    "0.50",
                    "--markers-per-type",
                    "25",
                )

            stages.append(
                Stage(
                    sample,
                    "batch_" + role,
                    (
                        ("histology", paths.histology(role)),
                        ("prototypes", paths.prototypes),
                        ("reference", paths.reference),
                        ("ood", paths.ood),
                    ),
                    paths.batch(role),
                    batch_command,
                    lambda output, p=paths, r=role: _validate_batch(output, p, r),
                )
            )
    return tuple(stages)


def _stage_recipe(stage: Stage, repository: Path) -> Dict[str, object]:
    return {
        "schema": RECEIPT_SCHEMA,
        "sample": stage.sample,
        "stage": stage.name,
        "inputs": {name: _record(path, repository) for name, path in stage.inputs},
        "output": _path_name(stage.output, repository),
        "recipe": RECIPE,
    }


def _complete_receipt(
    receipt_path: Path, recipe: Mapping[str, object], output: Path, repository: Path
) -> None:
    payload = dict(recipe)
    payload.update({"state": "complete", "output_artifact": _record(output, repository)})
    _atomic_json(receipt_path, payload)


def _run_stage(
    stage: Stage,
    *,
    repository: Path,
    receipt_root: Path,
    execute: bool,
    adopt_existing: bool,
) -> str:
    recipe = _stage_recipe(stage, repository)
    receipt_path = receipt_root / stage.sample / (stage.name + ".json")
    pending = receipt_root / stage.sample / (stage.name + ".pending.npz")
    receipt = _json(receipt_path) if receipt_path.is_file() else None
    if receipt is not None:
        observed_recipe = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "state",
                "output_artifact",
            }
        }
        if observed_recipe != recipe:
            raise RuntimeError("stale stage receipt: %s" % receipt_path)
        state = receipt.get("state")
        if state not in {"planned", "complete"}:
            raise RuntimeError("invalid stage receipt state: %s" % receipt_path)
        if state == "complete":
            if pending.exists() or not stage.output.is_file():
                raise RuntimeError("partial completed stage: %s" % stage.name)
            stage.validate(stage.output)
            output_record = receipt.get("output_artifact")
            if not isinstance(output_record, Mapping) or output_record != _record(
                stage.output, repository
            ):
                raise RuntimeError("stage output hash differs from receipt: %s" % stage.name)
            return "skipped_valid"
        if stage.output.exists() and pending.exists():
            raise RuntimeError("stage has both final and pending outputs: %s" % stage.name)
        if stage.output.is_file():
            stage.validate(stage.output)
            _complete_receipt(receipt_path, recipe, stage.output, repository)
            return "recovered_final"
        if pending.is_file():
            stage.validate(pending)
            stage.output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pending, stage.output)
            _complete_receipt(receipt_path, recipe, stage.output, repository)
            return "recovered_pending"
    else:
        if stage.output.exists():
            if not stage.output.is_file():
                raise RuntimeError("stage output is not a regular file: %s" % stage.output)
            if not adopt_existing:
                raise RuntimeError(
                    "untracked stage output exists; pass --adopt-existing to reproduce and "
                    "hash-verify it: %s" % stage.output
                )
            if not execute:
                print("VERIFY " + shlex.join(stage.command(pending)))
                return "planned_adoption"
            if not pending.exists():
                pending.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(stage.command(pending), cwd=repository, check=True)
            if not pending.is_file():
                raise RuntimeError("adoption command did not emit %s" % pending)
            stage.validate(stage.output)
            stage.validate(pending)
            if _sha256(pending) != _sha256(stage.output):
                raise RuntimeError(
                    "existing output is not byte-identical to the frozen CLI recipe: %s"
                    % stage.output
                )
            pending.unlink()
            _complete_receipt(receipt_path, recipe, stage.output, repository)
            return "adopted_reproduced"
        if pending.exists():
            raise RuntimeError("untracked pending stage output exists: %s" % pending)

    command = stage.command(pending)
    if not execute:
        print(shlex.join(stage.command(stage.output)))
        return "planned"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    planned = dict(recipe)
    planned["state"] = "planned"
    _atomic_json(receipt_path, planned)
    subprocess.run(command, cwd=repository, check=True)
    if not pending.is_file():
        raise RuntimeError("stage command did not emit %s" % pending)
    stage.validate(pending)
    stage.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending, stage.output)
    _complete_receipt(receipt_path, recipe, stage.output, repository)
    return "completed"


def _parser(repository: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=repository / "artifacts/snpatho")
    parser.add_argument("--scanvi-root", type=Path, default=None)
    parser.add_argument("--scanvi-input-root", type=Path, default=None)
    parser.add_argument("--scanvi-provenance", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--heir-command", default="heir")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help="re-run the frozen CLI recipe and adopt only byte-identical legacy outputs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    args = _parser(repository).parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    scanvi_root = (
        (source_root / "r1_scanvi").resolve()
        if args.scanvi_root is None
        else args.scanvi_root.expanduser().resolve()
    )
    scanvi_input_root = (
        (source_root / "r1_ffpe").resolve()
        if args.scanvi_input_root is None
        else args.scanvi_input_root.expanduser().resolve()
    )
    provenance_path = (
        scanvi_root / "provenance.json"
        if args.scanvi_provenance is None
        else args.scanvi_provenance.expanduser().resolve()
    )
    manifest_output = (
        scanvi_root / "preparation_manifest.json"
        if args.manifest_output is None
        else args.manifest_output.expanduser().resolve()
    )
    provenance, samples, input_records = _validate_upstream(
        repository=repository,
        source_root=source_root,
        scanvi_root=scanvi_root,
        scanvi_input_root=scanvi_input_root,
        provenance_path=provenance_path,
    )
    stages = build_stages(samples=samples, heir_command=args.heir_command)
    receipt_root = scanvi_root / ".preparation_receipts"
    statuses = []
    for stage in stages:
        status = _run_stage(
            stage,
            repository=repository,
            receipt_root=receipt_root,
            execute=args.execute,
            adopt_existing=args.adopt_existing,
        )
        statuses.append({"sample": stage.sample, "stage": stage.name, "status": status})

    if args.execute:
        outputs = {
            sample: {
                "prototypes": _record(paths.prototypes, repository),
                "residual_geometry": _record(paths.geometry, repository),
                "batch_train": _record(paths.batch("train"), repository),
                "batch_validation": _record(paths.batch("validation"), repository),
            }
            for sample, paths in samples.items()
        }
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "latent_space_id": provenance["latent_space_id"],
            "recipe": RECIPE,
            "inputs": input_records,
            "outputs": outputs,
        }
        if manifest_output.exists():
            if _json(manifest_output) != manifest:
                raise RuntimeError("existing preparation manifest is stale: %s" % manifest_output)
        else:
            _atomic_json(manifest_output, manifest)
    print(json.dumps({"execute": args.execute, "stages": statuses}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
