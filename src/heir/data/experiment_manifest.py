"""Immutable scientific contract for one morphology-state experiment arm."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Union

from heir.utils import sha256_file

PathLike = Union[str, Path]

EXPERIMENT_MANIFEST_SCHEMA = "heir.morphology_experiment_manifest.v1"


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible content with stable ordering and separators."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordered_ids_sha256(values: Sequence[object]) -> str:
    """Bind both identity and order of a vector of schema labels."""

    return canonical_sha256([str(value) for value in values])


def _mapping(value: object, name: str, required: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("experiment manifest %s is incomplete" % name)
    return value


def _strings(value: object, name: str, *, unique: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("experiment manifest %s must be a list" % name)
    result = tuple(str(item) for item in value)
    if not result or any(not item.strip() for item in result):
        raise ValueError("experiment manifest %s contains empty values" % name)
    if unique and len(set(result)) != len(result):
        raise ValueError("experiment manifest %s must be unique" % name)
    return result


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("experiment manifest %s must be a lowercase SHA-256" % name)
    return digest


@dataclass(frozen=True)
class ExperimentManifest:
    """Validated manifest content plus the exact file digest accepted by a benchmark."""

    path: Path
    sha256: str
    content: Mapping[str, object]
    experiment_role: str
    development_donors: tuple[str, ...]
    locked_test_donors: tuple[str, ...]
    gene_ids: tuple[str, ...]
    type_names: tuple[str, ...]

    @classmethod
    def load(cls, path: PathLike, *, verify_protocol: bool = True) -> "ExperimentManifest":
        resolved = Path(path).expanduser().resolve()
        try:
            content = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("experiment manifest is not valid JSON") from error
        if not isinstance(content, Mapping) or content.get("schema") != EXPERIMENT_MANIFEST_SCHEMA:
            raise ValueError("experiment manifest schema is unsupported")
        required = {
            "schema",
            "experiment_role",
            "scientific_scope",
            "code_commit_sha",
            "protocol",
            "source",
            "partitions",
            "encoder",
            "preprocessing",
            "target",
            "labels",
            "reference_pool",
            "nuisance_covariates",
            "gate",
        }
        if not required.issubset(content):
            raise ValueError("experiment manifest is incomplete")
        role = str(content["experiment_role"])
        allowed = {
            "regional_hescape_uni2h",
            "regional_hescape_uni2h_context",
            "regional_hescape_hoptimus1",
            "primary_hest_uni2h",
            "primary_hest_hoptimus1",
            "replication_hest_hoptimus1",
            "replication_hest_h0mini",
        }
        if role not in allowed and not role.startswith("external_confirmation_"):
            raise ValueError("experiment manifest role is unsupported")
        commit = str(content["code_commit_sha"])
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("experiment manifest code commit must be a full Git SHA")

        protocol = _mapping(content["protocol"], "protocol", {"path", "sha256"})
        protocol_sha = _sha256(protocol["sha256"], "protocol.sha256")
        if verify_protocol:
            protocol_path = Path(str(protocol["path"])).expanduser().resolve()
            if not protocol_path.is_file() or sha256_file(protocol_path) != protocol_sha:
                raise ValueError("experiment manifest protocol file or SHA differs")

        source = _mapping(
            content["source"],
            "source",
            {
                "schema",
                "schema_sha256",
                "observations_sha256",
                "cohort_id",
                "cohort_release",
                "assay",
                "observation_level",
                "donor_sections",
            },
        )
        _sha256(source["schema_sha256"], "source.schema_sha256")
        _sha256(source["observations_sha256"], "source.observations_sha256")
        donor_sections = source["donor_sections"]
        if not isinstance(donor_sections, Mapping) or not donor_sections:
            raise ValueError("experiment manifest source donor sections are missing")
        normalized_sections = {}
        for donor, sections in donor_sections.items():
            normalized_sections[str(donor)] = _strings(
                sections, "source.donor_sections", unique=True
            )

        partitions = _mapping(
            content["partitions"],
            "partitions",
            {"development_donors", "locked_test_donors"},
        )
        development = _strings(
            partitions["development_donors"], "partitions.development_donors", unique=True
        )
        locked = _strings(
            partitions["locked_test_donors"], "partitions.locked_test_donors", unique=True
        )
        if set(development) & set(locked):
            raise ValueError("experiment manifest donor partitions overlap")
        if set(normalized_sections) != set(development) | set(locked):
            raise ValueError("experiment manifest sections do not match donor partitions")

        encoder = _mapping(
            content["encoder"],
            "encoder",
            {"repository", "revision", "checkpoint_sha256", "feature_width"},
        )
        _sha256(encoder["checkpoint_sha256"], "encoder.checkpoint_sha256")
        if not str(encoder["repository"]).strip() or not str(encoder["revision"]).strip():
            raise ValueError("experiment manifest encoder identity is empty")
        expected_encoder = {
            "regional_hescape_uni2h": "MahmoodLab/UNI2-h",
            "regional_hescape_uni2h_context": "MahmoodLab/UNI2-h",
            "primary_hest_uni2h": "MahmoodLab/UNI2-h",
            "regional_hescape_hoptimus1": "bioptimus/H-optimus-1",
            "primary_hest_hoptimus1": "bioptimus/H-optimus-1",
            "replication_hest_hoptimus1": "bioptimus/H-optimus-1",
            "replication_hest_h0mini": "bioptimus/H0-mini",
        }.get(role)
        if expected_encoder is not None and str(encoder["repository"]) != expected_encoder:
            raise ValueError("experiment manifest role requires encoder %s" % expected_encoder)
        if not isinstance(encoder["feature_width"], int) or int(encoder["feature_width"]) <= 0:
            raise ValueError("experiment manifest encoder feature width must be positive")

        preprocessing = _mapping(
            content["preprocessing"],
            "preprocessing",
            {
                "implementation",
                "implementation_sha256",
                "crop_role",
                "crop_diameter_um",
                "source_mpp",
                "model_mpp",
                "model_input_pixels",
                "mask_mode",
            },
        )
        _sha256(preprocessing["implementation_sha256"], "preprocessing.implementation_sha256")
        for name in ("crop_diameter_um", "source_mpp", "model_mpp"):
            value = preprocessing[name]
            if not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError("experiment manifest preprocessing %s must be positive" % name)
        if (
            not isinstance(preprocessing["model_input_pixels"], int)
            or int(preprocessing["model_input_pixels"]) <= 0
        ):
            raise ValueError("experiment manifest model input pixels must be positive")
        if str(preprocessing["mask_mode"]) not in {
            "none",
            "nucleus",
            "cell",
            "context_annulus",
        }:
            raise ValueError("experiment manifest mask mode is unsupported")

        target = _mapping(
            content["target"],
            "target",
            {"construction", "schema", "gene_ids", "gene_panel_sha256"},
        )
        genes = _strings(target["gene_ids"], "target.gene_ids", unique=True)
        if _sha256(target["gene_panel_sha256"], "target.gene_panel_sha256") != ordered_ids_sha256(
            genes
        ):
            raise ValueError("experiment manifest ordered gene-panel hash differs")

        labels = _mapping(
            content["labels"],
            "labels",
            {
                "procedure",
                "source_sha256",
                "type_names",
                "marker_gene_ids",
                "conditioning_levels",
            },
        )
        _sha256(labels["source_sha256"], "labels.source_sha256")
        type_names = _strings(labels["type_names"], "labels.type_names", unique=True)
        marker_genes = _strings(labels["marker_gene_ids"], "labels.marker_gene_ids", unique=True)
        _strings(labels["conditioning_levels"], "labels.conditioning_levels", unique=True)
        if set(genes) & set(marker_genes):
            raise ValueError("experiment manifest marker genes leak into evaluation targets")

        reference = _mapping(
            content["reference_pool"],
            "reference_pool",
            {
                "construction",
                "spatially_disjoint",
                "minimum_per_donor_type",
                "observation_manifest_sha256",
            },
        )
        if reference["spatially_disjoint"] is not True:
            raise ValueError("experiment manifest reference pool must be spatially disjoint")
        if (
            not isinstance(reference["minimum_per_donor_type"], int)
            or int(reference["minimum_per_donor_type"]) <= 0
        ):
            raise ValueError("experiment manifest reference minimum must be positive")
        _sha256(
            reference["observation_manifest_sha256"],
            "reference_pool.observation_manifest_sha256",
        )

        _strings(content["nuisance_covariates"], "nuisance_covariates", unique=True)
        gate = _mapping(
            content["gate"],
            "gate",
            {
                "ranks",
                "ridge_penalties",
                "permutation_seeds",
                "permutations_per_seed",
                "minimum_support",
                "minimum_development_donors",
                "minimum_locked_donors",
                "minimum_coverage_fraction",
                "minimum_shuffled_fraction",
                "nulls",
                "thresholds",
            },
        )
        for name in ("ranks", "ridge_penalties", "permutation_seeds"):
            if not isinstance(gate[name], list) or not gate[name]:
                raise ValueError("experiment manifest gate %s cannot be empty" % name)
        if int(gate["permutations_per_seed"]) < 100:
            raise ValueError("experiment manifest needs at least 100 permutations per seed")
        for name in (
            "minimum_support",
            "minimum_development_donors",
            "minimum_locked_donors",
        ):
            if int(gate[name]) <= 0:
                raise ValueError("experiment manifest gate %s must be positive" % name)
        for name in ("minimum_coverage_fraction", "minimum_shuffled_fraction"):
            value = float(gate[name])
            if not 0 < value <= 1:
                raise ValueError("experiment manifest gate %s must be in (0, 1]" % name)
        nulls = _strings(gate["nulls"], "gate.nulls", unique=True)
        if set(nulls) != {"within_roi_derangement", "spatial_block_reassignment"}:
            raise ValueError("experiment manifest must freeze both registration nulls")
        thresholds = gate["thresholds"]
        if not isinstance(thresholds, Mapping) or not thresholds:
            raise ValueError("experiment manifest gate thresholds are missing")
        if any(not isinstance(value, (int, float)) for value in thresholds.values()):
            raise ValueError("experiment manifest gate thresholds must be numeric")

        return cls(
            path=resolved,
            sha256=sha256_file(resolved),
            content=content,
            experiment_role=role,
            development_donors=development,
            locked_test_donors=locked,
            gene_ids=genes,
            type_names=type_names,
        )

    def validate_source(self, source_path: PathLike, source_schema: str) -> None:
        """Verify the exact source artifact and schema bound by this manifest."""

        source = self.content["source"]
        if sha256_file(Path(source_path).expanduser().resolve()) != source["observations_sha256"]:
            raise ValueError("experiment source observations differ from immutable manifest")
        if str(source_schema) != str(source["schema"]):
            raise ValueError("experiment source schema differs from immutable manifest")


__all__ = [
    "EXPERIMENT_MANIFEST_SCHEMA",
    "ExperimentManifest",
    "canonical_sha256",
    "ordered_ids_sha256",
]
