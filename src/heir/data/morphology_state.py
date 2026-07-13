"""Fail-closed rows for measurement-qualified morphology-state experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> object:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError("morphology-ridge field %s must be scalar" % name)
    return value.item()


def _strings(value: object, name: str, *, unique: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("morphology-ridge %s must be a vector" % name)
    result = array.astype(str)
    if any(not item.strip() for item in result.tolist()):
        raise ValueError("morphology-ridge %s contains empty values" % name)
    if unique and len(set(result.tolist())) != len(result):
        raise ValueError("morphology-ridge %s must be unique" % name)
    return result


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("morphology-ridge %s must be a lowercase SHA-256" % name)
    return digest


def _boolean(archive: Mapping[str, np.ndarray], name: str) -> bool:
    value = np.asarray(archive[name])
    if value.ndim != 0 or value.dtype != np.bool_:
        raise ValueError("morphology-ridge %s must be a boolean scalar" % name)
    return bool(value.item())


def _json_mapping(archive: Mapping[str, np.ndarray], name: str) -> Mapping[str, object]:
    try:
        value = json.loads(str(_scalar(archive, name)))
    except json.JSONDecodeError as error:
        raise ValueError("morphology-ridge %s is not valid JSON" % name) from error
    if not isinstance(value, Mapping):
        raise ValueError("morphology-ridge %s must encode an object" % name)
    return value


def _matrix(archive: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(archive[name], dtype=np.float64)


@dataclass(frozen=True)
class MorphologyRidgeDatasetArtifact:
    """Evaluation rows plus every prespecified image and nuisance arm.

    Version three deliberately carries the effective experiment, rather than a
    single convenient feature matrix.  In particular, crop source is explicit;
    observation level is never used as a proxy for an intrinsic-morphology arm.
    """

    observation_ids: np.ndarray
    donor_ids: np.ndarray
    block_ids: np.ndarray
    roi_ids: np.ndarray
    type_labels: np.ndarray
    type_names: Tuple[str, ...]
    frozen_features: np.ndarray
    molecular_targets: np.ndarray
    reference_means: np.ndarray
    coordinate_features: np.ndarray
    stain_features: np.ndarray
    stain_feature_names: Tuple[str, ...]
    composition_features: np.ndarray
    composition_feature_names: Tuple[str, ...]
    technical_covariates: np.ndarray
    technical_covariate_names: Tuple[str, ...]
    gene_ids: Tuple[str, ...]
    type_marker_gene_ids: Tuple[str, ...]
    feature_space_id: str
    feature_checkpoint_sha256: str
    molecular_space_id: str
    reference_source_sha256: str
    label_source_sha256: str
    target_source_sha256: str
    registration_source_sha256: str
    exclusion_policy_sha256: str
    registration_method: str
    encoder_name: str
    crop_scale: str
    cohort_id: str
    cohort_release: str
    assay: str
    observation_level: str
    target_construction: str
    reference_pool_independent: bool
    labels_independent_of_images: bool
    registration_is_one_to_one: bool
    role: str

    section_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))
    disease_states: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))
    site_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))
    batch_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=str))
    image_feature_tensor: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64)
    )
    crop_ids: Tuple[str, ...] = ()
    crop_roles: Tuple[str, ...] = ()
    crop_comparison_families: Tuple[str, ...] = ()
    primary_crop_id: str = ""
    nuclear_morphometrics: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    nuclear_morphometric_names: Tuple[str, ...] = ()
    cell_morphometrics: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    cell_morphometric_names: Tuple[str, ...] = ()
    cellvit_context_features: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    cellvit_context_feature_names: Tuple[str, ...] = ()
    local_density_features: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    local_density_feature_names: Tuple[str, ...] = ()
    boundary_features: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    boundary_feature_names: Tuple[str, ...] = ()
    spatial_control_features: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    spatial_control_feature_names: Tuple[str, ...] = ()
    planned_stratum_ids: Tuple[str, ...] = ()
    planned_stratum_manifest_sha256: str = ""
    coverage_audit: Mapping[str, object] = field(default_factory=dict)
    reference_evaluation_balance: Mapping[str, object] = field(default_factory=dict)
    study_manifest_sha256: str = ""
    opening_receipt_sha256: str = ""
    measurement_receipt_sha256: str = ""
    measurement_source_sha256: str = ""
    hypothesis_ids: Tuple[str, ...] = ()
    scientific_scope: str = ""
    evidence_scope: str = ""
    authorizes_nucleus_intrinsic_claim: bool = False
    reference_split_ids: Tuple[str, ...] = ()
    reference_means_by_split: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64)
    )

    SCHEMA = "heir.morphology_ridge_dataset.v4"

    @classmethod
    def load_npz(cls, path: PathLike, *, role: str) -> "MorphologyRidgeDatasetArtifact":
        if role not in {"development", "locked_test"}:
            raise ValueError("morphology-ridge role must be development or locked_test")
        with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as archive:
            required = {
                "schema_version",
                "observation_ids",
                "donor_ids",
                "block_ids",
                "roi_ids",
                "section_ids",
                "disease_states",
                "site_ids",
                "batch_ids",
                "type_labels",
                "type_names",
                "frozen_features",
                "image_feature_tensor",
                "crop_ids",
                "crop_roles",
                "crop_comparison_families",
                "primary_crop_id",
                "molecular_targets",
                "reference_means",
                "coordinate_features",
                "stain_features",
                "stain_feature_names",
                "composition_features",
                "composition_feature_names",
                "technical_covariates",
                "technical_covariate_names",
                "nuclear_morphometrics",
                "nuclear_morphometric_names",
                "cell_morphometrics",
                "cell_morphometric_names",
                "cellvit_context_features",
                "cellvit_context_feature_names",
                "local_density_features",
                "local_density_feature_names",
                "boundary_features",
                "boundary_feature_names",
                "spatial_control_features",
                "spatial_control_feature_names",
                "gene_ids",
                "type_marker_gene_ids",
                "planned_stratum_ids",
                "planned_stratum_manifest_sha256",
                "coverage_audit_json",
                "reference_evaluation_balance_json",
                "study_manifest_sha256",
                "opening_receipt_sha256",
                "measurement_receipt_sha256",
                "measurement_source_sha256",
                "hypothesis_ids",
                "scientific_scope",
                "evidence_scope",
                "authorizes_nucleus_intrinsic_claim",
                "reference_split_ids",
                "reference_means_by_split",
                "feature_space_id",
                "feature_checkpoint_sha256",
                "molecular_space_id",
                "reference_source_sha256",
                "label_source_sha256",
                "target_source_sha256",
                "registration_source_sha256",
                "exclusion_policy_sha256",
                "registration_method",
                "encoder_name",
                "crop_scale",
                "cohort_id",
                "cohort_release",
                "assay",
                "observation_level",
                "target_construction",
                "reference_pool_independent",
                "labels_independent_of_images",
                "registration_is_one_to_one",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("morphology-ridge artifact is missing: %s" % ", ".join(missing))
            if str(_scalar(archive, "schema_version")) != cls.SCHEMA:
                raise ValueError("morphology-ridge dataset schema is unsupported")
            observations = _strings(archive["observation_ids"], "observation_ids", unique=True)
            labels = np.asarray(archive["type_labels"])
            if labels.dtype.kind not in "iu" or labels.shape != (len(observations),):
                raise ValueError("morphology-ridge type_labels must align with observations")
            value = cls(
                observation_ids=observations,
                donor_ids=_strings(archive["donor_ids"], "donor_ids"),
                block_ids=_strings(archive["block_ids"], "block_ids"),
                roi_ids=_strings(archive["roi_ids"], "roi_ids"),
                section_ids=_strings(archive["section_ids"], "section_ids"),
                disease_states=_strings(archive["disease_states"], "disease_states"),
                site_ids=_strings(archive["site_ids"], "site_ids"),
                batch_ids=_strings(archive["batch_ids"], "batch_ids"),
                type_labels=labels.astype(np.int64, copy=False),
                type_names=tuple(_strings(archive["type_names"], "type_names", unique=True)),
                frozen_features=_matrix(archive, "frozen_features"),
                image_feature_tensor=_matrix(archive, "image_feature_tensor"),
                crop_ids=tuple(_strings(archive["crop_ids"], "crop_ids", unique=True)),
                crop_roles=tuple(_strings(archive["crop_roles"], "crop_roles")),
                crop_comparison_families=tuple(
                    _strings(archive["crop_comparison_families"], "crop_comparison_families")
                ),
                primary_crop_id=str(_scalar(archive, "primary_crop_id")),
                molecular_targets=_matrix(archive, "molecular_targets"),
                reference_means=_matrix(archive, "reference_means"),
                coordinate_features=_matrix(archive, "coordinate_features"),
                stain_features=_matrix(archive, "stain_features"),
                stain_feature_names=tuple(
                    _strings(archive["stain_feature_names"], "stain_feature_names", unique=True)
                ),
                composition_features=_matrix(archive, "composition_features"),
                composition_feature_names=tuple(
                    _strings(
                        archive["composition_feature_names"],
                        "composition_feature_names",
                        unique=True,
                    )
                ),
                technical_covariates=_matrix(archive, "technical_covariates"),
                technical_covariate_names=tuple(
                    _strings(
                        archive["technical_covariate_names"],
                        "technical_covariate_names",
                        unique=True,
                    )
                ),
                nuclear_morphometrics=_matrix(archive, "nuclear_morphometrics"),
                nuclear_morphometric_names=tuple(
                    _strings(
                        archive["nuclear_morphometric_names"],
                        "nuclear_morphometric_names",
                        unique=True,
                    )
                ),
                cell_morphometrics=_matrix(archive, "cell_morphometrics"),
                cell_morphometric_names=tuple(
                    _strings(
                        archive["cell_morphometric_names"],
                        "cell_morphometric_names",
                        unique=True,
                    )
                ),
                cellvit_context_features=_matrix(archive, "cellvit_context_features"),
                cellvit_context_feature_names=tuple(
                    _strings(
                        archive["cellvit_context_feature_names"],
                        "cellvit_context_feature_names",
                        unique=True,
                    )
                ),
                local_density_features=_matrix(archive, "local_density_features"),
                local_density_feature_names=tuple(
                    _strings(
                        archive["local_density_feature_names"],
                        "local_density_feature_names",
                        unique=True,
                    )
                ),
                boundary_features=_matrix(archive, "boundary_features"),
                boundary_feature_names=tuple(
                    _strings(
                        archive["boundary_feature_names"],
                        "boundary_feature_names",
                        unique=True,
                    )
                ),
                spatial_control_features=_matrix(archive, "spatial_control_features"),
                spatial_control_feature_names=tuple(
                    _strings(
                        archive["spatial_control_feature_names"],
                        "spatial_control_feature_names",
                        unique=True,
                    )
                ),
                gene_ids=tuple(_strings(archive["gene_ids"], "gene_ids", unique=True)),
                type_marker_gene_ids=tuple(
                    _strings(archive["type_marker_gene_ids"], "type_marker_gene_ids", unique=True)
                ),
                planned_stratum_ids=tuple(
                    _strings(archive["planned_stratum_ids"], "planned_stratum_ids", unique=True)
                ),
                planned_stratum_manifest_sha256=_sha256(
                    _scalar(archive, "planned_stratum_manifest_sha256"),
                    "planned_stratum_manifest_sha256",
                ),
                coverage_audit=_json_mapping(archive, "coverage_audit_json"),
                reference_evaluation_balance=_json_mapping(
                    archive, "reference_evaluation_balance_json"
                ),
                study_manifest_sha256=_sha256(
                    _scalar(archive, "study_manifest_sha256"), "study_manifest_sha256"
                ),
                opening_receipt_sha256=str(_scalar(archive, "opening_receipt_sha256")),
                measurement_receipt_sha256=_sha256(
                    _scalar(archive, "measurement_receipt_sha256"),
                    "measurement_receipt_sha256",
                ),
                measurement_source_sha256=_sha256(
                    _scalar(archive, "measurement_source_sha256"),
                    "measurement_source_sha256",
                ),
                hypothesis_ids=tuple(
                    _strings(archive["hypothesis_ids"], "hypothesis_ids", unique=True)
                ),
                scientific_scope=str(_scalar(archive, "scientific_scope")),
                evidence_scope=str(_scalar(archive, "evidence_scope")),
                authorizes_nucleus_intrinsic_claim=_boolean(
                    archive, "authorizes_nucleus_intrinsic_claim"
                ),
                reference_split_ids=tuple(
                    _strings(archive["reference_split_ids"], "reference_split_ids", unique=True)
                ),
                reference_means_by_split=_matrix(archive, "reference_means_by_split"),
                feature_space_id=str(_scalar(archive, "feature_space_id")),
                feature_checkpoint_sha256=_sha256(
                    _scalar(archive, "feature_checkpoint_sha256"),
                    "feature_checkpoint_sha256",
                ),
                molecular_space_id=str(_scalar(archive, "molecular_space_id")),
                reference_source_sha256=_sha256(
                    _scalar(archive, "reference_source_sha256"), "reference_source_sha256"
                ),
                label_source_sha256=_sha256(
                    _scalar(archive, "label_source_sha256"), "label_source_sha256"
                ),
                target_source_sha256=_sha256(
                    _scalar(archive, "target_source_sha256"), "target_source_sha256"
                ),
                registration_source_sha256=_sha256(
                    _scalar(archive, "registration_source_sha256"),
                    "registration_source_sha256",
                ),
                exclusion_policy_sha256=_sha256(
                    _scalar(archive, "exclusion_policy_sha256"), "exclusion_policy_sha256"
                ),
                registration_method=str(_scalar(archive, "registration_method")),
                encoder_name=str(_scalar(archive, "encoder_name")),
                crop_scale=str(_scalar(archive, "crop_scale")),
                cohort_id=str(_scalar(archive, "cohort_id")),
                cohort_release=str(_scalar(archive, "cohort_release")),
                assay=str(_scalar(archive, "assay")),
                observation_level=str(_scalar(archive, "observation_level")),
                target_construction=str(_scalar(archive, "target_construction")),
                reference_pool_independent=_boolean(archive, "reference_pool_independent"),
                labels_independent_of_images=_boolean(archive, "labels_independent_of_images"),
                registration_is_one_to_one=_boolean(archive, "registration_is_one_to_one"),
                role=role,
            )
        value.validate()
        if role == "locked_test" and value.evidence_scope == "development_pilot":
            raise ValueError("development-pilot data cannot be opened as a locked test")
        return value

    def validate(self) -> None:
        observations = len(self.observation_ids)
        identities = (
            self.donor_ids,
            self.block_ids,
            self.roi_ids,
            self.section_ids,
            self.disease_states,
            self.site_ids,
            self.batch_ids,
        )
        if not observations or any(len(values) != observations for values in identities):
            raise ValueError("morphology-ridge row identities are empty or misaligned")
        if (
            not self.type_names
            or np.any(self.type_labels < 0)
            or np.any(self.type_labels >= len(self.type_names))
        ):
            raise ValueError("morphology-ridge labels fall outside the qualified ontology")
        named_matrices = {
            "stain_features": (self.stain_features, self.stain_feature_names),
            "composition_features": (
                self.composition_features,
                self.composition_feature_names,
            ),
            "technical_covariates": (
                self.technical_covariates,
                self.technical_covariate_names,
            ),
            "nuclear_morphometrics": (
                self.nuclear_morphometrics,
                self.nuclear_morphometric_names,
            ),
            "cell_morphometrics": (
                self.cell_morphometrics,
                self.cell_morphometric_names,
            ),
            "cellvit_context_features": (
                self.cellvit_context_features,
                self.cellvit_context_feature_names,
            ),
            "local_density_features": (
                self.local_density_features,
                self.local_density_feature_names,
            ),
            "boundary_features": (self.boundary_features, self.boundary_feature_names),
            "spatial_control_features": (
                self.spatial_control_features,
                self.spatial_control_feature_names,
            ),
        }
        matrices = {
            "frozen_features": self.frozen_features,
            "molecular_targets": self.molecular_targets,
            "reference_means": self.reference_means,
            "coordinate_features": self.coordinate_features,
            **{name: values[0] for name, values in named_matrices.items()},
        }
        for name, values in matrices.items():
            if values.ndim != 2 or len(values) != observations or not np.isfinite(values).all():
                raise ValueError("morphology-ridge %s is malformed" % name)
        for name, (values, names) in named_matrices.items():
            if values.shape[1] != len(names):
                raise ValueError("morphology-ridge %s differ from their names" % name)
        if not self.frozen_features.shape[1] or not self.coordinate_features.shape[1]:
            raise ValueError("morphology-ridge feature matrices cannot be empty")
        if self.molecular_targets.shape != self.reference_means.shape:
            raise ValueError("molecular targets and independent reference means must align")
        if self.molecular_targets.shape[1] != len(self.gene_ids):
            raise ValueError("molecular target width differs from measurement-qualified panel")
        if set(self.gene_ids) & set(self.type_marker_gene_ids):
            raise ValueError("type-marker genes leak into the molecular evaluation panel")
        crops = len(self.crop_ids)
        if (
            self.image_feature_tensor.ndim != 3
            or self.image_feature_tensor.shape[0] != observations
            or self.image_feature_tensor.shape[1] != crops
            or self.image_feature_tensor.shape[2] != self.frozen_features.shape[1]
            or not np.isfinite(self.image_feature_tensor).all()
            or len(self.crop_roles) != crops
            or len(self.crop_comparison_families) != crops
            or self.primary_crop_id not in self.crop_ids
        ):
            raise ValueError("morphology-ridge crop ladder is malformed")
        primary = self.crop_ids.index(self.primary_crop_id)
        if not np.allclose(self.frozen_features, self.image_feature_tensor[:, primary, :]):
            raise ValueError("primary frozen features differ from the declared crop tensor arm")
        if self.reference_means_by_split.ndim != 3 or (
            self.reference_means_by_split.shape
            != (observations, len(self.reference_split_ids), len(self.gene_ids))
        ):
            raise ValueError("reference means by frozen split are malformed")
        if not np.isfinite(self.reference_means_by_split).all():
            raise ValueError("reference means by frozen split contain non-finite values")
        if not self.reference_split_ids:
            raise ValueError("at least one frozen reference/evaluation split is required")
        if not np.array_equal(self.reference_means, self.reference_means_by_split[:, 0, :]):
            raise ValueError(
                "standalone reference means differ from the primary frozen reference split"
            )
        if not self.planned_stratum_ids or not self.coverage_audit:
            raise ValueError("planned biological coverage must be carried into the artifact")
        memberships = self.coverage_audit.get("reference_membership_sha256_by_split")
        if (
            not isinstance(memberships, Mapping)
            or set(str(value) for value in memberships) != set(self.reference_split_ids)
            or len(set(str(value) for value in memberships.values()))
            != len(self.reference_split_ids)
        ):
            raise ValueError("frozen reference-split memberships are missing or degenerate")
        for split_id, digest in memberships.items():
            _sha256(digest, "reference membership %s" % split_id)
        if not self.reference_evaluation_balance:
            raise ValueError("reference/evaluation balance diagnostics are required")
        if not (
            self.reference_pool_independent
            and self.labels_independent_of_images
            and self.registration_is_one_to_one
        ):
            raise ValueError("morphology-ridge independence declarations are not satisfied")
        if self.evidence_scope not in {"internal_locked_hest", "development_pilot"}:
            raise ValueError("morphology-ridge evidence scope is unsupported")
        if self.cohort_id == "HESCAPE" and self.evidence_scope != "development_pilot":
            raise ValueError("HESCAPE is restricted to development-pilot evidence")
        if self.cohort_id == "HEST":
            _sha256(self.opening_receipt_sha256, "opening_receipt_sha256")
        elif self.opening_receipt_sha256:
            raise ValueError("development-pilot artifacts cannot claim an opening receipt")
        declared = (
            self.feature_space_id,
            self.molecular_space_id,
            self.registration_method,
            self.encoder_name,
            self.crop_scale,
            self.cohort_id,
            self.cohort_release,
            self.assay,
            self.observation_level,
            self.target_construction,
            self.primary_crop_id,
            self.scientific_scope,
        )
        if not all(value.strip() for value in declared) or not self.hypothesis_ids:
            raise ValueError("morphology-ridge scientific identities must be explicit")
        sources = {
            self.feature_checkpoint_sha256,
            self.reference_source_sha256,
            self.label_source_sha256,
            self.target_source_sha256,
            self.registration_source_sha256,
            self.exclusion_policy_sha256,
        }
        if len(sources) != 6:
            raise ValueError("morphology-ridge evidence sources must be independently identifiable")

    def validate_compatible(self, locked_test: "MorphologyRidgeDatasetArtifact") -> None:
        if self.role != "development" or locked_test.role != "locked_test":
            raise ValueError("morphology-ridge compatibility requires development and locked_test")
        if self.evidence_scope == "development_pilot" or locked_test.evidence_scope == (
            "development_pilot"
        ):
            raise ValueError("development-pilot artifacts cannot enter a locked-test benchmark")
        for name in (
            "type_names",
            "gene_ids",
            "type_marker_gene_ids",
            "feature_space_id",
            "feature_checkpoint_sha256",
            "molecular_space_id",
            "registration_method",
            "encoder_name",
            "crop_scale",
            "crop_ids",
            "crop_roles",
            "crop_comparison_families",
            "primary_crop_id",
            "cohort_id",
            "cohort_release",
            "assay",
            "observation_level",
            "target_construction",
            "label_source_sha256",
            "registration_source_sha256",
            "exclusion_policy_sha256",
            "study_manifest_sha256",
            "opening_receipt_sha256",
            "measurement_receipt_sha256",
            "measurement_source_sha256",
            "hypothesis_ids",
            "scientific_scope",
            "evidence_scope",
            "reference_split_ids",
        ):
            if getattr(self, name) != getattr(locked_test, name):
                raise ValueError("development and locked-test morphology-ridge %s differ" % name)
        for name in (
            "frozen_features",
            "image_feature_tensor",
            "molecular_targets",
            "coordinate_features",
            "stain_features",
            "composition_features",
            "nuclear_morphometrics",
            "cell_morphometrics",
            "cellvit_context_features",
            "local_density_features",
            "boundary_features",
            "spatial_control_features",
        ):
            if getattr(self, name).shape[1:] != getattr(locked_test, name).shape[1:]:
                raise ValueError("development and locked-test %s widths differ" % name)
        for name in (
            "stain_feature_names",
            "composition_feature_names",
            "technical_covariate_names",
            "nuclear_morphometric_names",
            "cell_morphometric_names",
            "cellvit_context_feature_names",
            "local_density_feature_names",
            "boundary_feature_names",
            "spatial_control_feature_names",
        ):
            if getattr(self, name) != getattr(locked_test, name):
                raise ValueError("development and locked-test %s differ" % name)
        overlap = sorted(set(self.donor_ids) & set(locked_test.donor_ids))
        if overlap:
            raise ValueError("development and locked-test donors overlap: %s" % ", ".join(overlap))
        if self.reference_source_sha256 == locked_test.reference_source_sha256:
            raise ValueError("development and locked-test reference pools must be distinct")
        if self.target_source_sha256 == locked_test.target_source_sha256:
            raise ValueError("development and locked-test molecular targets must be distinct")


__all__ = ["MorphologyRidgeDatasetArtifact"]
