"""Fail-closed rows for the oracle-type morphology ridge experiment."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class MorphologyRidgeDatasetArtifact:
    """Registered evaluation rows with independent matched-reference means."""

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

    SCHEMA = "heir.morphology_ridge_dataset.v2"

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
                "type_labels",
                "type_names",
                "frozen_features",
                "molecular_targets",
                "reference_means",
                "coordinate_features",
                "stain_features",
                "stain_feature_names",
                "composition_features",
                "composition_feature_names",
                "technical_covariates",
                "technical_covariate_names",
                "gene_ids",
                "type_marker_gene_ids",
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
                type_labels=labels.astype(np.int64, copy=False),
                type_names=tuple(_strings(archive["type_names"], "type_names", unique=True)),
                frozen_features=np.asarray(archive["frozen_features"], dtype=np.float64),
                molecular_targets=np.asarray(archive["molecular_targets"], dtype=np.float64),
                reference_means=np.asarray(archive["reference_means"], dtype=np.float64),
                coordinate_features=np.asarray(archive["coordinate_features"], dtype=np.float64),
                stain_features=np.asarray(archive["stain_features"], dtype=np.float64),
                stain_feature_names=tuple(
                    _strings(archive["stain_feature_names"], "stain_feature_names", unique=True)
                ),
                composition_features=np.asarray(
                    archive["composition_features"], dtype=np.float64
                ),
                composition_feature_names=tuple(
                    _strings(
                        archive["composition_feature_names"],
                        "composition_feature_names",
                        unique=True,
                    )
                ),
                technical_covariates=np.asarray(archive["technical_covariates"], dtype=np.float64),
                technical_covariate_names=tuple(
                    _strings(
                        archive["technical_covariate_names"],
                        "technical_covariate_names",
                        unique=True,
                    )
                ),
                gene_ids=tuple(_strings(archive["gene_ids"], "gene_ids", unique=True)),
                type_marker_gene_ids=tuple(
                    _strings(archive["type_marker_gene_ids"], "type_marker_gene_ids", unique=True)
                ),
                feature_space_id=str(_scalar(archive, "feature_space_id")),
                feature_checkpoint_sha256=_sha256(
                    _scalar(archive, "feature_checkpoint_sha256"), "feature_checkpoint_sha256"
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
        return value

    def validate(self) -> None:
        observations = len(self.observation_ids)
        if not observations or any(
            len(values) != observations for values in (self.donor_ids, self.block_ids, self.roi_ids)
        ):
            raise ValueError("morphology-ridge row identities are empty or misaligned")
        if (
            not self.type_names
            or np.any(self.type_labels < 0)
            or np.any(self.type_labels >= len(self.type_names))
        ):
            raise ValueError("morphology-ridge labels fall outside the development ontology")
        matrices = {
            "frozen_features": self.frozen_features,
            "molecular_targets": self.molecular_targets,
            "reference_means": self.reference_means,
            "coordinate_features": self.coordinate_features,
            "stain_features": self.stain_features,
            "composition_features": self.composition_features,
            "technical_covariates": self.technical_covariates,
        }
        for name, values in matrices.items():
            if values.ndim != 2 or len(values) != observations or not np.isfinite(values).all():
                raise ValueError("morphology-ridge %s is malformed" % name)
        if not self.frozen_features.shape[1] or not self.coordinate_features.shape[1]:
            raise ValueError("morphology-ridge feature matrices cannot be empty")
        if self.stain_features.shape[1] != len(self.stain_feature_names):
            raise ValueError("stain features differ from their names")
        if self.composition_features.shape[1] != len(self.composition_feature_names):
            raise ValueError("composition features differ from their names")
        if self.molecular_targets.shape != self.reference_means.shape:
            raise ValueError("molecular targets and independent reference means must align")
        if self.molecular_targets.shape[1] != len(self.gene_ids):
            raise ValueError("molecular target width differs from frozen gene panel")
        if self.technical_covariates.shape[1] != len(self.technical_covariate_names):
            raise ValueError("technical covariates differ from their names")
        if set(self.gene_ids) & set(self.type_marker_gene_ids):
            raise ValueError("type-marker genes leak into the molecular evaluation panel")
        if not (
            self.reference_pool_independent
            and self.labels_independent_of_images
            and self.registration_is_one_to_one
        ):
            raise ValueError("morphology-ridge independence declarations are not satisfied")
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
        )
        if not all(value.strip() for value in declared):
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
            "cohort_id",
            "cohort_release",
            "assay",
            "observation_level",
            "target_construction",
            "label_source_sha256",
            "registration_source_sha256",
            "exclusion_policy_sha256",
        ):
            if getattr(self, name) != getattr(locked_test, name):
                raise ValueError("development and locked-test morphology-ridge %s differ" % name)
        for name in (
            "frozen_features",
            "molecular_targets",
            "coordinate_features",
            "stain_features",
            "composition_features",
        ):
            if getattr(self, name).shape[1] != getattr(locked_test, name).shape[1]:
                raise ValueError("development and locked-test %s widths differ" % name)
        if self.stain_feature_names != locked_test.stain_feature_names:
            raise ValueError("development and locked-test stain features differ")
        if self.composition_feature_names != locked_test.composition_feature_names:
            raise ValueError("development and locked-test composition features differ")
        if self.technical_covariate_names != locked_test.technical_covariate_names:
            raise ValueError("development and locked-test technical covariates differ")
        overlap = sorted(set(self.donor_ids) & set(locked_test.donor_ids))
        if overlap:
            raise ValueError("development and locked-test donors overlap: %s" % ", ".join(overlap))
        if self.reference_source_sha256 == locked_test.reference_source_sha256:
            raise ValueError("development and locked-test reference pools must be distinct")
        if self.target_source_sha256 == locked_test.target_source_sha256:
            raise ValueError("development and locked-test molecular targets must be distinct")


__all__ = ["MorphologyRidgeDatasetArtifact"]
