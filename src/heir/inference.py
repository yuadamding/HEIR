"""H&E nucleus inference and the versioned HEIR prediction artifact."""

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Sequence, Union

import numpy as np
import torch

from . import __version__
from .data.arrays import PrototypeSet
from .models.heir import HEIRModel
from .uncertainty.policy import apply_abstention_policy
from .utils import resolve_device


@dataclass(frozen=True)
class PredictionBundle:
    nucleus_ids: np.ndarray
    coordinates_um: np.ndarray
    type_probabilities: np.ndarray
    type_names: np.ndarray
    labels: np.ndarray
    prototype_probabilities: np.ndarray
    prototype_ids: np.ndarray
    latent_mean: np.ndarray
    latent_variance: np.ndarray
    expression_mean: np.ndarray
    expression_lower: np.ndarray
    expression_upper: np.ndarray
    gene_names: np.ndarray
    unknown_probability: np.ndarray
    abstain_score: np.ndarray
    abstain: np.ndarray
    ood_score: np.ndarray
    refinement_round: int
    expression_interval_semantics: str = ""
    expression_interval_available: Optional[np.ndarray] = None
    sample_id: str = ""
    donor_id: str = ""
    slide_id: str = ""
    checkpoint_sha256: str = ""
    prototype_sha256: str = ""
    histology_sha256: str = ""
    latent_space_id: str = ""
    model_version: str = ""
    parent_type_probabilities: Optional[np.ndarray] = None
    parent_type_names: Optional[np.ndarray] = None
    program_scores: Optional[np.ndarray] = None
    program_names: Optional[np.ndarray] = None
    program_sha256: str = ""
    program_training_donors: Optional[np.ndarray] = None
    ood_sha256: str = ""
    ood_training_donors: Optional[np.ndarray] = None
    inference_seed: Optional[int] = None
    latent_samples: Optional[int] = None
    probability_threshold: Optional[float] = None
    artifact_threshold: Optional[float] = None
    expression_space_id: str = ""

    CONTRACT: ClassVar[str] = "heir.prediction_bundle"
    CONTRACT_VERSION: ClassVar[int] = 7
    CONDITIONAL_KNOWN_STATE: ClassVar[str] = "conditional_on_measured_known_state"
    LEGACY_CONDITIONAL_KNOWN_STATE: ClassVar[str] = (
        "legacy_conditional_on_measured_known_state_unsuppressed"
    )
    CORE_FIELDS: ClassVar[frozenset] = frozenset(
        {
            "nucleus_ids",
            "coordinates_um",
            "type_probabilities",
            "type_names",
            "labels",
            "prototype_probabilities",
            "prototype_ids",
            "latent_mean",
            "latent_variance",
            "expression_mean",
            "expression_lower",
            "expression_upper",
            "gene_names",
            "unknown_probability",
            "abstain_score",
            "abstain",
            "ood_score",
            "refinement_round",
        }
    )

    @staticmethod
    def _names(values: np.ndarray, name: str, allow_empty: bool = False) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("%s must be a vector" % name)
        strings = [str(value) for value in array.tolist()]
        if not allow_empty and any(not value.strip() for value in strings):
            raise ValueError("%s cannot contain empty names" % name)
        if len(set(strings)) != len(strings):
            raise ValueError("%s must be unique" % name)
        return array

    @staticmethod
    def _finite(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("%s must be finite and numeric" % name)
        return array

    def validate(
        self,
        require_provenance: bool = False,
        allow_legacy_provenance: bool = False,
        allow_legacy_ood_provenance: bool = False,
        allow_legacy_decision_provenance: bool = False,
        allow_legacy_expression_provenance: bool = False,
    ) -> None:
        count = len(self.nucleus_ids)
        if count == 0:
            raise ValueError("prediction bundle must contain at least one nucleus")
        self._names(self.nucleus_ids, "nucleus_ids")
        row_arrays = {
            "coordinates_um": self.coordinates_um,
            "type_probabilities": self.type_probabilities,
            "labels": self.labels,
            "prototype_probabilities": self.prototype_probabilities,
            "latent_mean": self.latent_mean,
            "latent_variance": self.latent_variance,
            "expression_mean": self.expression_mean,
            "expression_lower": self.expression_lower,
            "expression_upper": self.expression_upper,
            "unknown_probability": self.unknown_probability,
            "abstain_score": self.abstain_score,
            "abstain": self.abstain,
            "ood_score": self.ood_score,
        }
        for name, values in row_arrays.items():
            if np.asarray(values).shape[0] != count:
                raise ValueError("%s does not align to nucleus_ids" % name)
        if self.coordinates_um.shape != (count, 2):
            raise ValueError("coordinates_um must have shape (nuclei, 2)")
        self._finite(self.coordinates_um, "coordinates_um")
        type_names = self._names(self.type_names, "type_names")
        if self.type_probabilities.ndim != 2 or self.type_probabilities.shape[1] != len(type_names):
            raise ValueError("type names do not match type probabilities")
        probabilities = self._finite(self.type_probabilities, "type_probabilities")
        if np.any(probabilities < 0) or np.any(probabilities.sum(axis=1) <= 0):
            raise ValueError("type probabilities must be non-negative with positive row mass")
        labels = np.asarray(self.labels)
        if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be an integer vector aligned to nuclei")
        if np.any(labels < -1) or np.any(labels >= len(type_names)):
            raise ValueError("labels contain an unavailable type index")

        prototype_ids = self._names(self.prototype_ids, "prototype_ids")
        if self.prototype_probabilities.ndim != 2 or self.prototype_probabilities.shape[1] != len(
            prototype_ids
        ):
            raise ValueError("prototype IDs do not match prototype probabilities")
        prototype_probabilities = self._finite(
            self.prototype_probabilities,
            "prototype_probabilities",
        )
        if np.any(prototype_probabilities < 0):
            raise ValueError("prototype probabilities must be non-negative")

        latent_mean = self._finite(self.latent_mean, "latent_mean")
        latent_variance = self._finite(self.latent_variance, "latent_variance")
        if latent_mean.ndim != 2 or latent_variance.shape != latent_mean.shape:
            raise ValueError("latent mean/variance must have matching nuclei-by-latent shapes")
        if np.any(latent_variance < 0):
            raise ValueError("latent variance must be non-negative")

        gene_names = self._names(self.gene_names, "gene_names")
        expression_mean = np.asarray(self.expression_mean)
        expression_lower = np.asarray(self.expression_lower)
        expression_upper = np.asarray(self.expression_upper)
        for name, values in (
            ("expression_mean", expression_mean),
            ("expression_lower", expression_lower),
            ("expression_upper", expression_upper),
        ):
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError("%s must be numeric" % name)
        if expression_mean.ndim != 2 or expression_mean.shape[1] != len(gene_names):
            raise ValueError("gene names do not match expression outputs")
        if expression_lower.shape != expression_mean.shape or expression_upper.shape != (
            expression_mean.shape
        ):
            raise ValueError("expression means and intervals must have identical shapes")
        if self.expression_interval_semantics not in {
            "",
            self.CONDITIONAL_KNOWN_STATE,
            self.LEGACY_CONDITIONAL_KNOWN_STATE,
        }:
            raise ValueError("expression interval semantics are unsupported")
        if self.expression_interval_available is None:
            available = np.ones(count, dtype=bool)
            for name, values in (
                ("expression_mean", expression_mean),
                ("expression_lower", expression_lower),
                ("expression_upper", expression_upper),
            ):
                if not np.isfinite(values).all():
                    raise ValueError("%s must be finite" % name)
        else:
            available = np.asarray(self.expression_interval_available)
            if available.shape != (count,) or available.dtype != bool:
                raise ValueError("expression_interval_available must be a boolean nuclei vector")
            if self.expression_interval_semantics != self.CONDITIONAL_KNOWN_STATE:
                raise ValueError("expression availability requires conditional interval semantics")
            unavailable = ~available
            if not np.isfinite(expression_mean).all():
                raise ValueError("expression_mean must remain finite for internal aggregation")
            for name, values in (
                ("expression_lower", expression_lower),
                ("expression_upper", expression_upper),
            ):
                if not np.isfinite(values[available]).all():
                    raise ValueError("available %s values must be finite" % name)
                if unavailable.any() and not np.isnan(values[unavailable]).all():
                    raise ValueError("unavailable %s values must be suppressed" % name)
            if np.any(available & np.asarray(self.abstain, dtype=bool)):
                raise ValueError("abstained cells cannot expose conditional expression")
        if np.any(expression_lower[available] > expression_upper[available]):
            raise ValueError("expression interval bounds are inverted")

        for name in ("unknown_probability", "abstain_score"):
            values = self._finite(getattr(self, name), name)
            if values.shape != (count,) or np.any(values < 0) or np.any(values > 1):
                raise ValueError("%s must be a nuclei vector in [0, 1]" % name)
        if np.asarray(self.abstain).shape != (count,) or np.asarray(self.abstain).dtype != bool:
            raise ValueError("abstain must be a boolean nuclei vector")
        ood = self._finite(self.ood_score, "ood_score")
        if ood.shape != (count,):
            raise ValueError("ood_score must be a nuclei vector")
        if self.ood_training_donors is not None:
            self._names(self.ood_training_donors, "ood_training_donors")
            if len(self.ood_training_donors) == 0:
                raise ValueError("ood_training_donors cannot be empty")
        has_ood_provenance = bool(self.ood_sha256) or self.ood_training_donors is not None
        if has_ood_provenance and (not self.ood_sha256 or self.ood_training_donors is None):
            raise ValueError("OOD provenance requires a detector hash and training donors")
        if (
            require_provenance
            and not allow_legacy_ood_provenance
            and np.any(ood != 0)
            and not has_ood_provenance
        ):
            raise ValueError("non-zero OOD scores require detector provenance")
        if int(self.refinement_round) != self.refinement_round or self.refinement_round < 0:
            raise ValueError("refinement_round must be a non-negative integer")

        missing_decisions = []
        if self.inference_seed is None:
            missing_decisions.append("inference_seed")
        else:
            if isinstance(self.inference_seed, (bool, np.bool_)) or not isinstance(
                self.inference_seed, (int, np.integer)
            ):
                raise TypeError("inference_seed must be an integer")
            if self.inference_seed < 0:
                raise ValueError("inference_seed must be non-negative")
        if self.latent_samples is None:
            missing_decisions.append("latent_samples")
        else:
            if isinstance(self.latent_samples, (bool, np.bool_)) or not isinstance(
                self.latent_samples, (int, np.integer)
            ):
                raise TypeError("latent_samples must be an integer")
            if self.latent_samples <= 0:
                raise ValueError("latent_samples must be positive")
        for name in ("probability_threshold", "artifact_threshold"):
            value = getattr(self, name)
            if value is None:
                missing_decisions.append(name)
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError("%s must be numeric" % name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("%s must lie in [0, 1]" % name)
        if require_provenance and not allow_legacy_decision_provenance and missing_decisions:
            raise ValueError(
                "inference decision provenance is missing: %s" % ", ".join(missing_decisions)
            )

        if (self.parent_type_probabilities is None) != (self.parent_type_names is None):
            raise ValueError("parent type probabilities and names must be supplied together")
        if self.parent_type_probabilities is not None:
            assert self.parent_type_names is not None
            parent_names = self._names(self.parent_type_names, "parent_type_names")
            parent_probabilities = self._finite(
                self.parent_type_probabilities,
                "parent_type_probabilities",
            )
            if parent_probabilities.shape != (count, len(parent_names)):
                raise ValueError("parent type probabilities have the wrong shape")
            if np.any(parent_probabilities < 0) or np.any(parent_probabilities.sum(axis=1) <= 0):
                raise ValueError(
                    "parent type probabilities must be non-negative with positive row mass"
                )

        if (self.program_scores is None) != (self.program_names is None):
            raise ValueError("program scores and names must be supplied together")
        if self.program_scores is not None:
            assert self.program_names is not None
            program_names = self._names(self.program_names, "program_names")
            scores = np.asarray(self.program_scores)
            if not np.issubdtype(scores.dtype, np.number):
                raise ValueError("program_scores must be numeric")
            if scores.shape != (count, len(program_names)):
                raise ValueError("program scores have the wrong shape")
            if not np.isfinite(scores).all():
                raise ValueError("program_scores must be finite")
            if self.program_training_donors is not None:
                self._names(self.program_training_donors, "program_training_donors")
            if require_provenance and not allow_legacy_provenance:
                if not self.program_sha256:
                    raise ValueError("program_scores require program_sha256 provenance")
                if self.program_training_donors is None or len(self.program_training_donors) == 0:
                    raise ValueError("program_scores require program training donors")
        elif self.program_sha256 or self.program_training_donors is not None:
            raise ValueError("program provenance requires program scores")

        provenance = {
            "sample_id": self.sample_id,
            "donor_id": self.donor_id,
            "slide_id": self.slide_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prototype_sha256": self.prototype_sha256,
            "histology_sha256": self.histology_sha256,
            "latent_space_id": self.latent_space_id,
            "expression_space_id": self.expression_space_id,
            "model_version": self.model_version,
            "ood_sha256": self.ood_sha256,
            "program_sha256": self.program_sha256,
        }
        if any(not isinstance(value, str) for value in provenance.values()):
            raise TypeError("prediction provenance fields must be strings")
        if require_provenance:
            required_names = {
                "sample_id",
                "donor_id",
                "slide_id",
                "checkpoint_sha256",
                "prototype_sha256",
                "latent_space_id",
                "model_version",
            }
            if not allow_legacy_provenance:
                required_names.add("histology_sha256")
            if not allow_legacy_expression_provenance:
                required_names.add("expression_space_id")
            missing = sorted(name for name in required_names if not provenance[name].strip())
            if missing:
                raise ValueError("prediction provenance is missing: %s" % ", ".join(missing))
        for name in (
            "checkpoint_sha256",
            "prototype_sha256",
            "histology_sha256",
            "ood_sha256",
            "program_sha256",
        ):
            value = provenance[name]
            if value and re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("%s must be a lowercase SHA-256 digest" % name)

    def to_npz(self, path: Union[str, Path]) -> None:
        """Atomically write the strict v7 prediction artifact."""

        self.validate(require_provenance=True)
        assert self.inference_seed is not None
        assert self.latent_samples is not None
        assert self.probability_threshold is not None
        assert self.artifact_threshold is not None
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        expression_available = (
            ~np.asarray(self.abstain, dtype=bool)
            if self.expression_interval_available is None
            else np.asarray(self.expression_interval_available, dtype=bool)
        )
        expression_mean = np.asarray(self.expression_mean, dtype=np.float32).copy()
        expression_lower = np.asarray(self.expression_lower, dtype=np.float32).copy()
        expression_upper = np.asarray(self.expression_upper, dtype=np.float32).copy()
        expression_lower[~expression_available] = np.nan
        expression_upper[~expression_available] = np.nan
        payload: Dict[str, np.ndarray] = dict(
            __contract__=np.asarray(self.CONTRACT, dtype=np.dtype("U")),
            __version__=np.asarray(self.CONTRACT_VERSION, dtype=np.int64),
            nucleus_ids=self.nucleus_ids.astype(np.str_),
            coordinates_um=self.coordinates_um.astype(np.float32),
            type_probabilities=self.type_probabilities.astype(np.float32),
            type_names=self.type_names.astype(np.str_),
            labels=self.labels.astype(np.int64),
            prototype_probabilities=self.prototype_probabilities.astype(np.float32),
            prototype_ids=self.prototype_ids.astype(np.str_),
            latent_mean=self.latent_mean.astype(np.float32),
            latent_variance=self.latent_variance.astype(np.float32),
            expression_mean=expression_mean,
            expression_lower=expression_lower,
            expression_upper=expression_upper,
            expression_interval_semantics=np.asarray(
                self.CONDITIONAL_KNOWN_STATE,
                dtype=np.dtype("U"),
            ),
            expression_interval_available=expression_available,
            gene_names=self.gene_names.astype(np.str_),
            unknown_probability=self.unknown_probability.astype(np.float32),
            abstain_score=self.abstain_score.astype(np.float32),
            abstain=self.abstain.astype(bool),
            ood_score=self.ood_score.astype(np.float32),
            refinement_round=np.asarray(self.refinement_round, dtype=np.int64),
            sample_id=np.asarray(self.sample_id, dtype=np.dtype("U")),
            donor_id=np.asarray(self.donor_id, dtype=np.dtype("U")),
            slide_id=np.asarray(self.slide_id, dtype=np.dtype("U")),
            checkpoint_sha256=np.asarray(self.checkpoint_sha256, dtype=np.dtype("U")),
            prototype_sha256=np.asarray(self.prototype_sha256, dtype=np.dtype("U")),
            histology_sha256=np.asarray(self.histology_sha256, dtype=np.dtype("U")),
            latent_space_id=np.asarray(self.latent_space_id, dtype=np.dtype("U")),
            expression_space_id=np.asarray(self.expression_space_id, dtype=np.dtype("U")),
            model_version=np.asarray(self.model_version, dtype=np.dtype("U")),
            ood_sha256=np.asarray(self.ood_sha256, dtype=np.dtype("U")),
            ood_training_donors=np.asarray(
                () if self.ood_training_donors is None else self.ood_training_donors,
                dtype=np.dtype("U"),
            ),
            inference_seed=np.asarray(self.inference_seed, dtype=np.int64),
            latent_samples=np.asarray(self.latent_samples, dtype=np.int64),
            probability_threshold=np.asarray(self.probability_threshold, dtype=np.float64),
            artifact_threshold=np.asarray(self.artifact_threshold, dtype=np.float64),
            program_sha256=np.asarray(self.program_sha256, dtype=np.dtype("U")),
            program_training_donors=np.asarray(
                () if self.program_training_donors is None else self.program_training_donors,
                dtype=np.dtype("U"),
            ),
            __present__parent_types=np.asarray(
                self.parent_type_probabilities is not None,
                dtype=bool,
            ),
            __present__programs=np.asarray(self.program_scores is not None, dtype=bool),
        )
        if self.parent_type_probabilities is not None:
            assert self.parent_type_names is not None
            payload["parent_type_probabilities"] = self.parent_type_probabilities.astype(np.float32)
            payload["parent_type_names"] = self.parent_type_names.astype(np.str_)
        if self.program_scores is not None:
            assert self.program_names is not None
            payload["program_scores"] = self.program_scores.astype(np.float32)
            payload["program_names"] = self.program_names.astype(np.str_)

        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=".npz.tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "PredictionBundle":
        """Load v2-v7 or migrate the original unversioned schema in memory.

        Legacy artifacts receive empty provenance and optional outputs. Saving
        that object again therefore requires callers to populate provenance
        explicitly; legacy data are never silently presented as audited v7.
        """

        with np.load(path, allow_pickle=False) as values:
            missing_core = sorted(cls.CORE_FIELDS - set(values.files))
            if missing_core:
                raise ValueError(
                    "prediction artifact is missing core arrays: %s" % ", ".join(missing_core)
                )
            versioned = "__contract__" in values or "__version__" in values
            version = 0
            if versioned:
                if "__contract__" not in values or "__version__" not in values:
                    raise ValueError("prediction artifact has partial contract metadata")
                contract = str(np.asarray(values["__contract__"]).item())
                version = int(np.asarray(values["__version__"]).item())
                if contract != cls.CONTRACT:
                    raise ValueError("artifact is not a HEIR PredictionBundle")
                if version not in {2, 3, 4, 5, 6, cls.CONTRACT_VERSION}:
                    raise ValueError("unsupported PredictionBundle version %d" % version)
                required_versioned = {
                    "sample_id",
                    "donor_id",
                    "slide_id",
                    "checkpoint_sha256",
                    "prototype_sha256",
                    "latent_space_id",
                    "model_version",
                    "__present__parent_types",
                    "__present__programs",
                }
                if version >= 3:
                    required_versioned.update(
                        {
                            "histology_sha256",
                            "program_sha256",
                            "program_training_donors",
                        }
                    )
                if version >= 4:
                    required_versioned.update({"ood_sha256", "ood_training_donors"})
                if version >= 5:
                    required_versioned.update(
                        {
                            "inference_seed",
                            "latent_samples",
                            "probability_threshold",
                            "artifact_threshold",
                        }
                    )
                if version >= 6:
                    required_versioned.add("expression_space_id")
                if version >= 7:
                    required_versioned.update(
                        {
                            "expression_interval_semantics",
                            "expression_interval_available",
                        }
                    )
                missing = sorted(required_versioned - set(values.files))
                if missing:
                    raise ValueError(
                        "versioned prediction artifact is missing: %s" % ", ".join(missing)
                    )
                has_parent = bool(np.asarray(values["__present__parent_types"]).item())
                has_programs = bool(np.asarray(values["__present__programs"]).item())
                if has_parent and not {
                    "parent_type_probabilities",
                    "parent_type_names",
                }.issubset(values.files):
                    raise ValueError("prediction marks missing parent arrays present")
                if has_programs and not {"program_scores", "program_names"}.issubset(values.files):
                    raise ValueError("prediction marks missing program arrays present")
                if not has_parent and {
                    "parent_type_probabilities",
                    "parent_type_names",
                } & set(values.files):
                    raise ValueError("prediction has unmarked parent arrays")
                if not has_programs and {"program_scores", "program_names"} & set(values.files):
                    raise ValueError("prediction has unmarked program arrays")
            else:
                has_parent = False
                has_programs = False
            abstain = np.array(values["abstain"], copy=True)
            expression_mean = np.array(values["expression_mean"], copy=True)
            expression_lower = np.array(values["expression_lower"], copy=True)
            expression_upper = np.array(values["expression_upper"], copy=True)
            program_scores = np.array(values["program_scores"], copy=True) if has_programs else None
            if versioned and version >= 7:
                expression_interval_semantics = str(
                    np.asarray(values["expression_interval_semantics"]).item()
                )
                expression_available = np.array(
                    values["expression_interval_available"],
                    copy=True,
                )
            else:
                # v2-v6 used the conditional-known-state sampler without an
                # availability mask. Preserve those immutable arrays exactly
                # for historical revalidation while labeling their weaker,
                # unsuppressed semantics explicitly in memory. Re-saving them
                # as v7 applies the current fail-closed mask in ``to_npz``.
                expression_interval_semantics = cls.LEGACY_CONDITIONAL_KNOWN_STATE
                expression_available = None
            result = cls(
                nucleus_ids=np.array(values["nucleus_ids"], copy=True),
                coordinates_um=np.array(values["coordinates_um"], copy=True),
                type_probabilities=np.array(values["type_probabilities"], copy=True),
                type_names=np.array(values["type_names"], copy=True),
                labels=np.array(values["labels"], copy=True),
                prototype_probabilities=np.array(values["prototype_probabilities"], copy=True),
                prototype_ids=np.array(values["prototype_ids"], copy=True),
                latent_mean=np.array(values["latent_mean"], copy=True),
                latent_variance=np.array(values["latent_variance"], copy=True),
                expression_mean=expression_mean,
                expression_lower=expression_lower,
                expression_upper=expression_upper,
                gene_names=np.array(values["gene_names"], copy=True),
                unknown_probability=np.array(values["unknown_probability"], copy=True),
                abstain_score=np.array(values["abstain_score"], copy=True),
                abstain=abstain,
                ood_score=np.array(values["ood_score"], copy=True),
                refinement_round=int(np.asarray(values["refinement_round"]).item()),
                expression_interval_semantics=expression_interval_semantics,
                expression_interval_available=expression_available,
                sample_id=(str(np.asarray(values["sample_id"]).item()) if versioned else ""),
                donor_id=(str(np.asarray(values["donor_id"]).item()) if versioned else ""),
                slide_id=(str(np.asarray(values["slide_id"]).item()) if versioned else ""),
                checkpoint_sha256=(
                    str(np.asarray(values["checkpoint_sha256"]).item()) if versioned else ""
                ),
                prototype_sha256=(
                    str(np.asarray(values["prototype_sha256"]).item()) if versioned else ""
                ),
                histology_sha256=(
                    str(np.asarray(values["histology_sha256"]).item())
                    if versioned and version >= 3
                    else ""
                ),
                latent_space_id=(
                    str(np.asarray(values["latent_space_id"]).item()) if versioned else ""
                ),
                expression_space_id=(
                    str(np.asarray(values["expression_space_id"]).item())
                    if versioned and version >= 6
                    else ""
                ),
                model_version=(
                    str(np.asarray(values["model_version"]).item()) if versioned else ""
                ),
                ood_sha256=(
                    str(np.asarray(values["ood_sha256"]).item())
                    if versioned and version >= 4
                    else ""
                ),
                ood_training_donors=(
                    np.array(values["ood_training_donors"], copy=True)
                    if versioned and version >= 4 and np.asarray(values["ood_training_donors"]).size
                    else None
                ),
                inference_seed=(
                    np.asarray(values["inference_seed"]).item()
                    if versioned and version >= 5
                    else None
                ),
                latent_samples=(
                    np.asarray(values["latent_samples"]).item()
                    if versioned and version >= 5
                    else None
                ),
                probability_threshold=(
                    np.asarray(values["probability_threshold"]).item()
                    if versioned and version >= 5
                    else None
                ),
                artifact_threshold=(
                    np.asarray(values["artifact_threshold"]).item()
                    if versioned and version >= 5
                    else None
                ),
                parent_type_probabilities=(
                    np.array(values["parent_type_probabilities"], copy=True) if has_parent else None
                ),
                parent_type_names=(
                    np.array(values["parent_type_names"], copy=True) if has_parent else None
                ),
                program_scores=program_scores,
                program_names=(
                    np.array(values["program_names"], copy=True) if has_programs else None
                ),
                program_sha256=(
                    str(np.asarray(values["program_sha256"]).item())
                    if versioned and version >= 3
                    else ""
                ),
                program_training_donors=(
                    np.array(values["program_training_donors"], copy=True)
                    if versioned
                    and version >= 3
                    and np.asarray(values["program_training_donors"]).size
                    else None
                ),
            )
        result.validate(
            require_provenance=versioned,
            allow_legacy_provenance=versioned and version == 2,
            allow_legacy_ood_provenance=versioned and version in {2, 3},
            allow_legacy_decision_provenance=versioned and version in {2, 3, 4},
            allow_legacy_expression_provenance=versioned and version in {2, 3, 4, 5},
        )
        return result


@torch.no_grad()
def predict_cells(
    model: HEIRModel,
    features: np.ndarray,
    coordinates_um: np.ndarray,
    nucleus_ids: Sequence[object],
    prototypes: PrototypeSet,
    type_names: Sequence[object],
    gene_names: Sequence[object],
    edge_index: Optional[np.ndarray] = None,
    edge_weight: Optional[np.ndarray] = None,
    latent_samples: int = 20,
    probability_threshold: float = 0.60,
    segmentation_confidence: Optional[np.ndarray] = None,
    ood_score: Optional[np.ndarray] = None,
    ood_threshold: Optional[float] = None,
    refinement_round: int = 0,
    device: str = "auto",
    sample_id: Optional[str] = None,
    donor_id: str = "",
    slide_id: str = "",
    checkpoint_sha256: str = "",
    prototype_sha256: str = "",
    histology_sha256: str = "",
    latent_space_id: Optional[str] = None,
    model_version: str = __version__,
    parent_type_names: Optional[Sequence[object]] = None,
    program_matrix: Optional[np.ndarray] = None,
    program_names: Optional[Sequence[object]] = None,
    program_sha256: str = "",
    program_training_donors: Optional[Sequence[object]] = None,
    ood_sha256: str = "",
    ood_training_donors: Optional[Sequence[object]] = None,
    inference_seed: Optional[int] = None,
    artifact_threshold: Optional[float] = None,
    expression_space_id: str = "",
    mixed_precision: Optional[bool] = None,
    mc_chunk_size: Optional[int] = None,
    use_model_abstain: bool = False,
) -> PredictionBundle:
    """Predict one graph bag and derive latent/gene credible intervals.

    ``program_matrix`` follows the canonical genes-by-programs orientation.
    Supplying hierarchy/program names exports those optional cell-level
    outputs. Provenance kwargs become mandatory when the returned bundle is
    persisted with :meth:`PredictionBundle.to_npz`. ``inference_seed`` is
    recorded only; callers remain responsible for configuring their RNGs
    before invoking this function. Molecular intervals are conditional on a
    measured known state and are suppressed for every abstained cell; the
    finite expression mean is retained only for internal aggregate scoring.
    """

    morphology = np.array(features, dtype=np.float32, copy=True)
    coordinates = np.array(coordinates_um, dtype=np.float32, copy=True)
    if morphology.ndim != 2 or morphology.shape[1] != model.config.morphology_dim:
        raise ValueError("features have the wrong HEIR morphology width")
    if coordinates.shape != (len(morphology), 2) or len(nucleus_ids) != len(morphology):
        raise ValueError("coordinates/nucleus_ids do not align to features")
    if isinstance(latent_samples, (bool, np.bool_)) or not isinstance(
        latent_samples, (int, np.integer)
    ):
        raise TypeError("latent_samples must be an integer")
    if latent_samples <= 0:
        raise ValueError("latent_samples must be positive")
    if mc_chunk_size is not None and mc_chunk_size <= 0:
        raise ValueError("mc_chunk_size must be positive when supplied")
    if not isinstance(use_model_abstain, bool):
        raise TypeError("use_model_abstain must be boolean")
    if inference_seed is not None:
        if isinstance(inference_seed, (bool, np.bool_)) or not isinstance(
            inference_seed, (int, np.integer)
        ):
            raise TypeError("inference_seed must be an integer")
        if inference_seed < 0:
            raise ValueError("inference_seed must be non-negative")
    for name, value in (("probability_threshold", probability_threshold),):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("%s must be numeric" % name)
    if artifact_threshold is not None and (
        isinstance(artifact_threshold, (bool, np.bool_))
        or not isinstance(artifact_threshold, (int, float, np.integer, np.floating))
    ):
        raise TypeError("artifact_threshold must be numeric")
    if not np.isfinite(probability_threshold) or not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must lie in [0, 1]")
    if artifact_threshold is not None and (
        not np.isfinite(artifact_threshold) or not 0.0 <= artifact_threshold <= 1.0
    ):
        raise ValueError("artifact_threshold must lie in [0, 1]")
    if prototypes.means.shape[1] != model.config.latent_dim:
        raise ValueError("prototype latent width differs from HEIR")
    unique_samples = np.unique(prototypes.sample_ids)
    if len(unique_samples) != 1:
        raise ValueError("predict_cells requires one sample-specific prototype bank")
    prototype_sample = str(unique_samples[0])
    resolved_sample_id = prototype_sample if sample_id is None else str(sample_id)
    if resolved_sample_id != prototype_sample:
        raise ValueError("prediction sample_id differs from the prototype bank")
    resolved_latent_space_id = (
        prototypes.latent_space_id if latent_space_id is None else str(latent_space_id)
    )
    if (
        latent_space_id is not None
        and prototypes.latent_space_id
        and resolved_latent_space_id != prototypes.latent_space_id
    ):
        raise ValueError("prediction latent_space_id differs from the prototype bank")
    resolved_type_names = [str(value) for value in type_names]
    resolved_gene_names = [str(value) for value in gene_names]
    if len(resolved_type_names) != model.config.num_cell_types:
        raise ValueError("type names do not match the model cell-type width")
    if len(set(resolved_type_names)) != len(resolved_type_names) or any(
        not value.strip() for value in resolved_type_names
    ):
        raise ValueError("type names must be unique and non-empty")
    if len(resolved_gene_names) != model.config.expression_dim:
        raise ValueError("gene names do not match the model expression width")
    if len(set(resolved_gene_names)) != len(resolved_gene_names) or any(
        not value.strip() for value in resolved_gene_names
    ):
        raise ValueError("gene names must be unique and non-empty")
    type_lookup = {name: index for index, name in enumerate(resolved_type_names)}
    missing_types = sorted(set(prototypes.cell_type_labels.tolist()) - set(type_lookup))
    if missing_types:
        raise ValueError(
            "prototype types are absent from the model ontology: %s" % ", ".join(missing_types)
        )
    target = resolve_device(device)
    model = model.to(target).eval()
    use_mixed_precision = (
        target.type == "cuda" if mixed_precision is None else bool(mixed_precision)
    )
    feature_tensor = torch.from_numpy(morphology).to(target)
    edge_tensor = (
        torch.empty((2, 0), dtype=torch.long, device=target)
        if edge_index is None
        else torch.from_numpy(np.array(edge_index, dtype=np.int64, copy=True)).to(target)
    )
    weight_tensor = (
        None
        if edge_weight is None
        else torch.from_numpy(np.array(edge_weight, dtype=np.float32, copy=True)).to(target)
    )
    # Artifact contracts are intentionally read-only. Copy before exposing the
    # storage to PyTorch, which otherwise warns about undefined write behavior.
    prototype_means = torch.from_numpy(np.array(prototypes.means, copy=True)).to(target)
    prototype_types = torch.tensor(
        [type_lookup[str(value)] for value in prototypes.cell_type_labels],
        dtype=torch.long,
        device=target,
    )
    prototype_weights = torch.from_numpy(
        np.array(prototypes.weights, dtype=np.float32, copy=True)
    ).to(target)
    prototype_variances = torch.from_numpy(
        np.array(prototypes.variances, dtype=np.float32, copy=True)
    ).to(target)

    with torch.autocast(
        device_type=target.type,
        dtype=torch.float16 if target.type == "cuda" else torch.bfloat16,
        enabled=use_mixed_precision,
    ):
        deterministic = model(
            feature_tensor,
            edge_tensor,
            weight_tensor,
            prototype_means=prototype_means,
            prototype_variances=prototype_variances,
            prototype_types=prototype_types,
            prototype_weights=prototype_weights,
            sample_latent=False,
        )
    latent_draws = []
    expression_draws = []
    # Graph context is deterministic at inference. Reuse it while sampling the
    # detached molecular-state mixture, prototype covariance, and morphology
    # residual. This captures between-state as well as within-state uncertainty.
    if mc_chunk_size is None:
        elements_per_draw = max(
            1,
            len(morphology) * (model.config.latent_dim + model.config.expression_dim),
        )
        resolved_chunk_size = max(1, min(latent_samples, 8_000_000 // elements_per_draw))
    else:
        resolved_chunk_size = min(latent_samples, mc_chunk_size)
    latent_mu = deterministic.latent_mu
    conditional = deterministic.conditional_prototype_probabilities.float()
    known_rows = conditional.sum(dim=-1) > torch.finfo(conditional.dtype).eps
    safe_conditional = conditional
    if conditional.shape[1]:
        safe_conditional = conditional.clone()
        safe_conditional[~known_rows, 0] = 1.0
    for start in range(0, latent_samples, resolved_chunk_size):
        draws = min(resolved_chunk_size, latent_samples - start)
        residual_draws = model.sample_residuals(deterministic, draws)
        if conditional.shape[1]:
            sampled_indices = torch.multinomial(
                safe_conditional,
                num_samples=draws,
                replacement=True,
            ).transpose(0, 1)
            sampled_means = prototype_means.index_select(0, sampled_indices.reshape(-1)).reshape(
                draws,
                latent_mu.shape[0],
                latent_mu.shape[1],
            )
            sampled_variances = prototype_variances.index_select(
                0, sampled_indices.reshape(-1)
            ).reshape_as(sampled_means)
            prototype_noise = torch.randn_like(sampled_means)
            prototype_draws = (
                sampled_means
                + prototype_noise
                * sampled_variances.clamp_min(model.config.prototype_variance_floor).sqrt()
            )
            prototype_draws = torch.where(
                known_rows.reshape(1, -1, 1),
                prototype_draws,
                deterministic.prototype_latent.unsqueeze(0),
            )
        else:
            prototype_draws = deterministic.prototype_latent.unsqueeze(0).expand(draws, -1, -1)
        sampled_latent = prototype_draws + residual_draws
        with torch.autocast(
            device_type=target.type,
            dtype=torch.float16 if target.type == "cuda" else torch.bfloat16,
            enabled=use_mixed_precision,
        ):
            sampled_expression = model.expression_decoder(
                sampled_latent.reshape(-1, sampled_latent.shape[-1])
            ).reshape(draws, latent_mu.shape[0], -1)
        latent_draws.append(sampled_latent.float().cpu().numpy())
        expression_draws.append(sampled_expression.float().cpu().numpy())
    latent_stack = np.concatenate(latent_draws, axis=0)
    expression_stack = np.concatenate(expression_draws, axis=0)
    probabilities = deterministic.type_probabilities.float().cpu().numpy()
    ood_values = (
        np.zeros(len(morphology), dtype=np.float32)
        if ood_score is None
        else np.array(ood_score, dtype=np.float32, copy=True)
    )
    if ood_values.shape != (len(morphology),):
        raise ValueError("ood_score must align to cells")
    if ood_score is not None:
        if not ood_sha256 or ood_training_donors is None or len(ood_training_donors) == 0:
            raise ValueError("ood_score requires detector hash and training donors")
    elif ood_sha256 or ood_training_donors is not None:
        raise ValueError("OOD provenance was supplied without ood_score")
    ood_mask = None if ood_threshold is None else ood_values > ood_threshold
    decision = apply_abstention_policy(
        probabilities,
        probability_threshold=probability_threshold,
        ood_mask=ood_mask,
        segmentation_confidence=segmentation_confidence,
        unknown_probability=deterministic.unknown_probability.float().cpu().numpy(),
        unknown_threshold=max(0.0, 1.0 - probability_threshold),
        model_abstain=(deterministic.abstain.cpu().numpy() if use_model_abstain else None),
    )
    parent_probabilities = None
    resolved_parent_names = None
    if deterministic.parent_type_probabilities is not None:
        if parent_type_names is None:
            raise ValueError("hierarchical HEIR inference requires parent_type_names")
        resolved_parent_names = np.asarray([str(value) for value in parent_type_names])
        parent_probabilities = deterministic.parent_type_probabilities.float().cpu().numpy()
        if parent_probabilities.shape[1] != len(resolved_parent_names):
            raise ValueError("parent type names do not match the model hierarchy")
    elif parent_type_names is not None:
        raise ValueError("parent_type_names were supplied to a non-hierarchical model")

    resolved_program_names = None
    program_scores = None
    expression_available = ~decision.abstain
    expression_mean = expression_stack.mean(axis=0)
    expression_lower = np.quantile(expression_stack, 0.05, axis=0).astype(
        np.float32,
        copy=False,
    )
    expression_upper = np.quantile(expression_stack, 0.95, axis=0).astype(
        np.float32,
        copy=False,
    )
    expression_lower[~expression_available] = np.nan
    expression_upper[~expression_available] = np.nan
    if (program_matrix is None) != (program_names is None):
        raise ValueError("program_matrix and program_names must be supplied together")
    if program_matrix is not None:
        assert program_names is not None
        matrix = np.asarray(program_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(resolved_gene_names):
            raise ValueError("program_matrix must have shape (genes, programs)")
        if not np.isfinite(matrix).all():
            raise ValueError("program_matrix must be finite")
        resolved_program_names = np.asarray([str(value) for value in program_names])
        if matrix.shape[1] != len(resolved_program_names):
            raise ValueError("program names do not match program_matrix")
        program_scores = expression_mean.dot(matrix)
    result = PredictionBundle(
        nucleus_ids=np.asarray([str(value) for value in nucleus_ids]),
        coordinates_um=coordinates,
        type_probabilities=probabilities,
        type_names=np.asarray(resolved_type_names),
        labels=decision.label_indices,
        prototype_probabilities=deterministic.prototype_probabilities.float().cpu().numpy(),
        prototype_ids=prototypes.prototype_ids,
        latent_mean=latent_stack.mean(axis=0),
        latent_variance=latent_stack.var(axis=0),
        expression_mean=expression_mean,
        expression_lower=expression_lower,
        expression_upper=expression_upper,
        gene_names=np.asarray(resolved_gene_names),
        unknown_probability=deterministic.unknown_probability.float().cpu().numpy(),
        abstain_score=deterministic.abstain_score.float().cpu().numpy(),
        abstain=decision.abstain,
        ood_score=ood_values,
        refinement_round=refinement_round,
        expression_interval_semantics=PredictionBundle.CONDITIONAL_KNOWN_STATE,
        expression_interval_available=expression_available,
        sample_id=resolved_sample_id,
        donor_id=str(donor_id),
        slide_id=str(slide_id),
        checkpoint_sha256=str(checkpoint_sha256),
        prototype_sha256=str(prototype_sha256),
        histology_sha256=str(histology_sha256),
        latent_space_id=resolved_latent_space_id,
        expression_space_id=expression_space_id,
        model_version=str(model_version),
        ood_sha256=str(ood_sha256),
        ood_training_donors=(
            None
            if ood_training_donors is None
            else np.asarray([str(value) for value in ood_training_donors])
        ),
        inference_seed=None if inference_seed is None else int(inference_seed),
        latent_samples=int(latent_samples),
        probability_threshold=float(probability_threshold),
        artifact_threshold=(None if artifact_threshold is None else float(artifact_threshold)),
        parent_type_probabilities=parent_probabilities,
        parent_type_names=resolved_parent_names,
        program_scores=program_scores,
        program_names=resolved_program_names,
        program_sha256=str(program_sha256),
        program_training_donors=(
            None
            if program_training_donors is None
            else np.asarray([str(value) for value in program_training_donors])
        ),
    )
    result.validate()
    return result
