"""Tensor and NPZ contract for one spatially coherent HEIR graph bag.

The serialized form is deliberately framework-light: tensors are written as
plain NumPy arrays, strings are Unicode arrays, and loading never enables
pickle.  This lets image preparation and model training run in separate
environments without weakening the provenance boundary between them.
"""

import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .stages import StageInputs, TrainingStage


@dataclass
class HEIRTrainingBatch:
    morphology: Tensor
    edge_index: Tensor
    edge_weight: Optional[Tensor]
    prototype_means: Tensor
    prototype_variances: Tensor
    prototype_types: Tensor
    prototype_weights: Tensor
    target_composition: Tensor
    target_pseudobulk: Tensor
    prototype_mask: Optional[Tensor] = None
    cell_weights: Optional[Tensor] = None
    molecular_responsibilities: Optional[Tensor] = None
    anchor_labels: Optional[Tensor] = None
    anchor_weights: Optional[Tensor] = None
    parent_anchor_labels: Optional[Tensor] = None
    parent_anchor_weights: Optional[Tensor] = None
    marker_centroids: Optional[Tensor] = None
    marker_mask: Optional[Tensor] = None
    program_matrix: Optional[Tensor] = None
    target_program_scores: Optional[Tensor] = None
    unknown_targets: Optional[Tensor] = None
    domain_labels: Optional[Tensor] = None
    segmentation_confidence: Optional[Tensor] = None
    ood_mask: Optional[Tensor] = None
    spot_assignment: Optional[Tensor] = None
    target_spatial_expression: Optional[Tensor] = None
    scgpt_type_prototypes: Optional[Tensor] = None
    scgpt_type_variances: Optional[Tensor] = None
    sample_id: str = "sample"
    bag_id: str = "bag0"
    donor_id: str = ""
    block_id: str = ""
    analysis_role: str = "train"
    latent_space_id: str = "unspecified"
    feature_space_id: str = "unspecified"
    expression_space_id: str = "unspecified"
    scgpt_space_id: str = ""
    weak_target_scope_id: str = "unspecified"
    weak_target_granularity: str = "legacy_unspecified"
    nucleus_ids: Tuple[str, ...] = ()
    type_names: Tuple[str, ...] = ()
    gene_names: Tuple[str, ...] = ()
    prototype_ids: Tuple[str, ...] = ()
    spot_ids: Tuple[str, ...] = ()
    source_artifacts: Tuple[str, ...] = ()
    source_sha256: Tuple[str, ...] = ()
    source_roles: Tuple[str, ...] = ()
    molecular_training_donors: Tuple[str, ...] = ()

    CONTRACT: ClassVar[str] = "heir.training_batch"
    CONTRACT_VERSION: ClassVar[int] = 6

    _OPTIONAL_TENSORS: ClassVar[Tuple[str, ...]] = (
        "edge_weight",
        "prototype_mask",
        "cell_weights",
        "molecular_responsibilities",
        "anchor_labels",
        "anchor_weights",
        "parent_anchor_labels",
        "parent_anchor_weights",
        "marker_centroids",
        "marker_mask",
        "program_matrix",
        "target_program_scores",
        "unknown_targets",
        "domain_labels",
        "segmentation_confidence",
        "ood_mask",
        "spot_assignment",
        "target_spatial_expression",
        "scgpt_type_prototypes",
        "scgpt_type_variances",
    )

    _LONG_TENSORS: ClassVar[Tuple[str, ...]] = (
        "edge_index",
        "prototype_types",
        "anchor_labels",
        "parent_anchor_labels",
        "domain_labels",
    )
    _BOOL_TENSORS: ClassVar[Tuple[str, ...]] = (
        "prototype_mask",
        "marker_mask",
        "ood_mask",
    )

    def validate(self, stage: TrainingStage) -> None:
        if self.morphology.ndim != 2 or not torch.is_floating_point(self.morphology):
            raise ValueError("morphology must be a floating cells-by-features tensor")
        cells = self.morphology.shape[0]
        if cells == 0 or self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("batch needs cells and a (2, edges) edge_index")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must be long")
        if self.edge_index.numel() and (
            bool((self.edge_index < 0).any()) or int(self.edge_index.max()) >= cells
        ):
            raise ValueError("edge_index contains an invalid cell")
        if self.edge_weight is not None and self.edge_weight.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_weight must have one value per edge")
        if (
            self.prototype_means.ndim != 2
            or self.prototype_variances.shape != self.prototype_means.shape
        ):
            raise ValueError("prototype means/variances must have matching 2-D shapes")
        prototypes = self.prototype_means.shape[0]
        if self.prototype_types.shape != (prototypes,) or self.prototype_types.dtype != torch.long:
            raise ValueError("prototype_types must be a long vector")
        if self.prototype_weights.shape != (prototypes,):
            raise ValueError("prototype_weights must align to prototypes")
        if torch.any(self.prototype_variances <= 0) or torch.any(self.prototype_weights < 0):
            raise ValueError("prototype variances must be positive and weights non-negative")
        if not torch.isfinite(self.prototype_weights).all() or not bool(
            self.prototype_weights.sum() > 0
        ):
            raise ValueError("prototype_weights must have finite positive mass")
        if self.prototype_mask is not None and self.prototype_mask.shape != (prototypes,):
            raise ValueError("prototype_mask must align to prototypes")
        if self.target_composition.ndim != 1 or self.target_pseudobulk.ndim != 1:
            raise ValueError("sample composition and pseudobulk must be vectors")
        types = self.target_composition.shape[0]
        genes = self.target_pseudobulk.shape[0]
        if types < 2 or genes == 0:
            raise ValueError("training batches require at least two cell types and one gene")
        if (
            bool((self.target_composition < 0).any())
            or not torch.isfinite(self.target_composition).all()
        ):
            raise ValueError("target_composition must be finite and non-negative")
        if float(self.target_composition.sum()) <= 0:
            raise ValueError("target_composition must have positive mass")
        if not torch.isfinite(self.target_pseudobulk).all():
            raise ValueError("target_pseudobulk must be finite")
        if prototypes and (
            bool((self.prototype_types < 0).any()) or bool((self.prototype_types >= types).any())
        ):
            raise ValueError("prototype_types contains a type outside target_composition")
        if not torch.isfinite(self.morphology).all():
            raise ValueError("morphology must be finite")
        for name in ("prototype_means", "prototype_variances", "prototype_weights"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError("%s must be finite" % name)
        if self.cell_weights is not None:
            if self.cell_weights.shape != (cells,):
                raise ValueError("cell_weights must align to cells")
            if not torch.isfinite(self.cell_weights).all() or bool((self.cell_weights < 0).any()):
                raise ValueError("cell_weights must be finite and non-negative")
        if self.molecular_responsibilities is not None:
            responsibilities = self.molecular_responsibilities
            if responsibilities.shape != (cells, prototypes):
                raise ValueError("molecular_responsibilities must have shape (cells, prototypes)")
            if (
                not torch.is_floating_point(responsibilities)
                or not torch.isfinite(responsibilities).all()
                or bool((responsibilities < 0).any())
            ):
                raise ValueError("molecular_responsibilities must be finite and non-negative")
            row_mass = responsibilities.sum(dim=1)
            if bool((row_mass > 1.0 + 1.0e-4).any()):
                raise ValueError(
                    "molecular responsibility rows must contain at most unit known mass"
                )
        if self.anchor_labels is not None and self.anchor_labels.shape != (cells,):
            raise ValueError("anchor_labels must align to cells")
        if self.anchor_weights is not None:
            if self.anchor_weights.shape != (cells,):
                raise ValueError("anchor_weights must align to cells")
            if not torch.isfinite(self.anchor_weights).all() or bool(
                (self.anchor_weights < 0).any()
            ):
                raise ValueError("anchor_weights must be finite and non-negative")
        if self.parent_anchor_labels is not None:
            if (
                self.parent_anchor_labels.shape != (cells,)
                or self.parent_anchor_labels.dtype != torch.long
            ):
                raise ValueError("parent_anchor_labels must be a long vector aligned to cells")
        if self.parent_anchor_weights is not None:
            if self.parent_anchor_weights.shape != (cells,):
                raise ValueError("parent_anchor_weights must align to cells")
            if not torch.isfinite(self.parent_anchor_weights).all() or bool(
                (self.parent_anchor_weights < 0).any()
            ):
                raise ValueError("parent_anchor_weights must be finite and non-negative")
        if self.parent_anchor_labels is None and self.parent_anchor_weights is not None:
            raise ValueError("parent_anchor_weights require parent_anchor_labels")
        if self.domain_labels is not None:
            if self.domain_labels.shape != (cells,) or self.domain_labels.dtype != torch.long:
                raise ValueError("domain_labels must be a long vector aligned to cells")
        if self.segmentation_confidence is not None:
            if self.segmentation_confidence.shape != (cells,):
                raise ValueError("segmentation_confidence must align to cells")
            if (
                not torch.isfinite(self.segmentation_confidence).all()
                or bool((self.segmentation_confidence < 0).any())
                or bool((self.segmentation_confidence > 1).any())
            ):
                raise ValueError("segmentation_confidence must be finite and lie in [0, 1]")
        if self.ood_mask is not None:
            if self.ood_mask.shape != (cells,) or self.ood_mask.dtype != torch.bool:
                raise ValueError("ood_mask must be boolean and align to cells")
        if self.unknown_targets is not None:
            if self.unknown_targets.shape != (cells,):
                raise ValueError("unknown_targets must align to cells")
            if (
                not torch.isfinite(self.unknown_targets).all()
                or bool((self.unknown_targets < 0).any())
                or bool((self.unknown_targets > 1).any())
            ):
                raise ValueError("unknown_targets must be finite and lie in [0, 1]")
        if (self.program_matrix is None) != (self.target_program_scores is None):
            raise ValueError("program matrix and target scores must be supplied together")
        if self.marker_centroids is not None:
            if self.marker_centroids.shape != (types, genes):
                raise ValueError("marker_centroids must have shape (cell types, genes)")
            if (
                self.marker_mask is not None
                and self.marker_mask.shape != self.marker_centroids.shape
            ):
                raise ValueError("marker_mask must align to marker_centroids")
            if self.marker_mask is not None and self.marker_mask.dtype != torch.bool:
                raise TypeError("marker_mask must be boolean")
        elif self.marker_mask is not None:
            raise ValueError("marker_mask requires marker_centroids")
        if self.program_matrix is not None:
            if self.program_matrix.ndim != 2 or self.program_matrix.shape[0] != genes:
                raise ValueError("program_matrix must have shape (genes, programs)")
            if self.target_program_scores is None or self.target_program_scores.shape not in {
                (self.program_matrix.shape[1],),
                (types, self.program_matrix.shape[1]),
            }:
                raise ValueError("target_program_scores must be programs or cell-types-by-programs")
        if (self.scgpt_type_prototypes is None) != (self.scgpt_type_variances is None):
            raise ValueError("scGPT prototypes and variances must be supplied together")
        if self.scgpt_type_prototypes is not None:
            assert self.scgpt_type_variances is not None
            if self.scgpt_type_prototypes.ndim != 2:
                raise ValueError("scgpt_type_prototypes must be two-dimensional")
            if self.scgpt_type_prototypes.shape[0] != types:
                raise ValueError("scgpt_type_prototypes must have one row per cell type")
            if self.scgpt_type_variances.shape != self.scgpt_type_prototypes.shape:
                raise ValueError("scgpt_type_variances must align to scgpt_type_prototypes")
            if bool((self.scgpt_type_variances < 0).any()):
                raise ValueError("scgpt_type_variances must be non-negative")
            if (
                not torch.isfinite(self.scgpt_type_prototypes).all()
                or not torch.isfinite(self.scgpt_type_variances).all()
            ):
                raise ValueError("scGPT prototype tensors must be finite")
            if not self.scgpt_space_id.strip():
                raise ValueError("scGPT supervision requires scgpt_space_id")
        elif self.scgpt_space_id:
            raise ValueError("scgpt_space_id requires scGPT supervision")
        if self.spot_assignment is not None:
            if self.spot_assignment.ndim != 2 or self.spot_assignment.shape[1] != cells:
                raise ValueError("spot_assignment must have shape (spots, cells)")
            if (
                not torch.isfinite(self.spot_assignment).all()
                or bool((self.spot_assignment < 0).any())
                or bool((self.spot_assignment.sum(dim=1) <= 0).any())
            ):
                raise ValueError(
                    "spot_assignment must be finite, non-negative, and give every spot mass"
                )
        if (self.spot_assignment is None) != (self.target_spatial_expression is None):
            raise ValueError(
                "spot assignment and target spatial expression must be supplied together"
            )
        if self.spot_assignment is not None:
            assert self.target_spatial_expression is not None
            if self.target_spatial_expression.shape != (self.spot_assignment.shape[0], genes):
                raise ValueError("target_spatial_expression must have shape (spots, genes)")
            if not torch.isfinite(self.target_spatial_expression).all():
                raise ValueError("target_spatial_expression must be finite")
            if len(self.spot_ids) != self.spot_assignment.shape[0]:
                raise ValueError("spot_ids must align to spot_assignment")
            if any(not value.strip() for value in self.spot_ids) or len(set(self.spot_ids)) != len(
                self.spot_ids
            ):
                raise ValueError("spot_ids must be unique and non-empty")
        elif self.spot_ids:
            raise ValueError("spot_ids require spatial targets")
        if self.type_names and len(self.type_names) != types:
            raise ValueError("type_names must align to target_composition")
        if self.nucleus_ids:
            if len(self.nucleus_ids) != cells or len(set(self.nucleus_ids)) != cells:
                raise ValueError("nucleus_ids must be unique and align to morphology")
            if any(not value.strip() for value in self.nucleus_ids):
                raise ValueError("nucleus_ids cannot contain empty values")
        if self.gene_names and len(self.gene_names) != genes:
            raise ValueError("gene_names must align to target_pseudobulk")
        if self.prototype_ids and len(self.prototype_ids) != prototypes:
            raise ValueError("prototype_ids must align to prototypes")
        for name, values in (
            ("type_names", self.type_names),
            ("gene_names", self.gene_names),
            ("prototype_ids", self.prototype_ids),
            ("spot_ids", self.spot_ids),
        ):
            if any(not value.strip() for value in values) or len(set(values)) != len(values):
                raise ValueError("%s must contain unique non-empty strings" % name)
        if not (len(self.source_artifacts) == len(self.source_sha256) == len(self.source_roles)):
            raise ValueError("source artifacts, SHA-256 values, and roles must align")
        allowed_source_roles = {
            "sample_assay",
            "shared_manifest",
            "shared_teacher",
            "frozen_e_step",
        }
        if any(value not in allowed_source_roles for value in self.source_roles):
            raise ValueError("source_roles contains an unsupported provenance role")
        if any(not value.strip() for value in self.molecular_training_donors) or len(
            set(self.molecular_training_donors)
        ) != len(self.molecular_training_donors):
            raise ValueError("molecular_training_donors must be unique non-empty strings")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.source_sha256
        ):
            raise ValueError("source_sha256 must contain lowercase SHA-256 digests")
        if not self.sample_id.strip() or not self.bag_id.strip():
            raise ValueError("sample_id and bag_id cannot be empty")
        if (
            not self.latent_space_id.strip()
            or not self.feature_space_id.strip()
            or not self.expression_space_id.strip()
            or not self.weak_target_scope_id.strip()
            or not self.weak_target_granularity.strip()
        ):
            raise ValueError(
                "latent, feature, expression, and weak-target identities cannot be empty"
            )
        if self.weak_target_scope_id != "unspecified":
            prefix = "sha256:"
            digest = self.weak_target_scope_id[len(prefix) :]
            if (
                not self.weak_target_scope_id.startswith(prefix)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    "weak_target_scope_id must be unspecified or sha256:<lowercase digest>"
                )
        StageInputs(
            histology_features=self.morphology,
            matched_rna=self.prototype_means,
            target_spatial_expression=self.target_spatial_expression,
            analysis_role=self.analysis_role,
        ).validate(stage)

    def to(self, device: torch.device) -> "HEIRTrainingBatch":
        values: Dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, Tensor) else value
        return HEIRTrainingBatch(**values)  # type: ignore[arg-type]

    @staticmethod
    def _numpy_tensor(value: Tensor, name: str) -> np.ndarray:
        if value.layout != torch.strided:
            raise ValueError("%s must be a dense strided tensor" % name)
        return value.detach().cpu().numpy()

    def save_npz(self, path: Union[str, os.PathLike], compressed: bool = True) -> None:
        """Atomically write a pickle-free, versioned batch artifact."""

        payload: Dict[str, np.ndarray] = {
            "__contract__": np.asarray(self.CONTRACT, dtype=np.dtype("U")),
            "__version__": np.asarray(self.CONTRACT_VERSION, dtype=np.int64),
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in self._OPTIONAL_TENSORS:
                payload["__present__%s" % item.name] = np.asarray(
                    value is not None,
                    dtype=bool,
                )
            if isinstance(value, Tensor):
                payload[item.name] = self._numpy_tensor(value, item.name)
            elif isinstance(value, tuple):
                payload[item.name] = np.asarray(value, dtype=np.dtype("U"))
            elif isinstance(value, str):
                payload[item.name] = np.asarray(value, dtype=np.dtype("U"))

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=".npz.tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                saver = np.savez_compressed if compressed else np.savez
                saver(handle, **payload)
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
    def load_npz(cls, path: Union[str, os.PathLike]) -> "HEIRTrainingBatch":
        """Load a batch without pickle and reject partial/foreign contracts."""

        required_tensors = {
            "morphology",
            "edge_index",
            "prototype_means",
            "prototype_variances",
            "prototype_types",
            "prototype_weights",
            "target_composition",
            "target_pseudobulk",
        }
        metadata = {
            "sample_id",
            "bag_id",
            "donor_id",
            "block_id",
            "analysis_role",
            "latent_space_id",
            "feature_space_id",
            "expression_space_id",
            "scgpt_space_id",
            "weak_target_scope_id",
            "weak_target_granularity",
            "type_names",
            "gene_names",
            "prototype_ids",
            "spot_ids",
            "source_artifacts",
            "source_sha256",
            "source_roles",
            "molecular_training_donors",
        }
        with np.load(path, allow_pickle=False) as archive:
            if "__contract__" not in archive or "__version__" not in archive:
                raise ValueError("NPZ artifact has no HEIR training-batch metadata")
            contract = str(np.asarray(archive["__contract__"]).item())
            version = int(np.asarray(archive["__version__"]).item())
            if contract != cls.CONTRACT:
                raise ValueError("expected %s NPZ contract, found %s" % (cls.CONTRACT, contract))
            if version != cls.CONTRACT_VERSION:
                if version == 1:
                    raise ValueError(
                        "HEIR training-batch version 1 lacks required provenance; "
                        "regenerate it with assemble-batch"
                    )
                if version not in {2, 3, 4, 5}:
                    raise ValueError("unsupported HEIR training-batch version %d" % version)
            optional_legacy_metadata = (
                {
                    "feature_space_id",
                    "expression_space_id",
                    "scgpt_space_id",
                    "spot_ids",
                }
                if version in {2, 3}
                else set()
            )
            if version < 6:
                optional_legacy_metadata.update({"weak_target_scope_id", "weak_target_granularity"})
            required_metadata = metadata - optional_legacy_metadata
            missing = sorted((required_tensors | required_metadata) - set(archive.files))
            if missing:
                raise ValueError("training-batch artifact is missing: %s" % ", ".join(missing))

            values: Dict[str, object] = {}
            for name in required_tensors:
                array = np.array(archive[name], copy=True)
                dtype = torch.long if name in cls._LONG_TENSORS else torch.float32
                values[name] = torch.as_tensor(array, dtype=dtype)
            for name in cls._OPTIONAL_TENSORS:
                presence = "__present__%s" % name
                if presence not in archive:
                    if name in {
                        "parent_anchor_labels",
                        "parent_anchor_weights",
                        "molecular_responsibilities",
                    }:
                        values[name] = None
                        continue
                    raise ValueError("training-batch artifact lacks presence flag for %s" % name)
                if not bool(np.asarray(archive[presence]).item()):
                    values[name] = None
                    continue
                if name not in archive:
                    raise ValueError("training-batch artifact marks absent array %s present" % name)
                array = np.array(archive[name], copy=True)
                if name in cls._LONG_TENSORS:
                    dtype = torch.long
                elif name in cls._BOOL_TENSORS:
                    dtype = torch.bool
                else:
                    dtype = torch.float32
                values[name] = torch.as_tensor(array, dtype=dtype)
            for name in (
                "sample_id",
                "bag_id",
                "donor_id",
                "block_id",
                "analysis_role",
                "latent_space_id",
                "feature_space_id",
                "expression_space_id",
                "scgpt_space_id",
                "weak_target_scope_id",
                "weak_target_granularity",
            ):
                values[name] = (
                    "unspecified"
                    if name in {"feature_space_id", "expression_space_id"} and name not in archive
                    else ""
                    if name == "scgpt_space_id" and name not in archive
                    else "unspecified"
                    if name == "weak_target_scope_id" and name not in archive
                    else "legacy_unspecified"
                    if name == "weak_target_granularity" and name not in archive
                    else str(np.asarray(archive[name]).item())
                )
            for name in (
                "nucleus_ids",
                "type_names",
                "gene_names",
                "prototype_ids",
                "spot_ids",
                "source_artifacts",
                "source_sha256",
                "source_roles",
                "molecular_training_donors",
            ):
                if name not in archive:
                    if name in {"nucleus_ids", "spot_ids"}:
                        values[name] = ()
                        continue
                    raise ValueError("training-batch artifact is missing metadata %s" % name)
                array = np.asarray(archive[name])
                if array.ndim != 1:
                    raise ValueError("%s metadata must be one-dimensional" % name)
                values[name] = tuple(str(value) for value in array.tolist())
        return cls(**values)  # type: ignore[arg-type]
