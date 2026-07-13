"""Donor-safe molecular, pretraining, and personalized training stages."""

from .batch import HEIRTrainingBatch
from .contracts import (
    MolecularEStepArtifact,
    ValidatedInitializationReceipt,
    array_content_sha256,
    donor_cross_type_permutation,
    frozen_transport_telemetry,
    ordered_identity_sha256,
    recompute_initialization_validation,
    validate_primary_claim_exclusions,
)
from .rna import RNATrainingResult, train_rna_vae
from .splits import (
    grouped_fold_assignment,
    spatial_block_split_masks,
    subset_histology_bag,
    validate_grouped_splits,
)
from .stages import StageInputs, TrainingStage
from .trainer import HEIRTrainer, HEIRTrainingResult, aggregate_to_spots

__all__ = [
    "RNATrainingResult",
    "HEIRTrainingBatch",
    "MolecularEStepArtifact",
    "ValidatedInitializationReceipt",
    "HEIRTrainer",
    "HEIRTrainingResult",
    "StageInputs",
    "TrainingStage",
    "grouped_fold_assignment",
    "spatial_block_split_masks",
    "subset_histology_bag",
    "train_rna_vae",
    "aggregate_to_spots",
    "array_content_sha256",
    "donor_cross_type_permutation",
    "frozen_transport_telemetry",
    "ordered_identity_sha256",
    "recompute_initialization_validation",
    "validate_primary_claim_exclusions",
    "validate_grouped_splits",
]
