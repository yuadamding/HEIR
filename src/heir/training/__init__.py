"""Donor-safe molecular, pretraining, and personalized training stages."""

from .batch import HEIRTrainingBatch
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
    "HEIRTrainer",
    "HEIRTrainingResult",
    "StageInputs",
    "TrainingStage",
    "grouped_fold_assignment",
    "spatial_block_split_masks",
    "subset_histology_bag",
    "train_rna_vae",
    "aggregate_to_spots",
    "validate_grouped_splits",
]
