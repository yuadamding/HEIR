"""Modality gates that make target-spatial leakage impossible by default."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch import Tensor


class TrainingStage(str, Enum):
    MOLECULAR = "molecular"
    GENERIC_SPATIAL_PRETRAINING = "generic_spatial_pretraining"
    PERSONALIZED = "personalized"
    REFINEMENT = "refinement"
    DISTILLATION = "distillation"
    CALIBRATION = "calibration"


@dataclass
class StageInputs:
    """Presence flags and tensors crossing a training-stage boundary."""

    histology_features: Optional[Tensor] = None
    matched_rna: Optional[Tensor] = None
    target_spatial_expression: Optional[Tensor] = None
    teacher_predictions: Optional[Tensor] = None
    cell_labels: Optional[Tensor] = None
    analysis_role: str = "train"

    def validate(self, stage: TrainingStage) -> None:
        role = self.analysis_role.strip().lower()
        locked = role in {
            "validation",
            "spatial_validation",
            "external_validation",
            "locked_validation",
            "locked_test",
            "test",
            "external_test",
        }
        if stage == TrainingStage.MOLECULAR:
            if locked:
                raise ValueError("locked-test RNA cannot fit the cohort-level molecular teacher")
            if self.matched_rna is None:
                raise ValueError("molecular training requires RNA")
            if self.histology_features is not None or self.target_spatial_expression is not None:
                raise ValueError("molecular training consumes RNA only")
        elif stage == TrainingStage.GENERIC_SPATIAL_PRETRAINING:
            if role != "pretraining":
                raise ValueError("generic spatial pretraining requires analysis_role=pretraining")
            if self.histology_features is None or self.target_spatial_expression is None:
                raise ValueError(
                    "generic pretraining requires histology and non-target spatial data"
                )
            if locked:
                raise ValueError("locked accessions cannot enter generic spatial pretraining")
        elif stage in {TrainingStage.PERSONALIZED, TrainingStage.REFINEMENT}:
            if self.histology_features is None or self.matched_rna is None:
                raise ValueError("personalized/refinement training requires H&E plus matched RNA")
            if self.target_spatial_expression is not None:
                raise ValueError("target spatial expression is validation-only")
            # A locked donor's H&E and matched RNA are intended inputs in
            # personalized mode. Only its spatial expression remains hidden.
        elif stage == TrainingStage.DISTILLATION:
            if locked:
                raise ValueError(
                    "locked-test teacher predictions cannot train the distilled student"
                )
            if self.histology_features is None or self.teacher_predictions is None:
                raise ValueError("distillation requires histology and frozen-teacher outputs")
            if self.matched_rna is not None or self.target_spatial_expression is not None:
                raise ValueError("the H&E-only student cannot consume RNA or spatial expression")
        elif stage == TrainingStage.CALIBRATION:
            if role not in {"calibration", "development", "inner_validation"}:
                raise ValueError("calibration must use a development donor split")
            if self.cell_labels is None:
                raise ValueError("calibration requires orthogonal labels")
