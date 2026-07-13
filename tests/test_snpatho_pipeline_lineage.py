"""Focused donor-lineage checks for the default snPATHO pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_snpatho_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_snpatho_pipeline_lineage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


def _batches():
    train = SimpleNamespace(molecular_training_donors=("B1",))
    validation = SimpleNamespace(molecular_training_donors=("B1",))
    return train, validation


def test_pipeline_accepts_direct_target_with_complete_upstream_exposure_lineage() -> None:
    train, validation = _batches()
    metadata = {
        "direct_training_donors": ["4066"],
        "validation_donors": ["4066"],
        "initial_heir_training_donors": [],
        "rna_vae_training_donors": ["B1"],
        "residual_geometry_training_donors": [],
        "training_donors": ["4066", "B1"],
    }

    PIPELINE._validate_checkpoint_donor_lineage(
        metadata,
        train,
        validation,
        sample="4066",
    )


def test_pipeline_rejects_legacy_singleton_that_omits_upstream_exposure() -> None:
    train, validation = _batches()
    metadata = {
        "direct_training_donors": ["4066"],
        "validation_donors": ["4066"],
        "initial_heir_training_donors": [],
        "rna_vae_training_donors": ["B1"],
        "residual_geometry_training_donors": [],
        "training_donors": ["4066"],
    }

    with pytest.raises(ValueError, match="all-exposure"):
        PIPELINE._validate_checkpoint_donor_lineage(
            metadata,
            train,
            validation,
            sample="4066",
        )
