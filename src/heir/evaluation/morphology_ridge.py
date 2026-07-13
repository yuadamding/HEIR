"""Compatibility imports for the modular morphology statistical core."""

from .morphology_gate import (
    MORPHOLOGY_RIDGE_REPORT_SCHEMA,
    _evaluate_permutation_null,
    evaluate_morphology_ridge_gate,
    validate_experiment_identity,
)
from .permutations import donor_type_block_permutation, donor_type_roi_permutation
from .ridge_probe import (
    MolecularTargetFit,
    OracleRidgeFit,
    fit_oracle_ridge_probe,
    predict_oracle_ridge,
)

__all__ = [
    "MORPHOLOGY_RIDGE_REPORT_SCHEMA",
    "MolecularTargetFit",
    "OracleRidgeFit",
    "donor_type_block_permutation",
    "donor_type_roi_permutation",
    "evaluate_morphology_ridge_gate",
    "fit_oracle_ridge_probe",
    "predict_oracle_ridge",
    "validate_experiment_identity",
    "_evaluate_permutation_null",
]
