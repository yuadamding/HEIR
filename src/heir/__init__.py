"""HEIR: matched-single-nucleus spatialization on histology.

The package deliberately separates personalized inference (H&E plus a matched
RNA reference) from distilled H&E-only inference. Spatial transcriptomics is
an evaluation modality and is never consumed by personalized refinement.
"""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
__version__ = "0.1.0"
