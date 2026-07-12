"""RNA-manifold summaries and optional molecular-teacher adapters."""

from .programs import GenePrograms, fit_gene_programs
from .prototypes import PrototypeSet, build_sample_prototypes
from .residual_geometry import RNAResidualGeometry, fit_rna_residual_geometry
from .scgpt_adapter import SCGPTTeacherArtifact
from .scvi_adapter import SCVIAdapter

__all__ = [
    "GenePrograms",
    "PrototypeSet",
    "RNAResidualGeometry",
    "SCGPTTeacherArtifact",
    "SCVIAdapter",
    "build_sample_prototypes",
    "fit_gene_programs",
    "fit_rna_residual_geometry",
]
