"""OOD detection and abstention policies used by the core pipeline."""

from .ood import MahalanobisOOD
from .policy import AbstentionDecision, apply_abstention_policy

__all__ = [
    "AbstentionDecision",
    "MahalanobisOOD",
    "apply_abstention_policy",
]
