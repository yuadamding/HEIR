"""Training-only gene-program construction and score transforms."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from sklearn.decomposition import NMF


@dataclass(frozen=True)
class GenePrograms:
    """Non-negative genes-by-program loadings with frozen gene ordering."""

    gene_names: np.ndarray
    loadings: np.ndarray
    names: np.ndarray
    training_donors: np.ndarray

    def validate(self) -> None:
        if self.gene_names.ndim != 1 or self.loadings.ndim != 2:
            raise ValueError("gene_names must be 1-D and loadings 2-D")
        if self.loadings.shape[0] != len(self.gene_names):
            raise ValueError("loadings rows must follow gene_names")
        if self.loadings.shape[1] != len(self.names) or self.loadings.shape[1] == 0:
            raise ValueError("program names must match loading columns")
        if np.any(self.loadings < 0) or not np.isfinite(self.loadings).all():
            raise ValueError("program loadings must be finite and non-negative")
        if len(np.unique(self.gene_names.astype(str))) != len(self.gene_names):
            raise ValueError("gene names must be unique")
        if len(self.training_donors) == 0:
            raise ValueError("training donor provenance is required")

    def transform(self, expression: np.ndarray) -> np.ndarray:
        values = np.asarray(expression, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.loadings.shape[0]:
            raise ValueError("expression must have shape (cells, genes)")
        return (values @ self.loadings).astype(np.float32)

    def to_npz(self, path: Union[str, Path]) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            gene_names=self.gene_names.astype(np.str_),
            loadings=self.loadings.astype(np.float32),
            names=self.names.astype(np.str_),
            training_donors=self.training_donors.astype(np.str_),
        )

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "GenePrograms":
        with np.load(path, allow_pickle=False) as values:
            result = cls(
                gene_names=values["gene_names"],
                loadings=values["loadings"],
                names=values["names"],
                training_donors=values["training_donors"],
            )
        result.validate()
        return result


def fit_gene_programs(
    expression: np.ndarray,
    gene_names: Sequence[object],
    donor_ids: Sequence[object],
    analysis_roles: Sequence[object],
    num_programs: int = 32,
    seed: int = 17,
    max_iter: int = 500,
) -> GenePrograms:
    """Fit NMF programs and fail if locked-test observations are supplied.

    The explicit role check prevents a subtle but common leak: selecting genes
    or modules after looking at the held-out spatial cohort.
    """

    values = np.asarray(expression, dtype=np.float64)
    genes = np.asarray([str(value) for value in gene_names], dtype=np.str_)
    donors = np.asarray([str(value) for value in donor_ids], dtype=np.str_)
    roles = np.asarray([str(value).lower() for value in analysis_roles], dtype=np.str_)
    if values.ndim != 2 or values.shape != (len(donors), len(genes)):
        raise ValueError("expression, donor_ids, and gene_names are misaligned")
    if roles.shape != donors.shape:
        raise ValueError("analysis_roles must have one entry per expression row")
    allowed_roles = {"train", "inner_train", "training"}
    disallowed = sorted(set(roles.tolist()) - allowed_roles)
    if disallowed:
        raise ValueError("gene programs cannot use non-training roles: %s" % ", ".join(disallowed))
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("NMF expression must be finite and non-negative")
    if num_programs <= 0 or num_programs > min(values.shape):
        raise ValueError("num_programs must be in [1, min(cells, genes)]")
    model = NMF(
        n_components=num_programs,
        init="nndsvda",
        random_state=seed,
        max_iter=max_iter,
    )
    model.fit(values)
    loadings = model.components_.T
    column_sums = loadings.sum(axis=0, keepdims=True)
    loadings = loadings / np.maximum(column_sums, 1.0e-12)
    result = GenePrograms(
        gene_names=genes,
        loadings=loadings.astype(np.float32),
        names=np.asarray(["program_%02d" % (index + 1) for index in range(num_programs)]),
        training_donors=np.unique(donors),
    )
    result.validate()
    return result
