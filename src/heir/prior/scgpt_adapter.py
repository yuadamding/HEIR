"""Stable artifact boundary for an externally adapted scGPT teacher.

The scGPT environment is intentionally decoupled from the HEIR graph runtime.
An adaptation job exports cell embeddings and provenance; HEIR consumes the
artifact without importing a particular scGPT release at training time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass(frozen=True)
class SCGPTTeacherArtifact:
    cell_ids: np.ndarray
    embeddings: np.ndarray
    type_names: np.ndarray
    type_indices: np.ndarray
    gene_vocabulary: np.ndarray
    checkpoint_id: str
    training_donors: np.ndarray

    def validate(self) -> None:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.cell_ids):
            raise ValueError("scGPT cell IDs and embeddings are misaligned")
        if self.embeddings.shape[1] == 0:
            raise ValueError("scGPT embeddings must contain features")
        for name, values in (
            ("cell_ids", self.cell_ids),
            ("type_names", self.type_names),
            ("gene_vocabulary", self.gene_vocabulary),
            ("training_donors", self.training_donors),
        ):
            text = [str(value).strip() for value in np.asarray(values).tolist()]
            if not text or any(not value for value in text) or len(set(text)) != len(text):
                raise ValueError("scGPT %s must contain unique non-empty values" % name)
        if self.type_indices.shape != (len(self.cell_ids),):
            raise ValueError("scGPT type indices are misaligned")
        if np.any(self.type_indices < 0) or np.any(self.type_indices >= len(self.type_names)):
            raise ValueError("scGPT type index is out of range")
        if len(self.type_names) == 0 or not np.array_equal(
            np.unique(self.type_indices), np.arange(len(self.type_names))
        ):
            raise ValueError("each declared scGPT type needs at least one cell")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("scGPT embeddings must be finite")
        if not self.checkpoint_id or len(self.training_donors) == 0:
            raise ValueError("scGPT checkpoint and donor provenance are required")

    def type_prototypes(self) -> np.ndarray:
        self.validate()
        return np.stack(
            [
                self.embeddings[self.type_indices == index].mean(axis=0)
                for index in range(len(self.type_names))
            ]
        ).astype(np.float32)

    def type_variances(self) -> np.ndarray:
        """Return finite diagonal within-type variances for moment matching."""

        self.validate()
        values = []
        for index in range(len(self.type_names)):
            selected = self.embeddings[self.type_indices == index]
            if len(selected) == 0:
                raise ValueError("each declared scGPT type needs at least one embedding")
            values.append(selected.var(axis=0))
        return np.stack(values).astype(np.float32)

    def to_npz(self, path: Union[str, Path]) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            cell_ids=self.cell_ids.astype(np.str_),
            embeddings=self.embeddings.astype(np.float32),
            type_names=self.type_names.astype(np.str_),
            type_indices=self.type_indices.astype(np.int64),
            gene_vocabulary=self.gene_vocabulary.astype(np.str_),
            checkpoint_id=np.asarray(self.checkpoint_id, dtype=np.str_),
            training_donors=self.training_donors.astype(np.str_),
        )

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "SCGPTTeacherArtifact":
        with np.load(path, allow_pickle=False) as values:
            result = cls(
                cell_ids=values["cell_ids"],
                embeddings=values["embeddings"],
                type_names=values["type_names"],
                type_indices=values["type_indices"],
                gene_vocabulary=values["gene_vocabulary"],
                checkpoint_id=str(values["checkpoint_id"].item()),
                training_donors=values["training_donors"],
            )
        result.validate()
        return result
