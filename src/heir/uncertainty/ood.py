"""Shrinkage-Mahalanobis OOD scores for cached pathology features."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class MahalanobisOOD:
    mean: Optional[np.ndarray] = None
    precision: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    quantile: float = 0.95
    training_donors: Tuple[str, ...] = ()
    source_sha256: Tuple[str, ...] = ()
    feature_space_id: str = ""

    CONTRACT = "heir.mahalanobis_ood"
    CONTRACT_VERSION = 2

    def fit(
        self,
        features: np.ndarray,
        analysis_role: str,
        quantile: Optional[float] = None,
        training_donors: Tuple[str, ...] = (),
        feature_space_id: str = "",
    ) -> "MahalanobisOOD":
        role = analysis_role.strip().lower()
        if role not in {"train", "development", "inner_train"}:
            raise ValueError("OOD reference must be fitted without locked-test observations")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
            raise ValueError("features must have shape (items>=2, dimensions)")
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        selected_quantile = self.quantile if quantile is None else float(quantile)
        if not 0.0 < selected_quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        donors = tuple(sorted(set(str(value).strip() for value in training_donors)))
        if any(not value for value in donors):
            raise ValueError("training_donors cannot contain empty values")
        if not feature_space_id.strip():
            raise ValueError("feature_space_id is required for an OOD detector")
        estimator = LedoitWolf().fit(values)
        self.mean = estimator.location_.astype(np.float64)
        self.precision = estimator.precision_.astype(np.float64)
        self.quantile = selected_quantile
        self.threshold = float(np.quantile(self.score(values), selected_quantile))
        self.training_donors = donors
        self.feature_space_id = feature_space_id.strip()
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision is None:
            raise RuntimeError("fit or load the OOD detector first")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("feature width differs from the fitted OOD reference")
        if not np.isfinite(values).all():
            raise ValueError("OOD features must be finite")
        delta = values - self.mean
        # Let NumPy choose a BLAS-backed contraction path.  The unoptimized
        # three-operand einsum performs the dense quadratic form element by
        # element and is prohibitively slow for the 1,546-dimensional
        # pathology vectors used by snPATHO.
        squared = np.einsum(
            "ni,ij,nj->n",
            delta,
            self.precision,
            delta,
            optimize=True,
        )
        return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)

    def is_ood(self, features: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("the OOD threshold has not been fitted")
        return self.score(features) > self.threshold

    def to_npz(self, path: Union[str, Path]) -> None:
        if self.mean is None or self.precision is None or self.threshold is None:
            raise RuntimeError("cannot save an unfitted OOD detector")
        self._validate_loaded()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            __contract__=np.asarray(self.CONTRACT, dtype=np.dtype("U")),
            __version__=np.asarray(self.CONTRACT_VERSION, dtype=np.int64),
            mean=self.mean,
            precision=self.precision,
            threshold=np.asarray(self.threshold),
            quantile=np.asarray(self.quantile),
            training_donors=np.asarray(self.training_donors, dtype=np.dtype("U")),
            source_sha256=np.asarray(self.source_sha256, dtype=np.dtype("U")),
            feature_space_id=np.asarray(self.feature_space_id, dtype=np.dtype("U")),
        )

    def _validate_loaded(self) -> None:
        if self.mean is None or self.precision is None or self.threshold is None:
            raise ValueError("OOD artifact is missing fitted parameters")
        mean = np.asarray(self.mean, dtype=np.float64)
        precision = np.asarray(self.precision, dtype=np.float64)
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("OOD mean must be a non-empty vector")
        if precision.shape != (mean.size, mean.size):
            raise ValueError("OOD precision must be square and align to the mean")
        if not np.isfinite(mean).all() or not np.isfinite(precision).all():
            raise ValueError("OOD fitted parameters must be finite")
        if not np.allclose(precision, precision.T, rtol=1.0e-5, atol=1.0e-7):
            raise ValueError("OOD precision must be symmetric")
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("OOD threshold must be finite and non-negative")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("OOD quantile must be in (0, 1)")
        if not self.feature_space_id.strip():
            raise ValueError("OOD artifact lacks feature_space_id")
        if any(not value.strip() for value in self.training_donors) or len(
            set(self.training_donors)
        ) != len(self.training_donors):
            raise ValueError("OOD training donors must be unique and non-empty")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_sha256
        ):
            raise ValueError("OOD source_sha256 entries must be lowercase SHA-256 digests")
        self.mean = mean
        self.precision = precision

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "MahalanobisOOD":
        with np.load(path, allow_pickle=False) as values:
            if "__contract__" not in values or "__version__" not in values:
                raise ValueError("OOD artifact lacks HEIR contract metadata")
            if str(np.asarray(values["__contract__"]).item()) != cls.CONTRACT:
                raise ValueError("artifact is not a HEIR Mahalanobis OOD detector")
            version = int(np.asarray(values["__version__"]).item())
            if version not in {1, cls.CONTRACT_VERSION}:
                raise ValueError("unsupported OOD artifact version")
            result = cls(
                mean=values["mean"],
                precision=values["precision"],
                threshold=float(values["threshold"]),
                quantile=float(values["quantile"]),
                training_donors=(
                    tuple(str(value) for value in values["training_donors"].tolist())
                    if "training_donors" in values
                    else ()
                ),
                source_sha256=(
                    tuple(str(value) for value in values["source_sha256"].tolist())
                    if "source_sha256" in values
                    else ()
                ),
                feature_space_id=(
                    str(np.asarray(values["feature_space_id"]).item())
                    if "feature_space_id" in values
                    else ""
                ),
            )
        if version == 1 and not result.feature_space_id:
            raise ValueError("legacy OOD artifact lacks required feature-space provenance")
        result._validate_loaded()
        return result
