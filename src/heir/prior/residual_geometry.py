"""RNA-derived geometry for biologically constrained latent residuals.

The image model should predict coefficients along molecular directions that
were measured in RNA, rather than learning an arbitrary latent rotation from
weak spatial supervision.  This module turns cell-level RNA latents into a
portable, deterministic artifact containing:

* one orthonormal residual basis per cell type;
* the locally identified PCA rank for each type; and
* a type-specific Euclidean residual bound calibrated to measured molecular
  state separation, prototype covariance, or empirical within-type residuals.

When prototypes are supplied, a cell's residual is measured from its nearest
same-type prototype.  Otherwise it is measured from its type centroid.  A
rank-deficient or rare type keeps all locally supported PCA directions and
borrows only its missing directions from pooled *within-type* RNA residuals.
If even the pooled residuals are rank deficient, a deterministic canonical
completion keeps the returned tensors well shaped and orthonormal.  The
``effective_ranks`` field records how many columns are actually identified by
that type's cells, so downstream code can audit or mask borrowed directions.

``RNAResidualGeometry.model_parameters`` returns arrays in any requested cell
type order.  Its first result can be copied directly into
``HEIRModel.residual_type_basis``; its second result is the per-type analogue
of ``residual_max_norm`` for models that support calibrated type-specific
bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import chi2  # type: ignore[import-untyped]

PathLike = Union[str, Path]
_SCALE_NAMES = ("state", "covariance", "residual")
_SCALE_SOURCES = set(_SCALE_NAMES) | {"pooled_residual", "minimum"}


def _labels(values: object, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("%s must be one-dimensional" % name)
    result = np.asarray([str(value).strip() for value in raw.tolist()], dtype=np.str_)
    if result.size == 0 or any(not value for value in result.tolist()):
        raise ValueError("%s must contain non-empty labels" % name)
    return result


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _sorted_rows(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return lexicographically sorted rows and their original indices."""

    if values.shape[0] <= 1:
        return values.copy(), np.arange(values.shape[0], dtype=np.int64)
    keys = tuple(values[:, index] for index in range(values.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    return values[order], order.astype(np.int64, copy=False)


def _canonical_sign(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            result[:, column] = -vector
    return result


def _principal_components(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Compute significant PCA directions with deterministic signs."""

    if values.shape[0] < 2:
        return np.empty((values.shape[1], 0), dtype=np.float64)
    ordered, _ = _sorted_rows(values)
    centered = ordered - ordered.mean(axis=0, keepdims=True)
    if not np.any(centered):
        return np.empty((values.shape[1], 0), dtype=np.float64)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0:
        return np.empty((values.shape[1], 0), dtype=np.float64)
    threshold = singular_values[0] * max(
        tolerance,
        np.finfo(np.float64).eps * max(centered.shape),
    )
    keep = int(np.count_nonzero(singular_values > threshold))
    return _canonical_sign(right[:keep].T)


def _complete_basis(
    local: np.ndarray,
    pooled: np.ndarray,
    latent_dim: int,
    rank: int,
    tolerance: float,
) -> np.ndarray:
    """Keep local directions, then deterministically complete an orthobasis."""

    accepted: List[np.ndarray] = []
    candidates: List[np.ndarray] = []
    candidates.extend(local[:, index] for index in range(local.shape[1]))
    candidates.extend(pooled[:, index] for index in range(pooled.shape[1]))
    candidates.extend(np.eye(latent_dim, dtype=np.float64)[:, index] for index in range(latent_dim))
    for candidate in candidates:
        vector = np.asarray(candidate, dtype=np.float64).copy()
        # Reorthogonalization avoids accumulating error when pooled and local
        # directions are nearly collinear.
        for _ in range(2):
            for existing in accepted:
                vector -= existing * float(np.dot(existing, vector))
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector = -vector
        accepted.append(vector)
        if len(accepted) == rank:
            break
    if len(accepted) != rank:  # pragma: no cover - canonical axes make this unreachable
        raise RuntimeError("could not construct the requested residual basis")
    return np.stack(accepted, axis=1)


def _state_scale(means: np.ndarray) -> float:
    if means.shape[0] < 2:
        return 0.0
    distances = []
    for first in range(means.shape[0] - 1):
        delta = means[first + 1 :] - means[first]
        distances.extend(np.linalg.norm(delta, axis=1).tolist())
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def _covariance_scale(
    variances: np.ndarray,
    basis: np.ndarray,
    quantile: float,
) -> float:
    if variances.shape[0] == 0:
        return 0.0
    rank = basis.shape[1]
    # For isotropic covariance this is the exact Gaussian radial quantile.
    # For diagonal anisotropic covariance it is a trace-matched approximation
    # that remains stable with small prototype banks.
    retained_weight = np.square(basis).sum(axis=1)
    projected_trace = variances @ retained_weight
    radial_factor = float(np.sqrt(chi2.ppf(quantile, df=rank) / rank))
    radii = np.sqrt(np.maximum(projected_trace, 0.0)) * radial_factor
    return float(np.median(radii))


@dataclass(frozen=True)
class RNAResidualGeometry:
    """Frozen type-specific molecular residual geometry.

    ``residual_type_basis`` has shape ``(types, latent_dim, rank)`` and every
    slice has orthonormal columns.  ``residual_type_max_norm`` has shape
    ``(types,)``.  Columns after ``effective_ranks[type]`` are pooled or
    canonical completion directions, not locally identified PCA components.
    ``state_scales``, ``covariance_scales``, and ``residual_scales`` preserve
    the calibration evidence even when another source wins according to
    ``scale_priority``.
    """

    type_names: np.ndarray
    residual_type_basis: np.ndarray
    residual_type_max_norm: np.ndarray
    effective_ranks: np.ndarray
    n_cells: np.ndarray
    n_prototypes: np.ndarray
    state_scales: np.ndarray
    covariance_scales: np.ndarray
    residual_scales: np.ndarray
    scale_sources: np.ndarray
    calibration_quantile: float
    bound_fraction: float
    minimum_bound: float
    maximum_bound: Optional[float]
    scale_priority: Tuple[str, ...]
    latent_space_id: str = ""
    source_reference_sha256: str = ""
    training_donors: Tuple[str, ...] = ()
    latent_transform_sha256: str = ""

    SCHEMA = "heir.rna_residual_geometry.v1"

    def __post_init__(self) -> None:
        names = _labels(self.type_names, "type_names")
        if len(set(names.tolist())) != len(names):
            raise ValueError("type_names must be unique")
        basis = _readonly(self.residual_type_basis, np.dtype(np.float32))
        bounds = _readonly(self.residual_type_max_norm, np.dtype(np.float32))
        ranks = _readonly(self.effective_ranks, np.dtype(np.int64))
        cell_counts = _readonly(self.n_cells, np.dtype(np.int64))
        prototype_counts = _readonly(self.n_prototypes, np.dtype(np.int64))
        states = _readonly(self.state_scales, np.dtype(np.float32))
        covariances = _readonly(self.covariance_scales, np.dtype(np.float32))
        residuals = _readonly(self.residual_scales, np.dtype(np.float32))
        sources = _labels(self.scale_sources, "scale_sources")
        sources.setflags(write=False)

        type_count = len(names)
        if basis.ndim != 3 or basis.shape[0] != type_count:
            raise ValueError("residual_type_basis must have shape (types, latent_dim, rank)")
        if basis.shape[1] == 0 or basis.shape[2] == 0 or basis.shape[2] > basis.shape[1]:
            raise ValueError("residual basis rank must be in [1, latent_dim]")
        for name, values in (
            ("residual_type_max_norm", bounds),
            ("effective_ranks", ranks),
            ("n_cells", cell_counts),
            ("n_prototypes", prototype_counts),
            ("state_scales", states),
            ("covariance_scales", covariances),
            ("residual_scales", residuals),
            ("scale_sources", sources),
        ):
            if values.shape != (type_count,):
                raise ValueError("%s must contain one value per type" % name)
        if not np.isfinite(basis).all() or not np.isfinite(bounds).all():
            raise ValueError("residual bases and bounds must be finite")
        if np.any(bounds <= 0):
            raise ValueError("residual bounds must be positive")
        if np.any(ranks < 0) or np.any(ranks > basis.shape[2]):
            raise ValueError("effective_ranks must be between zero and the requested rank")
        if np.any(cell_counts < 0) or np.any(prototype_counts < 0):
            raise ValueError("cell and prototype counts must be non-negative")
        if np.any(ranks > np.maximum(cell_counts - 1, 0)):
            raise ValueError("effective rank cannot exceed within-type sample rank")
        scales = np.concatenate((states, covariances, residuals))
        if not np.isfinite(scales).all() or np.any(scales < 0):
            raise ValueError("calibration scales must be finite and non-negative")
        unknown_sources = sorted(set(sources.tolist()) - _SCALE_SOURCES)
        if unknown_sources:
            raise ValueError("unknown scale source: %s" % ", ".join(unknown_sources))
        identity = np.eye(basis.shape[2], dtype=np.float64)
        for type_basis in basis.astype(np.float64):
            if not np.allclose(type_basis.T @ type_basis, identity, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError("each residual type basis must have orthonormal columns")
        if not 0 < self.calibration_quantile < 1:
            raise ValueError("calibration_quantile must be between zero and one")
        if not np.isfinite(self.bound_fraction) or self.bound_fraction <= 0:
            raise ValueError("bound_fraction must be finite and positive")
        if not np.isfinite(self.minimum_bound) or self.minimum_bound <= 0:
            raise ValueError("minimum_bound must be finite and positive")
        if self.maximum_bound is not None and (
            not np.isfinite(self.maximum_bound) or self.maximum_bound < self.minimum_bound
        ):
            raise ValueError("maximum_bound must be finite and at least minimum_bound")
        priority = tuple(str(value).strip().lower() for value in self.scale_priority)
        if not priority or len(set(priority)) != len(priority) or set(priority) - set(_SCALE_NAMES):
            raise ValueError("scale_priority must contain unique state/covariance/residual entries")
        if not isinstance(self.latent_space_id, str):
            raise TypeError("latent_space_id must be a string")
        if self.latent_space_id and not self.latent_space_id.strip():
            raise ValueError("latent_space_id cannot contain only whitespace")
        for name in ("source_reference_sha256", "latent_transform_sha256"):
            digest = getattr(self, name)
            if not isinstance(digest, str):
                raise TypeError("%s must be a string" % name)
            if digest and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("%s must be a lowercase SHA-256 digest" % name)
        donors = tuple(str(value).strip() for value in self.training_donors)
        if any(not value for value in donors) or len(set(donors)) != len(donors):
            raise ValueError("training_donors must contain unique non-empty identifiers")

        object.__setattr__(self, "type_names", _readonly(names, np.dtype(np.str_)))
        object.__setattr__(self, "residual_type_basis", basis)
        object.__setattr__(self, "residual_type_max_norm", bounds)
        object.__setattr__(self, "effective_ranks", ranks)
        object.__setattr__(self, "n_cells", cell_counts)
        object.__setattr__(self, "n_prototypes", prototype_counts)
        object.__setattr__(self, "state_scales", states)
        object.__setattr__(self, "covariance_scales", covariances)
        object.__setattr__(self, "residual_scales", residuals)
        object.__setattr__(self, "scale_sources", sources)
        object.__setattr__(self, "scale_priority", priority)
        object.__setattr__(self, "training_donors", donors)

    @property
    def latent_dim(self) -> int:
        return int(self.residual_type_basis.shape[1])

    @property
    def rank(self) -> int:
        return int(self.residual_type_basis.shape[2])

    def model_parameters(
        self,
        cell_type_names: Optional[Sequence[object]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(basis, bounds)`` aligned to a model's cell-type order.

        The returned arrays are writable copies.  Pass the basis to
        ``torch.as_tensor`` and copy it into ``model.residual_type_basis``.
        Models with per-type gates can consume the second array directly.
        Omitting ``cell_type_names`` preserves the artifact's stored order.
        """

        if cell_type_names is None:
            indices = np.arange(len(self.type_names), dtype=np.int64)
        else:
            requested = _labels(cell_type_names, "cell_type_names")
            if len(set(requested.tolist())) != len(requested):
                raise ValueError("cell_type_names must be unique")
            lookup = {name: index for index, name in enumerate(self.type_names.tolist())}
            missing = [name for name in requested.tolist() if name not in lookup]
            if missing:
                raise ValueError("residual geometry is missing types: %s" % ", ".join(missing))
            indices = np.asarray([lookup[name] for name in requested.tolist()], dtype=np.int64)
        return (
            self.residual_type_basis[indices].copy(),
            self.residual_type_max_norm[indices].copy(),
        )

    def to_npz(self, path: PathLike) -> None:
        """Save the frozen geometry without pickle-dependent fields."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            schema=np.asarray(self.SCHEMA, dtype=np.str_),
            type_names=self.type_names,
            residual_type_basis=self.residual_type_basis,
            residual_type_max_norm=self.residual_type_max_norm,
            effective_ranks=self.effective_ranks,
            n_cells=self.n_cells,
            n_prototypes=self.n_prototypes,
            state_scales=self.state_scales,
            covariance_scales=self.covariance_scales,
            residual_scales=self.residual_scales,
            scale_sources=self.scale_sources,
            calibration_quantile=np.asarray(self.calibration_quantile, dtype=np.float64),
            bound_fraction=np.asarray(self.bound_fraction, dtype=np.float64),
            minimum_bound=np.asarray(self.minimum_bound, dtype=np.float64),
            maximum_bound=np.asarray(
                np.nan if self.maximum_bound is None else self.maximum_bound,
                dtype=np.float64,
            ),
            scale_priority=np.asarray(self.scale_priority, dtype=np.str_),
            latent_space_id=np.asarray(self.latent_space_id, dtype=np.str_),
            source_reference_sha256=np.asarray(self.source_reference_sha256, dtype=np.str_),
            training_donors=np.asarray(self.training_donors, dtype=np.str_),
            latent_transform_sha256=np.asarray(self.latent_transform_sha256, dtype=np.str_),
        )

    @classmethod
    def from_npz(cls, path: PathLike) -> "RNAResidualGeometry":
        """Load and fully validate an RNA residual-geometry artifact."""

        with np.load(path, allow_pickle=False) as values:
            if "schema" not in values or str(np.asarray(values["schema"]).item()) != cls.SCHEMA:
                raise ValueError("not an %s artifact" % cls.SCHEMA)
            maximum = float(np.asarray(values["maximum_bound"]).item())
            return cls(
                type_names=values["type_names"],
                residual_type_basis=values["residual_type_basis"],
                residual_type_max_norm=values["residual_type_max_norm"],
                effective_ranks=values["effective_ranks"],
                n_cells=values["n_cells"],
                n_prototypes=values["n_prototypes"],
                state_scales=values["state_scales"],
                covariance_scales=values["covariance_scales"],
                residual_scales=values["residual_scales"],
                scale_sources=values["scale_sources"],
                calibration_quantile=float(np.asarray(values["calibration_quantile"]).item()),
                bound_fraction=float(np.asarray(values["bound_fraction"]).item()),
                minimum_bound=float(np.asarray(values["minimum_bound"]).item()),
                maximum_bound=None if np.isnan(maximum) else maximum,
                scale_priority=tuple(str(value) for value in values["scale_priority"].tolist()),
                latent_space_id=str(np.asarray(values["latent_space_id"]).item()),
                source_reference_sha256=str(np.asarray(values["source_reference_sha256"]).item()),
                training_donors=tuple(str(value) for value in values["training_donors"].tolist()),
                latent_transform_sha256=str(np.asarray(values["latent_transform_sha256"]).item()),
            )


def fit_rna_residual_geometry(
    latents: np.ndarray,
    cell_type_labels: Sequence[object],
    rank: int,
    *,
    type_names: Optional[Sequence[object]] = None,
    prototype_means: Optional[np.ndarray] = None,
    prototype_labels: Optional[Sequence[object]] = None,
    prototype_variances: Optional[np.ndarray] = None,
    calibration_quantile: float = 0.9,
    bound_fraction: float = 0.5,
    minimum_bound: float = 1.0e-3,
    maximum_bound: Optional[float] = None,
    minimum_calibration_cells: int = 3,
    scale_priority: Sequence[str] = _SCALE_NAMES,
    rank_tolerance: float = 1.0e-7,
    latent_space_id: str = "",
    source_reference_sha256: str = "",
    training_donors: Sequence[object] = (),
    latent_transform_sha256: str = "",
) -> RNAResidualGeometry:
    """Fit deterministic RNA PCA bases and molecularly calibrated bounds.

    Parameters
    ----------
    latents, cell_type_labels:
        Aligned RNA-cell latent representations and fine cell-type labels.
    rank:
        Number of residual directions expected by the image model.
    type_names:
        Optional authoritative model order.  With no explicit order, the union
        of cell and prototype labels is sorted lexicographically.
    prototype_means, prototype_labels, prototype_variances:
        Optional measured state bank.  Cells are residualized against their
        nearest same-type mean.  State separation is the median within-type
        pairwise mean distance.  Diagonal variances are converted to a
        trace-matched Gaussian radial quantile in the fitted subspace.
    scale_priority:
        Ordered subset of ``("state", "covariance", "residual")`` used to
        choose each type's scale.  Empirical pooled within-type residuals and
        finally ``minimum_bound`` are automatic rare-type fallbacks.

    The selected molecular scale is multiplied by ``bound_fraction``.  The
    default of one half places a state-geometry bound at the midpoint between
    typical same-type prototypes, reducing the chance that an image residual
    crosses a measured molecular state.  All calculations use float64 and no
    randomized decomposition; rows and states are sorted before factorization
    and PCA signs are canonicalized for reproducible artifacts.
    """

    latent = np.asarray(latents, dtype=np.float64)
    labels = _labels(cell_type_labels, "cell_type_labels")
    if latent.ndim != 2 or latent.shape[0] != len(labels) or latent.shape[0] == 0:
        raise ValueError("latents and cell_type_labels must contain aligned, non-empty rows")
    if latent.shape[1] == 0 or not np.isfinite(latent).all():
        raise ValueError("latents must contain finite features")
    if not isinstance(rank, (int, np.integer)) or rank <= 0 or rank > latent.shape[1]:
        raise ValueError("rank must be in [1, latent_dim]")
    if not 0 < calibration_quantile < 1:
        raise ValueError("calibration_quantile must be between zero and one")
    if not np.isfinite(bound_fraction) or bound_fraction <= 0:
        raise ValueError("bound_fraction must be finite and positive")
    if not np.isfinite(minimum_bound) or minimum_bound <= 0:
        raise ValueError("minimum_bound must be finite and positive")
    if maximum_bound is not None and (
        not np.isfinite(maximum_bound) or maximum_bound < minimum_bound
    ):
        raise ValueError("maximum_bound must be finite and at least minimum_bound")
    if minimum_calibration_cells < 2:
        raise ValueError("minimum_calibration_cells must be at least two")
    if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError("rank_tolerance must be between zero and one")
    priority = tuple(str(value).strip().lower() for value in scale_priority)
    if not priority or len(set(priority)) != len(priority) or set(priority) - set(_SCALE_NAMES):
        raise ValueError("scale_priority must contain unique state/covariance/residual entries")

    supplied_prototypes = prototype_means is not None or prototype_labels is not None
    if supplied_prototypes:
        if prototype_means is None or prototype_labels is None:
            raise ValueError("prototype_means and prototype_labels must be supplied together")
        means = np.asarray(prototype_means, dtype=np.float64)
        mean_labels = _labels(prototype_labels, "prototype_labels")
        if means.ndim != 2 or means.shape != (len(mean_labels), latent.shape[1]):
            raise ValueError("prototype means must align with labels and latent_dim")
        if not np.isfinite(means).all():
            raise ValueError("prototype means must be finite")
    else:
        means = np.empty((0, latent.shape[1]), dtype=np.float64)
        mean_labels = np.empty(0, dtype=np.str_)

    if prototype_variances is not None:
        if not supplied_prototypes:
            raise ValueError("prototype_variances require prototype means and labels")
        variances = np.asarray(prototype_variances, dtype=np.float64)
        if variances.shape != means.shape:
            raise ValueError("prototype variances must have the same shape as means")
        if not np.isfinite(variances).all() or np.any(variances < 0):
            raise ValueError("prototype variances must be finite and non-negative")
    else:
        variances = np.empty((means.shape[0], latent.shape[1]), dtype=np.float64)

    observed_names = set(labels.tolist()) | set(mean_labels.tolist())
    if type_names is None:
        names = np.asarray(sorted(observed_names), dtype=np.str_)
    else:
        names = _labels(type_names, "type_names")
        if len(set(names.tolist())) != len(names):
            raise ValueError("type_names must be unique")
        missing_names = sorted(observed_names - set(names.tolist()))
        if missing_names:
            raise ValueError("type_names omit observed labels: %s" % ", ".join(missing_names))

    residual_by_type = []
    centered_by_type = []
    ordered_means = []
    ordered_variances = []
    cell_counts = []
    prototype_counts = []
    for type_name in names.tolist():
        selected, _ = _sorted_rows(latent[labels == type_name])
        type_means, prototype_order = _sorted_rows(means[mean_labels == type_name])
        if prototype_variances is None:
            type_variances = np.empty((type_means.shape[0], latent.shape[1]), dtype=np.float64)
        else:
            type_variances = variances[mean_labels == type_name][prototype_order]
        if selected.shape[0] == 0:
            residual = np.empty((0, latent.shape[1]), dtype=np.float64)
        elif type_means.shape[0] > 0:
            squared_distance = np.square(selected[:, None, :] - type_means[None, :, :]).sum(axis=2)
            assignments = np.argmin(squared_distance, axis=1)
            residual = selected - type_means[assignments]
        else:
            residual = selected - selected.mean(axis=0, keepdims=True)
        centered = (
            residual - residual.mean(axis=0, keepdims=True)
            if residual.shape[0]
            else residual.copy()
        )
        residual_by_type.append(residual)
        centered_by_type.append(centered)
        ordered_means.append(type_means)
        ordered_variances.append(type_variances)
        cell_counts.append(selected.shape[0])
        prototype_counts.append(type_means.shape[0])

    pooled_rows = [values for values in centered_by_type if values.shape[0] > 0]
    pooled_residual = (
        np.concatenate(pooled_rows, axis=0)
        if pooled_rows
        else np.empty((0, latent.shape[1]), dtype=np.float64)
    )
    pooled_components = _principal_components(pooled_residual, rank_tolerance)
    all_raw_rows = [values for values in residual_by_type if values.shape[0] > 0]
    raw_pooled = (
        np.concatenate(all_raw_rows, axis=0)
        if all_raw_rows
        else np.empty((0, latent.shape[1]), dtype=np.float64)
    )
    pooled_scale = (
        float(np.quantile(np.linalg.norm(raw_pooled, axis=1), calibration_quantile))
        if raw_pooled.shape[0] >= minimum_calibration_cells
        else 0.0
    )

    bases = []
    effective_ranks = []
    state_scales = []
    covariance_scales = []
    residual_scales = []
    bounds = []
    sources = []
    for residual, type_means, type_variances in zip(
        residual_by_type,
        ordered_means,
        ordered_variances,
    ):
        local_components = _principal_components(residual, rank_tolerance)
        effective_rank = min(rank, local_components.shape[1])
        basis = _complete_basis(
            local_components[:, :effective_rank],
            pooled_components,
            latent.shape[1],
            rank,
            rank_tolerance,
        )
        state = _state_scale(type_means)
        covariance = (
            _covariance_scale(type_variances, basis, calibration_quantile)
            if prototype_variances is not None
            else 0.0
        )
        empirical = (
            float(np.quantile(np.linalg.norm(residual, axis=1), calibration_quantile))
            if residual.shape[0] >= minimum_calibration_cells
            else 0.0
        )
        component_scales = {
            "state": state,
            "covariance": covariance,
            "residual": empirical,
        }
        source = ""
        selected_scale = 0.0
        for candidate in priority:
            if component_scales[candidate] > rank_tolerance:
                source = candidate
                selected_scale = component_scales[candidate]
                break
        if not source and pooled_scale > rank_tolerance:
            source = "pooled_residual"
            selected_scale = pooled_scale
        if not source:
            source = "minimum"
        bound = max(minimum_bound, bound_fraction * selected_scale)
        if maximum_bound is not None:
            bound = min(bound, maximum_bound)

        bases.append(basis)
        effective_ranks.append(effective_rank)
        state_scales.append(state)
        covariance_scales.append(covariance)
        residual_scales.append(empirical)
        bounds.append(bound)
        sources.append(source)

    return RNAResidualGeometry(
        type_names=names,
        residual_type_basis=np.stack(bases).astype(np.float32),
        residual_type_max_norm=np.asarray(bounds, dtype=np.float32),
        effective_ranks=np.asarray(effective_ranks, dtype=np.int64),
        n_cells=np.asarray(cell_counts, dtype=np.int64),
        n_prototypes=np.asarray(prototype_counts, dtype=np.int64),
        state_scales=np.asarray(state_scales, dtype=np.float32),
        covariance_scales=np.asarray(covariance_scales, dtype=np.float32),
        residual_scales=np.asarray(residual_scales, dtype=np.float32),
        scale_sources=np.asarray(sources, dtype=np.str_),
        calibration_quantile=calibration_quantile,
        bound_fraction=bound_fraction,
        minimum_bound=minimum_bound,
        maximum_bound=maximum_bound,
        scale_priority=priority,
        latent_space_id=latent_space_id,
        source_reference_sha256=source_reference_sha256,
        training_donors=tuple(str(value) for value in training_donors),
        latent_transform_sha256=latent_transform_sha256,
    )
