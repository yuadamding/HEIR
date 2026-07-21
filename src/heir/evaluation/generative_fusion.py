"""Compact probabilistic primitives for the HEIR development experiment.

The module intentionally contains models and statistics, not a cohort runner.  Every
fitted object records the observation identifiers it saw, and raw-count entry points
reject non-finite, negative, or non-integral values.  NumPy owns deterministic data
preparation; Torch owns differentiable models and transparently follows CPU/CUDA
devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _counts(value: object, name: str = "counts") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain finite non-negative integer counts")
    return result


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _ids(value: Sequence[object], name: str, rows: int, *, unique: bool = False) -> np.ndarray:
    result = np.asarray([str(item) for item in value])
    if result.ndim != 1 or len(result) != rows or np.any(result == ""):
        raise ValueError(f"{name} must contain {rows} non-empty identifiers")
    if unique and len(set(result.tolist())) != rows:
        raise ValueError(f"{name} must be unique")
    return result


def _digest(counts: np.ndarray, identifiers: np.ndarray) -> str:
    canonical = np.ascontiguousarray(counts, dtype="<f8")
    payload = "\n".join(identifiers.tolist()).encode() + b"\0" + canonical.tobytes()
    return sha256(payload).hexdigest()


def _torch_counts(value: object, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError("counts must be a non-empty two-dimensional tensor")
    tensor = tensor.to(dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()) or not bool((tensor >= 0).all()):
        raise ValueError("counts must be finite and non-negative")
    if not bool((tensor == torch.floor(tensor)).all()):
        raise ValueError("counts must be integral")
    return tensor


@dataclass(frozen=True)
class NB2Dispersion:
    """Training-only gene-wise NB2 inverse dispersion."""

    theta: np.ndarray
    training_observation_ids: tuple[str, ...]
    training_donor_ids: tuple[str, ...]
    training_sha256: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.theta.shape

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return np.asarray(self.theta, dtype=dtype)

    def assert_excludes(self, heldout_observation_ids: Sequence[object]) -> None:
        overlap = set(self.training_observation_ids) & {
            str(value) for value in heldout_observation_ids
        }
        if overlap:
            raise ValueError(f"dispersion provenance overlaps held-out outcomes: {sorted(overlap)}")


def fit_nb2_dispersion(
    counts: object,
    training_observation_ids: Sequence[object] | None = None,
    *,
    training_donor_ids: Sequence[object] | None = None,
    library_size: object | None = None,
    minimum_theta: float = 1.0e-3,
    maximum_theta: float = 1.0e6,
) -> NB2Dispersion:
    """Fit exposure-aware gene-wise NB2 dispersion on training rows only.

    The fitted mean is ``library_size * rate``.  When donor identifiers are
    supplied, rates are estimated within donor before pooling residual moments;
    this prevents library depth and between-donor mean shifts from being counted
    as NB2 overdispersion.
    """

    values = _counts(counts)
    observations = _ids(
        [f"training-row-{index}" for index in range(len(values))]
        if training_observation_ids is None
        else training_observation_ids,
        "training_observation_ids",
        len(values),
        unique=True,
    )
    exposure = (
        np.ones(len(values), dtype=np.float64)
        if library_size is None
        else np.asarray(library_size, dtype=np.float64)
    )
    if (
        exposure.shape != (len(values),)
        or np.any(~np.isfinite(exposure))
        or np.any(exposure <= 0)
    ):
        raise ValueError("library_size must be finite and positive per training row")
    donors = (
        np.asarray([], dtype=str)
        if training_donor_ids is None
        else _ids(training_donor_ids, "training_donor_ids", len(values))
    )
    if len(values) < 2:
        raise ValueError("at least two training observations are required")
    lower, upper = float(minimum_theta), float(maximum_theta)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < upper):
        raise ValueError("dispersion bounds must be finite and ordered")
    groups = donors if len(donors) else np.full(len(values), "all", dtype=str)
    fitted_mean = np.zeros_like(values, dtype=np.float64)
    for donor in sorted(set(groups.tolist())):
        local = groups == donor
        rate = values[local].sum(axis=0) / exposure[local].sum()
        fitted_mean[local] = exposure[local, None] * rate[None]
    numerator = np.sum(np.square(fitted_mean), axis=0, dtype=np.float64)
    excess = np.sum(
        np.square(values - fitted_mean) - values,
        axis=0,
        dtype=np.float64,
    )
    theta = np.full(values.shape[1], upper, dtype=np.float64)
    estimable = (numerator > 0) & (excess > 0)
    theta[estimable] = numerator[estimable] / excess[estimable]
    theta = np.clip(theta, lower, upper)
    theta.setflags(write=False)
    provenance = sha256()
    provenance.update(_digest(values, observations).encode())
    provenance.update(np.ascontiguousarray(exposure, dtype="<f8").tobytes())
    provenance.update("\n".join(groups.tolist()).encode())
    return NB2Dispersion(
        theta=theta,
        training_observation_ids=tuple(observations.tolist()),
        training_donor_ids=tuple(sorted(set(donors.tolist()))),
        training_sha256=provenance.hexdigest(),
    )


def _as_like(value: object, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, NB2Dispersion):
        value = value.theta.copy()
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def nb2_log_likelihood(
    counts: object,
    mean: object,
    theta: object,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    """NB2 log likelihood with ``Var(Y)=mean+mean**2/theta``."""

    mu = torch.as_tensor(mean)
    if not mu.is_floating_point():
        mu = mu.to(torch.float32)
    y = _as_like(counts, mu)
    dispersion = _as_like(theta, mu)
    if y.shape != mu.shape or dispersion.ndim not in (0, 1, 2):
        raise ValueError("counts/mean shapes or theta rank are invalid")
    if (
        not bool(torch.isfinite(y).all())
        or not bool((y >= 0).all())
        or not bool((y == torch.floor(y)).all())
    ):
        raise ValueError("counts must be finite non-negative integers")
    if not bool(torch.isfinite(mu).all()) or not bool((mu > 0).all()):
        raise ValueError("NB2 means must be finite and positive")
    if not bool(torch.isfinite(dispersion).all()) or not bool((dispersion > 0).all()):
        raise ValueError("NB2 dispersion must be finite and positive")
    log_prob = (
        torch.lgamma(y + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(y + 1)
        + dispersion * (torch.log(dispersion) - torch.log(dispersion + mu))
        + y * (torch.log(mu) - torch.log(dispersion + mu))
    )
    if reduction == "none":
        return log_prob
    if reduction == "sum":
        return log_prob.sum()
    if reduction == "mean":
        return log_prob.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


nb2_log_prob = nb2_log_likelihood


def nb2_deviance(
    counts: object,
    mean: object,
    theta: object,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """NB2 likelihood deviance relative to the saturated mean."""

    mu = torch.as_tensor(mean)
    if not mu.is_floating_point():
        mu = mu.to(torch.float32)
    y = _as_like(counts, mu)
    saturated = torch.where(y > 0, y, torch.full_like(y, torch.finfo(mu.dtype).eps))
    deviance = 2 * (
        nb2_log_likelihood(y, saturated, theta, reduction="none")
        - nb2_log_likelihood(y, mu, theta, reduction="none")
    )
    deviance = torch.clamp(deviance, min=0)
    if reduction == "none":
        return deviance
    if reduction == "sum":
        return deviance.sum()
    if reduction == "mean":
        return deviance.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


@dataclass(frozen=True)
class CountSplitDispersion:
    concentration: np.ndarray
    fit: NB2Dispersion

    def assert_excludes(self, heldout_observation_ids: Sequence[object]) -> None:
        self.fit.assert_excludes(heldout_observation_ids)


@dataclass(frozen=True)
class CountSplit:
    first: np.ndarray
    second: np.ndarray
    fraction: float
    seed: int
    dispersion_sha256: str

    def __iter__(self):
        yield self.first
        yield self.second


def fit_count_split_dispersion(
    training_counts: object,
    training_observation_ids: Sequence[object],
    *,
    training_donor_ids: Sequence[object] | None = None,
) -> CountSplitDispersion:
    """Fit the beta-binomial/Dirichlet-multinomial concentration on training counts."""

    fit = fit_nb2_dispersion(
        training_counts,
        training_observation_ids,
        training_donor_ids=training_donor_ids,
    )
    concentration = np.asarray(fit.theta, dtype=np.float64).copy()
    concentration.setflags(write=False)
    return CountSplitDispersion(concentration=concentration, fit=fit)


def split_nb2_counts(
    counts: object,
    dispersion: CountSplitDispersion | NB2Dispersion | object,
    *,
    fraction: float = 0.5,
    seed: int = 17,
) -> CountSplit:
    """Two-way negative-binomial count split using a beta-binomial draw.

    A beta-binomial is the two-category Dirichlet-multinomial.  Its concentration
    is explicitly fitted on training observations and is never re-estimated here.
    """

    values = _counts(counts)
    probability = float(fraction)
    if not 0 < probability < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    if isinstance(dispersion, CountSplitDispersion):
        concentration = np.asarray(dispersion.concentration, dtype=np.float64)
        provenance = dispersion.fit.training_sha256
    elif isinstance(dispersion, NB2Dispersion):
        concentration = np.asarray(dispersion.theta, dtype=np.float64)
        provenance = dispersion.training_sha256
    else:
        concentration = np.asarray(dispersion, dtype=np.float64)
        provenance = sha256(np.ascontiguousarray(concentration, dtype="<f8").tobytes()).hexdigest()
    if concentration.shape != (values.shape[1],) or np.any(concentration <= 0):
        raise ValueError("split dispersion differs from the count gene panel")
    rng = np.random.default_rng(int(seed))
    alpha = np.broadcast_to(probability * concentration, values.shape)
    beta = np.broadcast_to((1 - probability) * concentration, values.shape)
    latent_probability = rng.beta(alpha, beta)
    first = rng.binomial(values.astype(np.int64), latent_probability).astype(np.int64)
    second = values.astype(np.int64) - first
    return CountSplit(
        first=first,
        second=second,
        fraction=probability,
        seed=int(seed),
        dispersion_sha256=provenance,
    )


@dataclass(frozen=True)
class VAEOutput:
    mean_counts: torch.Tensor
    latent_mean: torch.Tensor
    latent_log_variance: torch.Tensor
    latent: torch.Tensor
    theta: torch.Tensor
    modality: str


@dataclass(frozen=True)
class CrossAssayAlignment:
    """Training-only donor-pseudobulk latent-alignment receipt."""

    donor_ids: tuple[str, ...]
    # Backwards-compatible names for the raw matched-donor MSE.
    pre_mse: float
    post_mse: float
    weight: float
    pre_mismatched_mse: float
    post_mismatched_mse: float
    pre_matched_to_mismatched_ratio: float
    post_matched_to_mismatched_ratio: float
    pre_separation: float
    post_separation: float
    optimizer_applications_per_epoch: int
    optimizer_applications_total: int
    matched_pairs_closer_post: bool
    alignment_improved: bool
    support_criterion_met: bool
    support_criterion: str = (
        "positive_weight_and_every_minibatch_and_post_matched_to_mismatched_ratio"
        "_less_than_one_and_less_than_pre"
    )

    @property
    def pre_matched_mse(self) -> float:
        """Raw pre-fit matched-donor MSE (explicit alias for ``pre_mse``)."""

        return self.pre_mse

    @property
    def post_matched_mse(self) -> float:
        """Raw post-fit matched-donor MSE (explicit alias for ``post_mse``)."""

        return self.post_mse


@dataclass(frozen=True)
class DecoderMoments:
    """Moments after integrating a decoder over a diagonal-Gaussian latent."""

    mean_counts: torch.Tensor
    latent_variance_counts: torch.Tensor
    predictive_variance_counts: torch.Tensor
    theta: torch.Tensor
    gene_indices: torch.Tensor
    samples: int
    modality: str
    dispersion_source: str


def deterministic_diagonal_gaussian_samples(
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    samples: int = 32,
    seed: int = 17,
) -> torch.Tensor:
    """Return deterministic Sobol samples with leading sample axis.

    A tensor-product Gaussian quadrature is infeasible at the preregistered
    20-dimensional latent.  Scrambled Sobol normal draws provide a bounded,
    reproducible integration rule instead.  The same rule is shared across
    rows, which reduces Monte-Carlo noise in paired model comparisons.
    """

    if mean.ndim != 2 or not len(mean) or variance.shape != mean.shape:
        raise ValueError("mean and variance must be matching row-by-latent tensors")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(variance).all()):
        raise ValueError("Gaussian parameters must be finite")
    if not bool((variance >= 0).all()) or int(samples) < 2:
        raise ValueError("variance must be non-negative and samples must be at least two")
    engine = torch.quasirandom.SobolEngine(
        dimension=mean.shape[1], scramble=True, seed=int(seed)
    )
    uniform = engine.draw(int(samples), dtype=mean.dtype).to(mean.device)
    epsilon = torch.finfo(mean.dtype).eps
    normal = torch.erfinv(2 * uniform.clamp(epsilon, 1 - epsilon) - 1) * np.sqrt(2.0)
    return mean[None] + torch.sqrt(variance)[None] * normal[:, None]


class CountVAE(nn.Module):
    """Shallow 20-dimensional count VAE with separate snRNA and ST decoders."""

    modalities = ("scrna", "st")

    def __init__(
        self,
        n_genes: int,
        *,
        latent_dim: int = 20,
        hidden_dim: int = 64,
        seed: int = 17,
    ) -> None:
        super().__init__()
        if min(int(n_genes), int(latent_dim), int(hidden_dim)) <= 0:
            raise ValueError("model dimensions must be positive")
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.encoder = nn.Sequential(
                nn.Linear(self.n_genes + 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * latent_dim),
            )
            self.decoders = nn.ModuleDict(
                {
                    modality: nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, self.n_genes),
                    )
                    for modality in self.modalities
                }
            )
            self.log_theta = nn.ParameterDict(
                {modality: nn.Parameter(torch.zeros(self.n_genes)) for modality in self.modalities}
            )
        self.register_buffer("panel_fraction", torch.ones(2))
        self.training_observation_ids: tuple[str, ...] = ()
        self.training_sha256: str | None = None
        self.alignment_diagnostics: CrossAssayAlignment | None = None

    def _modality(self, modality: str) -> int:
        if modality == "snrna":
            modality = "scrna"
        if modality not in self.modalities:
            raise ValueError(f"modality must be one of {self.modalities}")
        return self.modalities.index(modality)

    @staticmethod
    def _canonical_modality(modality: str) -> str:
        return "scrna" if modality == "snrna" else modality

    def bind_training_provenance(
        self,
        observation_ids: Sequence[object],
        *,
        heldout_observation_ids: Sequence[object] = (),
    ) -> None:
        identifiers = _ids(observation_ids, "observation_ids", len(observation_ids), unique=True)
        heldout = {str(value) for value in heldout_observation_ids}
        overlap = set(identifiers.tolist()) & heldout
        if overlap:
            raise ValueError(f"training IDs overlap held-out outcomes: {sorted(overlap)}")
        if (
            self.training_observation_ids
            and tuple(identifiers.tolist()) != self.training_observation_ids
        ):
            raise RuntimeError("training provenance is immutable once bound")
        self.training_observation_ids = tuple(identifiers.tolist())
        self.training_sha256 = sha256("\n".join(self.training_observation_ids).encode()).hexdigest()

    def encode(self, counts: object, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        code = self._modality(modality)
        device = next(self.parameters()).device
        values = _torch_counts(counts, device=device)
        if values.shape[1] != self.n_genes:
            raise ValueError("counts differ from the frozen gene panel")
        library = values.sum(dim=1, keepdim=True).clamp_min(1)
        normalized = torch.log1p(values / library * 1.0e4)
        one_hot = F.one_hot(torch.full((len(values),), code, device=device), num_classes=2).to(
            normalized.dtype
        )
        encoded = self.encoder(torch.cat((normalized, one_hot), dim=1))
        return encoded.chunk(2, dim=1)

    def decode(
        self,
        latent: torch.Tensor,
        modality: str,
        library_size: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._modality(modality)
        modality = self._canonical_modality(modality)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent has the wrong shape")
        library = torch.as_tensor(library_size, dtype=latent.dtype, device=latent.device).reshape(
            -1, 1
        )
        if (
            len(library) != len(latent)
            or not bool(torch.isfinite(library).all())
            or not bool((library > 0).all())
        ):
            raise ValueError("library_size must be finite and positive per observation")
        proportions = torch.softmax(self.decoders[modality](latent), dim=1)
        theta = F.softplus(self.log_theta[modality]).clamp_min(1.0e-4)
        scale = self.panel_fraction[self.modalities.index(modality)].to(latent.dtype)
        return proportions * library * scale, theta

    def forward(
        self,
        counts: object,
        modality: str,
        *,
        sample: bool = False,
        seed: int = 17,
    ) -> VAEOutput:
        modality = self._canonical_modality(modality)
        values = _torch_counts(counts, device=next(self.parameters()).device)
        latent_mean, latent_log_variance = self.encode(values, modality)
        if sample:
            generator = torch.Generator(device=values.device)
            generator.manual_seed(int(seed))
            noise = torch.randn(
                latent_mean.shape,
                dtype=latent_mean.dtype,
                device=latent_mean.device,
                generator=generator,
            )
            latent = latent_mean + torch.exp(0.5 * latent_log_variance) * noise
        else:
            latent = latent_mean
        mean_counts, theta = self.decode(latent, modality, values.sum(dim=1).clamp_min(1))
        return VAEOutput(mean_counts, latent_mean, latent_log_variance, latent, theta, modality)

    def loss(self, counts: object, output: VAEOutput, *, kl_weight: float = 1.0) -> torch.Tensor:
        reconstruction = (
            -nb2_log_likelihood(
                counts, output.mean_counts.clamp_min(1.0e-8), output.theta, reduction="sum"
            )
            / output.mean_counts.shape[0]
        )
        kl = -0.5 * torch.mean(
            torch.sum(
                1
                + output.latent_log_variance
                - output.latent_mean.square()
                - output.latent_log_variance.exp(),
                dim=1,
            )
        )
        return reconstruction + float(kl_weight) * kl

    @staticmethod
    def _training_donor_pseudobulks(
        counts: np.ndarray,
        modality: np.ndarray,
        donor_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Aggregate row-aligned training assays for donors observed in both."""

        matched = tuple(
            sorted(
                set(donor_ids[modality == "st"].tolist())
                & set(donor_ids[modality == "scrna"].tolist())
            )
        )
        if len(matched) < 2:
            raise ValueError(
                "cross-assay alignment requires at least two donors represented in both assays"
            )
        st = np.vstack(
            [np.sum(counts[(donor_ids == donor) & (modality == "st")], axis=0) for donor in matched]
        )
        scrna = np.vstack(
            [
                np.sum(counts[(donor_ids == donor) & (modality == "scrna")], axis=0)
                for donor in matched
            ]
        )
        return st, scrna, matched

    @staticmethod
    def _alignment_terms(
        st_mean: torch.Tensor, scrna_mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return matched, mismatched, ratio, and separation alignment terms.

        The off-diagonal donor pairs provide a fold-local latent scale.  Their
        distance is detached in the optimization ratio: the loss pulls matched
        assays together without rewarding arbitrary expansion of donor-to-donor
        distances.  The resulting coefficient is dimensionless and remains
        interpretable when the encoder changes its absolute latent scale.
        """

        if (
            st_mean.ndim != 2
            or scrna_mean.shape != st_mean.shape
            or len(st_mean) < 2
        ):
            raise ValueError("alignment means must contain at least two paired donors")
        pairwise = torch.mean((st_mean[:, None] - scrna_mean[None]) ** 2, dim=2)
        diagonal = torch.eye(len(st_mean), dtype=torch.bool, device=st_mean.device)
        matched = torch.mean(pairwise.diagonal())
        mismatched = torch.mean(pairwise[~diagonal])
        epsilon = torch.finfo(pairwise.dtype).eps
        ratio = matched / mismatched.detach().clamp_min(epsilon)
        return matched, mismatched, ratio, 1.0 - ratio

    @torch.no_grad()
    def _alignment_statistics(
        self, st: np.ndarray, scrna: np.ndarray
    ) -> tuple[float, float, float, float]:
        self.eval()
        st_mean = self.encode(st, "st")[0]
        scrna_mean = self.encode(scrna, "scrna")[0]
        terms = self._alignment_terms(st_mean, scrna_mean)
        return tuple(float(value.detach().cpu()) for value in terms)

    @torch.no_grad()
    def _alignment_mse(self, st: np.ndarray, scrna: np.ndarray) -> float:
        """Return the legacy raw matched-donor MSE diagnostic."""

        return self._alignment_statistics(st, scrna)[0]

    def fit_model(
        self,
        counts: object,
        *,
        modality: Sequence[object],
        training_donor_ids: Sequence[object] | None = None,
        alignment_weight: float = 1.0,
        library_size: object | None = None,
        observation_ids: Sequence[object] | None = None,
        heldout_observation_ids: Sequence[object] = (),
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1.0e-3,
        seed: int = 17,
    ) -> "CountVAE":
        """Fit on explicitly supplied molecular training rows only.

        When ``training_donor_ids`` is supplied, same-donor ST and snRNA
        training pseudobulks receive an explicit latent-coordinate penalty.
        This aligns assays without sharing their count decoders or using a
        held-out donor's ST outcome.  Only donors represented in both training
        assays enter the penalty; the pre/post discrepancy is retained in
        :attr:`alignment_diagnostics`.
        """

        values = _counts(counts)
        labels = np.asarray(modality)
        if labels.shape != (len(values),):
            raise ValueError("modality must identify every training row")
        canonical = np.asarray(
            ["st" if str(value).lower() in ("1", "st") else "scrna" for value in labels]
        )
        unknown = {str(value).lower() for value in labels} - {"0", "1", "st", "scrna", "snrna"}
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        if not np.isfinite(alignment_weight) or float(alignment_weight) < 0:
            raise ValueError("alignment_weight must be finite and non-negative")
        alignment = None
        if training_donor_ids is not None:
            donors = _ids(training_donor_ids, "training_donor_ids", len(values))
            alignment = self._training_donor_pseudobulks(values, canonical, donors)
        identifiers = (
            [f"molecular-training-row-{index}" for index in range(len(values))]
            if observation_ids is None
            else observation_ids
        )
        self.bind_training_provenance(identifiers, heldout_observation_ids=heldout_observation_ids)
        if library_size is None:
            exposures = np.sum(values, axis=1)
        else:
            exposures = np.asarray(library_size, dtype=np.float64)
        if exposures.shape != (len(values),) or np.any(exposures <= 0):
            raise ValueError("library_size must be positive per training row")
        with torch.no_grad():
            for code, name in enumerate(self.modalities):
                local = canonical == name
                if np.any(local):
                    fraction = np.median(np.sum(values[local], axis=1) / exposures[local])
                    if not np.isfinite(fraction) or not 0 < fraction <= 1.0 + 1.0e-6:
                        raise ValueError(
                            "panel counts cannot exceed registered full-library exposure"
                        )
                    self.panel_fraction[code] = min(float(fraction), 1.0)
        if int(epochs) < 1 or int(batch_size) < 1:
            raise ValueError("epochs and batch_size must be positive")
        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        rng = np.random.default_rng(int(seed))
        device = next(self.parameters()).device
        pre_alignment = None if alignment is None else self._alignment_statistics(*alignment[:2])
        alignment_applications = 0
        minibatches_per_epoch = (len(values) + int(batch_size) - 1) // int(batch_size)
        self.train()
        for _ in range(int(epochs)):
            order = rng.permutation(len(values))
            for start in range(0, len(values), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                optimizer.zero_grad(set_to_none=True)
                objective = torch.zeros((), device=device)
                for name in self.modalities:
                    local = indices[canonical[indices] == name]
                    if not len(local):
                        continue
                    output = self.forward(values[local], name, sample=True, seed=int(seed) + start)
                    # Decode with the registered exposure rather than silently using depth.
                    mean, theta = self.decode(output.latent, name, exposures[local])
                    output = VAEOutput(
                        mean,
                        output.latent_mean,
                        output.latent_log_variance,
                        output.latent,
                        theta,
                        name,
                    )
                    objective = objective + self.loss(values[local], output)
                # The ratio to mismatched training-donor pairs is a
                # scale-normalized, dimensionless global regularizer.  Apply it
                # to every stochastic update so its declared coefficient is not
                # silently diluted by the number of molecular minibatches.
                if alignment is not None and float(alignment_weight) > 0:
                    st_pseudobulk, scrna_pseudobulk, _ = alignment
                    st_mean = self.encode(st_pseudobulk, "st")[0]
                    scrna_mean = self.encode(scrna_pseudobulk, "scrna")[0]
                    normalized_alignment = self._alignment_terms(st_mean, scrna_mean)[2]
                    objective = objective + float(alignment_weight) * normalized_alignment
                    alignment_applications += 1
                objective.backward()
                optimizer.step()
        self.eval()
        if alignment is None:
            self.alignment_diagnostics = None
        else:
            post_alignment = self._alignment_statistics(*alignment[:2])
            applications_per_epoch = (
                alignment_applications // int(epochs) if float(alignment_weight) > 0 else 0
            )
            if float(alignment_weight) > 0 and (
                alignment_applications != int(epochs) * minibatches_per_epoch
                or applications_per_epoch != minibatches_per_epoch
            ):
                raise RuntimeError("alignment was not applied to every molecular minibatch")
            matched_pairs_closer = post_alignment[2] < 1.0
            alignment_improved = post_alignment[2] < pre_alignment[2]
            support = bool(
                float(alignment_weight) > 0
                and applications_per_epoch == minibatches_per_epoch
                and matched_pairs_closer
                and alignment_improved
            )
            self.alignment_diagnostics = CrossAssayAlignment(
                donor_ids=alignment[2],
                pre_mse=pre_alignment[0],
                post_mse=post_alignment[0],
                weight=float(alignment_weight),
                pre_mismatched_mse=pre_alignment[1],
                post_mismatched_mse=post_alignment[1],
                pre_matched_to_mismatched_ratio=pre_alignment[2],
                post_matched_to_mismatched_ratio=post_alignment[2],
                pre_separation=pre_alignment[3],
                post_separation=post_alignment[3],
                optimizer_applications_per_epoch=applications_per_epoch,
                optimizer_applications_total=alignment_applications,
                matched_pairs_closer_post=matched_pairs_closer,
                alignment_improved=alignment_improved,
                support_criterion_met=support,
            )
        return self

    @torch.no_grad()
    def encode_numpy(self, counts: object, *, modality: str) -> np.ndarray:
        self.eval()
        return self.encode(counts, modality)[0].detach().cpu().numpy()

    @torch.no_grad()
    def decode_numpy(self, latent: object, *, library_size: object, modality: str) -> np.ndarray:
        self.eval()
        tensor = torch.as_tensor(latent, dtype=torch.float32, device=next(self.parameters()).device)
        return self.decode(tensor, modality, library_size)[0].detach().cpu().numpy()

    @torch.no_grad()
    def decode_diagonal_gaussian(
        self,
        latent_mean: object,
        latent_variance: object,
        *,
        library_size: object,
        modality: str,
        dispersion: object | None = None,
        endpoint_gene_indices: Sequence[int] | None = None,
        samples: int = 32,
        seed: int = 17,
        batch_size: int = 128,
    ) -> DecoderMoments:
        """Integrate the count decoder over a latent posterior.

        ``mean_counts`` estimates :math:`E[D(z)]`, rather than evaluating
        :math:`D(E[z])`. ``predictive_variance_counts`` additionally includes
        the conditional NB2 observation variance.  Pass the frozen primary
        scoring dispersion through ``dispersion`` to avoid using the VAE's
        nuisance reconstruction dispersion for endpoint intervals.  An
        ``endpoint_gene_indices`` subset lets an augmented decoder integrate
        only the frozen endpoint panel (for example, excluding an ``other``
        transcript bin). Row batching bounds the temporary
        ``samples x rows x genes`` tensor on CPU or CUDA.
        """

        self.eval()
        modality = self._canonical_modality(modality)
        self._modality(modality)
        device = next(self.parameters()).device
        mean = torch.as_tensor(latent_mean, dtype=torch.float32, device=device)
        variance = torch.as_tensor(latent_variance, dtype=torch.float32, device=device)
        if (
            mean.ndim != 2
            or not len(mean)
            or mean.shape[1] != self.latent_dim
            or variance.shape != mean.shape
        ):
            raise ValueError("latent Gaussian has the wrong shape")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        library = torch.as_tensor(library_size, dtype=mean.dtype, device=device).reshape(-1)
        if (
            len(library) != len(mean)
            or not bool(torch.isfinite(library).all())
            or not bool((library > 0).all())
        ):
            raise ValueError("library_size must be finite and positive per observation")
        if endpoint_gene_indices is None:
            gene_indices = torch.arange(self.n_genes, device=device)
        else:
            gene_indices = torch.as_tensor(endpoint_gene_indices, dtype=torch.int64, device=device)
            if (
                gene_indices.ndim != 1
                or not len(gene_indices)
                or len(torch.unique(gene_indices)) != len(gene_indices)
                or not bool(((gene_indices >= 0) & (gene_indices < self.n_genes)).all())
            ):
                raise ValueError("endpoint_gene_indices must be unique valid decoder columns")
        theta_override = None
        if dispersion is not None:
            theta_override = torch.as_tensor(dispersion, dtype=mean.dtype, device=device)
            if (
                theta_override.shape != (len(gene_indices),)
                or not bool(torch.isfinite(theta_override).all())
                or not bool((theta_override > 0).all())
            ):
                raise ValueError("dispersion must be a positive value for every decoder gene")
        means, latent_variances, predictive_variances = [], [], []
        used_theta = None
        for start in range(0, len(mean), int(batch_size)):
            stop = min(start + int(batch_size), len(mean))
            draws = deterministic_diagonal_gaussian_samples(
                mean[start:stop],
                variance[start:stop],
                samples=int(samples),
                seed=int(seed),
            )
            sample_count, rows, _ = draws.shape
            decoded, theta = self.decode(
                draws.reshape(sample_count * rows, self.latent_dim),
                modality,
                library[start:stop].repeat(sample_count),
            )
            decoded = decoded[:, gene_indices]
            theta = theta[gene_indices]
            if theta_override is not None:
                theta = theta_override
            used_theta = theta
            decoded = decoded.reshape(sample_count, rows, len(gene_indices))
            local_mean = decoded.mean(dim=0)
            latent_component = decoded.var(dim=0, unbiased=False)
            conditional_component = (
                decoded + decoded.square() / theta[None, None]
            ).mean(dim=0)
            means.append(local_mean)
            latent_variances.append(latent_component)
            predictive_variances.append(latent_component + conditional_component)
        return DecoderMoments(
            mean_counts=torch.cat(means),
            latent_variance_counts=torch.cat(latent_variances),
            predictive_variance_counts=torch.cat(predictive_variances),
            theta=used_theta,
            gene_indices=gene_indices,
            samples=int(samples),
            modality=modality,
            dispersion_source="provided_primary" if theta_override is not None else "vae_learned",
        )

    @torch.no_grad()
    def decode_diagonal_gaussian_numpy(self, *args, **kwargs) -> dict[str, np.ndarray | int | str]:
        """NumPy view of :meth:`decode_diagonal_gaussian`."""

        moments = self.decode_diagonal_gaussian(*args, **kwargs)
        return {
            "mean_counts": moments.mean_counts.cpu().numpy(),
            "latent_variance_counts": moments.latent_variance_counts.cpu().numpy(),
            "predictive_variance_counts": moments.predictive_variance_counts.cpu().numpy(),
            "theta": moments.theta.cpu().numpy(),
            "gene_indices": moments.gene_indices.cpu().numpy(),
            "samples": moments.samples,
            "modality": moments.modality,
            "dispersion_source": moments.dispersion_source,
        }


@dataclass(frozen=True)
class ReferenceMixture:
    """Natural-weight donor/type diagonal-Gaussian molecular-state mixture."""

    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    donor_ids: np.ndarray
    type_labels: np.ndarray
    component_ids: np.ndarray
    source_observation_ids: tuple[str, ...]
    source_donor_ids: tuple[str, ...]
    source_type_labels: tuple[str, ...]
    source_modality: str
    source_sha256: str

    @property
    def latent_dim(self) -> int:
        return int(self.means.shape[1])

    def component_indices(
        self,
        donor_id: object,
        type_label: object,
        *,
        allow_missing: bool = False,
    ) -> np.ndarray:
        indices = np.flatnonzero(
            (self.donor_ids == str(donor_id)) & (self.type_labels == str(type_label))
        )
        if not len(indices) and not allow_missing:
            raise ValueError(f"reference lacks donor/type {donor_id!s}/{type_label!s}")
        return indices

    def assert_no_outcome_overlap(self, outcome_ids: Sequence[object]) -> None:
        overlap = set(self.source_observation_ids) & {str(value) for value in outcome_ids}
        if overlap:
            raise ValueError(f"reference contains spatial outcomes: {sorted(overlap)}")

    @property
    def type_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.type_labels.tolist())))

    def type_weights(self, *, donor_equal: bool = False) -> np.ndarray:
        donors = np.asarray(self.source_donor_ids)
        types = np.asarray(self.source_type_labels)
        if not donor_equal:
            weights = np.asarray([np.mean(types == label) for label in self.type_names])
        else:
            unique_donors = sorted(set(donors.tolist()))
            weights = np.mean(
                [
                    [np.mean(types[donors == donor] == label) for label in self.type_names]
                    for donor in unique_donors
                ],
                axis=0,
            )
        return weights / np.sum(weights)

    def type_means(self, *, donor_equal: bool = False) -> np.ndarray:
        output = []
        source_donors = np.asarray(self.source_donor_ids)
        source_types = np.asarray(self.source_type_labels)
        for label in self.type_names:
            group_means, group_weights = [], []
            for donor in sorted(set(self.donor_ids[self.type_labels == label].tolist())):
                indices = self.component_indices(donor, label)
                local_weight = self.weights[indices] / np.sum(self.weights[indices])
                group_means.append(np.sum(local_weight[:, None] * self.means[indices], axis=0))
                group_weights.append(
                    1.0
                    if donor_equal
                    else np.sum((source_donors == donor) & (source_types == label))
                )
            output.append(np.average(np.vstack(group_means), axis=0, weights=group_weights))
        return np.vstack(output)

    def effective_sample_size(self, *, donor_id: object | None = None) -> float:
        mask = np.ones(len(self.weights), dtype=bool)
        if donor_id is not None:
            mask = self.donor_ids == str(donor_id)
            if not np.any(mask):
                raise ValueError(f"reference lacks donor {donor_id!s}")
        weights = self.weights[mask] / np.sum(self.weights[mask])
        return float(1.0 / np.sum(np.square(weights)))


def _stable_seed(identifier: str, seed: int) -> int:
    return int.from_bytes(sha256(f"{seed}:{identifier}".encode()).digest()[:8], "little")


def _soft_components(
    values: np.ndarray,
    observation_ids: np.ndarray,
    components: int,
    *,
    seed: int,
    iterations: int,
    temperature: float,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(observation_ids, kind="stable")
    values, observation_ids = values[order], observation_ids[order]
    first = min(
        range(len(values)),
        key=lambda i: (_stable_seed(observation_ids[i], seed), observation_ids[i]),
    )
    centers = [values[first].copy()]
    while len(centers) < components:
        distance = np.min(
            np.sum((values[:, None, :] - np.vstack(centers)[None, :, :]) ** 2, axis=2), axis=1
        )
        chosen = min(
            np.flatnonzero(distance == np.max(distance)).tolist(), key=lambda i: observation_ids[i]
        )
        centers.append(values[chosen].copy())
    means = np.vstack(centers)
    scale = max(float(np.median(np.sum((values - np.mean(values, axis=0)) ** 2, axis=1))), 1.0e-8)
    for _ in range(iterations):
        logits = -np.sum((values[:, None, :] - means[None, :, :]) ** 2, axis=2) / (
            2 * temperature * scale
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        responsibilities = np.exp(logits)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        mass = np.sum(responsibilities, axis=0)
        if np.any(mass <= np.finfo(np.float64).tiny):
            raise RuntimeError("soft reference mixture produced an empty component")
        means = (responsibilities.T @ values) / mass[:, None]
    residual = values[:, None, :] - means[None, :, :]
    variances = np.sum(responsibilities[:, :, None] * residual**2, axis=0) / mass[:, None]
    variances = np.maximum(variances, variance_floor)
    return means, variances, mass / np.sum(mass)


def build_reference_mixture(
    latent: object,
    donor_ids: Sequence[object] | None = None,
    type_labels: Sequence[object] | None = None,
    observation_ids: Sequence[object] | None = None,
    *,
    components_per_type: int = 4,
    type_ids: Sequence[object] | None = None,
    n_components: int | None = None,
    donor_equal: bool = False,
    seed: int = 17,
    iterations: int = 25,
    temperature: float = 1.0,
    variance_floor: float = 1.0e-4,
    source_modality: str = "snrna",
) -> ReferenceMixture:
    """Build deterministic soft state-aware donor/type mixtures with natural weights."""

    values = _matrix(latent, "reference latent")
    if donor_ids is None:
        donor_ids = ["reference"] * len(values)
    if type_labels is not None and type_ids is not None:
        raise ValueError("provide type_labels or type_ids, not both")
    if type_labels is None:
        type_labels = type_ids
    if type_labels is None:
        raise ValueError("type labels are required")
    if n_components is not None:
        if components_per_type != 4:
            raise ValueError("provide components_per_type or n_components, not both")
        components_per_type = n_components
    donors = _ids(donor_ids, "donor_ids", len(values))
    types = _ids(type_labels, "type_labels", len(values))
    observations = _ids(
        [f"snrna-reference-row-{index}" for index in range(len(values))]
        if observation_ids is None
        else observation_ids,
        "observation_ids",
        len(values),
        unique=True,
    )
    if source_modality not in {"snrna", "single_cell"}:
        raise ValueError("reference mixtures must be single-cell/nuclear, never held-out ST")
    if int(components_per_type) != components_per_type or int(components_per_type) < 1:
        raise ValueError("components_per_type must be a positive integer")
    if int(iterations) < 1 or not np.isfinite(temperature) or float(temperature) <= 0:
        raise ValueError("iterations and temperature must be positive")
    means, variances, weights, out_donors, out_types, component_ids = [], [], [], [], [], []
    for donor in sorted(set(donors.tolist())):
        for type_label in sorted(set(types[donors == donor].tolist())):
            indices = np.flatnonzero((donors == donor) & (types == type_label))
            count = min(int(components_per_type), len(indices))
            local = _soft_components(
                values[indices],
                observations[indices],
                count,
                seed=_stable_seed(f"{donor}:{type_label}", int(seed)),
                iterations=int(iterations),
                temperature=float(temperature),
                variance_floor=float(variance_floor),
            )
            for component in range(count):
                means.append(local[0][component])
                variances.append(local[1][component])
                weights.append(local[2][component])
                out_donors.append(donor)
                out_types.append(type_label)
                component_ids.append(f"{donor}::{type_label}::soft_state::{component}")
    result = ReferenceMixture(
        means=np.asarray(means),
        variances=np.asarray(variances),
        weights=np.asarray(weights),
        donor_ids=np.asarray(out_donors),
        type_labels=np.asarray(out_types),
        component_ids=np.asarray(component_ids),
        source_observation_ids=tuple(observations.tolist()),
        source_donor_ids=tuple(donors.tolist()),
        source_type_labels=tuple(types.tolist()),
        source_modality=source_modality,
        source_sha256=_digest(values, observations),
    )
    # ``donor_equal`` is a view-time weighting choice.  Accepting it here keeps the
    # M7 runner explicit without mutating the natural primary bank.
    if donor_equal:
        result.type_weights(donor_equal=True)
    return result


def diagonal_gaussian_mixture_poe(
    image_mean: torch.Tensor,
    image_variance: torch.Tensor,
    reference_means: torch.Tensor,
    reference_variances: torch.Tensor,
    reference_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic PoE between one diagonal Gaussian and a Gaussian mixture."""

    if image_mean.ndim != 1 or image_variance.shape != image_mean.shape:
        raise ValueError("image Gaussian must be one-dimensional")
    if reference_means.ndim != 2 or reference_means.shape != reference_variances.shape:
        raise ValueError("reference Gaussian arrays have invalid shapes")
    if reference_means.shape[1] != len(image_mean) or reference_weights.shape != (
        len(reference_means),
    ):
        raise ValueError("image/reference dimensions differ")
    if len(reference_means) < 2:
        raise ValueError("M3 requires a multi-component reference, not one centroid")
    if not bool((image_variance > 0).all()) or not bool((reference_variances > 0).all()):
        raise ValueError("Gaussian variances must be positive")
    if not bool((reference_weights > 0).all()):
        raise ValueError("natural reference weights must be positive")
    precision = image_variance.reciprocal()[None, :] + reference_variances.reciprocal()
    component_variance = precision.reciprocal()
    component_mean = component_variance * (
        image_mean[None, :] / image_variance[None, :] + reference_means / reference_variances
    )
    overlap_variance = image_variance[None, :] + reference_variances
    overlap = -0.5 * torch.sum(
        torch.log(2 * torch.pi * overlap_variance)
        + (image_mean[None, :] - reference_means).square() / overlap_variance,
        dim=1,
    )
    posterior_weights = torch.softmax(torch.log(reference_weights) + overlap, dim=0)
    mean = torch.sum(posterior_weights[:, None] * component_mean, dim=0)
    second = torch.sum(
        posterior_weights[:, None] * (component_variance + component_mean.square()), dim=0
    )
    return mean, torch.clamp(second - mean.square(), min=1.0e-8), posterior_weights


@dataclass(frozen=True)
class CompositionStateOutput:
    mean_counts: torch.Tensor
    composition: torch.Tensor
    state_mean: torch.Tensor
    state_variance: torch.Tensor
    reference_entropy: torch.Tensor
    mode: str
    reference_supported: torch.Tensor | None = None


class CompositionStateModel(nn.Module):
    """H&E composition/type-state model with one shared M0/M3 count decoder."""

    def __init__(
        self,
        image_dim: int,
        type_labels: Sequence[object] | None = None,
        n_genes: int | None = None,
        *,
        latent_dim: int = 20,
        hidden_dim: int = 64,
        n_types: int | None = None,
        seed: int = 17,
    ) -> None:
        super().__init__()
        if type_labels is None:
            if n_types is None or int(n_types) < 1:
                raise ValueError("provide type_labels or a positive n_types")
            type_labels = [f"type_{index}" for index in range(int(n_types))]
        if n_types is not None and int(n_types) != len(type_labels):
            raise ValueError("n_types differs from type_labels")
        labels = tuple(str(value) for value in type_labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("type_labels must be non-empty and unique")
        self.type_labels = labels
        if n_genes is None:
            n_genes = latent_dim
        self.image_dim, self.n_genes, self.latent_dim = (
            int(image_dim),
            int(n_genes),
            int(latent_dim),
        )
        if min(self.image_dim, self.n_genes, self.latent_dim, int(hidden_dim)) <= 0:
            raise ValueError("model dimensions must be positive")
        types = len(labels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.composition = nn.Sequential(
                nn.Linear(self.image_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, types)
            )
            self.state = nn.Sequential(
                nn.Linear(self.image_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, types * self.latent_dim),
            )
            # This exact decoder instance is used by both modes.
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim + types, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_genes),
            )
        # Per-type diagonal uncertainty is fitted after the state means by
        # maximizing the training residual likelihood.  Buffers survive a
        # state_dict round-trip and cannot be silently altered by prediction.
        self.register_buffer("calibrated_type_variance", torch.ones(types, self.latent_dim))
        self.register_buffer("variance_calibration_nll", torch.full((2,), float("nan")))
        self.register_buffer("variance_calibration_rows", torch.zeros((), dtype=torch.int64))

    def set_trainable_stage(self, stage: str) -> None:
        """Freeze all but one preregistered stage (composition, state, or decoder)."""

        modules = {"composition": self.composition, "state": self.state, "decoder": self.decoder}
        if stage not in (*modules, "frozen", "all"):
            raise ValueError("unknown training stage")
        for name, module in modules.items():
            enabled = stage == "all" or stage == name
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)

    def _image_state(
        self, image_features: object
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        image = torch.as_tensor(image_features, dtype=torch.float32, device=device)
        if (
            image.ndim != 2
            or image.shape[1] != self.image_dim
            or not bool(torch.isfinite(image).all())
        ):
            raise ValueError("image_features have invalid shape or values")
        rows, types = len(image), len(self.type_labels)
        composition = torch.softmax(self.composition(image), dim=1)
        state_mean = self.state(image).reshape(rows, types, self.latent_dim)
        state_variance = self.calibrated_type_variance[None].expand(rows, -1, -1)
        return composition, state_mean, state_variance

    @torch.no_grad()
    def predict_composition(self, image_features: object) -> np.ndarray:
        self.eval()
        return self._image_state(image_features)[0].cpu().numpy()

    def variance_calibration_receipt(self) -> dict[str, object]:
        """Return the persisted training-residual uncertainty calibration."""

        return {
            "method": "Gaussian_training_residual_likelihood",
            "rows": int(self.variance_calibration_rows.detach().cpu()),
            "nll_before": float(self.variance_calibration_nll[0].detach().cpu()),
            "nll_after": float(self.variance_calibration_nll[1].detach().cpu()),
            "per_type_variance": self.calibrated_type_variance.detach().cpu().numpy().copy(),
            "type_labels": self.type_labels,
        }

    def _calibrate_state_variance(
        self,
        image: np.ndarray,
        target: np.ndarray,
        *,
        batch_size: int,
        steps: int,
    ) -> None:
        """Fit constant per-type variances from aggregate training residual NLL."""

        device = next(self.parameters()).device
        compositions, squared_residuals = [], []
        self.eval()
        with torch.no_grad():
            for start in range(0, len(image), int(batch_size)):
                stop = min(start + int(batch_size), len(image))
                local_composition, local_state, _ = self._image_state(image[start:stop])
                expected = torch.sum(local_composition[:, :, None] * local_state, dim=1)
                local_target = torch.as_tensor(target[start:stop], device=device)
                compositions.append(local_composition.detach().cpu().numpy())
                squared_residuals.append((local_target - expected).square().cpu().numpy())
        composition = np.vstack(compositions).astype(np.float32, copy=False)
        residual2 = np.vstack(squared_residuals).astype(np.float32, copy=False)
        floor = 1.0e-6
        global_variance = np.maximum(np.mean(residual2, axis=0), floor)
        composition_scale = max(float(np.mean(np.sum(composition**2, axis=1))), floor)
        initial = np.broadcast_to(
            global_variance[None] / composition_scale,
            (len(self.type_labels), self.latent_dim),
        ).copy()

        def numpy_nll(variance: np.ndarray) -> float:
            aggregate = np.maximum(composition**2 @ variance, floor)
            return float(0.5 * np.mean(np.log(aggregate) + residual2 / aggregate))

        best_variance = initial
        before = numpy_nll(initial)
        best_nll = before
        raw_initial = np.empty_like(initial)
        large = initial > 20
        raw_initial[large] = initial[large]
        raw_initial[~large] = np.log(np.expm1(np.maximum(initial[~large], floor)))
        raw = nn.Parameter(torch.as_tensor(raw_initial, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam((raw,), lr=5.0e-2)
        total = float(residual2.size)
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(composition), int(batch_size)):
                stop = min(start + int(batch_size), len(composition))
                local_composition = torch.as_tensor(composition[start:stop], device=device)
                local_residual2 = torch.as_tensor(residual2[start:stop], device=device)
                variance = F.softplus(raw) + floor
                aggregate = local_composition.square() @ variance
                objective = 0.5 * torch.sum(
                    torch.log(aggregate.clamp_min(floor))
                    + local_residual2 / aggregate.clamp_min(floor)
                ) / total
                objective.backward()
            optimizer.step()
            candidate = (F.softplus(raw) + floor).detach().cpu().numpy()
            candidate_nll = numpy_nll(candidate)
            if candidate_nll < best_nll:
                best_nll, best_variance = candidate_nll, candidate.copy()
        with torch.no_grad():
            self.calibrated_type_variance.copy_(
                torch.as_tensor(best_variance, device=self.calibrated_type_variance.device)
            )
            self.variance_calibration_nll.copy_(
                torch.as_tensor([before, best_nll], device=self.variance_calibration_nll.device)
            )
            self.variance_calibration_rows.fill_(len(image))

    def fit_model(
        self,
        image_features: object,
        target_latent: object,
        *,
        composition_targets: object,
        type_ids: Sequence[object] | None = None,
        type_anchor_means: object | None = None,
        observation_ids: Sequence[object] | None = None,
        heldout_observation_ids: Sequence[object] = (),
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1.0e-3,
        seed: int = 17,
    ) -> "CompositionStateModel":
        """Fit composition then state on training rows; no reference enters fitting."""

        image = _matrix(image_features, "training image features").astype(np.float32)
        target = _matrix(target_latent, "training target latent").astype(np.float32)
        composition = _matrix(composition_targets, "composition targets").astype(np.float32)
        if len(image) != len(target) or target.shape[1] != self.latent_dim:
            raise ValueError("image and target latent differ")
        if composition.shape != (len(image), len(self.type_labels)) or np.any(composition < 0):
            raise ValueError("composition targets have invalid shape or values")
        row_sum = composition.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("composition targets must have positive row mass")
        composition /= row_sum
        if type_ids is not None:
            labels = tuple(str(value) for value in type_ids)
            if len(labels) != len(self.type_labels) or len(set(labels)) != len(labels):
                raise ValueError("fit type IDs differ from model capacity")
            self.type_labels = labels
        anchors = None
        if type_anchor_means is not None:
            anchors = _matrix(type_anchor_means, "type anchor means").astype(np.float32)
            if anchors.shape != (len(self.type_labels), self.latent_dim):
                raise ValueError("type anchor means have invalid shape")
        identifiers = _ids(
            [f"image-training-row-{index}" for index in range(len(image))]
            if observation_ids is None
            else observation_ids,
            "observation_ids",
            len(image),
            unique=True,
        )
        overlap = set(identifiers.tolist()) & {str(value) for value in heldout_observation_ids}
        if overlap:
            raise ValueError(f"image training IDs overlap held-out outcomes: {sorted(overlap)}")
        if int(epochs) < 1 or int(batch_size) < 1:
            raise ValueError("epochs and batch_size must be positive")
        rng = np.random.default_rng(int(seed))
        device = next(self.parameters()).device
        self.train()
        # Stage B: learn the disclosed training-ST composition proxy, then
        # freeze it so composition and continuous state cannot trade signal.
        self.set_trainable_stage("composition")
        optimizer = torch.optim.Adam(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=float(learning_rate),
        )
        for _ in range(max(1, int(epochs) // 2)):
            order = rng.permutation(len(image))
            for start in range(0, len(image), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                optimizer.zero_grad(set_to_none=True)
                local_composition = self._image_state(image[indices])[0]
                composition_tensor = torch.as_tensor(composition[indices], device=device)
                objective = -torch.mean(
                    torch.sum(
                        composition_tensor * torch.log(local_composition.clamp_min(1e-8)), dim=1
                    )
                )
                objective.backward()
                optimizer.step()
        # Stage C: freeze composition and learn per-type states.  snRNA type
        # means anchor otherwise weakly identified Visium mixture components.
        self.set_trainable_stage("state")
        optimizer = torch.optim.Adam(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=float(learning_rate),
        )
        anchor_tensor = None if anchors is None else torch.as_tensor(anchors, device=device)
        for _ in range(int(epochs)):
            order = rng.permutation(len(image))
            for start in range(0, len(image), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    local_composition = self._image_state(image[indices])[0]
                _, local_state, _ = self._image_state(image[indices])
                expected = torch.sum(local_composition[:, :, None] * local_state, dim=1)
                target_tensor = torch.as_tensor(target[indices], device=device)
                objective = F.mse_loss(expected, target_tensor)
                if anchor_tensor is not None:
                    objective = objective + 0.10 * torch.mean(
                        local_composition[:, :, None]
                        * (local_state - anchor_tensor[None]).square()
                    )
                objective.backward()
                optimizer.step()
        # Stage D: means and composition are now frozen.  Estimate diagonal
        # per-type uncertainty solely from the likelihood of training latent
        # residuals; no KL-only variance head enters prediction.
        self.set_trainable_stage("frozen")
        self._calibrate_state_variance(
            image,
            target,
            batch_size=int(batch_size),
            steps=max(25, min(100, 2 * int(epochs))),
        )
        self.eval()
        return self

    def _legacy_details(
        self,
        image_features: object,
        *,
        reference: ReferenceMixture | None,
        mode: str,
        missing_type_policy: str,
    ) -> dict[str, torch.Tensor]:
        composition, state_mean, state_variance = self._image_state(image_features)
        rows, types = composition.shape
        entropy = torch.zeros_like(composition)
        supported = torch.zeros((rows, types), dtype=torch.bool, device=composition.device)
        if missing_type_policy not in ("image_only", "error"):
            raise ValueError("missing_type_policy must be image_only or error")
        if mode in ("full_poe", "M3"):
            if reference is None or reference.latent_dim != self.latent_dim:
                raise ValueError("full PoE requires a dimension-matched reference")
            fused_mean, fused_variance = state_mean.clone(), state_variance.clone()
            for type_index, type_label in enumerate(self.type_labels):
                indices = np.flatnonzero(reference.type_labels == type_label)
                if len(indices) < 2 and missing_type_policy == "image_only":
                    continue
                if len(indices) < 2:
                    raise ValueError("full PoE requires multiple components per type")
                supported[:, type_index] = True
                ref_mean = torch.as_tensor(
                    reference.means[indices], dtype=state_mean.dtype, device=state_mean.device
                )
                ref_variance = torch.as_tensor(
                    reference.variances[indices], dtype=state_mean.dtype, device=state_mean.device
                )
                ref_weight = torch.as_tensor(
                    reference.weights[indices], dtype=state_mean.dtype, device=state_mean.device
                )
                ref_weight = ref_weight / ref_weight.sum()
                for row in range(rows):
                    local = diagonal_gaussian_mixture_poe(
                        state_mean[row, type_index],
                        state_variance[row, type_index],
                        ref_mean,
                        ref_variance,
                        ref_weight,
                    )
                    fused_mean[row, type_index], fused_variance[row, type_index] = local[:2]
                    entropy[row, type_index] = -torch.sum(
                        local[2] * torch.log(local[2].clamp_min(1e-12))
                    )
            state_mean, state_variance = fused_mean, fused_variance
        elif mode == "composition_reference_mean":
            if reference is None:
                raise ValueError("composition routing requires a reference")
            routed_mean, routed_variance = state_mean.clone(), state_variance.clone()
            for type_index, type_label in enumerate(self.type_labels):
                indices = np.flatnonzero(reference.type_labels == type_label)
                if not len(indices):
                    if missing_type_policy == "error":
                        raise ValueError(f"reference lacks type {type_label}")
                    continue
                local_weight = reference.weights[indices] / np.sum(reference.weights[indices])
                routed_mean[:, type_index] = torch.as_tensor(
                    np.sum(local_weight[:, None] * reference.means[indices], axis=0),
                    dtype=state_mean.dtype,
                    device=state_mean.device,
                )
                routed_variance[:, type_index] = 0
                supported[:, type_index] = True
            state_mean, state_variance = routed_mean, routed_variance
        elif mode not in ("image_only", "M0"):
            raise ValueError("unknown prediction mode")
        latent = torch.sum(composition[:, :, None] * state_mean, dim=1)
        # The spot latent is a weighted sum of independent type-state
        # posteriors, hence weights enter its uncertainty quadratically.  Keep
        # the random-cell/type-mixture dispersion separately to prevent callers
        # from confusing it with uncertainty of the aggregate spot mean.
        variance = torch.sum(composition[:, :, None].square() * state_variance, dim=1)
        type_mixture_variance = torch.sum(
            composition[:, :, None] * (state_variance + (state_mean - latent[:, None, :]).square()),
            dim=1,
        )
        return {
            "latent": latent,
            "variance": variance,
            "type_mixture_variance": type_mixture_variance,
            "composition": composition,
            "type_mean": state_mean,
            "type_variance": state_variance,
            "reference_entropy": entropy,
            "reference_supported": supported,
        }

    @torch.no_grad()
    def predict_details_numpy(
        self,
        image_features: object,
        *,
        reference: ReferenceMixture | None = None,
        mode: str = "image_only",
        missing_type_policy: str = "image_only",
    ) -> dict[str, np.ndarray]:
        self.eval()
        return {
            name: value.detach().cpu().numpy()
            for name, value in self._legacy_details(
                image_features,
                reference=reference,
                mode=mode,
                missing_type_policy=missing_type_policy,
            ).items()
        }

    def predict_numpy(
        self,
        image_features: object,
        *,
        reference: ReferenceMixture | None = None,
        mode: str = "image_only",
        missing_type_policy: str = "image_only",
    ) -> np.ndarray:
        return self.predict_details_numpy(
            image_features,
            reference=reference,
            mode=mode,
            missing_type_policy=missing_type_policy,
        )["latent"]

    def predict(
        self,
        image_features: object,
        *,
        mode: str,
        library_size: object,
        donor_ids: Sequence[object] | None = None,
        reference: ReferenceMixture | None = None,
        missing_type_policy: str = "image_only",
    ) -> CompositionStateOutput:
        if mode not in ("M0", "M3"):
            raise ValueError("mode must be M0 or M3")
        device = next(self.parameters()).device
        image = torch.as_tensor(image_features, dtype=torch.float32, device=device)
        rows, types = len(image), len(self.type_labels)
        composition, state_mean, state_variance = self._image_state(image)
        entropy = torch.zeros((rows, types), dtype=image.dtype, device=device)
        supported = torch.zeros((rows, types), dtype=torch.bool, device=device)
        if missing_type_policy not in ("image_only", "error"):
            raise ValueError("missing_type_policy must be image_only or error")
        if mode == "M3":
            if (
                reference is None
                or donor_ids is None
                or reference.source_modality not in {"snrna", "single_cell"}
            ):
                raise ValueError("M3 requires donor_ids and a single-cell/nuclear ReferenceMixture")
            donors = _ids(donor_ids, "donor_ids", rows)
            if reference.latent_dim != self.latent_dim:
                raise ValueError("reference latent dimension differs from the model")
            fused_mean, fused_variance = state_mean.clone(), state_variance.clone()
            for row, donor in enumerate(donors.tolist()):
                for type_index, type_label in enumerate(self.type_labels):
                    indices = reference.component_indices(
                        donor,
                        type_label,
                        allow_missing=missing_type_policy == "image_only",
                    )
                    if len(indices) < 2 and missing_type_policy == "image_only":
                        continue
                    supported[row, type_index] = True
                    reference_mean = torch.as_tensor(
                        reference.means[indices], dtype=image.dtype, device=device
                    )
                    reference_variance = torch.as_tensor(
                        reference.variances[indices], dtype=image.dtype, device=device
                    )
                    natural_weight = torch.as_tensor(
                        reference.weights[indices], dtype=image.dtype, device=device
                    )
                    local = diagonal_gaussian_mixture_poe(
                        state_mean[row, type_index],
                        state_variance[row, type_index],
                        reference_mean,
                        reference_variance,
                        natural_weight,
                    )
                    fused_mean[row, type_index], fused_variance[row, type_index] = local[:2]
                    entropy[row, type_index] = -torch.sum(
                        local[2] * torch.log(local[2].clamp_min(1e-12))
                    )
            state_mean, state_variance = fused_mean, fused_variance
        one_hot = torch.eye(types, dtype=image.dtype, device=device)[None].expand(rows, -1, -1)
        type_profiles = torch.softmax(self.decoder(torch.cat((state_mean, one_hot), dim=2)), dim=2)
        profile = torch.sum(composition[:, :, None] * type_profiles, dim=1)
        library = torch.as_tensor(library_size, dtype=image.dtype, device=device).reshape(-1, 1)
        if (
            len(library) != rows
            or not bool(torch.isfinite(library).all())
            or not bool((library > 0).all())
        ):
            raise ValueError("library_size must be finite and positive per observation")
        return CompositionStateOutput(
            profile * library,
            composition,
            state_mean,
            state_variance,
            entropy,
            mode,
            supported,
        )

    forward = predict


@dataclass(frozen=True)
class RetrievalOutput:
    retrieved_latent: torch.Tensor
    attention: torch.Tensor


class ContrastiveRetrieval(nn.Module):
    """BLEEP-style joint projection and soft molecular-bank retrieval primitive."""

    def __init__(
        self,
        image_dim: int,
        molecular_dim: int,
        *,
        projection_dim: int = 32,
        embedding_dim: int | None = None,
        seed: int = 17,
    ):
        super().__init__()
        if embedding_dim is not None:
            if projection_dim != 32:
                raise ValueError("provide projection_dim or embedding_dim, not both")
            projection_dim = embedding_dim
        if min(int(image_dim), int(molecular_dim), int(projection_dim)) <= 0:
            raise ValueError("projection dimensions must be positive")
        self.molecular_dim = int(molecular_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.image_projection = nn.Linear(image_dim, projection_dim)
            self.molecular_projection = nn.Linear(molecular_dim, projection_dim)

    def encode_image(self, values: object) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=next(self.parameters()).device)
        return F.normalize(self.image_projection(tensor), dim=1)

    def encode_molecular(self, values: object) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=next(self.parameters()).device)
        return F.normalize(self.molecular_projection(tensor), dim=1)

    def contrastive_loss(
        self, image: object, molecular: object, *, temperature: float = 0.07
    ) -> torch.Tensor:
        left, right = self.encode_image(image), self.encode_molecular(molecular)
        if left.shape != right.shape or len(left) < 2 or float(temperature) <= 0:
            raise ValueError(
                "paired projections must have equal shape and at least two rows; "
                "temperature must be positive"
            )
        logits = left @ right.T / float(temperature)
        targets = torch.arange(len(left), device=left.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    def retrieve(
        self,
        image: object,
        molecular_bank: object,
        *,
        temperature: float = 0.07,
        natural_weights: object | None = None,
    ) -> RetrievalOutput:
        bank = torch.as_tensor(
            molecular_bank, dtype=torch.float32, device=next(self.parameters()).device
        )
        if bank.ndim != 2 or bank.shape[1] != self.molecular_dim or float(temperature) <= 0:
            raise ValueError("molecular bank shape or temperature is invalid")
        logits = self.encode_image(image) @ self.encode_molecular(bank).T / float(temperature)
        if natural_weights is not None:
            weights = torch.as_tensor(natural_weights, dtype=logits.dtype, device=logits.device)
            if weights.shape != (len(bank),) or not bool((weights > 0).all()):
                raise ValueError("natural_weights must be positive per bank state")
            logits = logits + torch.log(weights / weights.sum())[None]
        attention = torch.softmax(logits, dim=1)
        return RetrievalOutput(attention @ bank, attention)

    def fit_model(
        self,
        image: object,
        molecular: object,
        *,
        hard_negative_molecular: object | None = None,
        hard_negative_weight: float = 0.5,
        observation_ids: Sequence[object] | None = None,
        heldout_observation_ids: Sequence[object] = (),
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1.0e-3,
        temperature: float = 0.07,
        seed: int = 17,
    ) -> "ContrastiveRetrieval":
        image_values = _matrix(image, "training image features").astype(np.float32)
        molecular_values = _matrix(molecular, "training molecular latent").astype(np.float32)
        if len(image_values) != len(molecular_values) or len(image_values) < 2:
            raise ValueError("contrastive pairs must be aligned and contain at least two rows")
        hard_negative = None
        if hard_negative_molecular is not None:
            hard_negative = _matrix(
                hard_negative_molecular, "hard-negative molecular latent"
            ).astype(np.float32)
            if hard_negative.shape != molecular_values.shape:
                raise ValueError("hard negatives must align with the positive molecular pairs")
        if not np.isfinite(hard_negative_weight) or float(hard_negative_weight) < 0:
            raise ValueError("hard_negative_weight must be finite and non-negative")
        identifiers = _ids(
            [f"retrieval-training-row-{index}" for index in range(len(image_values))]
            if observation_ids is None
            else observation_ids,
            "observation_ids",
            len(image_values),
            unique=True,
        )
        overlap = set(identifiers.tolist()) & {str(value) for value in heldout_observation_ids}
        if overlap:
            raise ValueError(f"retrieval training IDs overlap held-out outcomes: {sorted(overlap)}")
        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        rng = np.random.default_rng(int(seed))
        self.train()
        for _ in range(int(epochs)):
            order = rng.permutation(len(image_values))
            for start in range(0, len(image_values), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                if len(indices) < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                objective = self.contrastive_loss(
                    image_values[indices], molecular_values[indices], temperature=temperature
                )
                if hard_negative is not None and float(hard_negative_weight) > 0:
                    image_embedding = self.encode_image(image_values[indices])
                    positive_embedding = self.encode_molecular(molecular_values[indices])
                    negative_embedding = self.encode_molecular(hard_negative[indices])
                    positive_similarity = torch.sum(
                        image_embedding * positive_embedding, dim=1
                    )
                    negative_similarity = torch.sum(
                        image_embedding * negative_embedding, dim=1
                    )
                    objective = objective + float(hard_negative_weight) * torch.mean(
                        F.softplus(
                            (negative_similarity - positive_similarity) / float(temperature)
                        )
                    )
                objective.backward()
                optimizer.step()
        self.eval()
        return self

    @torch.no_grad()
    def retrieve_numpy(
        self,
        image: object,
        molecular_bank: object,
        *,
        return_entropy: bool = False,
        batch_size: int = 1024,
        temperature: float = 0.07,
        natural_weights: object | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        values = _matrix(image, "retrieval image features")
        retrieved, entropies = [], []
        self.eval()
        for start in range(0, len(values), int(batch_size)):
            output = self.retrieve(
                values[start : start + int(batch_size)],
                molecular_bank,
                temperature=temperature,
                natural_weights=natural_weights,
            )
            retrieved.append(output.retrieved_latent.cpu().numpy())
            normalized = -torch.sum(
                output.attention * torch.log(output.attention.clamp_min(1e-12)), dim=1
            ) / np.log(max(2, output.attention.shape[1]))
            entropies.append(normalized.cpu().numpy())
        result = np.vstack(retrieved)
        return (result, np.concatenate(entropies)) if return_entropy else result


@dataclass(frozen=True)
class SignFlipResult:
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    positive_fraction: float
    donors: int

    def __getitem__(self, key: str):
        aliases = {"mean": "statistic", "n": "donors"}
        return getattr(self, aliases.get(key, key))


def exact_sign_flip_test(
    differences: Sequence[float], *, alternative: str = "greater", alpha: float = 0.05
) -> SignFlipResult:
    """Exact paired donor sign-flip test (positive differences favor the candidate)."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not 2 <= len(values) <= 20 or not np.all(np.isfinite(values)):
        raise ValueError("exact sign-flip requires 2--20 finite donor differences")
    if alternative not in ("greater", "less", "two-sided") or not 0 < float(alpha) < 1:
        raise ValueError("invalid alternative or alpha")
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), dtype=np.float64)
    distribution = np.mean(signs * values[None], axis=1)
    observed = float(np.mean(values))
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 8
    if alternative == "greater":
        p_value = float(np.mean(distribution >= observed - tolerance))
    elif alternative == "less":
        p_value = float(np.mean(distribution <= observed + tolerance))
    else:
        p_value = float(np.mean(np.abs(distribution) >= abs(observed) - tolerance))
    centered_distribution = np.mean(signs * (values - observed)[None], axis=1)
    low, high = np.quantile(centered_distribution, [alpha / 2, 1 - alpha / 2])
    return SignFlipResult(
        statistic=observed,
        p_value=p_value,
        confidence_interval=(float(observed - high), float(observed - low)),
        positive_fraction=float(np.mean(values > 0)),
        donors=len(values),
    )


def holm_adjust(
    p_values: Sequence[float] | Mapping[str, float],
) -> np.ndarray | dict[str, float]:
    keys = tuple(p_values) if isinstance(p_values, Mapping) else None
    if isinstance(p_values, Mapping):
        p_values = [p_values[key] for key in keys]
    values = np.asarray(p_values, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or np.any(~np.isfinite(values))
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("p-values must be a non-empty finite vector in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    if keys is not None:
        return {key: float(value) for key, value in zip(keys, adjusted.tolist())}
    return adjusted


@dataclass(frozen=True)
class GateDecision:
    name: str
    reached: bool
    passed: bool
    reasons: tuple[str, ...]
    p_values: Mapping[str, float]
    adjusted_p_values: Mapping[str, float]


@dataclass(frozen=True)
class OrderedGateDecision:
    gates: tuple[GateDecision, ...]
    central_hypothesis_supported: bool
    full_model_chain_supported: bool
    personalization_supported: bool
    molecular_headroom_detected: bool
    relative_gain: float
    positive_donor_fraction: float


def evaluate_ordered_gates(
    losses: Mapping[str, object],
    *,
    indication_ids: Sequence[object] | None = None,
    alpha: float = 0.05,
    minimum_relative_gain: float = 0.05,
    minimum_positive_fraction: float = 0.70,
    maximum_indication_reversal: float = 0.05,
) -> OrderedGateDecision | dict[str, object]:
    """Apply preregistered Gates 1--5; Gate 5 is reported but nonblocking."""

    if "comparisons" in losses:
        comparisons = losses["comparisons"]
        if not isinstance(comparisons, Mapping):
            raise ValueError("comparisons must be a mapping")
        ordered = (
            ("gate_1", ("M3_vs_M0",), None),
            ("gate_2", ("M3_vs_M1", "M3_vs_M4"), "holm_gate2"),
            ("gate_3", ("M3_vs_M2",), None),
            ("gate_4", ("M3_vs_M6", "M3_vs_M7"), "holm_gate4"),
        )
        stopped_at = None
        gate_payload = {}
        for name, comparison_names, family_name in ordered:
            if stopped_at is not None:
                gate_payload[name] = {"reached": False, "passed": False}
                continue
            raw = {
                comparison: float(comparisons[comparison]["sign_flip"]["p_value"])
                for comparison in comparison_names
            }
            adjusted = losses.get(family_name, holm_adjust(raw)) if family_name else raw
            passed = all(float(adjusted[comparison]) <= alpha for comparison in comparison_names)
            if name == "gate_1":
                central = comparisons["M3_vs_M0"]
                positive = float(np.mean(np.asarray(central["donor_improvement"]) > 0))
                passed = (
                    passed
                    and float(losses.get("central_relative_gain", 0.0)) >= minimum_relative_gain
                    and positive >= minimum_positive_fraction
                    and bool(losses.get("no_severe_indication_reversal", False))
                )
            gate_payload[name] = {"reached": True, "passed": passed, "adjusted_p": adjusted}
            if not passed:
                stopped_at = name
        return {
            "supported": stopped_at is None,
            "stopped_at": stopped_at,
            "gates": gate_payload,
        }

    required = ("M0", "M1", "M2", "M3", "M4", "M6", "M7", "M8")
    if set(required) - set(losses):
        raise ValueError(f"losses lack models: {sorted(set(required) - set(losses))}")
    arrays = {name: np.asarray(losses[name], dtype=np.float64) for name in required}
    sizes = {len(value) for value in arrays.values()}
    if len(sizes) != 1 or any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("all model losses must be aligned one-dimensional donor vectors")
    donors = sizes.pop()
    if not 2 <= donors <= 20 or any(
        np.any(~np.isfinite(value)) or np.any(value < 0) for value in arrays.values()
    ):
        raise ValueError("losses must be finite/non-negative for 2--20 donors")
    indications = (
        np.asarray(["all"] * donors)
        if indication_ids is None
        else _ids(indication_ids, "indication_ids", donors)
    )
    tests = {
        name: exact_sign_flip_test(arrays[name] - arrays["M3"], alpha=alpha)
        for name in ("M0", "M1", "M2", "M4", "M6", "M7")
    }
    tests["M8"] = exact_sign_flip_test(arrays["M3"] - arrays["M8"], alpha=alpha)
    gain = arrays["M0"] - arrays["M3"]
    relative = (
        float(np.mean(gain) / np.mean(arrays["M0"])) if np.mean(arrays["M0"]) > 0 else -np.inf
    )
    severe_reversal = any(
        np.mean(gain[indications == label])
        < -maximum_indication_reversal * max(np.mean(arrays["M0"][indications == label]), 1.0e-12)
        for label in sorted(set(indications.tolist()))
    )
    gate1_reasons = []
    if tests["M0"].p_value > alpha:
        gate1_reasons.append("exact_sign_flip")
    if relative < minimum_relative_gain:
        gate1_reasons.append("relative_gain")
    if tests["M0"].positive_fraction < minimum_positive_fraction:
        gate1_reasons.append("donor_consistency")
    if tests["M0"].confidence_interval[0] <= 0:
        gate1_reasons.append("confidence_interval")
    if severe_reversal:
        gate1_reasons.append("indication_reversal")
    gate1 = GateDecision(
        "gate_1_incremental_value",
        True,
        not gate1_reasons,
        tuple(gate1_reasons),
        {"M3<M0": tests["M0"].p_value},
        {"M3<M0": tests["M0"].p_value},
    )

    def family_gate(name: str, comparators: tuple[str, ...], reached: bool) -> GateDecision:
        raw = np.asarray([tests[item].p_value for item in comparators])
        adjusted = holm_adjust(raw) if len(raw) > 1 else raw
        passed = reached and bool(np.all(adjusted <= alpha))
        return GateDecision(
            name,
            reached,
            passed,
            () if passed else (("not_reached",) if not reached else ("exact_sign_flip_holm",)),
            {f"M3<{item}": float(value) for item, value in zip(comparators, raw.tolist())},
            {f"M3<{item}": float(value) for item, value in zip(comparators, adjusted.tolist())},
        )

    gate2 = family_gate("gate_2_image_necessity", ("M1", "M4"), gate1.passed)
    gate3 = family_gate("gate_3_state_beyond_routing", ("M2",), gate2.passed)
    gate4 = family_gate("gate_4_matching_specificity", ("M6", "M7"), gate3.passed)
    gate5 = family_gate("gate_5_molecular_headroom", ("M8",), True)
    return OrderedGateDecision(
        gates=(gate1, gate2, gate3, gate4, gate5),
        central_hypothesis_supported=gate1.passed,
        full_model_chain_supported=gate1.passed and gate2.passed and gate3.passed,
        personalization_supported=gate1.passed and gate2.passed and gate3.passed and gate4.passed,
        molecular_headroom_detected=gate5.passed,
        relative_gain=relative,
        positive_donor_fraction=tests["M0"].positive_fraction,
    )


def calibration_slope(observed: object, predicted: object) -> float:
    truth, prediction = (
        np.asarray(observed, dtype=np.float64),
        np.asarray(predicted, dtype=np.float64),
    )
    if (
        prediction.shape != truth.shape
        or prediction.size < 2
        or not np.all(np.isfinite(prediction))
        or not np.all(np.isfinite(truth))
    ):
        raise ValueError("predicted and observed values must be aligned and finite")
    centered = prediction.ravel() - np.mean(prediction)
    denominator = float(centered @ centered)
    if denominator <= 0:
        raise ValueError("calibration slope is undefined for constant predictions")
    return float(centered @ (truth.ravel() - np.mean(truth)) / denominator)


@dataclass(frozen=True)
class ReliabilityAdjustedVariance:
    ratio: float
    predicted_variance: float
    signal_variance: float
    reliable_genes: int

    def __getitem__(self, key: str):
        return getattr(self, key)


def reliability_adjusted_variance(
    predicted: object, split_half_a: object, split_half_b: object
) -> ReliabilityAdjustedVariance:
    prediction = np.asarray(predicted, dtype=np.float64)
    first, second = (
        np.asarray(split_half_a, dtype=np.float64),
        np.asarray(split_half_b, dtype=np.float64),
    )
    if prediction.shape != first.shape or first.shape != second.shape or first.size < 2:
        raise ValueError("prediction and split halves must be aligned")
    if not np.all(np.isfinite(np.concatenate((prediction, first, second)))):
        raise ValueError("prediction and split halves must be finite")
    feature_first = first.reshape(first.shape[0], -1)
    feature_second = second.reshape(second.shape[0], -1)
    gene_signal = np.asarray(
        [
            np.cov(feature_first[:, index], feature_second[:, index], ddof=1)[0, 1]
            for index in range(feature_first.shape[1])
        ]
    )
    reliable = int(np.sum(gene_signal > 0))
    centered_first = first.ravel() - np.mean(first)
    centered_second = second.ravel() - np.mean(second)
    signal = float(centered_first @ centered_second / (first.size - 1))
    if signal <= 0:
        raise ValueError("split halves do not identify positive signal variance")
    predicted_variance = float(np.var(prediction, ddof=1))
    return ReliabilityAdjustedVariance(
        predicted_variance / signal, predicted_variance, signal, reliable
    )


def interval_coverage(
    posterior_samples: object,
    observed: object,
    upper: object | None = None,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
) -> dict[float, float] | dict[str, float]:
    if upper is not None:
        truth = np.asarray(posterior_samples, dtype=np.float64)
        low = np.asarray(observed, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if (
            truth.shape != low.shape
            or low.shape != high.shape
            or not np.all(np.isfinite(np.concatenate((truth.ravel(), low.ravel(), high.ravel()))))
        ):
            raise ValueError("observed and interval bounds must be aligned and finite")
        return {"coverage": float(np.mean((truth >= low) & (truth <= high)))}
    samples = np.asarray(posterior_samples, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    if samples.ndim < 2 or samples.shape[1:] != truth.shape or not np.all(np.isfinite(samples)):
        raise ValueError("posterior samples must be finite with leading sample axis")
    result = {}
    for level in levels:
        value = float(level)
        if not 0 < value < 1:
            raise ValueError("coverage levels must lie in (0,1)")
        low, high = np.quantile(samples, [(1 - value) / 2, 1 - (1 - value) / 2], axis=0)
        result[value] = float(np.mean((truth >= low) & (truth <= high)))
    return result
