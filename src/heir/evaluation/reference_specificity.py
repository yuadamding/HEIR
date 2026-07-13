"""Image-conditioned matched-reference utility with auditable bank provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _vector(values: object, name: str, *, rows: int, strings: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (rows,):
        raise ValueError("reference-utility %s must align with rows" % name)
    if strings:
        array = array.astype(str)
        if any(not value.strip() for value in array.tolist()):
            raise ValueError("reference-utility %s contains empty values" % name)
    return array


def _sha256(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("reference-utility %s must be a lowercase SHA-256" % name)
    return digest


@dataclass(frozen=True)
class ReferenceBank:
    """Molecular states and matching metadata for one bank substitution."""

    role: str
    latent: np.ndarray
    type_labels: np.ndarray
    donor_ids: np.ndarray
    observation_ids: np.ndarray
    section_ids: np.ndarray
    disease_states: np.ndarray
    site_ids: np.ndarray
    assay_ids: np.ndarray
    quality_bins: np.ndarray
    depth_bins: np.ndarray
    latent_model_sha256: str
    source_sha256: str

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, object]) -> "ReferenceBank":
        required = {
            "role",
            "latent",
            "type_labels",
            "donor_ids",
            "observation_ids",
            "section_ids",
            "disease_states",
            "site_ids",
            "assay_ids",
            "quality_bins",
            "depth_bins",
            "latent_model_sha256",
            "source_sha256",
        }
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise ValueError("reference bank is incomplete: %s" % name)
        latent = np.asarray(payload["latent"], dtype=np.float64)
        if latent.ndim != 2 or not len(latent) or not np.isfinite(latent).all():
            raise ValueError("reference bank latent is malformed: %s" % name)
        rows = len(latent)
        role = str(payload["role"])
        if role not in {"matched", "hard_wrong", "generic"}:
            raise ValueError("reference bank role is unsupported: %s" % name)
        value = cls(
            role=role,
            latent=latent,
            type_labels=_vector(payload["type_labels"], "type_labels", rows=rows),
            donor_ids=_vector(payload["donor_ids"], "donor_ids", rows=rows, strings=True),
            observation_ids=_vector(
                payload["observation_ids"], "observation_ids", rows=rows, strings=True
            ),
            section_ids=_vector(payload["section_ids"], "section_ids", rows=rows, strings=True),
            disease_states=_vector(
                payload["disease_states"], "disease_states", rows=rows, strings=True
            ),
            site_ids=_vector(payload["site_ids"], "site_ids", rows=rows, strings=True),
            assay_ids=_vector(payload["assay_ids"], "assay_ids", rows=rows, strings=True),
            quality_bins=_vector(payload["quality_bins"], "quality_bins", rows=rows, strings=True),
            depth_bins=_vector(payload["depth_bins"], "depth_bins", rows=rows, strings=True),
            latent_model_sha256=_sha256(payload["latent_model_sha256"], "latent_model_sha256"),
            source_sha256=_sha256(payload["source_sha256"], "source_sha256"),
        )
        if len(set(value.observation_ids.tolist())) != rows:
            raise ValueError("reference bank observation IDs are not unique: %s" % name)
        return value


def _bootstrap_interval(
    donor_effects: np.ndarray, *, seed: int, samples: int, confidence: float
) -> tuple[float, float]:
    if samples < 1000:
        raise ValueError("reference-utility donor bootstrap requires at least 1000 samples")
    if not 0.5 < confidence < 1.0:
        raise ValueError("reference-utility confidence must be between 0.5 and 1")
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        donor_effects,
        size=(samples, len(donor_effects)),
        replace=True,
    ).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def evaluate_reference_utility(
    image_state_latent: np.ndarray,
    molecular_target_latent: np.ndarray,
    query_types: np.ndarray,
    query_donors: Sequence[object],
    query_observation_ids: Sequence[object],
    query_section_ids: Sequence[object],
    query_disease_states: Sequence[object],
    query_site_ids: Sequence[object],
    query_assay_ids: Sequence[object],
    query_quality_bins: Sequence[object],
    query_depth_bins: Sequence[object],
    banks: Mapping[str, Mapping[str, object]],
    *,
    repeats: int = 100,
    minimum_effect: float = 0.01,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> Mapping[str, object]:
    """Swap only the molecular bank used to decode one frozen image-state prediction.

    Candidate banks are equalized separately for every locked query by exact type,
    disease, site, assay, quality-bin and depth-bin strata. The nearest candidate is
    selected by the frozen image-state latent, while utility is scored against the
    unchanged locked molecular target latent.
    """

    image = np.asarray(image_state_latent, dtype=np.float64)
    target = np.asarray(molecular_target_latent, dtype=np.float64)
    if (
        image.ndim != 2
        or target.shape != image.shape
        or not len(image)
        or not np.isfinite(image).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("reference-utility image and molecular query latents are malformed")
    rows = len(image)
    query = {
        "types": _vector(query_types, "query_types", rows=rows),
        "donors": _vector(query_donors, "query_donors", rows=rows, strings=True),
        "observations": _vector(
            query_observation_ids, "query_observation_ids", rows=rows, strings=True
        ),
        "sections": _vector(query_section_ids, "query_section_ids", rows=rows, strings=True),
        "disease": _vector(query_disease_states, "query_disease_states", rows=rows, strings=True),
        "site": _vector(query_site_ids, "query_site_ids", rows=rows, strings=True),
        "assay": _vector(query_assay_ids, "query_assay_ids", rows=rows, strings=True),
        "quality": _vector(query_quality_bins, "query_quality_bins", rows=rows, strings=True),
        "depth": _vector(query_depth_bins, "query_depth_bins", rows=rows, strings=True),
    }
    if len(set(query["observations"].tolist())) != rows:
        raise ValueError("reference-utility query observation IDs are not unique")
    if repeats < 100:
        raise ValueError("reference-utility gate requires at least 100 bank subsampling repeats")
    if minimum_effect <= 0:
        raise ValueError("reference-utility minimum effect must be positive")
    if not banks:
        raise ValueError("reference-utility requires molecular banks")

    parsed = {
        str(name): ReferenceBank.from_mapping(str(name), bank) for name, bank in banks.items()
    }
    roles = [bank.role for bank in parsed.values()]
    if roles.count("matched") != 1 or "hard_wrong" not in roles or "generic" not in roles:
        raise ValueError("reference-utility needs one matched, a hard-wrong and a generic bank")
    if any(bank.latent.shape[1] != image.shape[1] for bank in parsed.values()):
        raise ValueError("all banks and query states must share the same latent width")
    latent_hashes = {bank.latent_model_sha256 for bank in parsed.values()}
    if len(latent_hashes) != 1:
        raise ValueError("all banks must use one frozen latent preprocessing model")
    if len({bank.source_sha256 for bank in parsed.values()}) != len(parsed):
        raise ValueError("reference banks must have independently identifiable sources")

    query_observations = set(query["observations"].tolist())
    query_sections = set(query["sections"].tolist())
    seen_bank_observations: set[str] = set()
    for name, bank in parsed.items():
        observations = set(bank.observation_ids.tolist())
        if observations & query_observations:
            raise ValueError("query observations overlap reference bank: %s" % name)
        if observations & seen_bank_observations:
            raise ValueError("reference banks overlap one another: %s" % name)
        seen_bank_observations.update(observations)
        if set(bank.section_ids.tolist()) & query_sections:
            raise ValueError("query sections overlap reference bank: %s" % name)
        if bank.role == "generic" and set(bank.donor_ids.tolist()) & set(query["donors"].tolist()):
            raise ValueError("generic bank donors must be independent of query donors")

    def candidates(bank: ReferenceBank, index: int) -> np.ndarray:
        selected = (
            (bank.type_labels == query["types"][index])
            & (bank.disease_states == query["disease"][index])
            & (bank.site_ids == query["site"][index])
            & (bank.assay_ids == query["assay"][index])
            & (bank.quality_bins == query["quality"][index])
            & (bank.depth_bins == query["depth"][index])
        )
        if bank.role == "matched":
            selected &= bank.donor_ids == query["donors"][index]
        elif bank.role == "hard_wrong":
            selected &= bank.donor_ids != query["donors"][index]
        return np.flatnonzero(selected)

    candidate_rows: dict[str, list[np.ndarray]] = {name: [] for name in parsed}
    per_query_size = np.zeros(rows, dtype=np.int64)
    for index in range(rows):
        available = []
        for name, bank in parsed.items():
            selected = candidates(bank, index)
            candidate_rows[name].append(selected)
            available.append(len(selected))
        per_query_size[index] = min(available)
    if np.any(per_query_size < 2):
        failed = np.flatnonzero(per_query_size < 2)
        raise ValueError(
            "every query needs at least two exactly matched candidates in every bank; "
            "failed rows: %s" % ",".join(map(str, failed[:10].tolist()))
        )

    donors = sorted(set(query["donors"].tolist()))
    if len(donors) < 3:
        raise ValueError("reference-utility inference requires at least three locked donors")
    rng = np.random.default_rng(seed)
    repeated_errors = {name: [] for name in parsed}
    donor_repeated_errors = {name: {donor: [] for donor in donors} for name in parsed}
    for _ in range(repeats):
        for name, bank in parsed.items():
            row_error = np.zeros(rows, dtype=np.float64)
            for index in range(rows):
                selected = rng.choice(
                    candidate_rows[name][index],
                    size=int(per_query_size[index]),
                    replace=False,
                )
                latent = bank.latent[selected]
                nearest = int(np.argmin(np.square(latent - image[index]).sum(axis=1)))
                row_error[index] = float(np.square(latent[nearest] - target[index]).mean())
            donor_scores = []
            for donor in donors:
                donor_mask = query["donors"] == donor
                type_scores = []
                for type_index in sorted(set(query["types"][donor_mask].tolist())):
                    selected = donor_mask & (query["types"] == type_index)
                    type_scores.append(float(row_error[selected].mean()))
                donor_score = float(np.mean(type_scores))
                donor_repeated_errors[name][donor].append(donor_score)
                donor_scores.append(donor_score)
            repeated_errors[name].append(float(np.mean(donor_scores)))

    matched_name = next(name for name, bank in parsed.items() if bank.role == "matched")
    comparisons = []
    for position, (name, bank) in enumerate(sorted(parsed.items())):
        if name == matched_name:
            continue
        matched_values = np.asarray(repeated_errors[matched_name])
        comparator_values = np.asarray(repeated_errors[name])
        donor_effects = np.asarray(
            [
                np.mean(donor_repeated_errors[name][donor])
                - np.mean(donor_repeated_errors[matched_name][donor])
                for donor in donors
            ],
            dtype=np.float64,
        )
        lower, upper = _bootstrap_interval(
            donor_effects,
            seed=seed + 1009 * (position + 1),
            samples=bootstrap_samples,
            confidence=confidence,
        )
        mean_effect = float(donor_effects.mean())
        comparisons.append(
            {
                "bank": name,
                "bank_role": bank.role,
                "matched_minus_comparator_mse_reduction": mean_effect,
                "donor_confidence_interval": [lower, upper],
                "fraction_subsamples_matched_better": float(
                    np.mean(comparator_values > matched_values)
                ),
                "matched_mean_mse": float(matched_values.mean()),
                "comparator_mean_mse": float(comparator_values.mean()),
                "per_donor_effect": {
                    donor: float(value) for donor, value in zip(donors, donor_effects)
                },
                "passes_minimum_effect": bool(lower >= minimum_effect),
            }
        )
    passed = all(row["passes_minimum_effect"] for row in comparisons)
    return {
        "schema": "heir.matched_reference_utility.v1",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "image_conditioned": True,
        "frozen_image_queries": True,
        "locked_molecular_targets_unchanged": True,
        "only_reference_bank_changes": True,
        "repeats": repeats,
        "seed": seed,
        "minimum_effect": minimum_effect,
        "donor_bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "latent_model_sha256": next(iter(latent_hashes)),
        "query_count": rows,
        "donor_count": len(donors),
        "per_query_equalized_bank_size": {
            "minimum": int(per_query_size.min()),
            "median": float(np.median(per_query_size)),
            "maximum": int(per_query_size.max()),
        },
        "matching_factors": [
            "type",
            "disease",
            "site",
            "assay",
            "quality_bin",
            "depth_bin",
        ],
        "aggregation": "equal_donor_equal_type",
        "comparisons": comparisons,
        "scientific_scope": "image_conditioned_reference_bank_substitution",
    }


__all__ = ["ReferenceBank", "evaluate_reference_utility"]
