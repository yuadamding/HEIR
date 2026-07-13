"""Image-conditioned matched-reference utility with auditable bank provenance."""

from __future__ import annotations

import hashlib
import json
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
    source_sample_ids: np.ndarray
    source_material_ids: np.ndarray
    specimen_ids: np.ndarray
    preservation_methods: np.ndarray
    disease_states: np.ndarray
    site_ids: np.ndarray
    institution_ids: np.ndarray
    assay_ids: np.ndarray
    quality_bins: np.ndarray
    depth_bins: np.ndarray
    latent_model_sha256: str
    normalization_sha256: str
    assay_harmonization_sha256: str
    assay_mode: str
    latent_fit_donor_ids: tuple[str, ...]
    assay_harmonization_fit_donor_ids: tuple[str, ...]
    assay_harmonization_source_sha256: str
    calibrated_assay_pairs: tuple[str, ...]
    material_relationship_to_query: str
    independent_tissue_material: bool
    contains_registered_query_cells: bool
    selection_uses_query_truth: bool
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
            "source_sample_ids",
            "source_material_ids",
            "specimen_ids",
            "preservation_methods",
            "disease_states",
            "site_ids",
            "institution_ids",
            "assay_ids",
            "quality_bins",
            "depth_bins",
            "latent_model_sha256",
            "normalization_sha256",
            "assay_harmonization_sha256",
            "assay_mode",
            "latent_fit_donor_ids",
            "assay_harmonization_fit_donor_ids",
            "assay_harmonization_source_sha256",
            "calibrated_assay_pairs",
            "material_relationship_to_query",
            "independent_tissue_material",
            "contains_registered_query_cells",
            "selection_uses_query_truth",
            "source_sha256",
        }
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise ValueError("reference bank is incomplete: %s" % name)
        latent = np.asarray(payload["latent"], dtype=np.float64)
        if latent.ndim != 2 or not len(latent) or not np.isfinite(latent).all():
            raise ValueError("reference bank latent is malformed: %s" % name)
        rows = len(latent)
        role = str(payload["role"])
        if role not in {"matched", "hard_wrong", "generic", "population_leave_query_out"}:
            raise ValueError("reference bank role is unsupported: %s" % name)
        latent_fit_raw = payload["latent_fit_donor_ids"]
        if isinstance(latent_fit_raw, (str, bytes)):
            raise ValueError("reference bank latent-fit donors are malformed: %s" % name)
        latent_fit_donors = tuple(str(value) for value in latent_fit_raw)
        if (
            not latent_fit_donors
            or any(not value.strip() for value in latent_fit_donors)
            or len(set(latent_fit_donors)) != len(latent_fit_donors)
        ):
            raise ValueError("reference bank latent-fit donors are missing: %s" % name)
        harmonization_fit_raw = payload["assay_harmonization_fit_donor_ids"]
        if isinstance(harmonization_fit_raw, (str, bytes)):
            raise ValueError("reference bank assay-fit donors are malformed: %s" % name)
        harmonization_fit_donors = tuple(str(value) for value in harmonization_fit_raw)
        if (
            not harmonization_fit_donors
            or any(not value.strip() for value in harmonization_fit_donors)
            or len(set(harmonization_fit_donors)) != len(harmonization_fit_donors)
        ):
            raise ValueError("reference bank assay-fit donors are missing: %s" % name)
        pairs_raw = payload["calibrated_assay_pairs"]
        if isinstance(pairs_raw, (str, bytes)):
            raise ValueError("reference bank calibrated assay pairs are malformed: %s" % name)
        calibrated_pairs = tuple(str(value) for value in pairs_raw)
        if (
            not calibrated_pairs
            or any(not value.strip() for value in calibrated_pairs)
            or len(set(calibrated_pairs)) != len(calibrated_pairs)
        ):
            raise ValueError("reference bank calibrated assay pairs are missing: %s" % name)
        assay_mode = str(payload["assay_mode"])
        if assay_mode not in {"same_assay", "cross_assay_development_calibrated"}:
            raise ValueError("reference bank assay mode is unsupported: %s" % name)
        if not isinstance(payload["selection_uses_query_truth"], (bool, np.bool_)):
            raise ValueError("reference bank truth-selection flag is malformed: %s" % name)
        if not isinstance(
            payload["independent_tissue_material"], (bool, np.bool_)
        ) or not isinstance(payload["contains_registered_query_cells"], (bool, np.bool_)):
            raise ValueError(
                "reference bank biological-independence flags are malformed: %s" % name
            )
        relationship = str(payload["material_relationship_to_query"])
        if relationship not in {
            "independent_aliquot",
            "independent_specimen_block",
            "different_specimen_same_donor",
            "external_independent_donor_material",
        }:
            raise ValueError("reference bank material relationship is unsupported: %s" % name)
        value = cls(
            role=role,
            latent=latent,
            type_labels=_vector(payload["type_labels"], "type_labels", rows=rows),
            donor_ids=_vector(payload["donor_ids"], "donor_ids", rows=rows, strings=True),
            observation_ids=_vector(
                payload["observation_ids"], "observation_ids", rows=rows, strings=True
            ),
            section_ids=_vector(payload["section_ids"], "section_ids", rows=rows, strings=True),
            source_sample_ids=_vector(
                payload["source_sample_ids"], "source_sample_ids", rows=rows, strings=True
            ),
            source_material_ids=_vector(
                payload["source_material_ids"], "source_material_ids", rows=rows, strings=True
            ),
            specimen_ids=_vector(payload["specimen_ids"], "specimen_ids", rows=rows, strings=True),
            preservation_methods=_vector(
                payload["preservation_methods"], "preservation_methods", rows=rows, strings=True
            ),
            disease_states=_vector(
                payload["disease_states"], "disease_states", rows=rows, strings=True
            ),
            site_ids=_vector(payload["site_ids"], "site_ids", rows=rows, strings=True),
            institution_ids=_vector(
                payload["institution_ids"], "institution_ids", rows=rows, strings=True
            ),
            assay_ids=_vector(payload["assay_ids"], "assay_ids", rows=rows, strings=True),
            quality_bins=_vector(payload["quality_bins"], "quality_bins", rows=rows, strings=True),
            depth_bins=_vector(payload["depth_bins"], "depth_bins", rows=rows, strings=True),
            latent_model_sha256=_sha256(payload["latent_model_sha256"], "latent_model_sha256"),
            normalization_sha256=_sha256(payload["normalization_sha256"], "normalization_sha256"),
            assay_harmonization_sha256=_sha256(
                payload["assay_harmonization_sha256"], "assay_harmonization_sha256"
            ),
            assay_mode=assay_mode,
            latent_fit_donor_ids=latent_fit_donors,
            assay_harmonization_fit_donor_ids=harmonization_fit_donors,
            assay_harmonization_source_sha256=_sha256(
                payload["assay_harmonization_source_sha256"],
                "assay_harmonization_source_sha256",
            ),
            calibrated_assay_pairs=calibrated_pairs,
            material_relationship_to_query=relationship,
            independent_tissue_material=bool(payload["independent_tissue_material"]),
            contains_registered_query_cells=bool(payload["contains_registered_query_cells"]),
            selection_uses_query_truth=bool(payload["selection_uses_query_truth"]),
            source_sha256=_sha256(payload["source_sha256"], "source_sha256"),
        )
        if len(set(value.observation_ids.tolist())) != rows:
            raise ValueError("reference bank observation IDs are not unique: %s" % name)
        if value.selection_uses_query_truth:
            raise ValueError("reference bank cells cannot be selected using query truth: %s" % name)
        if not value.independent_tissue_material or value.contains_registered_query_cells:
            raise ValueError("reference bank is not biologically independent of queries: %s" % name)
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


def _unique_strings(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("reference-utility %s must be a sequence" % name)
    result = tuple(str(value) for value in values)
    if not result or any(not value.strip() for value in result) or len(set(result)) != len(result):
        raise ValueError("reference-utility %s must be non-empty and unique" % name)
    return result


def _aggregate_donor_type(
    row_error: np.ndarray,
    indices: np.ndarray,
    query_donors: np.ndarray,
    query_types: np.ndarray,
) -> tuple[float, dict[str, float]]:
    donor_scores: dict[str, float] = {}
    for donor in sorted(set(query_donors[indices].tolist())):
        donor_indices = indices[query_donors[indices] == donor]
        type_scores = []
        for type_index in sorted(set(query_types[donor_indices].tolist())):
            selected = donor_indices[query_types[donor_indices] == type_index]
            type_scores.append(float(row_error[selected].mean()))
        donor_scores[str(donor)] = float(np.mean(type_scores))
    return float(np.mean(tuple(donor_scores.values()))), donor_scores


def _validate_power_receipt(
    receipt: Mapping[str, object],
    *,
    receipt_sha256: str,
    minimum_relative_effect: float,
) -> Mapping[str, object]:
    required = {
        "schema",
        "simulation_sha256",
        "thresholds_sha256",
        "minimum_relative_effect",
        "minimum_query_donors",
        "minimum_comparison_query_donors",
        "maximum_familywise_false_positive_probability",
        "power_at_minimum_relative_effect",
        "uses_locked_query_outcomes",
        "scenario_families",
        "receipt_content_sha256",
    }
    if not isinstance(receipt, Mapping) or not required.issubset(receipt):
        raise ValueError("reference-utility power receipt is incomplete")
    if receipt["schema"] != "heir.reference_utility_power.v1":
        raise ValueError("reference-utility power receipt schema is unsupported")
    _sha256(receipt_sha256, "power_analysis_receipt_sha256")
    _sha256(receipt["simulation_sha256"], "power simulation_sha256")
    _sha256(receipt["thresholds_sha256"], "power thresholds_sha256")
    recorded_content_hash = _sha256(
        receipt["receipt_content_sha256"], "power receipt_content_sha256"
    )
    core = {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != recorded_content_hash:
        raise ValueError("reference-utility power receipt content hash differs")
    try:
        receipt_effect = float(receipt["minimum_relative_effect"])
        query_minimum = int(receipt["minimum_query_donors"])
        comparison_minimum = int(receipt["minimum_comparison_query_donors"])
        false_positive = float(receipt["maximum_familywise_false_positive_probability"])
        power = float(receipt["power_at_minimum_relative_effect"])
    except (TypeError, ValueError) as error:
        raise ValueError("reference-utility power receipt values are malformed") from error
    scenarios = _unique_strings(receipt["scenario_families"], "power scenario families")
    required_scenarios = {
        "donor_count",
        "query_population_coverage",
        "sparse_exact_matching",
        "wrong_donor_eligibility",
        "assay_harmonization",
    }
    if (
        receipt["uses_locked_query_outcomes"] is not False
        or receipt_effect != minimum_relative_effect
        or query_minimum < 3
        or comparison_minimum < query_minimum
        or false_positive > 0.05
        or power < 0.8
        or not required_scenarios.issubset(scenarios)
    ):
        raise ValueError("reference-utility power receipt fails the prespecified design")
    return {
        "receipt_sha256": receipt_sha256,
        "receipt_content_sha256": recorded_content_hash,
        "simulation_sha256": str(receipt["simulation_sha256"]),
        "thresholds_sha256": str(receipt["thresholds_sha256"]),
        "minimum_query_donors": query_minimum,
        "minimum_comparison_query_donors": comparison_minimum,
        "maximum_familywise_false_positive_probability": false_positive,
        "power_at_minimum_relative_effect": power,
        "scenario_families": list(scenarios),
        "uses_locked_query_outcomes": False,
    }


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
    query_source_sample_ids: Sequence[object],
    query_source_material_ids: Sequence[object],
    query_specimen_ids: Sequence[object],
    query_preservation_methods: Sequence[object],
    query_institution_ids: Sequence[object],
    query_latent_model_sha256: str,
    query_normalization_sha256: str,
    query_assay_harmonization_sha256: str,
    query_assay_harmonization_fit_donor_ids: Sequence[object],
    query_assay_harmonization_source_sha256: str,
    query_calibrated_assay_pairs: Sequence[object],
    eligible_hard_wrong_donor_ids: Sequence[object],
    query_eligibility: Mapping[str, object],
    power_analysis_receipt: Mapping[str, object],
    power_analysis_receipt_sha256: str,
    frozen_image_model_sha256: str,
    query_source_sha256: str,
    morphology_evidence_binding: Mapping[str, object],
    repeats: int = 100,
    minimum_relative_effect: float = 0.05,
    minimum_positive_donor_fraction: float = 0.75,
    minimum_query_coverage: float = 0.8,
    require_full_donor_and_type_coverage: bool = True,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> Mapping[str, object]:
    """Swap only the molecular bank used to decode one frozen image-state prediction.

    Candidate banks are equalized pairwise for every locked query by exact type,
    disease, site, institution, quality-bin and depth-bin strata. Assay is also exact
    in same-assay mode; cross-assay use instead requires one development-calibrated
    harmonization transform. The nearest candidate is selected by the frozen image
    state, while utility is scored against unchanged locked molecular truth.
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
        "source_samples": _vector(
            query_source_sample_ids, "query_source_sample_ids", rows=rows, strings=True
        ),
        "source_materials": _vector(
            query_source_material_ids, "query_source_material_ids", rows=rows, strings=True
        ),
        "specimens": _vector(query_specimen_ids, "query_specimen_ids", rows=rows, strings=True),
        "preservation": _vector(
            query_preservation_methods, "query_preservation_methods", rows=rows, strings=True
        ),
        "disease": _vector(query_disease_states, "query_disease_states", rows=rows, strings=True),
        "site": _vector(query_site_ids, "query_site_ids", rows=rows, strings=True),
        "institution": _vector(
            query_institution_ids, "query_institution_ids", rows=rows, strings=True
        ),
        "assay": _vector(query_assay_ids, "query_assay_ids", rows=rows, strings=True),
        "quality": _vector(query_quality_bins, "query_quality_bins", rows=rows, strings=True),
        "depth": _vector(query_depth_bins, "query_depth_bins", rows=rows, strings=True),
    }
    if len(set(query["observations"].tolist())) != rows:
        raise ValueError("reference-utility query observation IDs are not unique")
    if repeats < 100:
        raise ValueError("reference-utility gate requires at least 100 bank subsampling repeats")
    if minimum_relative_effect <= 0 or minimum_relative_effect >= 1:
        raise ValueError("reference-utility minimum relative effect must be between zero and one")
    if not 0.5 <= minimum_positive_donor_fraction <= 1:
        raise ValueError("reference-utility positive-donor fraction must be between 0.5 and 1")
    if not 0 < minimum_query_coverage <= 1:
        raise ValueError("reference-utility minimum query coverage must be in (0, 1]")
    if not banks:
        raise ValueError("reference-utility requires molecular banks")
    frozen_image_hash = _sha256(frozen_image_model_sha256, "frozen_image_model_sha256")
    query_source_hash = _sha256(query_source_sha256, "query_source_sha256")
    evidence_required = {
        "primary_feature_checkpoint_sha256",
        "primary_query_source_sha256",
        "primary_report_sha256",
        "external_report_sha256s",
    }
    if not isinstance(morphology_evidence_binding, Mapping) or not evidence_required.issubset(
        morphology_evidence_binding
    ):
        raise ValueError("reference-utility morphology evidence binding is incomplete")
    if (
        _sha256(
            morphology_evidence_binding["primary_feature_checkpoint_sha256"],
            "primary morphology feature checkpoint",
        )
        != frozen_image_hash
        or _sha256(
            morphology_evidence_binding["primary_query_source_sha256"],
            "primary morphology query source",
        )
        != query_source_hash
    ):
        raise ValueError("reference queries differ from the passing primary morphology evidence")
    primary_report_hash = _sha256(
        morphology_evidence_binding["primary_report_sha256"],
        "primary morphology report_sha256",
    )
    external_report_hashes = _unique_strings(
        morphology_evidence_binding["external_report_sha256s"],
        "external morphology report SHA-256 values",
    )
    for index, value in enumerate(external_report_hashes):
        _sha256(value, "external morphology report_sha256[%d]" % index)

    eligibility_required = {
        "total_eligible_query_count",
        "excluded_query_count_by_reason",
        "eligible_donor_ids",
        "eligible_type_labels",
        "power_analysis_sha256",
        "power_justified_minimum_donors",
    }
    if not isinstance(query_eligibility, Mapping) or not eligibility_required.issubset(
        query_eligibility
    ):
        raise ValueError("reference-utility query eligibility denominator is incomplete")
    try:
        total_eligible = int(query_eligibility["total_eligible_query_count"])
        power_minimum_donors = int(query_eligibility["power_justified_minimum_donors"])
    except (TypeError, ValueError) as error:
        raise ValueError("reference-utility eligibility counts are malformed") from error
    exclusions_raw = query_eligibility["excluded_query_count_by_reason"]
    if not isinstance(exclusions_raw, Mapping) or any(
        not str(reason).strip() or not isinstance(count, (int, np.integer)) or int(count) < 0
        for reason, count in exclusions_raw.items()
    ):
        raise ValueError("reference-utility exclusion denominator is malformed")
    exclusions = {str(reason): int(count) for reason, count in exclusions_raw.items()}
    if total_eligible < rows or sum(exclusions.values()) != total_eligible - rows:
        raise ValueError("reference-utility eligibility denominator does not reconcile")
    eligible_donors = _unique_strings(query_eligibility["eligible_donor_ids"], "eligible donor IDs")
    eligible_types = tuple(str(value) for value in query_eligibility["eligible_type_labels"])
    if not eligible_types or len(set(eligible_types)) != len(eligible_types):
        raise ValueError("reference-utility eligible type labels must be non-empty and unique")
    power_analysis_sha256 = _sha256(
        query_eligibility["power_analysis_sha256"], "power_analysis_sha256"
    )
    validated_power = _validate_power_receipt(
        power_analysis_receipt,
        receipt_sha256=power_analysis_receipt_sha256,
        minimum_relative_effect=minimum_relative_effect,
    )
    if (
        power_analysis_sha256 != validated_power["receipt_sha256"]
        or power_minimum_donors != validated_power["minimum_query_donors"]
    ):
        raise ValueError("reference-utility eligibility differs from its power receipt")
    included_donors = sorted(set(query["donors"].tolist()))
    included_types = sorted({str(value) for value in query["types"].tolist()})
    lost_donors = sorted(set(eligible_donors) - set(included_donors))
    lost_types = sorted(set(eligible_types) - set(included_types))
    unexpected_donors = sorted(set(included_donors) - set(eligible_donors))
    unexpected_types = sorted(set(included_types) - set(eligible_types))
    if unexpected_donors or unexpected_types:
        raise ValueError("reference-utility queries are outside the frozen eligible population")
    query_coverage = rows / total_eligible

    eligible_wrong_donors = _unique_strings(
        eligible_hard_wrong_donor_ids, "eligible hard-wrong donor IDs"
    )
    query_latent_hash = _sha256(query_latent_model_sha256, "query_latent_model_sha256")
    query_normalization_hash = _sha256(query_normalization_sha256, "query_normalization_sha256")
    query_harmonization_hash = _sha256(
        query_assay_harmonization_sha256, "query_assay_harmonization_sha256"
    )
    query_harmonization_fit_donors = _unique_strings(
        query_assay_harmonization_fit_donor_ids,
        "query assay-harmonization fit donor IDs",
    )
    query_harmonization_source_hash = _sha256(
        query_assay_harmonization_source_sha256,
        "query_assay_harmonization_source_sha256",
    )
    query_calibrated_pairs = _unique_strings(
        query_calibrated_assay_pairs, "query calibrated assay pairs"
    )

    parsed = {
        str(name): ReferenceBank.from_mapping(str(name), bank) for name, bank in banks.items()
    }
    roles = [bank.role for bank in parsed.values()]
    if (
        roles.count("matched") != 1
        or roles.count("generic") != 1
        or roles.count("population_leave_query_out") != 1
        or "hard_wrong" not in roles
    ):
        raise ValueError(
            "reference-utility needs one matched, every hard-wrong, one generic and one "
            "leave-query-donor-out population bank"
        )
    hard_wrong_banks = {name: bank for name, bank in parsed.items() if bank.role == "hard_wrong"}
    hard_wrong_bank_donors: dict[str, str] = {}
    for name, bank in hard_wrong_banks.items():
        donors = set(bank.donor_ids.tolist())
        if len(donors) != 1:
            raise ValueError("each hard-wrong bank must represent exactly one donor: %s" % name)
        hard_wrong_bank_donors[name] = next(iter(donors))
    if set(hard_wrong_bank_donors.values()) != set(eligible_wrong_donors) or len(
        hard_wrong_bank_donors
    ) != len(eligible_wrong_donors):
        raise ValueError("reference-utility must include one bank for every eligible wrong donor")
    if any(bank.latent.shape[1] != image.shape[1] for bank in parsed.values()):
        raise ValueError("all banks and query states must share the same latent width")
    latent_hashes = {bank.latent_model_sha256 for bank in parsed.values()}
    normalization_hashes = {bank.normalization_sha256 for bank in parsed.values()}
    harmonization_hashes = {bank.assay_harmonization_sha256 for bank in parsed.values()}
    assay_modes = {bank.assay_mode for bank in parsed.values()}
    latent_fit_sets = {tuple(sorted(bank.latent_fit_donor_ids)) for bank in parsed.values()}
    harmonization_fit_sets = {
        tuple(sorted(bank.assay_harmonization_fit_donor_ids)) for bank in parsed.values()
    }
    harmonization_source_hashes = {
        bank.assay_harmonization_source_sha256 for bank in parsed.values()
    }
    calibrated_pair_sets = {tuple(sorted(bank.calibrated_assay_pairs)) for bank in parsed.values()}
    if latent_hashes != {query_latent_hash}:
        raise ValueError("all banks must use one frozen latent preprocessing model")
    if normalization_hashes != {query_normalization_hash}:
        raise ValueError("all queries and banks must use one frozen normalization")
    if harmonization_hashes != {query_harmonization_hash}:
        raise ValueError("all queries and banks must use one assay harmonization transform")
    if len(assay_modes) != 1:
        raise ValueError("all reference banks must declare one assay comparison mode")
    if len(latent_fit_sets) != 1:
        raise ValueError("all reference banks must use one frozen latent-fit donor set")
    latent_fit_donors = next(iter(latent_fit_sets))
    if set(latent_fit_donors) & set(eligible_donors):
        raise ValueError("locked query donors cannot fit or select the molecular latent model")
    if harmonization_fit_sets != {tuple(sorted(query_harmonization_fit_donors))}:
        raise ValueError("queries and banks must use one assay-harmonization fit donor set")
    if harmonization_source_hashes != {query_harmonization_source_hash}:
        raise ValueError("queries and banks must bind one assay-harmonization source")
    if calibrated_pair_sets != {tuple(sorted(query_calibrated_pairs))}:
        raise ValueError("queries and banks must bind one calibrated assay-pair contract")
    if set(query_harmonization_fit_donors) & set(eligible_donors):
        raise ValueError("locked query donors cannot fit assay harmonization")
    assay_mode = next(iter(assay_modes))
    if assay_mode == "same_assay":
        if query_calibrated_pairs != ("same_assay",):
            raise ValueError("same-assay mode must declare only the same_assay contract")
    else:
        declared_pairs = set(query_calibrated_pairs)
        query_assays = set(query["assay"].tolist())
        reference_assays = set(
            value for bank in parsed.values() for value in bank.assay_ids.tolist()
        )
        actual_pairs = {
            "%s::%s" % (query_assay, reference_assay)
            for query_assay in query_assays
            for reference_assay in reference_assays
        }
        if not actual_pairs or not actual_pairs.issubset(declared_pairs):
            raise ValueError("cross-assay banks contain an undeclared assay calibration pair")

    query_observations = set(query["observations"].tolist())
    query_sections = set(query["sections"].tolist())
    query_source_samples = set(query["source_samples"].tolist())
    query_source_materials = set(query["source_materials"].tolist())
    query_specimens = set(query["specimens"].tolist())
    for name, bank in parsed.items():
        observations = set(bank.observation_ids.tolist())
        if observations & query_observations:
            raise ValueError("query observations overlap reference bank: %s" % name)
        if set(bank.section_ids.tolist()) & query_sections:
            raise ValueError("query sections overlap reference bank: %s" % name)
        if set(bank.source_sample_ids.tolist()) & query_source_samples:
            raise ValueError("query source samples overlap reference bank: %s" % name)
        if set(bank.source_material_ids.tolist()) & query_source_materials:
            raise ValueError("query tissue material overlaps reference bank: %s" % name)
        if set(bank.specimen_ids.tolist()) & query_specimens:
            raise ValueError("query specimens overlap reference bank: %s" % name)
        if bank.role == "generic" and set(bank.donor_ids.tolist()) & set(query["donors"].tolist()):
            raise ValueError("generic bank donors must be independent of query donors")

    matched_name = next(name for name, bank in parsed.items() if bank.role == "matched")
    matched_bank = parsed[matched_name]
    if not set(included_donors).issubset(set(matched_bank.donor_ids.tolist())):
        raise ValueError("matched bank does not cover every included query donor")

    def candidates(bank: ReferenceBank, index: int) -> np.ndarray:
        selected = (
            (bank.type_labels == query["types"][index])
            & (bank.disease_states == query["disease"][index])
            & (bank.site_ids == query["site"][index])
            & (bank.institution_ids == query["institution"][index])
            & (bank.quality_bins == query["quality"][index])
            & (bank.depth_bins == query["depth"][index])
        )
        if assay_mode == "same_assay":
            selected &= bank.assay_ids == query["assay"][index]
        if bank.role == "matched":
            selected &= bank.donor_ids == query["donors"][index]
        elif bank.role in {"hard_wrong", "population_leave_query_out"}:
            selected &= bank.donor_ids != query["donors"][index]
        return np.flatnonzero(selected)

    matched_candidates = [candidates(matched_bank, index) for index in range(rows)]
    failed_matched = [
        index for index, selected in enumerate(matched_candidates) if len(selected) < 2
    ]
    if failed_matched:
        raise ValueError(
            "every query needs at least two exactly matched candidates in the matched bank; "
            "failed rows: %s" % ",".join(map(str, failed_matched[:10]))
        )

    coverage_pass = (
        query_coverage >= minimum_query_coverage
        and len(included_donors) >= power_minimum_donors
        and (not require_full_donor_and_type_coverage or (not lost_donors and not lost_types))
    )
    rng = np.random.default_rng(seed)
    comparisons = []
    for position, (name, bank) in enumerate(sorted(parsed.items())):
        if name == matched_name:
            continue
        comparator_candidates = [candidates(bank, index) for index in range(rows)]
        eligible_indices = np.asarray(
            [
                index
                for index in range(rows)
                if not (
                    bank.role == "hard_wrong"
                    and query["donors"][index] == hard_wrong_bank_donors[name]
                )
            ],
            dtype=np.int64,
        )
        missing = [
            int(index)
            for index in eligible_indices
            if min(len(matched_candidates[index]), len(comparator_candidates[index])) < 2
        ]
        if missing:
            raise ValueError(
                "every eligible query needs two exactly matched candidates in bank %s; "
                "failed rows: %s" % (name, ",".join(map(str, missing[:10])))
            )
        comparison_donors = sorted(set(query["donors"][eligible_indices].tolist()))
        comparison_minimum_donors = int(validated_power["minimum_comparison_query_donors"])
        if len(comparison_donors) < comparison_minimum_donors:
            raise ValueError(
                "reference comparison is below its power-justified donor minimum: %s" % name
            )
        pair_sizes = np.asarray(
            [
                min(len(matched_candidates[index]), len(comparator_candidates[index]))
                for index in eligible_indices
            ],
            dtype=np.int64,
        )
        matched_values = []
        comparator_values = []
        donor_matched = {donor: [] for donor in comparison_donors}
        donor_comparator = {donor: [] for donor in comparison_donors}
        for _ in range(repeats):
            matched_error = np.full(rows, np.nan, dtype=np.float64)
            comparator_error = np.full(rows, np.nan, dtype=np.float64)
            for index, sample_size in zip(eligible_indices, pair_sizes):
                selected_matched = rng.choice(
                    matched_candidates[index], size=int(sample_size), replace=False
                )
                selected_comparator = rng.choice(
                    comparator_candidates[index], size=int(sample_size), replace=False
                )
                matched_latent = matched_bank.latent[selected_matched]
                comparator_latent = bank.latent[selected_comparator]
                matched_nearest = int(
                    np.argmin(np.square(matched_latent - image[index]).sum(axis=1))
                )
                comparator_nearest = int(
                    np.argmin(np.square(comparator_latent - image[index]).sum(axis=1))
                )
                matched_error[index] = float(
                    np.square(matched_latent[matched_nearest] - target[index]).mean()
                )
                comparator_error[index] = float(
                    np.square(comparator_latent[comparator_nearest] - target[index]).mean()
                )
            matched_macro, matched_by_donor = _aggregate_donor_type(
                matched_error, eligible_indices, query["donors"], query["types"]
            )
            comparator_macro, comparator_by_donor = _aggregate_donor_type(
                comparator_error, eligible_indices, query["donors"], query["types"]
            )
            matched_values.append(matched_macro)
            comparator_values.append(comparator_macro)
            for donor in comparison_donors:
                donor_matched[donor].append(matched_by_donor[donor])
                donor_comparator[donor].append(comparator_by_donor[donor])
        matched_values_array = np.asarray(matched_values)
        comparator_values_array = np.asarray(comparator_values)
        donor_absolute_effects = np.asarray(
            [
                np.mean(donor_comparator[donor]) - np.mean(donor_matched[donor])
                for donor in comparison_donors
            ],
            dtype=np.float64,
        )
        donor_relative_effects = np.asarray(
            [
                absolute / max(float(np.mean(donor_comparator[donor])), np.finfo(float).eps)
                for donor, absolute in zip(comparison_donors, donor_absolute_effects)
            ],
            dtype=np.float64,
        )
        lower, upper = _bootstrap_interval(
            donor_relative_effects,
            seed=seed + 1009 * (position + 1),
            samples=bootstrap_samples,
            confidence=confidence,
        )
        target_variance = float(np.mean(np.var(target[eligible_indices], axis=0, ddof=1)))
        if target_variance <= np.finfo(float).eps:
            raise ValueError("reference-utility target variance is zero for comparison: %s" % name)
        positive_fraction = float(np.mean(donor_relative_effects > 0))
        comparison_pass = bool(
            lower >= minimum_relative_effect
            and positive_fraction >= minimum_positive_donor_fraction
        )
        comparisons.append(
            {
                "bank": name,
                "bank_role": bank.role,
                "hard_wrong_donor_id": hard_wrong_bank_donors.get(name),
                "relative_error_reduction": float(donor_relative_effects.mean()),
                "relative_error_reduction_donor_confidence_interval": [lower, upper],
                "variance_normalized_mse_reduction": float(
                    donor_absolute_effects.mean() / target_variance
                ),
                "fraction_query_donors_positive": positive_fraction,
                "fraction_subsamples_matched_better": float(
                    np.mean(comparator_values_array > matched_values_array)
                ),
                "matched_mean_mse": float(matched_values_array.mean()),
                "comparator_mean_mse": float(comparator_values_array.mean()),
                "per_donor_relative_error_reduction": {
                    donor: float(value)
                    for donor, value in zip(comparison_donors, donor_relative_effects)
                },
                "per_donor_absolute_mse_reduction": {
                    donor: float(value)
                    for donor, value in zip(comparison_donors, donor_absolute_effects)
                },
                "query_count": int(len(eligible_indices)),
                "query_donor_count": len(comparison_donors),
                "excluded_same_donor_query_count": rows - int(len(eligible_indices)),
                "per_query_equalized_bank_size": {
                    "minimum": int(pair_sizes.min()),
                    "median": float(np.median(pair_sizes)),
                    "maximum": int(pair_sizes.max()),
                },
                "passes_prespecified_effect": comparison_pass,
            }
        )
    hard_wrong_rows = [row for row in comparisons if row["bank_role"] == "hard_wrong"]
    worst_hard_wrong = min(hard_wrong_rows, key=lambda row: row["relative_error_reduction"])
    passed = coverage_pass and all(row["passes_prespecified_effect"] for row in comparisons)
    return {
        "schema": "heir.matched_reference_utility.v2",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "image_conditioned": True,
        "frozen_image_queries": True,
        "locked_molecular_targets_unchanged": True,
        "only_reference_bank_changes": True,
        "morphology_evidence_binding": {
            "frozen_image_model_sha256": frozen_image_hash,
            "query_source_sha256": query_source_hash,
            "primary_report_sha256": primary_report_hash,
            "external_report_sha256s": list(external_report_hashes),
        },
        "repeats": repeats,
        "seed": seed,
        "minimum_relative_error_reduction": minimum_relative_effect,
        "minimum_positive_donor_fraction": minimum_positive_donor_fraction,
        "donor_bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "latent_model_sha256": query_latent_hash,
        "normalization_sha256": query_normalization_hash,
        "assay_harmonization_sha256": query_harmonization_hash,
        "assay_harmonization_source_sha256": query_harmonization_source_hash,
        "assay_harmonization_fit_donor_ids": list(query_harmonization_fit_donors),
        "calibrated_assay_pairs": list(query_calibrated_pairs),
        "assay_mode": assay_mode,
        "latent_fit_scope": "development_or_external_donors_only",
        "latent_fit_donor_ids": list(latent_fit_donors),
        "power_analysis": validated_power,
        "query_preservation_methods": sorted(set(query["preservation"].tolist())),
        "reference_bank_biological_provenance": {
            name: {
                "material_relationship_to_query": bank.material_relationship_to_query,
                "independent_tissue_material": bank.independent_tissue_material,
                "contains_registered_query_cells": bank.contains_registered_query_cells,
                "preservation_methods": sorted(set(bank.preservation_methods.tolist())),
                "assay_ids": sorted(set(bank.assay_ids.tolist())),
                "source_sha256": bank.source_sha256,
            }
            for name, bank in sorted(parsed.items())
        },
        "query_count": rows,
        "donor_count": len(included_donors),
        "query_population_coverage": {
            "total_eligible_query_count": total_eligible,
            "included_query_count": rows,
            "excluded_query_count": total_eligible - rows,
            "excluded_query_count_by_reason": exclusions,
            "retained_fraction": query_coverage,
            "minimum_retained_fraction": minimum_query_coverage,
            "eligible_donor_ids": list(eligible_donors),
            "included_donor_ids": included_donors,
            "lost_donor_ids": lost_donors,
            "eligible_type_labels": list(eligible_types),
            "included_type_labels": included_types,
            "lost_type_labels": lost_types,
            "power_analysis_sha256": power_analysis_sha256,
            "power_justified_minimum_donors": power_minimum_donors,
            "require_full_donor_and_type_coverage": require_full_donor_and_type_coverage,
            "pass": coverage_pass,
        },
        "eligible_hard_wrong_donor_ids": list(eligible_wrong_donors),
        "hard_wrong_bank_count": len(hard_wrong_banks),
        "all_eligible_hard_wrong_donors_tested": True,
        "worst_hard_wrong_comparator": {
            "bank": worst_hard_wrong["bank"],
            "donor_id": worst_hard_wrong["hard_wrong_donor_id"],
            "relative_error_reduction": worst_hard_wrong["relative_error_reduction"],
            "pass": worst_hard_wrong["passes_prespecified_effect"],
        },
        "matching_factors": [
            "type",
            "disease",
            "site",
            "institution",
            "quality_bin",
            "depth_bin",
        ]
        + (["assay"] if assay_mode == "same_assay" else ["development_calibrated_assay_domain"]),
        "aggregation": "equal_donor_equal_type",
        "comparisons": comparisons,
        "scientific_scope": "image_conditioned_reference_bank_substitution",
    }


__all__ = ["ReferenceBank", "evaluate_reference_utility"]
