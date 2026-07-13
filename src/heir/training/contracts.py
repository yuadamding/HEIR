"""Strict provenance contracts for personalized morphology/RNA training.

The molecular E-step is intentionally a separate, pickle-free artifact.  A
personalized M-step may consume its fixed responsibilities, but it must never
recreate those targets from the live student while optimizing that student.
"""

import hashlib
import io
import json
import math
import os
import pickle
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, os.PathLike]


def ordered_identity_sha256(values: Sequence[object]) -> str:
    """Hash an ordered string identity without locale-dependent formatting."""

    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def donor_cross_type_permutation(
    donors: Sequence[object], labels: object, seed: int
) -> Tuple[np.ndarray, Mapping[str, float]]:
    """Build the deterministic donor-local cross-type image-shuffle null.

    The largest type group determines the rotation, which maximizes cross-type
    reassignment subject to staying within each donor. Keeping this algorithm
    in the shared contract lets receipt consumers reproduce the exact control
    identities instead of trusting producer-reported hashes and mismatch rates.
    """

    donor_array = np.asarray([str(value) for value in donors])
    label_array = np.asarray(labels)
    if donor_array.ndim != 1 or label_array.shape != donor_array.shape:
        raise ValueError("shuffle-control donor and label arrays must align")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("shuffle-control seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    result = np.arange(len(donor_array), dtype=np.int64)
    mismatch_by_donor = {}
    for donor in sorted(set(donor_array.tolist())):
        indices = np.flatnonzero(donor_array == donor)
        donor_labels = label_array[indices]
        unique_labels = np.unique(donor_labels)
        if len(unique_labels) < 2:
            raise ValueError("every held-out donor needs at least two types for shuffle controls")
        label_order = np.array(unique_labels, copy=True)
        rng.shuffle(label_order)
        groups = []
        maximum_group = 0
        for label in label_order:
            group = np.array(indices[donor_labels == label], copy=True)
            rng.shuffle(group)
            groups.append(group)
            maximum_group = max(maximum_group, len(group))
        ordered = np.concatenate(groups)
        shuffled = np.roll(ordered, -maximum_group)
        result[ordered] = shuffled
        mismatch = float(np.mean(label_array[ordered] != label_array[shuffled]))
        if mismatch < 0.5:
            raise ValueError(
                "held-out donor %s is too type-imbalanced for a robust shuffle null" % donor
            )
        mismatch_by_donor[donor] = mismatch
    return result, mismatch_by_donor


def validate_primary_claim_exclusions(
    metadata: Mapping[str, object], *, artifact: str = "checkpoint"
) -> None:
    """Reject non-canonical or excluded primary-claim checkpoint metadata.

    Historical truthiness checks accidentally accepted malformed values such
    as ``0`` or an empty string for the exclusion flag.  A primary artifact is
    eligible only when the flag is absent or the JSON/Python boolean ``False``
    and reasons are absent or a canonical empty list.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("%s lacks provenance metadata" % artifact)
    if "excluded_from_primary_claims" in metadata:
        excluded = metadata["excluded_from_primary_claims"]
        if excluded is not False:
            if excluded is True:
                raise ValueError("excluded %s cannot support primary claims" % artifact)
            raise ValueError("%s primary-claim exclusion flag is malformed" % artifact)
    if "exclusion_reasons" in metadata:
        reasons = metadata["exclusion_reasons"]
        if not isinstance(reasons, list):
            raise ValueError("%s primary-claim exclusion reasons are malformed" % artifact)
        if reasons:
            raise ValueError("excluded %s cannot support primary claims" % artifact)


def _initialization_latent_metrics(predicted: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
    if predicted.shape != truth.shape:
        raise ValueError("image-to-latent predictions and targets must align")
    predicted_norm = np.linalg.norm(predicted, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    valid = (predicted_norm > 1.0e-12) & (truth_norm > 1.0e-12)
    if not np.any(valid):
        raise ValueError("initialization evidence has no nonzero latent pairs")
    cosine = np.sum(predicted[valid] * truth[valid], axis=1) / (
        predicted_norm[valid] * truth_norm[valid]
    )
    rmse = np.sqrt(np.mean(np.square(predicted - truth), dtype=np.float64))
    return float(np.mean(cosine, dtype=np.float64)), float(rmse)


def _initialization_classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_types: int,
) -> Mapping[str, float]:
    if probabilities.shape != (len(labels), num_types):
        raise ValueError("initializer probabilities and labels must align")
    if not np.isfinite(probabilities).all():
        raise ValueError("initializer probabilities must be finite")
    predicted = probabilities.argmax(axis=1)
    per_type_f1 = []
    for type_index in range(num_types):
        true_positive = int(np.sum((predicted == type_index) & (labels == type_index)))
        false_positive = int(np.sum((predicted == type_index) & (labels != type_index)))
        false_negative = int(np.sum((predicted != type_index) & (labels == type_index)))
        denominator = 2 * true_positive + false_positive + false_negative
        per_type_f1.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    one_hot = np.eye(num_types, dtype=np.float64)[labels]
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        selected = (confidence > boundaries[index]) & (confidence <= boundaries[index + 1])
        if index == 0:
            selected |= confidence == 0.0
        if np.any(selected):
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    support = np.bincount(labels, minlength=num_types)
    return {
        "macro_f1": float(np.mean(per_type_f1, dtype=np.float64)),
        "ece": ece,
        "brier": brier,
        "predicted_class_occupancy_fraction": float(len(np.unique(predicted)) / num_types),
        "minimum_per_type_support": float(support.min()),
    }


def recompute_initialization_validation(
    *,
    checkpoint: Mapping[str, object],
    morphology: object,
    edge_index: object,
    edge_weight: object,
    labels: object,
    target_latent: object,
    donor_ids: Sequence[object],
    seeds: Sequence[int],
) -> Mapping[str, object]:
    """Replay a checkpoint and derive every initializer-validation metric.

    Both report production and receipt consumption use this exact path.  It is
    intentionally CPU-only, evaluation-mode, inference-only, and float32 so a
    receipt never depends on CUDA kernels, autocast, dropout, or a producer's
    precomputed predictions.
    """

    try:
        import torch

        from heir.models import HEIRModel
    except ImportError as error:  # pragma: no cover - package installation failure
        raise ValueError("HEIR checkpoint replay dependencies are unavailable") from error

    if not isinstance(checkpoint, Mapping):
        raise ValueError("initializer checkpoint must contain a mapping")
    try:
        model = HEIRModel.from_checkpoint(checkpoint).to(device="cpu", dtype=torch.float32).eval()
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("initialization checkpoint cannot be replayed as a HEIR model") from error
    morphology_array = np.ascontiguousarray(np.asarray(morphology, dtype=np.float32))
    edge_array = np.ascontiguousarray(np.asarray(edge_index, dtype=np.int64))
    weight_array = (
        None
        if edge_weight is None
        else np.ascontiguousarray(np.asarray(edge_weight, dtype=np.float32))
    )
    label_array = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    target_array = np.ascontiguousarray(np.asarray(target_latent, dtype=np.float32))
    donor_array = np.asarray([str(value) for value in donor_ids])
    cells = len(morphology_array)
    if (
        morphology_array.ndim != 2
        or morphology_array.shape[1] != model.config.morphology_dim
        or edge_array.ndim != 2
        or edge_array.shape[0] != 2
        or (weight_array is not None and weight_array.shape != (edge_array.shape[1],))
        or label_array.shape != (cells,)
        or target_array.shape != (cells, model.config.latent_dim)
        or donor_array.shape != (cells,)
        or not np.isfinite(morphology_array).all()
        or not np.isfinite(target_array).all()
        or (weight_array is not None and not np.isfinite(weight_array).all())
    ):
        raise ValueError("initialization checkpoint replay inputs are malformed")
    if np.any(label_array < 0) or np.any(label_array >= model.config.num_cell_types):
        raise ValueError("initialization checkpoint replay labels are outside the ontology")
    if edge_array.size and (edge_array.min() < 0 or edge_array.max() >= cells):
        raise ValueError("initialization checkpoint replay graph is out of range")
    raw_seeds = tuple(seeds)
    if (
        len(raw_seeds) < 3
        or any(
            isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            for seed in raw_seeds
        )
        or len(set(int(seed) for seed in raw_seeds)) != len(raw_seeds)
    ):
        raise ValueError("initialization checkpoint replay seeds are malformed")

    morphology_tensor = torch.from_numpy(morphology_array)
    edge_tensor = torch.from_numpy(edge_array)
    weight_tensor = None if weight_array is None else torch.from_numpy(weight_array)

    def predict(features: "torch.Tensor") -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            _, probabilities, image_latent = model.encode_frozen_morphology(
                features,
                edge_tensor,
                weight_tensor,
            )
        return (
            probabilities.to(dtype=torch.float32).cpu().numpy(),
            image_latent.to(dtype=torch.float32).cpu().numpy(),
        )

    real_probabilities, real_latent = predict(morphology_tensor)
    pooled_classification = _initialization_classification_metrics(
        real_probabilities, label_array, model.config.num_cell_types
    )
    pooled_cosine, pooled_rmse = _initialization_latent_metrics(real_latent, target_array)
    pooled = {
        **pooled_classification,
        "latent_cosine": pooled_cosine,
        "latent_rmse": pooled_rmse,
    }
    held_out_donors = tuple(sorted(set(donor_array.tolist())))
    donor_metrics = []
    for donor in held_out_donors:
        selected = donor_array == donor
        classification = _initialization_classification_metrics(
            real_probabilities[selected], label_array[selected], model.config.num_cell_types
        )
        cosine, rmse = _initialization_latent_metrics(real_latent[selected], target_array[selected])
        donor_metrics.append(
            {
                "donor_id": donor,
                **classification,
                "latent_cosine": cosine,
                "latent_rmse": rmse,
            }
        )
    donor_rows = {str(row["donor_id"]): row for row in donor_metrics}
    controls = []
    for raw_seed in raw_seeds:
        seed = int(raw_seed)
        permutation, mismatch_by_donor = donor_cross_type_permutation(
            donor_array.tolist(), label_array, seed
        )
        shuffled_probabilities, shuffled_latent = predict(morphology_tensor[permutation])
        for donor in held_out_donors:
            selected = donor_array == donor
            shuffled_classification = _initialization_classification_metrics(
                shuffled_probabilities[selected],
                label_array[selected],
                model.config.num_cell_types,
            )
            shuffled_cosine, _ = _initialization_latent_metrics(
                shuffled_latent[selected], target_array[selected]
            )
            controls.append(
                {
                    "seed": seed,
                    "donor_id": donor,
                    "permutation_sha256": ordered_identity_sha256(permutation[selected].tolist()),
                    "cross_type_mismatch_fraction": mismatch_by_donor[donor],
                    "image_shuffle_macro_f1": shuffled_classification["macro_f1"],
                    "real_minus_image_shuffle_macro_f1": (
                        donor_rows[donor]["macro_f1"] - shuffled_classification["macro_f1"]
                    ),
                    "image_shuffle_latent_cosine": shuffled_cosine,
                    "real_minus_image_shuffle_latent_cosine": (
                        donor_rows[donor]["latent_cosine"] - shuffled_cosine
                    ),
                }
            )
    return {
        "metrics": pooled,
        "donor_metrics": donor_metrics,
        "shuffle_controls": controls,
    }


def array_content_sha256(value: object) -> str:
    """Hash an array's exact dtype, shape, order-normalized values, and bytes."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("content hashing does not permit object arrays")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def frozen_transport_telemetry(
    *,
    raw_transport_plan: object,
    transport_cost: object,
    source_mass: object,
    target_weights: object,
    fixed_unknown_mass: float,
    epsilon: float,
    marginal_relaxation: float,
) -> Mapping[str, float]:
    """Recompute solver telemetry from the exact serialized raw coupling."""

    raw = np.asarray(raw_transport_plan, dtype=np.float64)
    cost = np.asarray(transport_cost, dtype=np.float64)
    source = np.asarray(source_mass, dtype=np.float64)
    target = np.asarray(target_weights, dtype=np.float64)
    if (
        raw.ndim != 2
        or cost.shape != raw.shape
        or source.shape != (raw.shape[0],)
        or target.shape != (raw.shape[1] - 1,)
        or not np.isfinite(raw).all()
        or not np.isfinite(cost).all()
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or np.any(raw < 0)
        or np.any(source < 0)
        or np.any(target < 0)
        or float(source.sum()) <= 0
        or float(target.sum()) <= 0
    ):
        raise ValueError("frozen transport telemetry inputs are malformed")
    if (
        not math.isfinite(fixed_unknown_mass)
        or not 0.0 <= fixed_unknown_mass < 1.0
        or not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(marginal_relaxation)
        or marginal_relaxation <= 0
    ):
        raise ValueError("frozen transport telemetry parameters are malformed")
    desired_source = source / source.sum(dtype=np.float64)
    desired_target = np.concatenate(
        (
            target / target.sum(dtype=np.float64) * (1.0 - fixed_unknown_mass),
            np.asarray([fixed_unknown_mass], dtype=np.float64),
        )
    )
    raw_source = raw.sum(axis=1)
    raw_target = raw.sum(axis=0)
    epsilon_floor = 1.0e-8

    def generalized_kl(marginal: np.ndarray, desired: np.ndarray) -> float:
        selected = desired > 0
        safe_marginal = np.maximum(marginal[selected], epsilon_floor)
        safe_desired = np.maximum(desired[selected], epsilon_floor)
        terms = (
            marginal[selected] * (np.log(safe_marginal) - np.log(safe_desired))
            - marginal[selected]
            + desired[selected]
        )
        return float(terms.sum())

    objective = float((raw * cost).sum())
    objective += epsilon * float((raw * (np.log(np.maximum(raw, epsilon_floor)) - 1.0)).sum())
    objective += marginal_relaxation * (
        generalized_kl(raw_source, desired_source) + generalized_kl(raw_target, desired_target)
    )
    return {
        "solver_source_marginal_error": float(np.abs(raw_source - desired_source).sum()),
        "solver_target_marginal_error": float(np.abs(raw_target - desired_target).sum()),
        "transport_objective": objective,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_string_vector(value: np.ndarray, name: str) -> Tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("%s must be one-dimensional" % name)
    result = tuple(str(item) for item in array.tolist())
    if any(not item.strip() for item in result) or len(set(result)) != len(result):
        raise ValueError("%s must contain unique non-empty strings" % name)
    return result


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> object:
    if name not in archive:
        raise ValueError("molecular E-step artifact is missing %s" % name)
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError("molecular E-step field %s must be scalar" % name)
    return value.item()


def _write_deterministic_npz(path: PathLike, payload: Mapping[str, np.ndarray]) -> None:
    """Atomically write byte-deterministic, pickle-free NPZ content."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".npz.tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(payload):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.asarray(payload[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo("%s.npy" % name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


_INITIALIZATION_THRESHOLD_KEYS = frozenset(
    {
        "minimum_macro_f1",
        "minimum_image_shuffle_macro_f1_delta",
        "minimum_latent_cosine",
        "minimum_image_shuffle_latent_cosine_delta",
        "maximum_latent_rmse",
        "maximum_ece",
        "maximum_brier",
        "minimum_predicted_class_occupancy_fraction",
        "minimum_per_type_support",
    }
)
_INITIALIZATION_CHECK_KEYS = frozenset(
    {
        "macro_f1",
        "image_shuffle_macro_f1_delta",
        "latent_cosine",
        "image_shuffle_latent_cosine_delta",
        "latent_rmse",
        "ece",
        "brier",
        "predicted_class_occupancy",
        "per_type_support",
    }
)


def _report_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("initialization evidence %s must be numeric" % name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("initialization evidence %s must be finite" % name)
    return result


def _strict_string_sequence(value: object, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("initialization evidence %s must be a JSON array" % name)
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result) or len(set(result)) != len(result):
        raise ValueError("initialization evidence %s must be unique and non-empty" % name)
    return result


def validate_initialization_evidence_report(
    path: PathLike,
    *,
    checkpoint_sha256: str,
    feature_space_id: str,
    latent_space_id: str,
    type_ontology_sha256: str,
    training_donors: Sequence[str],
    held_out_donors: Sequence[str],
) -> Mapping[str, object]:
    """Validate a v1 initializer report against its prespecified plan.

    This is deliberately shared by receipt creation and receipt consumption.
    Hashing an arbitrary JSON file is not validation: the report's donor/seed
    rows and all nine gates are recomputed here from the bound plan.
    """

    source = Path(path).expanduser().resolve()
    try:
        with source.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("initialization evidence report is not valid JSON") from error
    required_report_keys = {
        "schema",
        "status",
        "pass",
        "checkpoint",
        "plan",
        "evidence_artifact",
        "label_source",
        "latent_target_source",
        "feature_space_id",
        "latent_space_id",
        "type_ontology_sha256",
        "training_donors",
        "held_out_donors",
        "capabilities",
        "thresholds",
        "metrics",
        "donor_metrics",
        "shuffle_controls",
        "checks",
        "execution",
    }
    if not isinstance(report, Mapping) or set(report) != required_report_keys:
        raise ValueError("initialization evidence report schema is invalid")
    if (
        report.get("schema") != "heir.initialization_validation_evidence.v1"
        or report.get("status") != "complete"
    ):
        raise ValueError("initialization evidence report schema is invalid")

    def bound_artifact(
        record: object,
        *,
        base: Path,
        label: str,
    ) -> Tuple[Path, str]:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError("initialization evidence report %s binding is malformed" % label)
        digest = str(record.get("sha256", ""))
        if not _is_sha256(digest):
            raise ValueError("initialization evidence report %s hash is malformed" % label)
        artifact = Path(str(record.get("path", ""))).expanduser()
        if not artifact.is_absolute():
            artifact = base / artifact
        artifact = artifact.resolve()
        if not artifact.is_file() or _sha256_path(artifact) != digest:
            raise ValueError("initialization evidence report %s binding is stale" % label)
        return artifact, digest

    report_checkpoint, report_checkpoint_sha256 = bound_artifact(
        report["checkpoint"], base=source.parent, label="checkpoint"
    )
    plan_path, plan_sha256 = bound_artifact(report["plan"], base=source.parent, label="plan")
    report_evidence, report_evidence_sha256 = bound_artifact(
        report["evidence_artifact"], base=source.parent, label="evidence artifact"
    )
    report_labels, report_labels_sha256 = bound_artifact(
        report["label_source"], base=source.parent, label="label source"
    )
    report_latent, report_latent_sha256 = bound_artifact(
        report["latent_target_source"], base=source.parent, label="latent target source"
    )
    if report_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("initialization evidence report checkpoint binding differs")
    try:
        import torch

        try:
            checkpoint_payload = torch.load(
                report_checkpoint, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint_payload = torch.load(report_checkpoint, map_location="cpu")
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError("initialization checkpoint is not a valid checkpoint artifact") from error
    if not isinstance(checkpoint_payload, Mapping) or not isinstance(
        checkpoint_payload.get("metadata"), Mapping
    ):
        raise ValueError("initialization checkpoint lacks provenance metadata")
    checkpoint_metadata = checkpoint_payload["metadata"]
    validate_primary_claim_exclusions(
        checkpoint_metadata,
        artifact="initialization checkpoint",
    )
    checkpoint_training = tuple(
        sorted(str(value) for value in checkpoint_metadata.get("training_donors", ()))
    )
    checkpoint_type_names = tuple(str(value) for value in checkpoint_metadata.get("type_names", ()))
    if (
        checkpoint_training != tuple(sorted(str(value) for value in training_donors))
        or str(checkpoint_metadata.get("feature_space_id", "")) != feature_space_id
        or str(checkpoint_metadata.get("latent_space_id", "")) != latent_space_id
        or ordered_identity_sha256(checkpoint_type_names) != type_ontology_sha256
    ):
        raise ValueError("initialization checkpoint provenance differs from the receipt scope")
    validation_source_hashes = {
        report_evidence_sha256,
        report_labels_sha256,
        report_latent_sha256,
    }
    if checkpoint_sha256 in validation_source_hashes or len(validation_source_hashes) != 3:
        raise ValueError("initialization evidence sources are not independent artifacts")

    try:
        with plan_path.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("initialization validation plan is not valid JSON") from error
    required_plan_keys = {
        "schema",
        "status",
        "checkpoint",
        "evaluation_artifact",
        "label_source",
        "latent_target_source",
        "held_out_donors",
        "seeds",
        "thresholds",
    }
    if (
        not isinstance(plan, Mapping)
        or set(plan) != required_plan_keys
        or plan.get("schema") != "heir.initialization_validation_plan.v1"
        or plan.get("status") != "ready"
    ):
        raise ValueError("initialization validation plan schema is invalid")
    plan_checkpoint, plan_checkpoint_sha256 = bound_artifact(
        plan["checkpoint"], base=plan_path.parent, label="plan checkpoint"
    )
    plan_evidence, plan_evidence_sha256 = bound_artifact(
        plan["evaluation_artifact"], base=plan_path.parent, label="plan evidence artifact"
    )
    plan_labels, plan_labels_sha256 = bound_artifact(
        plan["label_source"], base=plan_path.parent, label="plan label source"
    )
    plan_latent, plan_latent_sha256 = bound_artifact(
        plan["latent_target_source"], base=plan_path.parent, label="plan latent target source"
    )
    if (
        (report_checkpoint, report_checkpoint_sha256) != (plan_checkpoint, plan_checkpoint_sha256)
        or (report_evidence, report_evidence_sha256) != (plan_evidence, plan_evidence_sha256)
        or (report_labels, report_labels_sha256) != (plan_labels, plan_labels_sha256)
        or (report_latent, report_latent_sha256) != (plan_latent, plan_latent_sha256)
        or plan_sha256 != str(report["plan"]["sha256"])
    ):
        raise ValueError("initialization evidence report differs from its prespecified plan")

    expected_training = tuple(sorted(str(value) for value in training_donors))
    expected_holdout = tuple(sorted(str(value) for value in held_out_donors))
    report_training = tuple(
        sorted(_strict_string_sequence(report["training_donors"], "training_donors"))
    )
    report_holdout = tuple(
        sorted(_strict_string_sequence(report["held_out_donors"], "held_out_donors"))
    )
    plan_holdout = tuple(
        sorted(_strict_string_sequence(plan["held_out_donors"], "plan held_out_donors"))
    )
    if report_training != expected_training or report_holdout != expected_holdout:
        raise ValueError("initialization evidence report donor scope differs")
    if plan_holdout != expected_holdout:
        raise ValueError("initialization validation plan held-out donors differ")
    if report.get("feature_space_id") != feature_space_id:
        raise ValueError("initialization evidence report feature space differs")
    if report.get("latent_space_id") != latent_space_id:
        raise ValueError("initialization evidence report latent space differs")
    if report.get("type_ontology_sha256") != type_ontology_sha256:
        raise ValueError("initialization evidence report ontology differs")
    capabilities = report.get("capabilities")
    if capabilities != {"broad_type": True, "image_to_latent": True}:
        raise ValueError("initialization evidence report lacks required capabilities")

    def npz_scalar(archive: Mapping[str, np.ndarray], name: str) -> object:
        if name not in archive or np.asarray(archive[name]).ndim != 0:
            raise ValueError("initialization validation source %s is malformed" % name)
        return np.asarray(archive[name]).item()

    try:
        with np.load(report_labels, allow_pickle=False) as archive:
            required = {
                "schema",
                "nucleus_ids",
                "donor_ids",
                "type_labels",
                "type_names",
                "independent_of_checkpoint",
            }
            if not required.issubset(archive.files):
                raise ValueError("independent initialization label source is incomplete")
            if (
                str(npz_scalar(archive, "schema")) != "heir.independent_initialization_labels.v1"
                or npz_scalar(archive, "independent_of_checkpoint") is not True
            ):
                raise ValueError("independent initialization label source is invalid")
            source_nucleus_ids = np.asarray(archive["nucleus_ids"]).astype(str)
            source_donor_ids = np.asarray(archive["donor_ids"]).astype(str)
            source_type_labels = np.asarray(archive["type_labels"])
            source_type_names = np.asarray(archive["type_names"]).astype(str)
        with np.load(report_latent, allow_pickle=False) as archive:
            required = {
                "schema",
                "nucleus_ids",
                "target_latent",
                "latent_space_id",
                "independent_of_checkpoint",
            }
            if not required.issubset(archive.files):
                raise ValueError("registered latent-target source is incomplete")
            if (
                str(npz_scalar(archive, "schema")) != "heir.registered_image_latent_targets.v1"
                or npz_scalar(archive, "independent_of_checkpoint") is not True
            ):
                raise ValueError("registered latent-target source is invalid")
            latent_nucleus_ids = np.asarray(archive["nucleus_ids"]).astype(str)
            source_target_latent = np.asarray(archive["target_latent"], dtype=np.float32)
            source_latent_space = str(npz_scalar(archive, "latent_space_id"))
        with np.load(report_evidence, allow_pickle=False) as archive:
            required = {
                "morphology",
                "edge_index",
                "nucleus_ids",
                "donor_ids",
                "type_labels",
                "type_names",
                "target_latent",
                "feature_space_id",
                "latent_space_id",
                "label_source_sha256",
                "latent_target_source_sha256",
                "labels_independent_of_checkpoint",
                "latent_targets_independent_of_checkpoint",
            }
            if not required.issubset(archive.files):
                raise ValueError("initialization evaluation artifact is incomplete")
            evidence_morphology = np.asarray(archive["morphology"], dtype=np.float32)
            evidence_edges = np.asarray(archive["edge_index"])
            evidence_edge_weight = (
                np.asarray(archive["edge_weight"], dtype=np.float32)
                if "edge_weight" in archive
                else None
            )
            evidence_nucleus_ids = np.asarray(archive["nucleus_ids"]).astype(str)
            evidence_donor_ids = np.asarray(archive["donor_ids"]).astype(str)
            evidence_type_labels = np.asarray(archive["type_labels"])
            evidence_type_names = np.asarray(archive["type_names"]).astype(str)
            evidence_target_latent = np.asarray(archive["target_latent"], dtype=np.float32)
            evidence_feature_space = str(npz_scalar(archive, "feature_space_id"))
            evidence_latent_space = str(npz_scalar(archive, "latent_space_id"))
            evidence_label_hash = str(npz_scalar(archive, "label_source_sha256"))
            evidence_latent_hash = str(npz_scalar(archive, "latent_target_source_sha256"))
            labels_independent = npz_scalar(archive, "labels_independent_of_checkpoint")
            latent_independent = npz_scalar(archive, "latent_targets_independent_of_checkpoint")
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            (
                "independent initialization",
                "registered latent",
                "initialization evaluation",
                "initialization validation source",
            )
        ):
            raise
        raise ValueError("initialization validation sources are not valid NPZ artifacts") from error

    if evidence_nucleus_ids.ndim != 1:
        raise ValueError("initialization validation nucleus identities are malformed")
    cells = len(evidence_nucleus_ids)
    source_arrays_align = (
        source_nucleus_ids.ndim == 1
        and source_donor_ids.shape == (cells,)
        and source_type_labels.shape == (cells,)
        and source_type_names.ndim == 1
        and latent_nucleus_ids.shape == (cells,)
        and source_target_latent.ndim == 2
        and source_target_latent.shape[0] == cells
        and evidence_morphology.ndim == 2
        and evidence_morphology.shape[0] == cells
        and evidence_donor_ids.shape == (cells,)
        and evidence_type_labels.shape == (cells,)
        and evidence_type_names.ndim == 1
        and evidence_target_latent.shape == source_target_latent.shape
    )
    if (
        not source_arrays_align
        or cells == 0
        or len(set(evidence_nucleus_ids.tolist())) != cells
        or not np.issubdtype(source_type_labels.dtype, np.integer)
        or not np.issubdtype(evidence_type_labels.dtype, np.integer)
        or not np.issubdtype(evidence_edges.dtype, np.integer)
        or evidence_edges.ndim != 2
        or evidence_edges.shape[0] != 2
        or (evidence_edges.size and (evidence_edges.min() < 0 or evidence_edges.max() >= cells))
        or (
            evidence_edge_weight is not None
            and (
                evidence_edge_weight.shape != (evidence_edges.shape[1],)
                or not np.isfinite(evidence_edge_weight).all()
                or np.any(evidence_edge_weight < 0)
            )
        )
        or not np.isfinite(evidence_morphology).all()
        or not np.isfinite(evidence_target_latent).all()
        or not np.array_equal(source_nucleus_ids, evidence_nucleus_ids)
        or not np.array_equal(latent_nucleus_ids, evidence_nucleus_ids)
        or not np.array_equal(source_donor_ids, evidence_donor_ids)
        or not np.array_equal(source_type_labels, evidence_type_labels)
        or not np.array_equal(source_type_names, evidence_type_names)
        or not np.array_equal(source_target_latent, evidence_target_latent)
        or source_latent_space != latent_space_id
        or evidence_latent_space != latent_space_id
        or evidence_feature_space != feature_space_id
        or evidence_label_hash != report_labels_sha256
        or evidence_latent_hash != report_latent_sha256
        or labels_independent is not True
        or latent_independent is not True
        or ordered_identity_sha256(evidence_type_names.tolist()) != type_ontology_sha256
        or set(evidence_donor_ids.tolist()) != set(expected_holdout)
    ):
        raise ValueError("initialization validation sources are inconsistent")
    if evidence_edges.size and np.any(
        evidence_donor_ids[evidence_edges[0]] != evidence_donor_ids[evidence_edges[1]]
    ):
        raise ValueError("initialization validation graph connects different donors")
    num_types = len(evidence_type_names)
    if (
        num_types == 0
        or np.any(evidence_type_labels < 0)
        or np.any(evidence_type_labels >= num_types)
    ):
        raise ValueError("initialization validation labels are outside the ontology")
    thresholds = report.get("thresholds")
    plan_thresholds = plan.get("thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or not isinstance(plan_thresholds, Mapping)
        or set(thresholds) != _INITIALIZATION_THRESHOLD_KEYS
        or set(plan_thresholds) != _INITIALIZATION_THRESHOLD_KEYS
    ):
        raise ValueError("initialization evidence thresholds are malformed")
    threshold_values = {
        name: _report_number(thresholds[name], "threshold %s" % name)
        for name in _INITIALIZATION_THRESHOLD_KEYS
    }
    plan_threshold_values = {
        name: _report_number(plan_thresholds[name], "plan threshold %s" % name)
        for name in _INITIALIZATION_THRESHOLD_KEYS
    }
    if threshold_values != plan_threshold_values:
        raise ValueError("initialization evidence thresholds differ from the plan")
    if (
        not 0.65 <= threshold_values["minimum_macro_f1"] <= 1.0
        or not 0.05 <= threshold_values["minimum_image_shuffle_macro_f1_delta"] <= 1.0
        or not 0.0 <= threshold_values["minimum_latent_cosine"] <= 1.0
        or not 0.01 <= threshold_values["minimum_image_shuffle_latent_cosine_delta"] <= 2.0
        or threshold_values["maximum_latent_rmse"] <= 0.0
        or not 0.0 <= threshold_values["maximum_ece"] <= 0.10
        or not 0.0 <= threshold_values["maximum_brier"] <= 0.25
        or not 0.75 <= threshold_values["minimum_predicted_class_occupancy_fraction"] <= 1.0
        or threshold_values["minimum_per_type_support"] < 2.0
    ):
        raise ValueError("initialization evidence thresholds violate fail-closed bounds")

    raw_seeds = plan.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) < 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_seeds
        )
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ValueError("initialization validation plan seeds are malformed")
    execution = report.get("execution")
    if (
        not isinstance(execution, Mapping)
        or set(execution) != {"device", "seeds"}
        or execution.get("device") != "cpu-float32"
        or execution.get("seeds") != raw_seeds
    ):
        raise ValueError("initialization evidence execution differs from the plan")

    replay = recompute_initialization_validation(
        checkpoint=checkpoint_payload,
        morphology=evidence_morphology,
        edge_index=evidence_edges,
        edge_weight=evidence_edge_weight,
        labels=evidence_type_labels,
        target_latent=evidence_target_latent,
        donor_ids=evidence_donor_ids.tolist(),
        seeds=raw_seeds,
    )

    def replay_number_matches(reported: float, expected: object) -> bool:
        return math.isclose(
            reported,
            _report_number(expected, "recomputed metric"),
            rel_tol=1.0e-7,
            abs_tol=1.0e-8,
        )

    metric_keys = {
        "macro_f1",
        "ece",
        "brier",
        "predicted_class_occupancy_fraction",
        "minimum_per_type_support",
        "latent_cosine",
        "latent_rmse",
    }
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != metric_keys:
        raise ValueError("initialization evidence pooled metrics are malformed")

    def metric_values(row: Mapping[str, object], label: str) -> Mapping[str, float]:
        values = {name: _report_number(row[name], "%s %s" % (label, name)) for name in metric_keys}
        if (
            not 0.0 <= values["macro_f1"] <= 1.0
            or not 0.0 <= values["ece"] <= 1.0
            or not 0.0 <= values["brier"] <= 2.0
            or not 0.0 <= values["predicted_class_occupancy_fraction"] <= 1.0
            or values["minimum_per_type_support"] < 0.0
            or not -1.0 <= values["latent_cosine"] <= 1.0
            or values["latent_rmse"] < 0.0
        ):
            raise ValueError("initialization evidence %s is outside metric bounds" % label)
        return values

    pooled_metric_values = metric_values(metrics, "pooled metrics")
    replay_pooled = replay["metrics"]
    if not isinstance(replay_pooled, Mapping) or any(
        not replay_number_matches(pooled_metric_values[name], replay_pooled[name])
        for name in metric_keys
    ):
        raise ValueError("initialization evidence pooled metrics differ from checkpoint replay")

    donor_metrics = report.get("donor_metrics")
    if not isinstance(donor_metrics, list) or len(donor_metrics) != len(expected_holdout):
        raise ValueError("initialization evidence donor metrics are incomplete")
    donor_rows = {}
    for row in donor_metrics:
        if not isinstance(row, Mapping) or set(row) != metric_keys | {"donor_id"}:
            raise ValueError("initialization evidence donor metric row is malformed")
        donor = str(row["donor_id"])
        if donor in donor_rows or donor not in expected_holdout:
            raise ValueError("initialization evidence donor metric identities are invalid")
        donor_rows[donor] = metric_values(row, "donor metrics")
    if set(donor_rows) != set(expected_holdout):
        raise ValueError("initialization evidence donor metrics are incomplete")
    replay_donor_rows = {
        str(row["donor_id"]): row
        for row in replay["donor_metrics"]
        if isinstance(row, Mapping) and "donor_id" in row
    }
    if set(replay_donor_rows) != set(expected_holdout) or any(
        not replay_number_matches(donor_rows[donor][name], replay_donor_rows[donor][name])
        for donor in expected_holdout
        for name in metric_keys
    ):
        raise ValueError("initialization evidence donor metrics differ from checkpoint replay")

    control_keys = {
        "seed",
        "donor_id",
        "permutation_sha256",
        "cross_type_mismatch_fraction",
        "image_shuffle_macro_f1",
        "real_minus_image_shuffle_macro_f1",
        "image_shuffle_latent_cosine",
        "real_minus_image_shuffle_latent_cosine",
    }
    controls = report.get("shuffle_controls")
    if not isinstance(controls, list) or len(controls) != len(expected_holdout) * len(raw_seeds):
        raise ValueError("initialization evidence shuffle controls are incomplete")
    replay_control_rows = {
        (str(row["donor_id"]), int(row["seed"])): row
        for row in replay["shuffle_controls"]
        if isinstance(row, Mapping) and "donor_id" in row and "seed" in row
    }
    control_rows = {}
    for row in controls:
        if not isinstance(row, Mapping) or set(row) != control_keys:
            raise ValueError("initialization evidence shuffle-control row is malformed")
        donor = str(row["donor_id"])
        seed = row["seed"]
        if donor not in expected_holdout or seed not in raw_seeds or isinstance(seed, bool):
            raise ValueError("initialization evidence shuffle-control identity is invalid")
        key = (donor, int(seed))
        if key in control_rows or not _is_sha256(str(row["permutation_sha256"])):
            raise ValueError("initialization evidence shuffle-control identity is invalid")
        values = {
            name: _report_number(row[name], "shuffle metric %s" % name)
            for name in control_keys - {"seed", "donor_id", "permutation_sha256"}
        }
        if not 0.5 <= values["cross_type_mismatch_fraction"] <= 1.0:
            raise ValueError("initialization evidence shuffle null is not cross-type")
        if (
            not 0.0 <= values["image_shuffle_macro_f1"] <= 1.0
            or not -1.0 <= values["real_minus_image_shuffle_macro_f1"] <= 1.0
            or not -1.0 <= values["image_shuffle_latent_cosine"] <= 1.0
            or not -2.0 <= values["real_minus_image_shuffle_latent_cosine"] <= 2.0
        ):
            raise ValueError("initialization evidence shuffle metric is outside bounds")
        replay_row = replay_control_rows.get(key)
        if replay_row is None or str(row["permutation_sha256"]) != str(
            replay_row["permutation_sha256"]
        ):
            raise ValueError("initialization evidence shuffle permutation is not reproducible")
        if any(
            not replay_number_matches(values[name], replay_row[name])
            for name in control_keys - {"seed", "donor_id", "permutation_sha256"}
        ):
            raise ValueError(
                "initialization evidence shuffle metrics differ from checkpoint replay"
            )
        control_rows[key] = values
    expected_control_keys = {(donor, seed) for donor in expected_holdout for seed in raw_seeds}
    if set(control_rows) != expected_control_keys:
        raise ValueError("initialization evidence shuffle controls are incomplete")

    expected_checks = {
        "macro_f1": min(row["macro_f1"] for row in replay_donor_rows.values())
        >= threshold_values["minimum_macro_f1"],
        "image_shuffle_macro_f1_delta": min(
            row["real_minus_image_shuffle_macro_f1"] for row in replay_control_rows.values()
        )
        >= threshold_values["minimum_image_shuffle_macro_f1_delta"],
        "latent_cosine": min(row["latent_cosine"] for row in replay_donor_rows.values())
        >= threshold_values["minimum_latent_cosine"],
        "image_shuffle_latent_cosine_delta": min(
            row["real_minus_image_shuffle_latent_cosine"] for row in replay_control_rows.values()
        )
        >= threshold_values["minimum_image_shuffle_latent_cosine_delta"],
        "latent_rmse": max(row["latent_rmse"] for row in replay_donor_rows.values())
        <= threshold_values["maximum_latent_rmse"],
        "ece": max(row["ece"] for row in replay_donor_rows.values())
        <= threshold_values["maximum_ece"],
        "brier": max(row["brier"] for row in replay_donor_rows.values())
        <= threshold_values["maximum_brier"],
        "predicted_class_occupancy": min(
            row["predicted_class_occupancy_fraction"] for row in replay_donor_rows.values()
        )
        >= threshold_values["minimum_predicted_class_occupancy_fraction"],
        "per_type_support": min(
            row["minimum_per_type_support"] for row in replay_donor_rows.values()
        )
        >= threshold_values["minimum_per_type_support"],
    }
    checks = report.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _INITIALIZATION_CHECK_KEYS
        or any(not isinstance(value, bool) for value in checks.values())
        or dict(checks) != expected_checks
    ):
        raise ValueError("initialization evidence checks contradict reported metrics")
    expected_pass = bool(all(expected_checks.values()))
    if report.get("pass") is not expected_pass or not expected_pass:
        raise ValueError("initialization evidence report is not a completed passing result")
    return report


@dataclass(frozen=True)
class MolecularEStepArtifact:
    """Immutable output of an independently frozen morphology/RNA E-step."""

    transport_plan: np.ndarray
    raw_transport_plan: np.ndarray
    transport_cost: np.ndarray
    source_mass: np.ndarray
    nucleus_ids: Tuple[str, ...]
    prototype_ids: Tuple[str, ...]
    source_artifacts: Tuple[str, ...]
    source_sha256: Tuple[str, ...]
    source_roles: Tuple[str, ...]
    teacher_checkpoint: str
    teacher_checkpoint_sha256: str
    initialization_receipt: str
    initialization_receipt_sha256: str
    teacher_role: str
    teacher_training_donors: Tuple[str, ...]
    target_donor: str
    feature_space_id: str
    latent_space_id: str
    type_ontology_sha256: str
    morphology_sha256: str
    prototype_means_sha256: str
    prototype_variances_sha256: str
    prototype_types_sha256: str
    prototype_weights_sha256: str
    image_latent_sha256: str
    type_probabilities_sha256: str
    transport_cost_sha256: str
    source_mass_sha256: str
    artifact_threshold: float
    type_cost_weight: float
    unknown_cost: float
    fixed_unknown_mass: float
    uot_epsilon: float
    uot_marginal_relaxation: float
    uot_iterations: int
    uot_iterations_run: int
    uot_convergence_tolerance: float
    uot_maximum_marginal_residual: float
    converged: bool
    source_marginal_residual: float
    target_marginal_residual: float
    solver_source_marginal_error: float
    solver_target_marginal_error: float
    source_dual_residual: float
    target_dual_residual: float
    transport_objective: float
    e_step_round: int = 0

    CONTRACT = "heir.molecular_e_step"
    CONTRACT_VERSION = 3
    TRUSTED_TEACHER_ROLES = frozenset(
        {
            "generic_crossmodal_pretraining",
            "independent_crossmodal_bridge",
            "registered_spatial_teacher",
        }
    )
    REQUIRED_SOURCE_ROLES = frozenset({"histology", "prototype_bank", "rna_reference"})

    def validate(self) -> None:
        cells = len(self.nucleus_ids)
        prototypes = len(self.prototype_ids)
        plan = np.asarray(self.transport_plan)
        raw_plan = np.asarray(self.raw_transport_plan)
        transport_cost = np.asarray(self.transport_cost)
        source_mass = np.asarray(self.source_mass)
        if plan.shape != (cells, prototypes + 1):
            raise ValueError("transport_plan must have cells-by-(prototypes plus dustbin) shape")
        if raw_plan.shape != plan.shape or transport_cost.shape != plan.shape:
            raise ValueError("raw transport plan and cost must align to transport_plan")
        if source_mass.shape != (cells,):
            raise ValueError("source_mass must contain one value per nucleus")
        for name, value in (
            ("transport_plan", plan),
            ("raw_transport_plan", raw_plan),
            ("transport_cost", transport_cost),
            ("source_mass", source_mass),
        ):
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError("%s must be floating point" % name)
            if not np.isfinite(value).all():
                raise ValueError("%s must be finite" % name)
        if np.any(plan < 0) or np.any(raw_plan < 0) or np.any(source_mass < 0):
            raise ValueError("transport plans and source mass must be non-negative")
        if not np.any(source_mass > 0):
            raise ValueError("source_mass must contain positive M-step mass")
        if np.any(plan.sum(axis=1) <= 0):
            raise ValueError("transport_plan must give every nucleus positive transported mass")
        raw_row_mass = raw_plan.sum(axis=1)
        positive = source_mass > 0
        if np.any(raw_row_mass[positive] <= 0) or np.any(raw_row_mass[~positive] > 1.0e-12):
            raise ValueError("raw transport rows do not agree with source_mass support")
        expected_plan = np.zeros_like(plan, dtype=np.float64)
        expected_plan[positive] = raw_plan[positive] / raw_row_mass[positive, None]
        expected_plan[~positive, -1] = 1.0
        if not np.allclose(plan, expected_plan, atol=1.0e-6, rtol=0.0):
            raise ValueError("transport_plan is not the declared row-conditional raw plan")
        for name, values in (
            ("nucleus_ids", self.nucleus_ids),
            ("prototype_ids", self.prototype_ids),
            ("source_artifacts", self.source_artifacts),
            ("source_roles", self.source_roles),
            ("teacher_training_donors", self.teacher_training_donors),
        ):
            if any(not value.strip() for value in values) or len(set(values)) != len(values):
                raise ValueError("%s must contain unique non-empty strings" % name)
        if not (len(self.source_artifacts) == len(self.source_sha256) == len(self.source_roles)):
            raise ValueError("E-step sources, hashes, and roles must align")
        if any(not _is_sha256(value) for value in self.source_sha256):
            raise ValueError("E-step source_sha256 must contain lowercase SHA-256 digests")
        if set(self.source_roles) != self.REQUIRED_SOURCE_ROLES:
            raise ValueError(
                "E-step source_roles must contain exactly histology, prototype_bank, and "
                "rna_reference"
            )
        if self.teacher_role not in self.TRUSTED_TEACHER_ROLES:
            raise ValueError("molecular E-step teacher role is not independently grounded")
        if not self.teacher_training_donors:
            raise ValueError("molecular E-step teacher training donors cannot be empty")
        if not self.teacher_checkpoint.strip() or not self.initialization_receipt.strip():
            raise ValueError("molecular E-step teacher/receipt paths cannot be empty")
        if not _is_sha256(self.teacher_checkpoint_sha256):
            raise ValueError("teacher_checkpoint_sha256 must be a lowercase SHA-256 digest")
        if not _is_sha256(self.initialization_receipt_sha256):
            raise ValueError("initialization_receipt_sha256 must be a lowercase SHA-256 digest")
        for name, value in (
            ("type_ontology_sha256", self.type_ontology_sha256),
            ("morphology_sha256", self.morphology_sha256),
            ("prototype_means_sha256", self.prototype_means_sha256),
            ("prototype_variances_sha256", self.prototype_variances_sha256),
            ("prototype_types_sha256", self.prototype_types_sha256),
            ("prototype_weights_sha256", self.prototype_weights_sha256),
            ("image_latent_sha256", self.image_latent_sha256),
            ("type_probabilities_sha256", self.type_probabilities_sha256),
            ("transport_cost_sha256", self.transport_cost_sha256),
            ("source_mass_sha256", self.source_mass_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError("%s must be a lowercase SHA-256 digest" % name)
        if not self.target_donor.strip():
            raise ValueError("target_donor cannot be empty")
        if self.target_donor in set(self.teacher_training_donors):
            raise ValueError("molecular E-step teacher was trained on the target donor")
        if not self.feature_space_id.strip() or not self.latent_space_id.strip():
            raise ValueError("molecular E-step feature and latent spaces cannot be empty")
        if not 0.0 <= self.fixed_unknown_mass < 1.0:
            raise ValueError("fixed_unknown_mass must lie in [0, 1)")
        if not 0.0 <= self.artifact_threshold <= 1.0:
            raise ValueError("artifact_threshold must lie in [0, 1]")
        for name, value in (
            ("type_cost_weight", self.type_cost_weight),
            ("unknown_cost", self.unknown_cost),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
        if array_content_sha256(transport_cost) != self.transport_cost_sha256:
            raise ValueError("transport_cost_sha256 differs from the stored cost")
        if array_content_sha256(source_mass) != self.source_mass_sha256:
            raise ValueError("source_mass_sha256 differs from the stored M-step weights")
        for name, value in (
            ("uot_epsilon", self.uot_epsilon),
            ("uot_marginal_relaxation", self.uot_marginal_relaxation),
            ("uot_convergence_tolerance", self.uot_convergence_tolerance),
            ("uot_maximum_marginal_residual", self.uot_maximum_marginal_residual),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if self.uot_maximum_marginal_residual > 2.0:
            raise ValueError("uot_maximum_marginal_residual cannot exceed 2")
        if (
            self.uot_iterations <= 0
            or self.uot_iterations_run <= 0
            or self.uot_iterations_run > self.uot_iterations
            or self.e_step_round < 0
        ):
            raise ValueError("UOT iterations must be positive and E-step round non-negative")
        if not self.converged:
            raise ValueError("molecular E-step artifact records nonconverged transport")
        for name, value in (
            ("source_marginal_residual", self.source_marginal_residual),
            ("target_marginal_residual", self.target_marginal_residual),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
            if value > self.uot_maximum_marginal_residual:
                raise ValueError("%s exceeds the prespecified marginal tolerance" % name)
        for name, value in (
            ("solver_source_marginal_error", self.solver_source_marginal_error),
            ("solver_target_marginal_error", self.solver_target_marginal_error),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
        for name, value in (
            ("source_dual_residual", self.source_dual_residual),
            ("target_dual_residual", self.target_dual_residual),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
            if value > self.uot_convergence_tolerance:
                raise ValueError("%s exceeds the declared convergence tolerance" % name)
        if not math.isfinite(self.transport_objective):
            raise ValueError("transport_objective must be finite")
        row_sums = plan.sum(axis=1, dtype=np.float64)
        realized_source_residual = float(np.max(np.abs(row_sums - 1.0)))
        if not math.isclose(
            realized_source_residual,
            self.source_marginal_residual,
            rel_tol=0.0,
            abs_tol=max(1.0e-6, self.uot_convergence_tolerance),
        ):
            raise ValueError("transport_plan source marginal contradicts its residual")
        normalized_source_mass = source_mass.astype(np.float64) / source_mass.sum(dtype=np.float64)
        realized_unknown_mass = float((plan[:, -1] * normalized_source_mass).sum())
        if abs(realized_unknown_mass - self.fixed_unknown_mass) > (
            self.target_marginal_residual + self.uot_convergence_tolerance
        ):
            raise ValueError("transport_plan dustbin marginal differs from fixed_unknown_mass")

    @property
    def responsibilities(self) -> np.ndarray:
        """Known-state subprobabilities normalized by complete row mass."""

        plan = np.asarray(self.transport_plan, dtype=np.float64)
        responsibilities = plan[:, :-1] / plan.sum(axis=1, keepdims=True)
        return np.asarray(responsibilities, dtype=np.float32)

    def validate_binding(
        self,
        *,
        nucleus_ids: Sequence[object],
        prototype_ids: Sequence[object],
        source_sha256_by_role: Mapping[str, str],
        target_donor: str,
        feature_space_id: str,
        latent_space_id: str,
        type_names: Sequence[object],
        morphology: object,
        edge_index: object,
        edge_weight: object,
        prototype_means: object,
        prototype_variances: object,
        prototype_types: object,
        prototype_weights: object,
        cell_weights: object,
        artifact_threshold: float,
    ) -> None:
        """Fail on reordered identities, stale inputs, or an unreplayable E-step.

        The serialized cost and couplings are not trusted merely because their
        internal hashes and telemetry agree.  The independently bound teacher is
        replayed in evaluation-mode float32 on the exact morphology graph, the
        Gaussian/type cost is reconstructed, and the declared realized number of
        Sinkhorn updates is rerun.  Small tolerances account for CUDA-versus-CPU
        float32 kernels used by the producer; they are deliberately far below a
        biologically meaningful change in a cost or responsibility.
        """

        self.validate()
        for label, raw_path, expected_sha256 in (
            (
                "teacher checkpoint",
                self.teacher_checkpoint,
                self.teacher_checkpoint_sha256,
            ),
            (
                "initialization receipt",
                self.initialization_receipt,
                self.initialization_receipt_sha256,
            ),
        ):
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise ValueError("molecular E-step %s is unavailable" % label)
            if _sha256_path(path) != expected_sha256:
                raise ValueError("molecular E-step %s hash differs" % label)
        if tuple(str(value) for value in nucleus_ids) != self.nucleus_ids:
            raise ValueError("molecular E-step nucleus order differs from the HistologyBag")
        if tuple(str(value) for value in prototype_ids) != self.prototype_ids:
            raise ValueError("molecular E-step prototype order differs from the PrototypeSet")
        recorded = dict(zip(self.source_roles, self.source_sha256))
        if recorded != dict(source_sha256_by_role):
            raise ValueError("molecular E-step source hashes differ from its bound inputs")
        if self.target_donor != target_donor:
            raise ValueError("molecular E-step target donor differs from the training batch")
        if self.feature_space_id != feature_space_id:
            raise ValueError("molecular E-step pathology feature space differs")
        if self.latent_space_id != latent_space_id:
            raise ValueError("molecular E-step latent space differs")
        if self.type_ontology_sha256 != ordered_identity_sha256(type_names):
            raise ValueError("molecular E-step cell-type ontology differs")
        receipt = ValidatedInitializationReceipt.load_json(self.initialization_receipt)
        receipt.validate_binding(
            checkpoint_sha256=self.teacher_checkpoint_sha256,
            feature_space_id=self.feature_space_id,
            latent_space_id=self.latent_space_id,
            type_names=type_names,
            target_donors=(self.target_donor,),
            receipt_path=self.initialization_receipt,
        )
        if set(receipt.training_donors) != set(self.teacher_training_donors):
            raise ValueError(
                "molecular E-step receipt training donors differ from teacher provenance"
            )
        for label, observed, expected in (
            ("morphology", array_content_sha256(morphology), self.morphology_sha256),
            (
                "prototype means",
                array_content_sha256(prototype_means),
                self.prototype_means_sha256,
            ),
            (
                "prototype variances",
                array_content_sha256(prototype_variances),
                self.prototype_variances_sha256,
            ),
            (
                "prototype types",
                array_content_sha256(prototype_types),
                self.prototype_types_sha256,
            ),
            (
                "prototype weights",
                array_content_sha256(prototype_weights),
                self.prototype_weights_sha256,
            ),
            ("M-step cell weights", array_content_sha256(cell_weights), self.source_mass_sha256),
        ):
            if observed != expected:
                raise ValueError("molecular E-step %s tensor content differs" % label)
        morphology_array = np.array(morphology, dtype=np.float32, order="C", copy=True)
        edge_array = np.array(edge_index, dtype=np.int64, order="C", copy=True)
        edge_weight_array = (
            None
            if edge_weight is None
            else np.array(edge_weight, dtype=np.float32, order="C", copy=True)
        )
        means_array = np.array(prototype_means, dtype=np.float32, order="C", copy=True)
        variances_array = np.array(prototype_variances, dtype=np.float32, order="C", copy=True)
        types_array = np.array(prototype_types, dtype=np.int64, order="C", copy=True)
        if (
            morphology_array.ndim != 2
            or morphology_array.shape[0] != len(self.nucleus_ids)
            or edge_array.ndim != 2
            or edge_array.shape[0] != 2
            or (edge_weight_array is not None and edge_weight_array.shape != (edge_array.shape[1],))
            or means_array.ndim != 2
            or means_array.shape[0] != len(self.prototype_ids)
            or variances_array.shape != means_array.shape
            or types_array.shape != (len(self.prototype_ids),)
            or not np.isfinite(morphology_array).all()
            or not np.isfinite(means_array).all()
            or not np.isfinite(variances_array).all()
            or (edge_weight_array is not None and not np.isfinite(edge_weight_array).all())
            or np.any(variances_array <= 0)
            or np.any(types_array < 0)
            or np.any(types_array >= len(tuple(type_names)))
        ):
            raise ValueError("molecular E-step replay tensors are malformed")
        if edge_array.size and (
            int(edge_array.min()) < 0 or int(edge_array.max()) >= len(self.nucleus_ids)
        ):
            raise ValueError("molecular E-step replay graph is out of range")
        weights = np.asarray(prototype_weights, dtype=np.float64)
        if weights.shape != (len(self.prototype_ids),) or np.any(weights < 0):
            raise ValueError("molecular E-step prototype weights are malformed")
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            raise ValueError("molecular E-step prototype weights need positive finite mass")
        if not math.isclose(
            float(artifact_threshold),
            self.artifact_threshold,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("molecular E-step artifact threshold differs from assembly")
        m_step_mass = np.asarray(cell_weights, dtype=np.float64)
        if m_step_mass.shape != (len(self.nucleus_ids),) or np.any(m_step_mass < 0):
            raise ValueError("molecular E-step M-step cell weights are malformed")
        if not np.isfinite(m_step_mass).all() or float(m_step_mass.sum()) <= 0:
            raise ValueError("molecular E-step M-step cell weights need positive finite mass")

        try:
            import torch

            from heir.losses import unbalanced_sinkhorn
            from heir.models import HEIRModel
        except ImportError as error:  # pragma: no cover - broken package installation
            raise ValueError("molecular E-step replay dependencies are unavailable") from error
        try:
            try:
                checkpoint = torch.load(
                    Path(self.teacher_checkpoint).expanduser().resolve(),
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:  # pragma: no cover - older supported torch
                checkpoint = torch.load(
                    Path(self.teacher_checkpoint).expanduser().resolve(), map_location="cpu"
                )
            if not isinstance(checkpoint, Mapping):
                raise ValueError("teacher checkpoint must contain a mapping")
            teacher = (
                HEIRModel.from_checkpoint(checkpoint).to(device="cpu", dtype=torch.float32).eval()
            )
        except (
            EOFError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            raise ValueError("molecular E-step teacher checkpoint cannot be replayed") from error
        if (
            teacher.config.morphology_dim != morphology_array.shape[1]
            or teacher.config.latent_dim != means_array.shape[1]
            or teacher.config.num_cell_types != len(tuple(type_names))
        ):
            raise ValueError("molecular E-step teacher dimensions differ from bound inputs")

        morphology_tensor = torch.from_numpy(morphology_array)
        edge_tensor = torch.from_numpy(edge_array)
        edge_weight_tensor = (
            None if edge_weight_array is None else torch.from_numpy(edge_weight_array)
        )
        means_tensor = torch.from_numpy(means_array)
        variances_tensor = torch.from_numpy(variances_array)
        types_tensor = torch.from_numpy(types_array)
        source_tensor = torch.from_numpy(
            np.array(m_step_mass, dtype=np.float32, order="C", copy=True)
        )
        target_tensor = torch.from_numpy(
            np.array(prototype_weights, dtype=np.float32, order="C", copy=True)
        )
        with torch.inference_mode():
            _, replay_type_probabilities, replay_image_latent = teacher.encode_frozen_morphology(
                morphology_tensor,
                edge_tensor,
                edge_weight_tensor,
            )
            variance_tensor = variances_tensor.clamp_min(teacher.config.prototype_variance_floor)
            gaussian_cost = 0.5 * (
                (replay_image_latent.unsqueeze(1) - means_tensor.unsqueeze(0)).square()
                / variance_tensor.unsqueeze(0)
                + variance_tensor.unsqueeze(0).log()
            ).mean(dim=2)
            type_cost = (
                -replay_type_probabilities.index_select(1, types_tensor).clamp_min(1.0e-8).log()
            )
            replay_known_cost = gaussian_cost + self.type_cost_weight * type_cost
            replay_full_cost = torch.cat(
                (
                    replay_known_cost,
                    replay_known_cost.new_full((len(replay_known_cost), 1), self.unknown_cost),
                ),
                dim=1,
            )

            # Fixed realized iterations avoid a CPU/CUDA threshold-boundary
            # disagreement while still binding the exact update count reported
            # by the producer.  The artifact separately requires the final dual
            # residual to meet its declared convergence tolerance.
            replay_transport = unbalanced_sinkhorn(
                replay_known_cost,
                source_mass=source_tensor,
                target_mass=target_tensor,
                epsilon=self.uot_epsilon,
                marginal_relaxation=self.uot_marginal_relaxation,
                iterations=self.uot_iterations_run,
                convergence_tolerance=None,
                unknown_mass=self.fixed_unknown_mass,
                unknown_cost=self.unknown_cost,
                add_unknown=True,
            )
        replay_raw_plan = replay_transport.plan.to(dtype=torch.float32).cpu().numpy()
        replay_row_mass = replay_raw_plan.sum(axis=1, keepdims=True)
        replay_conditional = np.zeros_like(replay_raw_plan, dtype=np.float32)
        positive_source = np.asarray(m_step_mass > 0)
        if np.any(replay_row_mass[positive_source] <= 0):
            raise ValueError("replayed molecular E-step has an empty positive-mass row")
        replay_conditional[positive_source] = (
            replay_raw_plan[positive_source] / replay_row_mass[positive_source]
        )
        replay_conditional[~positive_source, -1] = 1.0

        replay_rtol = 5.0e-4
        replay_atol = 2.0e-5
        if not np.allclose(
            np.asarray(self.transport_cost, dtype=np.float32),
            replay_full_cost.to(dtype=torch.float32).cpu().numpy(),
            rtol=replay_rtol,
            atol=replay_atol,
        ):
            raise ValueError("molecular E-step transport cost differs from teacher replay")
        if not np.allclose(
            np.asarray(self.raw_transport_plan, dtype=np.float32),
            replay_raw_plan,
            rtol=replay_rtol,
            atol=replay_atol,
        ):
            raise ValueError("molecular E-step raw transport plan differs from Sinkhorn replay")
        if not np.allclose(
            np.asarray(self.transport_plan, dtype=np.float32),
            replay_conditional,
            rtol=replay_rtol,
            atol=replay_atol,
        ):
            raise ValueError(
                "molecular E-step conditional transport plan differs from Sinkhorn replay"
            )
        replay_source_dual = float(replay_transport.source_dual_residual.item())
        replay_target_dual = float(replay_transport.target_dual_residual.item())
        if max(replay_source_dual, replay_target_dual) > (
            self.uot_convergence_tolerance + replay_atol
        ):
            raise ValueError("molecular E-step declared iteration did not replay as converged")
        desired_target = np.concatenate(
            (
                weights / weights.sum() * (1.0 - self.fixed_unknown_mass),
                np.asarray([self.fixed_unknown_mass], dtype=np.float64),
            )
        )
        plan = np.asarray(self.transport_plan, dtype=np.float64)
        raw_plan = np.asarray(self.raw_transport_plan, dtype=np.float64)
        realized_target = (
            plan * (m_step_mass / m_step_mass.sum(dtype=np.float64))[:, np.newaxis]
        ).sum(axis=0)
        realized_residual = float(np.abs(realized_target - desired_target).sum())
        if not math.isclose(
            realized_residual,
            self.target_marginal_residual,
            rel_tol=0.0,
            abs_tol=max(1.0e-6, self.uot_convergence_tolerance),
        ):
            raise ValueError("molecular E-step target marginal residual is inconsistent")
        telemetry = frozen_transport_telemetry(
            raw_transport_plan=raw_plan,
            transport_cost=self.transport_cost,
            source_mass=m_step_mass,
            target_weights=weights,
            fixed_unknown_mass=self.fixed_unknown_mass,
            epsilon=self.uot_epsilon,
            marginal_relaxation=self.uot_marginal_relaxation,
        )
        if not math.isclose(
            telemetry["solver_source_marginal_error"],
            self.solver_source_marginal_error,
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise ValueError("raw transport source marginal differs from solver telemetry")
        if not math.isclose(
            telemetry["solver_target_marginal_error"],
            self.solver_target_marginal_error,
            rel_tol=1.0e-5,
            abs_tol=1.0e-6,
        ):
            raise ValueError("raw transport target marginal differs from solver telemetry")
        if not math.isclose(
            telemetry["transport_objective"],
            self.transport_objective,
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            raise ValueError("raw transport objective differs from solver telemetry")

    def save_npz(self, path: PathLike) -> None:
        self.validate()
        payload = {
            "__contract__": np.asarray(self.CONTRACT, dtype=np.dtype("U")),
            "__version__": np.asarray(self.CONTRACT_VERSION, dtype=np.int64),
            "transport_plan": np.asarray(self.transport_plan, dtype=np.float32),
            "raw_transport_plan": np.asarray(self.raw_transport_plan, dtype=np.float32),
            "transport_cost": np.asarray(self.transport_cost, dtype=np.float32),
            "source_mass": np.asarray(self.source_mass, dtype=np.float32),
            "nucleus_ids": np.asarray(self.nucleus_ids, dtype=np.dtype("U")),
            "prototype_ids": np.asarray(self.prototype_ids, dtype=np.dtype("U")),
            "source_artifacts": np.asarray(self.source_artifacts, dtype=np.dtype("U")),
            "source_sha256": np.asarray(self.source_sha256, dtype=np.dtype("U")),
            "source_roles": np.asarray(self.source_roles, dtype=np.dtype("U")),
            "teacher_checkpoint": np.asarray(self.teacher_checkpoint),
            "teacher_checkpoint_sha256": np.asarray(self.teacher_checkpoint_sha256),
            "initialization_receipt": np.asarray(self.initialization_receipt),
            "initialization_receipt_sha256": np.asarray(self.initialization_receipt_sha256),
            "teacher_role": np.asarray(self.teacher_role),
            "teacher_training_donors": np.asarray(
                self.teacher_training_donors, dtype=np.dtype("U")
            ),
            "target_donor": np.asarray(self.target_donor),
            "feature_space_id": np.asarray(self.feature_space_id),
            "latent_space_id": np.asarray(self.latent_space_id),
            "type_ontology_sha256": np.asarray(self.type_ontology_sha256),
            "morphology_sha256": np.asarray(self.morphology_sha256),
            "prototype_means_sha256": np.asarray(self.prototype_means_sha256),
            "prototype_variances_sha256": np.asarray(self.prototype_variances_sha256),
            "prototype_types_sha256": np.asarray(self.prototype_types_sha256),
            "prototype_weights_sha256": np.asarray(self.prototype_weights_sha256),
            "image_latent_sha256": np.asarray(self.image_latent_sha256),
            "type_probabilities_sha256": np.asarray(self.type_probabilities_sha256),
            "transport_cost_sha256": np.asarray(self.transport_cost_sha256),
            "source_mass_sha256": np.asarray(self.source_mass_sha256),
            "artifact_threshold": np.asarray(self.artifact_threshold, dtype=np.float64),
            "type_cost_weight": np.asarray(self.type_cost_weight, dtype=np.float64),
            "unknown_cost": np.asarray(self.unknown_cost, dtype=np.float64),
            "fixed_unknown_mass": np.asarray(self.fixed_unknown_mass, dtype=np.float64),
            "uot_epsilon": np.asarray(self.uot_epsilon, dtype=np.float64),
            "uot_marginal_relaxation": np.asarray(self.uot_marginal_relaxation, dtype=np.float64),
            "uot_iterations": np.asarray(self.uot_iterations, dtype=np.int64),
            "uot_iterations_run": np.asarray(self.uot_iterations_run, dtype=np.int64),
            "uot_convergence_tolerance": np.asarray(
                self.uot_convergence_tolerance, dtype=np.float64
            ),
            "uot_maximum_marginal_residual": np.asarray(
                self.uot_maximum_marginal_residual, dtype=np.float64
            ),
            "converged": np.asarray(self.converged, dtype=np.bool_),
            "source_marginal_residual": np.asarray(self.source_marginal_residual, dtype=np.float64),
            "target_marginal_residual": np.asarray(self.target_marginal_residual, dtype=np.float64),
            "solver_source_marginal_error": np.asarray(
                self.solver_source_marginal_error, dtype=np.float64
            ),
            "solver_target_marginal_error": np.asarray(
                self.solver_target_marginal_error, dtype=np.float64
            ),
            "source_dual_residual": np.asarray(self.source_dual_residual, dtype=np.float64),
            "target_dual_residual": np.asarray(self.target_dual_residual, dtype=np.float64),
            "transport_objective": np.asarray(self.transport_objective, dtype=np.float64),
            "e_step_round": np.asarray(self.e_step_round, dtype=np.int64),
        }
        _write_deterministic_npz(path, payload)

    @classmethod
    def load_npz(cls, path: PathLike) -> "MolecularEStepArtifact":
        with np.load(path, allow_pickle=False) as archive:
            contract = str(_scalar(archive, "__contract__"))
            version = int(_scalar(archive, "__version__"))
            if contract != cls.CONTRACT or version != cls.CONTRACT_VERSION:
                raise ValueError("unsupported molecular E-step artifact contract")
            required_vectors = (
                "nucleus_ids",
                "prototype_ids",
                "source_artifacts",
                "source_sha256",
                "source_roles",
                "teacher_training_donors",
            )
            for name in required_vectors:
                if name not in archive:
                    raise ValueError("molecular E-step artifact is missing %s" % name)
            for name in (
                "transport_plan",
                "raw_transport_plan",
                "transport_cost",
                "source_mass",
            ):
                if name not in archive:
                    raise ValueError("molecular E-step artifact is missing %s" % name)
            artifact = cls(
                transport_plan=np.array(archive["transport_plan"], dtype=np.float32, copy=True),
                raw_transport_plan=np.array(
                    archive["raw_transport_plan"], dtype=np.float32, copy=True
                ),
                transport_cost=np.array(archive["transport_cost"], dtype=np.float32, copy=True),
                source_mass=np.array(archive["source_mass"], dtype=np.float32, copy=True),
                nucleus_ids=_as_string_vector(archive["nucleus_ids"], "nucleus_ids"),
                prototype_ids=_as_string_vector(archive["prototype_ids"], "prototype_ids"),
                source_artifacts=_as_string_vector(archive["source_artifacts"], "source_artifacts"),
                source_sha256=_as_string_vector(archive["source_sha256"], "source_sha256"),
                source_roles=_as_string_vector(archive["source_roles"], "source_roles"),
                teacher_checkpoint=str(_scalar(archive, "teacher_checkpoint")),
                teacher_checkpoint_sha256=str(_scalar(archive, "teacher_checkpoint_sha256")),
                initialization_receipt=str(_scalar(archive, "initialization_receipt")),
                initialization_receipt_sha256=str(
                    _scalar(archive, "initialization_receipt_sha256")
                ),
                teacher_role=str(_scalar(archive, "teacher_role")),
                teacher_training_donors=_as_string_vector(
                    archive["teacher_training_donors"], "teacher_training_donors"
                ),
                target_donor=str(_scalar(archive, "target_donor")),
                feature_space_id=str(_scalar(archive, "feature_space_id")),
                latent_space_id=str(_scalar(archive, "latent_space_id")),
                type_ontology_sha256=str(_scalar(archive, "type_ontology_sha256")),
                morphology_sha256=str(_scalar(archive, "morphology_sha256")),
                prototype_means_sha256=str(_scalar(archive, "prototype_means_sha256")),
                prototype_variances_sha256=str(_scalar(archive, "prototype_variances_sha256")),
                prototype_types_sha256=str(_scalar(archive, "prototype_types_sha256")),
                prototype_weights_sha256=str(_scalar(archive, "prototype_weights_sha256")),
                image_latent_sha256=str(_scalar(archive, "image_latent_sha256")),
                type_probabilities_sha256=str(_scalar(archive, "type_probabilities_sha256")),
                transport_cost_sha256=str(_scalar(archive, "transport_cost_sha256")),
                source_mass_sha256=str(_scalar(archive, "source_mass_sha256")),
                artifact_threshold=float(_scalar(archive, "artifact_threshold")),
                type_cost_weight=float(_scalar(archive, "type_cost_weight")),
                unknown_cost=float(_scalar(archive, "unknown_cost")),
                fixed_unknown_mass=float(_scalar(archive, "fixed_unknown_mass")),
                uot_epsilon=float(_scalar(archive, "uot_epsilon")),
                uot_marginal_relaxation=float(_scalar(archive, "uot_marginal_relaxation")),
                uot_iterations=int(_scalar(archive, "uot_iterations")),
                uot_iterations_run=int(_scalar(archive, "uot_iterations_run")),
                uot_convergence_tolerance=float(_scalar(archive, "uot_convergence_tolerance")),
                uot_maximum_marginal_residual=float(
                    _scalar(archive, "uot_maximum_marginal_residual")
                ),
                converged=bool(_scalar(archive, "converged")),
                source_marginal_residual=float(_scalar(archive, "source_marginal_residual")),
                target_marginal_residual=float(_scalar(archive, "target_marginal_residual")),
                solver_source_marginal_error=float(
                    _scalar(archive, "solver_source_marginal_error")
                ),
                solver_target_marginal_error=float(
                    _scalar(archive, "solver_target_marginal_error")
                ),
                source_dual_residual=float(_scalar(archive, "source_dual_residual")),
                target_dual_residual=float(_scalar(archive, "target_dual_residual")),
                transport_objective=float(_scalar(archive, "transport_objective")),
                e_step_round=int(_scalar(archive, "e_step_round")),
            )
        artifact.validate()
        return artifact


@dataclass(frozen=True)
class ValidatedInitializationReceipt:
    """Evidence binding an independently validated morphology checkpoint."""

    checkpoint_sha256: str
    feature_space_id: str
    latent_space_id: str
    type_ontology_sha256: str
    training_donors: Tuple[str, ...]
    held_out_donors: Tuple[str, ...]
    capabilities: Tuple[str, ...]
    evidence_report: str
    evidence_report_sha256: str

    SCHEMA = "heir.validated_initialization.v1"
    EVIDENCE_SCHEMA = "heir.initialization_validation_evidence.v1"
    REQUIRED_CAPABILITIES = frozenset({"broad_type", "image_to_latent"})

    @classmethod
    def load_json(cls, path: PathLike) -> "ValidatedInitializationReceipt":
        source = Path(path).expanduser().resolve()
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("initialization receipt is not valid JSON") from error
        required = {
            "schema",
            "status",
            "pass",
            "checkpoint_sha256",
            "feature_space_id",
            "latent_space_id",
            "type_ontology_sha256",
            "training_donors",
            "held_out_donors",
            "capabilities",
            "evidence_report",
            "evidence_report_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("initialization receipt must contain a JSON object")
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError("initialization receipt has an unsupported schema")
        if payload.get("status") != "complete" or payload.get("pass") is not True:
            raise ValueError("initialization receipt is not a completed passing result")
        capabilities = payload.get("capabilities")
        if capabilities != {"broad_type": True, "image_to_latent": True}:
            raise ValueError("initialization receipt capabilities are malformed")
        capability_names = tuple(sorted(capabilities))
        training_donors = payload.get("training_donors")
        held_out_donors = payload.get("held_out_donors")
        if not isinstance(training_donors, list) or not isinstance(held_out_donors, list):
            raise ValueError("initialization receipt donor scopes must be JSON arrays")
        receipt = cls(
            checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
            feature_space_id=str(payload.get("feature_space_id", "")),
            latent_space_id=str(payload.get("latent_space_id", "")),
            type_ontology_sha256=str(payload.get("type_ontology_sha256", "")),
            training_donors=tuple(str(value) for value in training_donors),
            held_out_donors=tuple(str(value) for value in held_out_donors),
            capabilities=capability_names,
            evidence_report=str(payload.get("evidence_report", "")),
            evidence_report_sha256=str(payload.get("evidence_report_sha256", "")),
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        for name, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("type_ontology_sha256", self.type_ontology_sha256),
            ("evidence_report_sha256", self.evidence_report_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError("initialization receipt %s is not a SHA-256 digest" % name)
        if not self.feature_space_id.strip() or not self.latent_space_id.strip():
            raise ValueError("initialization receipt feature/latent spaces cannot be empty")
        for name, values in (
            ("training_donors", self.training_donors),
            ("held_out_donors", self.held_out_donors),
            ("capabilities", self.capabilities),
        ):
            if any(not value.strip() for value in values) or len(set(values)) != len(values):
                raise ValueError("initialization receipt %s must be unique and non-empty" % name)
        if not self.training_donors or not self.held_out_donors:
            raise ValueError(
                "initialization receipt requires non-empty training and held-out donors"
            )
        if set(self.training_donors) & set(self.held_out_donors):
            raise ValueError("initialization receipt training and held-out donors overlap")
        missing = sorted(self.REQUIRED_CAPABILITIES - set(self.capabilities))
        if missing:
            raise ValueError(
                "initialization receipt lacks required capabilities: %s" % ", ".join(missing)
            )
        if not self.evidence_report.strip():
            raise ValueError("initialization receipt must name its evidence report")

    def validate_binding(
        self,
        *,
        checkpoint_sha256: str,
        feature_space_id: str,
        latent_space_id: str,
        type_names: Sequence[object],
        target_donors: Sequence[str],
        receipt_path: PathLike,
    ) -> None:
        self.validate()
        if self.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("initialization receipt is bound to a different checkpoint")
        if self.feature_space_id != feature_space_id:
            raise ValueError("initialization receipt pathology feature space differs")
        if self.latent_space_id != latent_space_id:
            raise ValueError("initialization receipt latent space differs")
        if self.type_ontology_sha256 != ordered_identity_sha256(type_names):
            raise ValueError("initialization receipt cell-type ontology differs")
        target = set(target_donors)
        overlap = sorted(target & set(self.training_donors))
        if overlap:
            raise ValueError(
                "initialization checkpoint was trained on target donors: %s" % ", ".join(overlap)
            )
        missing_holdout = sorted(target - set(self.held_out_donors))
        if missing_holdout:
            raise ValueError(
                "initialization receipt did not hold out target donors: %s"
                % ", ".join(missing_holdout)
            )
        receipt_dir = Path(receipt_path).expanduser().resolve().parent
        evidence = Path(self.evidence_report).expanduser()
        if not evidence.is_absolute():
            evidence = receipt_dir / evidence
        evidence = evidence.resolve()
        if not evidence.is_file():
            raise ValueError("initialization evidence report does not exist")
        digest = _sha256_path(evidence)
        if digest != self.evidence_report_sha256:
            raise ValueError("initialization evidence report hash differs from the receipt")
        validate_initialization_evidence_report(
            evidence,
            checkpoint_sha256=self.checkpoint_sha256,
            feature_space_id=self.feature_space_id,
            latent_space_id=self.latent_space_id,
            type_ontology_sha256=self.type_ontology_sha256,
            training_donors=self.training_donors,
            held_out_donors=self.held_out_donors,
        )
