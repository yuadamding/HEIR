"""Validated, pickle-free exports of relative-expression predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .schema import Prediction, identifiers

_SCHEMA = "heir.personalized.prediction.v2"
_AXES = ("cell_ids", "gene_ids", "state_ids")
_ARRAYS = (
    "coordinates_um",
    "assignment",
    "conditional_assignment",
    "transfer",
    "log_correction",
    "expression",
    "assignment_entropy",
    "reference_support",
    "registration_sensitivity",
    "supervised_genes",
    "independently_evaluated_genes",
)
_DERIVED = ("reference_transferred_genes", "unsupported_probability", "expression_adjustment")


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate(prediction: Prediction) -> dict[str, np.ndarray]:
    """Check exported axes and values independently of the model constructor."""
    for name in _AXES:
        identifiers(tuple(getattr(prediction, name)), name)
    if prediction.state_ids[0] != "unsupported":
        raise ValueError("the first prediction state must be unsupported")
    if prediction.expression_scale != "relative":
        raise ValueError("prediction exports require relative expression")
    if prediction.reference_support_calibrated is not False:
        raise ValueError("reference support is uncalibrated in this prediction schema")
    expected_policy = (
        "he_only" if len(prediction.state_ids) == 1 else "conditional_reference_unvalidated"
    )
    if prediction.unsupported_policy != expected_policy:
        raise ValueError("prediction unsupported policy does not match its reference state axis")
    for name in ("patient_id", "specimen_id"):
        value = getattr(prediction, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prediction {name} is required")
    n, g, k = (len(getattr(prediction, name)) for name in _AXES)
    shapes = {
        "coordinates_um": (n, 2),
        "assignment": (n, k),
        "conditional_assignment": (n, k - 1),
        "transfer": (n, g),
        "log_correction": (n, g),
        "expression": (n, g),
        "assignment_entropy": (n,),
        "reference_support": (n,),
        "registration_sensitivity": (n,),
        "supervised_genes": (g,),
        "independently_evaluated_genes": (g,),
    }
    arrays = {name: _numpy(getattr(prediction, name)) for name in _ARRAYS}
    for name, values in arrays.items():
        if values.shape != shapes[name]:
            raise ValueError(f"prediction {name} shape differs from its named axes")
        if name in {"supervised_genes", "independently_evaluated_genes"}:
            if values.dtype != bool:
                raise ValueError(f"prediction {name} must be boolean")
        elif values.dtype.kind not in "fiu":
            raise ValueError(f"prediction {name} must be real numeric data")
        elif name == "registration_sensitivity":
            if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
                raise ValueError("registration sensitivity must be nonnegative or NaN (unassessed)")
        elif not np.isfinite(values).all():
            raise ValueError(f"prediction {name} must be finite")
    for name in (
        "assignment",
        "conditional_assignment",
        "transfer",
        "expression",
        "assignment_entropy",
    ):
        if np.any(arrays[name] < 0):
            raise ValueError(f"prediction {name} must be nonnegative")
    if not np.allclose(arrays["assignment"].sum(1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("prediction assignment probabilities must sum to one")
    conditional = arrays["conditional_assignment"]
    if k > 1:
        if not np.allclose(conditional.sum(1), 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("conditional assignment probabilities must sum to one")
        # Joint probabilities can legitimately underflow to zero. The independently
        # exported conditional distribution remains available in that case.
        joint_known = arrays["assignment"][:, 1:]
        joint_total = joint_known.sum(1, keepdims=True)
        if not np.allclose(joint_known, conditional * joint_total, atol=1e-5, rtol=1e-5):
            raise ValueError("conditional assignment differs from known-state joint probabilities")
    support = arrays["reference_support"]
    if np.any(support < 0) or np.any(support > 1):
        raise ValueError("prediction reference support must lie in [0, 1]")
    if not np.allclose(support, 1 - arrays["assignment"][:, 0], atol=1e-5, rtol=1e-5):
        raise ValueError("reference support differs from unsupported assignment mass")
    q = arrays["assignment"]
    entropy = -(q * np.log(np.clip(q, 1e-12, None))).sum(1)
    if not np.allclose(arrays["assignment_entropy"], entropy, atol=1e-5, rtol=1e-5):
        raise ValueError("assignment entropy differs from exported probabilities")
    return arrays


def _derived(prediction: Prediction, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    has_reference = tuple(prediction.state_ids) != ("unsupported",)
    return {
        "reference_transferred_genes": np.full(len(prediction.gene_ids), has_reference, dtype=bool),
        "unsupported_probability": arrays["assignment"][:, 0],
        "expression_adjustment": arrays["expression"] - arrays["transfer"],
    }


def save_prediction(
    prediction: Prediction,
    path: str | Path,
    provenance: Optional[dict] = None,
) -> None:
    """Write one numeric/Unicode NPZ without overwriting an existing artifact.

    Provenance records supplied split/training identities; serialization does not
    establish independence or calibrate any uncertainty indicator.
    """
    arrays = _validate(prediction)
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("prediction provenance must be a JSON object")
    metadata = {
        "schema": _SCHEMA,
        "patient_id": prediction.patient_id,
        "specimen_id": prediction.specimen_id,
        "expression_scale": prediction.expression_scale,
        "reference_support_calibrated": prediction.reference_support_calibrated,
        "unsupported_policy": prediction.unsupported_policy,
        "provenance": {} if provenance is None else provenance,
        "uncertainty_interpretation": "diagnostic indicators; not calibrated expression intervals",
    }
    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    payload = {name: np.asarray(tuple(getattr(prediction, name)), dtype=str) for name in _AXES}
    payload.update(arrays)
    payload.update(_derived(prediction, arrays))
    payload["metadata_json"] = np.asarray(metadata_json)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier outputs even if callers race to publish.
    with path.open("xb") as handle:
        try:
            np.savez(handle, **payload)
        except BaseException:
            handle.close()
            path.unlink()
            raise


def load_prediction(path: str | Path) -> tuple[Prediction, dict]:
    """Read a validated prediction and its supplied provenance, with pickle disabled."""
    required = {*_AXES, *_ARRAYS, *_DERIVED, "metadata_json"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("prediction archive fields differ from the versioned schema")
        data = {name: archive[name].copy() for name in required}
    if data["metadata_json"].ndim != 0 or data["metadata_json"].dtype.kind != "U":
        raise ValueError("prediction metadata must be one Unicode JSON string")
    metadata = json.loads(data["metadata_json"].item())
    if not isinstance(metadata, dict) or metadata.get("schema") != _SCHEMA:
        raise ValueError("unsupported prediction archive schema")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("prediction provenance must be a JSON object")
    axes = {}
    for name in _AXES:
        if data[name].ndim != 1 or data[name].dtype.kind != "U":
            raise ValueError(f"prediction {name} must be a Unicode vector")
        axes[name] = tuple(data[name].tolist())
    # Validate array dtypes before passing untrusted archive values to torch.
    for name in _ARRAYS:
        if data[name].dtype.kind not in "fiub":
            raise ValueError(f"prediction {name} must be numeric or boolean")
    prediction = Prediction(
        **axes,
        **{name: torch.from_numpy(data[name]) for name in _ARRAYS},
        expression_scale=metadata.get("expression_scale"),
        patient_id=metadata.get("patient_id"),
        specimen_id=metadata.get("specimen_id"),
        reference_support_calibrated=metadata.get("reference_support_calibrated"),
        unsupported_policy=metadata.get("unsupported_policy"),
    )
    arrays = _validate(prediction)
    for name, expected in _derived(prediction, arrays).items():
        actual = data[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"derived prediction {name} dtype or shape mismatch")
        if not np.array_equal(actual, expected):
            raise ValueError(f"derived prediction {name} differs from primary values")
    return prediction, provenance
