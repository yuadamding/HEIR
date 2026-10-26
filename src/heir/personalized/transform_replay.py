"""Replay declared coordinate operations; never estimate a registration.

Units name the supplied coordinate frames. Reproducing an author analysis frame
does not itself verify physical calibration or biological tissue correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import identifiers


def _xy(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError("replay coordinates must be finite N-by-2 values")
    return result


@dataclass(frozen=True)
class TransformStep:
    name: str
    matrix: np.ndarray
    input_units: str
    output_units: str
    domain_maximum: tuple = (None, None)

    def __post_init__(self):
        if any(
            not isinstance(s, str) or not s.strip()
            for s in (self.name, self.input_units, self.output_units)
        ):
            raise ValueError("transform names and input/output units must be explicit")
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[2], [0, 0, 1])
            or np.linalg.det(matrix[:2, :2]) == 0
        ):
            raise ValueError("transform requires a finite nonsingular 2D affine matrix")
        if len(self.domain_maximum) != 2 or any(
            value is not None and not np.isfinite(value) for value in self.domain_maximum
        ):
            raise ValueError("transform domain maxima must be finite values or None")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    def to_dict(self):
        return {
            "name": self.name,
            "matrix": self.matrix.tolist(),
            "input_units": self.input_units,
            "output_units": self.output_units,
            "domain_maximum": list(self.domain_maximum),
        }


def replay_coordinates(coordinates, steps, *, input_units):
    """Apply an ordered affine chain to row-vector coordinates, without axis flips for display."""
    values = _xy(coordinates).copy()
    if not steps:
        raise ValueError("coordinate replay requires an explicit transform chain")
    units = input_units
    for step in steps:
        if not isinstance(step, TransformStep) or step.input_units != units:
            raise ValueError("transform chain input/output units do not match")
        for axis, maximum in enumerate(step.domain_maximum):
            if maximum is not None and np.any(values[:, axis] > maximum):
                raise ValueError("coordinate lies outside the declared reflection domain")
        values = values @ step.matrix[:2, :2].T + step.matrix[:2, 2]
        if not np.isfinite(values).all():
            raise ValueError("coordinate replay produced nonfinite values")
        units = step.output_units
    return values


def replay_piecewise(coordinates, piece_ids, transforms, *, input_units):
    """Replay independently declared fragment chains while preserving the original row order."""
    values = _xy(coordinates)
    labels = np.asarray(piece_ids)
    if labels.shape != (len(values),) or labels.dtype.kind not in "US":
        raise ValueError("piece IDs must be a string vector matching coordinates")
    if any(not label.strip() for label in labels) or set(labels) - set(transforms):
        raise ValueError("every coordinate piece must have a declared transform chain")
    outputs = {chain[-1].output_units for chain in transforms.values() if chain}
    if len(outputs) != 1 or any(not chain for chain in transforms.values()):
        raise ValueError("piecewise transforms must share one explicit output unit")
    result = np.empty_like(values)
    for label in set(labels):
        selected = labels == label
        result[selected] = replay_coordinates(
            values[selected], transforms[label], input_units=input_units
        )
    return result


def author_rigid_step(
    name,
    *,
    rotation_origin,
    degrees,
    shift,
    input_units,
    output_units,
    flip=None,
    reflection_maximum=None,
):
    """Encode spatial_utils.rotate: pre-flip origin, optional reflection, rotation, then shift.

    The author names ``v`` for x reflection and ``h`` for y reflection. Their
    abs(coordinate-maximum) equals maximum-coordinate on the original population;
    the declared maximum is enforced so this affine replay cannot silently extend
    that equality to points beyond the original reflection domain.
    """
    origin, translation = np.asarray(rotation_origin, float), np.asarray(shift, float)
    if (
        origin.shape != (2,)
        or translation.shape != (2,)
        or not np.isfinite([*origin, *translation, degrees]).all()
    ):
        raise ValueError("rotation requires finite origin, degrees and translation")
    if flip not in (None, "v", "h"):
        raise ValueError("author reflection must be None, 'v' (x), or 'h' (y)")
    reflection, offset, bounds = np.eye(2), np.zeros(2), [None, None]
    if flip is not None:
        if reflection_maximum is None or not np.isfinite(reflection_maximum):
            raise ValueError("author reflection requires the predeclared coordinate maximum")
        axis = 0 if flip == "v" else 1
        reflection[axis, axis], offset[axis], bounds[axis] = (
            -1,
            reflection_maximum,
            float(reflection_maximum),
        )
    elif reflection_maximum is not None:
        raise ValueError("reflection maximum supplied without a reflection")
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation @ reflection
    matrix[:2, 2] = (offset - origin) @ rotation.T + origin + translation
    return TransformStep(name, matrix, input_units, output_units, tuple(bounds))


def observation_join(source_ids, target_ids, *, target_suffix=""):
    """Join stable observation IDs with only an explicitly declared concatenate suffix removed."""
    sources, targets = tuple(source_ids), tuple(target_ids)
    identifiers(sources, "native observation IDs")
    identifiers(targets, "registered observation IDs")
    if not isinstance(target_suffix, str):
        raise ValueError("target observation suffix must be a string")
    if target_suffix and any(not name.endswith(target_suffix) for name in targets):
        raise ValueError("registered observation ID lacks its declared replicate suffix")
    names = tuple(name.removesuffix(target_suffix) if target_suffix else name for name in targets)
    identifiers(names, "joined native observation IDs")
    lookup = {name: row for row, name in enumerate(sources)}
    missing = set(names) - set(lookup)
    if missing:
        raise ValueError(
            f"registered observations absent from native source: {sorted(missing)[:5]}"
        )
    return np.asarray([lookup[name] for name in names], dtype=np.int64)
