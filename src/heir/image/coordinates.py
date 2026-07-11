"""Coordinate transforms shared by histology, nuclei, and spatial assays.

All coordinates use ``(x, y)`` order.  Native pixels refer to level-zero slide
pixels; micrometers refer to a physical sample coordinate system.  The
transform is represented as a homogeneous 3-by-3 affine matrix so translations,
anisotropic pixel sizes, rotations, and a flipped y axis remain round-trippable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np

Number = Union[int, float, np.number]
MPP = Union[Number, Sequence[Number]]


def normalize_mpp(native_mpp: MPP) -> Tuple[float, float]:
    """Return a validated ``(mpp_x, mpp_y)`` pair.

    A scalar denotes isotropic pixels.  Values must be finite and strictly
    positive; accepting missing or zero MPP silently would make cross-slide
    neighborhood distances biologically meaningless.
    """

    array = np.asarray(native_mpp, dtype=np.float64)
    if array.ndim == 0:
        value = float(array)
        values = (value, value)
    elif array.shape == (2,):
        values = (float(array[0]), float(array[1]))
    else:
        raise ValueError("native_mpp must be a scalar or a two-value sequence")
    if not np.isfinite(values).all() or values[0] <= 0.0 or values[1] <= 0.0:
        raise ValueError("native_mpp values must be finite and strictly positive")
    return values


def _point_array(points: np.ndarray, name: str = "points") -> Tuple[np.ndarray, Tuple[int, ...]]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError("%s must have shape (..., 2)" % name)
    if not np.isfinite(values).all():
        raise ValueError("%s must contain only finite values" % name)
    return values.reshape(-1, 2), values.shape


def _pair(values: Sequence[Number], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("%s must contain two finite values" % name)
    return result


@dataclass(frozen=True)
class AffineTransform2D:
    """An invertible homogeneous affine transform in two dimensions."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if matrix.shape != (3, 3):
            raise ValueError("matrix must have shape (3, 3)")
        if not np.isfinite(matrix).all():
            raise ValueError("matrix must contain only finite values")
        if not np.allclose(matrix[2], np.array([0.0, 0.0, 1.0]), atol=1e-12):
            raise ValueError("matrix must be an affine homogeneous transform")
        if abs(float(np.linalg.det(matrix[:2, :2]))) <= np.finfo(np.float64).eps:
            raise ValueError("matrix must be invertible")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls) -> "AffineTransform2D":
        """Construct an identity transform."""

        return cls(np.eye(3, dtype=np.float64))

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Apply the forward transform while preserving leading dimensions."""

        flat, shape = _point_array(points)
        result = flat @ self.matrix[:2, :2].T + self.matrix[:2, 2]
        return result.reshape(shape)

    apply = transform

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """Apply the exact inverse transform."""

        return self.inverse().transform(points)

    def inverse(self) -> "AffineTransform2D":
        """Return the inverse affine transform."""

        return AffineTransform2D(np.linalg.inv(self.matrix))

    def compose(self, after: "AffineTransform2D") -> "AffineTransform2D":
        """Return a transform that applies ``self`` and then ``after``."""

        if not isinstance(after, AffineTransform2D):
            raise TypeError("after must be an AffineTransform2D")
        return AffineTransform2D(after.matrix @ self.matrix)


class PixelMicronTransform:
    """Bidirectional native-pixel to micrometer affine transform.

    Args:
        native_mpp: Scalar or ``(mpp_x, mpp_y)`` pixel spacing.
        pixel_origin: Native pixel mapped to ``micron_origin``.
        micron_origin: Physical coordinate corresponding to ``pixel_origin``.
        rotation_degrees: Counter-clockwise physical rotation.
        flip_y: Reverse the native y axis before rotation.  This is useful when
            registering image coordinates to a Cartesian assay coordinate frame.
    """

    def __init__(
        self,
        native_mpp: MPP,
        pixel_origin: Sequence[Number] = (0.0, 0.0),
        micron_origin: Sequence[Number] = (0.0, 0.0),
        rotation_degrees: float = 0.0,
        flip_y: bool = False,
    ) -> None:
        self._native_mpp = normalize_mpp(native_mpp)
        pixel_origin_array = _pair(pixel_origin, "pixel_origin")
        micron_origin_array = _pair(micron_origin, "micron_origin")
        angle = np.deg2rad(float(rotation_degrees))
        if not np.isfinite(angle):
            raise ValueError("rotation_degrees must be finite")
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
        spacing = np.diag(
            (self._native_mpp[0], -self._native_mpp[1] if flip_y else self._native_mpp[1])
        )
        linear = rotation @ spacing
        translation = micron_origin_array - linear @ pixel_origin_array
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = linear
        matrix[:2, 2] = translation
        self._affine = AffineTransform2D(matrix)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "PixelMicronTransform":
        """Construct from an already calibrated pixel-to-micron matrix."""

        affine = AffineTransform2D(matrix)
        instance = cls.__new__(cls)
        instance._affine = affine
        # Column norms are the physical sizes of native x/y basis vectors.
        instance._native_mpp = (
            float(np.linalg.norm(affine.matrix[:2, 0])),
            float(np.linalg.norm(affine.matrix[:2, 1])),
        )
        return instance

    @property
    def native_mpp(self) -> Tuple[float, float]:
        return self._native_mpp

    @property
    def matrix(self) -> np.ndarray:
        return self._affine.matrix

    @property
    def affine(self) -> AffineTransform2D:
        return self._affine

    def native_to_microns(self, points: np.ndarray) -> np.ndarray:
        """Convert level-zero ``(x, y)`` pixels to micrometers."""

        return self._affine.transform(points)

    def microns_to_native(self, points: np.ndarray) -> np.ndarray:
        """Convert micrometer ``(x, y)`` positions to level-zero pixels."""

        return self._affine.inverse_transform(points)

    # Explicit aliases make call sites self-documenting without forcing one
    # naming convention on downstream code.
    pixels_to_microns = native_to_microns
    microns_to_pixels = microns_to_native
    native_pixels_to_microns = native_to_microns
    microns_to_native_pixels = microns_to_native


NativePixelMicronTransform = PixelMicronTransform


def native_pixels_to_microns(
    points: np.ndarray,
    native_mpp: MPP,
    pixel_origin: Sequence[Number] = (0.0, 0.0),
    micron_origin: Sequence[Number] = (0.0, 0.0),
) -> np.ndarray:
    """Convenience conversion for an axis-aligned calibration."""

    return PixelMicronTransform(native_mpp, pixel_origin, micron_origin).native_to_microns(points)


def microns_to_native_pixels(
    points: np.ndarray,
    native_mpp: MPP,
    pixel_origin: Sequence[Number] = (0.0, 0.0),
    micron_origin: Sequence[Number] = (0.0, 0.0),
) -> np.ndarray:
    """Inverse of :func:`native_pixels_to_microns`."""

    return PixelMicronTransform(native_mpp, pixel_origin, micron_origin).microns_to_native(points)


__all__ = [
    "AffineTransform2D",
    "PixelMicronTransform",
    "NativePixelMicronTransform",
    "normalize_mpp",
    "native_pixels_to_microns",
    "microns_to_native_pixels",
]
