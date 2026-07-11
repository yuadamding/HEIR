"""Lazy histology slide backends and patch-level image quality helpers."""

from __future__ import annotations

import abc
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from PIL import Image
except ImportError as error:  # pragma: no cover - exercised in minimal installations
    raise RuntimeError(
        "histology slide access requires Pillow; install heir-spatial[image]"
    ) from error

from .coordinates import MPP, PixelMicronTransform, normalize_mpp

PathLike = Union[str, Path]
Size = Tuple[int, int]


def _size(values: Sequence[int], name: str = "size") -> Size:
    result = tuple(int(value) for value in values)
    if len(result) != 2 or result[0] <= 0 or result[1] <= 0:
        raise ValueError("%s must contain two positive integers" % name)
    return result


def _location(values: Sequence[float]) -> Tuple[int, int]:
    result = tuple(float(value) for value in values)
    if len(result) != 2 or not np.isfinite(result).all():
        raise ValueError("location must contain two finite values")
    return int(round(result[0])), int(round(result[1]))


class SlideBackend(abc.ABC):
    """Common lazy interface for level-zero-coordinate histology access."""

    backend_name = "abstract"

    def __init__(self, path: PathLike, native_mpp: MPP) -> None:
        self.path = Path(path).expanduser().resolve()
        self._native_mpp = normalize_mpp(native_mpp)
        self._closed = False

    @property
    @abc.abstractmethod
    def dimensions(self) -> Size:
        """Level-zero ``(width, height)`` in pixels."""

    @property
    @abc.abstractmethod
    def level_dimensions(self) -> Tuple[Size, ...]:
        """Dimensions for all available pyramid levels."""

    @property
    @abc.abstractmethod
    def level_downsamples(self) -> Tuple[float, ...]:
        """Downsampling relative to level zero for every level."""

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def native_mpp(self) -> Tuple[float, float]:
        return self._native_mpp

    @property
    def mpp_x(self) -> float:
        return self._native_mpp[0]

    @property
    def mpp_y(self) -> float:
        return self._native_mpp[1]

    @property
    def coordinate_transform(self) -> PixelMicronTransform:
        return PixelMicronTransform(self.native_mpp)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("slide is closed")

    @abc.abstractmethod
    def read_region(
        self,
        location: Sequence[float],
        level: int,
        size: Sequence[int],
        mode: str = "RGB",
    ) -> Image.Image:
        """Read a patch.

        ``location`` is always expressed in native level-zero pixels, matching
        OpenSlide semantics. ``size`` is expressed in pixels at ``level``.
        """

    def read_native_region(
        self,
        location: Sequence[float],
        size: Sequence[int],
        mode: str = "RGB",
    ) -> Image.Image:
        """Read a level-zero patch without materializing the whole slide."""

        return self.read_region(location, 0, size, mode=mode)

    def read_region_um(
        self,
        origin_um: Sequence[float],
        size_um: Sequence[float],
        output_size: Optional[Sequence[int]] = None,
        mode: str = "RGB",
    ) -> Image.Image:
        """Read a physical-size level-zero patch using the slide calibration."""

        origin = self.coordinate_transform.microns_to_native(np.asarray(origin_um, dtype=float))
        physical_size = np.asarray(size_um, dtype=np.float64)
        if physical_size.shape != (2,) or not np.isfinite(physical_size).all():
            raise ValueError("size_um must contain two finite values")
        if bool((physical_size <= 0).any()):
            raise ValueError("size_um values must be positive")
        pixel_size = np.maximum(
            1,
            np.ceil(physical_size / np.asarray(self.native_mpp)).astype(np.int64),
        )
        patch = self.read_native_region(
            (float(origin[0]), float(origin[1])),
            (int(pixel_size[0]), int(pixel_size[1])),
            mode=mode,
        )
        if output_size is not None:
            target = _size(output_size, "output_size")
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            patch = patch.resize(target, resample=resampling)
        return patch

    @abc.abstractmethod
    def get_thumbnail(self, max_size: Sequence[int], mode: str = "RGB") -> Image.Image:
        """Return an aspect-preserving thumbnail bounded by ``max_size``."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release file handles."""

    def __enter__(self) -> "SlideBackend":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class OpenSlideBackend(SlideBackend):
    """Pyramidal slide backend powered by the optional ``openslide`` package."""

    backend_name = "openslide"

    def __init__(self, path: PathLike, native_mpp: Optional[MPP] = None) -> None:
        slide_path = Path(path).expanduser().resolve()
        if not slide_path.is_file():
            raise FileNotFoundError(str(slide_path))
        try:
            import openslide  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "OpenSlide support requires openslide-python and the OpenSlide shared library"
            ) from error
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as error:
            raise ValueError("OpenSlide could not open %s: %s" % (slide_path, error)) from error

        if native_mpp is None:
            properties = slide.properties
            x_key = getattr(openslide, "PROPERTY_NAME_MPP_X", "openslide.mpp-x")
            y_key = getattr(openslide, "PROPERTY_NAME_MPP_Y", "openslide.mpp-y")
            x_value = properties.get(x_key) or properties.get("aperio.MPP")
            y_value = properties.get(y_key) or x_value
            if x_value is None or y_value is None:
                slide.close()
                raise ValueError(
                    "slide has no trustworthy MPP metadata; provide native_mpp explicitly"
                )
            native_mpp = (float(x_value), float(y_value))
        try:
            super().__init__(slide_path, native_mpp)
        except Exception:
            slide.close()
            raise
        self._slide = slide
        self._dimensions: Size = (int(slide.dimensions[0]), int(slide.dimensions[1]))
        self._level_dimensions: Tuple[Size, ...] = tuple(
            (int(dimensions[0]), int(dimensions[1])) for dimensions in slide.level_dimensions
        )
        self._level_downsamples = tuple(float(value) for value in slide.level_downsamples)

    @property
    def dimensions(self) -> Size:
        return self._dimensions

    @property
    def level_dimensions(self) -> Tuple[Size, ...]:
        return self._level_dimensions

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        return self._level_downsamples

    def read_region(
        self,
        location: Sequence[float],
        level: int,
        size: Sequence[int],
        mode: str = "RGB",
    ) -> Image.Image:
        self._ensure_open()
        if int(level) != level or not 0 <= int(level) < self.level_count:
            raise ValueError("level is outside the slide pyramid")
        patch = self._slide.read_region(_location(location), int(level), _size(size))
        return patch.convert(mode)

    def get_thumbnail(self, max_size: Sequence[int], mode: str = "RGB") -> Image.Image:
        self._ensure_open()
        return self._slide.get_thumbnail(_size(max_size, "max_size")).convert(mode)

    def close(self) -> None:
        if not self._closed:
            self._slide.close()
            self._closed = True


_PIL_OPEN_LOCK = threading.Lock()


def _open_large_pil_image(path: Path) -> Image.Image:
    """Open trusted local pathology imagery without Pillow's web-image limit.

    Only the header is opened here; pixels remain lazy.  The global setting is
    restored while holding a lock so other Pillow users retain normal bomb
    protection.
    """

    with _PIL_OPEN_LOCK:
        previous_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                return Image.open(path)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit


class PILSlideBackend(SlideBackend):
    """Single-level lazy Pillow backend for non-pyramidal TIFF and common images.

    Pillow images do not provide reliable pathology calibration.  Therefore an
    explicit ``native_mpp`` override is mandatory rather than interpreting DPI
    metadata as microscopy MPP.
    """

    backend_name = "pil"

    def __init__(self, path: PathLike, native_mpp: Optional[MPP] = None) -> None:
        slide_path = Path(path).expanduser().resolve()
        if not slide_path.is_file():
            raise FileNotFoundError(str(slide_path))
        if native_mpp is None:
            raise ValueError("PIL slides require an explicit native_mpp override")
        image = _open_large_pil_image(slide_path)
        try:
            super().__init__(slide_path, native_mpp)
        except Exception:
            image.close()
            raise
        self._image = image
        self._dimensions: Size = (int(image.size[0]), int(image.size[1]))
        self._lock = threading.RLock()

    @property
    def dimensions(self) -> Size:
        return self._dimensions

    @property
    def level_dimensions(self) -> Tuple[Size, ...]:
        return (self._dimensions,)

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        return (1.0,)

    def read_region(
        self,
        location: Sequence[float],
        level: int,
        size: Sequence[int],
        mode: str = "RGB",
    ) -> Image.Image:
        self._ensure_open()
        if int(level) != 0:
            raise ValueError("PILSlideBackend has only level 0")
        x, y = _location(location)
        width, height = _size(size)
        # Crop is region based and keeps image opening lazy.  Depending on TIFF
        # compression Pillow may decode more than the requested strip, but this
        # API never creates a full-slide NumPy array.
        with self._lock:
            return self._image.crop((x, y, x + width, y + height)).convert(mode)

    def get_thumbnail(self, max_size: Sequence[int], mode: str = "RGB") -> Image.Image:
        self._ensure_open()
        target = _size(max_size, "max_size")
        with self._lock:
            thumbnail = self._image.copy()
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        thumbnail.thumbnail(target, resample=resampling)
        return thumbnail.convert(mode)

    def close(self) -> None:
        if not self._closed:
            self._image.close()
            self._closed = True


def open_slide(
    path: PathLike,
    native_mpp: Optional[MPP] = None,
    backend: str = "auto",
) -> SlideBackend:
    """Open a slide through OpenSlide or the single-level Pillow fallback.

    ``backend='auto'`` tries OpenSlide first and falls back to Pillow only when
    the file is unsupported.  A supplied ``native_mpp`` always overrides slide
    metadata and is required for the Pillow fallback.
    """

    selected = backend.strip().lower()
    if selected not in ("auto", "openslide", "pil", "pillow"):
        raise ValueError("backend must be 'auto', 'openslide', or 'pil'")
    if selected == "openslide":
        return OpenSlideBackend(path, native_mpp=native_mpp)
    if selected in ("pil", "pillow"):
        return PILSlideBackend(path, native_mpp=native_mpp)

    try:
        return OpenSlideBackend(path, native_mpp=native_mpp)
    except RuntimeError:
        # The optional dependency is unavailable; Pillow may still handle the
        # image.  It will issue the correct explicit-MPP error if needed.
        return PILSlideBackend(path, native_mpp=native_mpp)
    except ValueError as openslide_error:
        try:
            return PILSlideBackend(path, native_mpp=native_mpp)
        except Exception as pil_error:
            raise ValueError(
                "neither OpenSlide nor Pillow could open %s (OpenSlide: %s; Pillow: %s)"
                % (path, openslide_error, pil_error)
            ) from pil_error


def _rgb_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, Image.Image):
        values = np.asarray(image.convert("RGB"))
    else:
        values = np.asarray(image)
        if values.ndim == 2:
            values = np.repeat(values[..., None], 3, axis=-1)
        if values.ndim != 3 or values.shape[-1] not in (3, 4):
            raise ValueError("image array must have shape (height, width, 3 or 4)")
        values = values[..., :3]
    if values.size == 0:
        raise ValueError("image cannot be empty")
    if np.issubdtype(values.dtype, np.integer):
        maximum = float(np.iinfo(values.dtype).max)
        rgb = values.astype(np.float32) / maximum
    else:
        rgb = values.astype(np.float32)
        if not np.isfinite(rgb).all():
            raise ValueError("image must contain only finite values")
        if float(rgb.max()) > 1.0:
            rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def tissue_fraction(
    image: Union[Image.Image, np.ndarray],
    saturation_threshold: float = 0.05,
    maximum_luminance: float = 0.95,
) -> float:
    """Estimate stained tissue fraction in a patch without OpenCV."""

    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be in [0, 1]")
    if not 0.0 <= maximum_luminance <= 1.0:
        raise ValueError("maximum_luminance must be in [0, 1]")
    rgb = _rgb_array(image)
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = (maximum - minimum) / np.maximum(maximum, np.finfo(np.float32).eps)
    mask = (saturation >= saturation_threshold) & (_luminance(rgb) <= maximum_luminance)
    return float(mask.mean())


def blur_score(image: Union[Image.Image, np.ndarray]) -> float:
    """Variance of a discrete luminance Laplacian; larger means sharper."""

    gray = _luminance(_rgb_array(image))
    if min(gray.shape) < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(np.var(laplacian, dtype=np.float64))


@dataclass(frozen=True)
class ExposureMetrics:
    mean_intensity: float
    standard_deviation: float
    dark_fraction: float
    bright_fraction: float


def exposure_metrics(
    image: Union[Image.Image, np.ndarray],
    dark_threshold: float = 0.05,
    bright_threshold: float = 0.95,
) -> ExposureMetrics:
    """Summarize under/overexposure using patch luminance."""

    if not 0.0 <= dark_threshold < bright_threshold <= 1.0:
        raise ValueError("exposure thresholds must satisfy 0 <= dark < bright <= 1")
    gray = _luminance(_rgb_array(image))
    return ExposureMetrics(
        mean_intensity=float(gray.mean()),
        standard_deviation=float(gray.std()),
        dark_fraction=float((gray <= dark_threshold).mean()),
        bright_fraction=float((gray >= bright_threshold).mean()),
    )


@dataclass(frozen=True)
class PatchQC:
    tissue_fraction: float
    blur_score: float
    exposure: ExposureMetrics
    passed: bool
    issues: Tuple[str, ...]


def assess_patch_qc(
    image: Union[Image.Image, np.ndarray],
    minimum_tissue_fraction: float = 0.1,
    minimum_blur_score: float = 1e-4,
    maximum_dark_fraction: float = 0.8,
    maximum_bright_fraction: float = 0.8,
) -> PatchQC:
    """Apply transparent tissue, blur, and exposure checks to one patch."""

    for name, value in (
        ("minimum_tissue_fraction", minimum_tissue_fraction),
        ("maximum_dark_fraction", maximum_dark_fraction),
        ("maximum_bright_fraction", maximum_bright_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0, 1]" % name)
    if minimum_blur_score < 0.0 or not np.isfinite(minimum_blur_score):
        raise ValueError("minimum_blur_score must be finite and nonnegative")
    tissue = tissue_fraction(image)
    blur = blur_score(image)
    exposure = exposure_metrics(image)
    issues = []
    if tissue < minimum_tissue_fraction:
        issues.append("low_tissue")
    if blur < minimum_blur_score:
        issues.append("blurred")
    if exposure.dark_fraction > maximum_dark_fraction:
        issues.append("underexposed")
    if exposure.bright_fraction > maximum_bright_fraction:
        issues.append("overexposed")
    return PatchQC(tissue, blur, exposure, not issues, tuple(issues))


# Common concise alias used by preprocessing pipelines.
patch_qc = assess_patch_qc


__all__ = [
    "SlideBackend",
    "OpenSlideBackend",
    "PILSlideBackend",
    "open_slide",
    "ExposureMetrics",
    "PatchQC",
    "tissue_fraction",
    "blur_score",
    "exposure_metrics",
    "assess_patch_qc",
    "patch_qc",
]
