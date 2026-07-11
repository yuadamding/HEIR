import csv

import numpy as np
import pytest
from PIL import Image

from heir.image.nuclei import load_feature_bundle, load_nuclei
from heir.image.slides import (
    PILSlideBackend,
    assess_patch_qc,
    blur_score,
    exposure_metrics,
    open_slide,
    tissue_fraction,
)


def _tiny_rgb():
    image = np.full((12, 16, 3), 255, dtype=np.uint8)
    image[2:10, 3:13] = [160, 70, 150]
    image[4:8, 6:10] = [60, 35, 100]
    return image


def test_pil_slide_backend_is_lazy_region_based_and_calibrated(tmp_path):
    path = tmp_path / "single-level.tiff"
    pixels = _tiny_rgb()
    Image.fromarray(pixels).save(path)

    with PILSlideBackend(path, native_mpp=(0.5, 0.25)) as slide:
        assert slide.backend_name == "pil"
        assert slide.dimensions == (16, 12)
        assert slide.level_count == 1
        assert slide.level_dimensions == ((16, 12),)
        assert slide.native_mpp == (0.5, 0.25)
        patch = np.asarray(slide.read_native_region((3, 2), (4, 5)))
        np.testing.assert_array_equal(patch, pixels[2:7, 3:7])
        physical_patch = slide.read_region_um((1.5, 0.5), (2.0, 1.25))
        assert physical_patch.size == (4, 5)
        np.testing.assert_allclose(
            slide.coordinate_transform.native_to_microns([[3.0, 2.0]]),
            [[1.5, 0.5]],
        )
    with pytest.raises(RuntimeError, match="closed"):
        slide.read_native_region((0, 0), (2, 2))


def test_auto_backend_falls_back_to_pillow_and_requires_explicit_mpp(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(_tiny_rgb()).save(path)
    with pytest.raises(ValueError, match="native_mpp"):
        PILSlideBackend(path)
    with open_slide(path, native_mpp=0.5, backend="auto") as slide:
        assert slide.dimensions == (16, 12)
        assert slide.native_mpp == (0.5, 0.5)
    with pytest.raises(ValueError, match="backend"):
        open_slide(path, native_mpp=0.5, backend="unknown")


def test_patch_qc_tissue_blur_and_exposure_helpers():
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    tissue = np.empty((16, 16, 3), dtype=np.uint8)
    tissue[:, ::2] = [190, 100, 170]
    tissue[:, 1::2] = [70, 30, 100]

    assert tissue_fraction(white) == 0.0
    assert tissue_fraction(tissue) > 0.95
    assert blur_score(tissue) > blur_score(np.full_like(tissue, [150, 80, 140]))
    assert exposure_metrics(white).bright_fraction == 1.0
    assert exposure_metrics(black).dark_fraction == 1.0

    good = assess_patch_qc(
        tissue,
        minimum_tissue_fraction=0.5,
        minimum_blur_score=0.0,
        maximum_bright_fraction=0.9,
    )
    assert good.passed
    bad = assess_patch_qc(white, minimum_tissue_fraction=0.1, minimum_blur_score=1e-6)
    assert not bad.passed
    assert "low_tissue" in bad.issues
    assert "blurred" in bad.issues
    assert "overexposed" in bad.issues


def test_nucleus_csv_import_is_canonical_and_preserves_morphology(tmp_path):
    path = tmp_path / "nuclei.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cell_id",
                "centroid_x",
                "centroid_y",
                "area",
                "eccentricity",
                "equiv_diameter",
                "confidence",
                "cell_type",
                "tile",
            ]
        )
        writer.writerow(["7", 10.5, 20.25, 42.0, 0.3, 7.3, 0.9, "Lymphocyte", "a"])
        writer.writerow(["8", 30.0, 40.0, 61.0, 0.7, 8.8, 0.6, "Tumor", "b"])

    nuclei = load_nuclei(path, sample_id="sample-1")
    np.testing.assert_array_equal(nuclei.nucleus_ids, ["sample-1::7", "sample-1::8"])
    np.testing.assert_array_equal(nuclei.source_ids, ["7", "8"])
    np.testing.assert_allclose(nuclei.centroids_px, [[10.5, 20.25], [30.0, 40.0]])
    assert nuclei.morphology_names == ("area", "eccentricity", "equiv_diameter")
    np.testing.assert_allclose(nuclei.morphology, [[42.0, 0.3, 7.3], [61.0, 0.7, 8.8]])
    np.testing.assert_allclose(nuclei.confidence, [0.9, 0.6])
    np.testing.assert_array_equal(nuclei.cell_types, ["Lymphocyte", "Tumor"])
    np.testing.assert_array_equal(nuclei.metadata["tile"], ["a", "b"])


def test_feature_bundle_loader_validates_and_aligns_ids(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(
        path,
        nucleus_ids=np.array(["sample::b", "sample::a"]),
        features=np.array([[2.0, 20.0], [1.0, 10.0]], dtype=np.float32),
        feature_names=np.array(["morphology", "context"]),
        centroids_px=np.array([[4.0, 5.0], [1.0, 2.0]]),
        quality=np.array([0.8, 0.9]),
    )
    bundle = load_feature_bundle(path, expected_ids=["sample::a", "sample::b"])
    np.testing.assert_array_equal(bundle.nucleus_ids, ["sample::a", "sample::b"])
    np.testing.assert_allclose(bundle.features, [[1.0, 10.0], [2.0, 20.0]])
    np.testing.assert_allclose(bundle.coordinates, [[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_allclose(bundle.metadata["quality"], [0.9, 0.8])
    assert bundle.feature_names == ("morphology", "context")

    with pytest.raises(ValueError, match="do not match"):
        load_feature_bundle(path, expected_ids=["sample::a", "sample::missing"])
