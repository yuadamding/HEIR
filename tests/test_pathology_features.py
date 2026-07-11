import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from heir.image.features import (
    EncoderDescriptor,
    _devices_compatible,
    extract_nucleus_pathology_features,
    pathology_feature_space_id,
    save_pathology_feature_npz,
)
from heir.image.nuclei import load_feature_bundle
from heir.image.slides import PILSlideBackend


class TinyFrozenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, -1.0, 0.5],
                    ]
                )
            )
        self.requires_grad_(False).eval()

    def forward(self, images):
        pooled = images.mean(dim=(2, 3))
        return torch.nn.functional.normalize(self.projection(pooled), dim=-1)


def _descriptor():
    return EncoderDescriptor(
        name="tiny-test-encoder",
        architecture="rgb-linear",
        package="test",
        package_version="1",
        checkpoint_sha256=hashlib.sha256(b"tiny-test-weights").hexdigest(),
        output_dim=4,
        input_size=16,
        normalization_mean=(0.5, 0.5, 0.5),
        normalization_std=(0.25, 0.25, 0.25),
        scientific_role="unit test",
    )


def _slide_pixels():
    yy, xx = np.mgrid[:96, :112]
    image = np.empty((96, 112, 3), dtype=np.uint8)
    image[..., 0] = 80 + (xx % 80)
    image[..., 1] = 30 + (yy % 100)
    image[..., 2] = 100 + ((xx + yy) % 100)
    image[:8] = 255
    return image


def test_multiscale_extraction_is_batched_bounded_and_self_identifying(tmp_path):
    slide_path = tmp_path / "slide.png"
    Image.fromarray(_slide_pixels()).save(slide_path)
    identifiers = np.asarray(["n0", "n1", "n2"])
    centroids = np.asarray([[24.0, 30.0], [60.0, 42.0], [100.0, 88.0]])
    descriptor = _descriptor()

    with PILSlideBackend(slide_path, native_mpp=1.0) as slide:
        result = extract_nucleus_pathology_features(
            slide,
            identifiers,
            centroids,
            TinyFrozenEncoder(),
            descriptor,
            patch_diameters_um=(16.0, 32.0),
            batch_size=2,
            device="cpu",
            mixed_precision=False,
        )

    assert result.features.shape == (3, 8)
    assert result.tissue_fraction_by_scale.shape == (3, 2)
    assert result.telemetry.batches == 2
    assert result.telemetry.peak_cuda_memory_bytes == 0
    assert result.telemetry.mixed_precision is False
    assert result.feature_space_id.startswith("tiny-test-encoder:sha256:")
    # Every scale is independently normalized before concatenation.
    np.testing.assert_allclose(
        np.linalg.norm(result.features.reshape(3, 2, 4), axis=2),
        np.ones((3, 2)),
        atol=1.0e-6,
    )

    output = tmp_path / "features.npz"
    digest = hashlib.sha256(b"source").hexdigest()
    save_pathology_feature_npz(
        result,
        output,
        slide_sha256=digest,
        nuclei_sha256=digest,
    )
    bundle = load_feature_bundle(output, expected_ids=identifiers)
    np.testing.assert_allclose(bundle.features, result.features)
    np.testing.assert_allclose(bundle.coordinates, centroids)
    assert bundle.feature_names == result.feature_names
    np.testing.assert_allclose(
        bundle.metadata["tissue_fraction_by_scale"],
        result.tissue_fraction_by_scale,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["feature_space_id"].item()) == result.feature_space_id
        assert str(archive["slide_sha256"].item()) == digest
        assert str(archive["__contract__"].item()) == "heir.nucleus_pathology_features"
    with pytest.raises(FileExistsError):
        save_pathology_feature_npz(
            result,
            output,
            slide_sha256=digest,
            nuclei_sha256=digest,
        )


def test_feature_space_identity_covers_scales_precision_and_checkpoint():
    descriptor = _descriptor()
    base = pathology_feature_space_id(descriptor, (16.0, 32.0), mixed_precision=False)
    assert base == pathology_feature_space_id(descriptor, (16.0, 32.0), mixed_precision=False)
    assert base != pathology_feature_space_id(descriptor, (32.0, 16.0), mixed_precision=False)
    assert base != pathology_feature_space_id(descriptor, (16.0, 32.0), mixed_precision=True)
    changed = EncoderDescriptor(
        **{
            **descriptor.__dict__,
            "checkpoint_sha256": hashlib.sha256(b"different").hexdigest(),
        }
    )
    assert base != pathology_feature_space_id(changed, (16.0, 32.0), mixed_precision=False)


def test_implicit_and_explicit_cuda_zero_devices_are_compatible():
    assert _devices_compatible(torch.device("cuda"), torch.device("cuda:0"))
    assert _devices_compatible(torch.device("cuda:0"), torch.device("cuda"))
    assert not _devices_compatible(torch.device("cuda:0"), torch.device("cuda:1"))
    assert not _devices_compatible(torch.device("cpu"), torch.device("cuda"))


def test_extractor_rejects_duplicate_ids_and_cpu_amp(tmp_path):
    slide_path = tmp_path / "slide.png"
    Image.fromarray(_slide_pixels()).save(slide_path)
    with PILSlideBackend(slide_path, native_mpp=1.0) as slide:
        with pytest.raises(ValueError, match="unique"):
            extract_nucleus_pathology_features(
                slide,
                ["same", "same"],
                np.asarray([[20.0, 20.0], [40.0, 40.0]]),
                TinyFrozenEncoder(),
                _descriptor(),
                device="cpu",
                mixed_precision=False,
            )
        with pytest.raises(ValueError, match="only on CUDA"):
            extract_nucleus_pathology_features(
                slide,
                ["n0"],
                np.asarray([[20.0, 20.0]]),
                TinyFrozenEncoder(),
                _descriptor(),
                device="cpu",
                mixed_precision=True,
            )
