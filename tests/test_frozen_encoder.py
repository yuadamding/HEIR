"""Frozen feature extraction rejects accidental backbone adaptation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from heir.features.base import TorchPatchEncoder, load_encoder_manifest


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 3, bias=False)

    def forward(self, pixels):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        return self.projection(pixels.mean((2, 3)))


def _encoder():
    manifest = load_encoder_manifest(Path(__file__).parents[1] / "manifests/encoders/uni2h.json")
    manifest = replace(manifest, feature_width=3, input_pixels=4)
    return TorchPatchEncoder(TinyBackbone(), manifest, "cpu")


def test_extraction_freezes_backbone_and_repeats_without_weight_changes():
    encoder = _encoder()
    before = {name: value.clone() for name, value in encoder._model.state_dict().items()}
    patches = np.arange(2 * 5 * 6 * 3, dtype=np.uint8).reshape(2, 5, 6, 3)
    first, second = encoder.encode(patches), encoder.encode(patches)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 3) and np.isfinite(first).all()
    assert all(not parameter.requires_grad for parameter in encoder._model.parameters())
    assert all(parameter.grad is None for parameter in encoder._model.parameters())
    for name, value in encoder._model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("change", ["train_mode", "child_train_mode", "trainable_weight"])
def test_extraction_rejects_accidentally_unfrozen_model(change):
    encoder = _encoder()
    if change == "train_mode":
        encoder._model.train()
    elif change == "child_train_mode":
        encoder._model.projection.train()
    else:
        next(encoder._model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen encoder"):
        encoder.encode(np.zeros((1, 4, 4, 3), dtype=np.uint8))
