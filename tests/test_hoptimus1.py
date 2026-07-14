from __future__ import annotations

from pathlib import Path

from heir.features import hoptimus1 as adapter
from heir.features import load_encoder_manifest


class _Model:
    def __init__(self) -> None:
        self.loaded: tuple[object, bool] | None = None

    def load_state_dict(self, state: object, *, strict: bool) -> None:
        self.loaded = (state, strict)


def test_hoptimus1_uses_checkpoint_native_224_pixel_geometry(monkeypatch, tmp_path: Path) -> None:
    manifest = load_encoder_manifest(
        Path(__file__).parents[1] / "manifests" / "encoders" / "hoptimus1.json"
    )
    calls: list[tuple[str, dict[str, object]]] = []
    model = _Model()

    def create_model(architecture: str, **kwargs: object) -> _Model:
        calls.append((architecture, kwargs))
        return model

    monkeypatch.setattr("timm.create_model", create_model)
    monkeypatch.setattr(adapter, "verified_model_file", lambda *_args: tmp_path / "model")
    monkeypatch.setattr(adapter, "load_local_state_dict", lambda _path: {"weight": object()})
    monkeypatch.setattr(adapter.TorchPatchEncoder, "__init__", lambda *_args: None)

    adapter.HOptimus1Encoder(tmp_path, manifest, device="cuda")

    assert len(calls) == 1
    architecture, kwargs = calls[0]
    assert architecture == "vit_giant_patch14_reg4_dinov2"
    assert kwargs["img_size"] == 224
    assert kwargs["dynamic_img_size"] is False
    assert kwargs["init_values"] == 1.0e-5
    assert kwargs["pretrained"] is False
    assert kwargs["num_classes"] == 0
    assert model.loaded is not None
    assert model.loaded[1] is True
