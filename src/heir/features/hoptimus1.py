"""Local adapter for the access-controlled H-optimus-1 encoder."""

from __future__ import annotations

from pathlib import Path

from .base import EncoderManifest, TorchPatchEncoder, load_local_state_dict, verified_model_file

HOPTIMUS1_REPOSITORY = "bioptimus/H-optimus-1"


class HOptimus1Encoder(TorchPatchEncoder):
    """Load H-optimus-1 only after a real checkpoint is pinned in its manifest."""

    def __init__(self, model_dir: Path, manifest: EncoderManifest, device: str = "cuda"):
        try:
            import timm
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install HEIR with the hest optional dependencies") from error
        if not manifest.available:
            raise ValueError("H-optimus-1 is inaccessible: %s" % manifest.status_reason)
        if manifest.repository != HOPTIMUS1_REPOSITORY or manifest.implementation != "timm_local":
            raise ValueError("H-optimus-1 manifest differs from the supported adapter")
        checkpoint = verified_model_file(
            model_dir, manifest.checkpoint_filename, manifest.checkpoint_sha256
        )
        model = timm.create_model(
            manifest.architecture,
            pretrained=False,
            num_classes=0,
            img_size=manifest.input_pixels,
            init_values=1.0e-5,
            dynamic_img_size=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        state = load_local_state_dict(checkpoint)
        model.load_state_dict(state, strict=True)
        super().__init__(model, manifest, device)
