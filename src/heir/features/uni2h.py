"""Offline, checksum-pinned UNI2-h patch encoder."""

from __future__ import annotations

from pathlib import Path

from .base import EncoderManifest, TorchPatchEncoder, load_local_state_dict, verified_model_file

UNI2H_REPOSITORY = "MahmoodLab/UNI2-h"
UNI2H_ARCHITECTURE = "vit_giant_patch14_224"


class UNI2HEncoder(TorchPatchEncoder):
    """Load the official UNI2-h state dict without network access."""

    def __init__(self, model_dir: Path, manifest: EncoderManifest, device: str = "cuda"):
        try:
            import timm
            import torch
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install HEIR with the hest optional dependencies") from error
        expected = {
            "availability": "available",
            "implementation": "uni2h_timm",
            "repository": UNI2H_REPOSITORY,
            "architecture": UNI2H_ARCHITECTURE,
            "feature_width": 1536,
            "input_pixels": 224,
            "pooling_rule": "direct_features",
        }
        for name, value in expected.items():
            if getattr(manifest, name) != value:
                raise ValueError("UNI2-h manifest %s differs from the supported adapter" % name)
        checkpoint = verified_model_file(
            model_dir, manifest.checkpoint_filename, manifest.checkpoint_sha256
        )
        if manifest.config_filename:
            verified_model_file(model_dir, manifest.config_filename, manifest.config_sha256)
        model = timm.create_model(
            model_name=UNI2H_ARCHITECTURE,
            pretrained=False,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1.0e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        state = load_local_state_dict(checkpoint)
        model.load_state_dict(state, strict=True)
        if int(getattr(model, "num_features", -1)) != manifest.feature_width:
            raise ValueError("loaded UNI2-h feature width differs from its manifest")
        super().__init__(model, manifest, device)
