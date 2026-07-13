"""Small deterministic RGB summary used only as a negative-control encoder."""

from __future__ import annotations

import hashlib

import numpy as np


class HandcraftedPatchEncoder:
    """Return channel mean, standard deviation, and quartiles (12 features)."""

    feature_width = 12
    manifest_sha256 = hashlib.sha256(b"heir.handcrafted_rgb_moments.v1").hexdigest()

    def encode(self, patches: np.ndarray) -> np.ndarray:
        values = np.asarray(patches)
        if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
            raise ValueError("handcrafted patches must be NHWC uint8 RGB")
        normalized = values.astype(np.float64) / 255.0
        mean = normalized.mean(axis=(1, 2))
        standard_deviation = normalized.std(axis=(1, 2))
        quartiles = np.quantile(normalized, (0.25, 0.75), axis=(1, 2)).transpose(1, 0, 2)
        result = np.column_stack((mean, standard_deviation, quartiles.reshape(len(values), -1)))
        return result.astype(np.float32)
