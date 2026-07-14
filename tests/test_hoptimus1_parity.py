from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _script():
    path = Path(__file__).parents[1] / "scripts" / "qualify_hoptimus1_parity.py"
    spec = importlib.util.spec_from_file_location("qualify_hoptimus1_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARITY = _script()


def _candidate(
    index: int,
    *,
    tissue: float,
    brightness: float,
    contrast: float,
    hematoxylin: float,
    dark: float,
):
    return PARITY.Candidate(
        x=index,
        y=index + 100,
        tissue_fraction=tissue,
        mean_luminance=brightness,
        luminance_std=contrast,
        hematoxylin_p90=hematoxylin,
        dark_fraction=dark,
    )


def test_frozen_contract_matches_available_hash_bound_manifest() -> None:
    manifest = PARITY.load_encoder_manifest(PARITY.DEFAULT_MANIFEST)
    assert manifest.sha256 == PARITY.MANIFEST_SHA256
    assert manifest.revision == PARITY.REVISION
    assert manifest.checkpoint_sha256 == PARITY.CHECKPOINT_SHA256
    assert manifest.config_sha256 == PARITY.CONFIG_SHA256
    assert PARITY.sha256_file(PARITY.DEFAULT_MODEL_DIR / "README.md") == PARITY.README_SHA256
    assert manifest.feature_width == 1536
    assert manifest.input_pixels == 224
    assert manifest.model_mpp == 0.5
    assert manifest.fine_tuning == "prohibited"


def test_patch_regimes_are_selected_deterministically_and_without_reuse() -> None:
    candidates = (
        _candidate(0, tissue=0.01, brightness=0.99, contrast=0.01, hematoxylin=0.01, dark=0),
        _candidate(1, tissue=0.70, brightness=0.40, contrast=0.22, hematoxylin=2.1, dark=0.4),
        _candidate(2, tissue=0.20, brightness=0.78, contrast=0.02, hematoxylin=0.4, dark=0),
        _candidate(3, tissue=0.95, brightness=0.60, contrast=0.30, hematoxylin=1.1, dark=0.2),
    )
    first = PARITY._select_candidates(candidates)
    second = PARITY._select_candidates(tuple(reversed(candidates)))
    assert {name: value.x for name, value in first.items()} == {
        "natural_hest_tissue": 3,
        "mostly_background": 0,
        "dark_hematoxylin_rich": 1,
        "low_contrast_tissue": 2,
    }
    assert {name: value.x for name, value in first.items()} == {
        name: value.x for name, value in second.items()
    }
    assert len({(value.x, value.y) for value in first.values()}) == 4


def test_physical_crop_is_resampled_once_to_exact_model_canvas(monkeypatch) -> None:
    native_pixels = PARITY._native_crop_pixels()
    assert native_pixels == round(112.0 / 0.2125)
    source = np.full((native_pixels, native_pixels, 3), 173, dtype=np.uint8)
    output = PARITY._resample_once(source)
    assert output.shape == (224, 224, 3)
    assert output.dtype == np.uint8

    class FakeReader:
        width = 10_000
        height = 10_000

        def read_with_padding(self, center, size):
            patch = np.full((size, size, 3), (center[0] + center[1]) % 255, dtype=np.uint8)
            return patch, 0.75 if center == (0, 0) else 0.0

    selected = {
        category: _candidate(
            index + 1000,
            tissue=0.5,
            brightness=0.5,
            contrast=0.1,
            hematoxylin=0.5,
            dark=0.1,
        )
        for index, category in enumerate(PARITY.PATCH_CATEGORIES[:-1])
    }
    monkeypatch.setattr(PARITY, "_scan_candidates", lambda _reader, _grid: tuple(selected.values()))
    monkeypatch.setattr(PARITY, "_select_candidates", lambda _candidates: selected)
    calls = []

    def resize_once(patch):
        calls.append(patch.shape)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    patches, receipts = PARITY._build_patch_suite(FakeReader(), 4, resize=resize_once)
    assert patches.shape == (5, 224, 224, 3)
    assert calls == [(native_pixels, native_pixels, 3)] * 5
    assert all(receipt.resampling_count == 1 for receipt in receipts)
    assert receipts[-1].category == "padded_border"
    assert receipts[-1].padding_fraction == 0.75


def test_parity_metrics_enforce_shape_finiteness_and_cosine_thresholds() -> None:
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(5, 1536)).astype(np.float32)
    PARITY._validate_embedding_matrix(reference, 5)
    same = PARITY._comparison(reference, reference.copy())
    assert same["minimum_cosine"] == pytest.approx(1.0)
    assert same["mean_absolute_error"] == 0.0
    assert same["maximum_absolute_error"] == 0.0
    with pytest.raises(ValueError, match="N x 1536"):
        PARITY._validate_embedding_matrix(reference[:, :-1], 5)
    broken = reference.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PARITY._validate_embedding_matrix(broken, 5)


def test_giant_models_are_loaded_and_released_sequentially(monkeypatch) -> None:
    state = {"live": 0, "maximum": 0, "events": []}

    class FakeCuda:
        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 123

        @staticmethod
        def is_available():
            return False

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))

    def load(name):
        def loader():
            state["live"] += 1
            state["maximum"] = max(state["maximum"], state["live"])
            state["events"].append(f"load:{name}")
            return name

        return loader

    def encode(model, precision):
        state["events"].append(f"encode:{model}:{precision}")
        return np.ones((5, 1536), dtype=np.float32)

    def release(model):
        state["events"].append(f"release:{model}")
        state["live"] -= 1

    result = PARITY._run_sequential_inference(
        np.zeros((5, 224, 224, 3), dtype=np.uint8),
        load("official"),
        load("local"),
        encode,
        release,
    )
    assert state["maximum"] == 1
    assert state["live"] == 0
    assert state["events"] == [
        "load:official",
        "encode:official:fp32",
        "release:official",
        "load:local",
        "encode:local:fp32",
        "encode:local:fp16",
        "release:local",
    ]
    assert result[0].shape == (5, 1536)
    assert result[3] == {
        "official_fp32_cuda_peak_bytes": 123,
        "local_fp32_fp16_cuda_peak_bytes": 123,
    }


def test_markdown_receipt_is_metrics_only() -> None:
    patch = {
        "category": "natural_hest_tissue",
        "padding_fraction": 0.0,
        "tissue_fraction": 0.8,
        "input_rgb_sha256": "a" * 64,
    }
    comparison = {
        "minimum_cosine": 1.0,
        "minimum_required_cosine": 0.999999,
        "mean_absolute_error": 0.0,
        "maximum_absolute_error": 0.0,
        "passed": True,
    }
    receipt = {
        "status": "passed",
        "model": {
            "repository": PARITY.REPOSITORY,
            "revision": PARITY.REVISION,
            "manifest_sha256": PARITY.MANIFEST_SHA256,
            "config_sha256": PARITY.CONFIG_SHA256,
            "checkpoint_sha256": PARITY.CHECKPOINT_SHA256,
            "official_readme_sha256": PARITY.README_SHA256,
        },
        "hest_patch_suite": {"patches": [patch]},
        "comparisons": {
            "official_fp32_vs_local_fp32": comparison,
            "local_fp32_vs_local_fp16": {**comparison, "minimum_required_cosine": 0.9999},
        },
        "resources": {"batch_size": 1},
        "production_runtime_contract": {
            "code_sha256": {"src/heir/features/hoptimus1.py": "b" * 64},
            "resampling_probe": {"output_sha256": "c" * 64},
        },
    }
    markdown = PARITY._markdown(receipt)
    assert "implementation only" in markdown
    assert "at most one giant model" in markdown
    assert "model.safetensors" not in markdown
    assert "tensor(" not in markdown
