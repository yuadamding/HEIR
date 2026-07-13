"""Regression tests for standalone benchmark output/input collision guards."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_test_" + name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _alias(source: Path, destination: Path, kind: str) -> None:
    if kind == "hardlink":
        destination.hardlink_to(source)
    else:
        destination.symlink_to(source)


@pytest.mark.parametrize("alias_kind", ("hardlink", "symlink"))
def test_snpatho_cli_rejects_bound_input_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    script = _load_script("benchmark_snpatho")
    source = tmp_path / "gene-panel.tsv"
    source.write_text("g1\n", encoding="utf-8")
    output = tmp_path / "report.json"
    _alias(source, output, alias_kind)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = SimpleNamespace(source_path=plan_path, gene_panel=source, cases=())
    monkeypatch.setattr(script, "load_snpatho_plan", lambda _: plan)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script.__file__), "--plan", str(plan_path), "--output", str(output)],
    )

    with pytest.raises(ValueError, match="would overwrite a bound input"):
        script.main()
    assert source.read_text(encoding="utf-8") == "g1\n"


@pytest.mark.parametrize("alias_kind", ("hardlink", "symlink"))
def test_deepbench_cli_rejects_bound_input_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    script = _load_script("benchmark_snpatho_deepbench")
    historical = tmp_path / "historical.json"
    historical.write_text("{}", encoding="utf-8")
    output = tmp_path / "report.json"
    _alias(historical, output, alias_kind)
    source_plan = tmp_path / "deepbench.yaml"
    frozen_plan = tmp_path / "frozen.json"
    source_plan.write_text("name: test\n", encoding="utf-8")
    frozen_plan.write_text("{}", encoding="utf-8")
    plan = SimpleNamespace(
        source_path=source_plan,
        frozen_plan=frozen_plan,
        historical_report=historical,
        optional_artifacts={},
    )
    locked = SimpleNamespace(gene_panel=tmp_path / "panel.tsv", cases=())
    monkeypatch.setattr(script, "load_deepbench_plan", lambda _: plan)
    monkeypatch.setattr(script, "load_snpatho_plan", lambda _: locked)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script.__file__), "--plan", str(source_plan), "--output", str(output)],
    )

    with pytest.raises(ValueError, match="would overwrite a bound input"):
        script.main()
    assert historical.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("alias_kind", ("hardlink", "symlink"))
def test_deepbench_cli_rejects_transitive_optional_manifest_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    script = _load_script("benchmark_snpatho_deepbench")
    latent_reference = tmp_path / "latent-reference.npz"
    latent_reference.write_bytes(b"frozen-reference")
    optional_manifest = tmp_path / "native-manifest.json"
    optional_manifest.write_text(
        json.dumps({"specimens": {"sample": {"latent_reference": latent_reference.name}}}),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    _alias(latent_reference, output, alias_kind)
    source_plan = tmp_path / "deepbench.yaml"
    frozen_plan = tmp_path / "frozen.json"
    historical = tmp_path / "historical.json"
    source_plan.write_text("name: test\n", encoding="utf-8")
    frozen_plan.write_text("{}", encoding="utf-8")
    historical.write_text("{}", encoding="utf-8")
    plan = SimpleNamespace(
        source_path=source_plan,
        frozen_plan=frozen_plan,
        historical_report=historical,
        optional_artifacts={"native_scanvi_checkpoint": optional_manifest},
    )
    locked = SimpleNamespace(gene_panel=tmp_path / "panel.tsv", cases=())
    monkeypatch.setattr(script, "load_deepbench_plan", lambda _: plan)
    monkeypatch.setattr(script, "load_snpatho_plan", lambda _: locked)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script.__file__), "--plan", str(source_plan), "--output", str(output)],
    )

    with pytest.raises(ValueError, match="would overwrite a bound input"):
        script.main()
    assert latent_reference.read_bytes() == b"frozen-reference"


@pytest.mark.parametrize("alias_kind", ("hardlink", "symlink"))
def test_broad_type_cli_rejects_declared_artifact_alias(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    script = _load_script("benchmark_broad_types")
    labels = tmp_path / "labels.tsv"
    labels.write_text("nucleus_id\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"tasks": [{"datasets": [{"labels": {"path": labels.name}}]}]}),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    _alias(labels, output, alias_kind)

    with pytest.raises(ValueError, match="would overwrite a bound input"):
        script.main(["--plan", str(plan), "--output", str(output), "--inspect"])
    assert labels.read_text(encoding="utf-8") == "nucleus_id\n"
