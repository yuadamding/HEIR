"""Published aggregates are independently checkable without workstation arrays."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

PUBLICATION = Path(__file__).resolve().parents[1] / "reproducibility/v013"


@pytest.fixture(scope="module")
def audit_script():
    specification = importlib.util.spec_from_file_location(
        "independent_residual_audit", PUBLICATION / "audit_residual_moments.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_published_aggregate_algebra_without_original_arrays(audit_script):
    provenance = json.loads((PUBLICATION / "provenance.json").read_text())
    assert (
        hashlib.sha256((PUBLICATION / "audit_residual_moments.py").read_bytes()).hexdigest()
        == (provenance["reconstruction_procedure"]["sha256"])
    )
    result = audit_script.verify_aggregates(PUBLICATION)
    assert result["status"] == "passed"
    assert result["summary_groups"] == 48
    assert result["maximum_absolute_discrepancy"] < 1e-10
    assert "aggregate algebra only" in result["scope"]


@pytest.mark.parametrize("rebind_table_hash", (False, True))
def test_changed_aggregate_is_rejected(audit_script, tmp_path, rebind_table_hash):
    for name in ("aggregates.tsv", "provenance.json"):
        shutil.copyfile(PUBLICATION / name, tmp_path / name)
    table = tmp_path / "aggregates.tsv"
    rows = table.read_text().splitlines()
    column = rows[0].split("\t").index("mse_change")
    values = rows[1].split("\t")
    values[column] = str(float(values[column]) + 1)
    rows[1] = "\t".join(values)
    table.write_text("\n".join(rows) + "\n")
    if rebind_table_hash:
        provenance = tmp_path / "provenance.json"
        metadata = json.loads(provenance.read_text())
        metadata["aggregate_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
        provenance.write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match="numeric|source digest"):
        audit_script.verify_aggregates(tmp_path)
