"""Independent algebra and weighting checks for retrospective residual diagnosis."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from test_feature_probes import GENES, sample

from heir.personalized.probe_workflow import _truth, score_probe
from heir.personalized.residual_diagnostics import (
    MOMENTS,
    _compare_scores,
    _derived,
    _historical_module_hash,
    _moments,
    _Sources,
    aggregate_residual_reports,
    residual_decomposition,
)


def test_energy_alignment_identity_uses_replicate_weights_and_local_centering():
    current = sample("p", n=12)
    current = replace(current, observation_groups=("small",) * 3 + ("large",) * 9)
    truth = _truth(current, GENES, GENES)[0].astype(float)
    error = np.arange(36, dtype=float).reshape(12, 3) / 10 - 2
    delta = np.sin(error) + np.array([0.2, -0.6, 1.5])
    baseline = truth - error
    result, replicate, genes = residual_decomposition(baseline, delta, current, GENES, GENES)
    expected_energy = (np.mean(delta[:3] ** 2) + np.mean(delta[3:] ** 2)) / 2
    expected_alignment = (np.mean(error[:3] * delta[:3]) + np.mean(error[3:] * delta[3:])) / 2
    expected_centered = []
    for x, d in ((error[:3], delta[:3]), (error[3:], delta[3:])):
        xc, dc = x - x.mean(0), d - d.mean(0)
        expected_centered.append((np.mean(dc**2), np.mean(xc * dc)))
    assert result["residual_energy"] == pytest.approx(expected_energy)
    assert result["error_alignment"] == pytest.approx(expected_alignment)
    assert result["centered_residual_energy"] == pytest.approx(
        np.mean(expected_centered, axis=0)[0]
    )
    assert result["centered_error_alignment"] == pytest.approx(
        np.mean(expected_centered, axis=0)[1]
    )
    assert result["mse_change"] == pytest.approx(expected_energy - 2 * expected_alignment)
    assert result["mse_change"] == pytest.approx(
        result["centered_mse_change"] + result["squared_bias_change"]
    )
    assert result["residual_energy"] != pytest.approx(np.mean(delta**2))
    assert sum(r["weight_in_specimen"] for r in genes) == pytest.approx(1)
    for base, prefix in ((baseline, "baseline"), (baseline + delta, "final")):
        score, _ = score_probe(base, current, GENES, GENES)
        assert result[prefix + "_mse"] == pytest.approx(score["normalized_log_mse"])
        assert result["centered_" + prefix + "_mse"] == pytest.approx(score["centered_mse"])
    assert len(replicate) == 2 and len(genes) == 6


@pytest.mark.parametrize(
    "error,delta,direction,alpha,shrink",
    [
        ([1, 2, 3], [0, 0, 0], "zero_residual", None, None),
        ([1, 2, 3], [-1, -2, -3], "noncorrective_direction", -1, 0),
        ([1, 2, 3], [3, 6, 9], "aligned_but_multiplier_one_too_large", 1 / 3, 1 / 3),
        ([1, 2, 3], [1, 2, 3], "beneficial_at_multiplier_one", 1, 1),
        ([1, 2, 3], [2, 4, 6], "break_even_at_multiplier_one", 0.5, 0.5),
    ],
)
def test_direction_and_target_informed_multiplier_are_analytic_diagnostics(
    error, delta, direction, alpha, shrink
):
    error, delta = np.asarray(error, float), np.asarray(delta, float)
    result = _derived(_moments(error, delta))
    assert result["direction"] == direction
    assert result["centered_direction"] == direction
    assert (
        result["diagnostic_optimal_multiplier"] == pytest.approx(alpha)
        if alpha is not None
        else result["diagnostic_optimal_multiplier"] is None
    )
    assert (
        result["diagnostic_shrink_multiplier"] == pytest.approx(shrink)
        if shrink is not None
        else result["diagnostic_shrink_multiplier"] is None
    )
    if alpha is not None:
        expected = np.mean((error - alpha * delta) ** 2 - error**2)
        assert result["diagnostic_optimal_mse_change"] == pytest.approx(expected)
        assert result["diagnostic_shrink_mse_change"] <= 1e-12


def test_constant_level_residual_has_zero_centered_effect():
    result = _derived(_moments(np.array([1.0, 3, 5]), np.full(3, 2.0)))
    assert result["direction"] == "beneficial_at_multiplier_one"
    assert result["centered_direction"] == "zero_residual"
    assert result["centered_mse_change"] == 0
    assert result["squared_bias_change"] == result["mse_change"]


def test_common_support_is_fixed_and_empty_replicate_remains_undefined():
    current = sample("p")
    truth = _truth(current, GENES, GENES)[0].astype(float)
    delta = np.ones_like(truth)
    selected = np.ones(len(truth), bool)
    selected[[1, 4, 7]] = False
    result, _, _ = residual_decomposition(truth, delta, current, GENES, GENES, row_mask=selected)
    assert result["n_observations"] == 9
    assert result["mse_change"] == 1
    selected[:6] = False
    missing, _, _ = residual_decomposition(truth, delta, current, GENES, GENES, row_mask=selected)
    assert missing["missing_coverage_replicate"]
    assert missing["mse_change"] is None and missing["direction"] == "undefined_support"
    with pytest.raises(ValueError, match="boolean"):
        residual_decomposition(truth, delta, current, GENES, GENES, row_mask=selected.astype(int))


def test_named_observation_mapping_and_direct_precision_are_preserved():
    current = sample("p")
    truth = _truth(current, GENES, GENES)[0].astype(float)
    original = truth.copy()
    delta = np.zeros_like(truth)
    shuffled = replace(
        current, observation=replace(current.observation, columns=np.arange(12)[::-1])
    )
    result, _, _ = residual_decomposition(truth[::-1].copy(), delta, shuffled, GENES, GENES)
    assert result["baseline_mse"] == 0 and result["final_mse"] == 0
    np.testing.assert_array_equal(truth, original)
    with pytest.raises(ValueError, match="float64"):
        residual_decomposition(truth.astype(np.float32), delta, current, GENES, GENES)


def test_repeated_fold_instances_do_not_weight_patients_or_specimens_twice():
    def record(patient, specimen, fold, energy):
        return {
            "patient_id": patient,
            "specimen_id": specimen,
            "outer_fold": fold,
            **_derived(_moments(np.zeros(2), np.full(2, np.sqrt(energy)))),
        }

    reports = [
        record("p1", "s1", 0, 1),
        record("p1", "s1", 1, 9),
        record("p1", "s2", 0, 3),
        record("p2", "s3", 0, 10),
    ]
    result = aggregate_residual_reports(reports)
    # p1: average the two s1 appearances ->5, then s1/s2 ->4; p2 ->10; cohort ->7.
    assert result["residual_energy"] == pytest.approx(7)
    assert result["mse_change"] == pytest.approx(7)
    assert result["n_fold_instances"] == 4 and result["n_patients"] == 2
    with pytest.raises(ValueError, match="unique"):
        aggregate_residual_reports(reports + [reports[0]])
    altered = [dict(row) for row in reports]
    altered[0].update({key: None for key in MOMENTS})
    assert aggregate_residual_reports(altered)["mse_change"] is None


def test_source_ledger_rejects_changed_files_and_escaping_inventory(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"value":1}')
    ledger = _Sources()
    assert ledger.json(path) == {"value": 1}
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="source identity changed"):
        ledger.bind(path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "sha256.json").write_text('{"../source.json":"a"}')
    with pytest.raises(ValueError, match="escapes"):
        _Sources().inventory(bundle)


@pytest.mark.parametrize(
    "original",
    (
        "/old/checkout/src/heir/personalized/residual_model.py",
        "/old/site-packages/heir/personalized/residual_model.py",
        "src/heir/personalized/residual_model.py",
        "heir/personalized/residual_model.py",
    ),
)
def test_relocated_module_preserves_exact_source_hash(tmp_path, original):
    path = tmp_path / "new/checkout/src/heir/personalized/residual_model.py"
    path.parent.mkdir(parents=True)
    content = b"# byte-identical numerical module\n"
    path.write_bytes(content)
    hashes = {original: sha256(content).hexdigest()}
    expected = _historical_module_hash(path, hashes)
    assert _Sources().bind(path, expected) == path.resolve()
    path.write_bytes(content + b"# changed numerical source\n")
    with pytest.raises(ValueError, match="source identity changed"):
        _Sources().bind(path, _historical_module_hash(path, hashes))


def test_historical_module_identity_rejects_missing_ambiguous_or_invalid_hashes():
    current = "/current/src/heir/personalized/residual_model.py"
    original = "/old/src/heir/personalized/residual_model.py"
    digest = "a" * 64
    for hashes in (
        {},
        {"/unrelated/residual_model.py": digest},
        {"src/other/personalized/residual_model.py": digest},
        {original: digest, current: digest},
        {original: digest, "heir/personalized/residual_model.py": "b" * 64},
    ):
        with pytest.raises(ValueError, match="exactly one source"):
            _historical_module_hash(current, hashes)
    for invalid in (None, "", "a" * 63, "z" * 64):
        with pytest.raises(ValueError, match="requires a SHA-256"):
            _historical_module_hash(current, {original: invalid})


def test_score_replay_requires_same_definedness_and_values():
    _compare_scores({"m": None}, {"m": None}, ("m",))
    with pytest.raises(ValueError, match="saved residual scores"):
        _compare_scores({"m": None}, {"m": 0}, ("m",))
    with pytest.raises(ValueError, match="saved residual scores"):
        _compare_scores({"m": 1}, {"m": 1.01}, ("m",))


def test_saved_study_replay_is_read_only_and_checks_selected_validation(tmp_path, monkeypatch):
    """Real tiny bundles/exports exercise diagnostics; no optimizer is permitted during replay."""
    import json
    import shutil
    from pathlib import Path

    import test_residual_workflow as fixture

    from heir.personalized import crossfit, residual_train, residual_workflow
    from heir.personalized.probe_workflow import FrozenFeatureProbe
    from heir.personalized.residual_diagnostics import diagnose_residual_study

    make_folds = fixture.make_patient_folds
    monkeypatch.setattr(
        fixture, "make_patient_folds", lambda ids: make_folds(ids, validation_size=1)
    )
    config, protocol, _ = fixture._fixture(tmp_path, monkeypatch)
    # The orchestration fixture's middle coefficient is zero; use evaluable baseline predictions.
    for fold in range(5):
        bundle = Path(protocol["baseline_directory"]) / f"fold-{fold}/probe.bundle"
        baseline = FrozenFeatureProbe.load(bundle)
        baseline.coefficients[0, 4] = 0.025
        shutil.rmtree(bundle)
        baseline.save(bundle)
        protocol["baseline_folds"][str(fold)]["bundle_inventory_sha256"] = fixture.workflow._sha(
            bundle / "sha256.json"
        )
    config.write_text(json.dumps(protocol))
    original_fit, original_validation = residual_train.fit_residual, residual_train._validation

    def baseline_validation(fitted, samples):
        stage = fitted.selection
        fitted.selection = {"residual_enabled": False}
        result = original_validation(fitted, samples)
        fitted.selection = stage
        return result

    def tiny_fit(*args, **kwargs):
        kwargs["epochs"] = 1
        fitted = original_fit(*args, **kwargs)
        # Fixed-length history is a protocol fixture, not a claim of 100 test optimization steps.
        fitted.history = [{**fitted.history[0], "epoch": i} for i in range(101)]
        assert fitted.selection["selected_epoch"] == 0
        return fitted

    monkeypatch.setattr(residual_train, "fit_residual", tiny_fit)
    monkeypatch.setattr(residual_train, "_validation", baseline_validation)
    monkeypatch.setattr(
        crossfit,
        "verify_deployment_probe",
        lambda path, *a, **k: (FrozenFeatureProbe.load(path), {}),
    )
    study = tmp_path / "residual-study"
    result = residual_workflow.run_residual_study(config, study)
    # The real study contract has thirteen test specimens; expand no data, use the fixture count.
    assert result["export_counts"] == {"baseline": 5, "final": 30, "transfer": 20}
    # Relocate only the recorded checkout root; numerical bytes and data stay identical.
    identity_path = study / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity["source_file_sha256"] = {
        str(tmp_path / "original-checkout/src/heir/personalized" / Path(path).name): digest
        for path, digest in identity["source_file_sha256"].items()
    }
    identity_path.write_text(json.dumps(identity))

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic replay must not fit or solve")

    monkeypatch.setattr(residual_train, "fit_residual", forbidden)
    monkeypatch.setattr(crossfit, "crossfit_feature_probe", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    output = tmp_path / "diagnostics"
    # Counts are derived from frozen geometry/folds; the fixture intentionally has five patients.
    diagnosed = diagnose_residual_study(study, output)
    assert diagnosed["model_fits"] == 0
    assert diagnosed["exact_test_prediction_replays"] == 30
    assert diagnosed["source_roles"] == {"train": 15, "validation": 5, "test": 5}
    assert diagnosed["selected_validation_and_zero_candidate_replays"] == 30
    assert all(row["mse_change"] == 0 for row in diagnosed["summaries"])
    assert all(row["direction"] == "zero_residual" for row in diagnosed["summaries"])
    assert not list(output.glob("*.npz"))
    with pytest.raises(FileExistsError):
        diagnose_residual_study(study, output)
    selected = study / "fold-0/H.bundle/bundle.json"
    record = json.loads(selected.read_text())
    record["selection"]["validation"]["normalized_log_mse"] += 1
    selected.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="source identity changed|integrity"):
        diagnose_residual_study(study, tmp_path / "tampered")
