import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from heir.evaluation import evaluate_oracle_ladder

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_oracle_ladder.py"
SPEC = importlib.util.spec_from_file_location("benchmark_oracle_ladder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _oracle_inputs() -> dict[str, object]:
    truth_latent = np.asarray([[0.0], [0.4], [2.0], [2.4]])
    true_types = np.asarray([0, 0, 1, 1])
    prototype_means = np.asarray([[0.0], [0.4], [2.0], [2.4]])
    prototype_types = np.asarray([0, 0, 1, 1])
    prototype_expression = np.asarray(
        [[0.0, 1.0, 0.0], [0.4, 1.2, 0.0], [2.0, 3.0, 0.0], [2.4, 3.2, 0.0]]
    )
    truth_expression = np.asarray(
        [[0.0, 1.0, 0.0], [0.4, 1.2, 0.0], [2.0, 3.0, 0.0], [2.4, 3.2, 0.0]]
    )
    type_means = np.asarray([[0.2, 1.1, 0.0], [2.2, 3.1, 0.0]])
    type_probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
    return {
        "truth_expression": truth_expression,
        "truth_latent": truth_latent,
        "true_types": true_types,
        "decoder_expression": truth_expression + 0.05,
        "type_mean_expression": type_means,
        "prototype_means": prototype_means,
        "prototype_expression": prototype_expression,
        "prototype_types": prototype_types,
        "predicted_type_probabilities": type_probabilities,
        "oracle_type_conditioned_heir_expression": truth_expression,
        "residual_disabled_heir_expression": type_means[type_probabilities.argmax(axis=1)],
        "full_heir_expression": type_means[type_probabilities.argmax(axis=1)],
        "cell_ids": np.asarray(["c1", "c2", "c3", "c4"]),
        "gene_names": np.asarray(["g1", "g2", "constant"]),
        "spot_ids": np.asarray(["s1", "s1", "s2", "s2"]),
        "cell_rna_mass": np.asarray([1.0, 3.0, 2.0, 1.0]),
        "input_artifact_sha256": "a" * 64,
        "decoder_checkpoint_sha256": "b" * 64,
        "heir_checkpoint_sha256": "c" * 64,
    }


def test_oracle_ladder_uses_one_truth_mask_and_separates_type_from_state() -> None:
    report = evaluate_oracle_ladder(**_oracle_inputs())

    assert report["genes_total"] == 3
    assert report["genes_evaluated"] == 2
    endpoints = report["endpoints"]
    assert report["spots"] == 2
    assert report["schema"] == "heir.oracle_ladder.v5"
    assert "full_heir_residual_disabled" in endpoints
    assert (
        endpoints["full_heir_residual_disabled"]["control_semantics"]["learned_morphology_residual"]
        == "disabled_during_same_checkpoint_forward"
    )
    assert (
        endpoints["full_heir_residual_disabled"]["control_semantics"]["construction"]
        == "precomputed_exact_model_output_with_residual_branch_forced_off"
    )
    masks = {value["truth_gene_mask_sha256"] for value in endpoints.values()}
    assert len(masks) == 1
    assert endpoints["rna_decoder_ceiling"]["metrics"]["cell_gene_mse"] > 0.0
    assert endpoints["oracle_type_oracle_prototype"]["metrics"]["cell_gene_mse"] == 0.0
    assert endpoints["predicted_type_oracle_state"]["metrics"]["cell_gene_mse"] > 0.0
    assert endpoints["oracle_type_predicted_state"]["metrics"]["cell_gene_mse"] == 0.0
    assert (
        endpoints["oracle_type_predicted_state"]["control_semantics"]["construction"]
        == "precomputed_exact_same_checkpoint_oracle_type_conditioned_forward"
    )
    assert (
        endpoints["oracle_type_predicted_state"]["control_semantics"]["broad_type"]
        == "oracle_true_type_forced_during_same_checkpoint_forward"
    )
    assert endpoints["rna_decoder_ceiling"]["decoder_checkpoint_sha256"] == "b" * 64
    assert endpoints["full_heir_residual_disabled"]["heir_checkpoint_sha256"] == "c" * 64
    assert report["provenance"]["input_artifact_sha256"] == "a" * 64
    assert len(report["provenance"]["oracle_input_bundle_sha256"]) == 64
    assert set(report["provenance"]["array_sha256"]) == set(MODULE.FIELDS)
    assert all("spot_metrics" in endpoint for endpoint in endpoints.values())
    assert all("pseudobulk_metrics" in endpoint for endpoint in endpoints.values())
    assert all(len(endpoint["spot_prediction_sha256"]) == 64 for endpoint in endpoints.values())


def test_oracle_spatial_metrics_use_frozen_rna_mass() -> None:
    arguments = _oracle_inputs()
    original = evaluate_oracle_ladder(**arguments)
    changed = evaluate_oracle_ladder(
        **{**arguments, "cell_rna_mass": np.asarray([3.0, 1.0, 2.0, 1.0])}
    )

    assert (
        original["spatial_aggregation"]["cell_rna_mass_sha256"]
        != changed["spatial_aggregation"]["cell_rna_mass_sha256"]
    )
    assert (
        original["endpoints"]["full_heir"]["spot_prediction_sha256"]
        != changed["endpoints"]["full_heir"]["spot_prediction_sha256"]
    )


def test_oracle_residual_disabled_endpoint_uses_explicit_fixture_not_profile_mixture() -> None:
    arguments = _oracle_inputs()
    explicit = np.asarray(arguments["residual_disabled_heir_expression"]).copy()
    explicit[:, 0] += np.asarray([0.0, 0.2, 0.4, 0.6])
    arguments["residual_disabled_heir_expression"] = explicit
    report = evaluate_oracle_ladder(**arguments)

    endpoint = report["endpoints"]["full_heir_residual_disabled"]
    assert (
        endpoint["prediction_sha256"]
        == report["provenance"]["array_sha256"]["residual_disabled_heir_expression"]
    )
    assert (
        endpoint["prediction_sha256"]
        != report["endpoints"]["oracle_type_predicted_state"]["prediction_sha256"]
    )


def test_oracle_type_predicted_state_uses_explicit_same_checkpoint_fixture() -> None:
    arguments = _oracle_inputs()
    explicit = np.asarray(arguments["oracle_type_conditioned_heir_expression"]).copy()
    explicit[:, 1] += np.asarray([0.0, 0.1, 0.2, 0.3])
    arguments["oracle_type_conditioned_heir_expression"] = explicit
    report = evaluate_oracle_ladder(**arguments)

    endpoint = report["endpoints"]["oracle_type_predicted_state"]
    assert (
        endpoint["prediction_sha256"]
        == report["provenance"]["array_sha256"]["oracle_type_conditioned_heir_expression"]
    )
    assert endpoint["heir_checkpoint_sha256"] == arguments["heir_checkpoint_sha256"]
    assert endpoint["control_semantics"]["not_reconstructed_by_evaluator"] is True


def test_oracle_constant_prediction_correlations_are_zero_on_common_mask() -> None:
    arguments = _oracle_inputs()
    arguments["full_heir_expression"] = np.full((4, 3), 0.5)
    report = evaluate_oracle_ladder(**arguments)

    metrics = report["endpoints"]["full_heir"]["metrics"]
    assert metrics["per_gene_pearson"] == pytest.approx([0.0, 0.0])
    assert metrics["per_gene_spearman"] == pytest.approx([0.0, 0.0])
    assert metrics["median_gene_pearson"] == pytest.approx(0.0)
    assert metrics["median_gene_spearman"] == pytest.approx(0.0)
    assert metrics["constant_prediction_count"] == 2
    assert metrics["fraction_genes_defined"] == pytest.approx(1.0)
    common_mask = report["truth_gene_mask_sha256"]
    assert all(
        endpoint["truth_gene_mask_sha256"] == common_mask
        for endpoint in report["endpoints"].values()
    )


def test_oracle_provenance_binds_ordered_identities_and_every_array() -> None:
    original_arguments = _oracle_inputs()
    original = evaluate_oracle_ladder(**original_arguments)
    reordered_arguments = _oracle_inputs()
    reordered_arguments["cell_ids"] = np.asarray(["c2", "c1", "c3", "c4"])
    reordered = evaluate_oracle_ladder(**reordered_arguments)

    assert (
        original["provenance"]["ordered_cell_ids_sha256"]
        != reordered["provenance"]["ordered_cell_ids_sha256"]
    )
    assert (
        original["provenance"]["oracle_input_bundle_sha256"]
        != reordered["provenance"]["oracle_input_bundle_sha256"]
    )
    assert original["truth_gene_mask_sha256"] != reordered["truth_gene_mask_sha256"]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        evaluate_oracle_ladder(
            **{**original_arguments, "decoder_checkpoint_sha256": "not-a-checkpoint-hash"}
        )


def test_oracle_script_hashes_npz_and_requires_physical_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arguments = _oracle_inputs()
    input_path = tmp_path / "oracle_inputs.npz"
    checkpoint_path = tmp_path / "decoder.pt"
    heir_checkpoint_path = tmp_path / "heir.pt"
    output_path = tmp_path / "oracle_report.json"
    checkpoint_path.write_bytes(b"frozen decoder checkpoint")
    heir_checkpoint_path.write_bytes(b"frozen heir checkpoint")
    checkpoint_sha256 = MODULE._file_sha256(checkpoint_path)
    heir_checkpoint_sha256 = MODULE._file_sha256(heir_checkpoint_path)
    np.savez_compressed(
        input_path,
        **{name: arguments[name] for name in (*MODULE.FIELDS, *MODULE.IDENTITY_FIELDS)},
        decoder_checkpoint_sha256=np.asarray(checkpoint_sha256),
        heir_checkpoint_sha256=np.asarray(heir_checkpoint_sha256),
    )

    assert (
        MODULE.main(
            [
                "--input",
                str(input_path),
                "--decoder-checkpoint",
                str(checkpoint_path),
                "--heir-checkpoint",
                str(heir_checkpoint_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["provenance"]["input_artifact_sha256"] == MODULE._file_sha256(input_path)
    assert report["provenance"]["decoder_checkpoint_sha256"] == MODULE._file_sha256(checkpoint_path)
    assert report["provenance"]["heir_checkpoint_sha256"] == MODULE._file_sha256(
        heir_checkpoint_path
    )
    assert report["endpoints"]["rna_decoder_ceiling"][
        "decoder_checkpoint_sha256"
    ] == MODULE._file_sha256(checkpoint_path)
    assert report["endpoints"]["rna_decoder_ceiling"]["checkpoint_file_verification"] is True
    assert "matched" in report["provenance"]["checkpoint_file_verification"]["decoder"]
    assert "matched" in report["provenance"]["checkpoint_file_verification"]["heir"]
    assert set(report["provenance"]["source_sha256"]) == {
        "scripts.benchmark_oracle_ladder",
        "heir.evaluation.oracle",
    }
    assert report["claim_scope"]["eligible_for_primary_performance_claims"] is False

    input_bytes = input_path.read_bytes()
    with pytest.raises(ValueError, match="output would overwrite a bound input"):
        MODULE.main(
            [
                "--input",
                str(input_path),
                "--decoder-checkpoint",
                str(checkpoint_path),
                "--heir-checkpoint",
                str(heir_checkpoint_path),
                "--output",
                str(input_path),
            ]
        )
    assert input_path.read_bytes() == input_bytes

    late_alias_output = tmp_path / "late-alias-output.json"
    original_evaluate = MODULE.evaluate_oracle_ladder

    def create_late_output_alias(**kwargs):
        report = original_evaluate(**kwargs)
        late_alias_output.hardlink_to(input_path)
        return report

    with monkeypatch.context() as scoped:
        scoped.setattr(MODULE, "evaluate_oracle_ladder", create_late_output_alias)
        with pytest.raises(ValueError, match="output would overwrite a bound input"):
            MODULE.main(
                [
                    "--input",
                    str(input_path),
                    "--decoder-checkpoint",
                    str(checkpoint_path),
                    "--heir-checkpoint",
                    str(heir_checkpoint_path),
                    "--output",
                    str(late_alias_output),
                ]
            )
    assert input_path.read_bytes() == input_bytes

    checkpoint_path.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="differs from the supplied checkpoint"):
        MODULE.main(
            [
                "--input",
                str(input_path),
                "--decoder-checkpoint",
                str(checkpoint_path),
                "--heir-checkpoint",
                str(heir_checkpoint_path),
                "--output",
                str(output_path),
            ]
        )
