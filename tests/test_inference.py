from dataclasses import replace

import numpy as np
import pytest
import torch

from heir.data import PrototypeSet
from heir.inference import PredictionBundle, predict_cells
from heir.models.heir import HEIRConfig, HEIRModel


def _bundle() -> PredictionBundle:
    return PredictionBundle(
        nucleus_ids=np.asarray(["n1", "n2"]),
        coordinates_um=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        type_probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        type_names=np.asarray(["A", "B"]),
        labels=np.asarray([0, 1], dtype=np.int64),
        prototype_probabilities=np.asarray([[0.7, 0.1], [0.1, 0.8]], dtype=np.float32),
        prototype_ids=np.asarray(["pA", "pB"]),
        latent_mean=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        latent_variance=np.full((2, 2), 0.2, dtype=np.float32),
        expression_mean=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        expression_lower=np.asarray([[0.8, 1.8], [2.8, 3.8]], dtype=np.float32),
        expression_upper=np.asarray([[1.2, 2.2], [3.2, 4.2]], dtype=np.float32),
        gene_names=np.asarray(["g1", "g2"]),
        unknown_probability=np.asarray([0.2, 0.1], dtype=np.float32),
        abstain_score=np.asarray([0.3, 0.2], dtype=np.float32),
        abstain=np.asarray([False, False]),
        ood_score=np.asarray([0.0, 0.1], dtype=np.float32),
        refinement_round=2,
        sample_id="sample1",
        donor_id="donor1",
        slide_id="slide1",
        checkpoint_sha256="a" * 64,
        prototype_sha256="b" * 64,
        histology_sha256="c" * 64,
        latent_space_id="latent-v1",
        model_version="0.1.0",
        ood_sha256="e" * 64,
        ood_training_donors=np.asarray(["ood-training-donor"]),
        inference_seed=23,
        latent_samples=17,
        probability_threshold=0.7,
        artifact_threshold=0.4,
        expression_space_id="log1p-cpm10k-v1",
        parent_type_probabilities=np.ones((2, 1), dtype=np.float32),
        parent_type_names=np.asarray(["parent"]),
        program_scores=np.asarray([[1.0], [2.0]], dtype=np.float32),
        program_names=np.asarray(["program1"]),
        program_sha256="d" * 64,
        program_training_donors=np.asarray(["training-donor"]),
    )


def test_prediction_bundle_v9_roundtrip_and_legacy_load(tmp_path):
    bundle = _bundle()
    output = tmp_path / "prediction_v9.npz"
    bundle.to_npz(output)
    loaded = PredictionBundle.from_npz(output)
    assert loaded.sample_id == "sample1"
    assert loaded.checkpoint_sha256 == "a" * 64
    assert loaded.ood_sha256 == "e" * 64
    assert loaded.ood_training_donors.tolist() == ["ood-training-donor"]
    assert loaded.inference_seed == 23
    assert loaded.latent_samples == 17
    assert loaded.probability_threshold == pytest.approx(0.7)
    assert loaded.artifact_threshold == pytest.approx(0.4)
    assert loaded.expression_space_id == "log1p-cpm10k-v1"
    assert loaded.expression_interval_semantics == PredictionBundle.CONDITIONAL_KNOWN_STATE
    np.testing.assert_array_equal(loaded.expression_mean_available, [True, True])
    np.testing.assert_array_equal(loaded.expression_interval_available, [True, True])
    assert loaded.parent_type_names.tolist() == ["parent"]
    assert loaded.program_scores.shape == (2, 1)
    np.testing.assert_array_equal(
        loaded.transport_unassigned_probability,
        loaded.unknown_probability,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["__contract__"].item()) == PredictionBundle.CONTRACT
        assert int(archive["__version__"].item()) == 9
        np.testing.assert_array_equal(
            archive["transport_unassigned_probability"],
            archive["unknown_probability"],
        )
        legacy_payload = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if not name.startswith("__")
            and name
            not in {
                "sample_id",
                "donor_id",
                "slide_id",
                "checkpoint_sha256",
                "prototype_sha256",
                "histology_sha256",
                "latent_space_id",
                "model_version",
                "ood_sha256",
                "ood_training_donors",
                "inference_seed",
                "latent_samples",
                "probability_threshold",
                "artifact_threshold",
                "expression_space_id",
                "expression_interval_semantics",
                "expression_mean_available",
                "expression_interval_available",
                "parent_type_probabilities",
                "parent_type_names",
                "program_scores",
                "program_names",
                "program_sha256",
                "program_training_donors",
            }
        }
    legacy = tmp_path / "legacy_prediction.npz"
    np.savez_compressed(legacy, **legacy_payload)
    migrated = PredictionBundle.from_npz(legacy)
    assert migrated.sample_id == ""
    assert migrated.checkpoint_sha256 == ""
    assert migrated.parent_type_probabilities is None
    assert migrated.program_scores is None
    assert migrated.ood_sha256 == ""
    assert migrated.ood_training_donors is None
    assert migrated.inference_seed is None
    assert migrated.latent_samples is None
    assert migrated.probability_threshold is None
    assert migrated.artifact_threshold is None
    assert migrated.expression_space_id == ""
    assert migrated.expression_interval_semantics == PredictionBundle.LEGACY_CONDITIONAL_KNOWN_STATE
    np.testing.assert_array_equal(migrated.expression_mean_available, [True, True])
    assert migrated.expression_interval_available is None

    with np.load(output, allow_pickle=False) as archive:
        v7_payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    v7_payload["__version__"] = np.asarray(7, dtype=np.int64)
    del v7_payload["expression_mean_available"]
    v7 = tmp_path / "prediction_v7_compat.npz"
    np.savez_compressed(v7, **v7_payload)
    loaded_v7 = PredictionBundle.from_npz(v7)
    assert loaded_v7.expression_interval_semantics == PredictionBundle.CONDITIONAL_KNOWN_STATE
    np.testing.assert_array_equal(loaded_v7.expression_mean_available, [True, True])
    np.testing.assert_array_equal(loaded_v7.expression_interval_available, [True, True])

    abstained_v7_payload = dict(v7_payload)
    abstained_v7_payload["abstain"] = np.asarray([False, True])
    abstained_v7_payload["labels"] = np.asarray([0, -1], dtype=np.int64)
    abstained_v7_payload["expression_interval_available"] = np.asarray([True, False])
    abstained_v7_payload["expression_lower"] = np.asarray(
        [[0.8, 1.8], [np.nan, np.nan]],
        dtype=np.float32,
    )
    abstained_v7_payload["expression_upper"] = np.asarray(
        [[1.2, 2.2], [np.nan, np.nan]],
        dtype=np.float32,
    )
    abstained_v7 = tmp_path / "prediction_v7_abstained.npz"
    np.savez_compressed(abstained_v7, **abstained_v7_payload)
    migrated_v7_abstained = PredictionBundle.from_npz(abstained_v7)
    np.testing.assert_array_equal(migrated_v7_abstained.expression_mean_available, [True, False])
    assert np.isfinite(migrated_v7_abstained.internal_aggregate_expression_mean).all()
    assert np.isnan(migrated_v7_abstained.public_cell_expression_mean[1]).all()

    v6_payload = dict(v7_payload)
    v6_payload["__version__"] = np.asarray(6, dtype=np.int64)
    del v6_payload["expression_interval_semantics"]
    del v6_payload["expression_interval_available"]
    v6 = tmp_path / "prediction_v6_compat.npz"
    np.savez_compressed(v6, **v6_payload)
    loaded_v6 = PredictionBundle.from_npz(v6)
    assert (
        loaded_v6.expression_interval_semantics == PredictionBundle.LEGACY_CONDITIONAL_KNOWN_STATE
    )
    np.testing.assert_array_equal(loaded_v6.expression_mean_available, [True, True])
    assert loaded_v6.expression_interval_available is None

    abstained_v6_payload = dict(v6_payload)
    abstained_v6_payload["abstain"] = np.asarray([False, True])
    abstained_v6_payload["labels"] = np.asarray([0, -1], dtype=np.int64)
    abstained_v6 = tmp_path / "prediction_v6_abstained.npz"
    np.savez_compressed(abstained_v6, **abstained_v6_payload)
    migrated_abstained = PredictionBundle.from_npz(abstained_v6)
    np.testing.assert_array_equal(migrated_abstained.expression_mean_available, [True, False])
    assert migrated_abstained.expression_interval_available is None
    np.testing.assert_array_equal(
        migrated_abstained.expression_mean,
        abstained_v6_payload["expression_mean"],
    )
    assert np.isfinite(migrated_abstained.internal_aggregate_expression_mean).all()
    np.testing.assert_array_equal(
        migrated_abstained.public_cell_expression_mean[0],
        abstained_v6_payload["expression_mean"][0],
    )
    assert np.isnan(migrated_abstained.public_cell_expression_mean[1]).all()
    np.testing.assert_array_equal(
        migrated_abstained.expression_lower,
        abstained_v6_payload["expression_lower"],
    )
    np.testing.assert_array_equal(
        migrated_abstained.expression_upper,
        abstained_v6_payload["expression_upper"],
    )
    np.testing.assert_array_equal(
        migrated_abstained.program_scores,
        abstained_v6_payload["program_scores"],
    )
    upgraded_v8 = tmp_path / "prediction_v8_upgraded.npz"
    migrated_abstained.to_npz(upgraded_v8)
    upgraded = PredictionBundle.from_npz(upgraded_v8)
    assert upgraded.expression_interval_semantics == PredictionBundle.CONDITIONAL_KNOWN_STATE
    np.testing.assert_array_equal(upgraded.expression_mean_available, [True, False])
    np.testing.assert_array_equal(upgraded.expression_interval_available, [True, False])
    np.testing.assert_array_equal(
        upgraded.expression_mean,
        abstained_v6_payload["expression_mean"],
    )
    assert np.isnan(upgraded.expression_lower[1]).all()
    assert np.isnan(upgraded.expression_upper[1]).all()
    np.testing.assert_array_equal(
        upgraded.program_scores,
        abstained_v6_payload["program_scores"],
    )

    v5_payload = dict(v6_payload)
    v5_payload["__version__"] = np.asarray(5, dtype=np.int64)
    del v5_payload["expression_space_id"]
    v5 = tmp_path / "prediction_v5_compat.npz"
    np.savez_compressed(v5, **v5_payload)
    loaded_v5 = PredictionBundle.from_npz(v5)
    assert loaded_v5.expression_space_id == ""
    assert loaded_v5.inference_seed == 23
    with pytest.raises(ValueError, match="expression_space_id"):
        loaded_v5.to_npz(tmp_path / "legacy_cannot_masquerade_as_v8.npz")

    v4_payload = dict(v5_payload)
    v4_payload["__version__"] = np.asarray(4, dtype=np.int64)
    for name in (
        "inference_seed",
        "latent_samples",
        "probability_threshold",
        "artifact_threshold",
    ):
        del v4_payload[name]
    v4 = tmp_path / "prediction_v4_compat.npz"
    np.savez_compressed(v4, **v4_payload)
    loaded_v4 = PredictionBundle.from_npz(v4)
    assert loaded_v4.inference_seed is None
    assert loaded_v4.latent_samples is None
    assert loaded_v4.probability_threshold is None
    assert loaded_v4.artifact_threshold is None
    assert loaded_v4.ood_sha256 == "e" * 64
    with pytest.raises(ValueError, match="inference decision provenance"):
        loaded_v4.to_npz(tmp_path / "legacy_cannot_masquerade_as_v5.npz")

    v3_payload = dict(v4_payload)
    v3_payload["__version__"] = np.asarray(3, dtype=np.int64)
    for name in ("ood_sha256", "ood_training_donors"):
        del v3_payload[name]
    v3 = tmp_path / "prediction_v3_compat.npz"
    np.savez_compressed(v3, **v3_payload)
    loaded_v3 = PredictionBundle.from_npz(v3)
    assert loaded_v3.ood_sha256 == ""
    assert loaded_v3.ood_training_donors is None
    assert loaded_v3.histology_sha256 == "c" * 64

    v2_payload = dict(v3_payload)
    v2_payload["__version__"] = np.asarray(2, dtype=np.int64)
    for name in ("histology_sha256", "program_sha256", "program_training_donors"):
        del v2_payload[name]
    v2 = tmp_path / "prediction_v2_compat.npz"
    np.savez_compressed(v2, **v2_payload)
    loaded_v2 = PredictionBundle.from_npz(v2)
    assert loaded_v2.histology_sha256 == ""
    assert loaded_v2.program_sha256 == ""
    assert loaded_v2.program_training_donors is None


def test_prediction_bundle_rejects_malformed_optional_outputs_and_provenance(tmp_path):
    bundle = _bundle()
    with pytest.raises(ValueError, match="supplied together"):
        replace(bundle, parent_type_names=None).validate()
    with pytest.raises(ValueError, match="unique"):
        replace(
            bundle,
            program_scores=np.ones((2, 2), dtype=np.float32),
            program_names=np.asarray(["duplicate", "duplicate"]),
        ).validate()
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        replace(bundle, checkpoint_sha256="not-a-hash").to_npz(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="program_sha256"):
        replace(bundle, program_sha256="").to_npz(tmp_path / "missing_program.npz")
    with pytest.raises(ValueError, match="histology_sha256"):
        replace(bundle, histology_sha256="").to_npz(tmp_path / "missing_histology.npz")
    with pytest.raises(ValueError, match="detector provenance"):
        replace(bundle, ood_sha256="", ood_training_donors=None).to_npz(
            tmp_path / "missing_ood.npz"
        )
    with pytest.raises(ValueError, match="detector hash and training donors"):
        replace(bundle, ood_training_donors=None).validate()
    with pytest.raises(ValueError, match="ood_sha256"):
        replace(bundle, ood_sha256="not-a-hash").validate()
    with pytest.raises(ValueError, match="program training donors"):
        replace(bundle, program_training_donors=None).to_npz(
            tmp_path / "missing_program_donors.npz"
        )
    with pytest.raises(ValueError, match="inference decision provenance"):
        replace(bundle, inference_seed=None).to_npz(tmp_path / "missing_decisions.npz")
    with pytest.raises(ValueError, match="inference_seed must be non-negative"):
        replace(bundle, inference_seed=-1).validate()
    with pytest.raises(ValueError, match="latent_samples must be positive"):
        replace(bundle, latent_samples=0).validate()
    with pytest.raises(ValueError, match="probability_threshold"):
        replace(bundle, probability_threshold=-0.1).validate()
    with pytest.raises(ValueError, match="artifact_threshold"):
        replace(bundle, artifact_threshold=1.1).validate()
    with pytest.raises(ValueError, match="expression_space_id"):
        replace(bundle, expression_space_id="").to_npz(tmp_path / "missing_expression_space.npz")
    with pytest.raises(ValueError, match="cannot expose conditional expression"):
        replace(
            bundle,
            abstain=np.asarray([False, True]),
            expression_interval_available=np.asarray([True, True]),
            expression_interval_semantics=PredictionBundle.CONDITIONAL_KNOWN_STATE,
        ).validate()
    with pytest.raises(ValueError, match="expression_mean_available"):
        replace(bundle, expression_mean_available=np.asarray([1, 1], dtype=np.int8)).validate()
    with pytest.raises(ValueError, match="require an available expression mean"):
        replace(
            bundle,
            expression_mean_available=np.asarray([True, False]),
            expression_interval_available=np.asarray([True, True]),
            expression_interval_semantics=PredictionBundle.CONDITIONAL_KNOWN_STATE,
        ).validate()

    abstained = replace(
        bundle,
        labels=np.asarray([0, -1], dtype=np.int64),
        abstain=np.asarray([False, True]),
        expression_mean_available=np.asarray([True, False]),
        expression_interval_available=np.asarray([True, False]),
        expression_interval_semantics=PredictionBundle.CONDITIONAL_KNOWN_STATE,
        expression_lower=np.asarray([[0.8, 1.8], [np.nan, np.nan]], dtype=np.float32),
        expression_upper=np.asarray([[1.2, 2.2], [np.nan, np.nan]], dtype=np.float32),
    )
    abstained.validate()
    np.testing.assert_array_equal(
        abstained.public_cell_expression_mean[0],
        abstained.expression_mean[0],
    )
    assert np.isnan(abstained.public_cell_expression_mean[1]).all()
    assert np.isfinite(abstained.internal_aggregate_expression_mean).all()

    valid = tmp_path / "valid.npz"
    bundle.to_npz(valid)
    with np.load(valid, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    del payload["parent_type_names"]
    malformed = tmp_path / "malformed.npz"
    np.savez_compressed(malformed, **payload)
    with pytest.raises(ValueError, match="parent arrays"):
        PredictionBundle.from_npz(malformed)

    with np.load(valid, allow_pickle=False) as archive:
        malformed_seed_payload = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    malformed_seed_payload["inference_seed"] = np.asarray(1.5, dtype=np.float64)
    malformed_seed = tmp_path / "malformed_seed.npz"
    np.savez_compressed(malformed_seed, **malformed_seed_payload)
    with pytest.raises(TypeError, match="inference_seed must be an integer"):
        PredictionBundle.from_npz(malformed_seed)


def test_predict_cells_exports_hierarchy_programs_and_provenance():
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=2,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=4,
            graph_layers=1,
            trunk_hidden_dims=(5,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
            fine_to_parent=(0, 0),
            num_parent_types=1,
            hard_type_routing=False,
        )
    )
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["pA", "pB"]),
        sample_ids=np.asarray(["sample1", "sample1"]),
        cell_type_labels=np.asarray(["A", "B"]),
        means=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        variances=np.full((2, 2), 0.2, dtype=np.float32),
        weights=np.asarray([0.5, 0.5]),
        latent_space_id="latent-v1",
    )
    programs = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    result = predict_cells(
        model,
        np.zeros((3, 3), dtype=np.float32),
        np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float32),
        ["n1", "n2", "n3"],
        prototypes,
        ["A", "B"],
        ["g1", "g2"],
        latent_samples=2,
        device="cpu",
        sample_id="sample1",
        donor_id="donor1",
        slide_id="slide1",
        checkpoint_sha256="a" * 64,
        prototype_sha256="b" * 64,
        histology_sha256="c" * 64,
        latent_space_id="latent-v1",
        model_version="0.1.0",
        parent_type_names=["parent"],
        program_matrix=programs,
        program_names=["P1", "P2"],
        program_sha256="d" * 64,
        program_training_donors=["training-donor"],
        ood_score=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        ood_sha256="e" * 64,
        ood_training_donors=["ood-training-donor"],
        inference_seed=41,
        probability_threshold=0.65,
        artifact_threshold=0.35,
        expression_space_id="log1p-cpm10k-v1",
    )
    result.validate(require_provenance=True)
    assert result.parent_type_probabilities.shape == (3, 1)
    assert result.program_scores.shape == (3, 2)
    assert result.ood_sha256 == "e" * 64
    assert result.ood_training_donors.tolist() == ["ood-training-donor"]
    assert result.inference_seed == 41
    assert result.latent_samples == 2
    assert result.probability_threshold == pytest.approx(0.65)
    assert result.artifact_threshold == pytest.approx(0.35)
    assert result.expression_space_id == "log1p-cpm10k-v1"
    assert result.expression_lower.dtype == np.float32
    assert result.expression_upper.dtype == np.float32
    np.testing.assert_allclose(result.program_scores, result.expression_mean @ programs)


def test_predict_cells_model_abstain_is_opt_in() -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=2,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            dropout=0.0,
            hard_type_routing=False,
            abstain_threshold=0.1,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fine_type_head.bias.copy_(torch.tensor([20.0, -20.0]))
        model.unknown_head.bias.fill_(-1.5)
        model.residual_logvar_head.bias.fill_(8.0)
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["p0", "p1"]),
        sample_ids=np.asarray(["sample", "sample"]),
        cell_type_labels=np.asarray(["type0", "type1"]),
        means=np.zeros((2, 2), dtype=np.float32),
        variances=np.ones((2, 2), dtype=np.float32),
        weights=np.full(2, 0.5, dtype=np.float32),
    )
    arguments = dict(
        model=model,
        features=np.zeros((1, 3), dtype=np.float32),
        coordinates_um=np.zeros((1, 2), dtype=np.float32),
        nucleus_ids=["nucleus"],
        prototypes=prototypes,
        type_names=["type0", "type1"],
        gene_names=["g0", "g1"],
        latent_samples=1,
        probability_threshold=0.6,
        device="cpu",
    )

    explicit_gates = predict_cells(**arguments)
    composite_gate = predict_cells(**arguments, use_model_abstain=True)

    assert explicit_gates.type_probabilities[0, 0] > 0.99
    assert explicit_gates.unknown_probability[0] < 0.4
    assert explicit_gates.abstain_score[0] > model.config.abstain_threshold
    assert not explicit_gates.abstain[0]
    assert explicit_gates.labels[0] == 0
    assert composite_gate.abstain[0]
    assert composite_gate.labels[0] == -1
    np.testing.assert_array_equal(explicit_gates.expression_mean_available, [True])
    np.testing.assert_array_equal(composite_gate.expression_mean_available, [False])
    assert np.isfinite(composite_gate.internal_aggregate_expression_mean).all()
    assert np.isnan(composite_gate.public_cell_expression_mean).all()


def test_predict_cells_prototype_sample_mismatch_is_explicit_control_only() -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=2,
            num_cell_types=2,
            expression_dim=1,
            latent_dim=1,
            graph_hidden_dim=2,
            graph_output_dim=2,
            graph_layers=1,
            trunk_hidden_dims=(2,),
            decoder_hidden_dims=(2,),
            dropout=0.0,
        )
    )
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["donor-prototype"]),
        sample_ids=np.asarray(["donor-sample"]),
        cell_type_labels=np.asarray(["type0"]),
        means=np.zeros((1, 1), dtype=np.float32),
        variances=np.ones((1, 1), dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
    )
    arguments = dict(
        model=model,
        features=np.zeros((1, 2), dtype=np.float32),
        coordinates_um=np.zeros((1, 2), dtype=np.float32),
        nucleus_ids=["nucleus"],
        prototypes=prototypes,
        type_names=["type0", "type1"],
        gene_names=["gene"],
        latent_samples=1,
        device="cpu",
        sample_id="target-sample",
    )

    with pytest.raises(ValueError, match="sample_id differs"):
        predict_cells(**arguments)
    controlled = predict_cells(**arguments, allow_prototype_sample_mismatch=True)
    assert controlled.sample_id == "target-sample"
    unsupported = replace(prototypes, cell_type_labels=np.asarray(["unsupported"]))
    with pytest.raises(
        ValueError,
        match="prototype types are absent from the model ontology: unsupported",
    ):
        predict_cells(
            **{**arguments, "prototypes": unsupported},
            allow_prototype_sample_mismatch=True,
        )


def test_predict_cells_intervals_include_prototype_assignment_uncertainty() -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=2,
            num_cell_types=2,
            expression_dim=1,
            latent_dim=1,
            graph_hidden_dim=2,
            graph_output_dim=2,
            graph_layers=1,
            trunk_hidden_dims=(2,),
            decoder_hidden_dims=(2,),
            dropout=0.0,
            hard_type_routing=False,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.unknown_head.bias.fill_(-20.0)
        model.residual_logvar_head.bias.fill_(-12.0)
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["left", "right"]),
        sample_ids=np.asarray(["sample", "sample"]),
        cell_type_labels=np.asarray(["type0", "type1"]),
        means=np.asarray([[-5.0], [5.0]], dtype=np.float32),
        variances=np.full((2, 1), 1.0e-4, dtype=np.float32),
        weights=np.full(2, 0.5, dtype=np.float32),
    )
    torch.manual_seed(17)
    result = predict_cells(
        model,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        ["nucleus"],
        prototypes,
        ["type0", "type1"],
        ["gene"],
        latent_samples=256,
        device="cpu",
    )

    assert result.latent_variance[0, 0] > 20.0


def test_predict_cells_requires_provenance_for_explicit_ood_scores():
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=2,
            num_cell_types=2,
            expression_dim=1,
            latent_dim=1,
            graph_hidden_dim=2,
            graph_output_dim=2,
            graph_layers=1,
            trunk_hidden_dims=(2,),
            decoder_hidden_dims=(2,),
            dropout=0.0,
            hard_type_routing=False,
        )
    )
    prototypes = PrototypeSet(
        prototype_ids=np.asarray(["p0", "p1"]),
        sample_ids=np.asarray(["sample", "sample"]),
        cell_type_labels=np.asarray(["type0", "type1"]),
        means=np.zeros((2, 1), dtype=np.float32),
        variances=np.ones((2, 1), dtype=np.float32),
        weights=np.full(2, 0.5, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="detector hash and training donors"):
        predict_cells(
            model,
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            ["nucleus"],
            prototypes,
            ["type0", "type1"],
            ["gene"],
            ood_score=np.zeros(1, dtype=np.float32),
            latent_samples=1,
            device="cpu",
        )
