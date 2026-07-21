from __future__ import annotations

import numpy as np

from heir.evaluation.generative_fusion import (
    CompositionStateModel,
    ContrastiveRetrieval,
    CountVAE,
    build_reference_mixture,
    calibration_slope,
    evaluate_ordered_gates,
    exact_sign_flip_test,
    fit_nb2_dispersion,
    holm_adjust,
    interval_coverage,
    nb2_deviance,
    nb2_log_prob,
    reliability_adjusted_variance,
    split_nb2_counts,
)


def _counts(seed: int, rows: int, genes: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rate = rng.gamma(2.0, 2.0, size=(rows, genes))
    return rng.poisson(rate).astype(np.int32)


def test_nb2_likelihood_deviance_and_dispersion_are_finite() -> None:
    counts = _counts(1, 40, 7)
    libraries = counts.sum(axis=1) + 1
    theta = fit_nb2_dispersion(counts, library_size=libraries)
    mean = np.maximum(counts.mean(axis=0, keepdims=True), 0.1).repeat(len(counts), axis=0)
    log_prob = np.asarray(nb2_log_prob(counts, mean, theta))
    deviance = np.asarray(nb2_deviance(counts, mean, theta))
    assert theta.shape == (7,)
    assert np.isfinite(log_prob).all()
    assert np.isfinite(deviance).all()
    assert (deviance >= 0).all()
    assert np.allclose(nb2_deviance(counts, np.maximum(counts, 1e-8), theta), 0, atol=1e-5)


def test_dispersion_uses_exposure_and_donor_specific_mean_rates() -> None:
    exposure = np.asarray([10, 20, 30, 40, 10, 20, 30, 40], dtype=float)
    donors = np.asarray(["A"] * 4 + ["B"] * 4)
    counts = np.column_stack(
        (
            exposure * np.asarray([0.1] * 4 + [0.5] * 4),
            exposure * np.asarray([0.2] * 4 + [0.4] * 4),
        )
    ).astype(np.int32)
    fit = fit_nb2_dispersion(
        counts,
        training_donor_ids=donors,
        library_size=exposure,
    )
    assert np.array_equal(np.asarray(fit), np.full(2, 1.0e6))


def test_nb_compatible_split_reconstructs_and_is_deterministic() -> None:
    counts = _counts(2, 30, 5)
    theta = np.linspace(0.5, 5.0, 5)
    first, second = split_nb2_counts(counts, theta, seed=17)
    again_first, again_second = split_nb2_counts(counts, theta, seed=17)
    assert np.array_equal(first + second, counts)
    assert np.array_equal(first, again_first)
    assert np.array_equal(second, again_second)


def test_reference_mixture_preserves_types_and_natural_weights() -> None:
    rng = np.random.default_rng(3)
    latent = np.r_[rng.normal(-2, 0.2, (20, 4)), rng.normal(2, 0.2, (60, 4))]
    types = np.asarray(["A"] * 20 + ["B"] * 60)
    donors = np.asarray(["d1"] * 40 + ["d2"] * 40)
    mixture = build_reference_mixture(
        latent, type_ids=types, donor_ids=donors, n_components=2, seed=11
    )
    assert mixture.type_names == ("A", "B")
    # Components are donor/type-specific: A/d1, B/d1, and B/d2 each keep two
    # states rather than collapsing donor-specific B states into one centroid.
    assert mixture.means.shape == (6, 4)
    assert all(
        len(mixture.component_indices(donor, type_label)) == 2
        for donor, type_label in (("d1", "A"), ("d1", "B"), ("d2", "B"))
    )
    assert np.allclose(mixture.type_weights(), [0.25, 0.75], atol=1e-6)
    assert not np.allclose(mixture.type_means()[0], mixture.type_means()[1])


def test_count_vae_has_separate_modalities_and_decodes_exposure() -> None:
    st = _counts(4, 24, 6) + 1
    sc = _counts(5, 24, 6) + 1
    values = np.r_[st, sc]
    modality = np.r_[np.ones(len(st), dtype=int), np.zeros(len(sc), dtype=int)]
    model = CountVAE(n_genes=6, latent_dim=3, hidden_dim=12)
    model.fit_model(
        values,
        modality=modality,
        library_size=values.sum(axis=1),
        epochs=1,
        batch_size=16,
        seed=7,
    )
    z_st = model.encode_numpy(st, modality="st")
    z_sc = model.encode_numpy(sc, modality="scrna")
    decoded_one = model.decode_numpy(z_st, library_size=np.ones(len(st)), modality="st")
    decoded_ten = model.decode_numpy(z_st, library_size=np.full(len(st), 10), modality="st")
    assert z_st.shape == z_sc.shape == (24, 3)
    assert set(model.decoders) == {"st", "scrna"}
    assert np.allclose(decoded_one.sum(axis=1), 1, atol=1e-5)
    assert np.allclose(decoded_ten.sum(axis=1), 10, atol=1e-4)


def test_count_vae_aligns_training_donor_pseudobulks_without_heldout_rows() -> None:
    rng = np.random.default_rng(41)
    donor_profiles = rng.dirichlet(np.ones(6), size=3)
    values, modalities, donors, observation_ids = [], [], [], []
    for donor_index, profile in enumerate(donor_profiles):
        donor = f"d{donor_index}"
        for modality, assay_bias in (("st", 1.0), ("scrna", 1.4)):
            biased = profile * np.asarray([assay_bias, 1, 1, 1, 1, 1 / assay_bias])
            biased /= biased.sum()
            for replicate in range(4):
                values.append(rng.multinomial(300, biased))
                modalities.append(modality)
                donors.append(donor)
                observation_ids.append(f"{donor}-{modality}-{replicate}")
    counts = np.asarray(values, dtype=np.int32)
    aligned = CountVAE(n_genes=6, latent_dim=3, hidden_dim=12, seed=8)
    aligned.fit_model(
        counts,
        modality=modalities,
        training_donor_ids=donors,
        alignment_weight=1.0,
        observation_ids=observation_ids,
        heldout_observation_ids=("heldout-ST-outcome",),
        epochs=8,
        batch_size=12,
        seed=12,
    )
    receipt = aligned.alignment_diagnostics
    assert receipt is not None
    assert receipt.donor_ids == ("d0", "d1", "d2")
    assert receipt.pre_matched_mse == receipt.pre_mse
    assert receipt.post_matched_mse == receipt.post_mse
    assert receipt.pre_mismatched_mse > 0
    assert receipt.post_mismatched_mse > 0
    assert receipt.post_matched_to_mismatched_ratio < (
        receipt.pre_matched_to_mismatched_ratio
    )
    assert receipt.post_separation > receipt.pre_separation
    assert receipt.optimizer_applications_per_epoch == 2
    assert receipt.optimizer_applications_total == 16
    assert receipt.matched_pairs_closer_post is True
    assert receipt.alignment_improved is True
    assert receipt.support_criterion_met is True
    assert "heldout-ST-outcome" not in aligned.training_observation_ids

    unaligned = CountVAE(n_genes=6, latent_dim=3, hidden_dim=12, seed=8)
    unaligned.fit_model(
        counts,
        modality=modalities,
        training_donor_ids=donors,
        alignment_weight=0.0,
        observation_ids=observation_ids,
        heldout_observation_ids=("heldout-ST-outcome",),
        epochs=8,
        batch_size=12,
        seed=12,
    )
    unaligned_receipt = unaligned.alignment_diagnostics
    assert unaligned_receipt is not None
    assert unaligned_receipt.optimizer_applications_per_epoch == 0
    assert unaligned_receipt.optimizer_applications_total == 0
    assert unaligned_receipt.support_criterion_met is False
    assert receipt.post_matched_to_mismatched_ratio < (
        unaligned_receipt.post_matched_to_mismatched_ratio
    )


def test_count_vae_integrates_decoder_over_latent_uncertainty() -> None:
    model = CountVAE(n_genes=5, latent_dim=3, hidden_dim=10, seed=9)
    mean = np.zeros((4, 3), dtype=np.float32)
    variance = np.full_like(mean, 0.8)
    point = model.decode_numpy(mean, library_size=np.full(4, 100), modality="st")
    moments = model.decode_diagonal_gaussian_numpy(
        mean,
        variance,
        library_size=np.full(4, 100),
        modality="st",
        dispersion=np.full(5, 2.0),
        samples=64,
        batch_size=2,
        seed=4,
    )
    assert moments["mean_counts"].shape == (4, 5)
    assert moments["dispersion_source"] == "provided_primary"
    assert np.array_equal(moments["theta"], np.full(5, 2.0, dtype=np.float32))
    assert np.max(np.abs(moments["mean_counts"] - point)) > 1.0e-4
    assert np.all(moments["latent_variance_counts"] > 0)
    assert np.all(
        moments["predictive_variance_counts"] > moments["latent_variance_counts"]
    )
    endpoint = model.decode_diagonal_gaussian_numpy(
        mean,
        variance,
        library_size=np.full(4, 100),
        modality="st",
        endpoint_gene_indices=(0, 2, 4),
        dispersion=np.asarray([2.0, 3.0, 4.0]),
        samples=8,
    )
    assert endpoint["mean_counts"].shape == (4, 3)
    assert np.array_equal(endpoint["gene_indices"], [0, 2, 4])
    assert np.array_equal(endpoint["theta"], [2.0, 3.0, 4.0])
    zero = model.decode_diagonal_gaussian_numpy(
        mean,
        np.zeros_like(mean),
        library_size=np.full(4, 100),
        modality="st",
        samples=8,
    )
    assert np.allclose(zero["mean_counts"], point, atol=1.0e-5)


def test_staged_composition_state_modes_and_poe() -> None:
    rng = np.random.default_rng(6)
    image = rng.normal(size=(36, 8)).astype(np.float32)
    composition = np.zeros((36, 2), dtype=np.float32)
    composition[:18, 0] = 1
    composition[18:, 1] = 1
    target = np.where(composition[:, :1] > 0, -1.0, 1.0).repeat(3, axis=1)
    reference_latent = np.r_[rng.normal(-1, 0.1, (20, 3)), rng.normal(1, 0.1, (20, 3))]
    reference = build_reference_mixture(
        reference_latent,
        type_ids=np.asarray(["A"] * 20 + ["B"] * 20),
        donor_ids=np.asarray(["q"] * 40),
        n_components=2,
    )
    model = CompositionStateModel(image_dim=8, latent_dim=3, n_types=2, hidden_dim=12)
    model.fit_model(
        image,
        target.astype(np.float32),
        composition_targets=composition,
        type_ids=("A", "B"),
        type_anchor_means=reference.type_means(),
        epochs=1,
        batch_size=12,
        seed=9,
    )
    image_details = model.predict_details_numpy(image[:5], mode="image_only")
    fused = model.predict_details_numpy(image[:5], reference=reference, mode="full_poe")
    routed = model.predict_details_numpy(
        image[:5], reference=reference, mode="composition_reference_mean"
    )
    assert image_details["composition"].shape == (5, 2)
    assert fused["type_mean"].shape == (5, 2, 3)
    assert routed["latent"].shape == (5, 3)
    assert not np.allclose(fused["latent"], image_details["latent"])
    receipt = model.variance_calibration_receipt()
    assert receipt["rows"] == len(image)
    assert receipt["nll_after"] <= receipt["nll_before"]
    assert np.all(receipt["per_type_variance"] > 0)
    assert "calibrated_type_variance" in model.state_dict()
    assert np.allclose(
        image_details["type_variance"],
        np.broadcast_to(receipt["per_type_variance"], image_details["type_variance"].shape),
    )


def test_poe_retains_image_state_when_donor_reference_lacks_a_type() -> None:
    rng = np.random.default_rng(61)
    reference = build_reference_mixture(
        rng.normal(-1, 0.2, (20, 3)),
        type_ids=np.asarray(["A"] * 20),
        donor_ids=np.asarray(["q"] * 20),
        n_components=2,
    )
    model = CompositionStateModel(
        image_dim=4,
        type_labels=("A", "B"),
        n_genes=5,
        latent_dim=3,
        hidden_dim=8,
    )
    image = rng.normal(size=(3, 4)).astype(np.float32)
    image_only = model.predict(
        image, mode="M0", library_size=np.full(3, 100, dtype=np.float32)
    )
    fused = model.predict(
        image,
        mode="M3",
        library_size=np.full(3, 100, dtype=np.float32),
        donor_ids=np.asarray(["q"] * 3),
        reference=reference,
    )
    assert fused.reference_supported is not None
    assert np.array_equal(
        fused.reference_supported.cpu().numpy(),
        np.asarray([[True, False]] * 3),
    )
    assert np.allclose(
        fused.state_mean[:, 1].detach().cpu(),
        image_only.state_mean[:, 1].detach().cpu(),
    )


def test_contrastive_retrieval_reports_normalized_entropy() -> None:
    rng = np.random.default_rng(8)
    image = rng.normal(size=(20, 5)).astype(np.float32)
    latent = rng.normal(size=(20, 3)).astype(np.float32)
    model = ContrastiveRetrieval(5, 3, embedding_dim=3)
    model.fit_model(
        image,
        latent,
        hard_negative_molecular=np.roll(latent, 1, axis=0),
        hard_negative_weight=0.5,
        epochs=1,
        batch_size=10,
        temperature=0.07,
        seed=2,
    )
    retrieved, entropy = model.retrieve_numpy(image[:4], latent, return_entropy=True, batch_size=2)
    assert retrieved.shape == (4, 3)
    assert entropy.shape == (4,)
    assert np.all((entropy >= 0) & (entropy <= 1.00001))


def test_statistics_and_ordered_gates_fail_closed() -> None:
    effects = np.asarray([2.0] * 10 + [1.0] * 3)
    sign = exact_sign_flip_test(effects)
    assert sign["p_value"] < 0.01
    adjusted = holm_adjust({"a": 0.01, "b": 0.04})
    assert adjusted == {"a": 0.02, "b": 0.04}
    comparisons = {
        name: {
            "donor_improvement": effects.tolist(),
            "mean_improvement": float(effects.mean()),
            "sign_flip": sign,
            "relative_gain": 0.1,
        }
        for name in (
            "M3_vs_M0",
            "M3_vs_M1",
            "M3_vs_M4",
            "M3_vs_M2",
            "M3_vs_M6",
            "M3_vs_M7",
            "M8_vs_M3",
        )
    }
    result = evaluate_ordered_gates(
        {
            "comparisons": comparisons,
            "central_relative_gain": 0.1,
            "no_severe_indication_reversal": True,
            "holm_gate2": {"M3_vs_M1": 0.01, "M3_vs_M4": 0.01},
            "holm_gate4": {"M3_vs_M6": 0.01, "M3_vs_M7": 0.01},
        }
    )
    assert result["supported"] is True
    result_failed = evaluate_ordered_gates(
        {
            "comparisons": comparisons,
            "central_relative_gain": 0.01,
            "no_severe_indication_reversal": True,
        }
    )
    assert result_failed["supported"] is False
    assert result_failed["stopped_at"] == "gate_1"


def test_calibration_reliability_and_interval_metrics() -> None:
    truth = np.arange(12, dtype=float).reshape(4, 3)
    predicted = truth / 2
    assert np.isclose(calibration_slope(truth, predicted), 2.0)
    reliable = reliability_adjusted_variance(predicted, truth + 1, truth + 2)
    assert reliable["reliable_genes"] == 3
    coverage = interval_coverage(truth, truth - 1, truth + 1)
    assert coverage["coverage"] == 1.0
