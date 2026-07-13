"""Regression tests for patch-safe constrained HEIR refinement."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from heir.config import RefinementConfig
from heir.models import HEIRConfig, HEIRModel
from heir.refinement import IterativeRefiner
from heir.refinement import iterative as iterative_module
from heir.refinement.anchors import (
    AnchorStatus,
    select_anchors,
    update_anchor_lifecycle,
)
from heir.refinement.ema import EMATeacher
from heir.training import HEIRTrainingBatch, HEIRTrainingResult


class _RecordingTrainer:
    molecular_e_step_mode = "live_student_negative_control"

    def __init__(self, model: HEIRModel) -> None:
        self.model = model
        self.calls = []
        self.trainable_parameter_names = []

    def fit(self, training_batches, validation_batches):
        self.calls.append(tuple(training_batches))
        self.trainable_parameter_names.append(
            tuple(name for name, value in self.model.named_parameters() if value.requires_grad)
        )
        return HEIRTrainingResult(0, 1.0, tuple())

    def _epoch(self, batches, optimizer):
        del batches, optimizer
        return {
            "selection_total": 1.0,
            "uot": 0.0,
            "pseudobulk": 0.0,
            "dirichlet": 0.0,
            "calibration": 0.0,
        }

    def _anchor_constraints(self, batch):
        del batch
        return None

    def transport_responsibilities(self, batch, output):
        del batch
        responsibilities = output.conditional_prototype_probabilities.detach()
        result = SimpleNamespace(
            source_marginal_error=torch.tensor(0.0),
            target_marginal_error=torch.tensor(0.0),
            target_marginal=responsibilities.mean(dim=0),
        )
        return responsibilities, result


def _batch(cells: int, bag_id: str, offset: float) -> HEIRTrainingBatch:
    morphology = torch.full((cells, 3), offset)
    return HEIRTrainingBatch(
        morphology=morphology,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        prototype_means=torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        prototype_variances=torch.ones(2, 2),
        prototype_types=torch.tensor([0, 1]),
        prototype_weights=torch.tensor([0.5, 0.5]),
        target_composition=torch.tensor([0.5, 0.5]),
        target_pseudobulk=torch.zeros(2),
        sample_id="sample-a",
        bag_id=bag_id,
        donor_id="donor-a",
        block_id="block-a",
        analysis_role="development",
    )


def test_refinement_keeps_patch_state_separate_and_updates_one_sample_prior() -> None:
    torch.manual_seed(7)
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
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    config = RefinementConfig(
        maximum_rounds=2,
        broad_refinement_rounds=0,
        prior_old_weight=0.80,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=1,
    )
    batches = (_batch(3, "left", -2.0), _batch(5, "right", 2.0))
    result = IterativeRefiner(lambda: trainer, config).fit(batches, batches)
    assert len(result.rounds) == 2
    assert len(result.round_state_dicts) == 2
    assert len(trainer.calls) == 2
    first_round = trainer.calls[0]
    assert first_round[0].anchor_labels.shape == (3,)
    assert first_round[1].anchor_labels.shape == (5,)
    assert torch.all(first_round[0].anchor_labels == -100)
    assert torch.all(first_round[1].anchor_labels == -100)
    assert torch.all(trainer.calls[1][0].anchor_labels >= 0)
    assert torch.all(trainer.calls[1][1].anchor_labels >= 0)
    assert result.rounds[0].provisional == 8
    assert result.rounds[0].trusted == 0
    assert result.rounds[1].trusted == 8
    assert torch.equal(first_round[0].prototype_weights, first_round[1].prototype_weights)
    assert "donor-a::sample-a" in result.sample_prototype_weights


def test_hierarchical_refinement_uses_parent_then_fine_anchors() -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=3,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            fine_to_parent=(0, 0, 1),
            dropout=0.0,
            hard_type_routing=False,
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    batch = HEIRTrainingBatch(
        morphology=torch.randn(5, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=None,
        prototype_means=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        prototype_variances=torch.ones(3, 2),
        prototype_types=torch.tensor([0, 1, 2]),
        prototype_weights=torch.tensor([0.4, 0.3, 0.3]),
        target_composition=torch.tensor([0.4, 0.3, 0.3]),
        target_pseudobulk=torch.zeros(2),
        sample_id="hierarchical",
        bag_id="bag0",
        donor_id="donor-h",
        block_id="block-h",
        analysis_role="development",
    )
    config = RefinementConfig(
        maximum_rounds=4,
        broad_refinement_rounds=2,
        prior_old_weight=0.80,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=1,
    )
    IterativeRefiner(lambda: trainer, config).fit([batch], [batch])
    broad_provisional = trainer.calls[0][0]
    broad_trusted = trainer.calls[1][0]
    fine_provisional = trainer.calls[2][0]
    fine_trusted = trainer.calls[3][0]
    assert broad_provisional.parent_anchor_labels is not None
    assert torch.all(broad_provisional.parent_anchor_labels == -100)
    assert torch.all(broad_provisional.anchor_labels == -100)
    assert torch.all(broad_trusted.parent_anchor_labels >= 0)
    assert torch.equal(
        fine_provisional.parent_anchor_labels,
        broad_trusted.parent_anchor_labels,
    )
    assert fine_provisional.anchor_labels is not None
    assert torch.all(fine_provisional.anchor_labels == -100)
    assert torch.all(fine_trusted.anchor_labels >= 0)
    torch.testing.assert_close(broad_provisional.prototype_weights, batch.prototype_weights)
    torch.testing.assert_close(broad_trusted.prototype_weights, batch.prototype_weights)
    assert all(
        names and all(name.startswith("parent_type_head.") for name in names)
        for names in trainer.trainable_parameter_names[:2]
    )
    assert "fine_type_head.weight" in trainer.trainable_parameter_names[2]
    assert "fine_type_head.weight" in trainer.trainable_parameter_names[3]
    assert not any(
        name.startswith("expression_decoder.") for name in trainer.trainable_parameter_names[2]
    )


def test_broad_transport_uses_binary_parent_support_without_probability_reweighting() -> None:
    model = HEIRModel(
        HEIRConfig(
            morphology_dim=3,
            num_cell_types=3,
            expression_dim=2,
            latent_dim=2,
            graph_hidden_dim=4,
            graph_output_dim=3,
            graph_layers=1,
            trunk_hidden_dims=(4,),
            decoder_hidden_dims=(4,),
            fine_to_parent=(0, 0, 1),
            dropout=0.0,
            hard_type_routing=False,
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    refiner = IterativeRefiner(
        lambda: trainer,
        RefinementConfig(
            maximum_rounds=4,
            broad_refinement_rounds=2,
            require_view_agreement=False,
        ),
    )
    teacher = EMATeacher(model, 0.99)
    captured = {}
    original_forward = teacher.model.forward

    def recording_forward(self, *args, **kwargs):
        captured["constraints"] = kwargs.get("cell_type_constraints").detach().clone()
        return original_forward(*args, **kwargs)

    teacher.model.forward = MethodType(recording_forward, teacher.model)
    batch = replace(
        _batch(2, "broad", 0.0),
        prototype_means=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        prototype_variances=torch.ones(3, 2),
        prototype_types=torch.tensor([0, 1, 2]),
        prototype_weights=torch.full((3,), 1.0 / 3.0),
        target_composition=torch.full((3,), 1.0 / 3.0),
    )
    refiner._teacher_transport(
        trainer,
        teacher,
        batch,
        parent_probabilities=np.asarray([[0.51, 0.49], [0.1, 0.9]], dtype=np.float32),
        broad_level=True,
    )

    torch.testing.assert_close(
        captured["constraints"],
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_default_round_teacher_copies_the_accepted_student_exactly() -> None:
    student = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    )
    teacher = EMATeacher(student)
    assert teacher.decay == 0.0

    with torch.no_grad():
        for value in student.state_dict().values():
            value.fill_(3 if not torch.is_floating_point(value) else 3.25)
    teacher.update(student)

    for name, expected in student.state_dict().items():
        assert torch.equal(teacher.model.state_dict()[name], expected)


def test_spatial_selection_uses_frozen_score_and_weak_objectives_only_as_safety() -> None:
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
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    scores = iter((0.0, 0.4, 0.2))
    config = RefinementConfig(
        maximum_rounds=2,
        broad_refinement_rounds=0,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=3,
        round_selection_mode="spatial",
    )
    refiner = IterativeRefiner(
        lambda: trainer,
        config,
        spatial_validation_scorer=lambda _model, _batches: next(scores),
    )
    result = refiner.fit([_batch(3, "train", 0.0)], [_batch(3, "validation", 0.0)])

    assert result.round_zero_spatial_validation_score == pytest.approx(0.0)
    assert result.selected_round == 1
    assert [item.committed for item in result.rounds] == [True, False]
    assert result.rounds[0].spatial_validation_score == pytest.approx(0.4)
    assert result.stopped_reason == "validation_degraded_rollback"


def test_spatial_selection_requires_a_scorer() -> None:
    with pytest.raises(ValueError, match="requires a frozen spatial validation scorer"):
        IterativeRefiner(
            lambda: _RecordingTrainer(
                HEIRModel(
                    HEIRConfig(
                        morphology_dim=3,
                        num_cell_types=2,
                        expression_dim=2,
                    )
                )
            ),
            RefinementConfig(
                maximum_rounds=2,
                broad_refinement_rounds=0,
                round_selection_mode="spatial",
            ),
        )


def test_view_agreement_fails_closed_without_independent_views() -> None:
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
        )
    )
    trainer = _RecordingTrainer(model)
    with pytest.raises(ValueError, match="scale-held-out view predictions"):
        IterativeRefiner(
            lambda: trainer,
            RefinementConfig(
                broad_refinement_rounds=0,
                require_view_agreement=True,
            ),
        ).fit(
            [_batch(3, "bag0", 0.0)],
            [_batch(3, "bag1", 0.0)],
        )


def test_same_checkpoint_views_are_not_anchor_evidence_by_default(monkeypatch) -> None:
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
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    observed_views = []
    original = iterative_module.select_anchors

    def record_views(*args, **kwargs):
        observed_views.append(kwargs.get("view_predictions"))
        return original(*args, **kwargs)

    monkeypatch.setattr(iterative_module, "select_anchors", record_views)
    config = RefinementConfig(
        maximum_rounds=1,
        broad_refinement_rounds=0,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
    )
    batch = _batch(3, "bag0", 0.0)
    IterativeRefiner(lambda: trainer, config).fit(
        [batch],
        [batch],
        view_probabilities={"donor-a::sample-a::bag0": np.asarray([[0, 0, 0], [1, 1, 1]])},
    )

    assert observed_views == [None]


def test_broad_refinement_requires_a_hierarchical_model() -> None:
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
        )
    )
    trainer = _RecordingTrainer(model)

    with pytest.raises(ValueError, match="broad refinement requires a hierarchical model"):
        IterativeRefiner(
            lambda: trainer,
            RefinementConfig(require_view_agreement=False),
        ).fit(
            [_batch(3, "bag0", 0.0)],
            [_batch(3, "validation", 0.0)],
        )


def test_anchor_lifecycle_recomputes_confidence_and_can_relabel_or_revoke() -> None:
    def update(
        probabilities: np.ndarray,
        previous=None,
        *,
        ood: bool = False,
    ):
        selection = select_anchors(
            probabilities,
            min_probability=0.90,
            max_normalized_entropy=1.0,
            ood_mask=np.asarray([ood]),
        )
        return update_anchor_lifecycle(
            selection,
            probabilities,
            previous,
            min_probability=0.90,
        )

    provisional = update(np.asarray([[0.97, 0.03]]))
    assert provisional.status[0] == AnchorStatus.PROVISIONAL
    assert provisional.labels[0] == 0
    assert provisional.confidence[0] == pytest.approx(0.97)

    trusted = update(np.asarray([[0.96, 0.04]]), provisional)
    assert trusted.status[0] == AnchorStatus.TRUSTED
    assert trusted.confidence[0] == pytest.approx(0.96)

    # A soft dip is retained at the lower hysteresis threshold, using the new
    # posterior rather than the permanently frozen 0.96 confidence.
    retained = update(np.asarray([[0.85, 0.15]]), trusted)
    assert retained.status[0] == AnchorStatus.TRUSTED
    assert retained.confidence[0] == pytest.approx(0.85)

    challenged = update(np.asarray([[0.04, 0.96]]), retained)
    assert challenged.status[0] == AnchorStatus.CHALLENGED
    assert not challenged.accepted[0]

    relabelled = update(np.asarray([[0.03, 0.97]]), challenged)
    assert relabelled.status[0] == AnchorStatus.PROVISIONAL
    assert relabelled.labels[0] == 1
    assert relabelled.confidence[0] == pytest.approx(0.97)

    revoked = update(np.asarray([[0.02, 0.98]]), relabelled, ood=True)
    assert revoked.status[0] == AnchorStatus.REVOKED
    assert not revoked.accepted[0]


class _ScriptedRefiner(IterativeRefiner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prediction_round = 0

    def _teacher_probabilities(self, teacher, batch):
        del teacher
        first = self.prediction_round == 0
        self.prediction_round += 1
        probabilities = np.tile([0.95, 0.05] if first else [0.05, 0.95], (3, 1))
        return (
            probabilities,
            probabilities,
            np.zeros(3, dtype=bool),
            np.zeros(3, dtype=np.float32),
            None,
        )

    def _teacher_transport(
        self,
        trainer,
        teacher,
        batch,
        parent_probabilities=None,
        broad_level=False,
    ):
        del trainer, teacher, batch, parent_probabilities, broad_level
        probabilities = np.tile(
            [0.95, 0.05] if self.prediction_round == 1 else [0.05, 0.95],
            (3, 1),
        )
        return probabilities, 0.1, 0.0, probabilities.mean(axis=0)


class _DegradingTrainer(_RecordingTrainer):
    def __init__(self, model: HEIRModel) -> None:
        super().__init__(model)
        self.model_states = []

    def fit(self, training_batches, validation_batches):
        del validation_batches
        self.calls.append(tuple(training_batches))
        call_id = len(self.calls)
        with torch.no_grad():
            next(self.model.parameters()).fill_(float(call_id))
        self.model_states.append(
            {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        )
        loss = 0.9 if call_id == 1 else 2.0
        return HEIRTrainingResult(0, loss, tuple())


class _LossScriptTrainer(_RecordingTrainer):
    def __init__(self, model: HEIRModel, losses) -> None:
        super().__init__(model)
        self.losses = iter(losses)
        self.model_states = []

    def fit(self, training_batches, validation_batches):
        del validation_batches
        self.calls.append(tuple(training_batches))
        with torch.no_grad():
            next(self.model.parameters()).fill_(float(len(self.calls)))
        self.model_states.append(
            {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        )
        return HEIRTrainingResult(0, float(next(self.losses)), tuple())


def test_validation_safety_uses_immutable_round_zero_ceiling() -> None:
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
            abstain_threshold=1.0,
        )
    )
    trainer = _LossScriptTrainer(model, (1.009, 1.018))
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config = RefinementConfig(
        maximum_rounds=2,
        broad_refinement_rounds=0,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        maximum_validation_loss_degradation=0.01,
    )

    result = IterativeRefiner(lambda: trainer, config).fit(
        [_batch(3, "train", 0.0)],
        [_batch(3, "validation", 0.0)],
    )

    assert [round_.committed for round_ in result.rounds] == [True, False]
    assert result.selected_round == 0
    assert result.stopped_reason == "validation_degraded_rollback"
    for name, expected in initial_state.items():
        assert torch.equal(trainer.model.state_dict()[name], expected)


def test_fixed_schedule_retains_lowest_loss_safe_round() -> None:
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
            abstain_threshold=1.0,
        )
    )
    trainer = _LossScriptTrainer(model, (0.8, 0.9))
    config = RefinementConfig(
        maximum_rounds=2,
        broad_refinement_rounds=0,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
    )

    result = IterativeRefiner(lambda: trainer, config).fit(
        [_batch(3, "train", 0.0)],
        [_batch(3, "validation", 0.0)],
    )

    assert [round_.committed for round_ in result.rounds] == [True, True]
    assert result.selected_round == 1
    for name, expected in trainer.model_states[0].items():
        assert torch.equal(trainer.model.state_dict()[name], expected)


def test_strict_artifact_refinement_never_recomputes_responsibilities(monkeypatch) -> None:
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
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    trainer.molecular_e_step_mode = "strict_artifact"
    responsibilities = torch.tensor(
        [[0.80, 0.10], [0.05, 0.70], [0.00, 0.00]],
        dtype=torch.float32,
    )
    batch = replace(
        _batch(3, "bag0", 0.0),
        molecular_responsibilities=responsibilities,
        source_artifacts=("unused-frozen-e-step.npz",),
        source_sha256=("0" * 64,),
        source_roles=("frozen_e_step",),
    )
    monkeypatch.setattr(
        iterative_module.MolecularEStepArtifact,
        "load_npz",
        lambda path: SimpleNamespace(
            source_marginal_residual=0.0,
            target_marginal_residual=0.01,
        ),
    )

    class _NoLiveTeacherRefiner(IterativeRefiner):
        def _teacher_probabilities(self, teacher, current_batch):
            del teacher, current_batch
            raise AssertionError("strict artifact mode consulted the live teacher")

        def _teacher_transport(self, trainer_, teacher, current_batch, **kwargs):
            del trainer_, teacher, current_batch, kwargs
            raise AssertionError("strict artifact mode recomputed transport")

    config = RefinementConfig(
        maximum_rounds=2,
        broad_refinement_rounds=0,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
    )
    _NoLiveTeacherRefiner(lambda: trainer, config).fit(
        [batch],
        [batch],
        view_probabilities={"donor-a::sample-a::bag0": np.asarray([[1, 1, 1], [1, 1, 1]])},
    )

    assert len(trainer.calls) == 2
    for call in trainer.calls:
        torch.testing.assert_close(call[0].molecular_responsibilities, responsibilities)


def test_degraded_round_rolls_back_model_prior_and_skips_ema(monkeypatch) -> None:
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
            abstain_threshold=1.0,
        )
    )
    trainer = _DegradingTrainer(model)
    ema_updates = []

    class _CountingTeacher(EMATeacher):
        def update(self, student) -> None:
            ema_updates.append(len(trainer.calls))
            super().update(student)

    monkeypatch.setattr(iterative_module, "EMATeacher", _CountingTeacher)
    config = RefinementConfig(
        maximum_rounds=3,
        broad_refinement_rounds=0,
        prior_old_weight=0.80,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=3,
        objective_stability_tolerance=0.01,
    )
    result = _ScriptedRefiner(lambda: trainer, config).fit(
        [_batch(3, "bag0", 0.0)],
        [_batch(3, "validation", 0.0)],
    )

    assert result.stopped_reason == "validation_degraded_rollback"
    assert [item.committed for item in result.rounds] == [True, False]
    assert ema_updates == [1]
    assert len(trainer.calls) == 2
    for name, expected in trainer.model_states[0].items():
        assert torch.equal(trainer.model.state_dict()[name], expected)
    np.testing.assert_allclose(
        result.sample_prototype_weights["donor-a::sample-a"],
        trainer.calls[0][0].prototype_weights.numpy(),
    )
    assert not np.allclose(
        result.sample_prototype_weights["donor-a::sample-a"],
        trainer.calls[1][0].prototype_weights.numpy(),
    )


def test_first_degraded_round_restores_complete_round_zero_snapshot(monkeypatch) -> None:
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
            abstain_threshold=1.0,
        )
    )

    class _FirstRoundDegradingTrainer(_RecordingTrainer):
        def fit(self, training_batches, validation_batches):
            del validation_batches
            self.calls.append(tuple(training_batches))
            with torch.no_grad():
                next(self.model.parameters()).fill_(9.0)
            return HEIRTrainingResult(0, 2.0, tuple())

    trainer = _FirstRoundDegradingTrainer(model)
    initial_state = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }
    teacher_loads = []
    ema_updates = []

    class _ObservedTeacher(EMATeacher):
        def update(self, student) -> None:
            ema_updates.append(True)
            super().update(student)

        def load_state_dict(self, values) -> None:
            teacher_loads.append(
                {name: value.detach().clone() for name, value in values["model"].items()}
            )
            super().load_state_dict(values)

    monkeypatch.setattr(iterative_module, "EMATeacher", _ObservedTeacher)
    config = RefinementConfig(
        maximum_rounds=2,
        broad_refinement_rounds=0,
        prior_old_weight=0.80,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
        stable_rounds_required=3,
        objective_stability_tolerance=0.01,
    )
    original_batch = _batch(3, "bag0", 0.0)
    result = _ScriptedRefiner(lambda: trainer, config).fit(
        [original_batch],
        [_batch(3, "validation", 0.0)],
    )

    assert result.stopped_reason == "validation_degraded_rollback"
    assert result.round_zero_validation_loss == 1.0
    assert result.selected_round == 0
    assert len(result.rounds) == 1
    assert result.rounds[0].round_id == 1
    assert not result.rounds[0].committed
    assert len(trainer.calls) == 1
    assert ema_updates == []
    assert len(teacher_loads) == 1
    for name, expected in initial_state.items():
        assert torch.equal(trainer.model.state_dict()[name], expected)
        assert torch.equal(teacher_loads[0][name], expected)
    np.testing.assert_array_equal(
        result.sample_prototype_weights["donor-a::sample-a"],
        original_batch.prototype_weights.numpy(),
    )


def test_refinement_defaults_require_two_broad_rounds_and_fixed_prior() -> None:
    config = RefinementConfig()
    config.validate()
    assert config.maximum_rounds == 4
    assert config.broad_refinement_rounds == 2
    assert config.teacher_ema == 0.0
    assert not config.require_view_agreement
    assert config.prior_old_weight == 1.0
    assert config.prior_new_weight == 0.0

    with pytest.raises(ValueError, match="0 for fine-only refinement or at least 2"):
        RefinementConfig(broad_refinement_rounds=1).validate()
    with pytest.raises(ValueError, match="leave at least two subsequent fine rounds"):
        RefinementConfig(maximum_rounds=3, broad_refinement_rounds=2).validate()

    sensitivity = RefinementConfig(prior_old_weight=0.80)
    sensitivity.validate()
    assert sensitivity.prior_new_weight == pytest.approx(0.20)


def test_default_fixed_prior_is_preserved_without_renormalization() -> None:
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
            abstain_threshold=1.0,
        )
    )
    trainer = _RecordingTrainer(model)
    batch = _batch(3, "bag0", 0.0)
    batch.prototype_weights = torch.tensor([2.0, 1.0])
    config = RefinementConfig(
        maximum_rounds=1,
        broad_refinement_rounds=0,
        min_probability=0.0,
        max_normalized_entropy=1.0,
        minimum_segmentation_confidence=0.0,
        require_view_agreement=False,
    )

    result = IterativeRefiner(lambda: trainer, config).fit([batch], [batch])

    np.testing.assert_array_equal(
        result.sample_prototype_weights["donor-a::sample-a"],
        np.asarray([2.0, 1.0], dtype=np.float32),
    )
    torch.testing.assert_close(
        trainer.calls[0][0].prototype_weights,
        torch.tensor([2.0, 1.0]),
    )
