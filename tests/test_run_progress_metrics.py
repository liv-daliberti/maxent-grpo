from types import SimpleNamespace

from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.canonical_replay import (
    CanonicalReplayInverseController,
    CanonicalReplayLikelihoodController,
)
from oat_drgrpo.maxent_controllers import MaxEntDualController
from oat_drgrpo.maxent_length_controller import MaxEntLengthController
from oat_drgrpo.xdr_tau_controller import XdrTauController
from oat_drgrpo.xdr_sac_dual_controller import XdrSacDualController
import pytest


class _ProgressMetricHarness(ZeroMathRunMixin):
    def __init__(self):
        self.steps = 0


class _AveragingStrategy:
    def __init__(self, global_entropy):
        self.global_entropy = global_entropy
        self.calls = []

    def all_reduce(self, values):
        self.calls.append(values)
        return {next(iter(values)): self.global_entropy}


class _IdentityReduceStrategy:
    def __init__(self):
        self.calls = []

    def all_reduce(self, values):
        self.calls.append(dict(values))
        return dict(values)


class _TerminalEvalHarness(ZeroMathRunMixin):
    def __init__(self):
        self.args = SimpleNamespace(
            eval_steps=768,
            export_steps=-1,
            resume_steps=-1,
            save_ckpt=False,
            logging_steps=997,
        )
        self.steps = 3840
        self.global_step = 3840
        self.eval_prompts_dataloader = object()
        self.evaluated_steps = []

    def evaluate(self, _dataloader, steps):
        self.evaluated_steps.append(steps)
        return {}


def test_learning_progress_metrics_track_eval_and_rollout_uptick():
    learner = _ProgressMetricHarness()

    first_logs = {
        "eval/average/accuracy": 0.10,
        "eval/average/score": 0.20,
        "actor/rewards": 0.30,
    }
    learner._add_learning_progress_metrics(first_logs)

    assert first_logs["xdr/progress/eval_accuracy/gain_from_start"] == 0.0
    assert first_logs["xdr/progress/eval_score/gain_from_prev"] == 0.0
    assert first_logs["xdr/progress/rollout_reward_ema"] == 0.30

    learner.steps = 16
    second_logs = {
        "eval/average/accuracy": 0.16,
        "eval/average/score": 0.24,
        "actor/rewards": 0.50,
    }
    learner._add_learning_progress_metrics(second_logs)

    assert second_logs["xdr/progress/eval_accuracy/value"] == 0.16
    assert second_logs["xdr/progress/eval_accuracy/gain_from_start"] == 0.06
    assert second_logs["xdr/progress/eval_accuracy/gain_from_prev"] == 0.06
    assert second_logs["xdr/progress/eval_accuracy/best"] == 0.16
    assert second_logs["xdr/progress/eval_accuracy/best_step"] == 16
    assert second_logs["xdr/progress/rollout_reward/gain_from_start"] == 0.20
    assert second_logs["xdr/progress/rollout_reward_ema"] == 0.32
    assert second_logs[
        "xdr/progress/rollout_reward_ema_gain_from_start"
    ] == pytest.approx(0.02)


def test_terminal_eval_skips_unchanged_policy_after_scheduled_endpoint():
    learner = _TerminalEvalHarness()

    learner.eval_and_log({}, eval=True, save=False)
    learner.steps = 3841
    learner.eval_and_log({}, eval=True, save=True)

    assert learner.evaluated_steps == [3840]


def test_terminal_eval_runs_when_policy_changed_since_last_evaluation():
    learner = _TerminalEvalHarness()

    learner.eval_and_log({}, eval=True, save=False)
    learner.steps = 3841
    learner.global_step += 1
    learner.eval_and_log({}, eval=True, save=True)

    assert learner.evaluated_steps == [3840, 3841]


def test_learning_progress_metrics_track_steps_since_best():
    learner = _ProgressMetricHarness()
    learner._add_learning_progress_metrics({"eval/average/accuracy": 0.20})

    learner.steps = 16
    learner._add_learning_progress_metrics({"eval/average/accuracy": 0.30})

    learner.steps = 32
    logs = {"eval/average/accuracy": 0.25}
    learner._add_learning_progress_metrics(logs)

    assert logs["xdr/progress/eval_accuracy/best"] == 0.30
    assert logs["xdr/progress/eval_accuracy/best_step"] == 16
    assert logs["xdr/progress/eval_accuracy/steps_since_best"] == 16


def test_tau_controller_uses_distributed_entropy_for_next_update():
    learner = _ProgressMetricHarness()
    learner.strategy = _AveragingStrategy(global_entropy=0.25)
    learner._xdr_tau_controller = XdrTauController(
        base_tau=0.05,
        min_tau=0.005,
        target_ratio=0.8,
        warmup_steps=1,
        ema_decay=0.0,
        gain=20.0,
    )

    train_info = {"entropy": 0.10}
    learner._update_xdr_tau_controller(train_info)

    assert learner.strategy.calls == [
        {"xdr_tau_control_observed_entropy": 0.10}
    ]
    assert train_info["xdr_tau_control_target_entropy"] == pytest.approx(0.20)
    assert train_info["xdr_tau_control_next_tau"] == pytest.approx(0.05)


def test_tau_controller_fails_closed_without_entropy_metric():
    learner = _ProgressMetricHarness()
    learner.strategy = _AveragingStrategy(global_entropy=0.25)
    learner._xdr_tau_controller = XdrTauController(
        base_tau=0.05,
        min_tau=0.005,
        target_ratio=0.8,
        warmup_steps=1,
        ema_decay=0.0,
        gain=20.0,
    )

    with pytest.raises(RuntimeError, match="train/entropy"):
        learner._update_xdr_tau_controller({})


def test_sac_dual_controller_uses_its_distinct_distributed_metric():
    learner = _ProgressMetricHarness()
    learner.strategy = _AveragingStrategy(global_entropy=0.25)
    learner._xdr_tau_controller = XdrSacDualController(
        base_tau=0.05,
        min_tau=0.005,
        max_tau=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        alpha_lr=0.1,
        beta1=0.0,
        beta2=0.0,
    )

    train_info = {"entropy": 0.10}
    learner._update_xdr_tau_controller(train_info)

    assert learner.strategy.calls == [{"xdr_sac_dual_observed_entropy": 0.10}]
    assert train_info["xdr_sac_dual_target_entropy"] == pytest.approx(0.20)
    assert train_info["xdr_sac_dual_next_tau"] > 0.05


def test_maxent_dual_uses_the_same_sequence_entropy_as_the_actor_objective():
    learner = _ProgressMetricHarness()
    learner.strategy = _AveragingStrategy(global_entropy=0.25)
    learner._maxent_alpha_controller = MaxEntDualController(
        base_alpha=0.05,
        min_alpha=0.005,
        max_alpha=0.5,
        target_ratio=0.8,
        warmup_steps=1,
        alpha_lr=0.1,
        beta1=0.0,
        beta2=0.0,
        # Disable EMA smoothing so one post-warmup observation determines the
        # descent direction. At the default decay the EMA of (0.25, 0.10) is
        # 0.205, still above the 0.20 target, and alpha would correctly fall.
        ema_decay=0.0,
    )

    train_info = {"maxent_sequence_entropy": 0.10}
    learner._update_maxent_alpha_controller(train_info)

    assert learner.strategy.calls == [{"maxent_sequence_entropy": 0.10}]
    assert train_info["maxent_dual_target_entropy"] == pytest.approx(0.20)
    assert train_info["maxent_dual_next_alpha"] == pytest.approx(0.05)

    learner.strategy = _AveragingStrategy(global_entropy=0.10)
    train_info = {"maxent_sequence_entropy": 0.10}
    learner._update_maxent_alpha_controller(train_info)

    # The distributed entropy (0.10) now sits below the warmup-frozen target
    # (0.8 * 0.25), so dual descent raises alpha to buy entropy back.
    assert train_info["maxent_dual_target_entropy"] == pytest.approx(0.20)
    assert train_info["maxent_dual_entropy_error"] == pytest.approx(-0.10)
    assert train_info["maxent_dual_next_alpha"] > 0.05


def test_maxent_length_controller_uses_distributed_expected_length():
    learner = _ProgressMetricHarness()
    learner.strategy = _AveragingStrategy(global_entropy=24.0)
    learner._maxent_length_controller = MaxEntLengthController(
        target_length=16.0,
        init_lambda=0.0,
        max_lambda=0.02,
        dual_lr=0.002,
        ema_decay=0.0,
    )

    train_info = {"maxent_expected_length": 20.0}
    learner._update_maxent_length_controller(train_info)

    assert learner.strategy.calls == [{"maxent_expected_length": 20.0}]
    assert train_info["maxent_length_observed_length"] == pytest.approx(24.0)
    assert train_info["maxent_length_relative_violation"] == pytest.approx(0.5)
    assert train_info["maxent_length_lambda_next"] == pytest.approx(0.001)


def test_maxent_length_controller_fails_without_expected_length_metric():
    learner = _ProgressMetricHarness()
    learner.strategy = _AveragingStrategy(global_entropy=24.0)
    learner._maxent_length_controller = MaxEntLengthController(
        target_length=16.0,
        init_lambda=0.0,
        max_lambda=0.02,
        dual_lr=0.002,
        ema_decay=0.0,
    )

    with pytest.raises(RuntimeError, match="expected-length"):
        learner._update_maxent_length_controller({})


def test_canonical_replay_controller_uses_only_observed_model_score_entropy():
    learner = _ProgressMetricHarness()
    learner.strategy = _IdentityReduceStrategy()
    learner._canonical_replay_controller = CanonicalReplayInverseController(
        base_alpha=0.1,
        warmup_steps=1,
        ema_decay=0.0,
    )

    first = {
        "canonical_replay_normalized_model_entropy": 0.8,
        "canonical_replay_eligible_groups": 1.0,
    }
    learner._update_canonical_replay_controller(first)
    assert first["canonical_replay_next_alpha"] == pytest.approx(0.1)

    collapsed = {
        "canonical_replay_normalized_model_entropy": 0.08,
        "canonical_replay_eligible_groups": 1.0,
    }
    learner._update_canonical_replay_controller(collapsed)
    assert collapsed["canonical_replay_next_alpha"] == pytest.approx(1.0)
    assert collapsed["canonical_replay_projection_active"] == 0.0
    assert learner.strategy.calls[-1] == {
        "canonical_replay_entropy_weighted": 0.08,
        "canonical_replay_eligibility_weight": 1.0,
    }

    idle = {}
    observations = learner._canonical_replay_controller.observation_count
    learner._update_canonical_replay_controller(idle)
    assert idle["canonical_replay_observation_skipped"] == 1.0
    assert (
        learner._canonical_replay_controller.observation_count
        == observations
    )


def test_canonical_replay_mass_controller_uses_its_own_verified_surprisal():
    learner = _ProgressMetricHarness()
    learner.strategy = _IdentityReduceStrategy()
    learner._canonical_replay_mass_controller = (
        CanonicalReplayLikelihoodController(
            base_alpha=0.1,
            warmup_steps=1,
            ema_decay=0.0,
        )
    )

    first = {
        "canonical_replay_actuator_loss": 2.0,
        "canonical_replay_actuator_groups": 1.0,
    }
    learner._update_canonical_replay_mass_controller(first)
    assert first["canonical_replay_mass_next_alpha"] == pytest.approx(0.1)

    degraded = {
        "canonical_replay_actuator_loss": 8.0,
        "canonical_replay_actuator_groups": 1.0,
    }
    learner._update_canonical_replay_mass_controller(degraded)
    assert degraded["canonical_replay_mass_next_alpha"] == pytest.approx(0.4)
    assert degraded["canonical_replay_mass_projection_active"] == 0.0
    assert learner.strategy.calls[-1] == {
        "canonical_replay_mass_surprisal_weighted": 8.0,
        "canonical_replay_mass_eligibility_weight": 1.0,
    }


def test_length_controller_resume_presence_must_match_checkpoint():
    learner = _ProgressMetricHarness()
    learner._xdr_tau_controller = None
    learner._maxent_alpha_controller = None
    learner._maxent_length_controller = MaxEntLengthController(
        target_length=16.0,
        init_lambda=0.0,
        max_lambda=0.02,
        dual_lr=0.002,
        ema_decay=0.9,
    )

    with pytest.raises(ValueError, match="missing length-controller state"):
        learner._restore_training_progress_state({})

    state = learner._maxent_length_controller.state_dict()
    learner._maxent_length_controller = None
    with pytest.raises(ValueError, match="unconstrained run"):
        learner._restore_training_progress_state(
            {"maxent_length_controller_state": state}
        )
