from oat_drgrpo.learner.run import ZeroMathRunMixin
import pytest


class _ProgressMetricHarness(ZeroMathRunMixin):
    def __init__(self):
        self.steps = 0


def test_learning_progress_metrics_track_eval_and_rollout_uptick():
    learner = _ProgressMetricHarness()

    first_logs = {
        "eval/average/accuracy": 0.10,
        "eval/average/score": 0.20,
        "actor/rewards": 0.30,
    }
    learner._add_learning_progress_metrics(first_logs)

    assert first_logs["drx/progress/eval_accuracy/gain_from_start"] == 0.0
    assert first_logs["drx/progress/eval_score/gain_from_prev"] == 0.0
    assert first_logs["drx/progress/rollout_reward_ema"] == 0.30

    learner.steps = 16
    second_logs = {
        "eval/average/accuracy": 0.16,
        "eval/average/score": 0.24,
        "actor/rewards": 0.50,
    }
    learner._add_learning_progress_metrics(second_logs)

    assert second_logs["drx/progress/eval_accuracy/value"] == 0.16
    assert second_logs["drx/progress/eval_accuracy/gain_from_start"] == 0.06
    assert second_logs["drx/progress/eval_accuracy/gain_from_prev"] == 0.06
    assert second_logs["drx/progress/eval_accuracy/best"] == 0.16
    assert second_logs["drx/progress/eval_accuracy/best_step"] == 16
    assert second_logs["drx/progress/rollout_reward/gain_from_start"] == 0.20
    assert second_logs["drx/progress/rollout_reward_ema"] == 0.32
    assert second_logs[
        "drx/progress/rollout_reward_ema_gain_from_start"
    ] == pytest.approx(0.02)


def test_learning_progress_metrics_track_steps_since_best():
    learner = _ProgressMetricHarness()
    learner._add_learning_progress_metrics({"eval/average/accuracy": 0.20})

    learner.steps = 16
    learner._add_learning_progress_metrics({"eval/average/accuracy": 0.30})

    learner.steps = 32
    logs = {"eval/average/accuracy": 0.25}
    learner._add_learning_progress_metrics(logs)

    assert logs["drx/progress/eval_accuracy/best"] == 0.30
    assert logs["drx/progress/eval_accuracy/best_step"] == 16
    assert logs["drx/progress/eval_accuracy/steps_since_best"] == 16
