from __future__ import annotations

import pytest

from exp_scaling.check_e9_direct_maxent_smoke import inspect_rows


def _rows(*, terminal_length: float = 12.0, reward: float = 0.25):
    rows = []
    for step in range(1, 129):
        rows.append(
            {
                "trainer/global_step": step,
                "train/entropy": 0.2,
                "train/pg_loss": 0.1,
                "train/policy_grad_norm": 1.0,
                "train/maxent_alpha_used": 0.05,
                "train/maxent_sequence_entropy": 0.3,
                "train/maxent_entropy_surrogate": -0.2,
                "train/maxent_causal_score_term": -0.5,
                "train/maxent_entropy_loss": 0.00005,
                "train/maxent_reward_estimator_scale": 15.0 / 16.0,
                "train/maxent_control_observations": step,
                "train/maxent_control_target_entropy": 0.24,
                "train/maxent_dual_target_entropy": 0.24,
                "actor/response_tok_len": terminal_length,
                "actor/rewards": reward if step == 120 else 0.0,
            }
        )
    return rows


def test_e9_smoke_accepts_noncollapsed_rewarded_run():
    summary = inspect_rows(_rows(), arm="maxent_control", expected_step=128)
    assert summary["max_step"] == 128


def test_e9_smoke_rejects_missing_direct_gradient_telemetry():
    rows = _rows()
    del rows[-1]["train/maxent_entropy_loss"]
    with pytest.raises(RuntimeError, match="maxent_entropy_loss"):
        inspect_rows(rows, arm="maxent", expected_step=128)


def test_e9_smoke_rejects_wrong_reward_estimator_scale():
    rows = _rows()
    rows[-1]["train/maxent_reward_estimator_scale"] = 1.0
    with pytest.raises(RuntimeError, match="estimator scale"):
        inspect_rows(rows, arm="maxent", expected_step=128)


def test_e9_smoke_rejects_two_token_collapse():
    with pytest.raises(RuntimeError, match="EOS collapse"):
        inspect_rows(_rows(terminal_length=2.0), arm="maxent", expected_step=128)


def test_e9_smoke_rejects_zero_reward_tail():
    with pytest.raises(RuntimeError, match="no nonzero reward"):
        inspect_rows(_rows(reward=0.0), arm="maxent", expected_step=128)


def test_e9_smoke_rejects_controller_that_misses_its_entropy_target():
    rows = _rows()
    for row in rows[-16:]:
        row["train/maxent_sequence_entropy"] = 0.05
    with pytest.raises(RuntimeError, match="actuator guard failed"):
        inspect_rows(rows, arm="maxent_control", expected_step=128)
