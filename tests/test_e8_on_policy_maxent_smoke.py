from __future__ import annotations

import pytest

from exp_scaling.check_e8_on_policy_maxent_smoke import inspect_rows


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
                "train/maxent_entropy_advantage_abs_mean": 0.01,
                "train/maxent_control_observations": step,
                "actor/response_tok_len": terminal_length,
                "actor/rewards": reward if step == 120 else 0.0,
            }
        )
    return rows


def test_e8_smoke_accepts_noncollapsed_rewarded_run():
    summary = inspect_rows(_rows(), arm="maxent_control", expected_step=128)
    assert summary["max_step"] == 128


def test_e8_smoke_rejects_missing_direct_entropy_telemetry():
    rows = _rows()
    del rows[-1]["train/maxent_sequence_entropy"]
    with pytest.raises(RuntimeError, match="maxent_sequence_entropy"):
        inspect_rows(rows, arm="maxent", expected_step=128)


def test_e8_smoke_rejects_two_token_collapse():
    with pytest.raises(RuntimeError, match="EOS collapse"):
        inspect_rows(_rows(terminal_length=2.0), arm="maxent", expected_step=128)


def test_e8_smoke_rejects_zero_reward_tail():
    with pytest.raises(RuntimeError, match="no nonzero reward"):
        inspect_rows(_rows(reward=0.0), arm="maxent", expected_step=128)
