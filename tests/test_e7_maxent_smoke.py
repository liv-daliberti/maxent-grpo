from __future__ import annotations

import pytest

from exp_scaling.check_e7_maxent_smoke import inspect_rows


def _rows(*, terminal_length: float = 12.0, reward: float = 0.25):
    rows = []
    for step in range(1, 129):
        rows.append(
            {
                "trainer/global_step": step,
                "train/entropy": 0.2,
                "train/pg_loss": 0.1,
                "train/policy_grad_norm": 1.0,
                "train/candidate_maxent_projection_loss": 0.1,
                "train/xdr_tau_control_observations": step,
                "actor/response_tok_len": terminal_length,
                "actor/rewards": reward if step == 120 else 0.0,
            }
        )
    return rows


def test_e7_smoke_accepts_noncollapsed_rewarded_run():
    summary = inspect_rows(
        _rows(), arm="xdr_maxent_tau_control", expected_step=128
    )
    assert summary["max_step"] == 128


def test_e7_smoke_rejects_two_token_collapse():
    with pytest.raises(RuntimeError, match="EOS collapse"):
        inspect_rows(_rows(terminal_length=2.0), arm="xdr_maxent", expected_step=128)


def test_e7_smoke_rejects_zero_reward_tail():
    with pytest.raises(RuntimeError, match="no nonzero reward"):
        inspect_rows(_rows(reward=0.0), arm="xdr_maxent", expected_step=128)
