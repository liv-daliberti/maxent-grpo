from __future__ import annotations

import pytest
import torch
from pathlib import Path

from oat_drgrpo.ucpo import redistribute_ucpo_advantages


ROOT = Path(__file__).resolve().parents[1]


def apply(advantages, logps, rewards, *, tau=0.2, masks=None):
    if masks is None:
        masks = [1.0] * len(advantages)
    return redistribute_ucpo_advantages(
        torch.tensor(advantages, dtype=torch.float64),
        torch.tensor(logps, dtype=torch.float64),
        torch.tensor(rewards, dtype=torch.float64),
        torch.tensor(masks, dtype=torch.float64),
        num_samples=4,
        tau=tau,
    )


def test_tau_zero_is_exact_identity():
    source = [0.5, 0.5, -0.5, -0.5]
    result, diagnostics = apply(
        source, [-1.0, -4.0, -2.0, -2.0], [1, 1, 0, 0], tau=0
    )
    assert result.tolist() == source
    assert diagnostics.eligible_groups == 1


def test_rare_correct_response_receives_more_mass_and_total_is_preserved():
    result, diagnostics = apply(
        [0.5, 0.5, -0.5, -0.5],
        [-1.0, -4.0, -2.0, -2.0],
        [1, 1, 0, 0],
        tau=0.2,
    )
    assert result[1] > result[0]
    assert result[:2].sum().item() == pytest.approx(1.0)
    assert result[2:].tolist() == [-0.5, -0.5]
    assert diagnostics.correct_rows == 2
    assert diagnostics.mass_error_max < 1e-12


def test_all_correct_zero_advantage_stays_zero():
    result, _ = apply([0, 0, 0, 0], [-1, -2, -3, -4], [1, 1, 1, 1])
    assert result.tolist() == [0, 0, 0, 0]


def test_loss_masked_correct_row_is_not_in_target():
    result, diagnostics = apply(
        [0.5, 0.5, -0.5, -0.5],
        [-1, -4, -2, -2],
        [1, 1, 0, 0],
        masks=[1, 0, 1, 1],
    )
    assert result.tolist() == [0.5, 0.5, -0.5, -0.5]
    assert diagnostics.correct_rows == 1


def test_rejects_nonshared_correct_advantage():
    with pytest.raises(ValueError, match="shared base advantage"):
        apply([0.4, 0.6, -0.5, -0.5], [-1, -2, -2, -2], [1, 1, 0, 0])


@pytest.mark.parametrize("tau", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_tau(tau):
    with pytest.raises(ValueError, match="tau"):
        apply(
            [0.5, 0.5, -0.5, -0.5],
            [-1, -2, -2, -2],
            [1, 1, 0, 0],
            tau=tau,
        )


def test_runner_captures_then_isolates_ucpo_and_rlep_switches():
    source = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    assert 'UCPO_TAU="${OAT_ZERO_UCPO_TAU:-0.2}"' in source
    assert 'RLEP_EXPERIENCE_ROOT="${OAT_ZERO_RLEP_EXPERIENCE_ROOT:-}"' in source
    assert 'RLEP_REPLAY_COUNT="${OAT_ZERO_RLEP_REPLAY_COUNT:-2}"' in source
    assert "export OAT_ZERO_CRITIC_TYPE=drgrpo" in source
    assert 'RLEP_SPARSE_FALLBACK="${OAT_ZERO_RLEP_SPARSE_FALLBACK:-0}"' in source
    assert 'export OAT_ZERO_UCPO_TAU="$UCPO_TAU"' in source
    assert 'export OAT_ZERO_RLEP_EXPERIENCE_ROOT="$RLEP_EXPERIENCE_ROOT"' in source
    assert 'export OAT_ZERO_RLEP_REPLAY_COUNT="$RLEP_REPLAY_COUNT"' in source
    assert 'export OAT_ZERO_RLEP_SPARSE_FALLBACK="$RLEP_SPARSE_FALLBACK"' in source
