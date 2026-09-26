from __future__ import annotations

from pathlib import Path

import pytest
import torch

from oat_drgrpo.dapo import (
    dapo_group_diagnostics,
    dapo_resample_prompt_index,
    dapo_soft_overlong_penalty,
    dapo_token_level_policy_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def test_group_filter_counts_constant_and_eligible_groups():
    diagnostics = dapo_group_diagnostics(
        [0] * 16 + [1] * 16 + [0] * 8 + [1] * 8,
        num_samples=16,
    )
    assert diagnostics.groups == 3
    assert diagnostics.eligible_groups == 1
    assert diagnostics.all_zero_groups == 1
    assert diagnostics.all_one_groups == 1


@pytest.mark.parametrize(
    "rewards, message",
    [([0] * 15, "complete"), ([0] * 15 + [0.5], "binary")],
)
def test_group_filter_fails_closed_on_invalid_reward_contract(rewards, message):
    with pytest.raises(ValueError, match=message):
        dapo_group_diagnostics(rewards, num_samples=16)


def test_resample_index_is_deterministic_and_bounded():
    first = dapo_resample_prompt_index(
        base_seed=43, learner_step=7, generation_batch=2, dataset_size=384
    )
    second = dapo_resample_prompt_index(
        base_seed=43, learner_step=7, generation_batch=2, dataset_size=384
    )
    assert first == second
    assert 0 <= first < 384
    assert first != dapo_resample_prompt_index(
        base_seed=43, learner_step=7, generation_batch=3, dataset_size=384
    )


def test_soft_overlong_penalty_scales_last_twenty_percent():
    penalties, diagnostics = dapo_soft_overlong_penalty(
        torch.tensor([0, 80, 90, 100]),
        max_length=100,
        buffer_ratio=0.20,
        penalty_factor=1.0,
    )
    assert penalties.tolist() == pytest.approx([0.0, 0.0, -0.5, -1.0])
    assert diagnostics.shaped_rows == 2
    assert diagnostics.truncated_rows == 1


def test_token_level_reduction_weights_tokens_not_sequences():
    losses = torch.tensor([[2.0, 2.0], [8.0, 0.0]])
    response_masks = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    result, denominator = dapo_token_level_policy_loss(
        losses,
        response_masks,
        torch.tensor([1.0, 1.0]),
    )
    assert result.item() == pytest.approx(4.0)
    assert denominator.item() == 3.0


def test_shell_contract_exposes_full_isolated_dapo_recipe():
    runner = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    trainer = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    run_loop = (ROOT / "src/oat_drgrpo/learner/run.py").read_text(encoding="utf-8")
    assert "  dapo)" in runner
    assert "export OAT_ZERO_CRITIC_TYPE=grpo" in runner
    assert "export OAT_ZERO_DAPO_ENABLED=1" in runner
    assert "export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=0" in runner
    assert "--dapo-clip-low" in trainer
    assert "--dapo-clip-high" in trainer
    assert "--dapo-max-num-gen-batches" in trainer
    assert "--dapo-overlong-buffer-ratio" in trainer
    assert "def _collect_dapo_dynamic_feedback(" in run_loop
    assert "self.query_step += len(feedback_data)" in run_loop
    assert "self.prompt_consumed += len(feedback_data)" in run_loop
    assert 'elif bool(getattr(self.args, "dapo_enabled", False)):' in run_loop
