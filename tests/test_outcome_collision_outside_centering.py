from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from oat_drgrpo.learner.grpo import ZeroMathGrpoMixin


class _Policy:
    training = True

    def __call__(self, input_ids, *, attention_mask):
        del attention_mask
        return {
            "logits": torch.zeros(
                (*input_ids.shape, 8),
                dtype=torch.float32,
                device=input_ids.device,
            )
        }


class _CaptureLearner(ZeroMathGrpoMixin):
    def __init__(self, *, outside_centering: bool):
        self.args = SimpleNamespace(
            canonical_action_task="none",
            canonical_graph_actions=False,
            canonical_graph_action_count=3,
            critic_type="drgrpo",
            num_samples=16,
            reward_scale=1.0,
            prompt_template="qwen_boxed",
            outcome_collision_coef=0.1,
            outcome_collision_outside_centering=outside_centering,
            temperature=1.0,
            train_batch_size_per_device=16,
            xdr_tau=math.inf,
            seed_entropy_alpha=0.0,
            generate_max_length=8,
        )
        self.model = _Policy()
        self.ref_model = None
        self.captured = None

    def get_completion_mask(self, att_mask, prompt_id_lens):
        del prompt_id_lens
        mask = torch.zeros_like(att_mask, dtype=torch.bool)
        mask[:, 1] = True
        return mask

    def _seed_answer_keys_grouped(
        self,
        input_ids,
        response_masks,
        group_size,
        references_grouped=None,
    ):
        del input_ids, response_masks, references_grouped
        return [["same-answer"] * group_size]

    def _resolve_scoring_vocab_upper_bound(self, model):
        del model
        return 8

    def _sanitize_scoring_token_ids(self, input_ids, **kwargs):
        del kwargs
        return input_ids

    def _mask_invalid_scoring_logit_columns(self, logits, **kwargs):
        del kwargs
        return logits

    def _policy_logps_and_optional_entropy(
        self,
        logits,
        input_ids,
        response_masks,
        *,
        need_entropy,
    ):
        del logits, input_ids, need_entropy
        return torch.zeros_like(response_masks, dtype=torch.float32), None

    def _baseline_update_with_precomputed_advantages(self, **kwargs):
        self.captured = kwargs
        return kwargs["extra_infos"]


def _collapsed_trajectory():
    return {
        "input_ids": torch.zeros((16, 3), dtype=torch.long),
        "attention_mask": torch.ones((16, 3), dtype=torch.long),
        "rewards": [[1.0] for _ in range(16)],
        "prompt_ids_lens": [1] * 16,
        "loss_masks": [1.0] * 16,
        "references": [None] * 16,
    }


def test_e40_adds_collision_after_task_centering_once(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    learner = _CaptureLearner(outside_centering=True)

    infos = learner._grpo_learning_step_with_progress(_collapsed_trajectory())

    assert learner.captured is not None
    assert learner.captured["final_rewards"] == pytest.approx(
        torch.ones((16, 1))
    )
    assert learner.captured["advantages"] == pytest.approx(
        torch.full((16, 1), -0.09375)
    )
    assert infos["outcome_collision_outside_centering_active"].item() == 1.0
    assert infos["outcome_collision_reward_sent_to_centering_mean"].item() == 1.0
    assert infos["outcome_collision_augmented_reward_mean"].item() == pytest.approx(
        0.90625
    )
    assert (
        infos["outcome_collision_bonus_zero_spread_group_fraction"].item()
        == 1.0
    )
    assert infos["outcome_collision_centered_bonus_rms"].item() == 0.0
    assert infos["outcome_collision_outside_base_advantage_abs_mean"].item() == 0.0
    assert infos["outcome_collision_outside_base_advantage_rms"].item() == 0.0
    assert infos[
        "outcome_collision_outside_semantic_advantage_rms"
    ].item() == pytest.approx(0.09375)
    assert infos[
        "outcome_collision_outside_combined_advantage_rms"
    ].item() == pytest.approx(0.09375)
    assert (
        infos[
            "outcome_collision_outside_semantic_advantage_nonzero_fraction"
        ].item()
        == 1.0
    )
    assert (
        infos[
            "outcome_collision_outside_combined_advantage_nonzero_fraction"
        ].item()
        == 1.0
    )


def test_e37_still_centers_the_collision_reward(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    learner = _CaptureLearner(outside_centering=False)

    infos = learner._grpo_learning_step_with_progress(_collapsed_trajectory())

    assert learner.captured is not None
    assert learner.captured["advantages"] == pytest.approx(torch.zeros((16, 1)))
    assert infos["outcome_collision_outside_centering_active"].item() == 0.0
    assert (
        "outcome_collision_outside_semantic_advantage_nonzero_fraction"
        not in infos
    )
