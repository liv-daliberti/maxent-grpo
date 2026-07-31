from __future__ import annotations

import math

import pytest
import torch

from oat_drgrpo.interactive_episode_replay import (
    InteractiveDecisionRecord,
    InteractiveEpisodeRecord,
    VerifiedInteractiveReplayBank,
    fixed_replay_slots,
    interactive_transition_sha256,
    length_normalized_episode_scores,
    replay_gradient_surrogate,
)


def test_public_transition_hash_is_stable_and_action_sensitive():
    before = {
        "achieved_goal": [0.0, 1.0],
        "desired_goal": [2.0, 3.0],
        "velocity_xy": [0.1, -0.2],
        "remaining_actions": 96,
    }
    after = {
        "achieved_goal": [0.2, 1.1],
        "desired_goal": [2.0, 3.0],
        "velocity_xy": [0.3, -0.1],
        "remaining_actions": 95,
        "done": False,
        "success": False,
    }
    first = interactive_transition_sha256(before=before, action="N", after=after)
    assert first == interactive_transition_sha256(
        before=dict(reversed(list(before.items()))), action="n", after=after
    )
    assert first != interactive_transition_sha256(
        before=before, action="S", after=after
    )
    assert len(first) == 64


def _decision(action: int, suffix: str) -> InteractiveDecisionRecord:
    return InteractiveDecisionRecord(
        prompt_token_ids=(1, 2, 3),
        allowed_token_ids=(10, 11),
        selected_token_id=action,
        behavior_logprobs=(-math.log(2), -math.log(2)),
        transition_sha256=(suffix * 64)[:64],
    )


def _episode(key: str | None, actions: tuple[int, ...], *, reward: float = 1.0):
    return InteractiveEpisodeRecord(
        group_prompt_token_ids=(7, 8),
        outcome_key=key,
        task_reward=reward,
        decisions=tuple(
            _decision(action, chr(ord("a") + index))
            for index, action in enumerate(actions)
        ),
    )


def test_decision_rejects_support_escape_and_unnormalized_behavior_q():
    with pytest.raises(ValueError, match="selected token"):
        InteractiveDecisionRecord(
            prompt_token_ids=(1,),
            allowed_token_ids=(10, 11),
            selected_token_id=12,
            behavior_logprobs=(-math.log(2), -math.log(2)),
            transition_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="normalize"):
        InteractiveDecisionRecord(
            prompt_token_ids=(1,),
            allowed_token_ids=(10, 11),
            selected_token_id=10,
            behavior_logprobs=(-0.1, -0.1),
            transition_sha256="a" * 64,
        )


def test_replay_bank_admits_only_verified_positive_episodes_and_keeps_minimum():
    bank = VerifiedInteractiveReplayBank(capacity=16)
    longer = _episode("route-a", (11, 11))
    minimum = _episode("route-a", (10, 11))
    bank.observe_group(
        [
            longer,
            minimum,
            _episode(None, (10,), reward=1.0),
            _episode("wrong", (10,), reward=0.0),
        ]
    )

    group = bank.schedule_one_global_round_robin()

    assert group is not None
    assert group.outcome_keys == ("route-a",)
    assert group.episodes[0].action_token_ids == (10, 11)
    assert bank.tracked_prompt_count == 1
    assert bank.tracked_outcome_count == 1


def test_global_replay_scheduler_is_sorted_round_robin_and_state_exact():
    bank = VerifiedInteractiveReplayBank(capacity=16)
    first = _episode("route-a", (10,))
    second = InteractiveEpisodeRecord(
        group_prompt_token_ids=(9,),
        outcome_key="route-z",
        task_reward=1.0,
        decisions=(_decision(11, "c"),),
    )
    bank.observe_group([second])
    bank.observe_group([first])

    scheduled = [bank.schedule_one_global_round_robin() for _ in range(3)]
    assert [group.group_prompt_token_ids for group in scheduled] == [
        (7, 8),
        (9,),
        (7, 8),
    ]

    restored = VerifiedInteractiveReplayBank(capacity=16)
    restored.load_state_dict(bank.state_dict())
    assert restored.state_dict() == bank.state_dict()
    assert restored.schedule_one_global_round_robin().group_prompt_token_ids == (9,)


def test_fixed_replay_slots_pad_to_capacity_without_scientific_rows():
    bank = VerifiedInteractiveReplayBank(capacity=16)
    bank.observe_group([_episode("route-a", (10,)), _episode("route-b", (11,))])
    group = bank.schedule_one_global_round_robin()

    slots, active = fixed_replay_slots(group, slot_count=16)

    assert len(slots) == len(active) == 16
    assert active == (True, True) + (False,) * 14
    assert slots[:2] == group.episodes
    assert all(slot is None for slot in slots[2:])


def test_length_normalized_scores_ignore_padding_and_do_not_upweight_length():
    logprobs = torch.tensor([[-1.0, 0.0, 0.0], [-2.0, -2.0, -2.0]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)

    scores = length_normalized_episode_scores(logprobs, mask)

    torch.testing.assert_close(scores, torch.tensor([-1.0, -2.0]))


def test_compute_only_replay_traversal_has_exact_zero_parameter_derivative():
    live_scores = torch.tensor([-1.0, -2.0], requires_grad=True)
    raw_gradients = torch.tensor([0.25, -0.25])

    control = replay_gradient_surrogate(
        live_scores,
        raw_score_gradients=raw_gradients,
        compute_only=True,
        scale=15.0 / 256.0,
    )
    control.backward()
    torch.testing.assert_close(live_scores.grad, torch.zeros_like(live_scores))

    treatment_scores = torch.tensor([-1.0, -2.0], requires_grad=True)
    treatment = replay_gradient_surrogate(
        treatment_scores,
        raw_score_gradients=raw_gradients,
        compute_only=False,
        scale=15.0 / 256.0,
    )
    treatment.backward()
    torch.testing.assert_close(
        treatment_scores.grad,
        raw_gradients * (15.0 / 256.0),
    )
