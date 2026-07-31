"""Length-neutral episodic policy objectives for interactive ModeBench.

Environment observations between actions are context, not policy targets.
This module operates on already selected action-token log probabilities and
averages decisions within each episode before averaging episodes.
"""

from __future__ import annotations

import torch


def drgrpo_task_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Center terminal rewards within each prompt without variance scaling.

    ``rewards`` has shape ``[prompt_count, samples_per_prompt]``.
    """

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must be [prompts, at least two samples]")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")
    return rewards - rewards.mean(dim=1, keepdim=True)


def add_verified_advantage_outside_centering(
    task_advantages: torch.Tensor,
    verified_advantages: torch.Tensor,
) -> torch.Tensor:
    """Add detached verified-MaxEnt advantages after task centering."""

    if task_advantages.shape != verified_advantages.shape:
        raise ValueError("task and verified advantages must have identical shapes")
    if not bool(torch.isfinite(task_advantages).all()) or not bool(
        torch.isfinite(verified_advantages).all()
    ):
        raise ValueError("advantages must be finite")
    return task_advantages + verified_advantages.detach()


def length_neutral_clipped_episode_loss(
    *,
    new_action_logprobs: torch.Tensor,
    behavior_action_logprobs: torch.Tensor,
    decision_mask: torch.Tensor,
    episode_advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Return a PPO-style loss with equal statistical weight per episode.

    Log-probability tensors and ``decision_mask`` have shape
    ``[episode_count, max_decisions]``. Only model-selected action positions
    are present. The surrogate is averaged over active decisions in each
    episode and then over episodes.
    """

    if (
        new_action_logprobs.shape != behavior_action_logprobs.shape
        or new_action_logprobs.shape != decision_mask.shape
        or new_action_logprobs.ndim != 2
    ):
        raise ValueError("log probabilities and mask must share a 2D shape")
    if episode_advantages.ndim != 1 or episode_advantages.shape[0] != (
        new_action_logprobs.shape[0]
    ):
        raise ValueError("one episode advantage is required per row")
    if not 0.0 < float(clip_epsilon) < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if not bool(decision_mask.to(torch.bool).any(dim=1).all()):
        raise ValueError("every episode must contain at least one decision")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            new_action_logprobs,
            behavior_action_logprobs,
            episode_advantages,
        )
    ):
        raise ValueError("objective inputs must be finite")

    mask = decision_mask.to(dtype=new_action_logprobs.dtype)
    ratio = torch.exp(new_action_logprobs - behavior_action_logprobs)
    advantage = episode_advantages[:, None]
    unclipped = ratio * advantage
    clipped = torch.clamp(
        ratio,
        1.0 - float(clip_epsilon),
        1.0 + float(clip_epsilon),
    ) * advantage
    surrogate = torch.minimum(unclipped, clipped)
    per_episode = (surrogate * mask).sum(dim=1) / mask.sum(dim=1)
    return -per_episode.mean()
