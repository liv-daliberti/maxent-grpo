"""Verified episode replay primitives for interactive ModeBench tasks.

Interactive episodes cannot be replayed as static language-model responses:
each selected action was conditioned on a public environment state.  The
records below therefore retain every public prompt/support/action decision and
make replay score the current policy at those same decision states.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch


def interactive_transition_sha256(
    *,
    before: Mapping[str, Any],
    action: str,
    after: Mapping[str, Any],
) -> str:
    """Hash one public interactive transition for independent replay audits."""

    def public_state(value: Mapping[str, Any]) -> dict[str, Any]:
        required = ("achieved_goal", "desired_goal", "velocity_xy", "remaining_actions")
        if any(key not in value for key in required):
            raise ValueError("interactive transition lacks a public state field")
        return {
            "achieved_goal": [float(item) for item in value["achieved_goal"]],
            "desired_goal": [float(item) for item in value["desired_goal"]],
            "velocity_xy": [float(item) for item in value["velocity_xy"]],
            "remaining_actions": int(value["remaining_actions"]),
            "done": bool(value.get("done", False)),
            "success": bool(value.get("success", False)),
        }

    normalized_action = str(action).upper()
    if not normalized_action:
        raise ValueError("interactive transition action is empty")
    encoded = json.dumps(
        {
            "schema": "public-interactive-transition-v1",
            "before": public_state(before),
            "action": normalized_action,
            "after": public_state(after),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _token_tuple(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain nonnegative token IDs")
    return result


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class InteractiveDecisionRecord:
    """One model-selected action at one immutable public environment state."""

    prompt_token_ids: tuple[int, ...]
    allowed_token_ids: tuple[int, ...]
    selected_token_id: int
    behavior_logprobs: tuple[float, ...]
    transition_sha256: str

    def __post_init__(self) -> None:
        prompt = _token_tuple(self.prompt_token_ids, name="prompt_token_ids")
        support = _token_tuple(self.allowed_token_ids, name="allowed_token_ids")
        if len(set(support)) != len(support):
            raise ValueError("allowed token IDs must be unique")
        selected = int(self.selected_token_id)
        if selected not in support:
            raise ValueError("selected token must belong to the recorded support")
        logprobs = tuple(float(value) for value in self.behavior_logprobs)
        if len(logprobs) != len(support) or not all(
            math.isfinite(value) for value in logprobs
        ):
            raise ValueError("behavior log probabilities must be finite and match support")
        normalizer = math.log(sum(math.exp(value) for value in logprobs))
        if not math.isclose(normalizer, 0.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("behavior log probabilities must normalize over support")
        transition = str(self.transition_sha256)
        if not _is_sha256(transition):
            raise ValueError("transition_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "allowed_token_ids", support)
        object.__setattr__(self, "selected_token_id", selected)
        object.__setattr__(self, "behavior_logprobs", logprobs)
        object.__setattr__(self, "transition_sha256", transition)

    @property
    def selected_behavior_logprob(self) -> float:
        return self.behavior_logprobs[
            self.allowed_token_ids.index(self.selected_token_id)
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "allowed_token_ids": list(self.allowed_token_ids),
            "selected_token_id": self.selected_token_id,
            "behavior_logprobs": list(self.behavior_logprobs),
            "transition_sha256": self.transition_sha256,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "InteractiveDecisionRecord":
        return cls(
            prompt_token_ids=tuple(state["prompt_token_ids"]),
            allowed_token_ids=tuple(state["allowed_token_ids"]),
            selected_token_id=int(state["selected_token_id"]),
            behavior_logprobs=tuple(state["behavior_logprobs"]),
            transition_sha256=str(state["transition_sha256"]),
        )


@dataclass(frozen=True)
class InteractiveEpisodeRecord:
    """A terminal interactive episode and its verifier-only outcome key."""

    group_prompt_token_ids: tuple[int, ...]
    outcome_key: str | None
    task_reward: float
    decisions: tuple[InteractiveDecisionRecord, ...]

    def __post_init__(self) -> None:
        group_prompt = _token_tuple(
            self.group_prompt_token_ids, name="group_prompt_token_ids"
        )
        decisions = tuple(self.decisions)
        if not decisions or not all(
            isinstance(decision, InteractiveDecisionRecord) for decision in decisions
        ):
            raise ValueError("an interactive episode requires recorded decisions")
        reward = float(self.task_reward)
        if not math.isfinite(reward):
            raise ValueError("task reward must be finite")
        key = self.outcome_key
        if key is not None and (not isinstance(key, str) or not key):
            raise ValueError("outcome key must be a nonempty string or None")
        object.__setattr__(self, "group_prompt_token_ids", group_prompt)
        object.__setattr__(self, "task_reward", reward)
        object.__setattr__(self, "decisions", decisions)

    @property
    def action_token_ids(self) -> tuple[int, ...]:
        return tuple(decision.selected_token_id for decision in self.decisions)

    @property
    def transition_hashes(self) -> tuple[str, ...]:
        return tuple(decision.transition_sha256 for decision in self.decisions)

    def state_dict(self) -> dict[str, Any]:
        return {
            "group_prompt_token_ids": list(self.group_prompt_token_ids),
            "outcome_key": self.outcome_key,
            "task_reward": self.task_reward,
            "decisions": [decision.state_dict() for decision in self.decisions],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "InteractiveEpisodeRecord":
        return cls(
            group_prompt_token_ids=tuple(state["group_prompt_token_ids"]),
            outcome_key=state.get("outcome_key"),
            task_reward=float(state["task_reward"]),
            decisions=tuple(
                InteractiveDecisionRecord.from_state_dict(decision)
                for decision in state["decisions"]
            ),
        )


@dataclass(frozen=True)
class InteractiveReplayGroup:
    group_prompt_token_ids: tuple[int, ...]
    outcome_keys: tuple[str, ...]
    episodes: tuple[InteractiveEpisodeRecord, ...]


def _episode_rank(episode: InteractiveEpisodeRecord) -> tuple[Any, ...]:
    """Deterministic minimum-exemplar order, independent of arrival order."""

    return (
        len(episode.decisions),
        episode.action_token_ids,
        episode.transition_hashes,
    )


class VerifiedInteractiveReplayBank:
    """Retain one deterministic positive exemplar per prompt/outcome mode."""

    def __init__(self, *, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._groups: dict[
            tuple[int, ...], dict[str, InteractiveEpisodeRecord]
        ] = {}
        self._cursor = 0

    @property
    def tracked_prompt_count(self) -> int:
        return len(self._groups)

    @property
    def tracked_outcome_count(self) -> int:
        return sum(len(group) for group in self._groups.values())

    def observe_group(self, episodes: Sequence[InteractiveEpisodeRecord]) -> None:
        for episode in episodes:
            if not isinstance(episode, InteractiveEpisodeRecord):
                raise TypeError("replay bank accepts InteractiveEpisodeRecord values")
            if episode.task_reward <= 0.0 or episode.outcome_key is None:
                continue
            modes = self._groups.setdefault(episode.group_prompt_token_ids, {})
            existing = modes.get(episode.outcome_key)
            if existing is None or _episode_rank(episode) < _episode_rank(existing):
                modes[episode.outcome_key] = episode
            if len(modes) > self.capacity:
                retained = sorted(modes)[: self.capacity]
                self._groups[episode.group_prompt_token_ids] = {
                    key: modes[key] for key in retained
                }

    def schedule_one_global_round_robin(self) -> InteractiveReplayGroup | None:
        prompt_keys = sorted(key for key, modes in self._groups.items() if modes)
        if not prompt_keys:
            return None
        index = self._cursor % len(prompt_keys)
        prompt_key = prompt_keys[index]
        self._cursor = (index + 1) % len(prompt_keys)
        modes = self._groups[prompt_key]
        outcome_keys = tuple(sorted(modes))
        return InteractiveReplayGroup(
            group_prompt_token_ids=prompt_key,
            outcome_keys=outcome_keys,
            episodes=tuple(modes[key] for key in outcome_keys),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "verified-interactive-replay-bank-v1",
            "capacity": self.capacity,
            "cursor": self._cursor,
            "groups": [
                {
                    "group_prompt_token_ids": list(prompt_key),
                    "episodes": [
                        modes[key].state_dict() for key in sorted(modes)
                    ],
                }
                for prompt_key, modes in sorted(self._groups.items())
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "verified-interactive-replay-bank-v1":
            raise ValueError("interactive replay bank schema mismatch")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("interactive replay bank capacity mismatch")
        restored: dict[tuple[int, ...], dict[str, InteractiveEpisodeRecord]] = {}
        for group in state.get("groups", []):
            prompt_key = _token_tuple(
                group["group_prompt_token_ids"], name="group_prompt_token_ids"
            )
            modes: dict[str, InteractiveEpisodeRecord] = {}
            for raw_episode in group["episodes"]:
                episode = InteractiveEpisodeRecord.from_state_dict(raw_episode)
                if (
                    episode.group_prompt_token_ids != prompt_key
                    or episode.outcome_key is None
                    or episode.task_reward <= 0.0
                    or episode.outcome_key in modes
                ):
                    raise ValueError("invalid interactive replay bank episode")
                modes[episode.outcome_key] = episode
            if not modes or len(modes) > self.capacity or prompt_key in restored:
                raise ValueError("invalid interactive replay bank group")
            restored[prompt_key] = modes
        cursor = int(state.get("cursor", 0))
        if cursor < 0 or (restored and cursor >= len(restored)):
            raise ValueError("interactive replay bank cursor is out of range")
        if not restored and cursor != 0:
            raise ValueError("empty interactive replay bank requires cursor zero")
        self._groups = restored
        self._cursor = cursor


def fixed_replay_slots(
    group: InteractiveReplayGroup | None, *, slot_count: int
) -> tuple[
    tuple[InteractiveEpisodeRecord | None, ...],
    tuple[bool, ...],
]:
    """Pad one replay group to a fixed compute budget without scientific rows."""

    if int(slot_count) <= 0:
        raise ValueError("slot_count must be positive")
    episodes = () if group is None else group.episodes
    if len(episodes) > int(slot_count):
        raise ValueError("replay group exceeds the fixed slot budget")
    inactive = int(slot_count) - len(episodes)
    return (
        tuple(episodes) + (None,) * inactive,
        (True,) * len(episodes) + (False,) * inactive,
    )


def length_normalized_episode_scores(
    selected_action_logprobs: torch.Tensor,
    decision_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean current selected-action log probability for each active episode."""

    if (
        selected_action_logprobs.ndim != 2
        or selected_action_logprobs.shape != decision_mask.shape
    ):
        raise ValueError("log probabilities and decision mask must share a 2D shape")
    if not bool(torch.isfinite(selected_action_logprobs).all()):
        raise ValueError("selected-action log probabilities must be finite")
    mask = decision_mask.to(dtype=torch.bool)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every replay episode must contain an active decision")
    numeric_mask = mask.to(dtype=selected_action_logprobs.dtype)
    return (selected_action_logprobs * numeric_mask).sum(dim=1) / numeric_mask.sum(
        dim=1
    )


def replay_gradient_surrogate(
    live_scores: torch.Tensor,
    *,
    raw_score_gradients: torch.Tensor,
    compute_only: bool,
    scale: float,
) -> torch.Tensor:
    """Apply frozen replay score gradients, or traverse with exact-zero effect."""

    if live_scores.shape != raw_score_gradients.shape:
        raise ValueError("live scores and raw replay gradients must share a shape")
    if not bool(torch.isfinite(live_scores).all()) or not bool(
        torch.isfinite(raw_score_gradients).all()
    ):
        raise ValueError("replay surrogate inputs must be finite")
    coefficient = float(scale)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("replay scale must be finite and nonnegative")
    gradient = raw_score_gradients.detach()
    if bool(compute_only):
        gradient = torch.zeros_like(gradient)
    return (live_scores * gradient).sum() * coefficient
