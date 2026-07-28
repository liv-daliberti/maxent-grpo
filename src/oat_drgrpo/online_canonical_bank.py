"""Verified executable-outcome tracking and online canonical MaxEnt.

The bank is prompt-local and contains only canonical keys produced by
validator-positive, loss-active rollouts. A group is scored against an immutable
pre-group snapshot and committed only after every row has been scored.  This
keeps novelty independent of row order.

The entropy term is a bounded, leave-one-out score-function estimator over the
empirical distribution on the verified bank.  It is returned as a detached
policy advantage and must be added *after* Dr.GRPO task-reward centering.  With
both coefficients at zero the same bank is a passive tracker: it records
verified discoveries without changing ordinary Dr.GRPO.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


def _prompt_key(prompt_token_ids: Sequence[int]) -> str:
    normalized: list[int] = []
    for value in prompt_token_ids:
        if isinstance(value, bool):
            raise ValueError("prompt token ids must be integers, not booleans")
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("prompt token ids must be integers") from exc
        if token_id < 0 or token_id != value:
            raise ValueError("prompt token ids must be non-negative")
        normalized.append(token_id)
    if not normalized:
        raise ValueError("prompt token ids must not be empty")
    encoded = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OnlineCanonicalBankDiagnostics:
    entropy_estimate_mean: float
    normalized_entropy_mean: float
    normalized_entropy_ratio_mean: float
    normalized_entropy_ratio_eligible_fraction: float
    log_support_mean: float
    entropy_alpha_used: float
    entropy_advantage_mean: float
    entropy_advantage_rms: float
    novelty_advantage_mean: float
    novelty_advantage_rms: float
    combined_advantage_mean: float
    combined_advantage_rms: float
    eligible_fraction: float
    reward_positive_fraction: float
    canonicalizable_correct_fraction: float
    new_outcome_count: float
    new_outcome_row_fraction: float
    bank_size_before_mean: float
    bank_size_after_mean: float
    tracked_prompts: float
    tracked_outcomes: float
    support_at_least_two_prompt_fraction: float


@dataclass(frozen=True)
class VerifiedCanonicalReplayGroup:
    """One prompt and its deterministic validator-positive mode exemplars."""

    prompt_token_ids: tuple[int, ...]
    outcome_keys: tuple[str, ...]
    response_token_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class VerifiedProposalAdmissionDiagnostics:
    """Atomic support-only admission from conditioned proposal rollouts."""

    proposal_groups: int
    proposal_rows: int
    new_outcomes: int
    stored_exemplars: int
    tracked_prompts: int
    tracked_outcomes: int


def _normalized_token_tuple(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{label} must contain integers, not booleans")
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain integers") from exc
        if token_id < 0 or token_id != value:
            raise ValueError(f"{label} must contain non-negative integers")
        normalized.append(token_id)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return tuple(normalized)


class OnlineCanonicalBank:
    """Persistent verified outcome banks and their on-policy MaxEnt score."""

    def __init__(
        self,
        *,
        entropy_alpha: float,
        novelty_beta: float,
        pseudocount: float = 1.0,
        surprisal_clip: float = 5.0,
        retain_exemplars: bool = False,
        replay_capacity: int = 16,
        global_replay_groups_per_step: int = 0,
        global_replay_bootstrap_steps: int = 0,
        separate_proposal_objective_support: bool = False,
    ) -> None:
        for name, value in (
            ("entropy_alpha", entropy_alpha),
            ("novelty_beta", novelty_beta),
            ("pseudocount", pseudocount),
            ("surprisal_clip", surprisal_clip),
        ):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in {"pseudocount", "surprisal_clip"} and value <= 0:
                raise ValueError(f"{name} must be positive")
            if name in {"entropy_alpha", "novelty_beta"} and value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.entropy_alpha = float(entropy_alpha)
        self.novelty_beta = float(novelty_beta)
        self.pseudocount = float(pseudocount)
        self.surprisal_clip = float(surprisal_clip)
        self.retain_exemplars = bool(retain_exemplars)
        self.separate_proposal_objective_support = bool(
            separate_proposal_objective_support
        )
        if (
            self.separate_proposal_objective_support
            and not self.retain_exemplars
        ):
            raise ValueError(
                "separate proposal objective support requires replay exemplars"
            )
        if (
            isinstance(replay_capacity, bool)
            or int(replay_capacity) != replay_capacity
            or int(replay_capacity) < 2
        ):
            raise ValueError("replay_capacity must be an integer at least two")
        self.replay_capacity = int(replay_capacity)
        if (
            isinstance(global_replay_groups_per_step, bool)
            or int(global_replay_groups_per_step)
            != global_replay_groups_per_step
            or int(global_replay_groups_per_step) < 0
        ):
            raise ValueError(
                "global_replay_groups_per_step must be a non-negative integer"
            )
        self.global_replay_groups_per_step = int(
            global_replay_groups_per_step
        )
        if (
            isinstance(global_replay_bootstrap_steps, bool)
            or int(global_replay_bootstrap_steps)
            != global_replay_bootstrap_steps
            or int(global_replay_bootstrap_steps) < 0
        ):
            raise ValueError(
                "global_replay_bootstrap_steps must be a non-negative integer"
            )
        self.global_replay_bootstrap_steps = int(
            global_replay_bootstrap_steps
        )
        if (
            self.global_replay_bootstrap_steps > 0
            and self.global_replay_groups_per_step == 0
        ):
            raise ValueError(
                "global replay bootstrap requires positive global groups "
                "per step"
            )
        self._counts: dict[str, dict[str, int]] = {}
        self._prompt_token_ids: dict[str, tuple[int, ...]] = {}
        self._exemplars: dict[
            str,
            dict[str, tuple[int, ...]],
        ] = {}
        self._groups_scored = 0
        self._rows_scored = 0
        self._global_replay_cursor = 0
        self._global_replay_updates = 0
        self._proposal_groups = 0
        self._proposal_rows = 0
        self._proposal_new_outcomes = 0
        self._proposal_only_outcomes: dict[str, set[str]] = {}

    @property
    def tracked_prompt_count(self) -> int:
        return len(self._counts)

    @property
    def tracked_outcome_count(self) -> int:
        return sum(len(counts) for counts in self._counts.values())

    @property
    def objective_active(self) -> bool:
        """Whether this bank contributes a nonzero exploration objective."""

        return self.entropy_alpha > 0.0 or self.novelty_beta > 0.0

    @property
    def mean_support_per_prompt(self) -> float:
        """Mean on-policy support used by entropy/novelty advantages."""

        if self.tracked_prompt_count == 0:
            return 0.0
        return self.tracked_outcome_count / self.tracked_prompt_count

    @property
    def replay_mean_support_per_prompt(self) -> float:
        """Mean replay-exemplar support, including proposal-only outcomes."""

        nonempty = [
            exemplars
            for exemplars in self._exemplars.values()
            if exemplars
        ]
        if not nonempty:
            return 0.0
        return sum(len(exemplars) for exemplars in nonempty) / len(nonempty)

    @property
    def support_at_least_two_prompt_fraction(self) -> float:
        """Fraction of discovered prompts with at least two verified routes."""

        if self.tracked_prompt_count == 0:
            return 0.0
        return (
            sum(int(len(counts) >= 2) for counts in self._counts.values())
            / self.tracked_prompt_count
        )

    @property
    def global_replay_updates(self) -> int:
        """Number of non-empty finite-bootstrap global replay updates."""

        return self._global_replay_updates

    @property
    def proposal_groups(self) -> int:
        return self._proposal_groups

    @property
    def proposal_rows(self) -> int:
        return self._proposal_rows

    @property
    def proposal_new_outcomes(self) -> int:
        return self._proposal_new_outcomes

    @property
    def global_replay_bootstrap_active(self) -> bool:
        """Whether the finite global cold-start phase may still schedule."""

        return (
            self.global_replay_groups_per_step > 0
            and self.global_replay_bootstrap_steps > 0
            and self._global_replay_updates
            < self.global_replay_bootstrap_steps
        )

    def state_dict(self) -> dict[str, Any]:
        state = {
            "schema": "online_growing_support_canonical_maxent_v1",
            "entropy_alpha": self.entropy_alpha,
            "novelty_beta": self.novelty_beta,
            "pseudocount": self.pseudocount,
            "surprisal_clip": self.surprisal_clip,
            "groups_scored": self._groups_scored,
            "rows_scored": self._rows_scored,
            "counts": {
                prompt_key: dict(outcome_counts)
                for prompt_key, outcome_counts in self._counts.items()
            },
        }
        if self.retain_exemplars:
            state.update(
                {
                    "schema": (
                        "online_growing_support_canonical_maxent_"
                        "replay_separated_proposal_v3"
                        if self.separate_proposal_objective_support
                        else "online_growing_support_canonical_maxent_replay_v2"
                    ),
                    "retain_exemplars": True,
                    "separate_proposal_objective_support": (
                        self.separate_proposal_objective_support
                    ),
                    "replay_capacity": self.replay_capacity,
                    "global_replay_groups_per_step": (
                        self.global_replay_groups_per_step
                    ),
                    "global_replay_bootstrap_steps": (
                        self.global_replay_bootstrap_steps
                    ),
                    "global_replay_cursor": self._global_replay_cursor,
                    "global_replay_updates": self._global_replay_updates,
                    "proposal_groups": self._proposal_groups,
                    "proposal_rows": self._proposal_rows,
                    "proposal_new_outcomes": (
                        self._proposal_new_outcomes
                    ),
                    "prompt_token_ids": {
                        prompt_key: list(token_ids)
                        for prompt_key, token_ids
                        in self._prompt_token_ids.items()
                    },
                    "exemplars": {
                        prompt_key: {
                            outcome_key: list(token_ids)
                            for outcome_key, token_ids
                            in outcome_exemplars.items()
                        }
                        for prompt_key, outcome_exemplars
                        in self._exemplars.items()
                    },
                    "proposal_only_outcomes": {
                        prompt_key: sorted(outcomes)
                        for prompt_key, outcomes
                        in self._proposal_only_outcomes.items()
                        if outcomes
                    },
                }
            )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("invalid online canonical bank state")
        expected_schema = (
            (
                "online_growing_support_canonical_maxent_"
                "replay_separated_proposal_v3"
                if self.separate_proposal_objective_support
                else "online_growing_support_canonical_maxent_replay_v2"
            )
            if self.retain_exemplars
            else "online_growing_support_canonical_maxent_v1"
        )
        if state.get("schema") != expected_schema:
            raise ValueError(
                "online canonical bank state replay configuration mismatch"
            )
        for name in (
            "entropy_alpha",
            "novelty_beta",
            "pseudocount",
            "surprisal_clip",
        ):
            try:
                saved = float(state[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"online canonical bank state is missing valid {name}"
                ) from exc
            configured = float(getattr(self, name))
            if not math.isfinite(saved) or not math.isclose(
                saved, configured, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"online canonical bank resume mismatch for {name}: "
                    f"saved={saved!r} configured={configured!r}"
                )
        raw_counts = state.get("counts")
        if not isinstance(raw_counts, dict):
            raise ValueError("online canonical bank state has invalid counts")
        restored: dict[str, dict[str, int]] = {}
        for prompt_key, outcome_counts in raw_counts.items():
            if not isinstance(prompt_key, str) or not isinstance(
                outcome_counts, dict
            ):
                raise ValueError("online canonical bank state has invalid bank")
            restored[prompt_key] = {}
            for outcome_key, count in outcome_counts.items():
                if (
                    not isinstance(outcome_key, str)
                    or isinstance(count, bool)
                    or int(count) != count
                    or int(count) <= 0
                ):
                    raise ValueError(
                        "online canonical bank state has invalid outcome count"
                    )
                restored[prompt_key][outcome_key] = int(count)
        self._counts = restored
        if self.retain_exemplars:
            if state.get("retain_exemplars") is not True:
                raise ValueError(
                    "online canonical replay state lacks its configuration"
                )
            if int(state.get("replay_capacity", -1)) != self.replay_capacity:
                raise ValueError(
                    "online canonical bank resume mismatch for replay_capacity"
                )
            if (
                int(state.get("global_replay_groups_per_step", 0))
                != self.global_replay_groups_per_step
            ):
                raise ValueError(
                    "online canonical bank resume mismatch for "
                    "global_replay_groups_per_step"
                )
            if (
                int(state.get("global_replay_bootstrap_steps", 0))
                != self.global_replay_bootstrap_steps
            ):
                raise ValueError(
                    "online canonical bank resume mismatch for "
                    "global_replay_bootstrap_steps"
                )
            raw_prompts = state.get("prompt_token_ids")
            raw_exemplars = state.get("exemplars")
            if not isinstance(raw_prompts, dict) or not isinstance(
                raw_exemplars,
                dict,
            ):
                raise ValueError(
                    "online canonical replay state lacks exemplar maps"
                )
            restored_prompts: dict[str, tuple[int, ...]] = {}
            restored_exemplars: dict[
                str,
                dict[str, tuple[int, ...]],
            ] = {}
            if self.separate_proposal_objective_support:
                if (
                    state.get("separate_proposal_objective_support")
                    is not True
                ):
                    raise ValueError(
                        "online canonical replay state lacks separated "
                        "proposal support"
                    )
                raw_proposal_only = state.get("proposal_only_outcomes")
                if not isinstance(raw_proposal_only, dict):
                    raise ValueError(
                        "online canonical replay state lacks proposal-only "
                        "support"
                    )
                restored_proposal_only: dict[str, set[str]] = {}
                for prompt_key, outcomes in raw_proposal_only.items():
                    if (
                        not isinstance(prompt_key, str)
                        or not isinstance(outcomes, list)
                        or any(
                            not isinstance(outcome, str) or not outcome
                            for outcome in outcomes
                        )
                        or len(set(outcomes)) != len(outcomes)
                    ):
                        raise ValueError(
                            "online canonical replay state has invalid "
                            "proposal-only support"
                        )
                    restored_proposal_only[prompt_key] = set(outcomes)
                prompt_keys = set(raw_exemplars)
                if (
                    set(raw_prompts) != prompt_keys
                    or not set(restored).issubset(prompt_keys)
                    or not set(restored_proposal_only).issubset(prompt_keys)
                ):
                    raise ValueError(
                        "online canonical replay state contains unknown prompts"
                    )
            else:
                restored_proposal_only = {}
                prompt_keys = set(restored)
            for prompt_key in prompt_keys:
                counts = restored.get(prompt_key, {})
                if prompt_key not in raw_prompts:
                    raise ValueError(
                        "online canonical replay state is missing prompt tokens"
                    )
                restored_prompts[prompt_key] = _normalized_token_tuple(
                    raw_prompts[prompt_key],
                    label="stored canonical replay prompt token ids",
                )
                prompt_exemplars = raw_exemplars.get(prompt_key)
                invalid_exemplars = (
                    not isinstance(prompt_exemplars, dict)
                    or len(prompt_exemplars) > self.replay_capacity
                )
                if not invalid_exemplars:
                    exemplar_keys = set(prompt_exemplars)
                    if self.separate_proposal_objective_support:
                        proposal_keys = restored_proposal_only.get(
                            prompt_key,
                            set(),
                        )
                        invalid_exemplars = (
                            not proposal_keys.issubset(exemplar_keys)
                            or bool(proposal_keys & set(counts))
                            or not exemplar_keys.issubset(
                                set(counts) | proposal_keys
                            )
                        )
                    else:
                        invalid_exemplars = (
                            not exemplar_keys.issubset(counts)
                            or (
                                len(counts) <= self.replay_capacity
                                and exemplar_keys != set(counts)
                            )
                        )
                if invalid_exemplars:
                    raise ValueError(
                        "online canonical replay exemplars mismatch bank support"
                    )
                restored_exemplars[prompt_key] = {
                    outcome_key: _normalized_token_tuple(
                        prompt_exemplars[outcome_key],
                        label="stored canonical replay response token ids",
                    )
                    for outcome_key in prompt_exemplars
                }
            if (
                not self.separate_proposal_objective_support
                and (
                    set(raw_prompts) != set(restored)
                    or set(raw_exemplars) != set(restored)
                )
            ):
                raise ValueError(
                    "online canonical replay state contains unknown prompts"
                )
            self._prompt_token_ids = restored_prompts
            self._exemplars = restored_exemplars
            self._proposal_only_outcomes = restored_proposal_only
            raw_cursor = state.get("global_replay_cursor", 0)
            if (
                isinstance(raw_cursor, bool)
                or int(raw_cursor) != raw_cursor
                or int(raw_cursor) < 0
            ):
                raise ValueError(
                    "online canonical replay state has invalid global cursor"
                )
            self._global_replay_cursor = int(raw_cursor)
            raw_updates = state.get("global_replay_updates", 0)
            if (
                isinstance(raw_updates, bool)
                or int(raw_updates) != raw_updates
                or int(raw_updates) < 0
                or (
                    self.global_replay_bootstrap_steps > 0
                    and int(raw_updates)
                    > self.global_replay_bootstrap_steps
                )
            ):
                raise ValueError(
                    "online canonical replay state has invalid global "
                    "update count"
                )
            self._global_replay_updates = int(raw_updates)
            for field_name, attr_name in (
                ("proposal_groups", "_proposal_groups"),
                ("proposal_rows", "_proposal_rows"),
                ("proposal_new_outcomes", "_proposal_new_outcomes"),
            ):
                raw_value = state.get(field_name, 0)
                if (
                    isinstance(raw_value, bool)
                    or int(raw_value) != raw_value
                    or int(raw_value) < 0
                ):
                    raise ValueError(
                        "online canonical replay state has invalid "
                        f"{field_name}"
                    )
                setattr(self, attr_name, int(raw_value))
        else:
            self._prompt_token_ids = {}
            self._exemplars = {}
            self._global_replay_cursor = 0
            self._global_replay_updates = 0
            self._proposal_groups = 0
            self._proposal_rows = 0
            self._proposal_new_outcomes = 0
            self._proposal_only_outcomes = {}
        self._groups_scored = int(state.get("groups_scored", 0))
        self._rows_scored = int(state.get("rows_scored", 0))

    def admit_verified_proposals(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        outcome_keys: Sequence[str],
        response_token_ids: Sequence[Sequence[int]],
    ) -> VerifiedProposalAdmissionDiagnostics:
        """Atomically add novel validator-positive proposal outcomes.

        Proposal rows were sampled under a conditioned prompt, so they are not
        on-policy rows for the neutral prompt. This is a support-only update:
        every supplied distinct outcome enters with count one, irrespective of
        how often it appeared in the conditioned candidate group. The caller
        must supply only outcomes absent from the pre-proposal replay group.
        """

        if not self.retain_exemplars:
            raise RuntimeError("proposal admission requires replay exemplars")
        row_count = len(outcome_keys)
        if not (
            len(prompt_token_ids)
            == len(response_token_ids)
            == row_count
        ):
            raise ValueError(
                "verified proposal admission inputs must have equal lengths"
            )
        if row_count == 0:
            return VerifiedProposalAdmissionDiagnostics(
                proposal_groups=0,
                proposal_rows=0,
                new_outcomes=0,
                stored_exemplars=0,
                tracked_prompts=sum(
                    bool(exemplars)
                    for exemplars in self._exemplars.values()
                ),
                tracked_outcomes=sum(
                    len(exemplars)
                    for exemplars in self._exemplars.values()
                ),
            )

        normalized_rows: list[
            tuple[str, tuple[int, ...], str, tuple[int, ...]]
        ] = []
        seen_pairs: set[tuple[str, str]] = set()
        for raw_prompt, raw_key, raw_response in zip(
            prompt_token_ids,
            outcome_keys,
            response_token_ids,
        ):
            normalized_prompt = _normalized_token_tuple(
                raw_prompt,
                label="verified proposal prompt token ids",
            )
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(
                    "verified proposal outcome keys must be non-empty strings"
                )
            normalized_response = _normalized_token_tuple(
                raw_response,
                label="verified proposal response token ids",
            )
            prompt_key = _prompt_key(normalized_prompt)
            pair = (prompt_key, raw_key)
            if pair in seen_pairs:
                raise ValueError(
                    "verified proposal admission requires one row per "
                    "prompt/outcome pair"
                )
            seen_pairs.add(pair)
            normalized_rows.append(
                (
                    prompt_key,
                    normalized_prompt,
                    raw_key,
                    normalized_response,
                )
            )

        # Stage all maps before committing so a hash collision, duplicate, or
        # malformed later row cannot partially mutate the live replay bank.
        staged_counts = {
            prompt_key: dict(counts)
            for prompt_key, counts in self._counts.items()
        }
        staged_prompts = dict(self._prompt_token_ids)
        staged_exemplars = {
            prompt_key: dict(exemplars)
            for prompt_key, exemplars in self._exemplars.items()
        }
        staged_proposal_only = {
            prompt_key: set(outcomes)
            for prompt_key, outcomes in self._proposal_only_outcomes.items()
        }
        new_outcomes = 0
        stored_exemplars = 0
        proposal_prompt_keys: set[str] = set()
        for prompt_key, prompt_tokens, outcome_key, response_tokens in (
            normalized_rows
        ):
            proposal_prompt_keys.add(prompt_key)
            stored_prompt = staged_prompts.get(prompt_key)
            if stored_prompt is not None and stored_prompt != prompt_tokens:
                raise ValueError(
                    "canonical replay prompt hash collision or mutation"
                )
            counts = staged_counts.get(prompt_key, {})
            if not self.separate_proposal_objective_support:
                counts = staged_counts.setdefault(prompt_key, {})
            exemplars = staged_exemplars.setdefault(prompt_key, {})
            known_outcomes = (
                set(exemplars)
                if self.separate_proposal_objective_support
                else set(counts)
            )
            if outcome_key in known_outcomes:
                raise ValueError(
                    "verified proposal admission accepts only outcomes absent "
                    "from the current bank"
                )
            staged_prompts[prompt_key] = prompt_tokens
            if (
                self.separate_proposal_objective_support
                and len(exemplars) >= self.replay_capacity
            ):
                raise ValueError(
                    "verified proposal admission exceeds replay capacity"
                )
            if not self.separate_proposal_objective_support:
                counts[outcome_key] = 1
            if len(exemplars) < self.replay_capacity:
                exemplars[outcome_key] = response_tokens
                if self.separate_proposal_objective_support:
                    staged_proposal_only.setdefault(
                        prompt_key,
                        set(),
                    ).add(outcome_key)
                stored_exemplars += 1
            new_outcomes += 1

        self._counts = staged_counts
        self._prompt_token_ids = staged_prompts
        self._exemplars = staged_exemplars
        self._proposal_only_outcomes = staged_proposal_only
        proposal_groups = len(proposal_prompt_keys)
        self._proposal_groups += proposal_groups
        self._proposal_rows += row_count
        self._proposal_new_outcomes += new_outcomes
        return VerifiedProposalAdmissionDiagnostics(
            proposal_groups=proposal_groups,
            proposal_rows=row_count,
            new_outcomes=new_outcomes,
            stored_exemplars=stored_exemplars,
            tracked_prompts=sum(
                bool(exemplars) for exemplars in self._exemplars.values()
            ),
            tracked_outcomes=sum(
                len(exemplars) for exemplars in self._exemplars.values()
            ),
        )

    def replay_groups(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        *,
        min_modes: int = 2,
    ) -> list[VerifiedCanonicalReplayGroup]:
        """Return deterministic replay banks for the requested prompts."""

        if not self.retain_exemplars:
            raise RuntimeError("canonical replay exemplars are not retained")
        if isinstance(min_modes, bool) or int(min_modes) not in {1, 2}:
            raise ValueError("canonical replay min_modes must be one or two")
        min_modes = int(min_modes)
        groups: list[VerifiedCanonicalReplayGroup] = []
        seen: set[str] = set()
        for raw_prompt_tokens in prompt_token_ids:
            normalized_prompt = _normalized_token_tuple(
                raw_prompt_tokens,
                label="canonical replay prompt token ids",
            )
            prompt_key = _prompt_key(normalized_prompt)
            if prompt_key in seen:
                continue
            seen.add(prompt_key)
            exemplars = self._exemplars.get(prompt_key, {})
            if len(exemplars) < min_modes:
                continue
            outcome_keys = tuple(sorted(exemplars))
            groups.append(
                VerifiedCanonicalReplayGroup(
                    prompt_token_ids=normalized_prompt,
                    outcome_keys=outcome_keys,
                    response_token_ids=tuple(
                        exemplars[key] for key in outcome_keys
                    ),
                )
            )
        return groups

    def scheduled_global_replay_groups(
        self,
        *,
        min_modes: int = 1,
    ) -> list[VerifiedCanonicalReplayGroup]:
        """Round-robin a fixed compute budget over all verified prompt banks.

        Selection depends only on the model's accumulated validator-positive
        exemplars. It never consults exhaustive support, evaluation, or a
        desired entropy/mode target. The cursor is checkpointed so resume does
        not silently resample the replay schedule.
        """

        if not self.retain_exemplars:
            raise RuntimeError("canonical replay exemplars are not retained")
        if isinstance(min_modes, bool) or int(min_modes) not in {1, 2}:
            raise ValueError("canonical replay min_modes must be one or two")
        min_modes = int(min_modes)
        if self.global_replay_groups_per_step == 0:
            return []
        if (
            self.global_replay_bootstrap_steps > 0
            and self._global_replay_updates
            >= self.global_replay_bootstrap_steps
        ):
            return []
        prompt_keys = sorted(
            prompt_key
            for prompt_key, exemplars in self._exemplars.items()
            if len(exemplars) >= min_modes
        )
        if not prompt_keys:
            return []
        count = min(self.global_replay_groups_per_step, len(prompt_keys))
        start = self._global_replay_cursor % len(prompt_keys)
        selected = [
            prompt_keys[(start + offset) % len(prompt_keys)]
            for offset in range(count)
        ]
        self._global_replay_cursor = (start + count) % len(prompt_keys)
        if self.global_replay_bootstrap_steps > 0:
            self._global_replay_updates += 1
        groups: list[VerifiedCanonicalReplayGroup] = []
        for prompt_key in selected:
            exemplars = self._exemplars[prompt_key]
            outcome_keys = tuple(sorted(exemplars))
            groups.append(
                VerifiedCanonicalReplayGroup(
                    prompt_token_ids=self._prompt_token_ids[prompt_key],
                    outcome_keys=outcome_keys,
                    response_token_ids=tuple(
                        exemplars[key] for key in outcome_keys
                    ),
                )
            )
        return groups

    def score_and_update(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        outcome_keys: Sequence[str | None],
        task_rewards: Sequence[float],
        active_mask: Sequence[float | bool],
        num_samples: int,
        entropy_alpha_override: float | None = None,
        response_token_ids: Sequence[Sequence[int]] | None = None,
    ) -> tuple[list[float], OnlineCanonicalBankDiagnostics]:
        """Score complete groups against snapshots, then commit verified keys."""

        entropy_alpha = (
            self.entropy_alpha
            if entropy_alpha_override is None
            else float(entropy_alpha_override)
        )
        if not math.isfinite(entropy_alpha) or entropy_alpha < 0:
            raise ValueError(
                "entropy_alpha_override must be finite and non-negative"
            )
        row_count = len(outcome_keys)
        if num_samples <= 1:
            raise ValueError("num_samples must be greater than one")
        if row_count == 0 or row_count % num_samples != 0:
            raise ValueError("outcome_keys must contain complete groups")
        if not (
            len(prompt_token_ids)
            == len(task_rewards)
            == len(active_mask)
            == row_count
        ):
            raise ValueError("online canonical bank inputs must have equal lengths")
        if self.retain_exemplars:
            if response_token_ids is None or len(response_token_ids) != row_count:
                raise ValueError(
                    "canonical replay requires one response token sequence per row"
                )

        prompt_keys = [_prompt_key(tokens) for tokens in prompt_token_ids]
        combined = [0.0] * row_count
        entropy_values: list[float] = []
        entropy_advantages = [0.0] * row_count
        novelty_advantages = [0.0] * row_count
        eligible_rows = 0
        positive_rows = 0
        canonicalizable_positive_rows = 0
        new_outcome_count = 0
        new_outcome_rows = 0
        bank_sizes_before: list[int] = []
        bank_sizes_after: list[int] = []
        normalized_entropies: list[float] = []
        normalized_entropy_ratios: list[float] = []
        log_supports: list[float] = []
        group_count = row_count // num_samples

        for start in range(0, row_count, num_samples):
            stop = start + num_samples
            group_prompt_keys = prompt_keys[start:stop]
            if len(set(group_prompt_keys)) != 1:
                raise ValueError(
                    "each online canonical candidate group must share one prompt"
                )
            prompt_key = group_prompt_keys[0]
            historical = Counter(self._counts.get(prompt_key, {}))
            bank_sizes_before.append(len(historical))
            group_keys = list(outcome_keys[start:stop])
            group_rewards = [float(value) for value in task_rewards[start:stop]]
            group_active = [bool(value) for value in active_mask[start:stop]]
            for reward, active in zip(group_rewards, group_active):
                if active and reward > 0:
                    positive_rows += 1

            eligible = [
                active and reward > 0 and isinstance(key, str) and bool(key)
                for key, reward, active in zip(
                    group_keys, group_rewards, group_active
                )
            ]
            canonicalizable_positive_rows += sum(eligible)
            eligible_rows += sum(eligible)
            current = Counter(
                str(key) for key, keep in zip(group_keys, eligible) if keep
            )
            support = set(historical) | set(current)
            new_keys = set(current) - set(historical)
            new_outcome_count += len(new_keys)
            new_outcome_rows += sum(current[key] for key in new_keys)

            for offset, (key, keep) in enumerate(zip(group_keys, eligible)):
                if not keep:
                    continue
                key = str(key)
                loo = historical.copy()
                loo.update(current)
                loo[key] -= 1
                if loo[key] <= 0:
                    del loo[key]
                support_size = max(len(support), 1)
                denominator = (
                    sum(loo.values()) + self.pseudocount * support_size
                )
                probabilities = {
                    candidate: (
                        loo.get(candidate, 0) + self.pseudocount
                    )
                    / denominator
                    for candidate in support
                }
                probability = probabilities[key]
                clipped_surprisal = min(
                    -math.log(probability), self.surprisal_clip
                )
                clipped_entropy = sum(
                    value
                    * min(-math.log(value), self.surprisal_clip)
                    for value in probabilities.values()
                )
                entropy_values.append(clipped_entropy)
                entropy_advantage = entropy_alpha * (
                    clipped_surprisal - clipped_entropy
                )
                novelty_advantage = (
                    self.novelty_beta / current[key]
                    if key in new_keys
                    else 0.0
                )
                row_index = start + offset
                entropy_advantages[row_index] = entropy_advantage
                novelty_advantages[row_index] = novelty_advantage
                combined[row_index] = entropy_advantage + novelty_advantage

            updated = historical.copy()
            updated.update(current)
            if updated:
                self._counts[prompt_key] = dict(updated)
            if self.retain_exemplars and new_keys:
                normalized_prompt = _normalized_token_tuple(
                    prompt_token_ids[start],
                    label="canonical replay prompt token ids",
                )
                stored_prompt = self._prompt_token_ids.get(prompt_key)
                if (
                    stored_prompt is not None
                    and stored_prompt != normalized_prompt
                ):
                    raise ValueError(
                        "canonical replay prompt hash collision or mutation"
                    )
                self._prompt_token_ids[prompt_key] = normalized_prompt
                prompt_exemplars = self._exemplars.setdefault(
                    prompt_key,
                    {},
                )
                proposal_only = self._proposal_only_outcomes.setdefault(
                    prompt_key,
                    set(),
                )
                assert response_token_ids is not None
                for new_key in sorted(new_keys):
                    if (
                        len(prompt_exemplars) >= self.replay_capacity
                        and new_key not in proposal_only
                    ):
                        break
                    candidates = [
                        _normalized_token_tuple(
                            response_token_ids[start + offset],
                            label="canonical replay response token ids",
                        )
                        for offset, (key, keep) in enumerate(
                            zip(group_keys, eligible)
                        )
                        if keep and str(key) == new_key
                    ]
                    if not candidates:
                        raise RuntimeError(
                            "new canonical replay mode lacks an eligible exemplar"
                        )
                    prompt_exemplars[new_key] = min(candidates)
                    proposal_only.discard(new_key)
                if not proposal_only:
                    self._proposal_only_outcomes.pop(prompt_key, None)
            bank_sizes_after.append(len(updated))
            if len(updated) >= 2:
                support_size = len(updated)
                denominator = (
                    sum(updated.values())
                    + self.pseudocount * support_size
                )
                probabilities = [
                    (count + self.pseudocount) / denominator
                    for count in updated.values()
                ]
                exact_entropy = -sum(
                    probability * math.log(probability)
                    for probability in probabilities
                )
                log_support = math.log(support_size)
                normalized_entropies.append(exact_entropy)
                log_supports.append(log_support)
                normalized_entropy_ratios.append(
                    min(max(exact_entropy / log_support, 0.0), 1.0)
                )
            self._groups_scored += 1

        self._rows_scored += row_count

        def mean(values: Sequence[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def rms(values: Sequence[float]) -> float:
            return (
                math.sqrt(sum(value * value for value in values) / len(values))
                if values
                else 0.0
            )

        diagnostics = OnlineCanonicalBankDiagnostics(
            entropy_estimate_mean=mean(entropy_values),
            normalized_entropy_mean=mean(normalized_entropies),
            normalized_entropy_ratio_mean=mean(
                normalized_entropy_ratios
            ),
            normalized_entropy_ratio_eligible_fraction=(
                len(normalized_entropy_ratios) / group_count
            ),
            log_support_mean=mean(log_supports),
            entropy_alpha_used=entropy_alpha,
            entropy_advantage_mean=mean(entropy_advantages),
            entropy_advantage_rms=rms(entropy_advantages),
            novelty_advantage_mean=mean(novelty_advantages),
            novelty_advantage_rms=rms(novelty_advantages),
            combined_advantage_mean=mean(combined),
            combined_advantage_rms=rms(combined),
            eligible_fraction=eligible_rows / row_count,
            reward_positive_fraction=positive_rows / row_count,
            canonicalizable_correct_fraction=(
                canonicalizable_positive_rows / positive_rows
                if positive_rows
                else 0.0
            ),
            new_outcome_count=float(new_outcome_count),
            new_outcome_row_fraction=new_outcome_rows / row_count,
            bank_size_before_mean=mean(bank_sizes_before),
            bank_size_after_mean=mean(bank_sizes_after),
            tracked_prompts=float(self.tracked_prompt_count),
            tracked_outcomes=float(self.tracked_outcome_count),
            support_at_least_two_prompt_fraction=(
                self.support_at_least_two_prompt_fraction
            ),
        )
        return combined, diagnostics
