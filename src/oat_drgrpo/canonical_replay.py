"""Differentiable retention over verified canonical-mode exemplars.

The online count bank can shape modes that occur in the current rollout, but
it cannot assign positive probability to an observed mode that has disappeared
from the on-policy sample.  Canonical replay closes that actuator gap:

* one validator-positive token sequence is retained per observed prompt/mode;
* the current model assigns a length-normalized log score to every retained
  mode for a replayed prompt;
* a selectable target-free actuator either balances scores within that
  *observed* bank or raises the common likelihood of every observed verified
  exemplar;
* neither actuator consults exhaustive support or evaluation feedback.

The accompanying inverse controller observes only the model's normalized
entropy over its retained-mode scores.  Its coefficient has no lower or upper
projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class CanonicalReplayLoss:
    """Differentiable replay loss and detached model-score diagnostics."""

    loss: torch.Tensor
    normalized_entropy: torch.Tensor
    cross_entropy_excess: torch.Tensor
    score_gradients: torch.Tensor
    eligible_groups: int
    retained_modes: int
    actuator_groups: int
    actuator_modes: int


@dataclass(frozen=True)
class CanonicalReplayBatch:
    """Right-padded teacher-forcing batch over verified exemplars."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_masks: torch.Tensor
    group_sizes: tuple[int, ...]


@dataclass(frozen=True)
class CanonicalReplaySplitLoss:
    """Independent verified-mass and known-mode-balance replay terms."""

    mass_loss: torch.Tensor
    balance_loss: torch.Tensor
    normalized_entropy: torch.Tensor
    mass_score_gradients: torch.Tensor
    balance_score_gradients: torch.Tensor
    actuator_groups: int
    actuator_modes: int
    balance_eligible_groups: int
    balance_retained_modes: int


def materialize_canonical_replay_batch(
    groups: Sequence[Any],
    *,
    pad_token_id: int,
    device: torch.device | int | str,
) -> CanonicalReplayBatch:
    """Build causal-LM labels without changing any stored exemplar tokens."""

    if not groups:
        raise ValueError("canonical replay requires at least one group")
    if isinstance(pad_token_id, bool) or int(pad_token_id) < 0:
        raise ValueError("canonical replay pad_token_id must be non-negative")

    sequences: list[tuple[int, ...]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    group_sizes: list[int] = []
    for group in groups:
        prompt = tuple(int(value) for value in group.prompt_token_ids)
        responses = tuple(
            tuple(int(value) for value in response)
            for response in group.response_token_ids
        )
        if not prompt or not responses or any(not row for row in responses):
            raise ValueError(
                "canonical replay groups require a prompt and at least one "
                "non-empty response"
            )
        if any(value < 0 for value in prompt) or any(
            value < 0 for row in responses for value in row
        ):
            raise ValueError("canonical replay token ids must be non-negative")
        group_sizes.append(len(responses))
        for response in responses:
            sequences.append(prompt + response)
            prompt_lengths.append(len(prompt))
            response_lengths.append(len(response))

    max_length = max(len(row) for row in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    response_masks = torch.zeros(
        (len(sequences), max_length - 1),
        dtype=torch.bool,
        device=device,
    )
    for row_index, (sequence, prompt_length, response_length) in enumerate(
        zip(sequences, prompt_lengths, response_lengths)
    ):
        row_length = len(sequence)
        input_ids[row_index, :row_length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_index, :row_length] = 1
        response_start = prompt_length - 1
        response_masks[
            row_index,
            response_start : response_start + response_length,
        ] = True

    return CanonicalReplayBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_masks=response_masks,
        group_sizes=tuple(group_sizes),
    )


def canonical_replay_uniform_loss(
    mode_scores: torch.Tensor,
    group_sizes: Sequence[int],
) -> CanonicalReplayLoss:
    """Balance current model scores across each observed prompt-local bank.

    ``mode_scores`` contains one length-normalized teacher-forced log score per
    retained verified mode. ``group_sizes`` partitions those scores by prompt.
    Every group must contain at least two modes.  The loss is

    ``mean_g KL(U_g || softmax(scores_g))``.

    This reverse-direction KL retains a non-vanishing restorative gradient for
    a mode whose current probability is very small.  It uses only modes that
    the validator has actually observed; no gold support size appears.
    """

    if mode_scores.ndim != 1:
        raise ValueError("canonical replay mode_scores must be one-dimensional")
    sizes = tuple(int(value) for value in group_sizes)
    if not sizes or any(value < 2 for value in sizes):
        raise ValueError(
            "canonical replay groups must each contain at least two modes"
        )
    if sum(sizes) != int(mode_scores.numel()):
        raise ValueError("canonical replay group sizes do not partition scores")
    if not torch.isfinite(mode_scores).all():
        raise ValueError("canonical replay mode scores must be finite")

    losses: list[torch.Tensor] = []
    entropy_ratios: list[torch.Tensor] = []
    score_gradients: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        stop = start + size
        scores = mode_scores[start:stop].float()
        log_probabilities = torch.log_softmax(scores, dim=0)
        log_support = math.log(size)
        cross_entropy_excess = -log_probabilities.mean() - log_support
        score_gradients.append(
            (
                log_probabilities.detach().exp()
                - (1.0 / float(size))
            )
            / float(len(sizes))
        )
        # Keep the actuator in the model's ordinary precision, but compute the
        # collapse sensor in float64 so a very unlikely retained mode does not
        # disappear merely through float32 exponent underflow.
        sensor_log_probabilities = torch.log_softmax(
            mode_scores[start:stop].detach().double(),
            dim=0,
        )
        sensor_probabilities = sensor_log_probabilities.exp()
        entropy = -(
            sensor_probabilities * sensor_log_probabilities
        ).sum()
        losses.append(cross_entropy_excess)
        entropy_ratios.append(entropy / log_support)
        start = stop

    loss = torch.stack(losses).mean()
    normalized_entropy = torch.stack(entropy_ratios).mean().detach()
    return CanonicalReplayLoss(
        loss=loss,
        normalized_entropy=normalized_entropy,
        cross_entropy_excess=loss.detach(),
        score_gradients=torch.cat(score_gradients),
        eligible_groups=len(sizes),
        retained_modes=sum(sizes),
        actuator_groups=len(sizes),
        actuator_modes=sum(sizes),
    )


def canonical_replay_uniform_verified_likelihood_loss(
    mode_scores: torch.Tensor,
    group_sizes: Sequence[int],
) -> CanonicalReplayLoss:
    """Raise common verified-mode score while weighting observed modes equally.

    The bank-conditioned reverse KL in :func:`canonical_replay_uniform_loss`
    has score gradients ``q_bank - U_bank``.  They sum to zero within every
    prompt, so that objective can redistribute score among retained modes but
    cannot raise their common score against invalid outputs.

    This target-free successor instead minimizes

    ``mean_g mean_{k in B_g} -score(g, k)``.

    It is uniform maximum likelihood over only validator-positive exemplars
    the policy has actually discovered.  Its score gradients are
    ``-1 / (number_of_groups * modes_in_group)`` and therefore retain a
    non-zero common-mass component.  The conditioned-bank entropy and reverse
    KL remain detached diagnostics for the inverse controller and audit.  No
    exhaustive support size or evaluation signal appears in the objective.
    """

    sizes = tuple(int(value) for value in group_sizes)
    if mode_scores.ndim != 1:
        raise ValueError("canonical replay mode_scores must be one-dimensional")
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError(
            "verified-likelihood replay groups must each contain at least one mode"
        )
    if sum(sizes) != int(mode_scores.numel()):
        raise ValueError("canonical replay group sizes do not partition scores")
    if not torch.isfinite(mode_scores).all():
        raise ValueError("canonical replay mode scores must be finite")

    losses: list[torch.Tensor] = []
    balance_losses: list[torch.Tensor] = []
    entropy_ratios: list[torch.Tensor] = []
    score_gradients: list[torch.Tensor] = []
    entropy_eligible_modes = 0
    start = 0
    for size in sizes:
        stop = start + size
        scores = mode_scores[start:stop].float()
        losses.append(-scores.mean())
        score_gradients.append(
            torch.full_like(
                scores.detach(),
                -1.0 / float(len(sizes) * size),
            )
        )
        if size >= 2:
            log_probabilities = torch.log_softmax(scores, dim=0)
            log_support = math.log(size)
            balance_losses.append(
                -log_probabilities.mean() - log_support
            )
            sensor_log_probabilities = torch.log_softmax(
                mode_scores[start:stop].detach().double(),
                dim=0,
            )
            sensor_probabilities = sensor_log_probabilities.exp()
            entropy_ratios.append(
                -(
                    sensor_probabilities * sensor_log_probabilities
                ).sum()
                / log_support
            )
            entropy_eligible_modes += size
        start = stop

    if balance_losses:
        balance_loss = torch.stack(balance_losses).mean().detach()
        normalized_entropy = torch.stack(entropy_ratios).mean().detach()
    else:
        # Singleton banks have no canonical entropy. Keep the finite value as
        # a telemetry placeholder while eligible_groups=0 makes the inverse
        # controller idle; it must not consume this placeholder observation.
        balance_loss = mode_scores.detach().float().sum() * 0.0
        normalized_entropy = mode_scores.detach().double().new_tensor(1.0)

    return CanonicalReplayLoss(
        loss=torch.stack(losses).mean(),
        normalized_entropy=normalized_entropy,
        cross_entropy_excess=balance_loss,
        score_gradients=torch.cat(score_gradients),
        eligible_groups=len(balance_losses),
        retained_modes=entropy_eligible_modes,
        actuator_groups=len(sizes),
        actuator_modes=sum(sizes),
    )


def canonical_replay_split_mass_balance_loss(
    mode_scores: torch.Tensor,
    group_sizes: Sequence[int],
) -> CanonicalReplaySplitLoss:
    """Return actuator-aligned mass and balance terms from one score pass.

    The mass term is uniform verified likelihood over every group, including
    singletons. The balance term is ``KL(U || softmax(scores))`` over only
    groups with at least two modes. Its gradient is scattered back into the
    full replay-row layout with exact zeros for singleton rows.
    """

    mass = canonical_replay_uniform_verified_likelihood_loss(
        mode_scores,
        group_sizes,
    )
    sizes = tuple(int(value) for value in group_sizes)
    eligible_slices: list[tuple[int, int]] = []
    eligible_sizes: list[int] = []
    start = 0
    for size in sizes:
        stop = start + size
        if size >= 2:
            eligible_slices.append((start, stop))
            eligible_sizes.append(size)
        start = stop

    if eligible_slices:
        eligible_scores = torch.cat(
            [mode_scores[start:stop] for start, stop in eligible_slices]
        )
        balance = canonical_replay_uniform_loss(
            eligible_scores,
            eligible_sizes,
        )
        balance_gradients = torch.zeros_like(mode_scores.detach())
        source_start = 0
        for (target_start, target_stop), size in zip(
            eligible_slices,
            eligible_sizes,
        ):
            source_stop = source_start + size
            balance_gradients[target_start:target_stop] = (
                balance.score_gradients[source_start:source_stop]
            )
            source_start = source_stop
        balance_loss = balance.loss
        normalized_entropy = balance.normalized_entropy
        balance_groups = balance.eligible_groups
        balance_modes = balance.retained_modes
    else:
        balance_loss = mode_scores.float().sum() * 0.0
        normalized_entropy = mode_scores.detach().double().new_tensor(1.0)
        balance_gradients = torch.zeros_like(mode_scores.detach())
        balance_groups = 0
        balance_modes = 0

    return CanonicalReplaySplitLoss(
        mass_loss=mass.loss,
        balance_loss=balance_loss,
        normalized_entropy=normalized_entropy,
        mass_score_gradients=mass.score_gradients,
        balance_score_gradients=balance_gradients,
        actuator_groups=mass.actuator_groups,
        actuator_modes=mass.actuator_modes,
        balance_eligible_groups=balance_groups,
        balance_retained_modes=balance_modes,
    )


@dataclass
class CanonicalReplayInverseController:
    """Unprojected inverse control from the model's retained-mode entropy."""

    base_alpha: float
    warmup_steps: int
    ema_decay: float
    current_alpha: float | None = None
    entropy_ema: float | None = None
    reference_entropy: float | None = None
    observation_count: int = 0
    _warmup_entropy_sum: float = 0.0

    observation_metric_key: str = (
        "canonical_replay_normalized_model_entropy"
    )

    def __post_init__(self) -> None:
        self.base_alpha = self._positive_finite(
            self.base_alpha,
            "base_alpha",
        )
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        self.warmup_steps = int(self.warmup_steps)
        self.ema_decay = float(self.ema_decay)
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be finite and in [0, 1)")
        if self.current_alpha is None:
            self.current_alpha = self.base_alpha
        self.current_alpha = self._positive_finite(
            self.current_alpha,
            "current_alpha",
        )
        if self.entropy_ema is not None:
            self.entropy_ema = self._entropy(self.entropy_ema, "entropy_ema")
        if self.reference_entropy is not None:
            self.reference_entropy = self._entropy(
                self.reference_entropy,
                "reference_entropy",
                positive=True,
            )

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _entropy(
        value: float,
        name: str,
        *,
        positive: bool = False,
    ) -> float:
        result = float(value)
        if (
            not math.isfinite(result)
            or result < 0
            or result > 1 + 1e-6
            or (positive and result <= 0)
        ):
            qualifier = "positive " if positive else ""
            raise ValueError(
                f"{name} must be finite, {qualifier}and in [0, 1]"
            )
        return min(result, 1.0)

    def observe(self, normalized_entropy: float) -> dict[str, float]:
        """Advance only on an eligible model-score entropy observation."""

        value = self._entropy(
            normalized_entropy,
            "normalized_entropy",
            positive=True,
        )
        alpha_before = float(self.current_alpha)
        self.observation_count += 1
        if self.entropy_ema is None:
            self.entropy_ema = value
        else:
            self.entropy_ema = (
                self.ema_decay * self.entropy_ema
                + (1.0 - self.ema_decay) * value
            )

        if self.observation_count <= self.warmup_steps:
            self._warmup_entropy_sum += value
            self.current_alpha = self.base_alpha
            if self.observation_count == self.warmup_steps:
                self.reference_entropy = (
                    self._warmup_entropy_sum / self.warmup_steps
                )
            multiplier = 1.0
        else:
            if self.reference_entropy is None:
                raise RuntimeError(
                    "canonical replay inverse control lacks its warmup reference"
                )
            if self.entropy_ema is None or self.entropy_ema <= 0:
                raise ValueError(
                    "canonical replay entropy EMA must remain positive"
                )
            multiplier = self.reference_entropy / self.entropy_ema
            self.current_alpha = self._positive_finite(
                self.base_alpha * multiplier,
                "current_alpha",
            )

        diagnostics = {
            "canonical_replay_observed_normalized_entropy": value,
            "canonical_replay_entropy_ema": float(self.entropy_ema),
            "canonical_replay_inverse_multiplier": float(multiplier),
            "canonical_replay_alpha_before": alpha_before,
            "canonical_replay_next_alpha": float(self.current_alpha),
            "canonical_replay_observations": float(self.observation_count),
            "canonical_replay_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
            "canonical_replay_projection_active": 0.0,
        }
        if self.reference_entropy is not None:
            diagnostics["canonical_replay_reference_entropy"] = float(
                self.reference_entropy
            )
        return diagnostics

    def idle_diagnostics(self) -> dict[str, float]:
        """Report a bank-ineligible update without advancing warmup or EMA."""

        diagnostics = {
            "canonical_replay_inverse_multiplier": (
                float(self.current_alpha) / self.base_alpha
            ),
            "canonical_replay_next_alpha": float(self.current_alpha),
            "canonical_replay_observations": float(self.observation_count),
            "canonical_replay_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
            "canonical_replay_projection_active": 0.0,
            "canonical_replay_observation_skipped": 1.0,
        }
        if self.entropy_ema is not None:
            diagnostics["canonical_replay_entropy_ema"] = float(
                self.entropy_ema
            )
        if self.reference_entropy is not None:
            diagnostics["canonical_replay_reference_entropy"] = float(
                self.reference_entropy
            )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "canonical_replay_inverse",
            "controller_rule": (
                "unprojected_warmup_inverse_observed_bank_entropy_v1"
            ),
            "observation_metric_key": self.observation_metric_key,
            "base_alpha": self.base_alpha,
            "warmup_steps": self.warmup_steps,
            "ema_decay": self.ema_decay,
            "current_alpha": float(self.current_alpha),
            "entropy_ema": self.entropy_ema,
            "reference_entropy": self.reference_entropy,
            "observation_count": self.observation_count,
            "warmup_entropy_sum": self._warmup_entropy_sum,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("canonical replay controller state must be a dict")
        if state.get("controller_kind") != "canonical_replay_inverse":
            raise ValueError("checkpoint contains a different replay controller")
        if state.get("controller_rule") != (
            "unprojected_warmup_inverse_observed_bank_entropy_v1"
        ):
            raise ValueError("checkpoint uses an incompatible replay rule")
        if state.get("observation_metric_key") != self.observation_metric_key:
            raise ValueError("checkpoint uses an incompatible replay sensor")
        if not math.isclose(
            float(state.get("base_alpha", math.nan)),
            self.base_alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("resume mismatch for base_alpha")
        if int(state.get("warmup_steps", -1)) != self.warmup_steps:
            raise ValueError("resume mismatch for warmup_steps")
        if not math.isclose(
            float(state.get("ema_decay", math.nan)),
            self.ema_decay,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("resume mismatch for ema_decay")

        self.current_alpha = self._positive_finite(
            state.get("current_alpha"),
            "current_alpha",
        )
        raw_ema = state.get("entropy_ema")
        self.entropy_ema = (
            None
            if raw_ema is None
            else self._entropy(raw_ema, "entropy_ema")
        )
        raw_reference = state.get("reference_entropy")
        self.reference_entropy = (
            None
            if raw_reference is None
            else self._entropy(
                raw_reference,
                "reference_entropy",
                positive=True,
            )
        )
        self.observation_count = int(state.get("observation_count", -1))
        self._warmup_entropy_sum = float(
            state.get("warmup_entropy_sum", math.nan)
        )
        if (
            self.observation_count < 0
            or not math.isfinite(self._warmup_entropy_sum)
            or self._warmup_entropy_sum < 0
        ):
            raise ValueError("checkpoint contains invalid replay controller state")


@dataclass
class CanonicalReplayLikelihoodController:
    """Unprojected retention control from verified-exemplar surprisal.

    Bank entropy describes only relative probability among retained modes.
    This controller instead observes the positive uniform verified-likelihood
    loss that its actuator minimizes.  After warmup it applies

    ``alpha = base_alpha * surprisal_ema / surprisal_reference``.

    A rise in the model's own verified surprisal therefore strengthens the
    common-mass actuator.  No evaluation metric, support count, or coefficient
    projection enters the rule.
    """

    base_alpha: float
    warmup_steps: int
    ema_decay: float
    current_alpha: float | None = None
    surprisal_ema: float | None = None
    surprisal_reference: float | None = None
    observation_count: int = 0
    _warmup_surprisal_sum: float = 0.0

    observation_metric_key: str = "canonical_replay_actuator_loss"

    def __post_init__(self) -> None:
        self.base_alpha = self._positive_finite(
            self.base_alpha,
            "base_alpha",
        )
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        self.warmup_steps = int(self.warmup_steps)
        self.ema_decay = float(self.ema_decay)
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be finite and in [0, 1)")
        if self.current_alpha is None:
            self.current_alpha = self.base_alpha
        self.current_alpha = self._positive_finite(
            self.current_alpha,
            "current_alpha",
        )
        if self.surprisal_ema is not None:
            self.surprisal_ema = self._nonnegative_finite(
                self.surprisal_ema,
                "surprisal_ema",
            )
        if self.surprisal_reference is not None:
            self.surprisal_reference = self._positive_finite(
                self.surprisal_reference,
                "surprisal_reference",
            )

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _nonnegative_finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def observe(self, verified_surprisal: float) -> dict[str, float]:
        """Advance from one actuator-aligned verified-surprisal observation."""

        value = self._nonnegative_finite(
            verified_surprisal,
            "verified_surprisal",
        )
        alpha_before = float(self.current_alpha)
        self.observation_count += 1
        if self.surprisal_ema is None:
            self.surprisal_ema = value
        else:
            self.surprisal_ema = (
                self.ema_decay * self.surprisal_ema
                + (1.0 - self.ema_decay) * value
            )

        if self.observation_count <= self.warmup_steps:
            self._warmup_surprisal_sum += value
            self.current_alpha = self.base_alpha
            if self.observation_count == self.warmup_steps:
                reference = (
                    self._warmup_surprisal_sum / self.warmup_steps
                )
                self.surprisal_reference = self._positive_finite(
                    reference,
                    "warmup verified-surprisal reference",
                )
            multiplier = 1.0
        else:
            if self.surprisal_reference is None:
                raise RuntimeError(
                    "verified-likelihood control lacks its warmup reference"
                )
            if self.surprisal_ema is None or self.surprisal_ema <= 0:
                raise ValueError(
                    "verified-surprisal EMA must remain positive"
                )
            multiplier = (
                self.surprisal_ema / self.surprisal_reference
            )
            self.current_alpha = self._positive_finite(
                self.base_alpha * multiplier,
                "current_alpha",
            )

        diagnostics = {
            "canonical_replay_mass_observed_surprisal": value,
            "canonical_replay_mass_surprisal_ema": float(
                self.surprisal_ema
            ),
            "canonical_replay_mass_inverse_multiplier": float(multiplier),
            "canonical_replay_mass_alpha_before": alpha_before,
            "canonical_replay_mass_next_alpha": float(self.current_alpha),
            "canonical_replay_mass_observations": float(
                self.observation_count
            ),
            "canonical_replay_mass_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
            "canonical_replay_mass_projection_active": 0.0,
        }
        if self.surprisal_reference is not None:
            diagnostics[
                "canonical_replay_mass_surprisal_reference"
            ] = float(self.surprisal_reference)
        return diagnostics

    def idle_diagnostics(self) -> dict[str, float]:
        """Report a mass-ineligible update without advancing its state."""

        diagnostics = {
            "canonical_replay_mass_inverse_multiplier": (
                float(self.current_alpha) / self.base_alpha
            ),
            "canonical_replay_mass_next_alpha": float(self.current_alpha),
            "canonical_replay_mass_observations": float(
                self.observation_count
            ),
            "canonical_replay_mass_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
            "canonical_replay_mass_projection_active": 0.0,
            "canonical_replay_mass_observation_skipped": 1.0,
        }
        if self.surprisal_ema is not None:
            diagnostics["canonical_replay_mass_surprisal_ema"] = float(
                self.surprisal_ema
            )
        if self.surprisal_reference is not None:
            diagnostics[
                "canonical_replay_mass_surprisal_reference"
            ] = float(self.surprisal_reference)
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "canonical_replay_likelihood",
            "controller_rule": (
                "unprojected_warmup_verified_surprisal_ratio_v1"
            ),
            "observation_metric_key": self.observation_metric_key,
            "base_alpha": self.base_alpha,
            "warmup_steps": self.warmup_steps,
            "ema_decay": self.ema_decay,
            "current_alpha": float(self.current_alpha),
            "surprisal_ema": self.surprisal_ema,
            "surprisal_reference": self.surprisal_reference,
            "observation_count": self.observation_count,
            "warmup_surprisal_sum": self._warmup_surprisal_sum,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("likelihood controller state must be a dict")
        if state.get("controller_kind") != "canonical_replay_likelihood":
            raise ValueError(
                "checkpoint contains a different likelihood controller"
            )
        if state.get("controller_rule") != (
            "unprojected_warmup_verified_surprisal_ratio_v1"
        ):
            raise ValueError(
                "checkpoint uses an incompatible likelihood rule"
            )
        if state.get("observation_metric_key") != self.observation_metric_key:
            raise ValueError(
                "checkpoint uses an incompatible likelihood sensor"
            )
        if not math.isclose(
            float(state.get("base_alpha", math.nan)),
            self.base_alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("resume mismatch for base_alpha")
        if int(state.get("warmup_steps", -1)) != self.warmup_steps:
            raise ValueError("resume mismatch for warmup_steps")
        if not math.isclose(
            float(state.get("ema_decay", math.nan)),
            self.ema_decay,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("resume mismatch for ema_decay")

        self.current_alpha = self._positive_finite(
            state.get("current_alpha"),
            "current_alpha",
        )
        raw_ema = state.get("surprisal_ema")
        self.surprisal_ema = (
            None
            if raw_ema is None
            else self._nonnegative_finite(raw_ema, "surprisal_ema")
        )
        raw_reference = state.get("surprisal_reference")
        self.surprisal_reference = (
            None
            if raw_reference is None
            else self._positive_finite(
                raw_reference,
                "surprisal_reference",
            )
        )
        self.observation_count = int(state.get("observation_count", -1))
        self._warmup_surprisal_sum = float(
            state.get("warmup_surprisal_sum", math.nan)
        )
        if (
            self.observation_count < 0
            or not math.isfinite(self._warmup_surprisal_sum)
            or self._warmup_surprisal_sum < 0
        ):
            raise ValueError(
                "checkpoint contains invalid likelihood-controller state"
            )
