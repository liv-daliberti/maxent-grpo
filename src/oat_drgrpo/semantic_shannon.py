"""Predictive semantic Shannon-surprisal reward shaping.

The estimator is prompt-local and catalogue-free.  It remembers only answer
keys that the policy has actually produced.  A current candidate is scored
against counts from earlier groups plus its leave-one-out peers, and the full
group is added to history only after all of its candidates have been scored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from .outcome_collision import INVALID_OUTCOME_KEY


@dataclass(frozen=True)
class SemanticShannonDiagnostics:
    """Batch diagnostics for predictive Shannon reward shaping."""

    bonus_mean: float
    bonus_min: float
    bonus_max: float
    surprisal_mean: float
    normalized_surprisal_mean: float
    entropy_mean: float
    clipped_surprisal_mean: float
    clip_fraction: float
    predictive_probability_mean: float
    predictive_probability_min: float
    predictive_probability_max: float
    normalization_error_max: float
    unseen_fraction: float
    history_total_mean: float
    distinct_outcomes_mean: float
    distinct_fraction: float
    invalid_fraction: float
    parseable_fraction: float
    tracked_prompts: float
    tracked_outcomes: float


@dataclass(frozen=True)
class SemanticShannonAdvantageDiagnostics:
    """Diagnostics for predictive-baseline semantic policy advantages."""

    predictive_baseline_mean: float
    predictive_baseline_min: float
    predictive_baseline_max: float
    predictive_baseline_normalized_mean: float
    predictive_centering_error_max: float
    advantage_scale: float
    advantage_mean: float
    advantage_min: float
    advantage_max: float
    advantage_abs_mean: float
    advantage_rms: float
    advantage_positive_fraction: float
    advantage_negative_fraction: float
    advantage_zero_fraction: float


@dataclass(frozen=True)
class SemanticShannonQualityGatedDiagnostics:
    """Diagnostics for success-conditioned, positive-only semantic pressure."""

    raw_all_row_advantage_mean: float
    raw_all_row_advantage_min: float
    raw_all_row_advantage_max: float
    raw_all_row_advantage_abs_mean: float
    raw_all_row_advantage_rms: float
    raw_all_row_advantage_positive_fraction: float
    raw_all_row_advantage_negative_fraction: float
    raw_all_row_advantage_zero_fraction: float
    effective_advantage_mean: float
    effective_advantage_min: float
    effective_advantage_max: float
    effective_advantage_abs_mean: float
    effective_advantage_rms: float
    effective_advantage_positive_fraction: float
    effective_advantage_zero_fraction: float
    eligible_fraction: float
    gated_fraction: float
    active_fraction: float
    reward_positive_fraction: float
    parseable_fraction: float
    positive_only_zeroed_fraction: float
    cap_fraction: float
    advantage_cap: float
    predictive_baseline_mean: float
    predictive_centering_error_max: float
    predictive_probability_mean: float
    predictive_probability_min: float
    predictive_probability_max: float
    normalization_error_max: float
    history_total_before_mean: float
    history_rows_added: float
    history_groups_updated: float
    history_groups_skipped: float
    tracked_prompts: float
    tracked_outcomes: float


@dataclass(frozen=True)
class SemanticShannonSuccessConditionedSignedDiagnostics:
    """Diagnostics for E43/E56 success-only signed semantic advantage."""

    raw_eligible_advantage_mean: float
    raw_eligible_advantage_min: float
    raw_eligible_advantage_max: float
    raw_eligible_advantage_abs_mean: float
    raw_eligible_advantage_rms: float
    effective_advantage_mean: float
    effective_advantage_min: float
    effective_advantage_max: float
    effective_advantage_abs_mean: float
    effective_advantage_rms: float
    effective_advantage_positive_fraction: float
    effective_advantage_negative_fraction: float
    effective_advantage_zero_fraction: float
    eligible_fraction: float
    gated_fraction: float
    active_fraction: float
    reward_positive_fraction: float
    parseable_fraction: float
    positive_cap_fraction: float
    negative_cap_fraction: float
    advantage_cap: float
    predictive_baseline_mean: float
    predictive_centering_error_max: float
    predictive_probability_mean: float
    predictive_probability_min: float
    predictive_probability_max: float
    normalization_error_max: float
    history_total_before_mean: float
    history_rows_added: float
    history_groups_updated: float
    history_groups_skipped: float
    tracked_prompts: float
    tracked_outcomes: float
    open_set_inverse_adaptation_active: float
    open_set_coefficient_used: float
    open_set_observed_normalized_entropy: float
    open_set_entropy_ema: float
    open_set_reference_entropy: float
    open_set_inverse_multiplier: float
    open_set_next_coefficient: float
    open_set_observations: float
    open_set_warmup_complete: float
    open_set_observation_skipped: float
    open_set_projection_active: float


@dataclass(frozen=True)
class OpenSetSemanticSignal:
    """Coefficient-free pressure and entropy for one successful outcome."""

    centered_clipped_surprisal: float
    normalized_predictive_entropy: float
    realized_probability: float
    predictive_entropy: float
    explicit_support_size: int


def open_set_success_semantic_signal(
    *,
    explicit_counts: dict[str, int],
    sampled_key: str,
    pseudocount: float = 1.0,
    surprisal_clip: float = 5.0,
) -> OpenSetSemanticSignal:
    """Score one validator-positive outcome against an open-set predictor.

    ``explicit_counts`` contains only prior and leave-one-out peer successful
    outcomes. One structural unseen bucket is always added. The returned
    centered pressure is coefficient-free, so a projection-free controller
    can scale it without changing this predictive distribution.
    """

    pseudocount = float(pseudocount)
    surprisal_clip = float(surprisal_clip)
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be finite and positive")
    if not math.isfinite(surprisal_clip) or surprisal_clip <= 0:
        raise ValueError("surprisal_clip must be finite and positive")
    if not explicit_counts:
        raise ValueError(
            "open-set semantic signal requires one explicit successful mode"
        )
    normalized_counts: dict[str, int] = {}
    for key, raw_count in explicit_counts.items():
        if isinstance(raw_count, bool):
            raise ValueError("explicit counts must be positive integers")
        count = int(raw_count)
        if count <= 0 or count != raw_count:
            raise ValueError("explicit counts must be positive integers")
        normalized_counts[str(key)] = count
    sampled_key = str(sampled_key)
    support_size = len(normalized_counts)
    denominator = (
        sum(normalized_counts.values())
        + pseudocount * (support_size + 1)
    )
    masses = [
        (count + pseudocount) / denominator
        for count in normalized_counts.values()
    ]
    unseen_mass = pseudocount / denominator
    masses.append(unseen_mass)
    normalization_error = abs(sum(masses) - 1.0)
    if normalization_error > 1e-12:
        raise RuntimeError("open-set predictive distribution did not normalize")

    probability = (
        (normalized_counts[sampled_key] + pseudocount) / denominator
        if sampled_key in normalized_counts
        else unseen_mass
    )
    clipped = min(-math.log(probability), surprisal_clip)
    predictive_entropy = -sum(mass * math.log(mass) for mass in masses)
    clipped_expectation = sum(
        mass * min(-math.log(mass), surprisal_clip)
        for mass in masses
    )
    normalized_entropy = predictive_entropy / math.log(support_size + 1)
    return OpenSetSemanticSignal(
        centered_clipped_surprisal=(
            clipped - clipped_expectation
        ) / surprisal_clip,
        normalized_predictive_entropy=normalized_entropy,
        realized_probability=probability,
        predictive_entropy=predictive_entropy,
        explicit_support_size=support_size,
    )


@dataclass
class OpenSetSemanticInverseController:
    """Projection-free inverse control from open-set predictive entropy."""

    base_coefficient: float
    warmup_steps: int
    ema_decay: float
    current_coefficient: float | None = None
    entropy_ema: float | None = None
    reference_entropy: float | None = None
    observation_count: int = 0
    _warmup_entropy_sum: float = 0.0

    def __post_init__(self) -> None:
        self.base_coefficient = self._positive(
            self.base_coefficient,
            "base_coefficient",
        )
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        self.warmup_steps = int(self.warmup_steps)
        self.ema_decay = float(self.ema_decay)
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be finite and in [0, 1)")
        if self.current_coefficient is None:
            self.current_coefficient = self.base_coefficient
        self.current_coefficient = self._positive(
            self.current_coefficient,
            "current_coefficient",
        )
        if self.entropy_ema is not None:
            self.entropy_ema = self._entropy(
                self.entropy_ema,
                "entropy_ema",
            )
        if self.reference_entropy is not None:
            self.reference_entropy = self._entropy(
                self.reference_entropy,
                "reference_entropy",
            )

    @staticmethod
    def _positive(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _entropy(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or not 0 < result <= 1 + 1e-6:
            raise ValueError(f"{name} must be finite and in (0, 1]")
        return min(result, 1.0)

    def observe(self, normalized_predictive_entropy: float) -> dict[str, float]:
        value = self._entropy(
            normalized_predictive_entropy,
            "normalized_predictive_entropy",
        )
        coefficient_before = float(self.current_coefficient)
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
            self.current_coefficient = self.base_coefficient
            if self.observation_count == self.warmup_steps:
                self.reference_entropy = self._positive(
                    self._warmup_entropy_sum / self.warmup_steps,
                    "warmup entropy reference",
                )
            multiplier = 1.0
        else:
            if self.reference_entropy is None:
                raise RuntimeError(
                    "open-set semantic control lacks its warmup reference"
                )
            if self.entropy_ema is None or self.entropy_ema <= 0:
                raise ValueError(
                    "open-set predictive entropy EMA must remain positive"
                )
            multiplier = self.reference_entropy / self.entropy_ema
            self.current_coefficient = self._positive(
                self.base_coefficient * multiplier,
                "current_coefficient",
            )
        diagnostics = {
            "semantic_open_set_observed_normalized_entropy": value,
            "semantic_open_set_entropy_ema": float(self.entropy_ema),
            "semantic_open_set_inverse_multiplier": float(multiplier),
            "semantic_open_set_coefficient_before": coefficient_before,
            "semantic_open_set_next_coefficient": float(
                self.current_coefficient
            ),
            "semantic_open_set_observations": float(self.observation_count),
            "semantic_open_set_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
            "semantic_open_set_projection_active": 0.0,
        }
        if self.reference_entropy is not None:
            diagnostics["semantic_open_set_reference_entropy"] = float(
                self.reference_entropy
            )
        return diagnostics

    def idle_diagnostics(self) -> dict[str, float]:
        diagnostics = {
            "semantic_open_set_inverse_multiplier": (
                float(self.current_coefficient) / self.base_coefficient
            ),
            "semantic_open_set_next_coefficient": float(
                self.current_coefficient
            ),
            "semantic_open_set_observations": float(self.observation_count),
            "semantic_open_set_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
            "semantic_open_set_projection_active": 0.0,
            "semantic_open_set_observation_skipped": 1.0,
        }
        if self.entropy_ema is not None:
            diagnostics["semantic_open_set_entropy_ema"] = float(
                self.entropy_ema
            )
        if self.reference_entropy is not None:
            diagnostics["semantic_open_set_reference_entropy"] = float(
                self.reference_entropy
            )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "semantic_open_set_inverse",
            "controller_rule": (
                "unprojected_warmup_inverse_open_set_entropy_v1"
            ),
            "base_coefficient": self.base_coefficient,
            "warmup_steps": self.warmup_steps,
            "ema_decay": self.ema_decay,
            "current_coefficient": float(self.current_coefficient),
            "entropy_ema": self.entropy_ema,
            "reference_entropy": self.reference_entropy,
            "observation_count": self.observation_count,
            "warmup_entropy_sum": self._warmup_entropy_sum,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("open-set controller state must be a dict")
        if state.get("controller_kind") != "semantic_open_set_inverse":
            raise ValueError("checkpoint contains a different controller")
        if state.get("controller_rule") != (
            "unprojected_warmup_inverse_open_set_entropy_v1"
        ):
            raise ValueError("checkpoint uses an incompatible controller rule")
        for name in ("base_coefficient", "ema_decay"):
            if not math.isclose(
                float(state.get(name, math.nan)),
                float(getattr(self, name)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"resume mismatch for {name}")
        if int(state.get("warmup_steps", -1)) != self.warmup_steps:
            raise ValueError("resume mismatch for warmup_steps")
        self.current_coefficient = self._positive(
            state.get("current_coefficient"),
            "current_coefficient",
        )
        raw_ema = state.get("entropy_ema")
        self.entropy_ema = (
            None if raw_ema is None else self._entropy(raw_ema, "entropy_ema")
        )
        raw_reference = state.get("reference_entropy")
        self.reference_entropy = (
            None
            if raw_reference is None
            else self._entropy(raw_reference, "reference_entropy")
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
            raise ValueError(
                "checkpoint contains invalid open-set controller state"
            )


def _prompt_key(prompt_token_ids: Sequence[int]) -> str:
    """Return a stable identity for one unpadded tokenized prompt."""

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


class SemanticShannonTracker:
    r"""Prompt-specific predictive answer-count estimator.

    For row ``i`` in a group of size ``G``, let ``n_x(a)`` be the historical
    count for answer ``a`` and ``m_{-i}(a)`` its count among the current
    leave-one-out peers.  The explicit support is the union of historical keys
    and peer-observed keys.  With ``K_i`` explicit keys and one reserved unseen
    bucket, the predictive probability is

    ``p_i = (n_x(a_i) + m_{-i}(a_i) + alpha) /
            (N_x + G - 1 + alpha * (K_i + 1))``

    when ``a_i`` is in the explicit support.  Otherwise the answer is scored
    through the unseen bucket with numerator ``alpha``.  The intrinsic reward
    is ``c * (min(-log(p_i), S) / S - 1)``, hence it is always in ``[-c, 0]``.
    """

    def __init__(
        self,
        *,
        coefficient: float = 0.10,
        surprisal_clip: float = 5.0,
        pseudocount: float = 1.0,
        quality_gated_advantage: bool = False,
        quality_gated_cap: float = 0.05,
        success_conditioned_signed_advantage: bool = False,
        success_conditioned_signed_cap: float = 0.05,
        open_set_inverse_adaptation: bool = False,
        open_set_warmup_steps: int = 64,
        open_set_ema_decay: float = 0.9,
    ) -> None:
        coefficient = float(coefficient)
        surprisal_clip = float(surprisal_clip)
        pseudocount = float(pseudocount)
        quality_gated_advantage = bool(quality_gated_advantage)
        quality_gated_cap = float(quality_gated_cap)
        success_conditioned_signed_advantage = bool(
            success_conditioned_signed_advantage
        )
        success_conditioned_signed_cap = float(
            success_conditioned_signed_cap
        )
        open_set_inverse_adaptation = bool(open_set_inverse_adaptation)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("coefficient must be finite and non-negative")
        if not math.isfinite(surprisal_clip) or surprisal_clip <= 0.0:
            raise ValueError("surprisal_clip must be finite and positive")
        if not math.isfinite(pseudocount) or pseudocount <= 0.0:
            raise ValueError("pseudocount must be finite and positive")
        if not math.isfinite(quality_gated_cap) or quality_gated_cap <= 0.0:
            raise ValueError("quality_gated_cap must be finite and positive")
        if (
            not math.isfinite(success_conditioned_signed_cap)
            or success_conditioned_signed_cap <= 0.0
        ):
            raise ValueError(
                "success_conditioned_signed_cap must be finite and positive"
            )
        if quality_gated_advantage and success_conditioned_signed_advantage:
            raise ValueError(
                "quality-gated and success-conditioned signed modes are "
                "mutually exclusive"
            )
        if (
            open_set_inverse_adaptation
            and not success_conditioned_signed_advantage
        ):
            raise ValueError(
                "open-set inverse adaptation requires the "
                "success-conditioned signed mode"
            )
        self.coefficient = coefficient
        self.surprisal_clip = surprisal_clip
        self.pseudocount = pseudocount
        self.quality_gated_advantage = quality_gated_advantage
        self.quality_gated_cap = quality_gated_cap
        self.success_conditioned_signed_advantage = (
            success_conditioned_signed_advantage
        )
        self.success_conditioned_signed_cap = (
            success_conditioned_signed_cap
        )
        self.open_set_inverse_adaptation = open_set_inverse_adaptation
        self._open_set_controller = (
            OpenSetSemanticInverseController(
                base_coefficient=coefficient,
                warmup_steps=open_set_warmup_steps,
                ema_decay=open_set_ema_decay,
            )
            if open_set_inverse_adaptation
            else None
        )
        self._counts: dict[str, dict[str, int]] = {}
        self._groups_scored = 0
        self._rows_scored = 0

    @property
    def tracked_prompt_count(self) -> int:
        return len(self._counts)

    @property
    def tracked_outcome_count(self) -> int:
        return sum(len(counts) for counts in self._counts.values())

    def singleton_escape_gate_diagnostics(self) -> dict[str, float]:
        """Expose a target-free collapse gate for support-only proposals.

        The gate uses the same model-derived entropy controller that scales
        the open-set semantic coefficient. It activates only after fixed
        warmup and only when the entropy EMA is below the model's own warmup
        reference. It reads no task support, desired entropy, or evaluation
        signal.
        """

        controller = self._open_set_controller
        if controller is None:
            return {
                "available": 0.0,
                "warmup_complete": 0.0,
                "entropy_below_self_reference": 0.0,
                "active": 0.0,
                "inverse_multiplier": 1.0,
            }
        warmup_complete = (
            controller.observation_count >= controller.warmup_steps
            and controller.reference_entropy is not None
            and controller.entropy_ema is not None
        )
        inverse_multiplier = (
            float(controller.current_coefficient)
            / float(controller.base_coefficient)
        )
        entropy_below_self_reference = bool(
            warmup_complete
            and float(controller.entropy_ema)
            < float(controller.reference_entropy)
        )
        return {
            "available": 1.0,
            "warmup_complete": float(warmup_complete),
            "entropy_below_self_reference": float(
                entropy_below_self_reference
            ),
            "active": float(
                entropy_below_self_reference and inverse_multiplier > 1.0
            ),
            "inverse_multiplier": inverse_multiplier,
            "entropy_ema": (
                0.0
                if controller.entropy_ema is None
                else float(controller.entropy_ema)
            ),
            "reference_entropy": (
                0.0
                if controller.reference_entropy is None
                else float(controller.reference_entropy)
            ),
            "observations": float(controller.observation_count),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return all estimator state required for exact training resume."""

        state = {
            "schema": "semantic_shannon_tracker_v1",
            "coefficient": self.coefficient,
            "surprisal_clip": self.surprisal_clip,
            "pseudocount": self.pseudocount,
            "groups_scored": self._groups_scored,
            "rows_scored": self._rows_scored,
            "counts": {
                prompt_key: dict(answer_counts)
                for prompt_key, answer_counts in self._counts.items()
            },
        }
        if self.success_conditioned_signed_advantage:
            state.update(
                {
                    "schema": (
                        "semantic_shannon_tracker_v3_"
                        "success_conditioned_signed"
                    ),
                    "success_conditioned_signed_advantage": True,
                    "success_conditioned_signed_cap": (
                        self.success_conditioned_signed_cap
                    ),
                }
            )
            if self.open_set_inverse_adaptation:
                if self._open_set_controller is None:
                    raise RuntimeError("open-set controller is missing")
                state.update(
                    {
                        "schema": (
                            "semantic_shannon_tracker_v4_"
                            "open_set_inverse"
                        ),
                        "open_set_inverse_adaptation": True,
                        "open_set_controller": (
                            self._open_set_controller.state_dict()
                        ),
                    }
                )
        elif self.quality_gated_advantage:
            state.update(
                {
                    "schema": "semantic_shannon_tracker_v2_quality_gated",
                    "quality_gated_advantage": True,
                    "quality_gated_cap": self.quality_gated_cap,
                }
            )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore counts while rejecting any estimator-contract mismatch."""

        if self.open_set_inverse_adaptation:
            expected_schema = (
                "semantic_shannon_tracker_v4_open_set_inverse"
            )
        elif self.success_conditioned_signed_advantage:
            expected_schema = (
                "semantic_shannon_tracker_v3_success_conditioned_signed"
            )
        elif self.quality_gated_advantage:
            expected_schema = "semantic_shannon_tracker_v2_quality_gated"
        else:
            expected_schema = "semantic_shannon_tracker_v1"
        if not isinstance(state, dict) or state.get("schema") != expected_schema:
            raise ValueError("invalid semantic Shannon tracker state")
        if self.success_conditioned_signed_advantage:
            if state.get("success_conditioned_signed_advantage") is not True:
                raise ValueError(
                    "semantic Shannon tracker success-conditioned signed "
                    "mode is invalid"
                )
            try:
                saved_cap = float(state["success_conditioned_signed_cap"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "semantic Shannon tracker state is missing valid "
                    "success_conditioned_signed_cap"
                ) from exc
            if not math.isfinite(saved_cap) or not math.isclose(
                saved_cap,
                self.success_conditioned_signed_cap,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "semantic Shannon resume mismatch for "
                    "success_conditioned_signed_cap: "
                    f"saved={saved_cap!r} "
                    f"configured={self.success_conditioned_signed_cap!r}"
                )
            if self.open_set_inverse_adaptation:
                if state.get("open_set_inverse_adaptation") is not True:
                    raise ValueError(
                        "semantic Shannon open-set inverse mode is invalid"
                    )
                if self._open_set_controller is None:
                    raise RuntimeError("open-set controller is missing")
                self._open_set_controller.load_state_dict(
                    state.get("open_set_controller")
                )
        elif self.quality_gated_advantage:
            if state.get("quality_gated_advantage") is not True:
                raise ValueError(
                    "semantic Shannon tracker quality-gated mode is invalid"
                )
            try:
                saved_cap = float(state["quality_gated_cap"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "semantic Shannon tracker state is missing valid "
                    "quality_gated_cap"
                ) from exc
            if not math.isfinite(saved_cap) or not math.isclose(
                saved_cap,
                self.quality_gated_cap,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "semantic Shannon resume mismatch for quality_gated_cap: "
                    f"saved={saved_cap!r} configured={self.quality_gated_cap!r}"
                )
        for name, configured in (
            ("coefficient", self.coefficient),
            ("surprisal_clip", self.surprisal_clip),
            ("pseudocount", self.pseudocount),
        ):
            try:
                saved = float(state[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"semantic Shannon tracker state is missing valid {name}"
                ) from exc
            if not math.isfinite(saved) or not math.isclose(
                saved, configured, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"semantic Shannon resume mismatch for {name}: "
                    f"saved={saved!r} configured={configured!r}"
                )

        raw_counts = state.get("counts")
        if not isinstance(raw_counts, dict):
            raise ValueError("semantic Shannon tracker counts are missing")
        restored: dict[str, dict[str, int]] = {}
        for prompt_key, answer_counts in raw_counts.items():
            if not isinstance(prompt_key, str) or not prompt_key:
                raise ValueError(
                    "semantic Shannon tracker contains an invalid prompt key"
                )
            if not isinstance(answer_counts, dict) or not answer_counts:
                raise ValueError(
                    "semantic Shannon tracker contains malformed answer counts"
                )
            restored_answers: dict[str, int] = {}
            for answer_key, raw_count in answer_counts.items():
                if not isinstance(answer_key, str):
                    raise ValueError(
                        "semantic Shannon tracker contains an invalid answer key"
                    )
                if isinstance(raw_count, bool):
                    raise ValueError("semantic Shannon tracker counts are invalid")
                try:
                    count = int(raw_count)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "semantic Shannon tracker counts are invalid"
                    ) from exc
                if count <= 0 or count != raw_count:
                    raise ValueError("semantic Shannon tracker counts are invalid")
                restored_answers[answer_key] = count
            restored[prompt_key] = restored_answers

        def _nonnegative_int(name: str) -> int:
            raw_value = state.get(name)
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"semantic Shannon tracker {name} is invalid"
                )
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"semantic Shannon tracker {name} is invalid"
                ) from exc
            if value < 0 or value != raw_value:
                raise ValueError(
                    f"semantic Shannon tracker {name} is invalid"
                )
            return value

        groups_scored = _nonnegative_int("groups_scored")
        rows_scored = _nonnegative_int("rows_scored")
        if (groups_scored == 0) != (rows_scored == 0):
            raise ValueError("semantic Shannon tracker progress is inconsistent")
        restored_row_count = sum(
            count
            for answer_counts in restored.values()
            for count in answer_counts.values()
        )
        if restored_row_count != rows_scored:
            raise ValueError(
                "semantic Shannon tracker counts do not match rows_scored"
            )
        if groups_scored > rows_scored:
            raise ValueError(
                "semantic Shannon tracker progress is inconsistent"
            )
        self._counts = restored
        self._groups_scored = groups_scored
        self._rows_scored = rows_scored

    def _score_and_update_details(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        answer_keys: Sequence[str | None],
        num_samples: int,
    ) -> tuple[
        list[float],
        list[float],
        SemanticShannonDiagnostics,
        SemanticShannonAdvantageDiagnostics,
    ]:
        """Score complete prompt groups and then add them to history."""

        if int(num_samples) <= 1:
            raise ValueError("num_samples must be greater than one")
        group_size = int(num_samples)
        if len(prompt_token_ids) != len(answer_keys):
            raise ValueError("prompt and answer rows must have matching lengths")
        if len(answer_keys) % group_size != 0:
            raise ValueError("rows must contain complete candidate groups")

        normalized_answers = [
            INVALID_OUTCOME_KEY if key is None else str(key) for key in answer_keys
        ]
        prompt_keys = [_prompt_key(ids) for ids in prompt_token_ids]
        for start in range(0, len(prompt_keys), group_size):
            group_prompt_keys = prompt_keys[start : start + group_size]
            if any(key != group_prompt_keys[0] for key in group_prompt_keys):
                raise ValueError(
                    "every candidate group must share one unpadded prompt"
                )

        bonuses: list[float] = []
        surprisals: list[float] = []
        clipped_surprisals: list[float] = []
        predictive_entropies: list[float] = []
        probabilities: list[float] = []
        clipped_rows: list[bool] = []
        unseen_rows: list[bool] = []
        history_totals: list[int] = []
        distinct_counts: list[int] = []
        normalization_errors: list[float] = []
        predictive_baselines: list[float] = []
        predictive_centering_errors: list[float] = []
        semantic_advantages: list[float] = []

        for start in range(0, len(normalized_answers), group_size):
            stop = start + group_size
            group_prompt_keys = prompt_keys[start:stop]
            prompt_key = group_prompt_keys[0]
            group_answers = normalized_answers[start:stop]
            group_counts = Counter(group_answers)
            distinct_counts.append(len(group_counts))
            history = self._counts.get(prompt_key, {})
            history_total = sum(history.values())

            for answer_key in group_answers:
                peer_counts = group_counts.copy()
                peer_counts[answer_key] -= 1
                if peer_counts[answer_key] == 0:
                    del peer_counts[answer_key]
                explicit_support = set(history) | set(peer_counts)
                support_size = len(explicit_support)
                denominator = (
                    history_total
                    + group_size
                    - 1
                    + self.pseudocount * (support_size + 1)
                )
                if answer_key in explicit_support:
                    numerator = (
                        history.get(answer_key, 0)
                        + peer_counts.get(answer_key, 0)
                        + self.pseudocount
                    )
                    unseen = False
                else:
                    numerator = self.pseudocount
                    unseen = True
                probability = numerator / denominator
                surprisal = -math.log(probability)
                clipped = min(surprisal, self.surprisal_clip)
                bonus = self.coefficient * (
                    clipped / self.surprisal_clip - 1.0
                )

                predictive_masses = [
                    (
                        history.get(key, 0)
                        + peer_counts.get(key, 0)
                        + self.pseudocount
                    )
                    / denominator
                    for key in explicit_support
                ]
                predictive_masses.append(self.pseudocount / denominator)
                normalization_error = abs(sum(predictive_masses) - 1.0)
                if not math.isfinite(normalization_error) or normalization_error > 1e-9:
                    raise RuntimeError(
                        "semantic Shannon predictive distribution did not normalize"
                    )
                predictive_entropy = -sum(
                    mass * math.log(mass) for mass in predictive_masses
                )
                predictive_baseline = sum(
                    mass
                    * min(-math.log(mass), self.surprisal_clip)
                    for mass in predictive_masses
                )
                advantage_scale = self.coefficient / self.surprisal_clip
                semantic_advantage = advantage_scale * (
                    clipped - predictive_baseline
                )
                predictive_centering_error = abs(
                    sum(
                        mass
                        * advantage_scale
                        * (
                            min(-math.log(mass), self.surprisal_clip)
                            - predictive_baseline
                        )
                        for mass in predictive_masses
                    )
                )

                probabilities.append(probability)
                surprisals.append(surprisal)
                clipped_surprisals.append(clipped)
                predictive_entropies.append(predictive_entropy)
                bonuses.append(bonus)
                clipped_rows.append(surprisal > self.surprisal_clip)
                unseen_rows.append(unseen)
                history_totals.append(history_total)
                normalization_errors.append(normalization_error)
                predictive_baselines.append(predictive_baseline)
                predictive_centering_errors.append(
                    predictive_centering_error
                )
                semantic_advantages.append(semantic_advantage)

            updated_history = dict(history)
            for answer_key, count in group_counts.items():
                updated_history[answer_key] = (
                    updated_history.get(answer_key, 0) + int(count)
                )
            self._counts[prompt_key] = updated_history
            self._groups_scored += 1
            self._rows_scored += group_size

        if not bonuses:
            diagnostics = SemanticShannonDiagnostics(
                bonus_mean=0.0,
                bonus_min=0.0,
                bonus_max=0.0,
                surprisal_mean=0.0,
                normalized_surprisal_mean=0.0,
                entropy_mean=0.0,
                clipped_surprisal_mean=0.0,
                clip_fraction=0.0,
                predictive_probability_mean=0.0,
                predictive_probability_min=0.0,
                predictive_probability_max=0.0,
                normalization_error_max=0.0,
                unseen_fraction=0.0,
                history_total_mean=0.0,
                distinct_outcomes_mean=0.0,
                distinct_fraction=0.0,
                invalid_fraction=0.0,
                parseable_fraction=0.0,
                tracked_prompts=float(self.tracked_prompt_count),
                tracked_outcomes=float(self.tracked_outcome_count),
            )
            advantage_diagnostics = SemanticShannonAdvantageDiagnostics(
                predictive_baseline_mean=0.0,
                predictive_baseline_min=0.0,
                predictive_baseline_max=0.0,
                predictive_baseline_normalized_mean=0.0,
                predictive_centering_error_max=0.0,
                advantage_scale=self.coefficient / self.surprisal_clip,
                advantage_mean=0.0,
                advantage_min=0.0,
                advantage_max=0.0,
                advantage_abs_mean=0.0,
                advantage_rms=0.0,
                advantage_positive_fraction=0.0,
                advantage_negative_fraction=0.0,
                advantage_zero_fraction=0.0,
            )
            return bonuses, semantic_advantages, diagnostics, advantage_diagnostics

        row_count = len(bonuses)
        invalid_fraction = (
            sum(key == INVALID_OUTCOME_KEY for key in normalized_answers)
            / row_count
        )
        clipped_surprisal_mean = sum(clipped_surprisals) / row_count
        diagnostics = SemanticShannonDiagnostics(
            bonus_mean=sum(bonuses) / row_count,
            bonus_min=min(bonuses),
            bonus_max=max(bonuses),
            surprisal_mean=sum(surprisals) / row_count,
            normalized_surprisal_mean=(
                clipped_surprisal_mean / self.surprisal_clip
            ),
            entropy_mean=sum(predictive_entropies) / row_count,
            clipped_surprisal_mean=clipped_surprisal_mean,
            clip_fraction=sum(clipped_rows) / row_count,
            predictive_probability_mean=sum(probabilities) / row_count,
            predictive_probability_min=min(probabilities),
            predictive_probability_max=max(probabilities),
            normalization_error_max=max(normalization_errors),
            unseen_fraction=sum(unseen_rows) / row_count,
            history_total_mean=sum(history_totals) / row_count,
            distinct_outcomes_mean=sum(distinct_counts) / len(distinct_counts),
            distinct_fraction=(
                sum(distinct_counts) / (len(distinct_counts) * group_size)
            ),
            invalid_fraction=invalid_fraction,
            parseable_fraction=1.0 - invalid_fraction,
            tracked_prompts=float(self.tracked_prompt_count),
            tracked_outcomes=float(self.tracked_outcome_count),
        )
        advantage_diagnostics = SemanticShannonAdvantageDiagnostics(
            predictive_baseline_mean=sum(predictive_baselines) / row_count,
            predictive_baseline_min=min(predictive_baselines),
            predictive_baseline_max=max(predictive_baselines),
            predictive_baseline_normalized_mean=(
                sum(predictive_baselines)
                / (row_count * self.surprisal_clip)
            ),
            predictive_centering_error_max=max(predictive_centering_errors),
            advantage_scale=self.coefficient / self.surprisal_clip,
            advantage_mean=sum(semantic_advantages) / row_count,
            advantage_min=min(semantic_advantages),
            advantage_max=max(semantic_advantages),
            advantage_abs_mean=(
                sum(abs(value) for value in semantic_advantages) / row_count
            ),
            advantage_rms=math.sqrt(
                sum(value * value for value in semantic_advantages) / row_count
            ),
            advantage_positive_fraction=(
                sum(value > 0.0 for value in semantic_advantages) / row_count
            ),
            advantage_negative_fraction=(
                sum(value < 0.0 for value in semantic_advantages) / row_count
            ),
            advantage_zero_fraction=(
                sum(value == 0.0 for value in semantic_advantages) / row_count
            ),
        )
        return bonuses, semantic_advantages, diagnostics, advantage_diagnostics

    def score_and_update(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        answer_keys: Sequence[str | None],
        num_samples: int,
    ) -> tuple[list[float], SemanticShannonDiagnostics]:
        """Return E38 reward bonuses and then add complete groups to history."""

        bonuses, _, diagnostics, _ = self._score_and_update_details(
            prompt_token_ids=prompt_token_ids,
            answer_keys=answer_keys,
            num_samples=num_samples,
        )
        return bonuses, diagnostics

    def score_separate_advantages_and_update(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        answer_keys: Sequence[str | None],
        num_samples: int,
    ) -> tuple[
        list[float],
        SemanticShannonDiagnostics,
        SemanticShannonAdvantageDiagnostics,
    ]:
        r"""Return E41 advantages centered under each predictive distribution.

        For row ``i``, this returns

        ``c / S * (clip(-log q_i(a_i), S)
                   - E_{a ~ q_i}[clip(-log q_i(a), S)])``.

        The detached predictive distribution ``q_i`` uses prompt history and
        current leave-one-out peers.  It is deliberately not the empirical
        mean of the current rollout group, so a collapsed group can retain a
        non-zero semantic learning signal.
        """

        _, advantages, diagnostics, advantage_diagnostics = (
            self._score_and_update_details(
                prompt_token_ids=prompt_token_ids,
                answer_keys=answer_keys,
                num_samples=num_samples,
            )
        )
        return advantages, diagnostics, advantage_diagnostics

    def score_quality_gated_advantages_and_update(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        answer_keys: Sequence[str | None],
        task_rewards: Sequence[float],
        active_mask: Sequence[float | bool],
        num_samples: int,
    ) -> tuple[list[float], SemanticShannonQualityGatedDiagnostics]:
        r"""Return capped positive novelty only for valid successful answers.

        The success-conditioned predictive distribution uses only historical
        and current leave-one-out outcomes whose task reward is positive and
        whose answer key is parseable. For row ``i`` the effective advantage
        is

        ``1[row_i active and reward_i > 0 and key_i parseable]
          * clamp(A_raw_i, 0, quality_gated_cap)``.

        Ineligible rows never enter current support or persistent history.
        Consequently, an all-wrong group returns exact zeros and leaves the
        tracker state unchanged. Raw-all-row diagnostics query every candidate
        against that success-only distribution solely to expose how much
        pressure the gate removed; only the effective values affect learning.
        """

        if not self.quality_gated_advantage:
            raise RuntimeError(
                "quality-gated scoring requires quality_gated_advantage=True"
            )
        if int(num_samples) <= 1:
            raise ValueError("num_samples must be greater than one")
        group_size = int(num_samples)
        row_count = len(answer_keys)
        if (
            len(prompt_token_ids) != row_count
            or len(task_rewards) != row_count
            or len(active_mask) != row_count
        ):
            raise ValueError(
                "prompt, answer, task-reward, and active-mask rows must have "
                "matching lengths"
            )
        if row_count % group_size != 0:
            raise ValueError("rows must contain complete candidate groups")

        prompt_keys = [_prompt_key(ids) for ids in prompt_token_ids]
        for start in range(0, row_count, group_size):
            group_prompt_keys = prompt_keys[start : start + group_size]
            if any(key != group_prompt_keys[0] for key in group_prompt_keys):
                raise ValueError(
                    "every candidate group must share one unpadded prompt"
                )

        normalized_rewards: list[float] = []
        for reward in task_rewards:
            if isinstance(reward, bool):
                raise ValueError("task rewards must be finite numbers")
            try:
                normalized_reward = float(reward)
            except (TypeError, ValueError) as exc:
                raise ValueError("task rewards must be finite numbers") from exc
            if not math.isfinite(normalized_reward):
                raise ValueError("task rewards must be finite numbers")
            normalized_rewards.append(normalized_reward)
        normalized_active: list[bool] = []
        for value in active_mask:
            if isinstance(value, bool):
                normalized_active.append(value)
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "active masks must be finite numbers or booleans"
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(
                    "active masks must be finite numbers or booleans"
                )
            normalized_active.append(numeric_value > 0.0)
        parseable = [
            key is not None and str(key) != INVALID_OUTCOME_KEY
            for key in answer_keys
        ]
        normalized_answers = [
            INVALID_OUTCOME_KEY if key is None else str(key) for key in answer_keys
        ]
        eligible = [
            is_active and reward > 0.0 and is_parseable
            for is_active, reward, is_parseable in zip(
                normalized_active, normalized_rewards, parseable
            )
        ]

        raw_advantages: list[float] = []
        effective_advantages: list[float] = []
        predictive_baselines: list[float] = []
        predictive_centering_errors: list[float] = []
        probabilities: list[float] = []
        normalization_errors: list[float] = []
        history_totals: list[int] = []
        positive_only_zeroed = 0
        capped = 0
        history_rows_added = 0
        history_groups_updated = 0
        history_groups_skipped = 0
        advantage_scale = self.coefficient / self.surprisal_clip

        for start in range(0, row_count, group_size):
            stop = start + group_size
            prompt_key = prompt_keys[start]
            group_answers = normalized_answers[start:stop]
            group_eligible = eligible[start:stop]
            eligible_counts = Counter(
                answer_key
                for answer_key, is_eligible in zip(
                    group_answers, group_eligible
                )
                if is_eligible
            )
            history = self._counts.get(prompt_key, {})
            history_total = sum(history.values())

            for answer_key, is_eligible in zip(
                group_answers, group_eligible
            ):
                peer_counts = eligible_counts.copy()
                if is_eligible:
                    peer_counts[answer_key] -= 1
                    if peer_counts[answer_key] == 0:
                        del peer_counts[answer_key]
                explicit_support = sorted(set(history) | set(peer_counts))
                support_size = len(explicit_support)
                peer_total = sum(peer_counts.values())
                denominator = (
                    history_total
                    + peer_total
                    + self.pseudocount * (support_size + 1)
                )
                if answer_key in explicit_support:
                    numerator = (
                        history.get(answer_key, 0)
                        + peer_counts.get(answer_key, 0)
                        + self.pseudocount
                    )
                else:
                    numerator = self.pseudocount
                probability = numerator / denominator
                clipped_surprisal = min(
                    -math.log(probability), self.surprisal_clip
                )

                predictive_masses = [
                    (
                        history.get(key, 0)
                        + peer_counts.get(key, 0)
                        + self.pseudocount
                    )
                    / denominator
                    for key in explicit_support
                ]
                predictive_masses.append(self.pseudocount / denominator)
                normalization_error = abs(sum(predictive_masses) - 1.0)
                if (
                    not math.isfinite(normalization_error)
                    or normalization_error > 1e-9
                ):
                    raise RuntimeError(
                        "semantic Shannon predictive distribution did not normalize"
                    )
                predictive_baseline = sum(
                    mass
                    * min(-math.log(mass), self.surprisal_clip)
                    for mass in predictive_masses
                )
                raw_advantage = advantage_scale * (
                    clipped_surprisal - predictive_baseline
                )
                predictive_centering_error = abs(
                    sum(
                        mass
                        * advantage_scale
                        * (
                            min(-math.log(mass), self.surprisal_clip)
                            - predictive_baseline
                        )
                        for mass in predictive_masses
                    )
                )
                if is_eligible:
                    if raw_advantage <= 0.0:
                        effective_advantage = 0.0
                        positive_only_zeroed += 1
                    elif raw_advantage > self.quality_gated_cap:
                        effective_advantage = self.quality_gated_cap
                        capped += 1
                    else:
                        effective_advantage = raw_advantage
                else:
                    effective_advantage = 0.0

                raw_advantages.append(raw_advantage)
                effective_advantages.append(effective_advantage)
                predictive_baselines.append(predictive_baseline)
                predictive_centering_errors.append(
                    predictive_centering_error
                )
                probabilities.append(probability)
                normalization_errors.append(normalization_error)
                history_totals.append(history_total)

            if eligible_counts:
                updated_history = dict(history)
                for answer_key, count in eligible_counts.items():
                    updated_history[answer_key] = (
                        updated_history.get(answer_key, 0) + int(count)
                    )
                self._counts[prompt_key] = updated_history
                eligible_row_count = sum(eligible_counts.values())
                self._groups_scored += 1
                self._rows_scored += eligible_row_count
                history_rows_added += eligible_row_count
                history_groups_updated += 1
            else:
                history_groups_skipped += 1

        def _mean(values: Sequence[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def _rms(values: Sequence[float]) -> float:
            return (
                math.sqrt(sum(value * value for value in values) / len(values))
                if values
                else 0.0
            )

        group_count = row_count // group_size
        diagnostics = SemanticShannonQualityGatedDiagnostics(
            raw_all_row_advantage_mean=_mean(raw_advantages),
            raw_all_row_advantage_min=min(raw_advantages, default=0.0),
            raw_all_row_advantage_max=max(raw_advantages, default=0.0),
            raw_all_row_advantage_abs_mean=_mean(
                [abs(value) for value in raw_advantages]
            ),
            raw_all_row_advantage_rms=_rms(raw_advantages),
            raw_all_row_advantage_positive_fraction=(
                sum(value > 0.0 for value in raw_advantages) / row_count
                if row_count
                else 0.0
            ),
            raw_all_row_advantage_negative_fraction=(
                sum(value < 0.0 for value in raw_advantages) / row_count
                if row_count
                else 0.0
            ),
            raw_all_row_advantage_zero_fraction=(
                sum(value == 0.0 for value in raw_advantages) / row_count
                if row_count
                else 0.0
            ),
            effective_advantage_mean=_mean(effective_advantages),
            effective_advantage_min=min(effective_advantages, default=0.0),
            effective_advantage_max=max(effective_advantages, default=0.0),
            effective_advantage_abs_mean=_mean(
                [abs(value) for value in effective_advantages]
            ),
            effective_advantage_rms=_rms(effective_advantages),
            effective_advantage_positive_fraction=(
                sum(value > 0.0 for value in effective_advantages) / row_count
                if row_count
                else 0.0
            ),
            effective_advantage_zero_fraction=(
                sum(value == 0.0 for value in effective_advantages) / row_count
                if row_count
                else 0.0
            ),
            eligible_fraction=(sum(eligible) / row_count if row_count else 0.0),
            gated_fraction=(
                1.0 - sum(eligible) / row_count if row_count else 0.0
            ),
            active_fraction=(
                sum(normalized_active) / row_count if row_count else 0.0
            ),
            reward_positive_fraction=(
                sum(reward > 0.0 for reward in normalized_rewards) / row_count
                if row_count
                else 0.0
            ),
            parseable_fraction=(
                sum(parseable) / row_count if row_count else 0.0
            ),
            positive_only_zeroed_fraction=(
                positive_only_zeroed / row_count if row_count else 0.0
            ),
            cap_fraction=(capped / row_count if row_count else 0.0),
            advantage_cap=self.quality_gated_cap,
            predictive_baseline_mean=_mean(predictive_baselines),
            predictive_centering_error_max=max(
                predictive_centering_errors, default=0.0
            ),
            predictive_probability_mean=_mean(probabilities),
            predictive_probability_min=min(probabilities, default=0.0),
            predictive_probability_max=max(probabilities, default=0.0),
            normalization_error_max=max(normalization_errors, default=0.0),
            history_total_before_mean=_mean(history_totals),
            history_rows_added=float(history_rows_added),
            history_groups_updated=float(history_groups_updated),
            history_groups_skipped=float(history_groups_skipped),
            tracked_prompts=float(self.tracked_prompt_count),
            tracked_outcomes=float(self.tracked_outcome_count),
        )
        if history_groups_updated + history_groups_skipped != group_count:
            raise RuntimeError(
                "semantic Shannon quality-gated group accounting failed"
            )
        return effective_advantages, diagnostics

    def score_success_conditioned_signed_advantages_and_update(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        answer_keys: Sequence[str | None],
        task_rewards: Sequence[float],
        active_mask: Sequence[float | bool],
        num_samples: int,
    ) -> tuple[
        list[float],
        SemanticShannonSuccessConditionedSignedDiagnostics,
    ]:
        r"""Return E43/E56 signed pressure on successful answers only.

        The predictor, leave-one-out support, and persistent history contain
        only active, parseable rows with positive task reward. Eligible rows
        retain the signed predictor-centered advantage

        ``clamp(A_raw_i, -success_conditioned_signed_cap,
                success_conditioned_signed_cap)``.

        In E56's open-set mode the same validator-only history is augmented by
        one structural unseen bucket. Its coefficient is controlled by that
        predictor's own normalized entropy with no coefficient projection and
        no gold support target. Every ineligible row receives exact zero and
        has no effect on scoring or history. Therefore an all-wrong group is
        an exact no-op.
        """

        if not self.success_conditioned_signed_advantage:
            raise RuntimeError(
                "success-conditioned signed scoring requires "
                "success_conditioned_signed_advantage=True"
            )
        if int(num_samples) <= 1:
            raise ValueError("num_samples must be greater than one")
        group_size = int(num_samples)
        row_count = len(answer_keys)
        if (
            len(prompt_token_ids) != row_count
            or len(task_rewards) != row_count
            or len(active_mask) != row_count
        ):
            raise ValueError(
                "prompt, answer, task-reward, and active-mask rows must have "
                "matching lengths"
            )
        if row_count % group_size != 0:
            raise ValueError("rows must contain complete candidate groups")

        prompt_keys = [_prompt_key(ids) for ids in prompt_token_ids]
        for start in range(0, row_count, group_size):
            group_prompt_keys = prompt_keys[start : start + group_size]
            if any(key != group_prompt_keys[0] for key in group_prompt_keys):
                raise ValueError(
                    "every candidate group must share one unpadded prompt"
                )

        normalized_rewards: list[float] = []
        for reward in task_rewards:
            if isinstance(reward, bool):
                raise ValueError("task rewards must be finite numbers")
            try:
                normalized_reward = float(reward)
            except (TypeError, ValueError) as exc:
                raise ValueError("task rewards must be finite numbers") from exc
            if not math.isfinite(normalized_reward):
                raise ValueError("task rewards must be finite numbers")
            normalized_rewards.append(normalized_reward)
        normalized_active: list[bool] = []
        for value in active_mask:
            if isinstance(value, bool):
                normalized_active.append(value)
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "active masks must be finite numbers or booleans"
                ) from exc
            if not math.isfinite(numeric_value):
                raise ValueError(
                    "active masks must be finite numbers or booleans"
                )
            normalized_active.append(numeric_value > 0.0)
        parseable = [
            key is not None and str(key) != INVALID_OUTCOME_KEY
            for key in answer_keys
        ]
        normalized_answers = [
            INVALID_OUTCOME_KEY if key is None else str(key) for key in answer_keys
        ]
        eligible = [
            is_active and reward > 0.0 and is_parseable
            for is_active, reward, is_parseable in zip(
                normalized_active, normalized_rewards, parseable
            )
        ]

        raw_eligible_advantages: list[float] = []
        effective_advantages: list[float] = []
        predictive_baselines: list[float] = []
        predictive_centering_errors: list[float] = []
        probabilities: list[float] = []
        normalization_errors: list[float] = []
        history_totals: list[int] = []
        positive_capped = 0
        negative_capped = 0
        history_rows_added = 0
        history_groups_updated = 0
        history_groups_skipped = 0
        open_set_controller = self._open_set_controller
        coefficient_used = (
            float(open_set_controller.current_coefficient)
            if open_set_controller is not None
            else self.coefficient
        )
        advantage_scale = coefficient_used / self.surprisal_clip
        normalized_predictive_entropies: list[float] = []

        for start in range(0, row_count, group_size):
            stop = start + group_size
            prompt_key = prompt_keys[start]
            group_answers = normalized_answers[start:stop]
            group_eligible = eligible[start:stop]
            eligible_counts = Counter(
                answer_key
                for answer_key, is_eligible in zip(
                    group_answers, group_eligible
                )
                if is_eligible
            )
            history = self._counts.get(prompt_key, {})
            history_total = sum(history.values())

            for answer_key, is_eligible in zip(
                group_answers, group_eligible
            ):
                if not is_eligible:
                    effective_advantages.append(0.0)
                    continue

                peer_counts = eligible_counts.copy()
                peer_counts[answer_key] -= 1
                if peer_counts[answer_key] == 0:
                    del peer_counts[answer_key]
                if open_set_controller is not None:
                    explicit_counts = dict(history)
                    for key, count in peer_counts.items():
                        explicit_counts[key] = (
                            explicit_counts.get(key, 0) + int(count)
                        )
                    if not explicit_counts:
                        raw_eligible_advantages.append(0.0)
                        effective_advantages.append(0.0)
                        history_totals.append(history_total)
                        continue
                    signal = open_set_success_semantic_signal(
                        explicit_counts=explicit_counts,
                        sampled_key=answer_key,
                        pseudocount=self.pseudocount,
                        surprisal_clip=self.surprisal_clip,
                    )
                    raw_advantage = (
                        coefficient_used
                        * signal.centered_clipped_surprisal
                    )
                    effective_advantage = raw_advantage
                    clipped_surprisal = min(
                        -math.log(signal.realized_probability),
                        self.surprisal_clip,
                    )
                    predictive_baseline = (
                        clipped_surprisal
                        - signal.centered_clipped_surprisal
                        * self.surprisal_clip
                    )
                    raw_eligible_advantages.append(raw_advantage)
                    effective_advantages.append(effective_advantage)
                    predictive_baselines.append(predictive_baseline)
                    predictive_centering_errors.append(0.0)
                    probabilities.append(signal.realized_probability)
                    normalization_errors.append(0.0)
                    history_totals.append(history_total)
                    normalized_predictive_entropies.append(
                        signal.normalized_predictive_entropy
                    )
                    continue
                explicit_support = sorted(set(history) | set(peer_counts))
                support_size = len(explicit_support)
                peer_total = sum(peer_counts.values())
                denominator = (
                    history_total
                    + peer_total
                    + self.pseudocount * (support_size + 1)
                )
                if answer_key in explicit_support:
                    numerator = (
                        history.get(answer_key, 0)
                        + peer_counts.get(answer_key, 0)
                        + self.pseudocount
                    )
                else:
                    numerator = self.pseudocount
                probability = numerator / denominator
                clipped_surprisal = min(
                    -math.log(probability), self.surprisal_clip
                )

                predictive_masses = [
                    (
                        history.get(key, 0)
                        + peer_counts.get(key, 0)
                        + self.pseudocount
                    )
                    / denominator
                    for key in explicit_support
                ]
                predictive_masses.append(self.pseudocount / denominator)
                normalization_error = abs(sum(predictive_masses) - 1.0)
                if (
                    not math.isfinite(normalization_error)
                    or normalization_error > 1e-9
                ):
                    raise RuntimeError(
                        "semantic Shannon predictive distribution did not normalize"
                    )
                predictive_baseline = sum(
                    mass
                    * min(-math.log(mass), self.surprisal_clip)
                    for mass in predictive_masses
                )
                raw_advantage = advantage_scale * (
                    clipped_surprisal - predictive_baseline
                )
                predictive_centering_error = abs(
                    sum(
                        mass
                        * advantage_scale
                        * (
                            min(-math.log(mass), self.surprisal_clip)
                            - predictive_baseline
                        )
                        for mass in predictive_masses
                    )
                )
                if raw_advantage > self.success_conditioned_signed_cap:
                    effective_advantage = (
                        self.success_conditioned_signed_cap
                    )
                    positive_capped += 1
                elif raw_advantage < -self.success_conditioned_signed_cap:
                    effective_advantage = (
                        -self.success_conditioned_signed_cap
                    )
                    negative_capped += 1
                else:
                    effective_advantage = raw_advantage

                raw_eligible_advantages.append(raw_advantage)
                effective_advantages.append(effective_advantage)
                predictive_baselines.append(predictive_baseline)
                predictive_centering_errors.append(
                    predictive_centering_error
                )
                probabilities.append(probability)
                normalization_errors.append(normalization_error)
                history_totals.append(history_total)

            if eligible_counts:
                updated_history = dict(history)
                for answer_key, count in eligible_counts.items():
                    updated_history[answer_key] = (
                        updated_history.get(answer_key, 0) + int(count)
                    )
                self._counts[prompt_key] = updated_history
                eligible_row_count = sum(eligible_counts.values())
                self._groups_scored += 1
                self._rows_scored += eligible_row_count
                history_rows_added += eligible_row_count
                history_groups_updated += 1
            else:
                history_groups_skipped += 1

        if open_set_controller is not None:
            if normalized_predictive_entropies:
                open_set_diagnostics = open_set_controller.observe(
                    sum(normalized_predictive_entropies)
                    / len(normalized_predictive_entropies)
                )
            else:
                open_set_diagnostics = (
                    open_set_controller.idle_diagnostics()
                )
        else:
            open_set_diagnostics = {
                "semantic_open_set_observed_normalized_entropy": 0.0,
                "semantic_open_set_entropy_ema": 0.0,
                "semantic_open_set_reference_entropy": 0.0,
                "semantic_open_set_inverse_multiplier": 1.0,
                "semantic_open_set_next_coefficient": self.coefficient,
                "semantic_open_set_observations": 0.0,
                "semantic_open_set_warmup_complete": 0.0,
                "semantic_open_set_projection_active": 0.0,
                "semantic_open_set_observation_skipped": 1.0,
            }

        def _mean(values: Sequence[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def _rms(values: Sequence[float]) -> float:
            return (
                math.sqrt(sum(value * value for value in values) / len(values))
                if values
                else 0.0
            )

        group_count = row_count // group_size
        diagnostics = SemanticShannonSuccessConditionedSignedDiagnostics(
            raw_eligible_advantage_mean=_mean(raw_eligible_advantages),
            raw_eligible_advantage_min=min(
                raw_eligible_advantages, default=0.0
            ),
            raw_eligible_advantage_max=max(
                raw_eligible_advantages, default=0.0
            ),
            raw_eligible_advantage_abs_mean=_mean(
                [abs(value) for value in raw_eligible_advantages]
            ),
            raw_eligible_advantage_rms=_rms(raw_eligible_advantages),
            effective_advantage_mean=_mean(effective_advantages),
            effective_advantage_min=min(effective_advantages, default=0.0),
            effective_advantage_max=max(effective_advantages, default=0.0),
            effective_advantage_abs_mean=_mean(
                [abs(value) for value in effective_advantages]
            ),
            effective_advantage_rms=_rms(effective_advantages),
            effective_advantage_positive_fraction=(
                sum(value > 0.0 for value in effective_advantages) / row_count
                if row_count
                else 0.0
            ),
            effective_advantage_negative_fraction=(
                sum(value < 0.0 for value in effective_advantages) / row_count
                if row_count
                else 0.0
            ),
            effective_advantage_zero_fraction=(
                sum(value == 0.0 for value in effective_advantages) / row_count
                if row_count
                else 0.0
            ),
            eligible_fraction=(sum(eligible) / row_count if row_count else 0.0),
            gated_fraction=(
                1.0 - sum(eligible) / row_count if row_count else 0.0
            ),
            active_fraction=(
                sum(normalized_active) / row_count if row_count else 0.0
            ),
            reward_positive_fraction=(
                sum(reward > 0.0 for reward in normalized_rewards) / row_count
                if row_count
                else 0.0
            ),
            parseable_fraction=(
                sum(parseable) / row_count if row_count else 0.0
            ),
            positive_cap_fraction=(
                positive_capped / row_count if row_count else 0.0
            ),
            negative_cap_fraction=(
                negative_capped / row_count if row_count else 0.0
            ),
            advantage_cap=(
                0.0
                if open_set_controller is not None
                else self.success_conditioned_signed_cap
            ),
            predictive_baseline_mean=_mean(predictive_baselines),
            predictive_centering_error_max=max(
                predictive_centering_errors, default=0.0
            ),
            predictive_probability_mean=_mean(probabilities),
            predictive_probability_min=min(probabilities, default=0.0),
            predictive_probability_max=max(probabilities, default=0.0),
            normalization_error_max=max(normalization_errors, default=0.0),
            history_total_before_mean=_mean(history_totals),
            history_rows_added=float(history_rows_added),
            history_groups_updated=float(history_groups_updated),
            history_groups_skipped=float(history_groups_skipped),
            tracked_prompts=float(self.tracked_prompt_count),
            tracked_outcomes=float(self.tracked_outcome_count),
            open_set_inverse_adaptation_active=float(
                open_set_controller is not None
            ),
            open_set_coefficient_used=coefficient_used,
            open_set_observed_normalized_entropy=float(
                open_set_diagnostics.get(
                    "semantic_open_set_observed_normalized_entropy",
                    0.0,
                )
            ),
            open_set_entropy_ema=float(
                open_set_diagnostics.get(
                    "semantic_open_set_entropy_ema",
                    0.0,
                )
            ),
            open_set_reference_entropy=float(
                open_set_diagnostics.get(
                    "semantic_open_set_reference_entropy",
                    0.0,
                )
            ),
            open_set_inverse_multiplier=float(
                open_set_diagnostics.get(
                    "semantic_open_set_inverse_multiplier",
                    1.0,
                )
            ),
            open_set_next_coefficient=float(
                open_set_diagnostics[
                    "semantic_open_set_next_coefficient"
                ]
            ),
            open_set_observations=float(
                open_set_diagnostics["semantic_open_set_observations"]
            ),
            open_set_warmup_complete=float(
                open_set_diagnostics[
                    "semantic_open_set_warmup_complete"
                ]
            ),
            open_set_observation_skipped=float(
                open_set_diagnostics.get(
                    "semantic_open_set_observation_skipped",
                    0.0,
                )
            ),
            open_set_projection_active=float(
                open_set_diagnostics[
                    "semantic_open_set_projection_active"
                ]
            ),
        )
        if history_groups_updated + history_groups_skipped != group_count:
            raise RuntimeError(
                "semantic Shannon success-conditioned signed group "
                "accounting failed"
            )
        return effective_advantages, diagnostics


def add_semantic_shannon_separate_advantage(
    task_advantages: Any,
    semantic_advantages: Any,
) -> Any:
    """Add a detached E41 semantic advantage without another centering step."""

    if task_advantages.shape != semantic_advantages.shape:
        raise ValueError(
            "task and semantic advantages must have identical shapes"
        )
    return task_advantages + semantic_advantages
