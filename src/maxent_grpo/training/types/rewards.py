# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-batch dataclasses shared across the pipeline, loss, and metrics code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import (
        Accelerator,
        EvaluationSettings,
        GenerationFn,
        PreTrainedModel,
        PreTrainedTokenizer,
        Tensor,
        torch,
    )
    from .logging import LoggingHandles
    from .runtime import RewardSpec
else:  # pragma: no cover - runtime imports are deferred in generation.helpers
    Accelerator = Any
    EvaluationSettings = Any
    GenerationFn = Any
    PreTrainedModel = Any
    PreTrainedTokenizer = Any
    Tensor = Any
    logging_stub = None
    try:
        from .runtime import torch
        from .runtime import RewardSpec
    except (
        ImportError,
        ModuleNotFoundError,
    ):  # pragma: no cover - fallback when runtime not ready
        torch = Any
        RewardSpec = Any
    try:
        from .logging import LoggingHandles
    except (ImportError, ModuleNotFoundError):

        class LoggingHandles:
            pass


@dataclass
class GenerationBatch:
    """Completions grouped per prompt after filtering."""

    prompts: List[str]
    answers: List[str]
    grouped_completions: List[List[str]]
    grouped_ref_meta: Optional[List[List[Optional[Any]]]]
    grouped_completion_info: Optional[List[List[Dict[str, Any]]]] = None


@dataclass
class PromptCompletionBatch:
    """Flattened prompt/completion pairs."""

    prompts: List[str]
    completions: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None


@dataclass
class AdvantageStats:
    """Grouped and flattened advantages."""

    grouped: List[List[float]]
    samples: List[float]


@dataclass
class QDistribution:
    """Sequence-level q-distribution."""

    grouped: List[List[float]]
    samples: List[float]


@dataclass
class RewardMoments:
    """Summary statistics for sequence rewards."""

    mean: float
    std: float


@dataclass
class RewardComputation:
    """Utility values and statistics computed per batch."""

    total_utils: List[float]
    per_reward_values: Dict[str, List[float]]
    advantage: AdvantageStats
    pairs: PromptCompletionBatch
    q_distribution: QDistribution
    moments: RewardMoments
    ref_logprob_meta: Optional[List[Optional[Any]]] = None
    completion_metadata: Optional[List[Dict[str, Any]]] = None

    @property
    def advantage_samples(self) -> List[float]:
        """Return flattened advantage samples for logging.

        :returns: Advantage samples concatenated across prompts.
        :rtype: list[float]
        """
        return self.advantage.samples

    @property
    def q_grouped(self) -> List[List[float]]:
        """Expose grouped q-values for downstream weighting.

        :returns: Per-prompt q values ready for weighting.
        :rtype: list[list[float]]
        """
        return self.q_distribution.grouped

    @property
    def train_reward_mean(self) -> float:
        """Return the cached mean reward.

        :returns: Average reward value for the processed batch.
        :rtype: float
        """
        return self.moments.mean

    @property
    def train_reward_std(self) -> float:
        """Return the cached reward standard deviation.

        :returns: Standard deviation of batch reward values.
        :rtype: float
        """
        return self.moments.std


@dataclass
class ScoreBatch:
    """Prompt cache entries and completion tokens ready for scoring."""

    prompt_entries: List["PromptCacheEntry"]
    completion_ids: Tensor
    completion_attention_mask: Tensor
    pad_token_id: int
    max_prompt_len: int
    slice_size: int
    total_sequences: int


@dataclass
class ReferenceLogprobs:
    """Reference-model log-prob summaries."""

    ref_logp_sum: Tensor
    ref_tok_counts: Tensor
    ref_logp_sum_raw: Tensor
    ref_logp_mean: float
    avg_completion_tokens: float


@dataclass
class LossScalarBundle:
    """Scalar contributions tracked for logging."""

    total_loss: float
    policy_loss: float
    clip_loss: Optional[float]
    kl_loss: float
    weighted_kl_loss: float


@dataclass
class LossOutputs:
    """Loss terms computed for a batch."""

    loss: torch.Tensor
    scalars: LossScalarBundle
    log_ratio_train: Tensor
    denom_tok_tensor: Tensor
    seed_loss: Optional[torch.Tensor] = None
    seed_loss_scalar: Optional[float] = None
    info_seed_entropy_term: Optional[torch.Tensor] = None
    info_seed_entropy_scalar: Optional[float] = None

    @property
    def total_loss_scalar(self) -> float:
        """Convenience accessor for the total loss scalar.

        :returns: Combined loss used for optimization/logging.
        :rtype: float
        """
        return self.scalars.total_loss

    @property
    def policy_loss_scalar(self) -> float:
        """Convenience accessor for the policy loss scalar.

        :returns: Policy loss contribution from ``scalars``.
        :rtype: float
        """
        return self.scalars.policy_loss

    @property
    def clip_loss_scalar(self) -> Optional[float]:
        """Convenience accessor for the clip-objective scalar.

        :returns: Optional clip objective scalar, if enabled.
        :rtype: float | None
        """
        return self.scalars.clip_loss

    @property
    def kl_loss_scalar(self) -> float:
        """Convenience accessor for the KL scalar.

        :returns: KL divergence scalar for the batch.
        :rtype: float
        """
        return self.scalars.kl_loss

    @property
    def weighted_kl_loss_scalar(self) -> float:
        """Convenience accessor for the weighted KL scalar.

        :returns: Weighted KL scalar using the configured beta.
        :rtype: float
        """
        return self.scalars.weighted_kl_loss

    @property
    def seed_loss_value(self) -> Optional[float]:
        """Convenience accessor for the auxiliary seed loss scalar."""

        return self.seed_loss_scalar

    @property
    def info_seed_entropy_value(self) -> Optional[float]:
        """Convenience accessor for the MI-style entropy term scalar."""

        return self.info_seed_entropy_scalar


@dataclass
class LengthStats:
    """Summary of completion lengths for metrics."""

    min_length: float
    mean_length: float
    max_length: float
    clipped_ratio: float
    min_terminated: float
    mean_terminated: float
    max_terminated: float


@dataclass
class BatchDiagnostics:
    """Additional scalar stats recorded for metrics."""

    kl_value: Optional[float]
    clip_ratio: float
    clip_ratio_low_mean: float
    clip_ratio_low_min: float
    clip_ratio_high_mean: float
    clip_ratio_high_max: float
    clip_ratio_region_mean: float
    kl_per_token_by_len_bucket: Dict[str, float]
    kl_token_count_by_len_bucket: Dict[str, float]


@dataclass
class ValidationContext:
    """Handles required for the optional evaluation loop."""

    evaluation: "EvaluationSettings"
    accelerator: "Accelerator"
    model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizer"
    reward: RewardSpec
    generator: "GenerationFn"
    logging: "LoggingHandles"
    eval_reward: Optional[RewardSpec] = None


@dataclass
class PromptCacheEntry:
    """Cached prompt tokenization used during scoring."""

    input_ids: List[int]
    attention_mask: List[int]

    @property
    def length(self) -> int:
        """Return cached prompt length.

        :returns: Number of tokens in the cached prompt.
        :rtype: int
        """
        return len(self.input_ids)


__all__ = [
    "AdvantageStats",
    "BatchDiagnostics",
    "GenerationBatch",
    "LengthStats",
    "LossOutputs",
    "LossScalarBundle",
    "PromptCacheEntry",
    "PromptCompletionBatch",
    "QDistribution",
    "ReferenceLogprobs",
    "RewardComputation",
    "RewardMoments",
    "ScoreBatch",
    "ValidationContext",
]
