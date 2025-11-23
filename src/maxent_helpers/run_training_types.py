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

"""Type definitions and dataclasses for the MaxEnt-GRPO training loop.

The training CLI threads many nested configurations (generation, reward,
optimization, controller paths, logging).  This module centralizes the
structure so that both the CLI and documentation share consistent definitions.
Each dataclass' docstring lists the most relevant fields to aid Sphinx when
rendering reference docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .run_helpers import (
    GenerationPenaltyConfig,
    GenerationSamplingConfig,
    require_accelerator,
    require_dataloader,
    require_torch,
    require_transformer_base_classes,
)

torch = require_torch("training_types")
Tensor = torch.Tensor
DataLoader = require_dataloader("training_types")
Accelerator = require_accelerator("training_types")
PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes("training_types")

GenerationFn = Callable[
    [List[str], int, Optional[List[int]]],
    Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]],
]


@dataclass
class RuntimeHandles:
    """Pointers to objects that should live for the entire training job.

    Attributes
    ----------
    accelerator:
        Accelerate instance managing distributed state.
    model / tokenizer:
        Active policy network and tokenizer.
    train_loader / train_sampler:
        PyTorch DataLoader and optional sampler for the training set.
    device:
        Primary device (``torch.device``) used for aux buffers.
    get_ref_model:
        Callable returning the reference model when needed.
    """

    accelerator: Accelerator
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    train_loader: DataLoader
    train_sampler: Optional[Any]
    device: torch.device
    get_ref_model: Callable[[], PreTrainedModel]


@dataclass
class RewardSpec:
    """Reward functions and their aggregation weights."""

    reward_funcs: Sequence[Any]
    reward_weights: List[float]


@dataclass
class GenerationSettings(GenerationSamplingConfig):
    """Configuration for sampling completions with penalty passthrough helpers."""

    penalty: GenerationPenaltyConfig = field(default_factory=GenerationPenaltyConfig)
    generation_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def gen_top_k(self) -> Optional[int]:
        """Expose penalty mixin attributes with backwards-compatible names."""
        return self.penalty.gen_top_k

    @gen_top_k.setter
    def gen_top_k(self, value: Optional[int]) -> None:
        """Update the top-k sampling limit."""
        self.penalty.gen_top_k = value

    @property
    def gen_best_of(self) -> Optional[int]:
        """Return the best-of sampling count."""
        return self.penalty.gen_best_of

    @gen_best_of.setter
    def gen_best_of(self, value: Optional[int]) -> None:
        """Update the best-of sampling count."""
        self.penalty.gen_best_of = value

    @property
    def gen_frequency_penalty(self) -> float:
        """Return the frequency penalty strength."""
        return self.penalty.gen_frequency_penalty

    @gen_frequency_penalty.setter
    def gen_frequency_penalty(self, value: float) -> None:
        """Update the frequency penalty strength."""
        self.penalty.gen_frequency_penalty = value

    @property
    def gen_presence_penalty(self) -> float:
        """Return the presence penalty strength."""
        return self.penalty.gen_presence_penalty

    @gen_presence_penalty.setter
    def gen_presence_penalty(self, value: float) -> None:
        """Update the presence penalty strength."""
        self.penalty.gen_presence_penalty = value

    @property
    def gen_stop_sequences(self) -> Optional[List[str]]:
        """Return the configured stop sequences."""
        return self.penalty.gen_stop_sequences

    @gen_stop_sequences.setter
    def gen_stop_sequences(self, value: Optional[List[str]]) -> None:
        """Update the set of stop sequences."""
        self.penalty.gen_stop_sequences = value


@dataclass
class EvaluationSettings:
    """Optional evaluation loop configuration."""

    enabled: bool
    rows: List[Dict[str, str]]
    batch_size: int
    every_n_steps: Optional[int]


@dataclass
class OptimizationSchedule:
    """Epoch/step configuration for the trainer."""

    num_epochs: int
    num_generations: int
    grad_accum_steps: int
    max_grad_norm: float
    steps_per_epoch: Optional[int]
    total_training_steps: int
    warmup_steps: int


@dataclass
class OptimizerHandles:
    """Pointers to optimizers and schedulers."""

    optimizer: torch.optim.Optimizer
    lr_scheduler: Optional[Any]
    base_optimizer: torch.optim.Optimizer
    learning_rate: float


@dataclass
class OptimizationSettings:
    """Combined optimization metadata."""

    schedule: OptimizationSchedule
    handles: OptimizerHandles


@dataclass
class QDistributionSettings:
    """Softmax temperature and smoothing for weighting."""

    temperature: float
    epsilon: float


@dataclass
class TauSchedule:
    """Hyperparameters controlling tau adaptation."""

    target_entropy: Optional[float]
    learning_rate: float
    minimum_value: float
    maximum_value: float
    warmup_steps: int


@dataclass
class KlControllerSettings:
    """Controller settings for KL regularization."""

    target: float
    horizon: int
    step_size: float


@dataclass
class WeightNormalizationSettings:
    """Length-normalization flag and denominator scaling."""

    denom: float
    len_norm_ref: bool


@dataclass
class WeightingSettings:
    """Sequence weighting hyperparameters.

    Notes
    -----
    The convenience properties (``denom``, ``q_temperature``, etc.) mirror
    nested dataclasses so callers can treat :class:`WeightingSettings` as a flat
    struct.  Each property forwards reads/writes to the underlying structures.
    """

    tau: float
    beta: float
    normalization: WeightNormalizationSettings
    q_distribution: QDistributionSettings
    tau_schedule: TauSchedule
    kl_controller: KlControllerSettings
    train_grpo_objective: bool

    @property
    def denom(self) -> float:
        """Return the denominator used for weight normalization."""
        return self.normalization.denom

    @denom.setter
    def denom(self, value: float) -> None:
        """Update the denominator used for weight normalization."""
        self.normalization.denom = value

    @property
    def len_norm_ref(self) -> bool:
        """Return whether reference log-probs are length-normalized."""
        return self.normalization.len_norm_ref

    @len_norm_ref.setter
    def len_norm_ref(self, value: bool) -> None:
        """Update the reference length-normalization flag."""
        self.normalization.len_norm_ref = value

    @property
    def q_temperature(self) -> float:
        """Return the q-distribution temperature."""
        return self.q_distribution.temperature

    @q_temperature.setter
    def q_temperature(self, value: float) -> None:
        """Update the q-distribution temperature."""
        self.q_distribution.temperature = value

    @property
    def q_epsilon(self) -> float:
        """Return the epsilon smoothing factor."""
        return self.q_distribution.epsilon

    @q_epsilon.setter
    def q_epsilon(self, value: float) -> None:
        """Update the epsilon smoothing factor."""
        self.q_distribution.epsilon = value

    @property
    def tau_target_entropy(self) -> Optional[float]:
        """Return the target weight entropy."""
        return self.tau_schedule.target_entropy

    @tau_target_entropy.setter
    def tau_target_entropy(self, value: Optional[float]) -> None:
        """Update the target weight entropy."""
        self.tau_schedule.target_entropy = value

    @property
    def tau_lr(self) -> float:
        """Return the learning rate for tau adaptation."""
        return self.tau_schedule.learning_rate

    @tau_lr.setter
    def tau_lr(self, value: float) -> None:
        """Update the learning rate for tau adaptation."""
        self.tau_schedule.learning_rate = value

    @property
    def tau_min(self) -> float:
        """Return the minimum tau value."""
        return self.tau_schedule.minimum_value

    @tau_min.setter
    def tau_min(self, value: float) -> None:
        """Update the minimum tau value."""
        self.tau_schedule.minimum_value = value

    @property
    def tau_max(self) -> float:
        """Return the maximum tau value."""
        return self.tau_schedule.maximum_value

    @tau_max.setter
    def tau_max(self, value: float) -> None:
        """Update the maximum tau value."""
        self.tau_schedule.maximum_value = value

    @property
    def tau_warmup_steps(self) -> int:
        """Return the tau warmup horizon."""
        return self.tau_schedule.warmup_steps

    @tau_warmup_steps.setter
    def tau_warmup_steps(self, value: int) -> None:
        """Update the tau warmup horizon."""
        self.tau_schedule.warmup_steps = value

    @property
    def kl_target(self) -> float:
        """Return the KL target."""
        return self.kl_controller.target

    @kl_target.setter
    def kl_target(self, value: float) -> None:
        """Update the KL target."""
        self.kl_controller.target = value

    @property
    def kl_horizon(self) -> int:
        """Return the KL controller horizon."""
        return self.kl_controller.horizon

    @kl_horizon.setter
    def kl_horizon(self, value: int) -> None:
        """Update the KL controller horizon."""
        self.kl_controller.horizon = value

    @property
    def kl_ctl_step_size(self) -> float:
        """Return the KL controller step size."""
        return self.kl_controller.step_size

    @kl_ctl_step_size.setter
    def kl_ctl_step_size(self, value: float) -> None:
        """Update the KL controller step size."""
        self.kl_controller.step_size = value


@dataclass
class ClipSettings:
    """PPO-style clipping configuration."""

    clip_range: float
    use_clip_objective: bool
    clip_objective_coef: float
    clip_adv_baseline: Optional[float]


@dataclass
class BatchingSettings:
    """Scoring batch/chunk hints."""

    logprob_chunk_size: int
    score_slice: int
    prompt_length_cache_get: Callable[[str], PromptCacheEntry]


@dataclass
class ScoringSettings:
    """Weights, clipping, and scoring related knobs."""

    weighting: WeightingSettings
    clipping: ClipSettings
    batching: BatchingSettings


@dataclass
class LoggingHandles:
    """Callbacks for logging and checkpointing."""

    log_metrics: Callable[[Dict[str, Any], int], None]
    save_checkpoint: Callable[[str], None]
    save_strategy: str
    save_steps: int
    wandb_run: Optional[Any]


@dataclass
class ControllerPaths:
    """Filesystem locations for adaptive controller state."""

    state_path: Optional[str]
    resume_from: Optional[str]
    overwrite_existing: bool = False


@dataclass
class LoopSettings:
    """Grouped training configuration shared across the loop."""

    generation: GenerationSettings
    evaluation: EvaluationSettings
    optimization: OptimizationSettings
    scoring: ScoringSettings
    controller: ControllerPaths


@dataclass
class TrainingLoopContext:
    """Top-level container describing the full training job."""

    runtime: RuntimeHandles
    reward: RewardSpec
    settings: LoopSettings
    logging: LoggingHandles

    @property
    def generation(self) -> GenerationSettings:
        """Active generation settings."""
        return self.settings.generation

    @property
    def evaluation(self) -> EvaluationSettings:
        """Evaluation configuration."""
        return self.settings.evaluation

    @property
    def optimization(self) -> OptimizationSettings:
        """Optimization handles and schedule."""
        return self.settings.optimization

    @property
    def scoring(self) -> ScoringSettings:
        """Scoring configuration."""
        return self.settings.scoring

    @property
    def controller(self) -> ControllerPaths:
        """Adaptive controller paths."""
        return self.settings.controller


@dataclass
class GenerationBatch:
    """Completions grouped per prompt after filtering."""

    prompts: List[str]
    answers: List[str]
    grouped_completions: List[List[str]]
    grouped_ref_meta: Optional[List[List[Optional[Any]]]]


@dataclass
class PromptCompletionBatch:
    """Flattened prompt/completion pairs."""

    prompts: List[str]
    completions: List[str]


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

    @property
    def advantage_samples(self) -> List[float]:
        """Return flattened advantage samples for logging."""
        return self.advantage.samples

    @property
    def q_grouped(self) -> List[List[float]]:
        """Expose grouped q-values for downstream weighting."""
        return self.q_distribution.grouped

    @property
    def train_reward_mean(self) -> float:
        """Return the cached mean reward."""
        return self.moments.mean

    @property
    def train_reward_std(self) -> float:
        """Return the cached reward standard deviation."""
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
class WeightStats:
    """Weights per completion and entropy diagnostics."""

    weights_grouped: List[List[float]]
    flat_weights: List[float]
    weight_entropy: float
    weight_entropy_min: float
    weight_entropy_max: float
    advantage_entropy: List[float]


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

    @property
    def total_loss_scalar(self) -> float:
        """Convenience accessor for the total loss scalar."""
        return self.scalars.total_loss

    @property
    def policy_loss_scalar(self) -> float:
        """Convenience accessor for the policy loss scalar."""
        return self.scalars.policy_loss

    @property
    def clip_loss_scalar(self) -> Optional[float]:
        """Convenience accessor for the clip-objective scalar."""
        return self.scalars.clip_loss

    @property
    def kl_loss_scalar(self) -> float:
        """Convenience accessor for the KL scalar."""
        return self.scalars.kl_loss

    @property
    def weighted_kl_loss_scalar(self) -> float:
        """Convenience accessor for the weighted KL scalar."""
        return self.scalars.weighted_kl_loss


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


@dataclass
class ValidationContext:
    """Handles required for the optional evaluation loop."""

    evaluation: EvaluationSettings
    accelerator: Accelerator
    model: PreTrainedModel
    reward: RewardSpec
    generator: GenerationFn
    logging: LoggingHandles


@dataclass
class LoggingConfigView:
    """Pointers to configs referenced while logging."""

    weighting: WeightingSettings
    clipping: ClipSettings
    schedule: OptimizationSchedule


@dataclass
class TokenUsageStats:
    """Aggregate completion/input token statistics."""

    avg_completion_tokens: float
    num_completion_tokens: float
    num_input_tokens: float


@dataclass
class TrainingScalarStats:
    """Scalar values that vary every logging step."""

    ref_logp_mean: float
    tokens: TokenUsageStats
    current_lr: float
    grad_norm_scalar: Optional[float]
    epoch_progress: float
    vllm_latency_ms: Optional[float]

    @property
    def avg_completion_tokens(self) -> float:
        """Return the average completion token length."""
        return self.tokens.avg_completion_tokens

    @avg_completion_tokens.setter
    def avg_completion_tokens(self, value: float) -> None:
        """Update the average completion token length."""
        self.tokens.avg_completion_tokens = value

    @property
    def num_completion_tokens(self) -> float:
        """Return the total completion token count processed."""
        return self.tokens.num_completion_tokens

    @num_completion_tokens.setter
    def num_completion_tokens(self, value: float) -> None:
        """Update the total completion token count processed."""
        self.tokens.num_completion_tokens = value

    @property
    def num_input_tokens(self) -> float:
        """Return the total input token count processed."""
        return self.tokens.num_input_tokens

    @num_input_tokens.setter
    def num_input_tokens(self, value: float) -> None:
        """Update the total input token count processed."""
        self.tokens.num_input_tokens = value


@dataclass
class RewardComponentStats:
    """Mean/std summary for an individual reward component."""

    mean: float
    std: float


@dataclass
class RewardLoggingView:
    """Aggregated reward/advantage statistics for logging."""

    reward_mean: float
    reward_std: float
    frac_zero_std: float
    advantage_mean: float
    advantage_std: float
    advantage_count: int
    per_reward: Dict[str, RewardComponentStats]


@dataclass
class WeightLoggingView:
    """Aggregated entropy statistics for logging."""

    entropy: float
    entropy_min: float
    entropy_max: float
    advantage_entropy_mean: float
    advantage_entropy_std: float


@dataclass
class TrainingMetricsPayload:
    """Container for scalar values used by the training logger."""

    reward_stats: RewardLoggingView
    weight_stats: WeightLoggingView
    loss_outputs: LossOutputs
    diagnostics: BatchDiagnostics
    length_stats: LengthStats
    config: LoggingConfigView
    scalars: TrainingScalarStats


@dataclass
class PromptCacheEntry:
    """Cached prompt tokenization used during scoring."""

    input_ids: List[int]
    attention_mask: List[int]

    @property
    def length(self) -> int:
        """Return cached prompt length."""
        return len(self.input_ids)


__all__ = [
    "Accelerator",
    "BatchDiagnostics",
    "BatchingSettings",
    "ClipSettings",
    "DataLoader",
    "EvaluationSettings",
    "GenerationBatch",
    "GenerationFn",
    "GenerationSettings",
    "LengthStats",
    "LoggingHandles",
    "LoopSettings",
    "ControllerPaths",
    "LossScalarBundle",
    "LossOutputs",
    "OptimizerHandles",
    "OptimizationSchedule",
    "OptimizationSettings",
    "PreTrainedModel",
    "PreTrainedTokenizer",
    "PromptCompletionBatch",
    "PromptCacheEntry",
    "AdvantageStats",
    "QDistribution",
    "RewardMoments",
    "QDistributionSettings",
    "ReferenceLogprobs",
    "RewardComputation",
    "RewardSpec",
    "RewardLoggingView",
    "RewardComponentStats",
    "RuntimeHandles",
    "ScoreBatch",
    "ScoringSettings",
    "Tensor",
    "KlControllerSettings",
    "TauSchedule",
    "TokenUsageStats",
    "TrainingLoopContext",
    "TrainingMetricsPayload",
    "TrainingScalarStats",
    "ValidationContext",
    "WeightStats",
    "WeightNormalizationSettings",
    "WeightingSettings",
    "LoggingConfigView",
    "WeightLoggingView",
]
