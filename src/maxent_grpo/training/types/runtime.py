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

"""Runtime handles and configuration dataclasses for the training loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

from maxent_grpo.training.runtime import (
    GenerationSamplingConfig,
    SeedAugmentationConfig,
    require_accelerator,
    require_dataloader,
    require_torch,
    require_transformer_base_classes,
)
from maxent_grpo.training.runtime.prompts import (
    GenerationPenaltyConfig,
    GenerationPenaltyPassthroughMixin,
)
from ..weighting import WeightingSettings
from .logging import LoggingHandles
from .rewards import PromptCacheEntry

if TYPE_CHECKING:
    from .batch import ValidationContext

    try:
        import torch as torch_module
        from torch import Tensor as TorchTensor
        from torch.optim import Optimizer as TorchOptimizer
        from torch.utils.data import (
            DataLoader as TorchDataLoader,
            Sampler as TorchSampler,
        )
    except ImportError:  # pragma: no cover - typing fallback
        torch_module = Any
        TorchTensor = Any
        TorchOptimizer = Any
        TorchDataLoader = Any
        TorchSampler = Any
    try:
        from accelerate import Accelerator as HFAccelerator
    except ImportError:  # pragma: no cover - typing fallback
        HFAccelerator = Any
    try:
        from transformers import (
            PreTrainedModel as HFPreTrainedModel,
            PreTrainedTokenizer as HFPreTrainedTokenizer,
        )
    except ImportError:  # pragma: no cover - typing fallback
        HFPreTrainedModel = Any
        HFPreTrainedTokenizer = Any
else:  # pragma: no cover - runtime dependency loading
    torch_module = require_torch("training_types")
    TorchTensor = getattr(torch_module, "Tensor", Any)
    try:
        from torch.optim import Optimizer as TorchOptimizer
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
    ):  # pragma: no cover - optional stub fallback
        TorchOptimizer = Any

    TorchDataLoader = require_dataloader("training_types")
    HFAccelerator = require_accelerator("training_types")
    HFPreTrainedModel, HFPreTrainedTokenizer = require_transformer_base_classes(
        "training_types"
    )
    try:
        from torch.utils.data import Sampler as TorchSampler
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
    ):  # pragma: no cover - optional stub fallback
        TorchSampler = Any

torch = torch_module
Tensor = TorchTensor
Optimizer = TorchOptimizer
DataLoader = TorchDataLoader
Sampler = TorchSampler
Accelerator = HFAccelerator
PreTrainedModel = HFPreTrainedModel
PreTrainedTokenizer = HFPreTrainedTokenizer

GenerationFn = Callable[
    [List[str], int, Optional[List[int]]],
    Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]],
]


@dataclass
class TrainingLoopState:
    """Mutable counters that track training progress across steps.

    :ivar int global_step: Number of optimizer steps that have completed successfully.
    :ivar bool stop_training: Flag toggled when :func:`training.state.check_stop_condition`
        determines the schedule has finished.
    :ivar float num_input_tokens_seen: Running total of prompt+completion tokens.
    :ivar dict[str, float] metric_sums: Accumulators used for windowed averages.
    :ivar dict[str, int] metric_counts: Sample counts matching ``metric_sums``.
    """

    global_step: int = 0
    stop_training: bool = False
    num_input_tokens_seen: float = 0.0
    metric_sums: Dict[str, float] = field(default_factory=dict)
    metric_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class StepBatchInfo:
    """Metadata describing the in-progress batch."""

    epoch: int
    step_in_epoch: int
    batch: Dict[str, List[str]]


@dataclass
class StepResources:
    """Reusable handles for each train step."""

    generator: GenerationFn
    validation_ctx: "ValidationContext"


@dataclass
class RuntimeHandles:
    """Pointers to objects that should live for the entire training job."""

    accelerator: Accelerator
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    train_loader: DataLoader
    train_sampler: Optional[Sampler[Any]]
    device: torch.device
    get_ref_model: Callable[[], PreTrainedModel]


@dataclass
class RewardSpec:
    """Reward functions and their aggregation weights."""

    reward_funcs: Sequence[Any]
    reward_weights: List[float]


@dataclass
class GenerationSettings(GenerationPenaltyPassthroughMixin, GenerationSamplingConfig):
    """Configuration for sampling completions with penalty passthrough helpers."""

    penalty: GenerationPenaltyConfig = field(default_factory=GenerationPenaltyConfig)
    generation_stats: Dict[str, int] = field(default_factory=dict)
    seed_augmentation: Optional[SeedAugmentationConfig] = None


@dataclass
class EvaluationSettings:
    """Optional evaluation loop configuration."""

    enabled: bool
    rows: List[Dict[str, str]]
    batch_size: int
    every_n_steps: Optional[int]
    seed_eval: Optional[Dict[str, Any]] = None


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

    optimizer: Optimizer
    lr_scheduler: Optional[Any]
    base_optimizer: Optimizer
    learning_rate: float


@dataclass
class OptimizationSettings:
    """Combined optimization metadata."""

    schedule: OptimizationSchedule
    handles: OptimizerHandles


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
    prompt_length_cache_get: Callable[[str], "PromptCacheEntry"]


@dataclass
class ScoringSettings:
    """Weights, clipping, and scoring related knobs."""

    weighting: WeightingSettings
    clipping: ClipSettings
    batching: BatchingSettings
    info_seed_lambda: float = 0.0
    info_seed_temperature: float = 0.1
    info_seed_loss_type: str = "infonce"
    info_seed_pooling: str = "mean"
    info_seed_alpha_entropy: float = 0.0


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
    logging: "LoggingHandles"

    @property
    def generation(self) -> GenerationSettings:
        """Active generation settings.

        :returns: Generation configuration backing the loop.
        :rtype: GenerationSettings
        """
        return self.settings.generation

    @property
    def evaluation(self) -> EvaluationSettings:
        """Evaluation configuration.

        :returns: Evaluation scheduling and dataset pointers.
        :rtype: EvaluationSettings
        """
        return self.settings.evaluation

    @property
    def optimization(self) -> OptimizationSettings:
        """Optimization handles and schedule.

        :returns: Optimizer handles and schedule metadata.
        :rtype: OptimizationSettings
        """
        return self.settings.optimization

    @property
    def scoring(self) -> ScoringSettings:
        """Scoring configuration.

        :returns: Scoring settings (weights, chunk sizes, etc.).
        :rtype: ScoringSettings
        """
        return self.settings.scoring

    @property
    def controller(self) -> ControllerPaths:
        """Adaptive controller paths.

        :returns: Filesystem locations used by the adaptive controller.
        :rtype: ControllerPaths
        """
        return self.settings.controller


__all__ = [
    "Accelerator",
    "BatchingSettings",
    "ClipSettings",
    "ControllerPaths",
    "DataLoader",
    "EvaluationSettings",
    "GenerationFn",
    "GenerationSettings",
    "LoopSettings",
    "Optimizer",
    "OptimizerHandles",
    "OptimizationSchedule",
    "OptimizationSettings",
    "PreTrainedModel",
    "PreTrainedTokenizer",
    "RewardSpec",
    "RuntimeHandles",
    "Sampler",
    "ScoringSettings",
    "SeedAugmentationConfig",
    "Tensor",
    "TrainingLoopContext",
]
