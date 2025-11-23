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

"""Dataclasses describing the MaxEnt-GRPO runner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from configs import GRPOConfig, GRPOScriptArguments
from .run_training_types import (
    EvaluationSettings,
    GenerationSettings,
    OptimizationSettings,
    PromptCacheEntry,
    RewardSpec,
    ScoringSettings,
)


@dataclass(frozen=True)
class FrameworkHandles:
    """Resolved runtime dependencies for the training loop."""

    torch: Any
    data_loader_cls: Any
    transformers: Any
    accelerator_cls: Any


@dataclass
class RuntimeArtifacts:
    """Pairing of resolved frameworks and the created accelerator."""

    frameworks: FrameworkHandles
    accelerator: Any


@dataclass
class ModelBundle:
    """Container for the primary model/tokenizer pair plus ref getter."""

    model: Any
    tokenizer: Any
    get_ref_model: Callable[[], Any]


@dataclass(frozen=True)
class PromptIOConfig:
    """Prompt/solution column configuration and cached tokenizer metadata."""

    prompt_column: str
    solution_column: str
    prompt_length_cache_get: Callable[[str], PromptCacheEntry]


@dataclass
class TrainDataBundle:
    """Dataset/dataloader handles and prompt metadata."""

    train_dataset: Any
    train_loader: Any
    train_sampler: Optional[Any]
    prompt_io: PromptIOConfig
    steps_per_epoch: Optional[int]
    batch_size: int

    @property
    def prompt_column(self) -> str:
        """Expose the configured prompt column without duplicating dataclass fields."""
        return self.prompt_io.prompt_column

    @property
    def solution_column(self) -> str:
        """Expose the configured solution column."""
        return self.prompt_io.solution_column

    @property
    def prompt_length_cache_get(self) -> Callable[[str], PromptCacheEntry]:
        """Expose the prompt length cache helper."""
        return self.prompt_io.prompt_length_cache_get


@dataclass(frozen=True)
class DatasetColumns:
    """Tuple describing the prompt/solution column names."""

    prompt: str
    solution: str


@dataclass(frozen=True)
class DatasetContext:
    """Bundle describing how to read and format dataset rows."""

    script_args: GRPOScriptArguments
    training_args: GRPOConfig
    tokenizer: Any


@dataclass(frozen=True)
class DataLoaderRuntimeOptions:
    """Runtime-specific DataLoader knobs bundled to limit attribute count."""

    torch_module: Optional[Any] = None
    accelerator: Optional[Any] = None
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False


@dataclass(frozen=True)
class DataPrepConfig:
    """Inputs required to build the training dataloader."""

    context: DatasetContext
    batch_size: int
    max_prompt_len: int
    data_loader_cls: Any
    runtime: DataLoaderRuntimeOptions = field(
        default_factory=DataLoaderRuntimeOptions
    )

    @property
    def torch_module(self) -> Optional[Any]:
        """Forward torch module resolution."""
        return self.runtime.torch_module

    @property
    def accelerator(self) -> Optional[Any]:
        """Forward accelerator handles."""
        return self.runtime.accelerator

    @property
    def num_workers(self) -> int:
        """Forward DataLoader worker count."""
        return self.runtime.num_workers

    @property
    def pin_memory(self) -> bool:
        """Forward the pin_memory option."""
        return self.runtime.pin_memory

    @property
    def drop_last(self) -> bool:
        """Forward the drop_last option."""
        return self.runtime.drop_last

    @property
    def persistent_workers(self) -> bool:
        """Forward the persistent_workers option."""
        return self.runtime.persistent_workers


@dataclass(frozen=True)
class LengthSettings:
    """Maximum prompt/completion lengths."""

    prompt: int
    completion: int


@dataclass(frozen=True)
class SamplingPenalties:
    """Penalty weights for decoding."""

    frequency: float = 0.0
    presence: float = 0.0


@dataclass(frozen=True)
class SamplingStopConfig:
    """Stopping and reranking configuration for decoding."""

    stop_sequences: Optional[List[str]] = None
    best_of: Optional[int] = None


@dataclass(frozen=True)
class SamplingParams:
    """Completion sampling metadata."""

    num_generations: int
    temperature: float
    top_p: float
    top_k: Optional[int] = None
    penalties: SamplingPenalties = field(default_factory=SamplingPenalties)
    stop_config: SamplingStopConfig = field(default_factory=SamplingStopConfig)

    @property
    def frequency_penalty(self) -> float:
        """Expose the frequency penalty weight."""
        return self.penalties.frequency

    @property
    def presence_penalty(self) -> float:
        """Expose the presence penalty weight."""
        return self.penalties.presence

    @property
    def stop_sequences(self) -> Optional[List[str]]:
        """Expose decoding stop sequences."""
        return self.stop_config.stop_sequences

    @property
    def best_of(self) -> Optional[int]:
        """Expose the best-of sampling parameter."""
        return self.stop_config.best_of


@dataclass(frozen=True)
class GenerationConfig:
    """Collection of sampling knobs shared by HF + vLLM."""

    lengths: LengthSettings
    sampling: SamplingParams
    use_vllm: bool
    vllm: Any  # VLLMClientConfig from run_helpers


@dataclass(frozen=True)
class HubPushConfig:
    """Describe optional pushes to the Hugging Face Hub."""

    enabled: bool
    model_id: Optional[str]
    token: Optional[str]


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint/save preferences resolved from training args."""

    output_dir: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    hub: HubPushConfig


@dataclass(frozen=True)
class EvaluationConfig:
    """Parameters describing how to run evaluation."""

    context: DatasetContext
    columns: DatasetColumns
    default_batch_size: int


@dataclass(frozen=True)
class OptimizerContext:
    """Pointers needed to create LR schedulers."""

    training_args: GRPOConfig
    optimizer: Any
    transformers_module: Any


@dataclass(frozen=True)
class TrainingScheduleConfig:
    """Simplified view of the training schedule."""

    num_epochs: int
    num_generations: int
    grad_accum_steps: int
    steps_per_epoch: Optional[int]


@dataclass(frozen=True)
class LearningConfig:
    """Basic optimizer hyperparameters."""

    learning_rate: float
    max_grad_norm: float


@dataclass(frozen=True)
class TrainingHyperParams:
    """User-friendly hyperparameter bundle pulled from GRPOConfig."""

    batch_size: int
    num_epochs: int
    grad_accum_steps: int
    learning_rate: float
    max_grad_norm: float


@dataclass
class RunnerSetup:
    """Artifacts required to construct the training loop."""

    runtime: RuntimeArtifacts
    hyperparams: TrainingHyperParams
    generation: GenerationConfig
    checkpoint: CheckpointConfig
    model_bundle: ModelBundle
    train_data: TrainDataBundle
    optimizer: Any

    @property
    def frameworks(self) -> FrameworkHandles:
        """Expose the resolved frameworks without inflating dataclass fields."""
        return self.runtime.frameworks

    @property
    def accelerator(self) -> Any:
        """Expose the accelerator handle from the runtime bundle."""
        return self.runtime.accelerator


@dataclass
class TrainingComponents:
    """High-level training components derived from the setup."""

    scoring: ScoringSettings
    evaluation: EvaluationSettings
    optimization: OptimizationSettings
    generation: GenerationSettings
    reward: RewardSpec
    wandb_config: Dict[str, Any]
    lr_warmup_steps: int


__all__ = [
    "CheckpointConfig",
    "DataLoaderRuntimeOptions",
    "DatasetColumns",
    "DatasetContext",
    "DataPrepConfig",
    "FrameworkHandles",
    "GenerationConfig",
    "HubPushConfig",
    "LearningConfig",
    "ModelBundle",
    "OptimizerContext",
    "PromptIOConfig",
    "RuntimeArtifacts",
    "RunnerSetup",
    "SamplingParams",
    "SamplingPenalties",
    "SamplingStopConfig",
    "TrainingComponents",
    "TrainingHyperParams",
    "TrainingScheduleConfig",
    "TrainDataBundle",
]
