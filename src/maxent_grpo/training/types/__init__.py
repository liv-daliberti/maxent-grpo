"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Public surface for the training type definitions.

from .runtime import (
    Accelerator,
    BatchingSettings,
    ClipSettings,
    ControllerPaths,
    DataLoader,
    EvaluationSettings,
    GenerationFn,
    GenerationSettings,
    SeedAugmentationConfig,
    LoopSettings,
    Optimizer,
    OptimizerHandles,
    OptimizationSchedule,
    OptimizationSettings,
    PreTrainedModel,
    PreTrainedTokenizer,
    RewardSpec,
    RuntimeHandles,
    ScoringSettings,
    StepBatchInfo,
    StepResources,
    Tensor,
    TrainingLoopContext,
    TrainingLoopState,
)
from .rewards import (
    AdvantageStats,
    BatchDiagnostics,
    GenerationBatch,
    LengthStats,
    LossOutputs,
    LossScalarBundle,
    PromptCacheEntry,
    PromptCompletionBatch,
    QDistribution,
    ReferenceLogprobs,
    RewardComputation,
    RewardMoments,
    ScoreBatch,
    ValidationContext,
)
from .logging import (
    LoggingConfigView,
    LoggingHandles,
    LogStepArtifacts,
    MetricState,
    MetricWriter,
    RewardComponentStats,
    RewardLoggingView,
    TokenUsageStats,
    TrainingMetricsPayload,
    TrainingScalarStats,
)
from ..weighting import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightLoggingView,
    WeightNormalizationSettings,
    WeightStats,
    WeightingSettings,
)

__all__ = [
    "Accelerator",
    "AdvantageStats",
    "BatchDiagnostics",
    "BatchingSettings",
    "ClipSettings",
    "ControllerPaths",
    "DataLoader",
    "EvaluationSettings",
    "GenerationBatch",
    "GenerationFn",
    "GenerationSettings",
    "KlControllerSettings",
    "LengthStats",
    "LoggingConfigView",
    "LoggingHandles",
    "LogStepArtifacts",
    "LoopSettings",
    "LossOutputs",
    "LossScalarBundle",
    "MetricState",
    "MetricWriter",
    "Optimizer",
    "OptimizerHandles",
    "OptimizationSchedule",
    "OptimizationSettings",
    "PreTrainedModel",
    "PreTrainedTokenizer",
    "PromptCacheEntry",
    "PromptCompletionBatch",
    "QDistribution",
    "QDistributionSettings",
    "ReferenceLogprobs",
    "RewardComputation",
    "RewardComponentStats",
    "RewardLoggingView",
    "RewardMoments",
    "RewardSpec",
    "RuntimeHandles",
    "ScoreBatch",
    "ScoringSettings",
    "SeedAugmentationConfig",
    "StepBatchInfo",
    "StepResources",
    "TauSchedule",
    "Tensor",
    "TokenUsageStats",
    "TrainingLoopContext",
    "TrainingLoopState",
    "TrainingMetricsPayload",
    "TrainingScalarStats",
    "ValidationContext",
    "WeightLoggingView",
    "WeightNormalizationSettings",
    "WeightStats",
    "WeightingSettings",
]
