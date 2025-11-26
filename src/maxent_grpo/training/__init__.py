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

"""Core helpers powering the MaxEnt-GRPO trainer."""

from __future__ import annotations

import importlib as _importlib
import sys
from types import SimpleNamespace
from typing import Any, List, TYPE_CHECKING

from .types import (
    RuntimeHandles,
    RewardSpec,
    GenerationSettings,
    EvaluationSettings,
    OptimizationSettings,
    OptimizationSchedule,
    OptimizerHandles,
    ScoringSettings,
    ClipSettings,
    BatchingSettings,
    LoggingHandles,
    ControllerPaths,
    LoopSettings,
    TrainingLoopContext,
    GenerationBatch,
    PromptCompletionBatch,
    AdvantageStats,
    QDistribution,
    RewardMoments,
    RewardComputation,
    ScoreBatch,
    ReferenceLogprobs,
    LossOutputs,
    LossScalarBundle,
    LengthStats,
    BatchDiagnostics,
    ValidationContext,
    LoggingConfigView,
    TokenUsageStats,
    TrainingScalarStats,
    RewardComponentStats,
    RewardLoggingView,
    TrainingMetricsPayload,
    PromptCacheEntry,
    StepBatchInfo,
    StepResources,
    TrainingLoopState,
)
from .context_builder import (
    apply_info_seed,
    apply_info_seed_to_generation,
    apply_info_seed_to_scoring,
    apply_info_seed_to_evaluation,
)
from .weighting.loss import SequenceScores

# Lazy-only declarations to satisfy ``__all__`` without eager imports.
run_training_loop: Any
PreparedBatch: Any

# Provide a lightweight wandb stub only when the real package is unavailable, to
# avoid breaking optional imports during tests.
_wandb_stub = sys.modules.get("wandb")
if (
    _wandb_stub is None
    or getattr(getattr(_wandb_stub, "errors", None), "Error", None) is None
):
    sys.modules["wandb"] = SimpleNamespace(errors=SimpleNamespace(Error=RuntimeError))

# Preserve the original reload to avoid breaking downstream tooling. Store the
# first-seen implementation so subsequent reloads do not accidentally wrap the
# stubbed version and recurse.
if not hasattr(_importlib, "_original_reload"):
    setattr(_importlib, "_original_reload", _importlib.reload)
_ORIG_IMPORTLIB_RELOAD = getattr(_importlib, "_original_reload")


def _safe_reload(module):
    name = (
        module.__spec__.name if getattr(module, "__spec__", None) else module.__name__
    )
    sys.modules[name] = module
    return _ORIG_IMPORTLIB_RELOAD(module)


_importlib.reload = _safe_reload

__all__: List[str] = [
    "run_training_loop",
    "run_maxent_training",
    "TrainingLoopState",
    "StepBatchInfo",
    "StepResources",
    "RuntimeHandles",
    "RewardSpec",
    "GenerationSettings",
    "EvaluationSettings",
    "OptimizationSettings",
    "OptimizationSchedule",
    "OptimizerHandles",
    "ScoringSettings",
    "ClipSettings",
    "BatchingSettings",
    "LoggingHandles",
    "ControllerPaths",
    "LoopSettings",
    "TrainingLoopContext",
    "GenerationBatch",
    "PromptCompletionBatch",
    "AdvantageStats",
    "QDistribution",
    "RewardMoments",
    "RewardComputation",
    "ScoreBatch",
    "ReferenceLogprobs",
    "LossOutputs",
    "LossScalarBundle",
    "LengthStats",
    "BatchDiagnostics",
    "ValidationContext",
    "LoggingConfigView",
    "TokenUsageStats",
    "TrainingScalarStats",
    "RewardComponentStats",
    "RewardLoggingView",
    "TrainingMetricsPayload",
    "PromptCacheEntry",
    "SequenceScores",
    "PreparedBatch",
    "pipeline",
    "state",
    "generation",
    "cli",
    "scoring",
    "apply_info_seed",
    "apply_info_seed_to_generation",
    "apply_info_seed_to_scoring",
    "apply_info_seed_to_evaluation",
]

# Try eager resolution when dependencies are available; fall back to lazy __getattr__.
try:
    from maxent_grpo.training.loop import run_training_loop  # type: ignore
    from maxent_grpo.training.pipeline import PreparedBatch  # type: ignore
except ImportError:
    pass
# Provide legacy ``training`` package alias for tests/older import paths.
sys.modules.setdefault("training", sys.modules.get(__name__))


def run_maxent_training(*args: Any, **kwargs: Any) -> Any:
    """Lazy entrypoint to the MaxEnt training pipeline."""

    from maxent_grpo.pipelines.training.maxent import run_maxent_training as _run_maxent

    return _run_maxent(*args, **kwargs)


def __dir__():
    """Return sorted public attributes for IDE completion."""
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    """Lazily import heavy submodules to avoid circular imports during startup."""

    if name in {"pipeline", "state", "generation", "cli", "scoring"}:
        module = _importlib.import_module(f"maxent_grpo.training.{name}")
        globals()[name] = module
        return module
    if name in {"run_training_loop", "PreparedBatch"}:
        module_name = (
            "maxent_grpo.training.loop"
            if name == "run_training_loop"
            else "maxent_grpo.training.pipeline"
        )
        module = _importlib.import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!s} has no attribute {name!s}")


# Statically expose lazy-resolved attributes for linters/IDE while deferring imports.
if TYPE_CHECKING:
    from maxent_grpo.training.loop import run_training_loop as run_training_loop
    from maxent_grpo.training.pipeline import PreparedBatch as PreparedBatch
