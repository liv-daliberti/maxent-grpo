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
import importlib.machinery as _importlib_machinery
import importlib.util as _importlib_util
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, List, TYPE_CHECKING

from .controller_objective import (
    AnalyticControllerObjective,
    TruncatedBackpropControllerObjective,
)
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

LOG = logging.getLogger(__name__)

# Lazy-only declarations to satisfy ``__all__`` without eager imports.
run_training_loop: Any
PreparedBatch: Any

# Provide a lightweight wandb stub only when the real package is unavailable.
# This must not override a real installation (even if not yet imported) or the
# Transformers WandbCallback import check will fail at runtime.
_wandb_stub = sys.modules.get("wandb")
_wandb_errors = getattr(getattr(_wandb_stub, "errors", None), "Error", None)
if _wandb_errors is None:
    try:
        _wandb_mod = _importlib.import_module("wandb")
    except (ImportError, ModuleNotFoundError):
        _wandb_mod = None
    if (
        _wandb_mod is None
        or getattr(getattr(_wandb_mod, "errors", None), "Error", None) is None
    ):
        # Only install a stub when the real package is missing or malformed.
        _wandb_stub_mod = ModuleType("wandb")
        setattr(_wandb_stub_mod, "errors", SimpleNamespace(Error=RuntimeError))
        sys.modules["wandb"] = _wandb_stub_mod

# Preserve the original reload to avoid breaking downstream tooling. Store the
# first-seen implementation so subsequent reloads do not accidentally wrap the
# stubbed version and recurse.
if not hasattr(_importlib, "_original_reload"):
    setattr(_importlib, "_original_reload", _importlib.reload)
_ORIG_IMPORTLIB_RELOAD = getattr(_importlib, "_original_reload")


def _safe_reload(module):
    spec = getattr(module, "__spec__", None)
    name = getattr(spec, "name", None) if spec is not None else None
    if not name:
        name = getattr(module, "__name__", None) or str(module)
    if spec is None or getattr(spec, "name", None) is None:
        try:
            module.__spec__ = _importlib_machinery.ModuleSpec(
                name, loader=getattr(spec, "loader", None), is_package=hasattr(module, "__path__")
            )
        except (ImportError, AttributeError, ValueError):
            module.__spec__ = None
    sys.modules[name] = module
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        if parent_name not in sys.modules:
            parent_mod = ModuleType(parent_name)
            parent_mod.__path__ = []
            sys.modules[parent_name] = parent_mod
    if getattr(module, "__spec__", None) is None:
        try:
            module.__spec__ = _importlib_util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            module.__spec__ = None
    return _ORIG_IMPORTLIB_RELOAD(module)


_importlib.reload = _safe_reload

__all__: List[str] = [
    "AnalyticControllerObjective",
    "TruncatedBackpropControllerObjective",
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
    "rollout",
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
    LOG.debug("Deferring training imports until dependencies are available.")


def run_maxent_training(*args: Any, **kwargs: Any) -> Any:
    """Lazy entrypoint to the MaxEnt training pipeline.

    This wrapper defers importing the full training stack until invoked.

    :param args: Positional arguments forwarded to
        :func:`maxent_grpo.pipelines.training.maxent.run_maxent_training`.
    :param kwargs: Keyword arguments forwarded to the training pipeline.
    :returns: Whatever the underlying training pipeline returns.
    :raises Exception: Propagates exceptions raised by the training pipeline.
    """

    from maxent_grpo.pipelines.training.maxent import run_maxent_training as _run_maxent

    return _run_maxent(*args, **kwargs)


def __dir__():
    """Return sorted public attributes for IDE completion.

    :returns: Sorted list of public attribute names.
    :rtype: list[str]
    """
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    """Lazily import heavy submodules to avoid circular imports during startup.

    :param name: Attribute name to resolve.
    :returns: Imported module or attribute for ``name`` when available.
    :raises AttributeError: If the requested name is not a known lazy attribute.
    """

    if name in {"pipeline", "state", "rollout", "cli", "scoring", "optim"}:
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
    from maxent_grpo.training import pipeline as pipeline
    from maxent_grpo.training import state as state
    from maxent_grpo.training import rollout as rollout
    from maxent_grpo.training import cli as cli
    from maxent_grpo.training import scoring as scoring
