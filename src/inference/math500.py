"""Compatibility shim forwarding to ``pipelines.inference.math500``."""

from pipelines.inference.math500 import (  # noqa: F401
    InferenceModelSpec,
    Math500EvalConfig,
    Math500InferenceResult,
    PromptRunner,
    RunnerFactory,
    TransformersPromptRunner,
    load_math500_dataset,
    run_math500_inference,
)

__all__ = [
    "InferenceModelSpec",
    "Math500EvalConfig",
    "Math500InferenceResult",
    "PromptRunner",
    "RunnerFactory",
    "TransformersPromptRunner",
    "load_math500_dataset",
    "run_math500_inference",
]
