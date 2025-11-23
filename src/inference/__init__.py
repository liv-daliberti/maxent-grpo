"""Inference helpers for evaluating trained models.

This package currently exposes utilities to run the math_500 benchmark across
multiple checkpoints produced by the GRPO or MaxEnt‑GRPO training flows.  See
``pipelines.inference.math500`` for the detailed implementation and public data
classes. This module re-exports the pipeline helpers for backwards
compatibility so existing imports such as ``from inference import
run_math500_inference`` continue to work.
"""

from pipelines.inference.math500 import (
    InferenceModelSpec,
    Math500EvalConfig,
    Math500InferenceResult,
    run_math500_inference,
    load_math500_dataset,
)

__all__ = [
    "InferenceModelSpec",
    "Math500EvalConfig",
    "Math500InferenceResult",
    "run_math500_inference",
    "load_math500_dataset",
]
