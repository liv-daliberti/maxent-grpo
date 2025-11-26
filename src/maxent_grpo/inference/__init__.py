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

"""Inference helpers for evaluating trained models.

This package exposes utilities to run common math benchmarks (e.g., math_500,
AIME24/25, AMC, Minerva) across checkpoints produced by the GRPO or
MaxEnt-GRPO training flows. See
``maxent_grpo.pipelines.inference.inference`` for the detailed implementation
and public data classes. This module re-exports the pipeline helpers for
backwards compatibility so existing imports such as
``from maxent_grpo.inference import run_math_inference`` continue to work.
Use the CLI alias ``maxent-grpo-math-eval`` (or ``maxent-grpo-inference command=math-eval``)
to run multi-benchmark evaluations from the console.
"""

from maxent_grpo.pipelines.inference.inference import (
    INFERENCE_DATASETS,
    InferenceModelSpec,
    MathEvalConfig,
    MathInferenceResult,
    list_inference_datasets,
    run_math_eval_inference,
    run_math_inference,
    resolve_inference_dataset,
    load_math_dataset,
)

__all__ = [
    "INFERENCE_DATASETS",
    "InferenceModelSpec",
    "MathEvalConfig",
    "MathInferenceResult",
    "list_inference_datasets",
    "run_math_eval_inference",
    "resolve_inference_dataset",
    "run_math_inference",
    "load_math_dataset",
]
