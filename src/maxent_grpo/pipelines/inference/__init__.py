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

Inference pipelines for evaluating trained checkpoints.
"""

from .inference import (
    INFERENCE_DATASETS,
    InferenceModelSpec,
    MathEvalConfig,
    MathInferenceResult,
    list_inference_datasets,
    load_math_dataset,
    run_math_eval_inference,
    resolve_inference_dataset,
    run_math_inference,
    TransformersPromptRunner,
)

__all__ = [
    "INFERENCE_DATASETS",
    "InferenceModelSpec",
    "MathEvalConfig",
    "MathInferenceResult",
    "list_inference_datasets",
    "resolve_inference_dataset",
    "run_math_eval_inference",
    "run_math_inference",
    "load_math_dataset",
    "TransformersPromptRunner",
]
