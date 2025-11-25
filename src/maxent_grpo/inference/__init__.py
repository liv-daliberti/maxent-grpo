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

This package currently exposes utilities to run the math_500 benchmark across
multiple checkpoints produced by the GRPO or MaxEnt-GRPO training flows. See
``maxent_grpo.pipelines.inference.math500`` for the detailed implementation
and public data classes. This module re-exports the pipeline helpers for
backwards compatibility so existing imports such as
``from maxent_grpo.inference import run_math500_inference`` continue to work.
"""

from maxent_grpo.pipelines.inference.math500 import (
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
