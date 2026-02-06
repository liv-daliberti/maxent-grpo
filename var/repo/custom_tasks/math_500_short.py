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

"""Custom LightEval task variants with reduced generation lengths."""

from __future__ import annotations

from lighteval.metrics.metrics import Metrics
from lighteval.tasks import default_prompts as prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig

# Override math_500 so local runs complete quickly on the HF backend. Setting a
# shorter generation_size keeps memory pressure low and dramatically reduces
# per-sample latency when debugging.
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["lighteval"],
    prompt_function=prompt.math_500,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metric=[
        Metrics.math_pass_at_1_1n,
        Metrics.math_pass_at_1_4n,
    ],
    version=3,
)

TASKS_TABLE = [math_500]
