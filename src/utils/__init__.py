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

# Lightweight utils package initializer.
#
# Avoid importing heavy optional dependencies at package import time to keep
# test collection and simple module imports fast and robust in minimal
# environments. Submodules should be imported directly, e.g. ``import
# utils.model_utils as MU``.

# Expose submodules via __all__ without importing them eagerly.
__all__ = [
    "data",
    "evaluation",
    "hub",
    "model_utils",
    "vllm_patch",
    "wandb_logging",
]
