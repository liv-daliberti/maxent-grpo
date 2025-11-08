"""
Lightweight utilities package.

Submodules are intentionally import‑light and avoid pulling heavy optional
dependencies at ``utils`` package import time. Import submodules directly to
use their helpers:
- ``utils.data``: Dataset loading and mixture creation.
- ``utils.evaluation``: Registration and Slurm launchers for LightEval tasks.
- ``utils.hub``: Hub push helpers and simple metadata inspection utilities.
- ``utils.model_utils``: Tokenizer/model loaders with quantization and device
  map helpers.
- ``utils.vllm_patch``: Robust helpers for talking to a vLLM ``/generate``
  endpoint (streaming and schema‑agnostic decoding).
- ``utils.wandb_logging``: Minimal W&B environment initialization.

License
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the specific language governing permissions and
limitations under the License.
"""

# The __all__ below lets users discover available submodules without importing
# them eagerly.

# Expose submodules via __all__ without importing them eagerly.
__all__ = [
    "data",
    "evaluation",
    "hub",
    "model_utils",
    "trl_patches",
    "vllm_patch",
    "wandb_logging",
]
