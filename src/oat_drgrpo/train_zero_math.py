# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch

from . import fused_adam_shim as _fused_adam_shim

_FUSED_ADAM_SHIM_INSTALLED = _fused_adam_shim.install()

from oat.algorithms.ppo import PPOLearner
from oat.args import default_args_validation, get_default_args

from .args import (
    ZeroMathArgs,
    validate_zero_math_args,
)
from .listwise import (
    build_padded_action_logprobs as _build_padded_action_logprobs,
)
from .learner.base import ZeroMathLearnerBaseMixin
from .learner.drx import ZeroMathDrxMixin
from .learner.grpo import ZeroMathGrpoMixin
from .learner.init import ZeroMathInitMixin
from .learner.run import ZeroMathRunMixin
from .runtime import (
    run_zero_math_rl,
)
from . import semantic_clusters as _semantic_clusters

_VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD = 0.999
build_padded_action_logprobs = _build_padded_action_logprobs

# Backward-compatible module attributes for tests and older scripts that
# monkeypatch these helpers through ``train_zero_math``.
build_connected_component_semantic_cluster_bundle = (
    _semantic_clusters.build_connected_component_semantic_cluster_bundle
)
build_semantic_cluster_bundle = _semantic_clusters.build_semantic_cluster_bundle
build_spectral_semantic_cluster_bundle = (
    _semantic_clusters.build_spectral_semantic_cluster_bundle
)


def build_runtime_semantic_cluster_bundle(
    *,
    args: Any,
    default_method: str,
    final_answer_keys_grouped,
    valid_row_mask_grouped: torch.Tensor,
    reasoning_signature_keys_grouped,
    reasoning_trace_embeddings_grouped: torch.Tensor | None,
    reasoning_trace_valid_row_mask_grouped: torch.Tensor | None,
):
    """Compatibility wrapper that honors train_zero_math-level monkeypatches."""

    return _semantic_clusters.build_runtime_semantic_cluster_bundle(
        args=args,
        default_method=default_method,
        final_answer_keys_grouped=final_answer_keys_grouped,
        valid_row_mask_grouped=valid_row_mask_grouped,
        reasoning_signature_keys_grouped=reasoning_signature_keys_grouped,
        reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
        reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
        greedy_cluster_builder=build_semantic_cluster_bundle,
        connected_component_cluster_builder=(
            build_connected_component_semantic_cluster_bundle
        ),
        spectral_cluster_builder=build_spectral_semantic_cluster_bundle,
    )


"""
4. Instantiate the learner based on PPOLearner. Here we adapt the `evaluate` logic to run multiple math benchmarks.
"""


class ZeroMathLearner(
    ZeroMathDrxMixin,
    ZeroMathGrpoMixin,
    ZeroMathLearnerBaseMixin,
    ZeroMathRunMixin,
    ZeroMathInitMixin,
    PPOLearner,
):
    """Concrete public learner assembled from narrow mixins."""


if __name__ == "__main__":
    args: ZeroMathArgs = get_default_args(ZeroMathArgs)
    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.

    args = default_args_validation(args)
    args = validate_zero_math_args(args)
    run_zero_math_rl(args)
