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

import functools
import gc
import itertools
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool, TimeoutError
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
import torch.distributed as dist
import tree
from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TrajectoryData
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.ops import entropy_from_logits, masked_mean, masked_sum
from torch.utils.data import DataLoader

from datasets import load_from_disk
from .listwise import (ListwiseControllerState,
                       aggregate_masked_row_values,
                       build_listwise_q_targets,
                       build_padded_action_logprobs,
                       cap_last_valid_token_pos_for_zero_advantage,
                       clamp_listwise_tau,
                       collect_weight_entropy_stats,
                       coerce_non_negative_float,
                       compute_learnable_tau_loss,
                       compute_listwise_centered_advantages,
                       compute_listwise_clip_advantages,
                       compute_sequence_clip_coefficients,
                       compute_listwise_sequence_coefficients,
                       compute_token_level_clip_loss,
                       compute_listwise_weights,
                       compute_listwise_weights_from_utilities,
                       build_drx_target_bundle,
                       compute_drx_projection_sequence_coefficients,
                       build_semantic_cluster_bundle,
                       compute_normalized_semantic_cluster_entropy,
                       flatten_prompt_major_tensor,
                       gather_selected_logps_chunked,
                       iter_budgeted_row_chunks,
                       iter_fixed_row_chunks,
                       iter_grouped_minibatch_indices,
                       mask_and_normalize_listwise_q_targets,
                       mask_invalid_logit_columns,
                       masked_group_log_softmax,
                       maybe_update_listwise_beta,
                       maybe_update_listwise_tau,
                       normalize_maxent_clip_mode,
                       normalize_oat_objective,
                       resolve_listwise_target_entropy,
                       resolve_token_id_upper_bound,
                       reshape_prompt_major_tensor,
                       sanitize_scoring_token_ids,
                       update_listwise_tau_metric_ema)
from .math_grader import (answer_tag_reward_fn, boxed_reward_fn,
                          extract_normalized_final_answer_for_clustering,
                          extract_reasoning_signature_for_clustering,
                          extract_reasoning_trace_for_clustering)
from .semantic_features import build_candidate_response_features
from tqdm import tqdm

"""
1. To do RL from base models, we use proper prompt template to make the base model answer questions.
"""


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def apply_no_template(question: str):
    return question


TEMPLATE_FACTORY = {
    "qwen_math": apply_qwen_math_template,
    "r1": apply_r1_template,
    "no": apply_no_template,
}


def _stack_scalar_stats(
    values: list[float | torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Stack scalar stats once at the end of a learning step."""

    target_device = device
    if target_device is None:
        for value in values:
            if isinstance(value, torch.Tensor):
                target_device = value.device
                break
    if target_device is None:
        target_device = torch.device("cpu")
    if not values:
        return torch.zeros(1, dtype=torch.float32, device=target_device)
    stacked: list[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            stacked.append(
                value.detach().to(device=target_device, dtype=torch.float32).reshape(1)
            )
        else:
            stacked.append(
                torch.tensor([float(value)], device=target_device, dtype=torch.float32)
            )
    return torch.cat(stacked)


def _concat_vector_stats(
    values: list[float | torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Concatenate vector-valued stats once at the end of a learning step."""

    target_device = device
    if target_device is None:
        for value in values:
            if isinstance(value, torch.Tensor):
                target_device = value.device
                break
    if target_device is None:
        target_device = torch.device("cpu")
    if not values:
        return torch.empty(0, dtype=torch.float32, device=target_device)
    pieces: list[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            pieces.append(
                value.detach().to(device=target_device, dtype=torch.float32).reshape(-1)
            )
        else:
            pieces.append(
                torch.tensor([float(value)], device=target_device, dtype=torch.float32)
            )
    return torch.cat(pieces, dim=0) if pieces else torch.empty(
        0,
        dtype=torch.float32,
        device=target_device,
    )


def _all_gather_variable_length_1d_tensor(values: torch.Tensor) -> torch.Tensor:
    """Gather a 1D tensor with variable local length across distributed ranks."""

    flat = values.reshape(-1)
    if not dist.is_available() or not dist.is_initialized():
        return flat
    length = torch.tensor([flat.numel()], device=flat.device, dtype=torch.int64)
    world_size = dist.get_world_size()
    gathered_lengths = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, length)
    max_length = int(torch.stack(gathered_lengths).max().item())
    if max_length <= 0:
        return flat.new_zeros((0,), dtype=flat.dtype)
    padded = flat.new_zeros((max_length,), dtype=flat.dtype)
    if flat.numel() > 0:
        padded[: flat.numel()] = flat
    gathered = [flat.new_zeros((max_length,), dtype=flat.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    chunks = []
    for shard, shard_length in zip(gathered, gathered_lengths):
        shard_count = int(shard_length.item())
        if shard_count > 0:
            chunks.append(shard[:shard_count])
    return torch.cat(chunks, dim=0) if chunks else flat.new_zeros((0,), dtype=flat.dtype)


"""
2. To train reasoning models that solve math questions, we need to define an oracle (environment) that provides rule-based verification rewards.
We instantiate the oracle based on Oat's OracleBase and implement the grading logic.
"""


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the math answer grading."""

    def __init__(self, template, verifier_version) -> None:
        super().__init__()
        if template == "r1":
            math_reward_fn = answer_tag_reward_fn
        else:
            math_reward_fn = boxed_reward_fn
        self.math_reward_fn = functools.partial(
            math_reward_fn, fast=verifier_version == "fast"
        )
        # Process pool is used to enable the timeout mechanism for answer grading in our distributed training setup.
        self.mp_pool = Pool(2)

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        # Parameters used by Oat when using model-based reward, here we don't need.
        del inputs, batch_size

        rewards = []
        infos = []
        for resp, ref in zip(responses, references):
            res = self.mp_pool.apply_async(self.math_reward_fn, (resp, ref))
            try:
                info, r = res.get(timeout=1)
                rewards.append(r)
                infos.append(info)
            except TimeoutError:
                rewards.append(0.0)
                infos.append({"formatted": False})

        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


"""
2. Define extra arguments needed besides Oat's PPOArgs, mainly about choosing the prompt template.
"""


@dataclass
class ZeroMathArgs(PPOArgs):
    # Template.
    prompt_template: Literal["qwen_math", "no", "r1"] = field(default="qwen_math")
    # Evaluation benchmarks used.
    test_split: str = "all"  # Use "aime,math" to only evaluate on selected benchmarks.
    # Verifier.
    verifier_version: Literal["fast", "math_verify"] = field(default="fast")
    # Objective routing. The default keeps upstream OAT DR.GRPO unchanged.
    objective: Literal["grpo", "maxent_listwise"] = field(default="grpo")
    maxent_tau: float = 0.5
    maxent_q_temperature: float = 2.0
    maxent_q_epsilon: float = 1e-6
    maxent_candidate_kl_coef: float = 0.0
    maxent_exact_drx_weight_source: Literal[
        "sequence_clipped", "clipped", "unclipped", "local_linear"
    ] = field(default="sequence_clipped")
    maxent_length_normalize_ref: bool = True
    maxent_length_normalize_policy: bool = True
    maxent_listwise_skip_zero_variance_groups: bool = True
    maxent_use_clip_objective: bool = True
    maxent_clip_objective_coef: float = 1.0
    maxent_clip_range: float | None = None
    maxent_clip_adv_baseline: float | None = None
    maxent_clip_preserve_reward_mass: bool = False
    maxent_clip_mode: Literal["sequence", "token", "none"] = field(
        default="sequence"
    )
    maxent_token_surrogate_primary: bool = False
    maxent_drgrpo_token_primary: bool = False
    maxent_sequence_aux_coef: float = 1.0
    maxent_neutral_projection_coef: float = 0.0
    maxent_semantic_similarity_threshold: float = 0.90
    maxent_semantic_embedding_max_tokens: int = 256
    maxent_competitive_mode_tau: float = 0.05
    maxent_competitive_mode_gap: float = 0.10
    maxent_competitive_mode_top_k: int = 3
    maxent_competitive_mode_budget_max: float = 0.10
    maxent_competitive_mode_budget_scale: float = 0.05
    maxent_competitive_mode_intra_tau: float = 0.01
    maxent_prompt_select_min_alpha_frac: float = 0.5
    maxent_competitive_mode_positive_only: bool = True
    maxent_verified_distinct_bonus_coef: float = 0.5
    maxent_verified_distinct_min_modes: int = 2
    maxent_verified_distinct_reward_threshold: float = 0.999
    maxent_semantic_guard_max_expected_len_delta: float = 24.0
    maxent_semantic_guard_max_expected_format_drop: float = 0.0
    maxent_branch_grad_diagnostics: bool = False
    maxent_branch_grad_diagnostics_interval: int = 1
    maxent_branch_grad_diagnostics_max_steps: int = 0
    maxent_logprob_chunk_size: int = 2
    maxent_backward_chunk_size: int = 4
    maxent_backward_token_budget: int = 4096
    baseline_zero_adv_response_tokens: int = 8
    maxent_reference_logprobs_source: Literal["model", "behavior"] = field(
        default="model"
    )
    maxent_tau_adaptation_metric: Literal[
        "semantic_entropy_mu",
        "exploration_gain_any_correct",
        "exploration_gain_drgrpo",
    ] = field(default="semantic_entropy_mu")
    maxent_tau_target_metric: float | None = None
    maxent_tau_target_metric_start: float | None = None
    maxent_tau_target_metric_peak: float | None = None
    maxent_tau_target_metric_peak_step: int = 0
    maxent_tau_target_metric_final: float | None = None
    maxent_tau_target_metric_horizon: int = 0
    # Deprecated legacy tau-controller target fields. Keep them here only so
    # older launchers fail loudly instead of silently driving tau from H(w*).
    maxent_target_weight_entropy: float | None = None
    maxent_target_weight_entropy_start: float | None = None
    maxent_target_weight_entropy_peak: float | None = None
    maxent_target_weight_entropy_peak_step: int = 0
    maxent_target_weight_entropy_final: float | None = None
    maxent_target_weight_entropy_horizon: int = 0
    maxent_tau_learnable: bool = False
    maxent_tau_controller_enabled: bool = False
    maxent_tau_lr: float = 0.0
    maxent_tau_min: float = 0.0
    maxent_tau_max: float = 0.0
    maxent_tau_warmup_steps: int = -1
    maxent_beta_controller_enabled: bool = False
    kl_target: float = 0.0
    kl_horizon: int = 0
    kl_ctl_step_size: float = 0.0


def validate_zero_math_args(args: ZeroMathArgs) -> ZeroMathArgs:
    args.objective = normalize_oat_objective(getattr(args, "objective", "grpo"))
    if args.beta < 0:
        raise ValueError("beta must be non-negative")
    if args.kl_target < 0:
        raise ValueError("kl_target must be non-negative")
    if args.kl_horizon < 0:
        raise ValueError("kl_horizon must be non-negative")
    if args.kl_ctl_step_size < 0:
        raise ValueError("kl_ctl_step_size must be non-negative")
    if args.maxent_clip_objective_coef < 0:
        raise ValueError("maxent_clip_objective_coef must be non-negative")
    args.maxent_clip_mode = normalize_maxent_clip_mode(
        getattr(args, "maxent_clip_mode", "sequence")
    )
    if args.maxent_tau_adaptation_metric not in {
        "semantic_entropy_mu",
        "exploration_gain_any_correct",
        "exploration_gain_drgrpo",
    }:
        raise ValueError(
            "maxent_tau_adaptation_metric must be one of: "
            "semantic_entropy_mu, exploration_gain_any_correct, exploration_gain_drgrpo"
        )
    if args.maxent_tau_target_metric is not None and not math.isfinite(
        float(args.maxent_tau_target_metric)
    ):
        raise ValueError("maxent_tau_target_metric must be finite when set")
    if args.maxent_tau_target_metric_start is not None and not math.isfinite(
        float(args.maxent_tau_target_metric_start)
    ):
        raise ValueError("maxent_tau_target_metric_start must be finite when set")
    if args.maxent_tau_target_metric_peak is not None and not math.isfinite(
        float(args.maxent_tau_target_metric_peak)
    ):
        raise ValueError("maxent_tau_target_metric_peak must be finite when set")
    if args.maxent_tau_target_metric_peak_step < 0:
        raise ValueError("maxent_tau_target_metric_peak_step must be non-negative")
    if args.maxent_tau_target_metric_final is not None and not math.isfinite(
        float(args.maxent_tau_target_metric_final)
    ):
        raise ValueError("maxent_tau_target_metric_final must be finite when set")
    if args.maxent_tau_target_metric_horizon < 0:
        raise ValueError("maxent_tau_target_metric_horizon must be non-negative")
    if (
        args.maxent_tau_target_metric_peak is not None
        and args.maxent_tau_target_metric_horizon > 0
        and args.maxent_tau_target_metric_peak_step
        > args.maxent_tau_target_metric_horizon
    ):
        raise ValueError(
            "maxent_tau_target_metric_peak_step must be <= "
            "maxent_tau_target_metric_horizon"
        )
    if args.maxent_tau_lr < 0:
        raise ValueError("maxent_tau_lr must be non-negative")
    if args.maxent_tau_min < 0:
        raise ValueError("maxent_tau_min must be non-negative")
    if args.maxent_tau_max < 0:
        raise ValueError("maxent_tau_max must be non-negative")
    if args.objective != "maxent_listwise":
        if bool(args.maxent_token_surrogate_primary):
            raise ValueError(
                "maxent_token_surrogate_primary requires objective=maxent_listwise"
            )
        if bool(args.maxent_drgrpo_token_primary):
            raise ValueError(
                "maxent_drgrpo_token_primary requires objective=maxent_listwise"
            )
        if bool(args.maxent_clip_preserve_reward_mass):
            raise ValueError(
                "maxent_clip_preserve_reward_mass requires objective=maxent_listwise"
            )
        return args
    if args.critic_type != "drgrpo":
        raise ValueError("Listwise MaxEnt currently requires critic_type=drgrpo")
    if args.num_samples <= 1:
        raise ValueError("Listwise MaxEnt requires num_samples > 1")
    if args.train_batch_size_per_device <= 0:
        raise ValueError("train_batch_size_per_device must be positive")
    if args.train_batch_size_per_device % args.num_samples != 0:
        raise ValueError(
            "Listwise MaxEnt requires train_batch_size_per_device to be divisible "
            "by num_samples so each microbatch preserves whole prompt groups."
        )
    if args.maxent_tau <= 0:
        raise ValueError("Listwise MaxEnt requires maxent_tau > 0")
    if args.maxent_logprob_chunk_size < 0:
        raise ValueError("maxent_logprob_chunk_size must be non-negative")
    if args.maxent_backward_chunk_size < 0:
        raise ValueError("maxent_backward_chunk_size must be non-negative")
    if args.maxent_backward_token_budget < 0:
        raise ValueError("maxent_backward_token_budget must be non-negative")
    if args.maxent_sequence_aux_coef < 0:
        raise ValueError("maxent_sequence_aux_coef must be non-negative")
    if args.maxent_candidate_kl_coef < 0:
        raise ValueError("maxent_candidate_kl_coef must be non-negative")
    if args.maxent_neutral_projection_coef < 0:
        raise ValueError("maxent_neutral_projection_coef must be non-negative")
    if not math.isfinite(float(args.maxent_semantic_similarity_threshold)):
        raise ValueError("maxent_semantic_similarity_threshold must be finite")
    if int(args.maxent_semantic_embedding_max_tokens) <= 0:
        raise ValueError("maxent_semantic_embedding_max_tokens must be positive")
    if args.maxent_competitive_mode_tau <= 0:
        raise ValueError("maxent_competitive_mode_tau must be positive")
    if args.maxent_competitive_mode_gap < 0:
        raise ValueError("maxent_competitive_mode_gap must be non-negative")
    if args.maxent_competitive_mode_top_k <= 0:
        raise ValueError("maxent_competitive_mode_top_k must be positive")
    if args.maxent_competitive_mode_budget_max < 0:
        raise ValueError("maxent_competitive_mode_budget_max must be non-negative")
    if args.maxent_competitive_mode_budget_scale <= 0:
        raise ValueError("maxent_competitive_mode_budget_scale must be positive")
    if args.maxent_competitive_mode_intra_tau <= 0:
        raise ValueError("maxent_competitive_mode_intra_tau must be positive")
    if not 0.0 <= float(args.maxent_prompt_select_min_alpha_frac) <= 1.0:
        raise ValueError("maxent_prompt_select_min_alpha_frac must be in [0, 1]")
    if args.maxent_verified_distinct_bonus_coef < 0:
        raise ValueError("maxent_verified_distinct_bonus_coef must be non-negative")
    if args.maxent_verified_distinct_min_modes < 2:
        raise ValueError("maxent_verified_distinct_min_modes must be >= 2")
    if not 0.0 <= float(args.maxent_verified_distinct_reward_threshold) <= 1.0:
        raise ValueError("maxent_verified_distinct_reward_threshold must be in [0, 1]")
    if args.maxent_semantic_guard_max_expected_len_delta < 0:
        raise ValueError("maxent_semantic_guard_max_expected_len_delta must be non-negative")
    if args.maxent_semantic_guard_max_expected_format_drop < 0:
        raise ValueError("maxent_semantic_guard_max_expected_format_drop must be non-negative")
    if args.maxent_exact_drx_weight_source not in {
        "sequence_clipped",
        "clipped",
        "unclipped",
        "local_linear",
    }:
        raise ValueError(
            "maxent_exact_drx_weight_source must be one of: "
            "sequence_clipped, clipped, unclipped, local_linear"
        )
    if args.maxent_branch_grad_diagnostics_interval <= 0:
        raise ValueError("maxent_branch_grad_diagnostics_interval must be positive")
    if args.maxent_branch_grad_diagnostics_max_steps < 0:
        raise ValueError("maxent_branch_grad_diagnostics_max_steps must be non-negative")
    if args.baseline_zero_adv_response_tokens < 0:
        raise ValueError("baseline_zero_adv_response_tokens must be non-negative")
    if bool(args.maxent_token_surrogate_primary):
        if not bool(args.maxent_use_clip_objective):
            raise ValueError(
                "maxent_token_surrogate_primary requires maxent_use_clip_objective"
            )
        if args.maxent_clip_mode != "token":
            raise ValueError(
                "maxent_token_surrogate_primary requires maxent_clip_mode=token"
            )
        if args.maxent_clip_objective_coef <= 0:
            raise ValueError(
                "maxent_token_surrogate_primary requires "
                "maxent_clip_objective_coef > 0"
            )
    if bool(args.maxent_drgrpo_token_primary):
        if bool(args.maxent_token_surrogate_primary):
            raise ValueError(
                "maxent_drgrpo_token_primary cannot be combined with "
                "maxent_token_surrogate_primary"
            )
        if args.beta > 0:
            raise ValueError(
                "maxent_drgrpo_token_primary currently requires beta=0; use "
                "maxent_candidate_kl_coef for the candidate-level trust region."
            )
    if (
        args.maxent_tau_max > 0
        and args.maxent_tau_min > 0
        and args.maxent_tau_max < args.maxent_tau_min
    ):
        raise ValueError("maxent_tau_max must be >= maxent_tau_min when both are positive")
    legacy_tau_target_fields = (
        args.maxent_target_weight_entropy,
        args.maxent_target_weight_entropy_start,
        args.maxent_target_weight_entropy_peak,
        args.maxent_target_weight_entropy_final,
        args.maxent_target_weight_entropy_peak_step
        if int(args.maxent_target_weight_entropy_peak_step) != 0
        else None,
        args.maxent_target_weight_entropy_horizon
        if int(args.maxent_target_weight_entropy_horizon) != 0
        else None,
    )
    using_legacy_tau_target = any(value is not None for value in legacy_tau_target_fields)
    if bool(args.maxent_tau_learnable) or bool(args.maxent_tau_controller_enabled):
        if using_legacy_tau_target:
            raise ValueError(
                "Adaptive listwise tau no longer accepts "
                "maxent_target_weight_entropy*; use maxent_tau_target_metric* "
                "with maxent_tau_adaptation_metric set to a rollout-side signal."
            )
        if (
            args.maxent_tau_target_metric is None
            and args.maxent_tau_target_metric_start is None
            and args.maxent_tau_target_metric_peak is None
            and args.maxent_tau_target_metric_final is None
        ):
            raise ValueError(
                "Adaptive listwise tau requires maxent_tau_target_metric* to be set."
            )
        if args.maxent_tau_lr <= 0:
            raise ValueError("Adaptive listwise tau requires maxent_tau_lr > 0")
    return args


"""
3. Instantiate the actor based on Oat's PPOActor, which controls the reasoning trace generation (`self.sampling_params`) and the rewarding (`self.oracle`).
"""


class ZeroMathActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: ZeroMathArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        # OAT 0.0.9 configures actor sampling in __init__, while newer OAT
        # versions populate these fields in init(actor_id, save_path).
        if hasattr(self, "sampling_params"):
            self._configure_math_actor()

    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self._configure_math_actor()

    def _configure_math_actor(self) -> None:
        self.oracle = MATHOracle(
            template=self.args.prompt_template,
            verifier_version=self.args.verifier_version,
        )

        if self.args.prompt_template in ["qwen_math", "no"]:
            # These two templates are better used for Qwen models, which can themselves stop generation. Hence we unset all external stopping conditions.
            self.sampling_params.stop = None
            self.sampling_params.stop_token_ids = None
            self.eval_sampling_params.stop = None
            self.eval_sampling_params.stop_token_ids = None
        elif self.args.prompt_template == "r1":
            # Let's stop when the model completes its answer.
            self.sampling_params.stop = ["</answer>"]
            self.sampling_params.include_stop_str_in_output = True
            self.eval_sampling_params.stop = ["</answer>"]
            self.eval_sampling_params.include_stop_str_in_output = True

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        """Main logic for the actor to generate trajectories (reasoning traces)."""
        assert not self.eval_mode
        info = {}
        logging.info(f"actor start")

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        resp_lens = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(outputs[i].outputs[k].finish_reason == "length")
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]
                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))

        info["actor/generate_time"] = time.time() - st

        # step 2. verify
        st = time.time()
        rewards, oracle_infos = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )

        info["actor/verify_time"] = time.time() - st
        logging.info(f"actor reward {rewards.mean()}")
        info["actor/rewards"] = rewards.mean().item()
        info["actor/num_data"] = rewards.numel()
        info["actor/formatted"] = np.mean([i["formatted"] for i in oracle_infos])
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)
        info["actor/no_eos_count"] = no_eos.sum()

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j].item()
                if no_eos[i][j]:
                    # Set zero reward for truncated outputs.
                    reward = 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory_data.append(
                    TrajectoryData(
                        prompt=prompt,
                        prompt_ids=prompt_token_ids[i],
                        response=candidates_per_prompt[j],
                        response_ids=response_ids[i][j],
                        response_logprobs=response_logprobs[i][j],
                        rewards=dense_rewards,
                        loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                        info=info,
                    )
                )
        logging.info(f"actor finished data_len={len(trajectory_data)}")
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle


"""
4. Instantiate the learner based on PPOLearner. Here we adapt the `evaluate` logic to run multiple math benchmarks.
"""


class ZeroMathLearner(PPOLearner):
    def _init(self, args: ZeroMathArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.eval_dataset_dict = load_from_disk(args.eval_data)  # TODO: get fro HF.
        if args.test_split != "all":
            self.eval_dataset_dict = {
                k: v for k, v in self.eval_dataset_dict.items() if k in args.test_split
            }
        self.args = args
        # Dr. GRPO Modification 1: Remove length bias by using masked_sum with a constant normalizer:
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )
        self.objective = normalize_oat_objective(args.objective)
        self._listwise_grad_norm_logging_disabled_warned = False
        self._listwise_zero_signal_skip_warned = False
        self._listwise_backward_token_budget_safety_warned = False
        self._listwise_branch_grad_probe_warned = False
        self._listwise_branch_grad_probe_runtime_disabled = False
        self._baseline_grad_norm_logging_disabled_warned = False
        self._invalid_scoring_token_ids_warned_contexts = set()
        self._invalid_logit_columns_warned_contexts = set()
        self._fixed_listwise_tau: float | None = None
        self._fixed_listwise_beta: float | None = None
        self._fixed_listwise_config: dict[str, Any] = {}
        self._policy_grad_probe_params: tuple[torch.nn.Parameter, ...] | None = None
        if self.objective == "maxent_listwise":
            self._fixed_listwise_config = {
                "maxent_q_temperature": float(args.maxent_q_temperature),
                "maxent_q_epsilon": float(args.maxent_q_epsilon),
                "maxent_candidate_kl_coef": float(args.maxent_candidate_kl_coef),
                "maxent_exact_drx_weight_source": str(
                    args.maxent_exact_drx_weight_source
                ),
                "maxent_length_normalize_ref": bool(args.maxent_length_normalize_ref),
                "maxent_length_normalize_policy": bool(
                    args.maxent_length_normalize_policy
                ),
                "maxent_listwise_skip_zero_variance_groups": bool(
                    args.maxent_listwise_skip_zero_variance_groups
                ),
                "maxent_use_clip_objective": bool(args.maxent_use_clip_objective),
                "maxent_clip_objective_coef": float(args.maxent_clip_objective_coef),
                "maxent_clip_range": (
                    None
                    if args.maxent_clip_range is None
                    else float(args.maxent_clip_range)
                ),
                "maxent_clip_adv_baseline": (
                    None
                    if args.maxent_clip_adv_baseline is None
                    else float(args.maxent_clip_adv_baseline)
                ),
                "maxent_clip_preserve_reward_mass": bool(
                    args.maxent_clip_preserve_reward_mass
                ),
                "maxent_clip_mode": str(args.maxent_clip_mode),
                "maxent_token_surrogate_primary": bool(
                    args.maxent_token_surrogate_primary
                ),
                "maxent_drgrpo_token_primary": bool(
                    args.maxent_drgrpo_token_primary
                ),
                "maxent_sequence_aux_coef": float(args.maxent_sequence_aux_coef),
                "maxent_neutral_projection_coef": float(
                    args.maxent_neutral_projection_coef
                ),
                "maxent_semantic_similarity_threshold": float(
                    args.maxent_semantic_similarity_threshold
                ),
                "maxent_semantic_embedding_max_tokens": int(
                    args.maxent_semantic_embedding_max_tokens
                ),
                "maxent_branch_grad_diagnostics": bool(
                    args.maxent_branch_grad_diagnostics
                ),
                "maxent_branch_grad_diagnostics_interval": int(
                    args.maxent_branch_grad_diagnostics_interval
                ),
                "maxent_branch_grad_diagnostics_max_steps": int(
                    args.maxent_branch_grad_diagnostics_max_steps
                ),
                "maxent_logprob_chunk_size": int(args.maxent_logprob_chunk_size),
                "maxent_backward_chunk_size": int(args.maxent_backward_chunk_size),
                "maxent_backward_token_budget": int(
                    args.maxent_backward_token_budget
                ),
                "maxent_reference_logprobs_source": str(
                    args.maxent_reference_logprobs_source
                ),
                "maxent_tau_adaptation_metric": str(
                    args.maxent_tau_adaptation_metric
                ),
                "maxent_tau_target_metric": (
                    None
                    if args.maxent_tau_target_metric is None
                    else float(args.maxent_tau_target_metric)
                ),
                "maxent_tau_target_metric_start": (
                    None
                    if args.maxent_tau_target_metric_start is None
                    else float(args.maxent_tau_target_metric_start)
                ),
                "maxent_tau_target_metric_peak": (
                    None
                    if args.maxent_tau_target_metric_peak is None
                    else float(args.maxent_tau_target_metric_peak)
                ),
                "maxent_tau_target_metric_peak_step": int(
                    args.maxent_tau_target_metric_peak_step
                ),
                "maxent_tau_target_metric_final": (
                    None
                    if args.maxent_tau_target_metric_final is None
                    else float(args.maxent_tau_target_metric_final)
                ),
                "maxent_tau_target_metric_horizon": int(
                    args.maxent_tau_target_metric_horizon
                ),
            }
            if not bool(args.maxent_tau_learnable) and not bool(
                args.maxent_tau_controller_enabled
            ):
                self._fixed_listwise_tau = float(args.maxent_tau)
            if not bool(args.maxent_beta_controller_enabled):
                self._fixed_listwise_beta = float(args.beta)
        self._maxent_controller_state = ListwiseControllerState(
            tau_log=math.log(max(float(args.maxent_tau), 1e-8))
        )
        self._maxent_tau_log: torch.nn.Parameter | None = None
        self._maxent_tau_optimizer: torch.optim.Optimizer | None = None
        if bool(args.maxent_tau_learnable):
            self._maxent_tau_log = torch.nn.Parameter(
                torch.tensor(
                    float(self._maxent_controller_state.tau_log),
                    dtype=torch.float32,
                )
            )
            self._maxent_tau_optimizer = torch.optim.Adam(
                [self._maxent_tau_log],
                lr=float(args.maxent_tau_lr),
            )
            self._sync_maxent_tau_from_state()
        if bool(args.maxent_beta_controller_enabled) and float(args.beta) <= 0.0:
            logging.warning(
                "Listwise beta controller is enabled with initial beta=%s; the "
                "multiplicative KL controller will remain at zero until beta is "
                "initialized above zero.",
                args.beta,
            )
        if self._fixed_listwise_tau is not None or self._fixed_listwise_beta is not None:
            logging.info(
                "Locking listwise hyperparameters: tau=%s beta=%s "
                "(tau_learnable=%s tau_controller=%s beta_controller=%s)",
                (
                    self._fixed_listwise_tau
                    if self._fixed_listwise_tau is not None
                    else float(args.maxent_tau)
                ),
                (
                    self._fixed_listwise_beta
                    if self._fixed_listwise_beta is not None
                    else float(args.beta)
                ),
                bool(args.maxent_tau_learnable),
                bool(args.maxent_tau_controller_enabled),
                bool(args.maxent_beta_controller_enabled),
            )
            self._enforce_fixed_listwise_hparams()

    def _enforce_fixed_listwise_hparams(self) -> None:
        if self.objective != "maxent_listwise":
            return
        for name, value in self._fixed_listwise_config.items():
            setattr(self.args, name, value)
        if self._fixed_listwise_tau is not None:
            fixed_tau = clamp_listwise_tau(
                float(self._fixed_listwise_tau),
                tau_min=self.args.maxent_tau_min,
                tau_max=self.args.maxent_tau_max,
            )
            self.args.maxent_tau = float(fixed_tau)
            self._maxent_controller_state.tau_log = math.log(max(fixed_tau, 1e-8))
            if self._maxent_tau_log is not None:
                with torch.no_grad():
                    self._maxent_tau_log.fill_(math.log(max(fixed_tau, 1e-8)))
        if self._fixed_listwise_beta is not None:
            self.args.beta = float(self._fixed_listwise_beta)

    def _sync_maxent_tau_from_state(self) -> float:
        self._enforce_fixed_listwise_hparams()
        if self._fixed_listwise_tau is not None:
            current_tau = clamp_listwise_tau(
                float(self._fixed_listwise_tau),
                tau_min=self.args.maxent_tau_min,
                tau_max=self.args.maxent_tau_max,
            )
            self.args.maxent_tau = float(current_tau)
            self._maxent_controller_state.tau_log = math.log(max(current_tau, 1e-8))
            if self._maxent_tau_log is not None:
                with torch.no_grad():
                    self._maxent_tau_log.fill_(math.log(max(current_tau, 1e-8)))
            return float(current_tau)
        current_tau = float(self.args.maxent_tau)
        if self._maxent_tau_log is not None:
            current_tau = math.exp(float(self._maxent_tau_log.detach().item()))
        current_tau = clamp_listwise_tau(
            current_tau,
            tau_min=self.args.maxent_tau_min,
            tau_max=self.args.maxent_tau_max,
        )
        if self._maxent_tau_log is not None:
            with torch.no_grad():
                self._maxent_tau_log.fill_(math.log(max(current_tau, 1e-8)))
            self._maxent_controller_state.tau_log = float(
                self._maxent_tau_log.detach().item()
            )
        else:
            self._maxent_controller_state.tau_log = math.log(max(current_tau, 1e-8))
        self.args.maxent_tau = float(current_tau)
        return float(current_tau)

    def _maybe_update_learnable_tau(
        self,
        *,
        measured_metric: float | None,
        target_metric: float | None,
        global_step: int,
    ) -> tuple[float, float | None]:
        if self._fixed_listwise_tau is not None:
            return self._sync_maxent_tau_from_state(), None
        current_tau = self._sync_maxent_tau_from_state()
        if self._maxent_tau_log is None or self._maxent_tau_optimizer is None:
            return current_tau, None
        if target_metric is None:
            return current_tau, None
        if global_step <= max(0, int(self.args.maxent_tau_warmup_steps)):
            return current_tau, None
        if not isinstance(measured_metric, (int, float)) or not math.isfinite(
            float(measured_metric)
        ):
            return current_tau, None

        tau_loss = compute_learnable_tau_loss(
            self._maxent_tau_log,
            measured_metric=float(measured_metric),
            target_metric=float(target_metric),
        )
        if tau_loss is None:
            return current_tau, None

        tau_loss_value = float(tau_loss.detach().cpu().item())
        self._maxent_tau_optimizer.zero_grad(set_to_none=True)
        tau_loss.backward()
        self._maxent_tau_optimizer.step()
        return self._sync_maxent_tau_from_state(), tau_loss_value

    def _resolve_tau_target_metric(self, *, global_step: int) -> float | None:
        return resolve_listwise_target_entropy(
            target_entropy=self.args.maxent_tau_target_metric,
            target_entropy_start=self.args.maxent_tau_target_metric_start,
            target_entropy_peak=self.args.maxent_tau_target_metric_peak,
            target_entropy_peak_step=self.args.maxent_tau_target_metric_peak_step,
            target_entropy_final=self.args.maxent_tau_target_metric_final,
            target_entropy_horizon=self.args.maxent_tau_target_metric_horizon,
            global_step=int(global_step),
        )

    def _select_tau_adaptation_metric(
        self,
        *,
        semantic_entropy_mu: float | None,
        exploration_gain_any_correct: float | None,
        exploration_gain_drgrpo: float | None,
    ) -> tuple[str, float | None]:
        metric_name = str(self.args.maxent_tau_adaptation_metric)
        metric_map = {
            "semantic_entropy_mu": semantic_entropy_mu,
            "exploration_gain_any_correct": exploration_gain_any_correct,
            "exploration_gain_drgrpo": exploration_gain_drgrpo,
        }
        return metric_name, metric_map.get(metric_name)

    def _use_instrumented_grpo_learning_step(self) -> bool:
        return self.objective == "grpo" and (
            int(getattr(self.args, "zero_stage", 0) or 0) >= 3
            or bool(getattr(self.args, "adam_offload", False))
        )

    def _should_skip_baseline_grad_norm_logging(self) -> bool:
        return self._use_instrumented_grpo_learning_step()

    def _baseline_progress_log_interval(self, total_micro_batches: int) -> int:
        if total_micro_batches <= 0:
            return 1
        return max(1, total_micro_batches // 8)

    def _baseline_should_log_progress(
        self,
        local_grad_step: int,
        total_micro_batches: int,
    ) -> bool:
        if not self.strategy.is_rank_0():
            return False
        if local_grad_step <= 1 or local_grad_step >= total_micro_batches:
            return True
        interval = self._baseline_progress_log_interval(total_micro_batches)
        return (local_grad_step % interval) == 0

    def _grpo_learning_step_with_progress(self, trajectory):
        args: ZeroMathArgs = self.args
        infos = {}
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        prompt_id_lens = trajectory["prompt_ids_lens"]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]

        logging.info(f"learn data size {input_ids.shape}")

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        logps = torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device
        )
        policy_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.model)
        with torch.no_grad():
            for i in range(0, len(input_ids), args.train_batch_size_per_device):
                batch_end = min(i + args.train_batch_size_per_device, len(input_ids))
                mini_batch_inds = torch.arange(i, batch_end, device=input_ids.device)
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]

                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_input_ids = self._sanitize_scoring_token_ids(
                    mb_input_ids,
                    upper_bound=policy_vocab_upper_bound,
                    context="baseline_policy_input",
                )

                batch_logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ]
                if args.temperature != 1:
                    batch_logits = batch_logits / args.temperature
                batch_logits = self._mask_invalid_scoring_logit_columns(
                    batch_logits,
                    valid_vocab_size=policy_vocab_upper_bound,
                    context="baseline_policy_logits",
                )
                batch_logps = self._gather_selected_logps(
                    batch_logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps

        if self.ref_model is not None:
            all_ref_logps = []
            ref_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.ref_model)
            with torch.no_grad():
                for i in range(0, len(input_ids), args.train_batch_size_per_device):
                    batch_end = min(i + args.train_batch_size_per_device, len(input_ids))
                    batch_inds = torch.arange(i, batch_end, device=input_ids.device)
                    batch_input_ids = self._sanitize_scoring_token_ids(
                        input_ids[batch_inds],
                        upper_bound=ref_vocab_upper_bound,
                        context="baseline_reference_input",
                    )

                    batch_ref_logits = self.ref_model(
                        batch_input_ids, attention_mask=att_mask[batch_inds]
                    )["logits"]
                    if args.temperature != 1:
                        batch_ref_logits = batch_ref_logits / args.temperature
                    batch_ref_logits = self._mask_invalid_scoring_logit_columns(
                        batch_ref_logits,
                        valid_vocab_size=ref_vocab_upper_bound,
                        context="baseline_reference_logits",
                    )
                    batch_ref_logps = self._gather_selected_logps(
                        batch_ref_logits,
                        batch_input_ids,
                        response_masks[batch_inds],
                    )
                    all_ref_logps.append(batch_ref_logps)
            ref_logps = torch.cat(all_ref_logps)

            kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
            rewards = kl_rewards.clone()
            del all_ref_logps
            torch.cuda.empty_cache()
            gc.collect()
        else:
            ref_logps = None
            rewards = torch.zeros_like(response_masks).float()

        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        if self.args.critic_type == "ppo":
            advantages, returns, values = self.compute_ppo_advantages(
                rewards, input_ids, att_mask, response_masks
            )
        elif self.args.critic_type in ["grpo", "drgrpo"]:
            advantages = self.compute_monte_carlo_advantages(rewards, response_masks)[
                :, None
            ]

        total_micro_batches = (
            args.num_ppo_epochs
            * math.ceil(len(input_ids) / max(args.train_batch_size_per_device, 1))
        )
        logging.info(
            "grpo prep done: logps=%s ref_logps=%s advantages=%s total_micro_batches=%s",
            tuple(logps.shape),
            None if ref_logps is None else tuple(ref_logps.shape),
            tuple(advantages.shape),
            total_micro_batches,
        )

        stats = defaultdict(list)
        local_grad_step = 0
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.train_batch_size_per_device):
                local_grad_step += 1
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                if (
                    args.beta <= 0
                    and self.args.critic_type in ["grpo", "drgrpo"]
                    and len(mb_advantage) == 1
                    and bool(torch.count_nonzero(mb_advantage).item() == 0)
                ):
                    prompt_len = int(prompt_id_lens[int(mini_batch_inds[0])])
                    mb_last_valid_token_pos = cap_last_valid_token_pos_for_zero_advantage(
                        prompt_len=prompt_len,
                        last_valid_token_pos=int(mb_last_valid_token_pos),
                        response_token_budget=int(
                            getattr(self.args, "baseline_zero_adv_response_tokens", 8)
                        ),
                    )
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_logps = mb_logps[:, : mb_last_valid_token_pos - 1]

                if self.args.critic_type == "ppo":
                    mb_return = returns[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_values = values[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_advantage = mb_advantage[:, : mb_last_valid_token_pos - 1]

                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)["logits"]
                if args.temperature != 1:
                    logits = logits / args.temperature
                logits = self._mask_invalid_scoring_logit_columns(
                    logits,
                    valid_vocab_size=policy_vocab_upper_bound,
                    context="baseline_policy_update_logits",
                )
                new_logps = self._gather_selected_logps(
                    logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                if args.reinforce_update:
                    pg_loss_max = -mb_advantage * new_logps
                else:
                    logprobs_diff = new_logps - mb_logps
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantage * ratio
                    pg_losses2 = -mb_advantage * torch.clamp(
                        ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    stats["logprobs_diff_max"].append(
                        torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["zero_pg_loss_count"].append(
                        (pg_loss_max == 0).detach().sum().item()
                    )

                pg_loss = self.masked_aggregator(pg_loss_max, mb_response_masks, axis=1)
                pg_loss = (pg_loss * mb_loss_masks).mean()
                infos["pg_loss"] = pg_loss.detach()
                loss = pg_loss
                if args.beta > 0:
                    mb_ref_logps = ref_logps[mini_batch_inds]
                    mb_ref_logps = mb_ref_logps[:, : mb_last_valid_token_pos - 1]
                    log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                    kl3 = torch.expm1(log_ratio) - log_ratio
                    infos["kl3"] = (kl3 * mb_response_masks).detach().sum(1).mean()

                    reg_loss = self.masked_aggregator(kl3, mb_response_masks, axis=1)
                    reg_loss = args.beta * (reg_loss * mb_loss_masks).mean()
                    infos["reg_loss"] = reg_loss.detach()
                    loss += reg_loss

                with torch.no_grad():
                    entropy = self._chunked_entropy_from_logits(logits)
                    entropy = masked_mean(entropy, mb_response_masks)
                    infos["entropy"] = entropy

                self.strategy.backward(loss, self.model, self.optimizer)

                if local_grad_step % self.strategy.grad_acc_step == 0:
                    if self._should_skip_baseline_grad_norm_logging():
                        if not self._baseline_grad_norm_logging_disabled_warned:
                            logging.warning(
                                "Skipping baseline policy_grad_norm logging for the "
                                "ZeRO-3/offload slow path on node-local 7B runs."
                            )
                            self._baseline_grad_norm_logging_disabled_warned = True
                        stats["policy_grad_norm"].append(0.0)
                        stats["get_grad_norm_time"].append(0.0)
                    else:
                        _st = time.time()
                        stats["policy_grad_norm"].append(
                            self.strategy.get_gradient_norm(self.model)
                        )
                        stats["get_grad_norm_time"].append(time.time() - _st)

                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                if self.args.critic_type == "ppo":
                    value_pred = self.critic(
                        input_ids=mb_input_ids, attention_mask=mb_att_mask
                    )[:, :-1]

                    value_pred_clipped = torch.clamp(
                        value_pred,
                        mb_values - args.cliprange_value,
                        mb_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(value_pred - mb_return)
                    vf_losses2 = torch.square(value_pred_clipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)

                    vf_loss = 0.5 * self.masked_aggregator(
                        vf_loss_max, mb_response_masks, axis=1
                    )
                    critic_loss = args.vf_coef * (vf_loss * mb_loss_masks).mean()

                    self.strategy.backward(
                        critic_loss, self.critic, self.critic_optimizer
                    )
                    self.strategy.optimizer_step(
                        self.critic_optimizer, self.critic, self.critic_scheduler
                    )
                    infos["critic_loss"] = critic_loss.detach()
                    infos["vf_clipfrac"] = masked_mean(
                        (vf_losses2 > vf_losses1).float(), mb_response_masks
                    ).detach()

                if self._baseline_should_log_progress(
                    local_grad_step, total_micro_batches
                ):
                    logging.info(
                        "grpo progress: microbatch=%s/%s seq_len=%s pg_loss=%.6f loss_mask_mean=%.3f",
                        local_grad_step,
                        total_micro_batches,
                        int(mb_input_ids.shape[1]),
                        float(pg_loss.detach().cpu().item()),
                        float(mb_loss_masks.float().mean().item()),
                    )

                with torch.no_grad():
                    if not args.reinforce_update:
                        pg_clipfrac = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks, axis=1
                        )
                        stats["pg_clipfrac"].append(pg_clipfrac.mean().min().item())

        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        infos["policy_grad_norm"] = torch.tensor(
            stats["policy_grad_norm"] or [0.0]
        ).max()
        infos["get_grad_norm_time"] = torch.tensor(
            sum(stats["get_grad_norm_time"] or [0.0])
        )
        if not args.reinforce_update:
            infos["logprobs_diff_max"] = torch.tensor(stats["logprobs_diff_max"]).max()
            infos["logprobs_diff_min"] = torch.tensor(stats["logprobs_diff_min"]).min()
            infos["zero_pg_loss_count"] = (
                torch.tensor(stats["zero_pg_loss_count"]).float().mean()
            )
            infos["pg_clipfrac"] = torch.tensor(stats["pg_clipfrac"]).mean()
        infos["adv_mean"] = advantages.mean().cpu()
        infos["adv_min"] = advantages.min().cpu()
        infos["adv_max"] = advantages.max().cpu()
        infos["all_zero_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 0).sum().cpu()
        )
        infos["all_one_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 1).sum().cpu()
        )
        return infos

    def learn(self, learning_round: int):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        if learning_round == 1:
            # The upstream PPO learner prints the full first training example.
            # For long math trajectories this can dominate startup time and bury
            # the actual learner-step logs without changing the optimization.
            self.strategy.print("Training example omitted for compact startup logs")

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=(True if self.args.critic_type == "ppo" else False),
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        step_bar = tqdm(
            range(len(dataloader)),
            desc="Train steps",
            disable=not self.strategy.is_rank_0(),
        )
        learn_batch_time = []

        self.model.train()
        if self.critic is not None:
            self.critic.train()
        st = time.time()

        logging.info(
            f"start learn() buffer_len={len(self.pi_buffer)} dl_len={len(dataloader)}"
        )
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.grad_acc_step
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.grad_acc_step == 0:
                self.gradient_update_elapse = time.time() - self.gradient_update_st
                st = time.time()
                self.gradient_update_st = time.time()

                local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            "total_time": time.time() - st,
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        # Keep distributed logging reductions aligned even when optional metrics
        # are populated by different local minibatch conditions.
        train_info = {key: train_info[key] for key in sorted(train_info)}
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        logging.info("finish learn()")
        return train_info

    # Dr. GRPO Modification 2: Remove difficulty bias by just computing the MC advantage without dividing by std:
    def compute_monte_carlo_advantages(self, rewards, response_masks=None):
        del response_masks
        rewards = rewards.sum(-1)
        # Compute monte carlo trajectory-level advantage
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if self.args.critic_type == "grpo":
            # Additionally normalize by std.
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages

    def _logprob_batch_size(self) -> int:
        configured = int(
            getattr(self.args, "maxent_logprob_chunk_size", 0)
            or self.args.train_batch_size_per_device
            or 1
        )
        return max(configured, 1)

    def _backward_batch_size(self) -> int:
        configured = int(
            getattr(self.args, "maxent_backward_chunk_size", 0)
            or self.args.num_samples
            or 1
        )
        return max(configured, 1)

    def _backward_token_budget(self) -> int:
        configured = int(getattr(self.args, "maxent_backward_token_budget", 0) or 0)
        return max(configured, 0)

    def _effective_backward_token_budget(
        self,
        att_mask: torch.Tensor,
        *,
        configured_chunk_size: int,
    ) -> tuple[int, list[int] | None]:
        configured_budget = self._backward_token_budget()
        if configured_budget > 0:
            synchronized_token_counts = self._synchronized_backward_token_counts(att_mask)
            return configured_budget, synchronized_token_counts

        safe_chunk_size = max(int(configured_chunk_size), 1)
        synchronized_token_counts = self._synchronized_backward_token_counts(att_mask)
        max_synchronized_tokens = max(synchronized_token_counts, default=1)
        safety_budget = 4096
        if safe_chunk_size > 1 and max_synchronized_tokens * safe_chunk_size > safety_budget:
            if not self._listwise_backward_token_budget_safety_warned:
                logging.warning(
                    "Listwise backward auto-enabled a synchronized token budget of %s "
                    "because fixed %s-row chunks would pad to %s tokens per rank "
                    "(max synchronized row length=%s). Override "
                    "maxent_backward_token_budget explicitly to change this cap.",
                    safety_budget,
                    safe_chunk_size,
                    max_synchronized_tokens * safe_chunk_size,
                    max_synchronized_tokens,
                )
                self._listwise_backward_token_budget_safety_warned = True
            return safety_budget, synchronized_token_counts

        return 0, synchronized_token_counts

    def _unwrap_scoring_model(self, model: torch.nn.Module) -> torch.nn.Module:
        base_model = model
        visited = set()
        while hasattr(base_model, "module"):
            next_model = getattr(base_model, "module")
            if not isinstance(next_model, torch.nn.Module):
                break
            next_id = id(next_model)
            if next_id in visited:
                break
            visited.add(next_id)
            base_model = next_model
        return base_model

    def _unwrap_hidden_state_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the deepest safe module for hidden-state extraction.

        OAT's ``LLM`` wrapper accepts only a logits-oriented forward signature,
        while the underlying HuggingFace model exposes ``output_hidden_states``.
        For semantic trace embeddings we want that deeper model, not the wrapper.
        """

        base_model = self._unwrap_scoring_model(model)
        visited = {id(base_model)}
        while hasattr(base_model, "model"):
            next_model = getattr(base_model, "model")
            if not isinstance(next_model, torch.nn.Module):
                break
            next_id = id(next_model)
            if next_id in visited:
                break
            visited.add(next_id)
            base_model = next_model
        return base_model

    def _resolve_scoring_vocab_upper_bound(
        self, model: torch.nn.Module
    ) -> int | None:
        tokenizer = getattr(self, "tokenizer", None)
        base_model = self._unwrap_scoring_model(model)
        return resolve_token_id_upper_bound(base_model, tokenizer)

    def _sanitize_scoring_token_ids(
        self,
        token_ids: torch.Tensor,
        *,
        upper_bound: int | None,
        context: str,
    ) -> torch.Tensor:
        sanitized = sanitize_scoring_token_ids(
            token_ids,
            upper_bound=upper_bound,
            tokenizer=getattr(self, "tokenizer", None),
        )
        if sanitized.invalid_count <= 0:
            return sanitized.token_ids
        warned_contexts = self._invalid_scoring_token_ids_warned_contexts
        if context not in warned_contexts:
            logging.warning(
                "Sanitized %d scoring token ids for %s outside upper_bound=%d using replacement_id=%s (min=%s max=%s)",
                sanitized.invalid_count,
                context,
                upper_bound,
                sanitized.replacement_id,
                sanitized.min_invalid,
                sanitized.max_invalid,
            )
            warned_contexts.add(context)
        return sanitized.token_ids

    def _mask_invalid_scoring_logit_columns(
        self,
        logits: torch.Tensor,
        *,
        valid_vocab_size: int | None,
        context: str,
    ) -> torch.Tensor:
        if not isinstance(valid_vocab_size, int) or valid_vocab_size <= 0:
            return logits
        if int(logits.size(-1)) <= valid_vocab_size:
            return logits
        warned_contexts = self._invalid_logit_columns_warned_contexts
        if context not in warned_contexts:
            logging.warning(
                "Masking %d tokenizer-inaccessible logit columns for %s (valid_vocab_size=%d, logits_width=%d).",
                int(logits.size(-1)) - valid_vocab_size,
                context,
                valid_vocab_size,
                int(logits.size(-1)),
            )
            warned_contexts.add(context)
        return mask_invalid_logit_columns(logits, valid_vocab_size=valid_vocab_size)

    def _distributed_mean_scalar(self, value: float | int | None) -> float | None:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        scalar = torch.tensor(float(value), device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
            scalar /= float(dist.get_world_size())
        return float(scalar.item())

    def _distributed_weighted_mean_scalar(
        self,
        value: float | int | torch.Tensor | None,
        *,
        weight: float | int | None,
    ) -> float | None:
        safe_weight = 0.0
        if isinstance(weight, (int, float)) and math.isfinite(float(weight)):
            safe_weight = max(float(weight), 0.0)
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        numerator = torch.zeros((), device=device, dtype=torch.float32)
        if safe_weight > 0.0:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1 and torch.isfinite(value).all():
                    numerator = value.detach().to(
                        device=device,
                        dtype=torch.float32,
                    ) * safe_weight
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                numerator = torch.tensor(
                    float(value) * safe_weight,
                    device=device,
                    dtype=torch.float32,
                )
        denominator = torch.tensor(
            safe_weight,
            device=device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        if denominator.item() <= 0.0:
            return None
        return float((numerator / denominator).item())

    def _policy_grad_probe_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        cached_params = self._policy_grad_probe_params
        if cached_params is not None:
            return cached_params
        model_module = getattr(self.model, "module", self.model)
        cached_params = tuple(
            param for param in model_module.parameters() if param.requires_grad
        )
        self._policy_grad_probe_params = cached_params
        return cached_params

    def _should_run_listwise_branch_grad_diagnostics(
        self,
        *,
        local_grad_step: int,
        grad_acc_step: int,
    ) -> tuple[bool, int]:
        if self._listwise_branch_grad_probe_runtime_disabled:
            return False, 0
        if not bool(getattr(self.args, "maxent_branch_grad_diagnostics", False)):
            return False, 0
        safe_grad_acc_step = max(int(grad_acc_step), 1)
        if local_grad_step % safe_grad_acc_step != 0:
            return False, 0
        update_index = max(local_grad_step // safe_grad_acc_step, 1)
        interval = max(
            int(getattr(self.args, "maxent_branch_grad_diagnostics_interval", 1)),
            1,
        )
        if (update_index - 1) % interval != 0:
            return False, update_index
        max_steps = max(
            int(getattr(self.args, "maxent_branch_grad_diagnostics_max_steps", 0)),
            0,
        )
        if max_steps > 0 and int(self.global_step) >= max_steps:
            return False, update_index
        return True, update_index

    def _synchronized_backward_token_counts(
        self,
        att_mask: torch.Tensor,
    ) -> list[int]:
        token_counts = att_mask.sum(dim=1).to(dtype=torch.int64)
        if (
            token_counts.numel() > 0
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            token_counts = token_counts.clone()
            dist.all_reduce(token_counts, op=dist.ReduceOp.MAX)
        return [max(int(count), 1) for count in token_counts.tolist()]

    def _synchronized_last_valid_token_pos(
        self,
        last_valid_token_pos: int,
        *,
        device: torch.device,
    ) -> int:
        safe_last_valid_token_pos = max(int(last_valid_token_pos), 1)
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            synced = torch.tensor(
                [safe_last_valid_token_pos],
                device=device,
                dtype=torch.int64,
            )
            dist.all_reduce(synced, op=dist.ReduceOp.MAX)
            safe_last_valid_token_pos = max(int(synced.item()), 1)
        return safe_last_valid_token_pos

    def _logprob_token_chunk_size(self) -> int:
        # Keep vocab-softmax work on short sequence slices so listwise runs fit on
        # smaller GPUs without changing the baseline DR.GRPO path.
        return 64

    def _trim_policy_batch(
        self,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        *extra_tensors: torch.Tensor | None,
        synchronize_last_valid_token_pos: bool = False,
    ) -> tuple[int, list[torch.Tensor | None]]:
        valid_token_count_per_pos = att_mask.sum(0)
        last_valid_token_pos = torch.where(valid_token_count_per_pos == 0)[0]
        if len(last_valid_token_pos) >= 1:
            last_valid_token_pos = int(last_valid_token_pos[0].item())
        else:
            last_valid_token_pos = int(att_mask.shape[1])
        if synchronize_last_valid_token_pos:
            # Keep listwise backward/autograd chunks aligned across ranks so one
            # straggler does not spend minutes in a much longer FlashAttention
            # recompute while the other ranks have already entered NCCL collectives.
            last_valid_token_pos = self._synchronized_last_valid_token_pos(
                last_valid_token_pos,
                device=att_mask.device,
            )
        trimmed: list[torch.Tensor | None] = [
            input_ids[:, :last_valid_token_pos],
            att_mask[:, :last_valid_token_pos],
            response_masks[:, : last_valid_token_pos - 1],
        ]
        for tensor in extra_tensors:
            if tensor is None:
                trimmed.append(None)
            else:
                trimmed.append(tensor[:, : last_valid_token_pos - 1])
        return last_valid_token_pos, trimmed

    def _compute_batched_logps(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = self._logprob_batch_size()
        vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(model)
        all_logps = torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device
        )
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_end = min(i + batch_size, len(input_ids))
                batch_inds = torch.arange(i, batch_end, device=input_ids.device)
                (
                    last_valid_token_pos,
                    [
                        mb_input_ids,
                        mb_att_mask,
                        mb_response_masks,
                    ],
                ) = self._trim_policy_batch(
                    input_ids[batch_inds],
                    att_mask[batch_inds],
                    response_masks[batch_inds],
                )
                mb_input_ids = self._sanitize_scoring_token_ids(
                    mb_input_ids,
                    upper_bound=vocab_upper_bound,
                    context="listwise_batch_input",
                )
                batch_logits = model(
                    mb_input_ids,
                    attention_mask=mb_att_mask,
                )["logits"]
                if self.args.temperature != 1:
                    batch_logits = batch_logits / self.args.temperature
                batch_logits = self._mask_invalid_scoring_logit_columns(
                    batch_logits,
                    valid_vocab_size=vocab_upper_bound,
                    context="listwise_batch_logits",
                )
                batch_logps = self._gather_selected_logps(
                    batch_logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                all_logps[batch_inds, : last_valid_token_pos - 1] = batch_logps
        return all_logps

    def _gather_selected_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        safe_labels = self._sanitize_scoring_token_ids(
            labels,
            upper_bound=int(logits.size(-1)),
            context="token_select",
        )
        return gather_selected_logps_chunked(
            logits,
            safe_labels,
            response_masks,
            token_chunk_size=self._logprob_token_chunk_size(),
        )

    def _chunked_entropy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        shifted_logits = logits[:, :-1, :]
        token_chunk_size = min(
            self._logprob_token_chunk_size(),
            max(int(shifted_logits.size(1)), 1),
        )
        entropy_chunks = []
        for start in range(0, int(shifted_logits.size(1)), token_chunk_size):
            stop = min(start + token_chunk_size, int(shifted_logits.size(1)))
            chunk_logits = shifted_logits[:, start:stop, :]
            chunk_logits_fp32 = (
                chunk_logits
                if chunk_logits.dtype == torch.float32
                else chunk_logits.float()
            )
            entropy_chunks.append(entropy_from_logits(chunk_logits_fp32))
        return torch.cat(entropy_chunks, dim=1)

    def _compute_policy_probe(
        self,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        chunk_size = min(self._logprob_batch_size(), max(int(input_ids.size(0)), 1))
        vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.model)
        while True:
            try:
                with torch.no_grad():
                    logp_chunks = []
                    entropy_chunks = []
                    entropy_failed = False
                    for start in range(0, int(input_ids.size(0)), chunk_size):
                        stop = min(start + chunk_size, int(input_ids.size(0)))
                        chunk_input_ids = self._sanitize_scoring_token_ids(
                            input_ids[start:stop],
                            upper_bound=vocab_upper_bound,
                            context="policy_probe_input",
                        )
                        chunk_logits = self.model(
                            chunk_input_ids,
                            attention_mask=att_mask[start:stop],
                        )["logits"]
                        if self.args.temperature != 1:
                            chunk_logits = chunk_logits / self.args.temperature
                        chunk_logits = self._mask_invalid_scoring_logit_columns(
                            chunk_logits,
                            valid_vocab_size=vocab_upper_bound,
                            context="policy_probe_logits",
                        )
                        logp_chunks.append(
                            self._gather_selected_logps(
                                chunk_logits,
                                chunk_input_ids,
                                response_masks[start:stop],
                            )
                        )
                        if not entropy_failed:
                            try:
                                entropy_chunks.append(
                                    self._chunked_entropy_from_logits(chunk_logits)
                                )
                            except torch.OutOfMemoryError:
                                entropy_failed = True
                                logging.warning(
                                    "Listwise entropy logging hit CUDA OOM; "
                                    "skipping entropy metric for this minibatch."
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        del chunk_logits
                    entropy = None
                    if not entropy_failed and entropy_chunks:
                        entropy = torch.cat(entropy_chunks, dim=0)
                return (
                    torch.cat(logp_chunks, dim=0),
                    entropy,
                    chunk_size,
                )
            except torch.OutOfMemoryError:
                if chunk_size <= 1:
                    raise
                next_chunk_size = max(1, chunk_size // 2)
                logging.warning(
                    "Listwise policy probe hit CUDA OOM at chunk size %s; "
                    "retrying with chunk size %s.",
                    chunk_size,
                    next_chunk_size,
                )
                chunk_size = next_chunk_size
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _backward_listwise_sequence_coefficients(
        self,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        seq_coeffs: torch.Tensor,
        *,
        length_normalize: bool,
        behavior_logps: torch.Tensor | None = None,
        row_advantages: torch.Tensor | None = None,
        active_row_mask: torch.Tensor | None = None,
        active_row_count_normalizer: int | None = None,
        clip_low: float = 0.0,
        clip_high: float = 0.0,
        clip_coef: float = 0.0,
    ) -> tuple[int, int]:
        total_rows = int(input_ids.size(0))
        if seq_coeffs.dim() != 1 or int(seq_coeffs.numel()) != total_rows:
            raise ValueError("seq_coeffs must provide one coefficient per input row.")
        configured_chunk_size = min(self._backward_batch_size(), max(total_rows, 1))
        token_budget, synchronized_token_counts = self._effective_backward_token_budget(
            att_mask,
            configured_chunk_size=configured_chunk_size,
        )
        row_chunks: list[tuple[int, int]]
        if token_budget > 0:
            row_chunks = list(
                iter_budgeted_row_chunks(
                    synchronized_token_counts or [],
                    max_rows=configured_chunk_size,
                    token_budget=token_budget,
                )
            )
        else:
            row_chunks = list(
                iter_fixed_row_chunks(
                    total_rows,
                    chunk_size=configured_chunk_size,
                )
            )
        safe_clip_coef = coerce_non_negative_float(clip_coef, default=0.0)
        use_token_clip = (
            safe_clip_coef > 0.0
            and behavior_logps is not None
            and row_advantages is not None
            and active_row_mask is not None
        )
        if use_token_clip:
            if behavior_logps.shape != response_masks.shape:
                raise ValueError(
                    "behavior_logps must match response_masks when token clip is enabled."
                )
            if active_row_mask.dim() != 1 or int(active_row_mask.numel()) != total_rows:
                raise ValueError(
                    "active_row_mask must provide one boolean flag per input row."
                )
            local_active_row_count = int(active_row_mask.to(torch.int64).sum().item())
            active_row_count = (
                local_active_row_count
                if active_row_count_normalizer is None
                else max(int(active_row_count_normalizer), 0)
            )
            use_token_clip = active_row_count > 0
        else:
            active_row_count = 0
        constant_normalizer = (
            self._listwise_token_clip_constant_normalizer() if use_token_clip else None
        )
        max_chunk_size_used = 0
        max_clip_chunk_size_used = 0
        for start, stop in row_chunks:
            row_inds = torch.arange(start, stop, device=input_ids.device)
            max_chunk_size_used = max(max_chunk_size_used, int(row_inds.numel()))
            if use_token_clip:
                (
                    _,
                    [
                        chunk_input_ids,
                        chunk_att_mask,
                        chunk_response_masks,
                        chunk_behavior_logps,
                    ],
                ) = self._trim_policy_batch(
                    input_ids[row_inds],
                    att_mask[row_inds],
                    response_masks[row_inds],
                    behavior_logps[row_inds],
                    synchronize_last_valid_token_pos=True,
                )
                max_clip_chunk_size_used = max(max_clip_chunk_size_used, int(row_inds.numel()))
            else:
                (
                    _,
                    [
                        chunk_input_ids,
                        chunk_att_mask,
                        chunk_response_masks,
                    ],
                ) = self._trim_policy_batch(
                    input_ids[row_inds],
                    att_mask[row_inds],
                    response_masks[row_inds],
                    synchronize_last_valid_token_pos=True,
                )
            vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.model)
            chunk_input_ids = self._sanitize_scoring_token_ids(
                chunk_input_ids,
                upper_bound=vocab_upper_bound,
                context="listwise_backward_input",
            )
            chunk_logits = self.model(
                chunk_input_ids,
                attention_mask=chunk_att_mask,
            )["logits"]
            if self.args.temperature != 1:
                chunk_logits = chunk_logits / self.args.temperature
            chunk_logits = self._mask_invalid_scoring_logit_columns(
                chunk_logits,
                valid_vocab_size=vocab_upper_bound,
                context="listwise_backward_logits",
            )
            chunk_logps = self._gather_selected_logps(
                chunk_logits,
                chunk_input_ids,
                chunk_response_masks,
            )
            seq_logps = (chunk_logps * chunk_response_masks).sum(dim=1)
            if length_normalize:
                seq_logps = seq_logps / chunk_response_masks.sum(dim=1).clamp(min=1).to(
                    seq_logps.dtype
                )
            chunk_surrogate_loss = torch.sum(
                seq_logps * seq_coeffs[row_inds].to(seq_logps.dtype)
            )
            # Reuse the exact same chunked forward pass for token-clip mode so every
            # rank participates in one shared distributed backward pattern.
            if use_token_clip:
                chunk_per_row_loss, _, _, _ = compute_token_level_clip_loss(
                    new_logps=chunk_logps,
                    behavior_logps=chunk_behavior_logps.to(chunk_logps.dtype),
                    response_masks=chunk_response_masks,
                    row_advantages=row_advantages[row_inds].to(chunk_logps.dtype),
                    clip_low=clip_low,
                    clip_high=clip_high,
                    constant_normalizer=constant_normalizer,
                )
                chunk_active = active_row_mask[row_inds].to(chunk_per_row_loss.dtype)
                chunk_surrogate_loss = chunk_surrogate_loss + (
                    safe_clip_coef
                    * (chunk_per_row_loss * chunk_active).sum()
                    / float(active_row_count)
                )
            if chunk_surrogate_loss.requires_grad:
                self.strategy.backward(chunk_surrogate_loss, self.model, self.optimizer)
        return max_chunk_size_used, max_clip_chunk_size_used

    def _measure_listwise_sequence_gradient_squared_norm(
        self,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        seq_coeffs: torch.Tensor,
        *,
        length_normalize: bool,
        behavior_logps: torch.Tensor | None = None,
        row_advantages: torch.Tensor | None = None,
        active_row_mask: torch.Tensor | None = None,
        active_row_count_normalizer: int | None = None,
        clip_low: float = 0.0,
        clip_high: float = 0.0,
        clip_coef: float = 0.0,
    ) -> tuple[torch.Tensor, int, int]:
        total_rows = int(input_ids.size(0))
        if seq_coeffs.dim() != 1 or int(seq_coeffs.numel()) != total_rows:
            raise ValueError("seq_coeffs must provide one coefficient per input row.")
        grad_params = self._policy_grad_probe_parameters()
        grad_sq_norm = torch.zeros(
            (),
            device=input_ids.device,
            dtype=torch.float64,
        )
        if not grad_params:
            return grad_sq_norm, 0, 0

        configured_chunk_size = min(self._backward_batch_size(), max(total_rows, 1))
        token_budget, synchronized_token_counts = self._effective_backward_token_budget(
            att_mask,
            configured_chunk_size=configured_chunk_size,
        )
        if token_budget > 0:
            row_chunks = list(
                iter_budgeted_row_chunks(
                    synchronized_token_counts or [],
                    max_rows=configured_chunk_size,
                    token_budget=token_budget,
                )
            )
        else:
            row_chunks = list(
                iter_fixed_row_chunks(
                    total_rows,
                    chunk_size=configured_chunk_size,
                )
            )

        safe_clip_coef = coerce_non_negative_float(clip_coef, default=0.0)
        use_token_clip = (
            safe_clip_coef > 0.0
            and behavior_logps is not None
            and row_advantages is not None
            and active_row_mask is not None
        )
        if use_token_clip:
            if behavior_logps.shape != response_masks.shape:
                raise ValueError(
                    "behavior_logps must match response_masks when token clip is enabled."
                )
            if active_row_mask.dim() != 1 or int(active_row_mask.numel()) != total_rows:
                raise ValueError(
                    "active_row_mask must provide one boolean flag per input row."
                )
            local_active_row_count = int(active_row_mask.to(torch.int64).sum().item())
            active_row_count = (
                local_active_row_count
                if active_row_count_normalizer is None
                else max(int(active_row_count_normalizer), 0)
            )
            use_token_clip = active_row_count > 0
        else:
            active_row_count = 0
        constant_normalizer = (
            self._listwise_token_clip_constant_normalizer() if use_token_clip else None
        )

        max_chunk_size_used = 0
        max_clip_chunk_size_used = 0
        for start, stop in row_chunks:
            row_inds = torch.arange(start, stop, device=input_ids.device)
            max_chunk_size_used = max(max_chunk_size_used, int(row_inds.numel()))
            if use_token_clip:
                (
                    _,
                    [
                        chunk_input_ids,
                        chunk_att_mask,
                        chunk_response_masks,
                        chunk_behavior_logps,
                    ],
                ) = self._trim_policy_batch(
                    input_ids[row_inds],
                    att_mask[row_inds],
                    response_masks[row_inds],
                    behavior_logps[row_inds],
                    synchronize_last_valid_token_pos=True,
                )
                max_clip_chunk_size_used = max(
                    max_clip_chunk_size_used,
                    int(row_inds.numel()),
                )
            else:
                (
                    _,
                    [
                        chunk_input_ids,
                        chunk_att_mask,
                        chunk_response_masks,
                    ],
                ) = self._trim_policy_batch(
                    input_ids[row_inds],
                    att_mask[row_inds],
                    response_masks[row_inds],
                    synchronize_last_valid_token_pos=True,
                )
            vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.model)
            chunk_input_ids = self._sanitize_scoring_token_ids(
                chunk_input_ids,
                upper_bound=vocab_upper_bound,
                context="listwise_grad_probe_input",
            )
            chunk_logits = self.model(
                chunk_input_ids,
                attention_mask=chunk_att_mask,
            )["logits"]
            if self.args.temperature != 1:
                chunk_logits = chunk_logits / self.args.temperature
            chunk_logits = self._mask_invalid_scoring_logit_columns(
                chunk_logits,
                valid_vocab_size=vocab_upper_bound,
                context="listwise_grad_probe_logits",
            )
            chunk_logps = self._gather_selected_logps(
                chunk_logits,
                chunk_input_ids,
                chunk_response_masks,
            )
            seq_logps = (chunk_logps * chunk_response_masks).sum(dim=1)
            if length_normalize:
                seq_logps = seq_logps / chunk_response_masks.sum(dim=1).clamp(min=1).to(
                    seq_logps.dtype
                )
            chunk_surrogate_loss = torch.sum(
                seq_logps * seq_coeffs[row_inds].to(seq_logps.dtype)
            )
            if use_token_clip:
                chunk_per_row_loss, _, _, _ = compute_token_level_clip_loss(
                    new_logps=chunk_logps,
                    behavior_logps=chunk_behavior_logps.to(chunk_logps.dtype),
                    response_masks=chunk_response_masks,
                    row_advantages=row_advantages[row_inds].to(chunk_logps.dtype),
                    clip_low=clip_low,
                    clip_high=clip_high,
                    constant_normalizer=constant_normalizer,
                )
                chunk_active = active_row_mask[row_inds].to(chunk_per_row_loss.dtype)
                chunk_surrogate_loss = chunk_surrogate_loss + (
                    safe_clip_coef
                    * (chunk_per_row_loss * chunk_active).sum()
                    / float(active_row_count)
                )
            if not chunk_surrogate_loss.requires_grad:
                continue
            chunk_grads = torch.autograd.grad(
                chunk_surrogate_loss,
                grad_params,
                retain_graph=False,
                allow_unused=True,
            )
            for grad in chunk_grads:
                if grad is None:
                    continue
                grad_fp32 = grad.detach().to(dtype=torch.float32)
                grad_sq_norm = grad_sq_norm + torch.sum(
                    grad_fp32 * grad_fp32,
                    dtype=torch.float64,
                )
            del chunk_grads

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_sq_norm, op=dist.ReduceOp.SUM)
        return grad_sq_norm, max_chunk_size_used, max_clip_chunk_size_used

    def _probe_listwise_branch_gradient_metrics(
        self,
        *,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        raw_seq_coeffs_grouped: torch.Tensor,
        length_normalize: bool,
        behavior_logps: torch.Tensor,
        row_advantages: torch.Tensor,
        active_row_mask: torch.Tensor,
        active_row_count_normalizer: int,
        clip_low: float,
        clip_high: float,
        sequence_aux_coef: float,
        global_active_group_count: int,
        update_index: int,
    ) -> dict[str, torch.Tensor] | None:
        if sequence_aux_coef <= 0.0:
            return None
        try:
            raw_seq_coeffs = flatten_prompt_major_tensor(raw_seq_coeffs_grouped).to(
                device=input_ids.device,
                dtype=torch.float32,
            )
            zero_seq_coeffs = torch.zeros_like(raw_seq_coeffs)
            token_grad_sq, _, token_clip_chunk = (
                self._measure_listwise_sequence_gradient_squared_norm(
                    input_ids,
                    att_mask,
                    response_masks,
                    zero_seq_coeffs,
                    length_normalize=length_normalize,
                    behavior_logps=behavior_logps,
                    row_advantages=row_advantages,
                    active_row_mask=active_row_mask,
                    active_row_count_normalizer=active_row_count_normalizer,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    clip_coef=1.0,
                )
            )
            if global_active_group_count > 0:
                listwise_grad_sq, seq_chunk_size, _ = (
                    self._measure_listwise_sequence_gradient_squared_norm(
                        input_ids,
                        att_mask,
                        response_masks,
                        raw_seq_coeffs,
                        length_normalize=length_normalize,
                    )
                )
                combined_grad_sq, _, combined_clip_chunk = (
                    self._measure_listwise_sequence_gradient_squared_norm(
                        input_ids,
                        att_mask,
                        response_masks,
                        raw_seq_coeffs,
                        length_normalize=length_normalize,
                        behavior_logps=behavior_logps,
                        row_advantages=row_advantages,
                        active_row_mask=active_row_mask,
                        active_row_count_normalizer=active_row_count_normalizer,
                        clip_low=clip_low,
                        clip_high=clip_high,
                        clip_coef=1.0,
                    )
                )
            else:
                listwise_grad_sq = torch.zeros_like(token_grad_sq)
                combined_grad_sq = token_grad_sq.clone()
                seq_chunk_size = 0
                combined_clip_chunk = token_clip_chunk

            token_grad_norm = token_grad_sq.sqrt().to(dtype=torch.float32)
            listwise_grad_norm = listwise_grad_sq.sqrt().to(dtype=torch.float32)
            combined_grad_norm = combined_grad_sq.sqrt().to(dtype=torch.float32)
            denom = token_grad_norm.clamp(min=1e-12)
            grad_ratio_unscaled = listwise_grad_norm / denom
            grad_ratio_scaled = grad_ratio_unscaled * float(sequence_aux_coef)
            cosine_valid = (
                token_grad_norm.detach().item() > 0.0
                and listwise_grad_norm.detach().item() > 0.0
            )
            if cosine_valid:
                cosine_numerator = (
                    combined_grad_sq - token_grad_sq - listwise_grad_sq
                ).to(dtype=torch.float32)
                cosine_denominator = (
                    2.0 * token_grad_norm * listwise_grad_norm
                ).clamp(min=1e-12)
                grad_cosine = torch.clamp(
                    cosine_numerator / cosine_denominator,
                    min=-1.0,
                    max=1.0,
                )
            else:
                grad_cosine = torch.zeros(
                    (),
                    device=input_ids.device,
                    dtype=torch.float32,
                )

            if self.strategy.is_rank_0():
                logging.info(
                    "listwise grad probe update %s: token_norm=%.6f "
                    "listwise_norm=%.6f scaled_ratio=%.6f cosine=%.6f "
                    "seq_chunk=%s token_clip_chunk=%s combined_clip_chunk=%s",
                    update_index,
                    float(token_grad_norm.detach().item()),
                    float(listwise_grad_norm.detach().item()),
                    float(grad_ratio_scaled.detach().item()),
                    float(grad_cosine.detach().item()),
                    seq_chunk_size,
                    token_clip_chunk,
                    combined_clip_chunk,
                )

            return {
                "listwise_grad_probe_enabled": torch.tensor(
                    1.0,
                    device=input_ids.device,
                    dtype=torch.float32,
                ),
                "listwise_grad_probe_update_index": torch.tensor(
                    float(update_index),
                    device=input_ids.device,
                    dtype=torch.float32,
                ),
                "listwise_grad_probe_valid": torch.tensor(
                    1.0 if cosine_valid else 0.0,
                    device=input_ids.device,
                    dtype=torch.float32,
                ),
                "listwise_grad_token_norm": token_grad_norm.detach(),
                "listwise_grad_sequence_norm": listwise_grad_norm.detach(),
                "listwise_grad_combined_norm": combined_grad_norm.detach(),
                "listwise_grad_ratio_unscaled": grad_ratio_unscaled.detach(),
                "listwise_grad_ratio_scaled": grad_ratio_scaled.detach(),
                "listwise_grad_cosine": grad_cosine.detach(),
            }
        except (RuntimeError, torch.OutOfMemoryError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._listwise_branch_grad_probe_runtime_disabled = True
            if not self._listwise_branch_grad_probe_warned:
                logging.exception(
                    "Disabling listwise branch gradient diagnostics after a probe failure."
                )
                self._listwise_branch_grad_probe_warned = True
            return None

    def _listwise_token_clip_constant_normalizer(self) -> float | None:
        if self.args.critic_type == "drgrpo":
            return float(max(int(self.args.generate_max_length), 1))
        return None

    def _backward_listwise_token_clip_loss(
        self,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        behavior_logps: torch.Tensor,
        row_advantages: torch.Tensor,
        active_row_mask: torch.Tensor,
        *,
        active_row_count_normalizer: int | None = None,
        clip_low: float,
        clip_high: float,
        clip_coef: float,
    ) -> int:
        safe_clip_coef = coerce_non_negative_float(clip_coef, default=0.0)
        local_active_row_count = int(active_row_mask.to(torch.int64).sum().item())
        if safe_clip_coef <= 0.0:
            return 0
        active_row_count = (
            local_active_row_count
            if active_row_count_normalizer is None
            else max(int(active_row_count_normalizer), 0)
        )
        if active_row_count <= 0:
            return 0

        # Do not short-circuit ranks with zero local active rows when another rank
        # still has active rows. Distributed optimizers expect every rank to
        # participate in the same token-clip backward pattern.

        total_rows = int(input_ids.size(0))
        configured_chunk_size = min(self._backward_batch_size(), max(total_rows, 1))
        token_budget, synchronized_token_counts = self._effective_backward_token_budget(
            att_mask,
            configured_chunk_size=configured_chunk_size,
        )
        if token_budget > 0:
            row_chunks = list(
                iter_budgeted_row_chunks(
                    synchronized_token_counts or [],
                    max_rows=configured_chunk_size,
                    token_budget=token_budget,
                )
            )
        else:
            row_chunks = list(
                iter_fixed_row_chunks(
                    total_rows,
                    chunk_size=configured_chunk_size,
                )
            )

        constant_normalizer = self._listwise_token_clip_constant_normalizer()
        max_chunk_size_used = 0
        for start, stop in row_chunks:
            row_inds = torch.arange(start, stop, device=input_ids.device)
            max_chunk_size_used = max(max_chunk_size_used, int(row_inds.numel()))
            (
                _,
                [
                    chunk_input_ids,
                    chunk_att_mask,
                    chunk_response_masks,
                    chunk_behavior_logps,
                ],
            ) = self._trim_policy_batch(
                input_ids[row_inds],
                att_mask[row_inds],
                response_masks[row_inds],
                behavior_logps[row_inds],
                synchronize_last_valid_token_pos=True,
            )
            vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.model)
            chunk_input_ids = self._sanitize_scoring_token_ids(
                chunk_input_ids,
                upper_bound=vocab_upper_bound,
                context="listwise_clip_backward_input",
            )
            chunk_logits = self.model(
                chunk_input_ids,
                attention_mask=chunk_att_mask,
            )["logits"]
            if self.args.temperature != 1:
                chunk_logits = chunk_logits / self.args.temperature
            chunk_logits = self._mask_invalid_scoring_logit_columns(
                chunk_logits,
                valid_vocab_size=vocab_upper_bound,
                context="listwise_clip_backward_logits",
            )
            chunk_new_logps = self._gather_selected_logps(
                chunk_logits,
                chunk_input_ids,
                chunk_response_masks,
            )
            chunk_per_row_loss, _, _, _ = compute_token_level_clip_loss(
                new_logps=chunk_new_logps,
                behavior_logps=chunk_behavior_logps.to(chunk_new_logps.dtype),
                response_masks=chunk_response_masks,
                row_advantages=row_advantages[row_inds].to(chunk_new_logps.dtype),
                clip_low=clip_low,
                clip_high=clip_high,
                constant_normalizer=constant_normalizer,
            )
            chunk_active = active_row_mask[row_inds].to(chunk_per_row_loss.dtype)
            chunk_loss = (
                safe_clip_coef
                * (chunk_per_row_loss * chunk_active).sum()
                / float(active_row_count)
            )
            if chunk_loss.requires_grad:
                self.strategy.backward(chunk_loss, self.model, self.optimizer)
        return max_chunk_size_used

    def _sequence_logps_grouped(
        self,
        per_token_logps: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        *,
        length_normalize: bool,
        context: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_logps = (per_token_logps * response_masks).sum(dim=1)
        grouped_seq_logps = reshape_prompt_major_tensor(seq_logps, group_size)
        token_counts = reshape_prompt_major_tensor(
            response_masks.sum(dim=1).to(torch.float32),
            group_size,
        )
        if grouped_seq_logps is None or token_counts is None:
            raise ValueError(f"Could not reshape {context} into whole prompt groups.")
        if length_normalize:
            grouped_seq_logps = grouped_seq_logps / token_counts.clamp(min=1.0)
        return grouped_seq_logps, token_counts



    def _response_texts_grouped(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
    ) -> list[list[str]]:
        label_ids = input_ids[:, 1:]
        rows: list[str] = []
        for row_ids, row_mask in zip(label_ids, response_masks):
            token_ids = row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()
            rows.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        grouped = [rows[i:i+group_size] for i in range(0, len(rows), group_size)]
        return grouped

    def _semantic_cluster_inputs(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
    ) -> tuple[
        list[list[str | None]],
        list[list[str | None]],
        list[list[str | None]],
    ]:
        response_texts_grouped = self._response_texts_grouped(input_ids, response_masks, group_size)
        final_answer_keys_grouped: list[list[str | None]] = []
        reasoning_trace_texts_grouped: list[list[str | None]] = []
        reasoning_signature_keys_grouped: list[list[str | None]] = []
        for prompt_rows in response_texts_grouped:
            final_answer_keys_grouped.append([
                extract_normalized_final_answer_for_clustering(
                    row_text,
                    template=str(self.args.prompt_template),
                )
                for row_text in prompt_rows
            ])
            reasoning_trace_texts_grouped.append([
                extract_reasoning_trace_for_clustering(
                    row_text,
                    template=str(self.args.prompt_template),
                )
                for row_text in prompt_rows
            ])
            reasoning_signature_keys_grouped.append([
                extract_reasoning_signature_for_clustering(
                    row_text,
                    template=str(self.args.prompt_template),
                )
                for row_text in prompt_rows
            ])
        return (
            final_answer_keys_grouped,
            reasoning_trace_texts_grouped,
            reasoning_signature_keys_grouped,
        )

    def _semantic_trace_embeddings_grouped(
        self,
        *,
        reasoning_trace_texts_grouped: list[list[str | None]],
        valid_row_mask_grouped: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if valid_row_mask_grouped.dim() != 2:
            raise ValueError("valid_row_mask_grouped must have shape [prompts, group].")
        num_prompts, group_size = valid_row_mask_grouped.shape
        if len(reasoning_trace_texts_grouped) != num_prompts:
            raise ValueError(
                "reasoning_trace_texts_grouped must match the prompt-major minibatch shape."
            )
        trace_model_wrapper = self.ref_model if self.ref_model is not None else self.model
        if trace_model_wrapper is None:
            return (
                torch.zeros(
                    (num_prompts, group_size, 1),
                    device=valid_row_mask_grouped.device,
                    dtype=torch.float32,
                ),
                torch.zeros_like(valid_row_mask_grouped, dtype=torch.bool),
                torch.tensor(
                    0.0,
                    device=valid_row_mask_grouped.device,
                    dtype=torch.float32,
                ),
            )
        trace_model = self._unwrap_hidden_state_model(trace_model_wrapper)

        flat_texts: list[str] = []
        flat_positions: list[tuple[int, int]] = []
        valid_mask = valid_row_mask_grouped.to(torch.bool)
        for p in range(num_prompts):
            prompt_rows = reasoning_trace_texts_grouped[p]
            if len(prompt_rows) != group_size:
                raise ValueError(
                    "Each prompt must provide one reasoning trace per candidate."
                )
            for i in range(group_size):
                if not bool(valid_mask[p, i].item()):
                    continue
                trace_text = prompt_rows[i]
                if trace_text is None:
                    continue
                trace_text = str(trace_text).strip()
                if not trace_text:
                    continue
                flat_texts.append(trace_text)
                flat_positions.append((p, i))

        if not flat_texts:
            return (
                torch.zeros(
                    (num_prompts, group_size, 1),
                    device=valid_row_mask_grouped.device,
                    dtype=torch.float32,
                ),
                torch.zeros_like(valid_row_mask_grouped, dtype=torch.bool),
                torch.tensor(
                    0.0,
                    device=valid_row_mask_grouped.device,
                    dtype=torch.float32,
                ),
            )

        batch_size = max(int(self.args.train_batch_size_per_device), 1)
        max_tokens = max(int(self.args.maxent_semantic_embedding_max_tokens), 1)
        model_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(trace_model)
        pooled_batches: list[torch.Tensor] = []
        batch_positions_all: list[list[tuple[int, int]]] = []
        truncated_trace_count = 0
        attempted_trace_count = 0

        with torch.no_grad():
            for start in range(0, len(flat_texts), batch_size):
                end = min(start + batch_size, len(flat_texts))
                batch_texts = flat_texts[start:end]
                batch_positions = flat_positions[start:end]
                batch_tokenized_no_trunc = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    truncation=False,
                )
                truncated_trace_count += sum(
                    1
                    for token_ids in batch_tokenized_no_trunc["input_ids"]
                    if len(token_ids) > max_tokens
                )
                attempted_trace_count += len(batch_texts)
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                trace_input_ids = encoded["input_ids"].to(valid_row_mask_grouped.device)
                trace_att_mask = encoded["attention_mask"].to(
                    valid_row_mask_grouped.device
                )
                trace_input_ids = self._sanitize_scoring_token_ids(
                    trace_input_ids,
                    upper_bound=model_vocab_upper_bound,
                    context="semantic_trace_input",
                )
                position_ids = trace_att_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(trace_att_mask == 0, 1)
                model_outputs = None
                model_call_kwargs = {
                    "input_ids": trace_input_ids,
                    "attention_mask": trace_att_mask,
                    "position_ids": position_ids,
                    "return_dict": True,
                }
                last_type_error: TypeError | None = None
                for optional_kwargs in (
                    {"output_hidden_states": True, "use_cache": False},
                    {"output_hidden_states": True},
                    {},
                ):
                    try:
                        model_outputs = trace_model(
                            **model_call_kwargs,
                            **optional_kwargs,
                        )
                        break
                    except TypeError as exc:
                        last_type_error = exc
                if model_outputs is None:
                    if last_type_error is not None:
                        raise last_type_error
                    raise RuntimeError(
                        "Trace embedding model did not return outputs for semantic clustering."
                    )
                hidden_states = getattr(model_outputs, "hidden_states", None)
                last_hidden = None
                if hidden_states is not None and len(hidden_states) > 0:
                    last_hidden = hidden_states[-1]
                else:
                    last_hidden = getattr(model_outputs, "last_hidden_state", None)
                if last_hidden is None and isinstance(model_outputs, (tuple, list)):
                    if len(model_outputs) > 0 and isinstance(
                        model_outputs[0], torch.Tensor
                    ):
                        last_hidden = model_outputs[0]
                if last_hidden is None:
                    get_embeddings = getattr(trace_model, "get_input_embeddings", None)
                    if not callable(get_embeddings):
                        raise RuntimeError(
                            "Trace embedding model does not expose hidden states or input embeddings."
                        )
                    embedding_module = get_embeddings()
                    last_hidden = embedding_module(trace_input_ids)
                last_hidden = last_hidden.to(torch.float32)
                trace_att_mask_f = trace_att_mask.to(torch.float32)
                pooled = (
                    last_hidden * trace_att_mask_f.unsqueeze(-1)
                ).sum(dim=1) / trace_att_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
                pooled_batches.append(pooled)
                batch_positions_all.append(batch_positions)

        embedding_dim = int(pooled_batches[0].shape[-1])
        reasoning_trace_embeddings_grouped = torch.zeros(
            (num_prompts, group_size, embedding_dim),
            device=valid_row_mask_grouped.device,
            dtype=torch.float32,
        )
        reasoning_trace_valid_row_mask_grouped = torch.zeros_like(
            valid_row_mask_grouped,
            dtype=torch.bool,
        )
        for pooled, batch_positions in zip(pooled_batches, batch_positions_all):
            for offset, (p, i) in enumerate(batch_positions):
                reasoning_trace_embeddings_grouped[p, i] = pooled[offset]
                reasoning_trace_valid_row_mask_grouped[p, i] = True
        semantic_trace_truncated_frac = torch.tensor(
            float(truncated_trace_count) / float(max(attempted_trace_count, 1)),
            device=valid_row_mask_grouped.device,
            dtype=torch.float32,
        )
        return (
            reasoning_trace_embeddings_grouped,
            reasoning_trace_valid_row_mask_grouped,
            semantic_trace_truncated_frac,
        )

    def _reference_seq_logps_grouped(
        self,
        *,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        behavior_seq_logps_grouped: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        needs_reference_scores = (
            float(self.args.beta) > 0.0
            or coerce_non_negative_float(
                getattr(self.args, "maxent_candidate_kl_coef", 0.0),
                default=0.0,
            )
            > 0.0
        )
        if not needs_reference_scores:
            return torch.zeros_like(behavior_seq_logps_grouped), None
        if self.args.maxent_reference_logprobs_source == "behavior":
            return behavior_seq_logps_grouped.detach(), None
        if self.ref_model is None:
            logging.warning(
                "Listwise MaxEnt requested model reference log-probs but no ref_model "
                "is available; falling back to rollout behavior log-probs."
            )
            return behavior_seq_logps_grouped.detach(), None

        ref_logps = self._compute_batched_logps(
            self.ref_model,
            input_ids,
            att_mask,
            response_masks,
        )
        ref_seq_logps_grouped, _ = self._sequence_logps_grouped(
            ref_logps,
            response_masks,
            group_size,
            length_normalize=bool(self.args.maxent_length_normalize_ref),
            context="reference log-probs",
        )
        return ref_seq_logps_grouped, ref_logps

    def _listwise_learning_step(self, trajectory):
        args: ZeroMathArgs = self.args
        self._enforce_fixed_listwise_hparams()
        infos = {}
        stats = defaultdict(list)
        prompt_diag_stats = defaultdict(list)

        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        prompt_id_lens = trajectory["prompt_ids_lens"]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]

        logging.info(f"learn data size {input_ids.shape}")
        if self.strategy.is_rank_0():
            logging.info(
                "listwise runtime config: tau=%s beta=%s candidate_kl_coef=%s exact_drx_weight_source=%s q_temperature=%s "
                "q_epsilon=%s length_normalize_ref=%s "
                "length_normalize_policy=%s skip_zero_variance_groups=%s "
                "use_clip_objective=%s clip_objective_coef=%s clip_range=%s "
                "clip_adv_baseline=%s clip_preserve_reward_mass=%s "
                "clip_mode=%s token_surrogate_primary=%s "
                "drgrpo_token_primary=%s sequence_aux_coef=%s "
                "exact_drx_weighted_drgrpo=%s neutral_projection_coef=%s "
                "semantic_similarity_threshold=%s "
                "semantic_embedding_max_tokens=%s "
                "competitive_mode_tau=%s competitive_mode_gap=%s "
                "competitive_mode_top_k=%s competitive_mode_budget_max=%s "
                "competitive_mode_budget_scale=%s competitive_mode_intra_tau=%s "
                "prompt_select_min_alpha_frac=%s competitive_mode_positive_only=%s "
                "verified_distinct_bonus_coef=%s verified_distinct_min_modes=%s "
                "verified_distinct_reward_threshold=%s "
                "semantic_guard_max_expected_len_delta=%s "
                "semantic_guard_max_expected_format_drop=%s "
                "tau_adaptation_metric=%s "
                "branch_grad_diagnostics=%s branch_grad_interval=%s "
                "branch_grad_max_steps=%s "
                "logprob_chunk_size=%s backward_chunk_size=%s "
                "backward_token_budget=%s reference_logprobs_source=%s",
                float(args.maxent_tau),
                float(args.beta),
                float(args.maxent_candidate_kl_coef),
                str(args.maxent_exact_drx_weight_source),
                float(args.maxent_q_temperature),
                float(args.maxent_q_epsilon),
                bool(args.maxent_length_normalize_ref),
                bool(args.maxent_length_normalize_policy),
                bool(args.maxent_listwise_skip_zero_variance_groups),
                bool(args.maxent_use_clip_objective),
                float(args.maxent_clip_objective_coef),
                (
                    float(args.maxent_clip_range)
                    if args.maxent_clip_range is not None
                    else float(args.cliprange)
                ),
                (
                    None
                    if args.maxent_clip_adv_baseline is None
                    else float(args.maxent_clip_adv_baseline)
                ),
                bool(args.maxent_clip_preserve_reward_mass),
                str(args.maxent_clip_mode),
                bool(args.maxent_token_surrogate_primary),
                bool(args.maxent_drgrpo_token_primary),
                float(args.maxent_sequence_aux_coef),
                bool(args.maxent_drgrpo_token_primary),
                float(args.maxent_neutral_projection_coef),
                float(args.maxent_semantic_similarity_threshold),
                int(args.maxent_semantic_embedding_max_tokens),
                float(args.maxent_competitive_mode_tau),
                float(args.maxent_competitive_mode_gap),
                int(args.maxent_competitive_mode_top_k),
                float(args.maxent_competitive_mode_budget_max),
                float(args.maxent_competitive_mode_budget_scale),
                float(args.maxent_competitive_mode_intra_tau),
                float(args.maxent_prompt_select_min_alpha_frac),
                bool(args.maxent_competitive_mode_positive_only),
                float(args.maxent_verified_distinct_bonus_coef),
                int(args.maxent_verified_distinct_min_modes),
                float(args.maxent_verified_distinct_reward_threshold),
                float(args.maxent_semantic_guard_max_expected_len_delta),
                float(args.maxent_semantic_guard_max_expected_format_drop),
                str(args.maxent_tau_adaptation_metric),
                bool(args.maxent_branch_grad_diagnostics),
                int(args.maxent_branch_grad_diagnostics_interval),
                int(args.maxent_branch_grad_diagnostics_max_steps),
                int(args.maxent_logprob_chunk_size),
                int(args.maxent_backward_chunk_size),
                int(args.maxent_backward_token_budget),
                str(args.maxent_reference_logprobs_source),
            )

        group_size = max(int(args.num_samples), 1)
        token_surrogate_primary = bool(args.maxent_token_surrogate_primary)
        drgrpo_token_primary = bool(args.maxent_drgrpo_token_primary)
        sequence_aux_coef = coerce_non_negative_float(
            getattr(args, "maxent_sequence_aux_coef", 1.0),
            default=1.0,
        )
        candidate_kl_coef = coerce_non_negative_float(
            getattr(args, "maxent_candidate_kl_coef", 0.0),
            default=0.0,
        )
        exact_drx_weight_source = str(
            getattr(args, "maxent_exact_drx_weight_source", "sequence_clipped")
        )
        neutral_projection_coef = coerce_non_negative_float(
            getattr(args, "maxent_neutral_projection_coef", 0.0),
            default=0.0,
        )
        competitive_mode_tau = coerce_non_negative_float(
            getattr(args, "maxent_competitive_mode_tau", 0.05),
            default=0.05,
        )
        competitive_mode_gap = coerce_non_negative_float(
            getattr(args, "maxent_competitive_mode_gap", 0.10),
            default=0.10,
        )
        competitive_mode_top_k = max(
            int(getattr(args, "maxent_competitive_mode_top_k", 3)),
            1,
        )
        competitive_mode_budget_max = coerce_non_negative_float(
            getattr(args, "maxent_competitive_mode_budget_max", 0.10),
            default=0.10,
        )
        competitive_mode_budget_scale = max(
            coerce_non_negative_float(
                getattr(args, "maxent_competitive_mode_budget_scale", 0.05),
                default=0.05,
            ),
            1e-8,
        )
        competitive_mode_intra_tau = max(
            coerce_non_negative_float(
                getattr(args, "maxent_competitive_mode_intra_tau", 0.01),
                default=0.01,
            ),
            1e-8,
        )
        prompt_select_min_alpha_frac = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_prompt_select_min_alpha_frac", 0.5),
                    default=0.5,
                ),
                0.0,
            ),
            1.0,
        )
        competitive_mode_positive_only = bool(
            getattr(args, "maxent_competitive_mode_positive_only", True)
        )
        verified_distinct_bonus_coef = coerce_non_negative_float(
            getattr(args, "maxent_verified_distinct_bonus_coef", 0.5),
            default=0.5,
        )
        verified_distinct_min_modes = max(
            int(getattr(args, "maxent_verified_distinct_min_modes", 2)),
            2,
        )
        verified_distinct_reward_threshold = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_verified_distinct_reward_threshold", 0.999),
                    default=0.999,
                ),
                0.0,
            ),
            1.0,
        )
        semantic_guard_max_expected_len_delta = coerce_non_negative_float(
            getattr(args, "maxent_semantic_guard_max_expected_len_delta", 24.0),
            default=24.0,
        )
        semantic_guard_max_expected_format_drop = coerce_non_negative_float(
            getattr(args, "maxent_semantic_guard_max_expected_format_drop", 0.0),
            default=0.0,
        )
        reward_values = final_rewards.squeeze(1)
        grouped_reward_values = reshape_prompt_major_tensor(reward_values, group_size)
        if grouped_reward_values is None:
            raise ValueError(
                "Listwise MaxEnt requires rollout data in prompt-major order with "
                "flat batch size divisible by num_samples."
            )
        response_length_values, formatted_values = build_candidate_response_features(
            trajectory["action_ids"],
            tokenizer=self.tokenizer,
            template=args.prompt_template,
            device=device,
            dtype=reward_values.dtype,
        )
        grouped_response_length_values = reshape_prompt_major_tensor(
            response_length_values,
            group_size,
        )
        if grouped_response_length_values is None:
            raise ValueError(
                "Could not reshape per-candidate response lengths into prompt groups."
            )
        grouped_formatted_values = reshape_prompt_major_tensor(
            formatted_values,
            group_size,
        )
        if grouped_formatted_values is None:
            raise ValueError(
                "Could not reshape per-candidate formatted indicators into prompt groups."
            )
        reward_scale_denom = max(abs(float(args.reward_scale)), 1e-8)
        grouped_correctness_values = (
            grouped_reward_values / reward_scale_denom
        ).clamp(min=0.0, max=1.0)
        q_grouped = None
        if not drgrpo_token_primary:
            q_grouped = build_listwise_q_targets(
                reward_values,
                group_size=group_size,
                temperature=args.maxent_q_temperature,
                epsilon=args.maxent_q_epsilon,
            ).to(device=device, dtype=torch.float32)
        advantages = (
            grouped_reward_values - grouped_reward_values.mean(dim=1, keepdim=True)
        ).reshape(-1, 1)

        behavior_logps = build_padded_action_logprobs(
            trajectory.get("action_logprobs", []),
            response_masks,
        ).to(device=input_ids.device, dtype=torch.float32)
        if behavior_logps.abs().sum().item() == 0:
            behavior_logps = self._compute_batched_logps(
                self.model, input_ids, att_mask, response_masks
            ).detach()
        logging.info("listwise prep ready: behavior_logps=%s", tuple(behavior_logps.shape))

        behavior_seq_logps_grouped, _ = self._sequence_logps_grouped(
            behavior_logps,
            response_masks,
            group_size,
            length_normalize=bool(args.maxent_length_normalize_policy),
            context="behavior log-probs",
        )
        behavior_seq_logps_grouped_raw, _ = self._sequence_logps_grouped(
            behavior_logps,
            response_masks,
            group_size,
            length_normalize=False,
            context="behavior log-probs (raw)",
        )
        ref_seq_logps_grouped, ref_logps = self._reference_seq_logps_grouped(
            input_ids=input_ids,
            att_mask=att_mask,
            response_masks=response_masks,
            group_size=group_size,
            behavior_seq_logps_grouped=behavior_seq_logps_grouped,
        )
        if ref_logps is not None:
            ref_seq_logps_grouped_raw, _ = self._sequence_logps_grouped(
                ref_logps,
                response_masks,
                group_size,
                length_normalize=False,
                context="reference log-probs (raw)",
            )
        elif bool(ref_seq_logps_grouped.abs().sum().item()):
            ref_seq_logps_grouped_raw = behavior_seq_logps_grouped_raw.detach()
        else:
            ref_seq_logps_grouped_raw = torch.zeros_like(behavior_seq_logps_grouped_raw)
        logging.info(
            "listwise prep ready: behavior_seq=%s ref_seq=%s ref_logps=%s",
            tuple(behavior_seq_logps_grouped.shape),
            tuple(ref_seq_logps_grouped.shape),
            None if ref_logps is None else tuple(ref_logps.shape),
        )

        total_rows = int(input_ids.size(0))
        grad_acc_step = max(int(self.strategy.grad_acc_step), 1)
        local_minibatches_per_epoch = max(
            math.ceil(total_rows / max(int(args.train_batch_size_per_device), 1)),
            1,
        )
        total_optimizer_updates = max(
            math.ceil(args.num_ppo_epochs * local_minibatches_per_epoch / grad_acc_step),
            1,
        )
        local_grad_step = 0
        for _ in range(args.num_ppo_epochs):
            prompt_permutation = np.random.permutation(
                int(grouped_reward_values.size(0))
            )
            for mini_batch_inds in iter_grouped_minibatch_indices(
                total_rows=total_rows,
                group_size=group_size,
                flat_batch_size=args.train_batch_size_per_device,
                device=input_ids.device,
                prompt_permutation=prompt_permutation,
            ):
                local_grad_step += 1
                prompt_batch_inds = (
                    mini_batch_inds.reshape(-1, group_size)[:, 0] // group_size
                )

                (
                    _,
                    [
                        mb_input_ids,
                        mb_att_mask,
                        mb_response_masks,
                        mb_behavior_logps,
                    ],
                ) = self._trim_policy_batch(
                    input_ids[mini_batch_inds],
                    att_mask[mini_batch_inds],
                    response_masks[mini_batch_inds],
                    behavior_logps[mini_batch_inds],
                )
                mb_loss_masks = loss_masks[mini_batch_inds]
                if local_grad_step == 1:
                    logging.info(
                        "listwise minibatch ready: input=%s att=%s response=%s",
                        tuple(mb_input_ids.shape),
                        tuple(mb_att_mask.shape),
                        tuple(mb_response_masks.shape),
                    )

                new_logps, entropy, policy_chunk_size = self._compute_policy_probe(
                    mb_input_ids,
                    mb_att_mask,
                    mb_response_masks,
                )
                if local_grad_step == 1:
                    logging.info(
                        "listwise policy probe done: new_logps=%s entropy=%s chunk=%s",
                        tuple(new_logps.shape),
                        None if entropy is None else tuple(entropy.shape),
                        policy_chunk_size,
                    )
                infos["listwise_policy_probe_chunk_size"] = torch.tensor(
                    float(policy_chunk_size),
                    device=new_logps.device,
                )

                length_normalize_policy = bool(args.maxent_length_normalize_policy)
                policy_seq_logps_grouped, token_counts_grouped = (
                    self._sequence_logps_grouped(
                        new_logps,
                        mb_response_masks,
                        group_size,
                        length_normalize=length_normalize_policy,
                        context="policy log-probs",
                    )
                )
                behavior_seq_grouped, _ = self._sequence_logps_grouped(
                    mb_behavior_logps,
                    mb_response_masks,
                    group_size,
                    length_normalize=length_normalize_policy,
                    context="behavior log-probs",
                )
                if drgrpo_token_primary and exact_drx_weight_source == "sequence_clipped":
                    policy_seq_logps_grouped_raw, _ = self._sequence_logps_grouped(
                        new_logps,
                        mb_response_masks,
                        group_size,
                        length_normalize=False,
                        context="policy log-probs (raw)",
                    )
                    behavior_seq_grouped_raw, _ = self._sequence_logps_grouped(
                        mb_behavior_logps,
                        mb_response_masks,
                        group_size,
                        length_normalize=False,
                        context="behavior log-probs (raw)",
                    )
                else:
                    policy_seq_logps_grouped_raw = None
                    behavior_seq_grouped_raw = None
                grouped_loss_masks = reshape_prompt_major_tensor(
                    mb_loss_masks.to(torch.bool),
                    group_size,
                )
                if grouped_loss_masks is None:
                    raise ValueError("Could not reshape loss masks into prompt groups.")
                mb_q_grouped = None
                if q_grouped is not None:
                    mb_q_grouped = mask_and_normalize_listwise_q_targets(
                        q_grouped[prompt_batch_inds].to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        ),
                        row_mask_grouped=grouped_loss_masks,
                        context="Listwise MaxEnt loss",
                    )
                mb_ref_seq_grouped = ref_seq_logps_grouped[prompt_batch_inds].to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_ref_seq_grouped_raw = ref_seq_logps_grouped_raw[prompt_batch_inds].to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_reward_grouped = grouped_reward_values[prompt_batch_inds].to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_response_len_grouped = grouped_response_length_values[
                    prompt_batch_inds
                ].to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_formatted_grouped = grouped_formatted_values[prompt_batch_inds].to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_advantage = advantages[mini_batch_inds].to(
                    device=new_logps.device,
                    dtype=new_logps.dtype,
                )
                probe_row_advantages = mb_advantage.reshape(-1).to(
                    device=new_logps.device,
                    dtype=new_logps.dtype,
                )
                probe_log_ratio = (
                    new_logps - mb_behavior_logps.to(new_logps.dtype)
                ).clamp(-40.0, 40.0)
                probe_token_ratio = torch.exp(probe_log_ratio).to(new_logps.dtype)
                probe_token_advantages = probe_row_advantages.unsqueeze(1)
                probe_drgrpo_pg_row_loss, _, _, _ = compute_token_level_clip_loss(
                    new_logps=new_logps,
                    behavior_logps=mb_behavior_logps.to(new_logps.dtype),
                    response_masks=mb_response_masks,
                    row_advantages=probe_row_advantages,
                    clip_low=float(args.cliprange),
                    clip_high=float(args.cliprange),
                    constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                )
                probe_drgrpo_reg_row_loss = torch.zeros_like(probe_drgrpo_pg_row_loss)
                if args.beta > 0:
                    mb_ref_logps = ref_logps[mini_batch_inds]
                    mb_ref_logps = mb_ref_logps[:, : int(new_logps.size(1))].to(
                        device=new_logps.device,
                        dtype=new_logps.dtype,
                    )
                    log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                    kl3 = torch.expm1(log_ratio) - log_ratio
                    probe_drgrpo_reg_row_loss = aggregate_masked_row_values(
                        kl3,
                        mb_response_masks,
                        constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                    ).to(
                        device=new_logps.device,
                        dtype=new_logps.dtype,
                    )
                probe_drgrpo_row_loss = probe_drgrpo_pg_row_loss + (
                    float(args.beta) * probe_drgrpo_reg_row_loss
                )
                probe_row_advantages_grouped = reshape_prompt_major_tensor(
                    probe_row_advantages.to(
                        device=policy_seq_logps_grouped.device,
                        dtype=policy_seq_logps_grouped.dtype,
                    ),
                    group_size,
                )
                if probe_row_advantages_grouped is None:
                    raise ValueError(
                        "Could not reshape the Dr.GRPO row advantages into prompt groups."
                    )
                if exact_drx_weight_source == "sequence_clipped":
                    if (
                        policy_seq_logps_grouped_raw is None
                        or behavior_seq_grouped_raw is None
                    ):
                        raise ValueError(
                            "sequence_clipped DrX weights require raw sequence log-probs."
                        )
                    probe_log_seq_ratio = (
                        policy_seq_logps_grouped_raw.detach()
                        - behavior_seq_grouped_raw.detach()
                    ).clamp(-40.0, 40.0)
                    probe_seq_ratio = torch.exp(probe_log_seq_ratio).to(
                        policy_seq_logps_grouped_raw.dtype
                    )
                    probe_seq_ratio_clipped = torch.clamp(
                        probe_seq_ratio,
                        1.0 - float(args.cliprange),
                        1.0 + float(args.cliprange),
                    )
                    probe_drgrpo_utility_grouped = torch.min(
                        probe_seq_ratio * probe_row_advantages_grouped,
                        probe_seq_ratio_clipped * probe_row_advantages_grouped,
                    )
                    probe_drgrpo_utility_grouped = torch.where(
                        grouped_loss_masks,
                        probe_drgrpo_utility_grouped,
                        torch.zeros_like(probe_drgrpo_utility_grouped),
                    )
                else:
                    if exact_drx_weight_source == "clipped":
                        probe_drgrpo_weight_scores = -probe_drgrpo_row_loss
                    elif exact_drx_weight_source == "unclipped":
                        probe_drgrpo_weight_scores = aggregate_masked_row_values(
                            probe_token_ratio * probe_token_advantages,
                            mb_response_masks,
                            constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                        ).to(
                            device=new_logps.device,
                            dtype=new_logps.dtype,
                        )
                    elif exact_drx_weight_source == "local_linear":
                        probe_drgrpo_weight_scores = aggregate_masked_row_values(
                            torch.ones_like(new_logps) * probe_token_advantages,
                            mb_response_masks,
                            constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                        ).to(
                            device=new_logps.device,
                            dtype=new_logps.dtype,
                        )
                    else:
                        raise ValueError(
                            "Unsupported maxent_exact_drx_weight_source: "
                            f"{exact_drx_weight_source}"
                        )
                    probe_drgrpo_utility_grouped = reshape_prompt_major_tensor(
                        probe_drgrpo_weight_scores.to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        ),
                        group_size,
                    )
                    if probe_drgrpo_utility_grouped is None:
                        raise ValueError(
                            "Could not reshape the Dr.GRPO per-candidate utilities into prompt groups."
                        )
                valid_group_counts = grouped_loss_masks.to(torch.int64).sum(dim=1)
                neutral_group_mask = torch.zeros(
                    int(prompt_batch_inds.numel()),
                    device=policy_seq_logps_grouped.device,
                    dtype=torch.bool,
                )
                if bool(args.maxent_listwise_skip_zero_variance_groups):
                    if drgrpo_token_primary:
                        utility_dtype_info = torch.finfo(probe_drgrpo_utility_grouped.dtype)
                        valid_utility_max = torch.where(
                            grouped_loss_masks,
                            probe_drgrpo_utility_grouped,
                            torch.full_like(
                                probe_drgrpo_utility_grouped,
                                utility_dtype_info.min,
                            ),
                        ).amax(dim=1)
                        valid_utility_min = torch.where(
                            grouped_loss_masks,
                            probe_drgrpo_utility_grouped,
                            torch.full_like(
                                probe_drgrpo_utility_grouped,
                                utility_dtype_info.max,
                            ),
                        ).amin(dim=1)
                        neutral_group_mask = (
                            (valid_group_counts <= 1)
                            | ((valid_utility_max - valid_utility_min) <= 1e-8)
                        )
                    else:
                        q_dtype_info = torch.finfo(mb_q_grouped.dtype)
                        valid_q_max = torch.where(
                            grouped_loss_masks,
                            mb_q_grouped,
                            torch.full_like(mb_q_grouped, q_dtype_info.min),
                        ).amax(dim=1)
                        valid_q_min = torch.where(
                            grouped_loss_masks,
                            mb_q_grouped,
                            torch.full_like(mb_q_grouped, q_dtype_info.max),
                        ).amin(dim=1)
                        neutral_group_mask = (
                            (valid_group_counts <= 1)
                            | ((valid_q_max - valid_q_min) <= 1e-8)
                        )
                contributing_group_mask = grouped_loss_masks.any(dim=1)
                active_group_mask = (~neutral_group_mask) & contributing_group_mask
                active_group_count = int(active_group_mask.to(torch.int64).sum().item())
                active_row_mask = active_group_mask.repeat_interleave(group_size)
                valid_row_mask = flatten_prompt_major_tensor(grouped_loss_masks).to(
                    torch.bool
                )
                contributing_group_count = int(
                    contributing_group_mask.to(torch.int64).sum().item()
                )
                global_active_group_count = active_group_count
                if dist.is_available() and dist.is_initialized():
                    active_group_count_tensor = torch.tensor(
                        float(active_group_count),
                        device=policy_seq_logps_grouped.device,
                    )
                    dist.all_reduce(active_group_count_tensor, op=dist.ReduceOp.SUM)
                    global_active_group_count = int(active_group_count_tensor.item())
                global_contributing_group_count = contributing_group_count
                if dist.is_available() and dist.is_initialized():
                    contributing_group_count_tensor = torch.tensor(
                        float(contributing_group_count),
                        device=policy_seq_logps_grouped.device,
                    )
                    dist.all_reduce(contributing_group_count_tensor, op=dist.ReduceOp.SUM)
                    global_contributing_group_count = int(
                        contributing_group_count_tensor.item()
                    )
                active_row_count = int(active_row_mask.to(torch.int64).sum().item())
                global_active_row_count = active_row_count
                if dist.is_available() and dist.is_initialized():
                    active_row_count_tensor = torch.tensor(
                        float(active_row_count),
                        device=policy_seq_logps_grouped.device,
                    )
                    dist.all_reduce(active_row_count_tensor, op=dist.ReduceOp.SUM)
                    global_active_row_count = int(active_row_count_tensor.item())


                current_tau = self._sync_maxent_tau_from_state()
                if drgrpo_token_primary:
                    (
                        final_answer_keys_grouped,
                        reasoning_trace_texts_grouped,
                        reasoning_signature_keys_grouped,
                    ) = self._semantic_cluster_inputs(
                        mb_input_ids,
                        mb_response_masks,
                        group_size,
                    )
                    semantic_trace_truncated_frac = torch.tensor(
                        0.0,
                        device=policy_seq_logps_grouped.device,
                        dtype=torch.float32,
                    )
                    semantic_cluster_bundle = build_semantic_cluster_bundle(
                        final_answer_keys_grouped=final_answer_keys_grouped,
                        valid_row_mask_grouped=grouped_loss_masks.detach(),
                        reasoning_signature_keys_grouped=reasoning_signature_keys_grouped,
                    )
                    answer_key_extracted_mask_grouped = torch.tensor(
                        [
                            [answer_key is not None for answer_key in prompt_keys]
                            for prompt_keys in final_answer_keys_grouped
                        ],
                        device=policy_seq_logps_grouped.device,
                        dtype=torch.bool,
                    ) & grouped_loss_masks
                    trace_extracted_mask_grouped = torch.tensor(
                        [
                            [trace_text is not None for trace_text in prompt_rows]
                            for prompt_rows in reasoning_trace_texts_grouped
                        ],
                        device=policy_seq_logps_grouped.device,
                        dtype=torch.bool,
                    ) & grouped_loss_masks
                    signature_extracted_mask_grouped = torch.tensor(
                        [
                            [signature is not None for signature in prompt_rows]
                            for prompt_rows in reasoning_signature_keys_grouped
                        ],
                        device=policy_seq_logps_grouped.device,
                        dtype=torch.bool,
                    ) & grouped_loss_masks
                    semantic_valid_row_mask_grouped = (
                        semantic_cluster_bundle.semantic_valid_row_mask_grouped.to(
                            device=policy_seq_logps_grouped.device
                        )
                    )
                    semantic_answer_key_extracted_count = answer_key_extracted_mask_grouped.to(
                        torch.float32
                    ).sum()
                    semantic_trace_extracted_count = trace_extracted_mask_grouped.to(
                        torch.float32
                    ).sum()
                    semantic_signature_extracted_count = signature_extracted_mask_grouped.to(
                        torch.float32
                    ).sum()
                    semantic_valid_row_count = semantic_valid_row_mask_grouped.to(
                        torch.float32
                    ).sum()
                    valid_row_count_for_semantics = grouped_loss_masks.to(
                        torch.float32
                    ).sum().clamp(min=1.0)
                    stats["listwise_semantic_answer_key_extracted_frac"].append(
                        (
                            semantic_answer_key_extracted_count
                            / valid_row_count_for_semantics
                        ).detach()
                    )
                    stats["listwise_semantic_trace_extracted_frac"].append(
                        (
                            semantic_trace_extracted_count
                            / valid_row_count_for_semantics
                        ).detach()
                    )
                    stats["listwise_semantic_signature_extracted_frac"].append(
                        (
                            semantic_signature_extracted_count
                            / valid_row_count_for_semantics
                        ).detach()
                    )
                    stats["listwise_semantic_cluster_valid_frac"].append(
                        (semantic_valid_row_count / valid_row_count_for_semantics).detach()
                    )
                    stats["listwise_semantic_trace_truncated_frac"].append(
                        semantic_trace_truncated_frac.detach()
                    )
                    semantic_ref_seq_grouped = (
                        mb_ref_seq_grouped_raw
                        if exact_drx_weight_source == "sequence_clipped"
                        else mb_ref_seq_grouped
                    ).detach()
                    if behavior_seq_grouped_raw is None:
                        behavior_seq_grouped_raw, _ = self._sequence_logps_grouped(
                            mb_behavior_logps,
                            mb_response_masks,
                            group_size,
                            length_normalize=False,
                            context="behavior log-probs (raw semantic entropy)",
                        )
                    behavior_log_probs_grouped_raw = masked_group_log_softmax(
                        behavior_seq_grouped_raw.to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        ),
                        grouped_loss_masks,
                    )
                    behavior_probs_grouped_raw = torch.where(
                        grouped_loss_masks,
                        torch.exp(behavior_log_probs_grouped_raw),
                        torch.zeros_like(behavior_log_probs_grouped_raw),
                    )
                    utility_dtype_info = torch.finfo(probe_drgrpo_utility_grouped.dtype)
                    best_drgrpo_utility_grouped = torch.where(
                        grouped_loss_masks,
                        probe_drgrpo_utility_grouped.detach(),
                        torch.full_like(
                            probe_drgrpo_utility_grouped,
                            utility_dtype_info.min,
                        ),
                    ).amax(dim=1)
                    has_valid_group = grouped_loss_masks.any(dim=1)
                    best_drgrpo_utility_grouped = torch.where(
                        has_valid_group,
                        best_drgrpo_utility_grouped,
                        torch.zeros_like(best_drgrpo_utility_grouped),
                    )
                    behavior_expected_drgrpo_utility_grouped = (
                        behavior_probs_grouped_raw.detach()
                        * torch.where(
                            grouped_loss_masks,
                            probe_drgrpo_utility_grouped.detach(),
                            torch.zeros_like(probe_drgrpo_utility_grouped),
                        )
                    ).sum(dim=1)
                    exploration_gain_drgrpo_t = (
                        best_drgrpo_utility_grouped
                        - behavior_expected_drgrpo_utility_grouped
                    )
                    semantic_alpha_raw_grouped = torch.clamp(
                        exploration_gain_drgrpo_t.detach()
                        / float(competitive_mode_budget_scale),
                        min=0.0,
                        max=1.0,
                    )
                    semantic_explore_budget_grouped = (
                        float(competitive_mode_budget_max)
                        * semantic_alpha_raw_grouped
                        * active_group_mask.to(
                            dtype=policy_seq_logps_grouped.dtype
                        )
                    )
                    drx_bundle = build_drx_target_bundle(
                        utility_grouped=probe_drgrpo_utility_grouped.detach(),
                        ref_seq_logps_grouped=semantic_ref_seq_grouped,
                        valid_row_mask_grouped=grouped_loss_masks.detach(),
                        cluster_ids_grouped=semantic_cluster_bundle.cluster_ids_grouped.detach(),
                        candidate_correctness_grouped=grouped_correctness_values[
                            prompt_batch_inds
                        ].detach(),
                        candidate_lengths_grouped=mb_response_len_grouped.detach(),
                        candidate_formatted_grouped=mb_formatted_grouped.detach(),
                        competitive_mode_tau=competitive_mode_tau,
                        competitive_mode_gap=competitive_mode_gap,
                        competitive_mode_top_k=competitive_mode_top_k,
                        competitive_mode_budget_grouped=semantic_explore_budget_grouped.detach(),
                        competitive_mode_budget_max=competitive_mode_budget_max,
                        competitive_mode_intra_tau=competitive_mode_intra_tau,
                        prompt_select_min_alpha_frac=prompt_select_min_alpha_frac,
                        competitive_mode_positive_only=competitive_mode_positive_only,
                        verified_distinct_bonus_coef=verified_distinct_bonus_coef,
                        verified_distinct_min_modes=verified_distinct_min_modes,
                        verified_distinct_reward_threshold=verified_distinct_reward_threshold,
                        semantic_guard_max_expected_len_delta=semantic_guard_max_expected_len_delta,
                        semantic_guard_max_expected_format_drop=semantic_guard_max_expected_format_drop,
                        tau=current_tau,
                        candidate_kl_coef=candidate_kl_coef,
                        neutral_eps=1e-8,
                        neutral_projection_coef=neutral_projection_coef,
                    )
                    weights_grouped = drx_bundle.w_star_grouped.to(
                        device=policy_seq_logps_grouped.device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    informative_group_mask = drx_bundle.informative_group_mask.to(
                        device=policy_seq_logps_grouped.device
                    )
                    neutral_group_mask = drx_bundle.neutral_group_mask.to(
                        device=policy_seq_logps_grouped.device
                    )
                    contributing_group_mask = drx_bundle.contributing_group_mask.to(
                        device=policy_seq_logps_grouped.device
                    )
                    active_group_mask = informative_group_mask
                    active_group_count = int(active_group_mask.to(torch.int64).sum().item())
                    contributing_group_count = int(contributing_group_mask.to(torch.int64).sum().item())
                    global_active_group_count = active_group_count
                    if dist.is_available() and dist.is_initialized():
                        active_group_count_tensor = torch.tensor(
                            float(active_group_count),
                            device=policy_seq_logps_grouped.device,
                        )
                        dist.all_reduce(active_group_count_tensor, op=dist.ReduceOp.SUM)
                        global_active_group_count = int(active_group_count_tensor.item())
                    global_contributing_group_count = contributing_group_count
                    if dist.is_available() and dist.is_initialized():
                        contributing_group_count_tensor = torch.tensor(
                            float(contributing_group_count),
                            device=policy_seq_logps_grouped.device,
                        )
                        dist.all_reduce(contributing_group_count_tensor, op=dist.ReduceOp.SUM)
                        global_contributing_group_count = int(contributing_group_count_tensor.item())
                    active_row_mask = active_group_mask.repeat_interleave(group_size)
                    active_row_count = int(active_row_mask.to(torch.int64).sum().item())
                    global_active_row_count = active_row_count
                    if dist.is_available() and dist.is_initialized():
                        active_row_count_tensor = torch.tensor(
                            float(active_row_count),
                            device=policy_seq_logps_grouped.device,
                        )
                        dist.all_reduce(active_row_count_tensor, op=dist.ReduceOp.SUM)
                        global_active_row_count = int(active_row_count_tensor.item())

                    target_weights_grouped = torch.where(
                        grouped_loss_masks,
                        weights_grouped,
                        torch.zeros_like(weights_grouped),
                    )
                    token_target_weights_grouped = torch.where(
                        grouped_loss_masks,
                        drx_bundle.token_target_grouped.to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        ),
                        torch.zeros_like(weights_grouped),
                    )
                    projection_target_grouped = torch.where(
                        grouped_loss_masks,
                        drx_bundle.projection_target_grouped.to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        ),
                        torch.zeros_like(weights_grouped),
                    )
                    projection_group_scale = drx_bundle.projection_group_scale.to(
                        device=policy_seq_logps_grouped.device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    # Normalize projection CE by the number of projection-eligible
                    # groups, not by the sum of scale coefficients. Otherwise a
                    # neutral scale such as 0.10 cancels out of both numerator and
                    # denominator and is no longer a real strength knob.
                    local_projection_group_weight = contributing_group_mask.to(
                        dtype=projection_group_scale.dtype
                    ).sum()
                    local_projection_scale_mass = projection_group_scale.sum()
                    global_projection_group_weight = float(
                        local_projection_group_weight.item()
                    )
                    global_projection_scale_mass = float(
                        local_projection_scale_mass.item()
                    )
                    if dist.is_available() and dist.is_initialized():
                        projection_group_weight_tensor = local_projection_group_weight.detach().clone()
                        dist.all_reduce(projection_group_weight_tensor, op=dist.ReduceOp.SUM)
                        global_projection_group_weight = float(projection_group_weight_tensor.item())
                        projection_scale_mass_tensor = local_projection_scale_mass.detach().clone()
                        dist.all_reduce(projection_scale_mass_tensor, op=dist.ReduceOp.SUM)
                        global_projection_scale_mass = float(
                            projection_scale_mass_tensor.item()
                        )
                else:
                    weights_grouped = compute_listwise_weights(
                        q_grouped=mb_q_grouped,
                        ref_seq_logps_grouped=mb_ref_seq_grouped,
                        tau=current_tau,
                        beta=args.beta,
                    ).to(
                        device=policy_seq_logps_grouped.device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    if bool(neutral_group_mask.any().item()):
                        valid_group_denoms = grouped_loss_masks.sum(
                            dim=1,
                            keepdim=True,
                        ).clamp(min=1).to(weights_grouped.dtype)
                        uniform_weights = torch.where(
                            grouped_loss_masks,
                            1.0 / valid_group_denoms,
                            torch.zeros_like(weights_grouped),
                        )
                        weights_grouped = torch.where(
                            neutral_group_mask.unsqueeze(1),
                            uniform_weights,
                            weights_grouped,
                        )
                    target_weights_grouped = torch.where(
                        grouped_loss_masks,
                        weights_grouped,
                        torch.zeros_like(weights_grouped),
                    )
                    token_target_weights_grouped = target_weights_grouped
                    projection_target_grouped = target_weights_grouped
                    projection_group_scale = active_group_mask.to(
                        device=policy_seq_logps_grouped.device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    global_projection_group_weight = float(max(global_active_group_count, 0))
                    global_projection_scale_mass = global_projection_group_weight

                if drgrpo_token_primary:
                    cluster_ids_grouped = semantic_cluster_bundle.cluster_ids_grouped.to(
                        device=weights_grouped.device
                    )
                    if behavior_seq_grouped_raw is None:
                        behavior_seq_grouped_raw, _ = self._sequence_logps_grouped(
                            mb_behavior_logps,
                            mb_response_masks,
                            group_size,
                            length_normalize=False,
                            context="behavior log-probs (raw semantic entropy)",
                        )
                    behavior_log_probs_grouped_raw = masked_group_log_softmax(
                        behavior_seq_grouped_raw.to(
                            device=weights_grouped.device,
                            dtype=weights_grouped.dtype,
                        ),
                        grouped_loss_masks,
                    )
                    behavior_probs_grouped_raw = torch.where(
                        grouped_loss_masks,
                        torch.exp(behavior_log_probs_grouped_raw),
                        torch.zeros_like(behavior_log_probs_grouped_raw),
                    )
                    semantic_cluster_entropy_t, semantic_cluster_count_t = (
                        compute_normalized_semantic_cluster_entropy(
                            candidate_probs_grouped=behavior_probs_grouped_raw.detach(),
                            cluster_ids_grouped=cluster_ids_grouped,
                            valid_row_mask_grouped=grouped_loss_masks,
                            normalizer_group_size=group_size,
                        )
                    )
                    stats["listwise_semantic_cluster_entropy"].append(
                        semantic_cluster_entropy_t.mean().detach()
                    )
                    stats["listwise_semantic_cluster_entropy_norm"].append(
                        semantic_cluster_entropy_t.mean().detach()
                    )
                    stats["listwise_semantic_cluster_count"].append(
                        semantic_cluster_count_t.mean().detach()
                    )
                    semantic_diag = drx_bundle.semantic_diagnostics
                    if semantic_diag is not None:
                        stats["listwise_semantic_competitive_mode_count"].append(
                            semantic_diag.mode_count_grouped.mean().detach()
                        )
                        stats["listwise_semantic_competitive_mode_eligible_count"].append(
                            semantic_diag.eligible_mode_count_grouped.mean().detach()
                        )
                        stats["listwise_semantic_competitive_mode_eligible_frac"].append(
                            semantic_diag.eligible_mode_frac_grouped.mean().detach()
                        )
                        stats["listwise_semantic_distinct_correct_mode_count"].append(
                            semantic_diag.distinct_correct_mode_count_grouped.mean().detach()
                        )
                        stats["listwise_semantic_distinct_correct_mode_frac"].append(
                            semantic_diag.distinct_correct_mode_frac_grouped.mean().detach()
                        )
                        stats["listwise_semantic_competitive_mode_best_score"].append(
                            semantic_diag.best_score_grouped.mean().detach()
                        )
                        stats["listwise_semantic_competitive_mode_second_score"].append(
                            semantic_diag.second_score_grouped.mean().detach()
                        )
                        stats["listwise_semantic_competitive_mode_gap"].append(
                            semantic_diag.competitive_gap_grouped.mean().detach()
                        )
                        stats["listwise_semantic_explore_budget_mean"].append(
                            semantic_diag.explore_budget_grouped.mean().detach()
                        )
                        stats["listwise_semantic_explore_budget_saturated_frac"].append(
                            semantic_diag.explore_budget_saturated_grouped.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_explore_applied_group_frac"].append(
                            semantic_diag.explore_applied_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_verified_bonus_applied_group_frac"].append(
                            semantic_diag.verified_bonus_applied_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_prompt_selected_frac"].append(
                            semantic_diag.prompt_selected_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_prompt_rejected_low_opp_frac"].append(
                            semantic_diag.prompt_rejected_low_opp_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_prompt_rejected_nonpositive_frac"].append(
                            semantic_diag.prompt_rejected_nonpositive_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_prompt_rejected_len_guard_frac"].append(
                            semantic_diag.prompt_rejected_len_guard_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_prompt_rejected_format_guard_frac"].append(
                            semantic_diag.prompt_rejected_format_guard_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_prompt_rejected_verified_bonus_len_guard_frac"].append(
                            semantic_diag.prompt_rejected_verified_bonus_len_guard_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_prompt_rejected_verified_bonus_format_guard_frac"].append(
                            semantic_diag.prompt_rejected_verified_bonus_format_guard_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            ).mean().detach()
                        )
                        stats["listwise_semantic_moved_mass_l1"].append(
                            semantic_diag.moved_mass_l1_grouped.mean().detach()
                        )
                        stats["listwise_semantic_alpha_raw_mean"].append(
                            semantic_diag.alpha_raw_grouped.mean().detach()
                        )
                        stats["listwise_semantic_alpha_applied_mean"].append(
                            semantic_diag.alpha_applied_grouped.mean().detach()
                        )
                        stats["listwise_semantic_verified_bonus_mean"].append(
                            semantic_diag.verified_bonus_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_utility_q"].append(
                            semantic_diag.expected_utility_q_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_utility_explore_target"].append(
                            semantic_diag.expected_utility_explore_target_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_utility_final_w"].append(
                            semantic_diag.expected_utility_final_w_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_len_q"].append(
                            semantic_diag.expected_len_q_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_len_explore_target"].append(
                            semantic_diag.expected_len_explore_target_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_len_final_w"].append(
                            semantic_diag.expected_len_final_w_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_format_q"].append(
                            semantic_diag.expected_format_q_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_format_explore_target"].append(
                            semantic_diag.expected_format_explore_target_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_format_final_w"].append(
                            semantic_diag.expected_format_final_w_grouped.mean().detach()
                        )

                    correct_indicator_grouped = (
                        (mb_reward_grouped > 0.0) & grouped_loss_masks
                    ).to(weights_grouped.dtype)
                    any_correct_grouped = correct_indicator_grouped.any(dim=1).to(
                        weights_grouped.dtype
                    )
                    behavior_correct_mass_grouped = (
                        behavior_probs_grouped_raw.detach() * correct_indicator_grouped
                    ).sum(dim=1)
                    exploration_gain_any_correct_t = (
                        any_correct_grouped - behavior_correct_mass_grouped
                    )

                    utility_dtype_info = torch.finfo(probe_drgrpo_utility_grouped.dtype)
                    best_drgrpo_utility_grouped = torch.where(
                        grouped_loss_masks,
                        probe_drgrpo_utility_grouped.detach(),
                        torch.full_like(
                            probe_drgrpo_utility_grouped,
                            utility_dtype_info.min,
                        ),
                    ).amax(dim=1)
                    has_valid_group = grouped_loss_masks.any(dim=1)
                    best_drgrpo_utility_grouped = torch.where(
                        has_valid_group,
                        best_drgrpo_utility_grouped,
                        torch.zeros_like(best_drgrpo_utility_grouped),
                    )
                    behavior_expected_drgrpo_utility_grouped = (
                        behavior_probs_grouped_raw.detach()
                        * torch.where(
                            grouped_loss_masks,
                            probe_drgrpo_utility_grouped.detach(),
                            torch.zeros_like(probe_drgrpo_utility_grouped),
                        )
                    ).sum(dim=1)
                    exploration_gain_drgrpo_t = (
                        best_drgrpo_utility_grouped
                        - behavior_expected_drgrpo_utility_grouped
                    )

                    gain_group_mask = semantic_cluster_count_t > 0.0

                    def _masked_prompt_mean(
                        values: torch.Tensor,
                        mask: torch.Tensor,
                    ) -> torch.Tensor:
                        mask_f = mask.to(values.dtype)
                        return (values * mask_f).sum() / mask_f.sum().clamp(min=1.0)

                    def _masked_prompt_corr(
                        lhs: torch.Tensor,
                        rhs: torch.Tensor,
                        mask: torch.Tensor,
                    ) -> torch.Tensor:
                        if int(mask.to(torch.int64).sum().item()) < 2:
                            return lhs.new_zeros(())
                        lhs_sel = lhs[mask].to(torch.float32)
                        rhs_sel = rhs[mask].to(torch.float32)
                        lhs_centered = lhs_sel - lhs_sel.mean()
                        rhs_centered = rhs_sel - rhs_sel.mean()
                        denom = lhs_centered.norm() * rhs_centered.norm()
                        if float(denom.item()) <= 1e-12:
                            return lhs.new_zeros(())
                        return (
                            (lhs_centered * rhs_centered).sum() / denom
                        ).to(dtype=lhs.dtype, device=lhs.device)

                    gain_group_count = int(gain_group_mask.to(torch.int64).sum().item())
                    semantic_entropy_prompt_mean_local = _masked_prompt_mean(
                        semantic_cluster_entropy_t.detach(),
                        gain_group_mask,
                    )
                    exploration_gain_any_correct_mean_local = _masked_prompt_mean(
                        exploration_gain_any_correct_t.detach(),
                        gain_group_mask,
                    )
                    exploration_gain_drgrpo_mean_local = _masked_prompt_mean(
                        exploration_gain_drgrpo_t.detach(),
                        gain_group_mask,
                    )

                    if bool(gain_group_mask.any().item()):
                        prompt_diag_stats["listwise_semantic_entropy_prompt"].append(
                            semantic_cluster_entropy_t.detach()[gain_group_mask]
                        )
                        prompt_diag_stats[
                            "listwise_semantic_exploration_gain_any_correct_prompt"
                        ].append(
                            exploration_gain_any_correct_t.detach()[gain_group_mask]
                        )
                        prompt_diag_stats[
                            "listwise_semantic_exploration_gain_drgrpo_prompt"
                        ].append(
                            exploration_gain_drgrpo_t.detach()[gain_group_mask]
                        )

                    stats["listwise_semantic_exploration_gain_any_correct"].append(
                        exploration_gain_any_correct_mean_local
                    )
                    stats["listwise_semantic_exploration_gain_drgrpo"].append(
                        exploration_gain_drgrpo_mean_local
                    )
                    ref_available = bool(candidate_kl_coef > 0.0)
                    ref_semantic_entropy_t = torch.zeros_like(semantic_cluster_entropy_t)
                    semantic_entropy_gain_t = semantic_cluster_entropy_t
                    if ref_available:
                        ref_log_probs_grouped = masked_group_log_softmax(
                            semantic_ref_seq_grouped.to(
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            ),
                            grouped_loss_masks,
                        )
                        ref_probs_grouped = torch.where(
                            grouped_loss_masks,
                            torch.exp(ref_log_probs_grouped),
                            torch.zeros_like(ref_log_probs_grouped),
                        )
                        ref_semantic_entropy_t, _ = compute_normalized_semantic_cluster_entropy(
                            candidate_probs_grouped=ref_probs_grouped.detach(),
                            cluster_ids_grouped=cluster_ids_grouped,
                            valid_row_mask_grouped=grouped_loss_masks,
                            normalizer_group_size=group_size,
                        )
                        semantic_entropy_gain_t = (
                            semantic_cluster_entropy_t - ref_semantic_entropy_t
                        )
                    stats["listwise_semantic_cluster_entropy_ref"].append(
                        ref_semantic_entropy_t.mean().detach()
                    )
                    stats["listwise_semantic_cluster_entropy_gain"].append(
                        semantic_entropy_gain_t.mean().detach()
                    )
                    stats["listwise_semantic_cluster_ref_available"].append(
                        torch.tensor(
                            1.0 if ref_available else 0.0,
                            device=weights_grouped.device,
                            dtype=weights_grouped.dtype,
                        )
                    )

                (
                    active_weight_entropy,
                    active_weight_entropy_min,
                    active_weight_entropy_max,
                ) = collect_weight_entropy_stats(weights_grouped[active_group_mask])
                (
                    weight_entropy_all,
                    weight_entropy_all_min,
                    weight_entropy_all_max,
                ) = collect_weight_entropy_stats(weights_grouped)
                stats["listwise_weight_entropy"].append(active_weight_entropy.detach())
                stats["listwise_weight_entropy_min"].append(
                    active_weight_entropy_min.detach()
                )
                stats["listwise_weight_entropy_max"].append(
                    active_weight_entropy_max.detach()
                )
                stats["listwise_weight_entropy_all"].append(weight_entropy_all.detach())
                stats["listwise_weight_entropy_all_min"].append(
                    weight_entropy_all_min.detach()
                )
                stats["listwise_weight_entropy_all_max"].append(
                    weight_entropy_all_max.detach()
                )

                policy_log_probs_grouped = masked_group_log_softmax(
                    policy_seq_logps_grouped,
                    grouped_loss_masks,
                )
                policy_probs_grouped = torch.where(
                    grouped_loss_masks,
                    torch.exp(policy_log_probs_grouped),
                    torch.zeros_like(policy_log_probs_grouped),
                )
                per_group_projection_ce = -(
                    target_weights_grouped * policy_log_probs_grouped
                ).sum(dim=1)
                projection_ce_loss = (per_group_projection_ce * 0.0).sum()
                if drgrpo_token_primary and global_projection_group_weight > 0.0:
                    projection_ce_loss = (
                        projection_group_scale.detach() * per_group_projection_ce
                    ).sum() / max(global_projection_group_weight, 1e-8)
                else:
                    ce_group_mask = active_group_mask
                    if bool(ce_group_mask.any().item()):
                        projection_ce_loss = per_group_projection_ce[ce_group_mask].mean()
                projection_ce_loss_effective = projection_ce_loss
                if drgrpo_token_primary:
                    projection_ce_loss_effective = (
                        float(sequence_aux_coef) * projection_ce_loss
                    )

                listwise_centered_adv_grouped = compute_listwise_centered_advantages(
                    weights_grouped=target_weights_grouped,
                    behavior_seq_logps_grouped=behavior_seq_grouped,
                    valid_row_mask_grouped=grouped_loss_masks,
                ).to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                loss = projection_ce_loss
                zero_component = projection_ce_loss.detach() * 0.0
                infos["listwise_ce_loss"] = projection_ce_loss.detach()
                infos["listwise_projection_ce_loss"] = projection_ce_loss.detach()
                infos["listwise_projection_ce_loss_effective"] = (
                    projection_ce_loss_effective.detach()
                )
                infos["drgrpo_primary_loss"] = zero_component
                infos["listwise_ce_reference_loss"] = projection_ce_loss.detach()
                infos["listwise_bonus_loss_raw"] = zero_component
                infos["listwise_bonus_loss_weighted"] = zero_component
                infos["listwise_bonus_loss_effective"] = zero_component
                infos["listwise_helpfulness_proxy"] = zero_component
                infos["listwise_helpfulness_proxy_valid"] = zero_component
                infos["listwise_auto_scale_factor"] = zero_component
                infos["listwise_raw_to_drgrpo_ratio"] = zero_component
                infos["listwise_post_scale_ratio"] = zero_component
                infos["objective_effective_total_loss"] = (
                    projection_ce_loss_effective.detach()
                )
                infos["clip_loss"] = zero_component
                infos["listwise_adv_abs_mean"] = zero_component
                infos["listwise_adv_abs_mean_scaled"] = zero_component
                infos["drgrpo_adv_abs_mean"] = zero_component
                infos["combined_adv_abs_mean"] = zero_component
                infos["combined_token_pg_loss"] = zero_component
                drgrpo_pg_loss = None
                drgrpo_row_adv_flat = None
                weighted_drgrpo_row_adv_flat = None
                weighted_drgrpo_multiplier_flat = None
                weighted_drgrpo_delta_adv_flat = None
                drgrpo_active_row_mask = None
                drgrpo_active_row_count = None
                global_drgrpo_active_row_count = 0
                weighted_drgrpo_pg_loss = None
                if drgrpo_token_primary:
                    if float(args.beta) > 0.0:
                        raise NotImplementedError(
                            "The exact DrX utility-lift path currently requires beta=0 "
                            "so the weighted objective remains the pure Dr.GRPO clip "
                            "surrogate. Candidate-level trust should use "
                            "maxent_candidate_kl_coef."
                        )
                    drgrpo_row_adv_flat = probe_row_advantages
                    drgrpo_pg_loss = (probe_drgrpo_pg_row_loss * mb_loss_masks).mean()
                    infos["drgrpo_pg_loss"] = drgrpo_pg_loss.detach()
                    infos["drgrpo_primary_loss"] = drgrpo_pg_loss.detach()

                    token_active_row_mask_grouped = informative_group_mask[:, None] & grouped_loss_masks
                    drgrpo_active_row_mask = flatten_prompt_major_tensor(
                        token_active_row_mask_grouped
                    ).to(torch.bool)
                    drgrpo_active_row_count = int(drgrpo_active_row_mask.to(torch.int64).sum().item())
                    global_drgrpo_active_row_count = drgrpo_active_row_count
                    if dist.is_available() and dist.is_initialized():
                        drgrpo_active_row_count_tensor = torch.tensor(
                            float(drgrpo_active_row_count),
                            device=policy_seq_logps_grouped.device,
                        )
                        dist.all_reduce(drgrpo_active_row_count_tensor, op=dist.ReduceOp.SUM)
                        global_drgrpo_active_row_count = int(drgrpo_active_row_count_tensor.item())

                    token_weight_row_flat = flatten_prompt_major_tensor(token_target_weights_grouped).to(
                        device=new_logps.device,
                        dtype=new_logps.dtype,
                    )
                    weighted_drgrpo_multiplier_flat = float(group_size) * token_weight_row_flat
                    weighted_drgrpo_row_adv_flat = (
                        weighted_drgrpo_multiplier_flat * drgrpo_row_adv_flat
                    )
                    weighted_drgrpo_delta_adv_flat = (
                        weighted_drgrpo_row_adv_flat - drgrpo_row_adv_flat
                    )
                    weighted_drgrpo_pg_row_loss, _, _, _ = compute_token_level_clip_loss(
                        new_logps=new_logps,
                        behavior_logps=mb_behavior_logps.to(new_logps.dtype),
                        response_masks=mb_response_masks,
                        row_advantages=weighted_drgrpo_row_adv_flat,
                        clip_low=float(args.cliprange),
                        clip_high=float(args.cliprange),
                        constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                    )
                    if bool(drgrpo_active_row_mask.any().item()):
                        weighted_drgrpo_pg_loss = (
                            weighted_drgrpo_pg_row_loss[drgrpo_active_row_mask]
                        ).mean()
                    else:
                        weighted_drgrpo_pg_loss = (weighted_drgrpo_pg_row_loss * 0.0).sum()

                    pg_clipfrac = masked_mean(
                        (
                            torch.clamp(
                                torch.exp(
                                    (new_logps - mb_behavior_logps.to(new_logps.dtype)).clamp(
                                        -40.0,
                                        40.0,
                                    )
                                ),
                                1.0 - args.cliprange,
                                1.0 + args.cliprange,
                            )
                            < torch.exp(
                                (new_logps - mb_behavior_logps.to(new_logps.dtype)).clamp(
                                    -40.0,
                                    40.0,
                                )
                            )
                        ).float(),
                        mb_response_masks,
                        axis=1,
                    )
                    stats["pg_clipfrac"].append(pg_clipfrac.mean().min().detach())
                    stats["zero_pg_loss_count"].append(
                        (probe_drgrpo_pg_row_loss == 0).detach().sum().to(torch.float32)
                    )
                    stats["logprobs_diff_max"].append(
                        torch.amax(
                            (new_logps - mb_behavior_logps.to(new_logps.dtype)).detach()
                            * mb_response_masks
                        ).to(torch.float32)
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(
                            (new_logps - mb_behavior_logps.to(new_logps.dtype)).detach()
                            * mb_response_masks
                        ).to(torch.float32)
                    )
                    if bool(drgrpo_active_row_mask.any().item()):
                        drgrpo_adv_abs_mean = drgrpo_row_adv_flat[
                            drgrpo_active_row_mask
                        ].abs().mean()
                        listwise_adv_abs_mean = listwise_centered_adv_grouped[
                            contributing_group_mask
                        ].abs().mean() if contributing_group_count > 0 else zero_component
                        listwise_delta_adv_abs_mean = weighted_drgrpo_delta_adv_flat[
                            drgrpo_active_row_mask
                        ].abs().mean()
                        combined_adv_abs_mean = weighted_drgrpo_row_adv_flat[
                            drgrpo_active_row_mask
                        ].abs().mean()
                        listwise_weight_deviation = (
                            weighted_drgrpo_multiplier_flat[drgrpo_active_row_mask] - 1.0
                        ).abs().mean()
                    else:
                        drgrpo_adv_abs_mean = zero_component
                        listwise_adv_abs_mean = zero_component
                        listwise_delta_adv_abs_mean = zero_component
                        combined_adv_abs_mean = zero_component
                        listwise_weight_deviation = zero_component
                    infos["pg_loss"] = weighted_drgrpo_pg_loss.detach()
                    infos["combined_token_pg_loss"] = weighted_drgrpo_pg_loss.detach()
                    loss = weighted_drgrpo_pg_loss + projection_ce_loss_effective
                    infos["listwise_adv_abs_mean"] = listwise_adv_abs_mean.detach()
                    infos["listwise_adv_abs_mean_scaled"] = (
                        listwise_delta_adv_abs_mean.detach()
                    )
                    infos["drgrpo_adv_abs_mean"] = drgrpo_adv_abs_mean.detach()
                    infos["combined_adv_abs_mean"] = combined_adv_abs_mean.detach()
                    infos["listwise_auto_scale_factor"] = zero_component
                    infos["listwise_raw_to_drgrpo_ratio"] = listwise_weight_deviation.detach()
                    infos["listwise_post_scale_ratio"] = (
                        torch.abs(listwise_delta_adv_abs_mean.detach())
                        / (torch.abs(drgrpo_adv_abs_mean.detach()) + 1e-8)
                    )
                    infos["listwise_bonus_loss_raw"] = (
                        weighted_drgrpo_pg_loss.detach() - drgrpo_pg_loss.detach()
                    )
                    infos["listwise_bonus_loss_weighted"] = infos[
                        "listwise_bonus_loss_raw"
                    ]
                    infos["listwise_bonus_loss_effective"] = infos[
                        "listwise_bonus_loss_raw"
                    ]
                    infos["listwise_helpfulness_proxy"] = (
                        torch.abs(infos["listwise_bonus_loss_effective"])
                        / (torch.abs(infos["drgrpo_primary_loss"]) + 1e-8)
                    )
                    infos["listwise_helpfulness_proxy_valid"] = torch.tensor(
                        1.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["objective_effective_total_loss"] = loss.detach()
                elif not token_surrogate_primary:
                    infos["pg_loss"] = policy_loss.detach()
                    infos["listwise_bonus_loss_raw"] = policy_loss.detach()
                    infos["listwise_bonus_loss_weighted"] = policy_loss.detach()
                    infos["listwise_bonus_loss_effective"] = policy_loss.detach()
                    infos["objective_effective_total_loss"] = policy_loss.detach()

                clip_loss = None
                clip_low = clip_high = None
                baseline_value = None
                baseline_grouped = None
                reward_mass_grouped = None
                seq_ratio = None
                is_low_clipped = None
                is_high_clipped = None
                clip_region = None
                clip_coef = 0.0
                clip_mode = normalize_maxent_clip_mode(
                    getattr(args, "maxent_clip_mode", "sequence")
                )
                effective_clip_mode = "none"
                token_clip_adv_flat = None
                raw_seq_coeffs_grouped = None
                if drgrpo_token_primary:
                    clip_low = clip_high = float(args.cliprange)
                    effective_clip_mode = "token"
                elif bool(args.maxent_use_clip_objective):
                    clip_coef = coerce_non_negative_float(
                        args.maxent_clip_objective_coef,
                        default=1.0,
                    )
                    if clip_coef > 0:
                        effective_clip_mode = clip_mode
                        clip_range = args.maxent_clip_range
                        if clip_range is None:
                            clip_low = clip_high = float(args.cliprange)
                        else:
                            clip_low = clip_high = coerce_non_negative_float(
                                clip_range,
                                default=float(args.cliprange),
                            )
                        baseline = args.maxent_clip_adv_baseline
                        if baseline is None:
                            valid_group_denoms = grouped_loss_masks.sum(
                                dim=1,
                                keepdim=True,
                            ).clamp(min=1).to(weights_grouped.dtype)
                            baseline_grouped = torch.where(
                                grouped_loss_masks,
                                1.0 / valid_group_denoms,
                                torch.zeros_like(weights_grouped),
                            )
                        else:
                            baseline_value = float(baseline)
                            baseline_grouped = torch.where(
                                grouped_loss_masks,
                                torch.full_like(weights_grouped, baseline_value),
                                torch.zeros_like(weights_grouped),
                            )
                        if bool(args.maxent_clip_preserve_reward_mass):
                            reward_mass_grouped = torch.where(
                                grouped_loss_masks,
                                mb_reward_grouped,
                                torch.zeros_like(mb_reward_grouped),
                            ).sum(dim=1, keepdim=True)
                        clip_adv = compute_listwise_clip_advantages(
                            weights_grouped=weights_grouped,
                            valid_row_mask_grouped=grouped_loss_masks,
                            baseline_value=baseline_value,
                            baseline_grouped=baseline_grouped,
                            reward_mass_grouped=reward_mass_grouped,
                        )
                        if active_group_count > 0 and reward_mass_grouped is not None:
                            infos["listwise_clip_reward_mass_mean"] = (
                                reward_mass_grouped[active_group_mask]
                                .mean()
                                .detach()
                            )
                run_listwise_clip_objective = (
                    effective_clip_mode != "none" and not drgrpo_token_primary
                )
                if run_listwise_clip_objective:
                    infos["listwise_clip_preserve_reward_mass"] = torch.tensor(
                        float(bool(args.maxent_clip_preserve_reward_mass)),
                        device=policy_seq_logps_grouped.device,
                    )
                    active_clip_group_mask = active_group_mask
                    active_clip_group_count = global_active_group_count
                    if effective_clip_mode == "sequence":
                        log_seq_ratio = (
                            policy_seq_logps_grouped - behavior_seq_grouped
                        ).clamp(-40.0, 40.0)
                        seq_ratio = torch.exp(log_seq_ratio)
                        seq_ratio_clipped = torch.clamp(
                            seq_ratio,
                            1.0 - clip_low,
                            1.0 + clip_high,
                        )
                        clip_objective = torch.min(
                            seq_ratio * clip_adv,
                            seq_ratio_clipped * clip_adv,
                        )
                        per_group_clip_loss = -clip_objective.sum(dim=1)
                        if int(active_clip_group_count) > 0:
                            clip_loss = per_group_clip_loss[active_clip_group_mask].mean()
                        else:
                            clip_loss = (per_group_clip_loss * 0.0).sum()
                        loss = loss + clip_coef * clip_loss
                        infos["clip_loss"] = clip_loss.detach()
                        infos["listwise_bonus_loss_raw"] = clip_loss.detach()
                        infos["listwise_bonus_loss_weighted"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["listwise_bonus_loss_effective"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["objective_effective_total_loss"] = (
                            infos["listwise_bonus_loss_effective"]
                        )

                        is_low_clipped = (seq_ratio < 1.0 - clip_low) & (clip_adv < 0.0)
                        is_high_clipped = (seq_ratio > 1.0 + clip_high) & (clip_adv > 0.0)
                        clip_region = is_low_clipped | is_high_clipped
                        active_valid_group_entries = (
                            grouped_loss_masks & active_clip_group_mask.unsqueeze(1)
                        )
                        active_valid_entry_count = active_valid_group_entries.to(
                            torch.float32
                        ).sum()
                        if bool(active_valid_entry_count.gt(0).item()):
                            stats["clip_ratio_low"].append(
                                is_low_clipped.to(torch.float32).sum().detach()
                                / active_valid_entry_count
                            )
                            stats["clip_ratio_high"].append(
                                is_high_clipped.to(torch.float32).sum().detach()
                                / active_valid_entry_count
                            )
                            stats["clip_ratio_region"].append(
                                clip_region.to(torch.float32).sum().detach()
                                / active_valid_entry_count
                            )
                        else:
                            zero_stat = torch.zeros(
                                (),
                                dtype=torch.float32,
                                device=policy_seq_logps_grouped.device,
                            )
                            stats["clip_ratio_low"].append(zero_stat)
                            stats["clip_ratio_high"].append(zero_stat)
                            stats["clip_ratio_region"].append(zero_stat)
                    elif effective_clip_mode == "token":
                        token_clip_adv_flat = flatten_prompt_major_tensor(clip_adv).to(
                            device=new_logps.device,
                            dtype=new_logps.dtype,
                        )
                        per_row_clip_loss, seq_ratio, is_low_clipped, is_high_clipped = (
                            compute_token_level_clip_loss(
                                new_logps=new_logps,
                                behavior_logps=mb_behavior_logps.to(new_logps.dtype),
                                response_masks=mb_response_masks,
                                row_advantages=token_clip_adv_flat,
                                clip_low=clip_low,
                                clip_high=clip_high,
                                constant_normalizer=(
                                    self._listwise_token_clip_constant_normalizer()
                                ),
                            )
                        )
                        active_row_count = int(active_row_mask.to(torch.int64).sum().item())
                        if active_row_count > 0:
                            clip_loss = (
                                per_row_clip_loss * active_row_mask.to(per_row_clip_loss.dtype)
                            ).sum() / float(active_row_count)
                        else:
                            clip_loss = (per_row_clip_loss * 0.0).sum()
                        if token_surrogate_primary:
                            loss = clip_coef * clip_loss
                        else:
                            loss = loss + clip_coef * clip_loss
                        infos["clip_loss"] = clip_loss.detach()
                        infos["listwise_bonus_loss_raw"] = clip_loss.detach()
                        infos["listwise_bonus_loss_weighted"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["listwise_bonus_loss_effective"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["objective_effective_total_loss"] = loss.detach()

                        clip_region = is_low_clipped | is_high_clipped
                        active_token_mask = (
                            mb_response_masks
                            & active_row_mask.unsqueeze(1)
                            & valid_row_mask.unsqueeze(1)
                        )
                        active_token_count = active_token_mask.to(torch.float32).sum()
                        if bool(active_token_count.gt(0).item()):
                            stats["clip_ratio_low"].append(
                                is_low_clipped.to(torch.float32).sum().detach()
                                / active_token_count
                            )
                            stats["clip_ratio_high"].append(
                                is_high_clipped.to(torch.float32).sum().detach()
                                / active_token_count
                            )
                            stats["clip_ratio_region"].append(
                                clip_region.to(torch.float32).sum().detach()
                                / active_token_count
                            )
                        else:
                            zero_stat = torch.zeros(
                                (),
                                dtype=torch.float32,
                                device=policy_seq_logps_grouped.device,
                            )
                            stats["clip_ratio_low"].append(zero_stat)
                            stats["clip_ratio_high"].append(zero_stat)
                            stats["clip_ratio_region"].append(zero_stat)
                has_token_signal = global_drgrpo_active_row_count > 0
                has_projection_signal = (
                    float(sequence_aux_coef) > 0.0
                    and global_projection_group_weight > 0.0
                    and global_projection_scale_mass > 0.0
                )
                if drgrpo_token_primary:
                    skip_zero_signal_update = not (has_token_signal or has_projection_signal)
                else:
                    skip_zero_signal_update = global_active_group_count <= 0
                skip_listwise_backward = skip_zero_signal_update
                if local_grad_step == 1 and skip_zero_signal_update:
                    if drgrpo_token_primary:
                        logging.info(
                            "listwise minibatch has no informative token signal and no "
                            "projection signal across any rank; skipping the exact DrX "
                            "update."
                        )
                    else:
                        logging.info(
                            "listwise minibatch has no active prompt groups across "
                            "any rank; skipping backward, optimizer, and controller "
                            "updates."
                        )
                if skip_listwise_backward:
                    backward_chunk_size = 0
                    clip_backward_chunk_size = 0
                else:
                    if token_surrogate_primary:
                        seq_coeffs_grouped = torch.zeros_like(
                            target_weights_grouped,
                            device=target_weights_grouped.device,
                            dtype=target_weights_grouped.dtype,
                        )
                    elif drgrpo_token_primary:
                        if (
                            float(sequence_aux_coef) > 0.0
                            and global_projection_group_weight > 0.0
                            and global_projection_scale_mass > 0.0
                        ):
                            seq_coeffs_grouped = compute_drx_projection_sequence_coefficients(
                                policy_seq_logps_grouped=policy_seq_logps_grouped.detach(),
                                projection_target_grouped=projection_target_grouped.detach(),
                                projection_group_scale=projection_group_scale.detach(),
                                valid_row_mask_grouped=grouped_loss_masks.detach(),
                                normalizer_total_group_weight=global_projection_group_weight,
                            ).to(
                                device=target_weights_grouped.device,
                                dtype=target_weights_grouped.dtype,
                            )
                            seq_coeffs_grouped = float(sequence_aux_coef) * seq_coeffs_grouped
                        else:
                            seq_coeffs_grouped = torch.zeros_like(
                                target_weights_grouped,
                                device=target_weights_grouped.device,
                                dtype=target_weights_grouped.dtype,
                            )
                        raw_seq_coeffs_grouped = seq_coeffs_grouped
                    else:
                        seq_coeffs_grouped = compute_listwise_sequence_coefficients(
                            policy_seq_logps_grouped=policy_seq_logps_grouped.detach(),
                            weights_grouped=target_weights_grouped.detach(),
                            active_group_mask=active_group_mask.detach(),
                            normalizer_active_group_count=global_active_group_count,
                            valid_row_mask_grouped=grouped_loss_masks.detach(),
                            behavior_seq_logps_grouped=(
                                behavior_seq_grouped.detach()
                                if clip_coef > 0 and effective_clip_mode == "sequence"
                                else None
                            ),
                            clip_row_mask_grouped=(
                                grouped_loss_masks.detach()
                                if clip_coef > 0 and effective_clip_mode == "sequence"
                                else None
                            ),
                            reward_mass_grouped=(
                                reward_mass_grouped.detach()
                                if reward_mass_grouped is not None
                                and clip_coef > 0
                                and effective_clip_mode == "sequence"
                                else None
                            ),
                            clip_low=0.0 if clip_low is None else clip_low,
                            clip_high=0.0 if clip_high is None else clip_high,
                            clip_coef=(
                                clip_coef if effective_clip_mode == "sequence" else 0.0
                            ),
                            baseline_value=baseline_value,
                            baseline_grouped=(
                                baseline_grouped.detach()
                                if baseline_grouped is not None
                                and clip_coef > 0
                                and effective_clip_mode == "sequence"
                                else None
                            ),
                        )
                        raw_seq_coeffs_grouped = seq_coeffs_grouped
                    token_clip_enabled = False
                    token_clip_adv_for_backward = None
                    token_clip_active_row_mask = None
                    token_clip_row_count_normalizer = None
                    token_clip_coef_for_backward = 0.0
                    if drgrpo_token_primary:
                        token_clip_enabled = (
                            weighted_drgrpo_row_adv_flat is not None
                            and drgrpo_active_row_mask is not None
                            and global_drgrpo_active_row_count > 0
                        )
                        token_clip_adv_for_backward = weighted_drgrpo_row_adv_flat
                        token_clip_active_row_mask = drgrpo_active_row_mask
                        token_clip_row_count_normalizer = global_drgrpo_active_row_count
                        token_clip_coef_for_backward = 1.0
                    elif (
                        effective_clip_mode == "token"
                        and clip_coef > 0.0
                        and token_clip_adv_flat is not None
                    ):
                        token_clip_enabled = True
                        token_clip_adv_for_backward = token_clip_adv_flat
                        token_clip_active_row_mask = active_row_mask
                        token_clip_row_count_normalizer = global_active_row_count
                        token_clip_coef_for_backward = clip_coef
                    run_grad_probe, grad_probe_update_index = (
                        self._should_run_listwise_branch_grad_diagnostics(
                            local_grad_step=local_grad_step,
                            grad_acc_step=grad_acc_step,
                        )
                    )
                    if (
                        run_grad_probe
                        and drgrpo_token_primary
                        and raw_seq_coeffs_grouped is not None
                        and drgrpo_row_adv_flat is not None
                        and drgrpo_active_row_mask is not None
                        and global_drgrpo_active_row_count > 0
                    ):
                        grad_probe_infos = self._probe_listwise_branch_gradient_metrics(
                            input_ids=mb_input_ids,
                            att_mask=mb_att_mask,
                            response_masks=mb_response_masks,
                            raw_seq_coeffs_grouped=raw_seq_coeffs_grouped.detach(),
                            length_normalize=length_normalize_policy,
                            behavior_logps=mb_behavior_logps,
                            row_advantages=drgrpo_row_adv_flat,
                            active_row_mask=drgrpo_active_row_mask,
                            active_row_count_normalizer=global_drgrpo_active_row_count,
                            clip_low=0.0 if clip_low is None else clip_low,
                            clip_high=0.0 if clip_high is None else clip_high,
                            sequence_aux_coef=(
                                float(sequence_aux_coef)
                            ),
                            global_active_group_count=global_contributing_group_count,
                            update_index=grad_probe_update_index,
                        )
                        if grad_probe_infos:
                            infos.update(grad_probe_infos)
                    (
                        backward_chunk_size,
                        clip_backward_chunk_size,
                    ) = self._backward_listwise_sequence_coefficients(
                        mb_input_ids,
                        mb_att_mask,
                        mb_response_masks,
                        flatten_prompt_major_tensor(seq_coeffs_grouped),
                        length_normalize=length_normalize_policy,
                        behavior_logps=(
                            mb_behavior_logps if token_clip_enabled else None
                        ),
                        row_advantages=(
                            token_clip_adv_for_backward if token_clip_enabled else None
                        ),
                        active_row_mask=(
                            token_clip_active_row_mask if token_clip_enabled else None
                        ),
                        active_row_count_normalizer=(
                            token_clip_row_count_normalizer
                            if token_clip_enabled
                            else None
                        ),
                        clip_low=0.0 if clip_low is None else clip_low,
                        clip_high=0.0 if clip_high is None else clip_high,
                        clip_coef=(
                            token_clip_coef_for_backward if token_clip_enabled else 0.0
                        ),
                    )
                    if local_grad_step == 1:
                        logging.info(
                            "listwise backward done: backward_chunk=%s clip_backward_chunk=%s active_groups=%s clip_mode=%s",
                            backward_chunk_size,
                            clip_backward_chunk_size,
                            active_group_count,
                            effective_clip_mode,
                        )
                infos["listwise_policy_backward_chunk_size"] = torch.tensor(
                    float(backward_chunk_size),
                    device=policy_seq_logps_grouped.device,
                )
                infos["listwise_clip_backward_chunk_size"] = torch.tensor(
                    float(clip_backward_chunk_size),
                    device=policy_seq_logps_grouped.device,
                )

                if local_grad_step % self.strategy.grad_acc_step == 0:
                    update_index = max(local_grad_step // grad_acc_step, 1)
                    if skip_zero_signal_update:
                        if not self._listwise_zero_signal_skip_warned:
                            logging.warning(
                                "Skipping a zero-signal listwise optimizer step "
                                "because every prompt group in the minibatch is "
                                "neutral. This preserves the no-op update without "
                                "forcing a distributed backward/optimizer pass."
                            )
                            self._listwise_zero_signal_skip_warned = True
                    else:
                        if not self._listwise_grad_norm_logging_disabled_warned:
                            logging.warning(
                                "Skipping listwise policy_grad_norm logging because the "
                                "chunked backward path uses per-sequence passes and "
                                "DeepSpeed gradient-norm collectives can hang while "
                                "ranks finish unevenly."
                            )
                            self._listwise_grad_norm_logging_disabled_warned = True
                    stats["policy_grad_norm"].append(torch.tensor(0.0))

                if not skip_zero_signal_update:
                    self.strategy.optimizer_step(
                        self.optimizer, self.model, self.scheduler
                    )
                if (
                    local_grad_step % grad_acc_step == 0
                    and self.strategy.is_rank_0()
                ):
                    logging.info(
                        "listwise optimizer update %s/%s: status=%s "
                        "local_active_groups=%s global_active_groups=%s "
                        "local_active_rows=%s global_active_rows=%s "
                        "policy_probe_chunk=%s backward_chunk=%s "
                        "clip_backward_chunk=%s clip_mode=%s",
                        update_index,
                        total_optimizer_updates,
                        (
                            "skipped_zero_signal"
                            if skip_zero_signal_update
                            else (
                                "applied_exact_drx_weighted_drgrpo"
                                if drgrpo_token_primary
                                else "applied"
                            )
                        ),
                        active_group_count,
                        global_active_group_count,
                        active_row_count,
                        global_active_row_count,
                        policy_chunk_size,
                        backward_chunk_size,
                        clip_backward_chunk_size,
                        effective_clip_mode,
                    )

                with torch.no_grad():
                    measured_kl_value = None
                    if entropy is not None:
                        infos["entropy"] = masked_mean(entropy, mb_response_masks)
                    infos["listwise_policy_prob_mean"] = policy_probs_grouped.mean().detach()
                    infos["listwise_weight_mean"] = weights_grouped.mean().detach()
                    infos["listwise_weight_std"] = (
                        weights_grouped.to(torch.float32).std(unbiased=False).detach()
                    )
                    infos["listwise_weight_entropy"] = active_weight_entropy.detach().to(
                        device=policy_seq_logps_grouped.device,
                        dtype=torch.float32,
                    )
                    infos["weight_entropy"] = infos["listwise_weight_entropy"]
                    infos["listwise_weight_entropy_active"] = infos[
                        "listwise_weight_entropy"
                    ]
                    infos["listwise_weight_entropy_min"] = (
                        active_weight_entropy_min.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    infos["weight_entropy_min"] = infos["listwise_weight_entropy_min"]
                    infos["listwise_weight_entropy_max"] = (
                        active_weight_entropy_max.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    infos["weight_entropy_max"] = infos["listwise_weight_entropy_max"]
                    infos["listwise_weight_entropy_all"] = (
                        weight_entropy_all.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    infos["listwise_weight_entropy_all_min"] = (
                        weight_entropy_all_min.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    infos["listwise_weight_entropy_all_max"] = (
                        weight_entropy_all_max.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    infos["listwise_neutral_group_frac"] = neutral_group_mask.to(
                        torch.float32
                    ).mean().detach()
                    infos["listwise_informative_group_frac"] = active_group_mask.to(
                        torch.float32
                    ).mean().detach()
                    infos["listwise_informative_group_count_global"] = torch.tensor(
                        float(global_active_group_count),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_contributing_group_frac"] = contributing_group_mask.to(
                        torch.float32
                    ).mean().detach()
                    infos["listwise_contributing_group_count_global"] = torch.tensor(
                        float(global_contributing_group_count),
                        device=policy_seq_logps_grouped.device,
                    )
                    if not drgrpo_token_primary:
                        infos["listwise_active_group_frac"] = infos[
                            "listwise_informative_group_frac"
                        ]
                        infos["listwise_active_group_count_global"] = infos[
                            "listwise_informative_group_count_global"
                        ]
                    else:
                        infos["listwise_active_group_frac"] = infos[
                            "listwise_informative_group_frac"
                        ]
                        infos["listwise_active_group_count_global"] = infos[
                            "listwise_informative_group_count_global"
                        ]
                    infos["listwise_valid_row_frac"] = grouped_loss_masks.to(
                        torch.float32
                    ).mean().detach()
                    infos["listwise_partial_group_frac"] = (
                        grouped_loss_masks.any(dim=1)
                        & (~grouped_loss_masks.all(dim=1))
                    ).to(torch.float32).mean().detach()
                    infos["listwise_valid_weight_mass"] = target_weights_grouped.sum(
                        dim=1
                    ).mean().detach()
                    infos["listwise_clip_mode_sequence"] = torch.tensor(
                        1.0 if effective_clip_mode == "sequence" else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_clip_mode_token"] = torch.tensor(
                        1.0 if effective_clip_mode == "token" else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_clip_mode_none"] = torch.tensor(
                        1.0 if effective_clip_mode == "none" else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_token_surrogate_primary"] = torch.tensor(
                        1.0 if token_surrogate_primary else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_drgrpo_token_primary"] = torch.tensor(
                        1.0 if drgrpo_token_primary else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_combined_auto_scale_enabled"] = torch.tensor(
                        0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_sequence_aux_coef"] = torch.tensor(
                        float(sequence_aux_coef),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_candidate_kl_coef"] = torch.tensor(
                        float(candidate_kl_coef),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_branch_grad_diagnostics"] = torch.tensor(
                        1.0 if bool(args.maxent_branch_grad_diagnostics) else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_branch_grad_diagnostics_interval"] = torch.tensor(
                        float(args.maxent_branch_grad_diagnostics_interval),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_branch_grad_diagnostics_max_steps"] = torch.tensor(
                        float(args.maxent_branch_grad_diagnostics_max_steps),
                        device=policy_seq_logps_grouped.device,
                    )
                    if "listwise_grad_probe_enabled" not in infos:
                        infos["listwise_grad_probe_enabled"] = torch.tensor(
                            0.0,
                            device=policy_seq_logps_grouped.device,
                        )
                        infos["listwise_grad_probe_valid"] = torch.tensor(
                            0.0,
                            device=policy_seq_logps_grouped.device,
                        )
                    infos["listwise_q_temperature"] = torch.tensor(
                        float(args.maxent_q_temperature),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_q_epsilon"] = torch.tensor(
                        float(args.maxent_q_epsilon),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_length_normalize_ref"] = torch.tensor(
                        1.0 if bool(args.maxent_length_normalize_ref) else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_length_normalize_policy"] = torch.tensor(
                        1.0 if bool(args.maxent_length_normalize_policy) else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_skip_zero_variance_groups"] = torch.tensor(
                        1.0
                        if bool(args.maxent_listwise_skip_zero_variance_groups)
                        else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_use_clip_objective"] = torch.tensor(
                        1.0 if bool(args.maxent_use_clip_objective) else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_clip_objective_coef"] = torch.tensor(
                        float(args.maxent_clip_objective_coef),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_clip_range"] = torch.tensor(
                        float(
                            args.maxent_clip_range
                            if args.maxent_clip_range is not None
                            else args.cliprange
                        ),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_clip_adv_baseline_override"] = torch.tensor(
                        0.0 if args.maxent_clip_adv_baseline is None else 1.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    if args.maxent_clip_adv_baseline is not None:
                        infos["listwise_clip_adv_baseline_value"] = torch.tensor(
                            float(args.maxent_clip_adv_baseline),
                            device=policy_seq_logps_grouped.device,
                        )
                    infos["listwise_logprob_chunk_size"] = torch.tensor(
                        float(args.maxent_logprob_chunk_size),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_backward_chunk_size"] = torch.tensor(
                        float(args.maxent_backward_chunk_size),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_backward_token_budget"] = torch.tensor(
                        float(args.maxent_backward_token_budget),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_reference_logprobs_from_model"] = torch.tensor(
                        1.0
                        if args.maxent_reference_logprobs_source == "model"
                        else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_reference_logprobs_from_behavior"] = (
                        torch.tensor(
                            1.0
                            if args.maxent_reference_logprobs_source == "behavior"
                            else 0.0,
                            device=policy_seq_logps_grouped.device,
                        )
                    )
                    infos["listwise_zero_signal_skip"] = torch.tensor(
                        1.0 if skip_zero_signal_update else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    if token_surrogate_primary:
                        infos["pg_loss"] = loss.detach()
                    elif not drgrpo_token_primary:
                        logprobs_diff = (new_logps - mb_behavior_logps) * mb_response_masks
                        stats["logprobs_diff_max"].append(
                            torch.amax(logprobs_diff.detach())
                        )
                        stats["logprobs_diff_min"].append(
                            torch.amin(logprobs_diff.detach())
                        )
                        stats["zero_pg_loss_count"].append(
                            (per_group_policy_loss == 0).detach().sum().to(
                                torch.float32
                            )
                        )

                    if ref_logps is not None:
                        mb_ref_logps = ref_logps[mini_batch_inds]
                        _, [_, _, _, mb_ref_logps] = self._trim_policy_batch(
                            input_ids[mini_batch_inds],
                            att_mask[mini_batch_inds],
                            response_masks[mini_batch_inds],
                            ref_logps[mini_batch_inds],
                        )
                        log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                        kl3 = torch.expm1(log_ratio) - log_ratio
                        infos["kl3"] = masked_mean(kl3, mb_response_masks).detach()
                        measured_kl_value = float(infos["kl3"].cpu().item())

                    active_tau_target_metric = self._resolve_tau_target_metric(
                        global_step=int(self.global_step),
                    )
                    reduced_weight_entropy = None
                    reduced_kl_value = None
                    reduced_semantic_entropy_mu = None
                    reduced_exploration_gain_any_correct = None
                    reduced_exploration_gain_drgrpo = None
                    tau_loss_value = None
                    if not skip_zero_signal_update:
                        # Tau only controls informative prompt groups. Neutral
                        # groups are overwritten with uniform weights, so
                        # including them in the controller statistic would
                        # spuriously push tau toward the floor.
                        reduced_weight_entropy = self._distributed_weighted_mean_scalar(
                            active_weight_entropy,
                            weight=active_group_count,
                        )
                        reduced_kl_value = self._distributed_mean_scalar(
                            measured_kl_value
                        )
                        if reduced_weight_entropy is not None:
                            infos["listwise_weight_entropy_controller"] = torch.tensor(
                                reduced_weight_entropy,
                                device=policy_seq_logps_grouped.device,
                            )
                        reduced_semantic_entropy_mu = self._distributed_weighted_mean_scalar(
                            semantic_entropy_prompt_mean_local,
                            weight=gain_group_count,
                        )
                        reduced_exploration_gain_any_correct = (
                            self._distributed_weighted_mean_scalar(
                                exploration_gain_any_correct_mean_local,
                                weight=gain_group_count,
                            )
                        )
                        reduced_exploration_gain_drgrpo = (
                            self._distributed_weighted_mean_scalar(
                                exploration_gain_drgrpo_mean_local,
                                weight=gain_group_count,
                            )
                        )
                        tau_metric_name, tau_metric_value = (
                            self._select_tau_adaptation_metric(
                                semantic_entropy_mu=reduced_semantic_entropy_mu,
                                exploration_gain_any_correct=(
                                    reduced_exploration_gain_any_correct
                                ),
                                exploration_gain_drgrpo=reduced_exploration_gain_drgrpo,
                            )
                        )
                        if tau_metric_value is not None:
                            infos["listwise_tau_adaptation_metric_value"] = torch.tensor(
                                float(tau_metric_value),
                                device=policy_seq_logps_grouped.device,
                            )
                        infos["listwise_tau_adaptation_metric_is_semantic_entropy_mu"] = torch.tensor(
                            1.0 if tau_metric_name == "semantic_entropy_mu" else 0.0,
                            device=policy_seq_logps_grouped.device,
                        )
                        infos["listwise_tau_adaptation_metric_is_exploration_gain_any_correct"] = torch.tensor(
                            1.0
                            if tau_metric_name == "exploration_gain_any_correct"
                            else 0.0,
                            device=policy_seq_logps_grouped.device,
                        )
                        infos["listwise_tau_adaptation_metric_is_exploration_gain_drgrpo"] = torch.tensor(
                            1.0 if tau_metric_name == "exploration_gain_drgrpo" else 0.0,
                            device=policy_seq_logps_grouped.device,
                        )
                        if reduced_semantic_entropy_mu is not None:
                            infos["listwise_tau_signal_semantic_entropy_mu"] = torch.tensor(
                                float(reduced_semantic_entropy_mu),
                                device=policy_seq_logps_grouped.device,
                            )
                        if reduced_exploration_gain_any_correct is not None:
                            infos["listwise_tau_signal_exploration_gain_any_correct"] = torch.tensor(
                                float(reduced_exploration_gain_any_correct),
                                device=policy_seq_logps_grouped.device,
                            )
                        if reduced_exploration_gain_drgrpo is not None:
                            infos["listwise_tau_signal_exploration_gain_drgrpo"] = torch.tensor(
                                float(reduced_exploration_gain_drgrpo),
                                device=policy_seq_logps_grouped.device,
                            )
                        if bool(args.maxent_tau_learnable):
                            update_listwise_tau_metric_ema(
                                self._maxent_controller_state,
                                measured_metric=tau_metric_value,
                            )
                            with torch.enable_grad():
                                args.maxent_tau, tau_loss_value = (
                                    self._maybe_update_learnable_tau(
                                        measured_metric=tau_metric_value,
                                        target_metric=active_tau_target_metric,
                                        global_step=int(self.global_step),
                                    )
                                )
                        elif bool(args.maxent_tau_controller_enabled):
                            args.maxent_tau = maybe_update_listwise_tau(
                                args.maxent_tau,
                                measured_metric=tau_metric_value,
                                global_step=int(self.global_step),
                                state=self._maxent_controller_state,
                                target_metric=args.maxent_tau_target_metric,
                                target_metric_start=args.maxent_tau_target_metric_start,
                                target_metric_peak=args.maxent_tau_target_metric_peak,
                                target_metric_peak_step=args.maxent_tau_target_metric_peak_step,
                                target_metric_final=args.maxent_tau_target_metric_final,
                                target_metric_horizon=args.maxent_tau_target_metric_horizon,
                                tau_lr=args.maxent_tau_lr,
                                tau_min=args.maxent_tau_min,
                                tau_max=args.maxent_tau_max,
                                tau_warmup_steps=args.maxent_tau_warmup_steps,
                            )
                        if bool(args.maxent_beta_controller_enabled):
                            args.beta = maybe_update_listwise_beta(
                                args.beta,
                                measured_kl=reduced_kl_value,
                                kl_target=args.kl_target,
                                kl_horizon=args.kl_horizon,
                                kl_ctl_step_size=args.kl_ctl_step_size,
                            )
                    self._enforce_fixed_listwise_hparams()
                    infos["tau"] = torch.tensor(
                        float(args.maxent_tau),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["beta"] = torch.tensor(
                        float(args.beta),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["weight_norm_denom"] = torch.tensor(
                        float(max(args.maxent_tau, 1e-8)),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["kl_controller_enabled"] = torch.tensor(
                        1.0 if bool(args.maxent_beta_controller_enabled) else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["tau_learnable_enabled"] = torch.tensor(
                        1.0 if bool(args.maxent_tau_learnable) else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["tau_controller_enabled"] = torch.tensor(
                        1.0 if bool(args.maxent_tau_controller_enabled) else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    if tau_loss_value is not None:
                        infos["tau_loss"] = torch.tensor(
                            tau_loss_value,
                            device=policy_seq_logps_grouped.device,
                        )
                    if active_tau_target_metric is not None:
                        infos["listwise_tau_target_metric"] = torch.tensor(
                            float(active_tau_target_metric),
                            device=policy_seq_logps_grouped.device,
                        )
                    tau_metric_ema = getattr(
                        self._maxent_controller_state, "tau_metric_ema", None
                    )
                    if isinstance(tau_metric_ema, (int, float)) and math.isfinite(
                        float(tau_metric_ema)
                    ):
                        infos["listwise_tau_metric_ema"] = torch.tensor(
                            float(tau_metric_ema),
                            device=policy_seq_logps_grouped.device,
                        )

        scalar_stat_device = (
            next(
                (
                    value.device
                    for values in stats.values()
                    for value in values
                    if isinstance(value, torch.Tensor)
                ),
                advantages.device,
            )
            if isinstance(advantages, torch.Tensor)
            else torch.device("cpu")
        )
        infos["policy_grad_norm"] = _stack_scalar_stats(
            stats["policy_grad_norm"],
            device=scalar_stat_device,
        ).max()
        if stats["listwise_weight_entropy"]:
            infos["listwise_weight_entropy"] = _stack_scalar_stats(
                stats["listwise_weight_entropy"],
                device=scalar_stat_device,
            ).mean()
            infos["weight_entropy"] = infos["listwise_weight_entropy"]
            infos["listwise_weight_entropy_active"] = infos["listwise_weight_entropy"]
            infos["listwise_weight_entropy_min"] = _stack_scalar_stats(
                stats["listwise_weight_entropy_min"],
                device=scalar_stat_device,
            ).min()
            infos["weight_entropy_min"] = infos["listwise_weight_entropy_min"]
            infos["listwise_weight_entropy_max"] = _stack_scalar_stats(
                stats["listwise_weight_entropy_max"],
                device=scalar_stat_device,
            ).max()
            infos["weight_entropy_max"] = infos["listwise_weight_entropy_max"]
        if stats["listwise_weight_entropy_all"]:
            infos["listwise_weight_entropy_all"] = _stack_scalar_stats(
                stats["listwise_weight_entropy_all"],
                device=scalar_stat_device,
            ).mean()
            infos["listwise_weight_entropy_all_min"] = _stack_scalar_stats(
                stats["listwise_weight_entropy_all_min"],
                device=scalar_stat_device,
            ).min()
            infos["listwise_weight_entropy_all_max"] = _stack_scalar_stats(
                stats["listwise_weight_entropy_all_max"],
                device=scalar_stat_device,
            ).max()
        if stats["listwise_semantic_cluster_entropy"]:
            infos["listwise_semantic_cluster_entropy"] = _stack_scalar_stats(
                stats["listwise_semantic_cluster_entropy"],
                device=scalar_stat_device,
            ).mean()
            infos["listwise_semantic_cluster_entropy_norm"] = infos[
                "listwise_semantic_cluster_entropy"
            ]
        if stats["listwise_semantic_cluster_count"]:
            infos["listwise_semantic_cluster_count"] = _stack_scalar_stats(
                stats["listwise_semantic_cluster_count"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_competitive_mode_count"]:
            infos["listwise_semantic_competitive_mode_count"] = _stack_scalar_stats(
                stats["listwise_semantic_competitive_mode_count"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_competitive_mode_eligible_count"]:
            infos["listwise_semantic_competitive_mode_eligible_count"] = _stack_scalar_stats(
                stats["listwise_semantic_competitive_mode_eligible_count"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_competitive_mode_eligible_frac"]:
            infos["listwise_semantic_competitive_mode_eligible_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_competitive_mode_eligible_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_distinct_correct_mode_count"]:
            infos["listwise_semantic_distinct_correct_mode_count"] = _stack_scalar_stats(
                stats["listwise_semantic_distinct_correct_mode_count"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_distinct_correct_mode_frac"]:
            infos["listwise_semantic_distinct_correct_mode_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_distinct_correct_mode_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_competitive_mode_best_score"]:
            infos["listwise_semantic_competitive_mode_best_score"] = _stack_scalar_stats(
                stats["listwise_semantic_competitive_mode_best_score"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_competitive_mode_second_score"]:
            infos["listwise_semantic_competitive_mode_second_score"] = _stack_scalar_stats(
                stats["listwise_semantic_competitive_mode_second_score"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_competitive_mode_gap"]:
            infos["listwise_semantic_competitive_mode_gap"] = _stack_scalar_stats(
                stats["listwise_semantic_competitive_mode_gap"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_explore_budget_mean"]:
            infos["listwise_semantic_explore_budget_mean"] = _stack_scalar_stats(
                stats["listwise_semantic_explore_budget_mean"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_explore_budget_saturated_frac"]:
            infos["listwise_semantic_explore_budget_saturated_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_explore_budget_saturated_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_explore_applied_group_frac"]:
            infos["listwise_semantic_explore_applied_group_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_explore_applied_group_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_verified_bonus_applied_group_frac"]:
            infos["listwise_semantic_verified_bonus_applied_group_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_verified_bonus_applied_group_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_prompt_selected_frac"]:
            infos["listwise_semantic_prompt_selected_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_prompt_selected_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_prompt_rejected_low_opp_frac"]:
            infos["listwise_semantic_prompt_rejected_low_opp_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_prompt_rejected_low_opp_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_prompt_rejected_nonpositive_frac"]:
            infos["listwise_semantic_prompt_rejected_nonpositive_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_prompt_rejected_nonpositive_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_prompt_rejected_len_guard_frac"]:
            infos["listwise_semantic_prompt_rejected_len_guard_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_prompt_rejected_len_guard_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_prompt_rejected_format_guard_frac"]:
            infos["listwise_semantic_prompt_rejected_format_guard_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_prompt_rejected_format_guard_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_prompt_rejected_verified_bonus_len_guard_frac"]:
            infos["listwise_semantic_prompt_rejected_verified_bonus_len_guard_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_prompt_rejected_verified_bonus_len_guard_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_prompt_rejected_verified_bonus_format_guard_frac"]:
            infos["listwise_semantic_prompt_rejected_verified_bonus_format_guard_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_prompt_rejected_verified_bonus_format_guard_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_moved_mass_l1"]:
            infos["listwise_semantic_moved_mass_l1"] = _stack_scalar_stats(
                stats["listwise_semantic_moved_mass_l1"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_alpha_raw_mean"]:
            infos["listwise_semantic_alpha_raw_mean"] = _stack_scalar_stats(
                stats["listwise_semantic_alpha_raw_mean"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_alpha_applied_mean"]:
            infos["listwise_semantic_alpha_applied_mean"] = _stack_scalar_stats(
                stats["listwise_semantic_alpha_applied_mean"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_verified_bonus_mean"]:
            infos["listwise_semantic_verified_bonus_mean"] = _stack_scalar_stats(
                stats["listwise_semantic_verified_bonus_mean"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_utility_q"]:
            infos["listwise_semantic_expected_utility_q"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_utility_q"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_utility_explore_target"]:
            infos["listwise_semantic_expected_utility_explore_target"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_utility_explore_target"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_utility_final_w"]:
            infos["listwise_semantic_expected_utility_final_w"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_utility_final_w"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_len_q"]:
            infos["listwise_semantic_expected_len_q"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_len_q"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_len_explore_target"]:
            infos["listwise_semantic_expected_len_explore_target"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_len_explore_target"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_len_final_w"]:
            infos["listwise_semantic_expected_len_final_w"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_len_final_w"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_format_q"]:
            infos["listwise_semantic_expected_format_q"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_format_q"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_format_explore_target"]:
            infos["listwise_semantic_expected_format_explore_target"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_format_explore_target"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_expected_format_final_w"]:
            infos["listwise_semantic_expected_format_final_w"] = _stack_scalar_stats(
                stats["listwise_semantic_expected_format_final_w"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_answer_key_extracted_frac"]:
            infos["listwise_semantic_answer_key_extracted_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_answer_key_extracted_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_trace_extracted_frac"]:
            infos["listwise_semantic_trace_extracted_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_trace_extracted_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_signature_extracted_frac"]:
            infos["listwise_semantic_signature_extracted_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_signature_extracted_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_cluster_valid_frac"]:
            infos["listwise_semantic_cluster_valid_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_cluster_valid_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_trace_truncated_frac"]:
            infos["listwise_semantic_trace_truncated_frac"] = _stack_scalar_stats(
                stats["listwise_semantic_trace_truncated_frac"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_cluster_entropy_ref"]:
            infos["listwise_semantic_cluster_entropy_ref"] = _stack_scalar_stats(
                stats["listwise_semantic_cluster_entropy_ref"],
                device=scalar_stat_device,
            ).mean()
        if stats["listwise_semantic_cluster_entropy_gain"]:
            infos["listwise_semantic_cluster_entropy_gain"] = _stack_scalar_stats(
                stats["listwise_semantic_cluster_entropy_gain"],
                device=scalar_stat_device,
            ).mean()
        semantic_prompt_entropy = _concat_vector_stats(
            prompt_diag_stats["listwise_semantic_entropy_prompt"],
            device=scalar_stat_device,
        )
        semantic_prompt_gain_any = _concat_vector_stats(
            prompt_diag_stats["listwise_semantic_exploration_gain_any_correct_prompt"],
            device=scalar_stat_device,
        )
        semantic_prompt_gain_drgrpo = _concat_vector_stats(
            prompt_diag_stats["listwise_semantic_exploration_gain_drgrpo_prompt"],
            device=scalar_stat_device,
        )
        if semantic_prompt_entropy.numel() > 0:
            semantic_prompt_entropy = _all_gather_variable_length_1d_tensor(
                semantic_prompt_entropy
            )
            semantic_prompt_gain_any = _all_gather_variable_length_1d_tensor(
                semantic_prompt_gain_any
            )
            semantic_prompt_gain_drgrpo = _all_gather_variable_length_1d_tensor(
                semantic_prompt_gain_drgrpo
            )

            def _prompt_corr(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
                if lhs.numel() < 2 or rhs.numel() < 2:
                    return lhs.new_zeros(())
                lhs_centered = lhs - lhs.mean()
                rhs_centered = rhs - rhs.mean()
                denom = lhs_centered.norm() * rhs_centered.norm()
                if float(denom.item()) <= 1e-12:
                    return lhs.new_zeros(())
                return (lhs_centered * rhs_centered).sum() / denom

            def _entropy_split_means(
                entropy: torch.Tensor,
                values: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if entropy.numel() < 2 or values.numel() < 2:
                    mean_value = (
                        values.mean() if values.numel() > 0 else values.new_zeros(())
                    )
                    return mean_value, mean_value
                order = torch.argsort(entropy)
                low_count = max(int(order.numel() // 2), 1)
                high_count = max(int(order.numel() - low_count), 1)
                low_idx = order[:low_count]
                high_idx = order[-high_count:]
                return values[low_idx].mean(), values[high_idx].mean()

            infos["listwise_semantic_exploration_prompt_count"] = torch.tensor(
                float(semantic_prompt_entropy.numel()),
                device=scalar_stat_device,
                dtype=torch.float32,
            )
            infos["listwise_semantic_exploration_entropy_std"] = semantic_prompt_entropy.std(
                unbiased=False
            )
            infos["listwise_semantic_exploration_gain_any_correct_std"] = (
                semantic_prompt_gain_any.std(unbiased=False)
            )
            infos["listwise_semantic_exploration_gain_drgrpo_std"] = (
                semantic_prompt_gain_drgrpo.std(unbiased=False)
            )
            infos["listwise_semantic_exploration_gain_any_correct_corr"] = _prompt_corr(
                semantic_prompt_entropy,
                semantic_prompt_gain_any,
            )
            infos["listwise_semantic_exploration_gain_drgrpo_corr"] = _prompt_corr(
                semantic_prompt_entropy,
                semantic_prompt_gain_drgrpo,
            )
            low_any, high_any = _entropy_split_means(
                semantic_prompt_entropy,
                semantic_prompt_gain_any,
            )
            low_drgrpo, high_drgrpo = _entropy_split_means(
                semantic_prompt_entropy,
                semantic_prompt_gain_drgrpo,
            )
            infos["listwise_semantic_exploration_gain_any_correct_low_entropy"] = (
                low_any
            )
            infos["listwise_semantic_exploration_gain_any_correct_high_entropy"] = (
                high_any
            )
            infos["listwise_semantic_exploration_gain_any_correct_high_minus_low"] = (
                high_any - low_any
            )
            infos["listwise_semantic_exploration_gain_drgrpo_low_entropy"] = (
                low_drgrpo
            )
            infos["listwise_semantic_exploration_gain_drgrpo_high_entropy"] = (
                high_drgrpo
            )
            infos["listwise_semantic_exploration_gain_drgrpo_high_minus_low"] = (
                high_drgrpo - low_drgrpo
            )
        if stats["listwise_semantic_exploration_gain_any_correct"]:
            infos["listwise_semantic_exploration_gain_any_correct"] = (
                _stack_scalar_stats(
                    stats["listwise_semantic_exploration_gain_any_correct"],
                    device=scalar_stat_device,
                ).mean()
            )
        if stats["listwise_semantic_exploration_gain_drgrpo"]:
            infos["listwise_semantic_exploration_gain_drgrpo"] = _stack_scalar_stats(
                stats["listwise_semantic_exploration_gain_drgrpo"],
                device=scalar_stat_device,
            ).mean()
        if semantic_prompt_entropy.numel() > 0:
            infos["listwise_semantic_exploration_gain_any_correct"] = (
                semantic_prompt_gain_any.mean()
            )
            infos["listwise_semantic_exploration_gain_drgrpo"] = (
                semantic_prompt_gain_drgrpo.mean()
            )
        if stats["listwise_semantic_cluster_ref_available"]:
            infos["listwise_semantic_cluster_ref_available"] = _stack_scalar_stats(
                stats["listwise_semantic_cluster_ref_available"],
                device=scalar_stat_device,
            ).mean()
        infos["logprobs_diff_max"] = _stack_scalar_stats(
            stats["logprobs_diff_max"],
            device=scalar_stat_device,
        ).max()
        infos["logprobs_diff_min"] = _stack_scalar_stats(
            stats["logprobs_diff_min"],
            device=scalar_stat_device,
        ).min()
        infos["zero_pg_loss_count"] = _stack_scalar_stats(
            stats["zero_pg_loss_count"],
            device=scalar_stat_device,
        ).mean()
        if stats["clip_ratio_low"]:
            infos["clip_ratio_low"] = _stack_scalar_stats(
                stats["clip_ratio_low"],
                device=scalar_stat_device,
            ).mean()
            infos["clip_ratio_high"] = _stack_scalar_stats(
                stats["clip_ratio_high"],
                device=scalar_stat_device,
            ).mean()
            infos["clip_ratio_region"] = _stack_scalar_stats(
                stats["clip_ratio_region"],
                device=scalar_stat_device,
            ).mean()
        if stats["pg_clipfrac"]:
            infos["pg_clipfrac"] = _stack_scalar_stats(
                stats["pg_clipfrac"],
                device=scalar_stat_device,
            ).mean()
        infos["adv_mean"] = advantages.mean().cpu()
        infos["adv_min"] = advantages.min().cpu()
        infos["adv_max"] = advantages.max().cpu()
        infos["all_zero_rewards_count"] = (
            (grouped_reward_values.mean(-1) == 0).sum().cpu()
        )
        infos["all_one_rewards_count"] = (
            (grouped_reward_values.mean(-1) == 1).sum().cpu()
        )
        optional_logging_metrics = {
            "listwise_clip_reward_mass_mean": 0.0,
            "listwise_grad_probe_enabled": 0.0,
            "listwise_grad_probe_update_index": 0.0,
            "listwise_grad_probe_valid": 0.0,
            "listwise_grad_token_norm": 0.0,
            "listwise_grad_sequence_norm": 0.0,
            "listwise_grad_combined_norm": 0.0,
            "listwise_grad_ratio_unscaled": 0.0,
            "listwise_grad_ratio_scaled": 0.0,
            "listwise_grad_cosine": 0.0,
            "listwise_projection_ce_loss_effective": 0.0,
            "listwise_semantic_cluster_entropy": 0.0,
            "listwise_semantic_cluster_entropy_norm": 0.0,
            "listwise_semantic_cluster_count": 0.0,
            "listwise_semantic_competitive_mode_count": 0.0,
            "listwise_semantic_competitive_mode_eligible_count": 0.0,
            "listwise_semantic_competitive_mode_eligible_frac": 0.0,
            "listwise_semantic_distinct_correct_mode_count": 0.0,
            "listwise_semantic_distinct_correct_mode_frac": 0.0,
            "listwise_semantic_competitive_mode_best_score": 0.0,
            "listwise_semantic_competitive_mode_second_score": 0.0,
            "listwise_semantic_competitive_mode_gap": 0.0,
            "listwise_semantic_answer_key_extracted_frac": 0.0,
            "listwise_semantic_trace_extracted_frac": 0.0,
            "listwise_semantic_signature_extracted_frac": 0.0,
            "listwise_semantic_cluster_valid_frac": 0.0,
            "listwise_semantic_trace_truncated_frac": 0.0,
            "listwise_semantic_cluster_entropy_ref": 0.0,
            "listwise_semantic_cluster_entropy_gain": 0.0,
            "listwise_semantic_exploration_gain_any_correct": 0.0,
            "listwise_semantic_exploration_gain_drgrpo": 0.0,
            "listwise_semantic_exploration_gain_any_correct_corr": 0.0,
            "listwise_semantic_exploration_gain_drgrpo_corr": 0.0,
            "listwise_semantic_exploration_prompt_count": 0.0,
            "listwise_semantic_exploration_entropy_std": 0.0,
            "listwise_semantic_exploration_gain_any_correct_std": 0.0,
            "listwise_semantic_exploration_gain_drgrpo_std": 0.0,
            "listwise_semantic_exploration_gain_any_correct_low_entropy": 0.0,
            "listwise_semantic_exploration_gain_any_correct_high_entropy": 0.0,
            "listwise_semantic_exploration_gain_any_correct_high_minus_low": 0.0,
            "listwise_semantic_exploration_gain_drgrpo_low_entropy": 0.0,
            "listwise_semantic_exploration_gain_drgrpo_high_entropy": 0.0,
            "listwise_semantic_exploration_gain_drgrpo_high_minus_low": 0.0,
            "listwise_semantic_cluster_ref_available": 0.0,
            "listwise_semantic_explore_budget_mean": 0.0,
            "listwise_semantic_explore_budget_saturated_frac": 0.0,
            "listwise_semantic_explore_applied_group_frac": 0.0,
            "listwise_semantic_verified_bonus_applied_group_frac": 0.0,
            "listwise_semantic_prompt_selected_frac": 0.0,
            "listwise_semantic_prompt_rejected_low_opp_frac": 0.0,
            "listwise_semantic_prompt_rejected_nonpositive_frac": 0.0,
            "listwise_semantic_prompt_rejected_len_guard_frac": 0.0,
            "listwise_semantic_prompt_rejected_format_guard_frac": 0.0,
            "listwise_semantic_prompt_rejected_verified_bonus_len_guard_frac": 0.0,
            "listwise_semantic_prompt_rejected_verified_bonus_format_guard_frac": 0.0,
            "listwise_semantic_moved_mass_l1": 0.0,
            "listwise_semantic_alpha_raw_mean": 0.0,
            "listwise_semantic_alpha_applied_mean": 0.0,
            "listwise_semantic_verified_bonus_mean": 0.0,
            "listwise_semantic_expected_utility_q": 0.0,
            "listwise_semantic_expected_utility_explore_target": 0.0,
            "listwise_semantic_expected_utility_final_w": 0.0,
            "listwise_semantic_expected_len_q": 0.0,
            "listwise_semantic_expected_len_explore_target": 0.0,
            "listwise_semantic_expected_len_final_w": 0.0,
            "listwise_semantic_expected_format_q": 0.0,
            "listwise_semantic_expected_format_explore_target": 0.0,
            "listwise_semantic_expected_format_final_w": 0.0,
            "listwise_tau_target_metric": 0.0,
            "listwise_tau_metric_ema": 0.0,
            "listwise_tau_adaptation_metric_value": 0.0,
            "listwise_tau_adaptation_metric_is_semantic_entropy_mu": 0.0,
            "listwise_tau_adaptation_metric_is_exploration_gain_any_correct": 0.0,
            "listwise_tau_adaptation_metric_is_exploration_gain_drgrpo": 0.0,
            "listwise_tau_signal_semantic_entropy_mu": 0.0,
            "listwise_tau_signal_exploration_gain_any_correct": 0.0,
            "listwise_tau_signal_exploration_gain_drgrpo": 0.0,
        }
        for key, default in optional_logging_metrics.items():
            infos.setdefault(
                key,
                torch.tensor(
                    default,
                    device=scalar_stat_device,
                    dtype=torch.float32,
                ),
            )
        return {key: infos[key] for key in sorted(infos)}

    def learning_step(self, trajectory):
        if self.objective == "maxent_listwise":
            return self._listwise_learning_step(trajectory)
        if self._use_instrumented_grpo_learning_step():
            return self._grpo_learning_step_with_progress(trajectory)
        return super().learning_step(trajectory)

    def _apply_template(self, example):
        problem = example[self.args.input_key]
        example[self.args.input_key] = TEMPLATE_FACTORY[self.args.prompt_template](
            problem
        )
        return example

    def prepare_data(self, strategy, tokenizer):
        prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        prompts_data = prompt_dataset[self.args.train_split].select(
            range(min(self.args.max_train, len(prompt_dataset[self.args.train_split])))
        )

        # Prepare the data: templated questions & gt final answers.
        prompts_data = prompts_data.map(lambda x: self._apply_template(x))

        self.prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=self.args.input_key,
            output_key=self.args.output_key,
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            self.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )

    def eval_dataloader_collate_fn(self, item_list):
        problems = []
        formatted_problems = []
        answers = []
        for item in item_list:
            problems.append(item["problem"])
            formatted_problems.append(
                TEMPLATE_FACTORY[self.args.prompt_template](item["problem"])
            )
            answers.append(item["answer"])
        return formatted_problems, problems, answers

    def evaluate(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        for benchmark_name, dataset in self.eval_dataset_dict.items():
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            metrics = super().evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            all_metrics.update(
                {
                    k.replace("eval/", f"eval/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval/average/accuracy": np.mean(accuracies),
                "eval/average/score": np.mean(scores),
                "eval/average/response_tok_len": np.mean(lens),
            }
        )
        return all_metrics


def run_zero_math_rl(args: ZeroMathArgs):
    # Define a distributed program that composes Actors and Learners.
    program, local_resources = get_program(
        args, learner_cls=ZeroMathLearner, actor_cls=ZeroMathActor
    )
    # Launch the program in a local, multi-processing way!
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args: ZeroMathArgs = get_default_args(ZeroMathArgs)
    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.

    args = default_args_validation(args)
    args = validate_zero_math_args(args)
    run_zero_math_rl(args)
