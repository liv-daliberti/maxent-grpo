"""Argument schema and validation for zero-math Dr.GRPO/Dr.X runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from oat.algorithms.ppo import PPOArgs

from .listwise import (
    normalize_maxent_clip_mode,
    normalize_tiebreak_anchor,
    normalize_oat_objective,
    normalize_semantic_cluster_method,
    normalize_semantic_remix_mode,
)


@dataclass
class ZeroMathArgs(PPOArgs):
    # Template.
    prompt_template: Literal["qwen_boxed", "qwen_math", "no", "r1"] = field(
        default="qwen_math"
    )
    # Evaluation benchmarks used.
    test_split: str = "all"
    # Verifier.
    verifier_version: Literal["fast", "math_verify"] = field(default="fast")
    # Objective routing. The default keeps upstream OAT DR.GRPO unchanged.
    objective: Literal[
        "grpo",
        "maxent_listwise",
    ] = field(default="grpo")
    semantic_entropy_lambda: float = 0.05
    policy_entropy_coef: float = 0.0
    # xDr.GRPO candidate-level tempered aggregation on the grpo objective.
    # inf disables the reweighting and reproduces Dr.GRPO bit-for-bit.
    xdr_tau: float = math.inf
    # SEED-Dr.GRPO per-prompt semantic-entropy scaling on the grpo objective.
    # 0 disables the scaling and reproduces Dr.GRPO exactly.
    seed_entropy_alpha: float = 0.0
    maxent_tau: float = 0.3
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
    maxent_clip_mode: Literal["sequence", "token", "none"] = field(default="sequence")
    maxent_token_clip_primary: bool = False
    maxent_drgrpo_token_primary: bool = False
    maxent_drgrpo_token_advantage_source: Literal[
        "weighted", "utility_centered", "maxent_centered"
    ] = field(default="weighted")
    maxent_drgrpo_token_length_normalizer: Literal["max_length", "response_length"] = (
        field(default="max_length")
    )
    maxent_sequence_aux_coef: float = 1.0
    maxent_sequence_aux_group_filter: Literal["all", "mixed", "has_correct"] = field(
        default="all"
    )
    maxent_sequence_aux_max_expected_len_drop: float = math.inf
    maxent_sequence_aux_max_expected_len_gain: float = math.inf
    maxent_sequence_aux_max_expected_format_drop: float = 1.0
    maxent_sequence_aux_min_expected_correctness_delta: float = -1.0
    maxent_neutral_projection_coef: float = 0.0
    maxent_semantic_cluster_method: Literal[
        "default", "answer_family", "greedy", "connected_components", "spectral"
    ] = field(default="default")
    maxent_semantic_similarity_threshold: float = 0.75
    maxent_semantic_embedding_similarity_threshold: float = 0.9
    maxent_semantic_embedding_max_tokens: int = 256
    maxent_semantic_cluster_max_tokens: int = 0
    maxent_semantic_spectral_max_clusters: int = 0
    maxent_semantic_spectral_eigengap_min: float = 0.05
    maxent_semantic_correctness_target_frac: float = 0.5
    maxent_semantic_correctness_sharpness: float = 4.0
    maxent_semantic_correctness_answer_level: bool = False
    maxent_semantic_correctness_min_answer_count: int = 1
    maxent_semantic_remix_mode: Literal[
        "competitive", "correctness_conditioned", "anchor_rare"
    ] = field(default="competitive")
    maxent_reward_shaping_alpha: float = 0.0
    maxent_tiebreak_anchor: Literal["hybrid", "behavior", "reference"] = field(
        default="hybrid"
    )
    maxent_tiebreak_clip_max: float = 1.0
    maxent_competitive_mode_tau: float = 0.05
    maxent_competitive_mode_gap: float = 0.10
    maxent_competitive_mode_top_k: int = 3
    maxent_competitive_mode_budget_max: float = 0.10
    maxent_competitive_mode_budget_scale: float = 0.05
    maxent_competitive_mode_intra_tau: float = 0.01
    maxent_prompt_select_min_alpha_frac: float = 0.5
    maxent_competitive_mode_positive_only: bool = True
    maxent_correctness_schedule_enabled: bool = True
    maxent_correctness_schedule_ema_decay: float = 0.997
    maxent_correctness_schedule_low: float = 0.45
    maxent_correctness_schedule_high: float = 0.90
    maxent_correctness_schedule_budget_max_early: float = 0.18
    maxent_correctness_schedule_budget_max_late: float = 0.06
    maxent_correctness_schedule_prompt_select_min_alpha_frac_early: float = 0.20
    maxent_correctness_schedule_prompt_select_min_alpha_frac_late: float = 0.50
    maxent_correctness_schedule_mode_tau_early: float = 0.08
    maxent_correctness_schedule_mode_tau_late: float = 0.03
    maxent_correctness_schedule_intra_tau_early: float = 0.03
    maxent_correctness_schedule_intra_tau_late: float = 0.005
    maxent_semantic_guard_max_expected_len_delta: float = 24.0
    maxent_semantic_guard_max_expected_format_drop: float = 0.0
    maxent_branch_grad_diagnostics: bool = False
    maxent_branch_grad_diagnostics_interval: int = 1
    maxent_branch_grad_diagnostics_max_steps: int = 0
    maxent_logprob_chunk_size: int = 2
    maxent_backward_chunk_size: int = 4
    maxent_backward_token_budget: int = 4096
    eval_mode_coverage_k: int = 0
    eval_mode_coverage_temperature: float = 1.0
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


def build_fixed_listwise_config(args: ZeroMathArgs) -> dict[str, object]:
    """Snapshot immutable listwise settings for a learner run."""

    return {
        "maxent_q_temperature": float(args.maxent_q_temperature),
        "maxent_q_epsilon": float(args.maxent_q_epsilon),
        "maxent_candidate_kl_coef": float(args.maxent_candidate_kl_coef),
        "maxent_exact_drx_weight_source": str(args.maxent_exact_drx_weight_source),
        "maxent_length_normalize_ref": bool(args.maxent_length_normalize_ref),
        "maxent_length_normalize_policy": bool(args.maxent_length_normalize_policy),
        "maxent_listwise_skip_zero_variance_groups": bool(
            args.maxent_listwise_skip_zero_variance_groups
        ),
        "maxent_use_clip_objective": bool(args.maxent_use_clip_objective),
        "maxent_clip_objective_coef": float(args.maxent_clip_objective_coef),
        "maxent_clip_range": (
            None if args.maxent_clip_range is None else float(args.maxent_clip_range)
        ),
        "maxent_clip_adv_baseline": (
            None
            if args.maxent_clip_adv_baseline is None
            else float(args.maxent_clip_adv_baseline)
        ),
        "maxent_clip_preserve_reward_mass": bool(args.maxent_clip_preserve_reward_mass),
        "maxent_clip_mode": str(args.maxent_clip_mode),
        "maxent_token_clip_primary": bool(args.maxent_token_clip_primary),
        "maxent_drgrpo_token_primary": bool(args.maxent_drgrpo_token_primary),
        "maxent_drgrpo_token_advantage_source": str(
            args.maxent_drgrpo_token_advantage_source
        ),
        "maxent_drgrpo_token_length_normalizer": str(
            args.maxent_drgrpo_token_length_normalizer
        ),
        "maxent_sequence_aux_coef": float(args.maxent_sequence_aux_coef),
        "maxent_sequence_aux_group_filter": str(
            args.maxent_sequence_aux_group_filter
        ),
        "maxent_sequence_aux_max_expected_len_drop": float(
            args.maxent_sequence_aux_max_expected_len_drop
        ),
        "maxent_sequence_aux_max_expected_len_gain": float(
            args.maxent_sequence_aux_max_expected_len_gain
        ),
        "maxent_sequence_aux_max_expected_format_drop": float(
            args.maxent_sequence_aux_max_expected_format_drop
        ),
        "maxent_sequence_aux_min_expected_correctness_delta": float(
            args.maxent_sequence_aux_min_expected_correctness_delta
        ),
        "maxent_neutral_projection_coef": float(args.maxent_neutral_projection_coef),
        "maxent_semantic_cluster_method": str(args.maxent_semantic_cluster_method),
        "maxent_semantic_similarity_threshold": float(
            args.maxent_semantic_similarity_threshold
        ),
        "maxent_semantic_embedding_similarity_threshold": float(
            args.maxent_semantic_embedding_similarity_threshold
        ),
        "maxent_semantic_embedding_max_tokens": int(
            args.maxent_semantic_embedding_max_tokens
        ),
        "maxent_semantic_cluster_max_tokens": int(
            args.maxent_semantic_cluster_max_tokens
        ),
        "maxent_semantic_spectral_max_clusters": int(
            args.maxent_semantic_spectral_max_clusters
        ),
        "maxent_semantic_spectral_eigengap_min": float(
            args.maxent_semantic_spectral_eigengap_min
        ),
        "maxent_semantic_correctness_target_frac": float(
            args.maxent_semantic_correctness_target_frac
        ),
        "maxent_semantic_correctness_sharpness": float(
            args.maxent_semantic_correctness_sharpness
        ),
        "maxent_semantic_correctness_answer_level": bool(
            args.maxent_semantic_correctness_answer_level
        ),
        "maxent_semantic_correctness_min_answer_count": int(
            args.maxent_semantic_correctness_min_answer_count
        ),
        "maxent_semantic_remix_mode": str(args.maxent_semantic_remix_mode),
        "maxent_reward_shaping_alpha": float(args.maxent_reward_shaping_alpha),
        "maxent_tiebreak_anchor": str(args.maxent_tiebreak_anchor),
        "maxent_tiebreak_clip_max": float(args.maxent_tiebreak_clip_max),
        "maxent_correctness_schedule_enabled": bool(
            args.maxent_correctness_schedule_enabled
        ),
        "maxent_correctness_schedule_ema_decay": float(
            args.maxent_correctness_schedule_ema_decay
        ),
        "maxent_correctness_schedule_low": float(args.maxent_correctness_schedule_low),
        "maxent_correctness_schedule_high": float(
            args.maxent_correctness_schedule_high
        ),
        "maxent_correctness_schedule_budget_max_early": float(
            args.maxent_correctness_schedule_budget_max_early
        ),
        "maxent_correctness_schedule_budget_max_late": float(
            args.maxent_correctness_schedule_budget_max_late
        ),
        "maxent_correctness_schedule_prompt_select_min_alpha_frac_early": float(
            args.maxent_correctness_schedule_prompt_select_min_alpha_frac_early
        ),
        "maxent_correctness_schedule_prompt_select_min_alpha_frac_late": float(
            args.maxent_correctness_schedule_prompt_select_min_alpha_frac_late
        ),
        "maxent_correctness_schedule_mode_tau_early": float(
            args.maxent_correctness_schedule_mode_tau_early
        ),
        "maxent_correctness_schedule_mode_tau_late": float(
            args.maxent_correctness_schedule_mode_tau_late
        ),
        "maxent_correctness_schedule_intra_tau_early": float(
            args.maxent_correctness_schedule_intra_tau_early
        ),
        "maxent_correctness_schedule_intra_tau_late": float(
            args.maxent_correctness_schedule_intra_tau_late
        ),
        "maxent_branch_grad_diagnostics": bool(args.maxent_branch_grad_diagnostics),
        "maxent_branch_grad_diagnostics_interval": int(
            args.maxent_branch_grad_diagnostics_interval
        ),
        "maxent_branch_grad_diagnostics_max_steps": int(
            args.maxent_branch_grad_diagnostics_max_steps
        ),
        "maxent_logprob_chunk_size": int(args.maxent_logprob_chunk_size),
        "maxent_backward_chunk_size": int(args.maxent_backward_chunk_size),
        "maxent_backward_token_budget": int(args.maxent_backward_token_budget),
        "maxent_reference_logprobs_source": str(args.maxent_reference_logprobs_source),
        "maxent_tau_adaptation_metric": str(args.maxent_tau_adaptation_metric),
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
        "maxent_tau_target_metric_horizon": int(args.maxent_tau_target_metric_horizon),
    }


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
    args.maxent_clip_mode = normalize_maxent_clip_mode(
        getattr(args, "maxent_clip_mode", "sequence")
    )
    args.maxent_tiebreak_anchor = normalize_tiebreak_anchor(
        getattr(args, "maxent_tiebreak_anchor", "hybrid")
    )
    args.maxent_semantic_cluster_method = normalize_semantic_cluster_method(
        getattr(args, "maxent_semantic_cluster_method", "default")
    )
    args.maxent_semantic_remix_mode = normalize_semantic_remix_mode(
        getattr(args, "maxent_semantic_remix_mode", "competitive")
    )
    if args.maxent_reference_logprobs_source not in {"model", "behavior"}:
        raise ValueError(
            "maxent_reference_logprobs_source must be one of: model, behavior"
        )
    if not math.isfinite(float(args.maxent_semantic_similarity_threshold)):
        raise ValueError("maxent_semantic_similarity_threshold must be finite")
    if not math.isfinite(float(args.maxent_semantic_embedding_similarity_threshold)):
        raise ValueError(
            "maxent_semantic_embedding_similarity_threshold must be finite"
        )
    if int(args.maxent_semantic_embedding_max_tokens) <= 0:
        raise ValueError("maxent_semantic_embedding_max_tokens must be positive")
    if int(args.maxent_semantic_cluster_max_tokens) < 0:
        raise ValueError("maxent_semantic_cluster_max_tokens must be non-negative")
    if int(args.maxent_semantic_spectral_max_clusters) < 0:
        raise ValueError("maxent_semantic_spectral_max_clusters must be non-negative")
    if args.maxent_semantic_spectral_eigengap_min < 0:
        raise ValueError("maxent_semantic_spectral_eigengap_min must be non-negative")
    if not math.isfinite(
        float(args.maxent_semantic_correctness_target_frac)
    ) or not 0.0 <= float(args.maxent_semantic_correctness_target_frac) <= 1.0:
        raise ValueError(
            "maxent_semantic_correctness_target_frac must be between 0 and 1"
        )
    if not math.isfinite(
        float(args.maxent_semantic_correctness_sharpness)
    ) or args.maxent_semantic_correctness_sharpness < 0:
        raise ValueError(
            "maxent_semantic_correctness_sharpness must be finite and non-negative"
        )
    if int(args.maxent_semantic_correctness_min_answer_count) < 1:
        raise ValueError(
            "maxent_semantic_correctness_min_answer_count must be positive"
        )
    if args.maxent_reward_shaping_alpha < 0:
        raise ValueError("maxent_reward_shaping_alpha must be non-negative")
    if args.semantic_entropy_lambda < 0:
        raise ValueError("semantic_entropy_lambda must be non-negative")
    if args.policy_entropy_coef < 0:
        raise ValueError("policy_entropy_coef must be non-negative")
    if math.isnan(args.xdr_tau) or args.xdr_tau <= 0:
        raise ValueError("xdr_tau must be positive (use inf to disable)")
    if math.isfinite(args.xdr_tau):
        if args.objective != "grpo":
            raise ValueError("finite xdr_tau requires objective=grpo")
        if args.critic_type != "drgrpo":
            raise ValueError("finite xdr_tau requires critic_type=drgrpo")
        if getattr(args, "reinforce_update", False):
            raise ValueError(
                "finite xdr_tau is incompatible with reinforce_update: the "
                "xdr utilities assume the clipped Dr.GRPO surrogate"
            )
    if math.isnan(args.seed_entropy_alpha) or args.seed_entropy_alpha < 0:
        raise ValueError("seed_entropy_alpha must be non-negative")
    if args.seed_entropy_alpha > 0:
        if args.objective != "grpo":
            raise ValueError("seed_entropy_alpha requires objective=grpo")
        if args.critic_type != "drgrpo":
            raise ValueError("seed_entropy_alpha requires critic_type=drgrpo")
        if math.isfinite(args.xdr_tau):
            raise ValueError(
                "seed_entropy_alpha and finite xdr_tau are separate "
                "comparative arms; enable at most one"
            )
    if args.maxent_tiebreak_clip_max < 0:
        raise ValueError("maxent_tiebreak_clip_max must be non-negative")
    if args.baseline_zero_adv_response_tokens < 0:
        raise ValueError("baseline_zero_adv_response_tokens must be non-negative")
    if args.maxent_reward_shaping_alpha > 0 and args.objective != "maxent_listwise":
        raise ValueError(
            "maxent_reward_shaping_alpha currently requires objective=maxent_listwise"
        )
    if args.maxent_reward_shaping_alpha > 0 and not args.maxent_drgrpo_token_primary:
        raise ValueError(
            "maxent_reward_shaping_alpha currently requires maxent_drgrpo_token_primary=1"
        )
    if args.maxent_clip_objective_coef < 0:
        raise ValueError("maxent_clip_objective_coef must be non-negative")
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
        if bool(args.maxent_token_clip_primary):
            raise ValueError(
                "maxent_token_clip_primary requires objective=maxent_listwise"
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
    row_sharded_exact_drx = bool(args.maxent_drgrpo_token_primary) and 0 < int(
        args.train_batch_size_per_device
    ) < int(args.num_samples)
    if (
        args.train_batch_size_per_device % args.num_samples != 0
        and not row_sharded_exact_drx
    ):
        raise ValueError(
            "Listwise MaxEnt requires train_batch_size_per_device to be divisible "
            "by num_samples so each microbatch preserves whole prompt groups, "
            "unless the narrow row-sharded exact DrX path is enabled with "
            "train_batch_size_per_device < num_samples."
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
    if args.maxent_sequence_aux_group_filter not in {
        "all",
        "mixed",
        "has_correct",
    }:
        raise ValueError(
            "maxent_sequence_aux_group_filter must be one of: "
            "all, mixed, has_correct"
        )
    if args.maxent_sequence_aux_max_expected_len_drop < 0:
        raise ValueError(
            "maxent_sequence_aux_max_expected_len_drop must be non-negative"
        )
    if args.maxent_sequence_aux_max_expected_len_gain < 0:
        raise ValueError(
            "maxent_sequence_aux_max_expected_len_gain must be non-negative"
        )
    if args.maxent_sequence_aux_max_expected_format_drop < 0:
        raise ValueError(
            "maxent_sequence_aux_max_expected_format_drop must be non-negative"
        )
    if (
        not -1.0
        <= float(args.maxent_sequence_aux_min_expected_correctness_delta)
        <= 1.0
    ):
        raise ValueError(
            "maxent_sequence_aux_min_expected_correctness_delta must be in [-1, 1]"
        )
    if args.maxent_drgrpo_token_advantage_source not in {
        "weighted",
        "utility_centered",
        "maxent_centered",
    }:
        raise ValueError(
            "maxent_drgrpo_token_advantage_source must be one of: "
            "weighted, utility_centered, maxent_centered"
        )
    if args.maxent_drgrpo_token_length_normalizer not in {
        "max_length",
        "response_length",
    }:
        raise ValueError(
            "maxent_drgrpo_token_length_normalizer must be one of: "
            "max_length, response_length"
        )
    if args.maxent_candidate_kl_coef < 0:
        raise ValueError("maxent_candidate_kl_coef must be non-negative")
    if args.maxent_neutral_projection_coef < 0:
        raise ValueError("maxent_neutral_projection_coef must be non-negative")
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
    if not 0.0 <= float(args.maxent_correctness_schedule_ema_decay) <= 1.0:
        raise ValueError("maxent_correctness_schedule_ema_decay must be in [0, 1]")
    if not 0.0 <= float(args.maxent_correctness_schedule_low) <= 1.0:
        raise ValueError("maxent_correctness_schedule_low must be in [0, 1]")
    if not 0.0 <= float(args.maxent_correctness_schedule_high) <= 1.0:
        raise ValueError("maxent_correctness_schedule_high must be in [0, 1]")
    if float(args.maxent_correctness_schedule_high) <= float(
        args.maxent_correctness_schedule_low
    ):
        raise ValueError(
            "maxent_correctness_schedule_high must be greater than "
            "maxent_correctness_schedule_low"
        )
    if args.maxent_correctness_schedule_budget_max_early < 0:
        raise ValueError(
            "maxent_correctness_schedule_budget_max_early must be non-negative"
        )
    if args.maxent_correctness_schedule_budget_max_late < 0:
        raise ValueError(
            "maxent_correctness_schedule_budget_max_late must be non-negative"
        )
    if (
        not 0.0
        <= float(args.maxent_correctness_schedule_prompt_select_min_alpha_frac_early)
        <= 1.0
    ):
        raise ValueError(
            "maxent_correctness_schedule_prompt_select_min_alpha_frac_early must be in [0, 1]"
        )
    if (
        not 0.0
        <= float(args.maxent_correctness_schedule_prompt_select_min_alpha_frac_late)
        <= 1.0
    ):
        raise ValueError(
            "maxent_correctness_schedule_prompt_select_min_alpha_frac_late must be in [0, 1]"
        )
    if args.maxent_correctness_schedule_mode_tau_early <= 0:
        raise ValueError("maxent_correctness_schedule_mode_tau_early must be positive")
    if args.maxent_correctness_schedule_mode_tau_late <= 0:
        raise ValueError("maxent_correctness_schedule_mode_tau_late must be positive")
    if args.maxent_correctness_schedule_intra_tau_early <= 0:
        raise ValueError("maxent_correctness_schedule_intra_tau_early must be positive")
    if args.maxent_correctness_schedule_intra_tau_late <= 0:
        raise ValueError("maxent_correctness_schedule_intra_tau_late must be positive")
    if args.maxent_semantic_guard_max_expected_len_delta < 0:
        raise ValueError(
            "maxent_semantic_guard_max_expected_len_delta must be non-negative"
        )
    if args.maxent_semantic_guard_max_expected_format_drop < 0:
        raise ValueError(
            "maxent_semantic_guard_max_expected_format_drop must be non-negative"
        )
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
        raise ValueError(
            "maxent_branch_grad_diagnostics_max_steps must be non-negative"
        )
    if bool(args.maxent_token_clip_primary):
        if not bool(args.maxent_use_clip_objective):
            raise ValueError(
                "maxent_token_clip_primary requires maxent_use_clip_objective"
            )
        if args.maxent_clip_mode != "token":
            raise ValueError(
                "maxent_token_clip_primary requires maxent_clip_mode=token"
            )
        if args.maxent_clip_objective_coef <= 0:
            raise ValueError(
                "maxent_token_clip_primary requires maxent_clip_objective_coef > 0"
            )
    if bool(args.maxent_drgrpo_token_primary):
        if bool(args.maxent_token_clip_primary):
            raise ValueError(
                "maxent_drgrpo_token_primary cannot be combined with "
                "maxent_token_clip_primary"
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
        raise ValueError(
            "maxent_tau_max must be >= maxent_tau_min when both are positive"
        )
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
    using_legacy_tau_target = any(
        value is not None for value in legacy_tau_target_fields
    )
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
