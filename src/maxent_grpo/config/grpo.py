"""
GRPO-specific configuration dataclasses with MaxEnt extensions.

These classes layer additional benchmarking, telemetry, weighting, and vLLM
generation controls on top of TRL's GRPO configuration so training recipes can
be expressed declaratively. The TRL dependency is optional during imports to
keep documentation builds and tests lightweight.

License
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

from __future__ import annotations

import json
import logging
from dataclasses import MISSING, dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .dataset import ScriptArguments, trl

LOG = logging.getLogger(__name__)

_BASE_CLIP_RANGE_DEFAULT = 0.0
_base_fields = getattr(trl.GRPOConfig, "__dataclass_fields__", None)
if isinstance(_base_fields, dict):
    _base_clip = _base_fields.get("clip_range")
    if _base_clip is not None:
        _base_default = getattr(_base_clip, "default", MISSING)
        if _base_default is not MISSING:
            _BASE_CLIP_RANGE_DEFAULT = _base_default


def _parse_log_level(value: int | str | None) -> Optional[int]:
    """Resolve a logging level specified as a name or numeric value."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.isdigit():
            try:
                return int(candidate)
            except ValueError:
                return None
        normalized = candidate.upper()
        level = getattr(logging, normalized, None)
        if isinstance(level, int):
            return level
    return None


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """GRPO configuration extended for MaxEnt-GRPO experiments.

    Adds logging hooks, benchmark orchestration, weighting controls, and vLLM
    integration options on top of :class:`trl.GRPOConfig`.

    :ivar benchmarks: Benchmarks to run after training completes.
    :ivar callbacks: Callback identifiers executed during training.
    :ivar chat_template: Optional chat template string used to render prompts.
    :ivar hub_model_revision: Hub model branch to push artifacts to.
    :ivar num_completions_to_print: Number of completions to print for inspection.
    :ivar overwrite_hub_revision: Whether to overwrite the destination Hub branch.
    :ivar push_to_hub_revision: Whether to push training outputs to the Hub.
    :ivar system_prompt: System prompt injected before training prompts.
    :ivar wandb_log_unique_prompts: Log unique prompts to W&B runs.
    :ivar wandb_entity: W&B entity/organization for run tracking.
    :ivar wandb_project: W&B project name.
    :ivar wandb_run_group: W&B group used to cluster related runs.
    :ivar maxent_alpha: Coefficient for the extra MaxEnt objective term.
        Set to 0 to force native GRPO loss/pathway parity.
    :ivar maxent_tau: Sequence-level entropy weight used by MaxEnt-GRPO.
    :ivar maxent_q_temperature: Temperature applied when forming listwise q values.
    :ivar maxent_q_epsilon: Minimum support added to q before normalization.
    :ivar maxent_length_normalize_ref: Length-normalize reference log-probs.
    :ivar maxent_logprob_chunk_size: Mini-batch size when computing log-probs.
    :ivar maxent_policy_entropy: Whether to compute policy entropy during scoring.
    :ivar maxent_policy_entropy_mode: Which entropy estimator to use ("exact" or "sample").
    :ivar policy_entropy_bonus_coef: Coefficient applied to per-token policy entropy
        when adding an entropy bonus to rewards (GRPO + entropy bonus).
    :ivar behavior_logprobs_source: Source for behavior-policy log-probs used in PPO ratios.
    :ivar maxent_target_weight_entropy: Target weight entropy for automatic tau tuning.
    :ivar maxent_target_weight_entropy_start: Optional starting entropy target for annealing.
    :ivar maxent_target_weight_entropy_final: Optional final entropy target for annealing.
    :ivar maxent_target_weight_entropy_horizon: Steps to interpolate between start/final targets.
    :ivar maxent_tau_lr: Learning rate applied during tau adaptation.
    :ivar maxent_tau_min: Lower bound enforced on tau during tuning.
    :ivar maxent_tau_max: Upper bound enforced on tau during tuning.
    :ivar maxent_tau_warmup_steps: Warmup steps before enabling tau adaptation.
    :ivar maxent_use_clip_objective: Blend a PPO-style clipped objective into the loss.
    :ivar maxent_clip_objective_coef: Scale for the clipped objective component.
    :ivar maxent_clip_adv_baseline: Baseline subtracted before clipping.
    :ivar train_grpo_objective: Disable MaxEnt weighting and run the standard GRPO objective.
    :ivar scale_rewards: Whether to scale GRPO advantages by group std (TRL default).
    :ivar maxent_clip_range: Override PPO clip range for the MaxEnt objective.
    :ivar kl_target: Target KL value for automatic beta adjustment.
    :ivar kl_horizon: Horizon in optimizer steps for the beta controller.
    :ivar kl_ctl_step_size: Maximum fractional beta change per controller step.
    :ivar clip_range: PPO clip range used for clipping ratios in training loss.
    :ivar clip_range_high: Upper PPO clip range (epsilon_high) for asymmetric clipping.
    :ivar clip_delta: Optional additional slack for two-sided clipping.
    :ivar grpo_loss_type: GRPO loss aggregation ("grpo", "bnpo", or "dr_grpo").
    :ivar gen_temperature: Temperature used for candidate generation.
    :ivar gen_top_p: Top-p nucleus sampling used for generation.
    :ivar vllm_mode: vLLM backend mode ("server" or "colocate").
    :ivar vllm_url: Base URL for the vLLM ``/generate`` endpoint.
    :ivar vllm_max_completion_rounds: Maximum number of retries to top off completions.
    :ivar vllm_retry_sleep: Seconds to sleep between vLLM retries.
    :ivar vllm_backfill_with_model: Fallback to local ``model.generate`` when vLLM misses completions.
    :ivar vllm_return_logprobs: Request per-token logprobs from vLLM.
    :ivar vllm_logprob_fail_after: Consecutive steps with missing vLLM logprobs before aborting (0 disables).
    :ivar vllm_logprob_fallback: When true, switch reference logprobs to the model after missing vLLM logprobs.
    :ivar vllm_client_tag_fail_fast: When true, abort vLLM retries immediately on client_tag mismatch.
    :ivar vllm_sync_interval_steps: Only sync weights every N optimizer steps when using vLLM sync.
    :ivar vllm_best_of: vLLM ``best_of`` parameter forwarded from TRL.
    :ivar vllm_frequency_penalty: Frequency penalty applied during sampling.
    :ivar vllm_presence_penalty: Presence penalty applied during sampling.
    :ivar vllm_top_k: Top-k sampling parameter forwarded to vLLM.
    :ivar vllm_stop_sequences: Stop sequences for vLLM (JSON list or ``'||'``-delimited string).
    :ivar eval_before_train: Run evaluation once before training begins (step 0).
    :ivar disable_distributed_sampler: Disable the DistributedSampler to avoid double sharding.
    :ivar dataloader_num_workers: Number of worker processes for the training dataloader.
    :ivar dataloader_pin_memory: Whether to pin memory in the training dataloader.
    :ivar dataloader_prefetch_factor: Prefetch factor per worker (only when num_workers > 0).
    :ivar dataloader_persistent_workers: Keep DataLoader workers alive between epochs.
    :raises ValueError: If validation detects negative or inconsistent hyperparameters.
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    eval_before_train: bool = field(
        default=False,
        metadata={"help": "Run evaluation once before training starts (step 0)."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["pure_accuracy_math"],
        metadata={
            "help": (
                "Reward functions to apply during training. Kept for compatibility "
                "with earlier CLIs; defaults to pure_accuracy_math."
            )
        },
    )
    reward_weights: list[float] = field(
        default_factory=lambda: [],
        metadata={
            "help": "Optional weights aligned with reward_funcs (defaults to 1.0 each)."
        },
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    reference_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional frozen reference model to score completions against. "
                "Defaults to model_name_or_path when unset."
            )
        },
    )
    reference_model_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional revision/branch for the frozen reference model."
        },
    )
    maxent_share_reference_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When true, reuse the trainable policy weights for reference scoring "
                "instead of loading a frozen copy."
            )
        },
    )
    num_completions_to_print: int = field(
        default=0,
        metadata={"help": "Number of completions to print."},
    )
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the Hub revision."},
    )
    push_to_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to push to a Hub revision/branch."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to log the unique prompts to wandb. This will create a "
                "new run for each unique prompt."
            )
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    log_like_grpo: bool = field(
        default=False,
        metadata={
            "help": (
                "Emit GRPOTrainer-style single-rank logs instead of aggregated metrics."
            )
        },
    )
    torch_compile: bool = field(
        default=False,
        metadata={
            "help": (
                "When true, wrap the loaded model with torch.compile for faster training "
                "(if supported by the installed torch version)."
            )
        },
    )
    init_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional checkpoint path to initialize weights/state from when resuming "
                "custom MaxEnt/InfoSeed loops."
            )
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional checkpoint path to resume training from (mirrors TRL/Trainer)."
            )
        },
    )
    maxent_alpha: float = field(
        default=1.0,
        metadata={
            "help": (
                "Coefficient for the extra MaxEnt objective term. "
                "Set to 0.0 to use the native GRPO loss pathway exactly."
            )
        },
    )
    maxent_tau: float = field(
        default=0.0,
        metadata={"help": "Sequence-level entropy weight τ for MaxEnt-GRPO."},
    )
    maxent_q_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature applied when turning utilities into listwise q via softmax."
        },
    )
    maxent_q_epsilon: float = field(
        default=1e-6,
        metadata={
            "help": "Epsilon floor added to q for full support before normalization.",
        },
    )
    maxent_length_normalize_ref: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to length-normalize reference log-probs when computing weights."
            )
        },
    )
    maxent_logprob_chunk_size: int = field(
        default=0,
        metadata={
            "help": (
                "If >0, score reference/policy log-probs in mini-batches of this size "
                "to reduce activation memory."
            )
        },
    )
    maxent_reference_logprobs_source: str = field(
        default="auto",
        metadata={
            "help": (
                "How to obtain reference log-prob statistics used for KL. "
                "'auto' uses vLLM metadata when valid, otherwise falls back to policy logprobs "
                "(KL ~= 0) and only runs the frozen reference model when needed; "
                "'model' always scores with the frozen reference model; "
                "'policy' always uses policy logprobs (no reference model); "
                "'none' is an alias for 'policy'."
            )
        },
    )
    maxent_trl_reference_scoring: bool = field(
        default=True,
        metadata={
            "help": (
                "When enabled, use TRL/open-r1-style reference scoring (always run the frozen "
                "reference model, logits_to_keep + selective_log_softmax, temperature scaling) "
                "and disable vLLM/policy fallbacks."
            )
        },
    )
    behavior_logprobs_source: str = field(
        default="model",
        metadata={
            "help": (
                "Source for behavior-policy log-probs used in PPO ratios: "
                "'model' uses HF forward-pass log-probs (TRL-style); "
                "'vllm' uses vLLM metadata when available."
            )
        },
    )
    maxent_allow_stale_reference_logprobs: bool = field(
        default=False,
        metadata={
            "help": (
                "When true, reuse the last cached reference log-prob statistics if reference scoring fails "
                "for a batch. This avoids skipping steps but can corrupt KL/weighting when the cached stats "
                "do not correspond to the current completions."
            )
        },
    )
    maxent_score_tail_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, only the final N tokens of each prompt+completion pair are "
                "used when computing reference/policy log-probs."
            )
        },
    )
    maxent_policy_entropy: bool = field(
        default=False,
        metadata={
            "help": (
                "Compute policy token entropy from logits during scoring so MaxEnt "
                "runs can log (and optionally use) policy entropy diagnostics."
            )
        },
    )
    maxent_policy_entropy_mode: str = field(
        default="exact",
        metadata={
            "help": (
                "Entropy estimator to use when policy entropy is requested. "
                "'exact' computes full-distribution entropy (slow, uses log_softmax); "
                "'sample' uses the negative log-prob of sampled tokens as an unbiased "
                "entropy estimator (fast, matches generation entropy on average)."
            )
        },
    )
    policy_entropy_bonus_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient applied to per-token policy entropy when adding an entropy "
                "bonus to rewards (GRPO + entropy bonus). The entropy bonus is z-scored "
                "within each prompt group and scaled by the batch reward std. Set to 0 to disable."
            )
        },
    )
    maxent_score_slice_prefetch: int = field(
        default=0,
        metadata={
            "help": (
                "Number of scoring slices to prefetch in a background thread while "
                "the model processes the current slice."
            )
        },
    )
    maxent_prompt_cache_size: int = field(
        default=10000,
        metadata={
            "help": (
                "Remember up to this many prompt tokenizations (LRU) so repeated prompts "
                "skip re-tokenization when building scoring batches. Defaults to an "
                "auto-sized value (min 10k, max 50k); set to 0 to disable."
            )
        },
    )
    maxent_target_weight_entropy: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If set, automatically tune the MaxEnt entropy weight τ to keep the "
                "average per-prompt weight entropy near this target."
            )
        },
    )
    maxent_target_weight_entropy_start: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional starting entropy target to anneal toward "
                "maxent_target_weight_entropy_final."
            )
        },
    )
    maxent_target_weight_entropy_final: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional final entropy target used alongside "
                "maxent_target_weight_entropy_start to enable annealing. Defaults to "
                "maxent_target_weight_entropy when unset."
            )
        },
    )
    maxent_target_weight_entropy_horizon: int = field(
        default=0,
        metadata={
            "help": (
                "Number of optimizer steps to linearly anneal the entropy target from "
                "start to final. When 0 or negative, annealing is disabled."
            )
        },
    )
    maxent_tau_lr: float = field(
        default=0.0,
        metadata={
            "help": "Learning rate used for automatic τ tuning (log-space).",
        },
    )
    maxent_tau_min: float = field(
        default=0.0,
        metadata={
            "help": "Lower bound for the learned sequence-level entropy weight τ."
        },
    )
    maxent_tau_max: float = field(
        default=0.0,
        metadata={
            "help": "Upper bound for the learned sequence-level entropy weight τ."
        },
    )
    maxent_tau_warmup_steps: int = field(
        default=-1,
        metadata={
            "help": (
                "Number of steps to keep τ on a warmup schedule before enabling automatic tuning. "
                "Set to a non-negative value to override, or -1 to match the learning-rate warmup."
            )
        },
    )
    controller_meta_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable the tau/beta meta-controller. When true, τ/β are updated "
                "according to controller_meta_method."
            )
        },
    )
    controller_meta_method: str = field(
        default="analytic",
        metadata={
            "help": (
                "Meta-controller update rule. 'analytic' applies closed-form gradients "
                "derived from the MaxEnt potential."
            )
        },
    )
    controller_meta_lr: float = field(
        default=0.0,
        metadata={
            "help": (
                "Learning rate for meta-controller updates. Leave at zero to disable gradient steps."
            )
        },
    )
    controller_meta_tau_lr: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional tau-specific learning rate for meta-controller updates. "
                "When > 0, overrides controller_meta_lr for tau updates."
            )
        },
    )
    controller_meta_beta_lr: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional beta-specific learning rate for meta-controller updates. "
                "When > 0, overrides controller_meta_lr for beta updates."
            )
        },
    )
    controller_meta_beta_grad_clip: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional absolute clip value applied to the meta-controller beta gradient "
                "(kl - kl_target) before applying the update. Set > 0 to enable."
            )
        },
    )
    controller_meta_update_interval: int = field(
        default=1,
        metadata={
            "help": (
                "Number of training steps between meta-controller updates when enabled."
            )
        },
    )
    controller_meta_objective: str = field(
        default="potential",
        metadata={
            "help": (
                "Name of the objective optimized by the meta-controller. The default "
                "optimizes the regularized potential."
            )
        },
    )
    controller_meta_analytic_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of inner steps to include when computing analytic meta-gradients."
            )
        },
    )
    controller_meta_optimizer: str = field(
        default="sgd",
        metadata={
            "help": (
                "Optimizer used for the meta-controller when truncated/first-order "
                "updates are enabled. Supported: 'sgd', 'adam', 'adamw'."
            )
        },
    )
    controller_meta_truncation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of truncated inner steps used for first-order/backprop-based meta updates."
            )
        },
    )
    controller_meta_use_hessian: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to include second-order (Hessian) information when approximating meta-gradients."
            )
        },
    )
    maxent_use_clip_objective: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to blend in a PPO-style clipped objective on top of the MaxEnt loss."
            )
        },
    )
    maxent_clip_objective_coef: float = field(
        default=1.0,
        metadata={
            "help": "Scaling applied to the PPO-style clipped objective when enabled."
        },
    )
    maxent_clip_adv_baseline: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional baseline (e.g., 1/K) subtracted from the MaxEnt advantages before clipping."
            )
        },
    )
    controller_overwrite_from_config: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming from checkpoints, overwrite the controller's tau/beta scalars "
                "with the recipe values before training continues."
            )
        },
    )
    train_grpo_objective: bool = field(
        default=False,
        metadata={
            "help": (
                "When true, disable the MaxEnt reference weighting term and train with the standard GRPO objective."
            )
        },
    )
    if not isinstance(_base_fields, dict) or "scale_rewards" not in _base_fields:
        scale_rewards: bool = field(
            default=True,
            metadata={
                "help": (
                    "Whether to divide group advantages by their standard deviation "
                    "(mirrors TRL's scale_rewards behavior)."
                )
            },
        )
    maxent_allow_empty_weight_fallback: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow training to fall back to uniform GRPO-style weights when MaxEnt weighting returns no samples."
            )
        },
    )
    maxent_clip_range: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If set, override the PPO clip range specifically for the MaxEnt objective "
                "(falls back to clip_range)."
            )
        },
    )
    clip_range: float = field(
        default=_BASE_CLIP_RANGE_DEFAULT,
        metadata={"help": "PPO clip range used for clipping ratios in training loss."},
    )
    clip_range_high: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Upper PPO clip range (epsilon_high). Defaults to clip_range when unset."
            )
        },
    )
    clip_delta: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional additional slack applied to both clip bounds (two-sided clipping)."
            )
        },
    )
    grpo_loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "GRPO loss aggregation: 'grpo', 'bnpo', or 'dr_grpo'."
        },
    )
    info_seed_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable InfoSeed auxiliary loss and seed-conditioned sampling. "
                "When false, seed augmentation and seed loss remain disabled."
            )
        },
    )
    info_seed_num_seeds: int = field(
        default=0,
        metadata={
            "help": "Number of seeds per prompt for InfoSeed augmentation. Ignored when disabled."
        },
    )
    info_seed_lambda: float = field(
        default=0.0,
        metadata={"help": "Scaling factor for the InfoSeed auxiliary loss."},
    )
    info_seed_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature applied to the InfoSeed contrastive loss."},
    )
    info_seed_alpha_entropy: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional entropy MI-style weight applied to H(orig)-H(seed_aug); "
                "kept for compatibility, unused by default."
            )
        },
    )
    info_seed_prompt_template: str = field(
        default="\n[seed={seed}]",
        metadata={
            "help": (
                "Template appended to prompts for seed-augmented completions. "
                "Supports {seed} placeholder; include {prompt} to override full rendering."
            )
        },
    )
    info_seed_loss_type: str = field(
        default="infonce",
        metadata={"help": "Seed loss type: 'ce' or 'infonce'."},
    )
    info_seed_pooling: str = field(
        default="mean",
        metadata={"help": "Pooling for seed representations: 'mean' or 'last'."},
    )
    kl_target: float = field(
        default=0.0,
        metadata={
            "help": (
                "Target KL value for the β controller. Leave at zero to disable automatic adjustment."
            )
        },
    )
    kl_horizon: int = field(
        default=0,
        metadata={
            "help": "Horizon (in optimizer steps) for the β controller. Zero disables the controller.",
        },
    )
    kl_ctl_step_size: float = field(
        default=0.0,
        metadata={
            "help": "Maximum fractional change allowed per β controller update. Zero disables adaptation.",
        },
    )
    gen_temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature used for candidate generation."},
    )
    gen_top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p used for candidate generation."},
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": (
                "vLLM backend mode: 'server' for HTTP API or 'colocate' for "
                "in-process generation."
            )
        },
    )
    vllm_url: Optional[str] = field(
        default="http://localhost:8000/generate",
        metadata={"help": "Base URL for vLLM /generate when use_vllm is true."},
    )
    vllm_max_completion_rounds: int = field(
        default=0,
        metadata={
            "help": (
                "Maximum number of vLLM /generate attempts to top off missing "
                "completions. Set to 0 to match the requested num_generations."
            )
        },
    )
    vllm_retry_sleep: float = field(
        default=0.5,
        metadata={
            "help": "Seconds to sleep between vLLM completion retries when n outputs are missing."
        },
    )
    vllm_backfill_with_model: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to fall back to the local HF model.generate() to fill missing "
                "completions after exhausting vLLM retries."
            )
        },
    )
    vllm_return_logprobs: bool = field(
        default=True,
        metadata={
            "help": (
                "Request per-token logprobs from vLLM so the frozen reference model "
                "does not need to rescore completions."
            )
        },
    )
    vllm_force_logprobs: bool = field(
        default=False,
        metadata={
            "help": (
                "Force vLLM logprob requests even when reference logprobs are sourced "
                "from the model/policy (useful for debugging)."
            )
        },
    )
    vllm_logprob_fail_after: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Consecutive steps with missing vLLM logprobs before aborting; "
                "set to 0 to disable (None uses env/default)."
            )
        },
    )
    vllm_logprob_fallback: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When true, switch reference logprobs to the model after missing "
                "vLLM logprobs (None uses env/default)."
            )
        },
    )
    vllm_client_tag_fail_fast: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Abort vLLM retries immediately on client_tag mismatch "
                "(None uses env/default)."
            )
        },
    )
    vllm_sync_interval_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Only sync weights to vLLM every N optimizer steps when vllm_sync_weights is true. "
                "Set to 0 to disable sync; None syncs every step. Default: 1."
            )
        },
    )
    vllm_sync_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "When using a long-lived vLLM server, push updated weights to the "
                "server before generation so completions reflect the latest policy."
            )
        },
    )
    vllm_best_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional best_of parameter forwarded to vLLM (matches TRL)."
        },
    )
    vllm_frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty applied during sampling (matches TRL)."},
    )
    vllm_presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty applied during sampling (matches TRL)."},
    )
    vllm_top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k sampling parameter forwarded to vLLM (matches TRL)."},
    )
    vllm_stop_sequences: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Stop sequences for vLLM; accepts JSON list or '||'-delimited string "
                "to match TRL's CLI."
            )
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of worker processes for the training DataLoader."},
    )
    dataloader_pin_memory: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to pin memory in the training DataLoader (None uses default)."
        },
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Prefetch factor per DataLoader worker (only valid when num_workers > 0)."
            )
        },
    )
    dataloader_persistent_workers: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Keep DataLoader workers alive between epochs (only when num_workers > 0)."
        },
    )
    disable_distributed_sampler: bool = field(
        default=False,
        metadata={
            "help": "Disable DistributedSampler to avoid double sharding under Accelerate."
        },
    )
    log_level: str | int = field(
        default="warning",
        metadata={"help": "Logging level applied to the training process."},
    )
    log_completions: bool = field(
        default=False,
        metadata={"help": "Log prompt/completion samples when W&B logging is enabled."},
    )

    def __post_init__(self) -> None:
        base_post_init = getattr(super(), "__post_init__", None)
        if callable(base_post_init):
            try:
                base_post_init()
            except (AttributeError, ImportError, ModuleNotFoundError) as exc:
                # TRL/Transformers may be absent in some test environments.
                LOG.debug("Skipping base __post_init__ (dependency unavailable): %s", exc)
        vllm_mode = str(getattr(self, "vllm_mode", "server") or "server").strip().lower()
        if vllm_mode in {"inline", "local", "inprocess", "in-process"}:
            vllm_mode = "colocate"
        if vllm_mode not in {"server", "colocate"}:
            raise ValueError("vllm_mode must be one of: server, colocate")
        setattr(self, "vllm_mode", vllm_mode)
        loss_type = str(getattr(self, "grpo_loss_type", "") or "").strip().lower()
        base_loss_type = getattr(self, "loss_type", None)
        if not loss_type and base_loss_type is not None:
            loss_type = str(base_loss_type or "").strip().lower()
        if not loss_type:
            loss_type = "bnpo"
        if loss_type not in {"grpo", "bnpo", "dr_grpo"}:
            LOG.warning(
                "Unknown grpo_loss_type=%s; defaulting to 'bnpo'.",
                loss_type,
            )
            loss_type = "bnpo"
        setattr(self, "grpo_loss_type", loss_type)
        if base_loss_type is None or str(base_loss_type).strip() == "":
            try:
                setattr(self, "loss_type", loss_type)
            except Exception:
                pass
        vllm_url = getattr(self, "vllm_url", None)
        if isinstance(vllm_url, str):
            normalized = vllm_url.strip()
            parsed = None
            if normalized:
                try:
                    parsed = urlparse(normalized)
                except ValueError:
                    parsed = None
            if (
                parsed
                and parsed.scheme
                and parsed.netloc
                and (parsed.path in ("", "/"))
                and not parsed.params
                and not parsed.query
                and not parsed.fragment
            ):
                normalized = f"{parsed.scheme}://{parsed.netloc}/generate"
                setattr(self, "vllm_url", normalized)
        def _parse_stop_sequences(raw: object) -> Optional[list[str]]:
            if raw is None:
                return None
            if isinstance(raw, (list, tuple)):
                cleaned = [
                    str(item)
                    for item in raw
                    if item is not None and str(item).strip()
                ]
                return cleaned or None
            if isinstance(raw, str):
                stripped = raw.strip()
                if not stripped:
                    return None
                parsed_val: Optional[object]
                try:
                    parsed_val = json.loads(stripped)
                except (TypeError, ValueError):
                    parsed_val = None
                if isinstance(parsed_val, list):
                    cleaned = [
                        str(item)
                        for item in parsed_val
                        if item is not None and str(item).strip()
                    ]
                    return cleaned or None
                if isinstance(parsed_val, str):
                    stripped = parsed_val.strip()
                if "||" in stripped:
                    parts = [part.strip() for part in stripped.split("||")]
                    cleaned = [part for part in parts if part]
                    return cleaned or None
                return [stripped]
            return [str(raw)]

        raw_stops = getattr(self, "vllm_stop_sequences", None)
        parsed_stops = _parse_stop_sequences(raw_stops)
        if parsed_stops is not None:
            setattr(self, "vllm_stop_sequences", parsed_stops)
        elif isinstance(raw_stops, str):
            setattr(self, "vllm_stop_sequences", None)
        if self.maxent_tau_min < 0.0:
            raise ValueError("maxent_tau_min must be non-negative")
        if self.maxent_tau_max < self.maxent_tau_min:
            raise ValueError("maxent_tau_max must be >= maxent_tau_min")
        if self.maxent_tau_lr < 0.0:
            raise ValueError("maxent_tau_lr must be non-negative")
        if (
            self.maxent_tau_warmup_steps is not None
            and self.maxent_tau_warmup_steps < -1
        ):
            raise ValueError("maxent_tau_warmup_steps must be >= -1")
        if self.controller_meta_lr < 0.0:
            raise ValueError("controller_meta_lr must be non-negative")
        if self.controller_meta_tau_lr < 0.0:
            raise ValueError("controller_meta_tau_lr must be non-negative")
        if self.controller_meta_beta_lr < 0.0:
            raise ValueError("controller_meta_beta_lr must be non-negative")
        if self.controller_meta_beta_grad_clip < 0.0:
            raise ValueError("controller_meta_beta_grad_clip must be non-negative")
        if self.controller_meta_update_interval < 1:
            raise ValueError("controller_meta_update_interval must be >= 1")
        if self.controller_meta_analytic_steps < 1:
            raise ValueError("controller_meta_analytic_steps must be >= 1")
        if self.controller_meta_truncation_steps < 1:
            raise ValueError("controller_meta_truncation_steps must be >= 1")
        if self.maxent_q_epsilon <= 0.0:
            raise ValueError("maxent_q_epsilon must be > 0 to avoid zero weights")
        if self.maxent_q_temperature <= 0.0:
            raise ValueError("maxent_q_temperature must be > 0")
        if self.maxent_alpha < 0.0:
            raise ValueError("maxent_alpha must be non-negative")
        if self.policy_entropy_bonus_coef < 0.0:
            raise ValueError("policy_entropy_bonus_coef must be non-negative")
        entropy_mode = (
            str(getattr(self, "maxent_policy_entropy_mode", "exact") or "exact")
            .strip()
            .lower()
        )
        if entropy_mode in {"", "none"}:
            entropy_mode = "exact"
        if entropy_mode in {"exact", "full", "distribution"}:
            entropy_mode = "exact"
        elif entropy_mode in {
            "sample",
            "estimate",
            "estimated",
            "approx",
            "approximate",
            "token",
            "token_logp",
            "nll",
            "logp",
        }:
            entropy_mode = "sample"
        else:
            raise ValueError("maxent_policy_entropy_mode must be one of: exact, sample")
        setattr(self, "maxent_policy_entropy_mode", entropy_mode)
        if self.maxent_logprob_chunk_size < 0:
            raise ValueError("maxent_logprob_chunk_size must be non-negative")
        ref_source = (
            str(getattr(self, "maxent_reference_logprobs_source", "auto") or "auto")
            .strip()
            .lower()
        )
        if ref_source not in {"auto", "model", "policy", "none"}:
            raise ValueError(
                "maxent_reference_logprobs_source must be one of: auto, model, policy, none"
            )
        behavior_source = (
            str(getattr(self, "behavior_logprobs_source", "model") or "model")
            .strip()
            .lower()
        )
        if behavior_source in {"metadata", "meta"}:
            behavior_source = "vllm"
        if behavior_source not in {"model", "vllm"}:
            raise ValueError(
                "behavior_logprobs_source must be one of: model, vllm"
            )
        setattr(self, "behavior_logprobs_source", behavior_source)
        clip_range_high = getattr(self, "clip_range_high", None)
        if clip_range_high is not None and clip_range_high < 0.0:
            raise ValueError("clip_range_high must be non-negative when set")
        clip_delta = getattr(self, "clip_delta", None)
        if clip_delta is not None and clip_delta < 0.0:
            raise ValueError("clip_delta must be non-negative when set")
        if (
            self.maxent_score_tail_tokens is not None
            and self.maxent_score_tail_tokens <= 0
        ):
            raise ValueError("maxent_score_tail_tokens must be positive when set")
        if self.maxent_score_slice_prefetch < 0:
            raise ValueError("maxent_score_slice_prefetch must be non-negative")
        if self.maxent_prompt_cache_size < 0:
            raise ValueError("maxent_prompt_cache_size must be non-negative")
        sync_interval = getattr(self, "vllm_sync_interval_steps", None)
        if sync_interval is not None:
            try:
                sync_interval = int(sync_interval)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "vllm_sync_interval_steps must be an integer when set"
                ) from exc
            if sync_interval < 0:
                raise ValueError("vllm_sync_interval_steps must be >= 0 when set")
            setattr(self, "vllm_sync_interval_steps", sync_interval)
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        prefetch = getattr(self, "dataloader_prefetch_factor", None)
        if prefetch is not None:
            try:
                prefetch = int(prefetch)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "dataloader_prefetch_factor must be an integer when set"
                ) from exc
            if prefetch <= 0:
                raise ValueError("dataloader_prefetch_factor must be > 0 when set")
            setattr(self, "dataloader_prefetch_factor", prefetch)
        vllm_logprob_fail_after = getattr(self, "vllm_logprob_fail_after", None)
        if vllm_logprob_fail_after is not None:
            try:
                vllm_logprob_fail_after = int(vllm_logprob_fail_after)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "vllm_logprob_fail_after must be an integer when set"
                ) from exc
            if vllm_logprob_fail_after < 0:
                raise ValueError("vllm_logprob_fail_after must be >= 0 when set")
            setattr(self, "vllm_logprob_fail_after", vllm_logprob_fail_after)
        if self.maxent_clip_objective_coef < 0.0:
            raise ValueError("maxent_clip_objective_coef must be non-negative")
        if self.maxent_clip_range is not None and self.maxent_clip_range < 0.0:
            raise ValueError("maxent_clip_range must be non-negative when set")
        for name in ("kl_target", "kl_horizon", "kl_ctl_step_size"):
            value = getattr(self, name, None)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def get_process_log_level(self) -> int:
        """Return the numeric log level honoring the configured overrides."""

        resolved = _parse_log_level(getattr(self, "log_level", None))
        if resolved is not None:
            return resolved
        try:
            parent_getter = super().get_process_log_level
        except AttributeError:
            return logging.INFO
        try:
            return parent_getter()
        except (TypeError, ValueError):
            return logging.INFO


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script.

    Extends :class:`~maxent_grpo.config.ScriptArguments` with reward, dataset,
    and evaluation knobs used by MaxEnt-GRPO training pipelines.

    :ivar cosine_min_value_wrong: Minimum reward when the answer is wrong.
    :ivar cosine_max_value_wrong: Maximum reward when the answer is wrong.
    :ivar cosine_min_value_correct: Minimum reward for correct answers.
    :ivar cosine_max_value_correct: Maximum reward for correct answers.
    :ivar cosine_max_len: Maximum length considered when scaling cosine reward.
    :ivar repetition_n_grams: N-gram size for repetition penalty rewards.
    :ivar repetition_max_penalty: Maximum negative penalty for repetition rewards.
    :ivar dataset_prompt_column: Column used as prompts during training.
    :ivar dataset_solution_column: Column containing the reference solution.
    :ivar eval_dataset_name: Dataset to use for evaluation when different from training.
    :ivar eval_dataset_config: Config name for the evaluation dataset.
    :ivar eval_dataset_split: Split to read from the evaluation dataset.
    :ivar eval_dataset_prompt_column: Prompt column for the evaluation dataset.
    :ivar eval_dataset_solution_column: Solution column for the evaluation dataset.
    :ivar max_completion_len: Maximum completion length in characters.
    :ivar soft_punish_cache: Minimum completion length before applying a soft penalty.
    :ivar span_kl_target: Per-token KL target used by the span KL controller.
    :ivar span_kl_beta0: Initial KL coefficient for span KL regularization.
    :ivar span_kl_horizon: Horizon (steps) for the span KL controller.
    """

    eval_reward_funcs: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Optional override for eval rewards; defaults to reward_funcs when empty."
        },
    )
    eval_reward_weights: list[float] = field(
        default_factory=list,
        metadata={
            "help": "Optional weights for eval rewards; length must match eval_reward_funcs."
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for repetition penalty reward"},
    )

    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Column to use as prompts for training."},
    )
    dataset_solution_column: str = field(
        default="answer",
        metadata={"help": "Column to use as the gold solution/answer for training."},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset to use exclusively for evaluation."},
    )
    eval_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Config name for the evaluation dataset when applicable."},
    )
    eval_dataset_split: str = field(
        default="validation",
        metadata={"help": "Split to load from the evaluation dataset."},
    )
    eval_dataset_prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Prompt column for the evaluation dataset (defaults to training column)."
        },
    )
    eval_dataset_solution_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Answer column for the evaluation dataset (defaults to training column)."
        },
    )
    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )
    span_kl_target: float = field(
        default=0.05, metadata={"help": "per-token KL target"}
    )
    span_kl_beta0: float = field(default=0.12, metadata={"help": "initial KL coeff"})
    span_kl_horizon: int = field(
        default=10000, metadata={"help": "KL controller horizon"}
    )


__all__ = ["GRPOConfig", "GRPOScriptArguments"]
