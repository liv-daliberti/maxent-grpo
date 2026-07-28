"""Arguments for the Dr.GRPO/xDr.GRPO training surface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from oat.algorithms.ppo import PPOArgs


@dataclass
class ZeroMathArgs(PPOArgs):
    """OAT PPO arguments plus the small set of knobs used in this project."""

    prompt_template: Literal[
        "qwen_boxed",
        "qwen_countdown_digits",
        "qwen_graph_digits",
        "qwen_math",
        "qwen_math_route",
        "no",
        "r1",
    ] = field(default="qwen_math")
    test_split: str = "all"
    verifier_version: Literal["fast", "math_verify"] = field(default="fast")

    # Dr.GRPO is recovered exactly when xdr_tau is infinite. Finite values
    # apply detached candidate weights softmax(U/tau); zero is the exact
    # argmax-set limit.
    xdr_tau: float = math.inf
    # E44 composes reward-directed xDr aggregation with a separately added
    # semantic policy-gradient advantage. When enabled, xDr's detached row
    # weights are computed from the ordinary task advantage captured before
    # semantic augmentation; the actor still optimizes the combined advantage.
    # This prevents semantic novelty from being counted once in the advantage
    # and again through the aggregation weights.
    xdr_task_advantage_weights: bool = False
    xdr_mode_adaptive: bool = False
    # Optional label-free feedback controller over xDr's aggregation
    # temperature. A positive target ratio enables the controller; zero keeps
    # the fixed-tau treatment exactly unchanged.
    xdr_tau_control_target_ratio: float = 0.0
    xdr_tau_control_warmup_steps: int = 64
    xdr_tau_control_min: float = 0.005
    xdr_tau_control_ema_decay: float = 0.9
    xdr_tau_control_gain: float = 20.0
    # Distinct Haarnoja-style dual controller.  It learns a positive inverse-
    # tau strength with Adam from the signed entropy-target error, rather than
    # applying the one-sided proportional rule above.
    xdr_sac_dual_target_ratio: float = 0.0
    xdr_sac_dual_warmup_steps: int = 64
    xdr_sac_dual_min_tau: float = 0.005
    xdr_sac_dual_max_tau: float = 0.5
    xdr_sac_dual_alpha_lr: float = 0.003

    # Direct on-policy maximum-entropy Dr.GRPO.  The coefficient multiplies
    # raw completion-policy sequence entropy H(pi(.|x)).  Entropy is
    # differentiated at sampled prefixes rather than injected into the
    # group-relative reward advantage.  The two optional controllers observe
    # the same raw sequence-entropy quantity used by the actor objective.
    maxent_alpha: float = 0.0
    # ``sequence`` is the literal trajectory-entropy formulation retained for
    # E11--E20. ``conditional_token_mean`` is E21's free-form, length-neutral
    # formulation: at each sampled response state it maximizes entropy over
    # non-EOS content tokens, does not importance-differentiate state
    # visitation, and averages within each response before averaging rows.
    maxent_objective: Literal[
        "sequence", "conditional_token_mean"
    ] = "sequence"
    maxent_control_target_ratio: float = 0.0
    maxent_control_target_entropy: float = 0.0
    maxent_control_warmup_steps: int = 64
    maxent_control_max_alpha: float = 0.5
    maxent_control_ema_decay: float = 0.9
    maxent_control_gain: float = 1.0
    maxent_dual_target_ratio: float = 0.0
    maxent_dual_target_entropy: float = 0.0
    maxent_dual_warmup_steps: int = 64
    maxent_dual_min_alpha: float = 0.005
    maxent_dual_max_alpha: float = 0.5
    maxent_dual_alpha_lr: float = 0.003
    # Smooth the per-prompt-group entropy sensor before dual Adam. A zero
    # decay explicitly recovers the historical instantaneous-feedback rule.
    maxent_dual_ema_decay: float = 0.7
    # Projection-free, memoryless inverse control of the coefficient on the
    # direct MaxEnt objective. It calibrates exclusively from the same entropy
    # quantity differentiated by that objective, never from evaluation labels
    # or the canonical bank. The reference dose is maxent_alpha.
    maxent_inverse_adaptation: bool = False
    maxent_inverse_warmup_steps: int = 64
    maxent_inverse_ema_decay: float = 0.9
    # Optional constrained-MaxEnt length dual. The actor optimizes
    # E[R] + alpha H - lambda(E[L] - target), with raw generated response
    # length L and a separate projected dual controller over lambda. It may
    # accompany fixed, proportional, or Haarnoja-controlled entropy alpha; a
    # zero target leaves those established MaxEnt paths unchanged.
    maxent_length_target: float = 0.0
    maxent_length_lambda_init: float = 0.0
    maxent_length_lambda_max: float = 0.02
    maxent_length_ema_decay: float = 0.9
    maxent_length_dual_lr: float = 0.0002

    # DIAYN-style answer-option binding.  When enabled, each rollout group is
    # split across K latent answer options by augmenting the prompt with z.
    # The learner adds beta * (log q(z | answer_repr) - log 1/K) to terminal
    # reward, where q is an EMA discriminator over extracted final answers.
    diayn_num_options: int = 0
    diayn_mi_beta: float = 0.0
    diayn_mi_ema_decay: float = 0.9
    diayn_mi_smoothing: float = 1.0
    diayn_mi_bonus_clip: float = 5.0
    diayn_mi_correct_only: bool = True
    diayn_mi_leave_one_out: bool = False

    # Group-local outcome-collision shaping for free-form Dr.GRPO. Every
    # observed canonical answer, including one shared INVALID key for parse
    # failures, receives -(coefficient / G) for each same-key peer in its
    # candidate group. Zero recovers the unmodified reward exactly.
    outcome_collision_coef: float = 0.0
    # Opt-in free-form semantic policy-gradient variant. When true, the
    # outcome-collision penalty is not added to terminal reward before
    # Dr.GRPO's group baseline. Instead, the learner adds that detached
    # leave-one-out penalty once to the already group-centered sequence
    # advantage. False preserves E37's reward-shaping behavior exactly.
    outcome_collision_outside_centering: bool = False

    # Prompt-specific predictive Shannon-surprisal shaping over observed
    # canonical answers. Counts from prior groups and current leave-one-out
    # peers define a smoothed answer probability with one unseen bucket. The
    # bounded non-positive bonus preserves task-reward ordering. Zero disables
    # the tracker and recovers the unmodified reward exactly.
    semantic_shannon_coef: float = 0.0
    semantic_shannon_surprisal_clip: float = 5.0
    semantic_shannon_pseudocount: float = 1.0
    # E41 keeps ordinary task reward as the only input to Dr.GRPO's group
    # baseline. It adds a detached semantic advantage afterward, centered
    # under the prompt-local predictive distribution rather than the current
    # candidate group's empirical mean. False preserves E38 exactly.
    semantic_shannon_separate_advantage: bool = False
    # Optional safety gate for a future separate-advantage treatment. Only
    # reward-positive rows with parseable answer keys can receive semantic
    # pressure or enter the predictive history. Negative raw semantic
    # advantages are zeroed, and positive ones are capped explicitly.
    # False preserves both E38 and E41 exactly.
    semantic_shannon_quality_gated_advantage: bool = False
    semantic_shannon_quality_gated_cap: float = 0.05
    # E43 keeps the E42 success-conditioned support/history contract but
    # retains both signs of the predictor-centered advantage. Only active,
    # parseable, reward-positive rows receive pressure; their signed
    # advantages are clamped symmetrically. False preserves E38/E41/E42.
    semantic_shannon_success_conditioned_signed_advantage: bool = False
    semantic_shannon_success_conditioned_signed_cap: float = 0.05
    # E56 optionally replaces E43's fixed, projected semantic coefficient with
    # a projection-free inverse controller driven only by the model's own
    # normalized open-set predictive entropy. No catalogue size, gold support,
    # or desired entropy is supplied.
    semantic_shannon_open_set_inverse_adaptation: bool = False
    semantic_shannon_open_set_warmup_steps: int = 64
    semantic_shannon_open_set_ema_decay: float = 0.9

    # Online growing-support canonical MaxEnt. A prompt-local bank contains
    # only validator-positive canonical strategies produced on-policy. The
    # learner adds a bounded bank-entropy score plus a one-time per-class
    # discovery bonus after ordinary Dr.GRPO task centering.
    # The bank also runs passively by default for ordinary Dr.GRPO so normal
    # runs report cumulative verified discoveries and mean verified support
    # per prompt without altering rewards, advantages, or gradients.
    verified_discovery_tracking: bool = True
    online_canonical_bank_alpha: float = 0.0
    online_canonical_novelty_beta: float = 0.0
    online_canonical_bank_pseudocount: float = 1.0
    online_canonical_bank_surprisal_clip: float = 5.0
    # Haarnoja-style log-alpha control against the exact post-update ratio
    # H(q_x) / log |B_x^+|. A zero target preserves fixed-alpha E44 exactly.
    online_canonical_dual_target_ratio: float = 0.0
    online_canonical_dual_min_alpha: float = 0.005
    # Positive infinity disables only the controller's upper projection.
    online_canonical_dual_max_alpha: float = 0.5
    online_canonical_dual_alpha_lr: float = 0.003
    online_canonical_dual_ema_decay: float = 0.9
    # E51's projection-free alternative to the canonical-bank Haarnoja dual. The
    # model's masked-mean token entropy is calibrated separately in every run.
    # After warmup, alpha is the base dose times the warmup-mean/entropy-EMA
    # ratio with no upper or lower projection, so falling entropy raises alpha.
    online_canonical_policy_entropy_adaptation: bool = False
    online_canonical_policy_entropy_warmup_steps: int = 64
    online_canonical_policy_entropy_ema_decay: float = 0.9
    # Default-off verified exemplar replay. Once a prompt has at least two
    # observed validator-positive modes, teacher-forced model scores over one
    # stored exemplar per mode are balanced with KL(U_bank || q_model).
    # Its separate inverse coefficient is calibrated only from the model's
    # normalized entropy over the observed bank and has no alpha projection.
    online_canonical_replay: bool = False
    online_canonical_replay_alpha: float = 0.1
    # "bank_balance" is E53's conditioned-bank reverse KL. The successor
    # "verified_likelihood" keeps the same target-free sensor but gives the
    # actuator a non-zero common verified-mode score gradient.
    online_canonical_replay_objective: Literal[
        "bank_balance",
        "verified_likelihood",
        "verified_likelihood_per_rollout",
        "split_mass_balance_per_rollout",
    ] = "bank_balance"
    # A compute budget, not a semantic-support target. The default equals the
    # standard rollout width and never changes in response to evaluation data.
    online_canonical_replay_capacity: int = 16
    # Default-off cross-prompt scheduling. A positive value replays this many
    # model-discovered verified prompt banks per optimizer update in persistent
    # round-robin order. It is a fixed compute budget, not a support target.
    online_canonical_replay_global_groups_per_step: int = 0
    # Optional finite global cold-start phase. When positive, cross-prompt
    # replay is used for exactly this many non-empty optimizer updates and then
    # replay returns to the current prompt. Zero preserves either prompt-local
    # replay (global_groups_per_step=0) or unlimited global replay. The phase is
    # checkpointed and never observes evaluation or exhaustive support.
    online_canonical_replay_global_bootstrap_steps: int = 0
    online_canonical_replay_warmup_steps: int = 64
    online_canonical_replay_ema_decay: float = 0.9
    online_canonical_replay_mass_alpha: float = 0.1
    online_canonical_replay_mass_warmup_steps: int = 64
    online_canonical_replay_mass_ema_decay: float = 0.9
    # Optional model-self-proposal actuator. Once the current prompt has one
    # model-generated validator-positive outcome, sample a fixed-budget
    # temperature sweep from the untouched original task prompt and admit only
    # genuinely new executable outcomes. Proposal rows are discarded before
    # PPO. The retry budget and proposal temperatures are search-compute
    # parameters, not support or entropy targets.
    online_canonical_counterfactual_proposals: bool = False
    # Keep proposal-derived replay exemplars out of the on-policy count table
    # used by the canonical entropy/novelty advantage. This permits a literal
    # E58 objective plus a replay-support actuator without off-policy proposal
    # outcomes changing any neutral-rollout advantage.
    online_canonical_counterfactual_separate_objective_support: bool = False
    # E65: allow support-only proposals only for a singleton verified bank
    # while the unprojected open-set controller reports entropy below the
    # model's own warmup reference. At most one novel outcome is admitted.
    online_canonical_counterfactual_singleton_entropy_gate: bool = False
    online_canonical_counterfactual_anchor_max_tokens: int = 256
    online_canonical_counterfactual_max_attempts: int = 3
    online_canonical_counterfactual_sampling_temperature: float = 1.0
    online_canonical_key_mode: Literal[
        "modebench_outcome",
        "math_verified_answer",
        "math_strategy_qwen72",
    ] = "modebench_outcome"
    # ``math_verified_answer`` is an external-validity track, not a
    # multi-mode strategy claim. The ordinary MATH verifier maps every
    # reward-positive solution for a prompt to one shared ``correct`` outcome.
    # Consequently verified-mass replay may anchor a discovered solution, but
    # known-mode balance remains structurally ineligible unless a future
    # executable proof/strategy verifier supplies distinct outcome keys.
    # E47-calibrated semantic strategy IDs for validator-positive MATH only.
    # The endpoint is an OpenAI-compatible /v1 service for the frozen 72B
    # judge. Two independent temperature-zero permutations are mandatory.
    math_strategy_endpoint: str = ""
    math_strategy_model: str = "qwen2.5-72b"
    math_strategy_timeout_seconds: int = 600
    math_strategy_workers: int = 4
    math_strategy_max_item_chars: int = 4000
    # E49T bootstrap successor: answer-positive natural derivations may be
    # mapped to one exact frozen menu route by two unanimous semantic audits.
    # This never creates an open-set strategy and is off by default.
    math_strategy_allow_unstructured_inference: bool = False
    # For finite-menu MATH prompts, only an answer-positive response that
    # passes the exact declaration parser and both semantic execution audits
    # retains its task reward. This matched contract can be enabled for both
    # Dr.GRPO and canonical-MaxEnt arms.
    math_strategy_gate_task_reward: bool = False

    # E14's finite canonical graph policy. Rollouts contain exactly
    # canonical_graph_action_count stochastic actions, each chosen from the
    # one-token support {"1", "2", "3"}; termination is deterministic.
    canonical_graph_actions: bool = False
    # Generic task selector for new finite policies. ``canonical_graph_actions``
    # remains the exact backward-compatible E14 switch.
    canonical_action_task: Literal["none", "graph_coloring", "countdown"] = "none"
    canonical_graph_action_count: int = 3
    canonical_graph_learner_sampling: bool = False
    canonical_graph_fixed_shape_sampling: bool = False
    # Four-rank free-form execution that preserves one prompt and one complete
    # num_samples candidate group per optimizer update. The rollout is
    # generated once, replicated for group-relative statistics, and sharded
    # only for the backward pass.
    replicated_freeform_sampling: bool = False
    # Pair each learner rank with one collocated one-GPU actor for concurrent
    # model-weight broadcasts. This avoids serializing the full 7B model from
    # rank zero into four tensor-parallel actor workers.
    local_actor_weight_sync: bool = False
    # vLLM sleep level 2 discards actor weights instead of copying four full
    # models into host RAM. The learner remaps empty weight storage, broadcasts
    # the updated policy, and restores only the KV cache after each update.
    vllm_sleep_level: int = 1

    # Controls retained because they are reported in the paper.
    policy_entropy_coef: float = 0.0
    seed_entropy_alpha: float = 0.0

    eval_mode_coverage_k: int = 0
    eval_mode_coverage_temperature: float = 1.0
    # Repeated, fixed-seed K-draws expose Monte Carlo evaluation variance.
    # The established headline keys remain the mean across draws; every raw
    # draw and its spread are logged separately by the learner.
    eval_mode_coverage_draws: int = 4
    eval_mode_coverage_seed: int = 1001
    baseline_zero_adv_response_tokens: int = 8

    # Storage lifecycle.  Evaluation is intentionally independent from both
    # model export and resumable DeepSpeed state.  ``export_steps=0`` means
    # terminal-only; a negative value disables exports entirely.  Resume
    # checkpoints are opt-in at the Python surface and resolved to one prompt
    # epoch by the shared experiment launcher.
    export_steps: int = 0
    export_from: int = 0
    resume_steps: int = -1
    resume_from: int = 0
    max_export_num: int = 1
    max_resume_num: int = 1
    max_export_mem: int = 64
    max_resume_mem: int = 256
    prune_resume_on_success: bool = True


def resolve_canonical_action_task(args: ZeroMathArgs) -> str:
    """Resolve the generic task selector and E14's legacy graph boolean."""

    requested = str(getattr(args, "canonical_action_task", "none"))
    if requested not in {"none", "graph_coloring", "countdown"}:
        raise ValueError("canonical_action_task must be none, graph_coloring, or countdown")
    legacy_graph = bool(getattr(args, "canonical_graph_actions", False))
    if legacy_graph and requested not in {"none", "graph_coloring"}:
        raise ValueError(
            "canonical_graph_actions conflicts with canonical_action_task"
        )
    return "graph_coloring" if legacy_graph else requested


def validate_zero_math_args(args: ZeroMathArgs) -> ZeroMathArgs:
    """Reject configurations outside the single supported training surface."""

    if args.critic_type != "drgrpo":
        raise ValueError("This project supports critic_type=drgrpo only")
    if args.num_samples <= 1:
        raise ValueError("Dr.GRPO requires num_samples > 1")
    for name in ("export_from", "resume_from"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if int(args.export_steps) < -1:
        raise ValueError("export_steps must be -1, 0, or a positive interval")
    if int(args.resume_steps) == 0 or int(args.resume_steps) < -1:
        raise ValueError("resume_steps must be -1 or a positive interval")
    for name in (
        "max_export_num",
        "max_resume_num",
        "max_export_mem",
        "max_resume_mem",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if math.isnan(float(args.xdr_tau)) or float(args.xdr_tau) < 0:
        raise ValueError(
            "xdr_tau must be non-negative (0 = argmax-set limit; inf = Dr.GRPO)"
        )
    maxent_alpha = float(args.maxent_alpha)
    if not math.isfinite(maxent_alpha) or maxent_alpha < 0:
        raise ValueError("maxent_alpha must be finite and non-negative")
    if maxent_alpha > 0:
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "direct MaxEnt and signed-surrogate xDr are separate treatments"
            )
        if args.xdr_mode_adaptive:
            raise ValueError(
                "direct MaxEnt and mode-adaptive xDr are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError("direct MaxEnt and SEED are separate treatments")
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "direct MaxEnt and the legacy token-entropy control are separate treatments"
            )
    maxent_objective = str(getattr(args, "maxent_objective", "sequence"))
    if maxent_objective not in {"sequence", "conditional_token_mean"}:
        raise ValueError(
            "maxent_objective must be sequence or conditional_token_mean"
        )
    diayn_num_options = int(getattr(args, "diayn_num_options", 0) or 0)
    diayn_beta = float(getattr(args, "diayn_mi_beta", 0.0) or 0.0)
    if diayn_num_options < 0:
        raise ValueError("diayn_num_options must be non-negative")
    if not math.isfinite(diayn_beta) or diayn_beta < 0:
        raise ValueError("diayn_mi_beta must be finite and non-negative")
    if diayn_num_options <= 1 and diayn_beta > 0:
        raise ValueError("diayn_mi_beta requires diayn_num_options > 1")
    outcome_collision_coef = float(
        getattr(args, "outcome_collision_coef", 0.0) or 0.0
    )
    outcome_collision_outside_centering = bool(
        getattr(args, "outcome_collision_outside_centering", False)
    )
    if not math.isfinite(outcome_collision_coef) or outcome_collision_coef < 0:
        raise ValueError(
            "outcome_collision_coef must be finite and non-negative"
        )
    if outcome_collision_outside_centering and outcome_collision_coef <= 0:
        raise ValueError(
            "outcome_collision_outside_centering requires a positive "
            "outcome_collision_coef"
        )
    semantic_shannon_coef = float(
        getattr(args, "semantic_shannon_coef", 0.0) or 0.0
    )
    semantic_shannon_surprisal_clip = float(
        getattr(args, "semantic_shannon_surprisal_clip", 5.0)
    )
    semantic_shannon_pseudocount = float(
        getattr(args, "semantic_shannon_pseudocount", 1.0)
    )
    semantic_shannon_separate_advantage = bool(
        getattr(args, "semantic_shannon_separate_advantage", False)
    )
    semantic_shannon_quality_gated_advantage = bool(
        getattr(args, "semantic_shannon_quality_gated_advantage", False)
    )
    semantic_shannon_quality_gated_cap = float(
        getattr(args, "semantic_shannon_quality_gated_cap", 0.05)
    )
    semantic_shannon_success_conditioned_signed_advantage = bool(
        getattr(
            args,
            "semantic_shannon_success_conditioned_signed_advantage",
            False,
        )
    )
    semantic_shannon_success_conditioned_signed_cap = float(
        getattr(
            args,
            "semantic_shannon_success_conditioned_signed_cap",
            0.05,
        )
    )
    semantic_shannon_open_set_inverse_adaptation = bool(
        getattr(
            args,
            "semantic_shannon_open_set_inverse_adaptation",
            False,
        )
    )
    semantic_shannon_open_set_warmup_steps = int(
        getattr(args, "semantic_shannon_open_set_warmup_steps", 64)
    )
    semantic_shannon_open_set_ema_decay = float(
        getattr(args, "semantic_shannon_open_set_ema_decay", 0.9)
    )
    online_canonical_bank_alpha = float(
        getattr(args, "online_canonical_bank_alpha", 0.0) or 0.0
    )
    online_canonical_novelty_beta = float(
        getattr(args, "online_canonical_novelty_beta", 0.0) or 0.0
    )
    online_canonical_bank_pseudocount = float(
        getattr(args, "online_canonical_bank_pseudocount", 1.0)
    )
    online_canonical_bank_surprisal_clip = float(
        getattr(args, "online_canonical_bank_surprisal_clip", 5.0)
    )
    online_canonical_dual_target_ratio = float(
        getattr(args, "online_canonical_dual_target_ratio", 0.0) or 0.0
    )
    online_canonical_dual_min_alpha = float(
        getattr(args, "online_canonical_dual_min_alpha", 0.005)
    )
    online_canonical_dual_max_alpha = float(
        getattr(args, "online_canonical_dual_max_alpha", 0.5)
    )
    online_canonical_dual_alpha_lr = float(
        getattr(args, "online_canonical_dual_alpha_lr", 0.003)
    )
    online_canonical_dual_ema_decay = float(
        getattr(args, "online_canonical_dual_ema_decay", 0.9)
    )
    online_canonical_policy_entropy_adaptation = bool(
        getattr(
            args,
            "online_canonical_policy_entropy_adaptation",
            False,
        )
    )
    online_canonical_policy_entropy_warmup_steps = int(
        getattr(
            args,
            "online_canonical_policy_entropy_warmup_steps",
            64,
        )
    )
    online_canonical_policy_entropy_ema_decay = float(
        getattr(
            args,
            "online_canonical_policy_entropy_ema_decay",
            0.9,
        )
    )
    online_canonical_replay = bool(
        getattr(args, "online_canonical_replay", False)
    )
    online_canonical_replay_alpha = float(
        getattr(args, "online_canonical_replay_alpha", 0.1)
    )
    online_canonical_replay_objective = str(
        getattr(
            args,
            "online_canonical_replay_objective",
            "bank_balance",
        )
    )
    online_canonical_replay_capacity = int(
        getattr(args, "online_canonical_replay_capacity", 16)
    )
    online_canonical_replay_global_groups_per_step = int(
        getattr(
            args,
            "online_canonical_replay_global_groups_per_step",
            0,
        )
    )
    online_canonical_replay_global_bootstrap_steps = int(
        getattr(
            args,
            "online_canonical_replay_global_bootstrap_steps",
            0,
        )
    )
    online_canonical_replay_warmup_steps = int(
        getattr(args, "online_canonical_replay_warmup_steps", 64)
    )
    online_canonical_replay_ema_decay = float(
        getattr(args, "online_canonical_replay_ema_decay", 0.9)
    )
    online_canonical_replay_mass_alpha = float(
        getattr(args, "online_canonical_replay_mass_alpha", 0.1)
    )
    online_canonical_replay_mass_warmup_steps = int(
        getattr(args, "online_canonical_replay_mass_warmup_steps", 64)
    )
    online_canonical_replay_mass_ema_decay = float(
        getattr(args, "online_canonical_replay_mass_ema_decay", 0.9)
    )
    online_canonical_counterfactual_proposals = bool(
        getattr(
            args,
            "online_canonical_counterfactual_proposals",
            False,
        )
    )
    online_canonical_counterfactual_separate_objective_support = bool(
        getattr(
            args,
            "online_canonical_counterfactual_separate_objective_support",
            False,
        )
    )
    online_canonical_counterfactual_singleton_entropy_gate = bool(
        getattr(
            args,
            "online_canonical_counterfactual_singleton_entropy_gate",
            False,
        )
    )
    online_canonical_counterfactual_anchor_max_tokens = int(
        getattr(
            args,
            "online_canonical_counterfactual_anchor_max_tokens",
            256,
        )
    )
    online_canonical_key_mode = str(
        getattr(args, "online_canonical_key_mode", "modebench_outcome")
    )
    math_strategy_gate_task_reward = bool(
        getattr(args, "math_strategy_gate_task_reward", False)
    )
    math_strategy_allow_unstructured_inference = bool(
        getattr(args, "math_strategy_allow_unstructured_inference", False)
    )
    online_canonical_bank_active = (
        online_canonical_bank_alpha > 0.0
        or online_canonical_novelty_beta > 0.0
    )
    online_canonical_objective_active = (
        online_canonical_bank_active or online_canonical_replay
    )
    for name, value in (
        ("online_canonical_bank_alpha", online_canonical_bank_alpha),
        ("online_canonical_novelty_beta", online_canonical_novelty_beta),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name, value in (
        ("online_canonical_bank_pseudocount", online_canonical_bank_pseudocount),
        (
            "online_canonical_bank_surprisal_clip",
            online_canonical_bank_surprisal_clip,
        ),
        ("online_canonical_dual_min_alpha", online_canonical_dual_min_alpha),
        ("online_canonical_dual_alpha_lr", online_canonical_dual_alpha_lr),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        math.isnan(online_canonical_dual_max_alpha)
        or online_canonical_dual_max_alpha <= 0
    ):
        raise ValueError(
            "online_canonical_dual_max_alpha must be positive or +inf"
        )
    if (
        not math.isfinite(online_canonical_dual_target_ratio)
        or not 0 <= online_canonical_dual_target_ratio <= 1
    ):
        raise ValueError(
            "online_canonical_dual_target_ratio must be finite and in [0, 1]"
        )
    if (
        not math.isfinite(online_canonical_dual_ema_decay)
        or not 0 <= online_canonical_dual_ema_decay < 1
    ):
        raise ValueError(
            "online_canonical_dual_ema_decay must be finite and in [0, 1)"
        )
    if online_canonical_policy_entropy_warmup_steps <= 0:
        raise ValueError(
            "online_canonical_policy_entropy_warmup_steps must be positive"
        )
    if (
        not math.isfinite(online_canonical_policy_entropy_ema_decay)
        or not 0 <= online_canonical_policy_entropy_ema_decay < 1
    ):
        raise ValueError(
            "online_canonical_policy_entropy_ema_decay must be finite and "
            "in [0, 1)"
        )
    if (
        not math.isfinite(online_canonical_replay_alpha)
        or online_canonical_replay_alpha <= 0
    ):
        raise ValueError(
            "online_canonical_replay_alpha must be finite and positive"
        )
    if online_canonical_replay_objective not in {
        "bank_balance",
        "verified_likelihood",
        "verified_likelihood_per_rollout",
        "split_mass_balance_per_rollout",
    }:
        raise ValueError(
            "online_canonical_replay_objective must be bank_balance or "
            "verified_likelihood or verified_likelihood_per_rollout or "
            "split_mass_balance_per_rollout"
        )
    if online_canonical_replay_warmup_steps <= 0:
        raise ValueError(
            "online_canonical_replay_warmup_steps must be positive"
        )
    if online_canonical_replay_capacity < 2:
        raise ValueError(
            "online_canonical_replay_capacity must be at least two"
        )
    if online_canonical_replay_global_groups_per_step < 0:
        raise ValueError(
            "online_canonical_replay_global_groups_per_step must be "
            "non-negative"
        )
    if online_canonical_replay_global_bootstrap_steps < 0:
        raise ValueError(
            "online_canonical_replay_global_bootstrap_steps must be "
            "non-negative"
        )
    if (
        online_canonical_replay_global_groups_per_step > 0
        and not online_canonical_replay
    ):
        raise ValueError(
            "online_canonical_replay_global_groups_per_step requires replay"
        )
    if (
        online_canonical_replay_global_bootstrap_steps > 0
        and online_canonical_replay_global_groups_per_step <= 0
    ):
        raise ValueError(
            "online_canonical_replay_global_bootstrap_steps requires "
            "positive global groups per step"
        )
    if (
        not math.isfinite(online_canonical_replay_ema_decay)
        or not 0 <= online_canonical_replay_ema_decay < 1
    ):
        raise ValueError(
            "online_canonical_replay_ema_decay must be finite and in [0, 1)"
        )
    if (
        not math.isfinite(online_canonical_replay_mass_alpha)
        or online_canonical_replay_mass_alpha <= 0
    ):
        raise ValueError(
            "online_canonical_replay_mass_alpha must be finite and positive"
        )
    if online_canonical_replay_mass_warmup_steps <= 0:
        raise ValueError(
            "online_canonical_replay_mass_warmup_steps must be positive"
        )
    if (
        not math.isfinite(online_canonical_replay_mass_ema_decay)
        or not 0 <= online_canonical_replay_mass_ema_decay < 1
    ):
        raise ValueError(
            "online_canonical_replay_mass_ema_decay must be finite and "
            "in [0, 1)"
        )
    if online_canonical_counterfactual_anchor_max_tokens <= 0:
        raise ValueError(
            "online_canonical_counterfactual_anchor_max_tokens must be positive"
        )
    online_canonical_counterfactual_max_attempts = int(
        getattr(
            args,
            "online_canonical_counterfactual_max_attempts",
            3,
        )
    )
    if online_canonical_counterfactual_max_attempts <= 0:
        raise ValueError(
            "online_canonical_counterfactual_max_attempts must be positive"
        )
    online_canonical_counterfactual_sampling_temperature = float(
        getattr(
            args,
            "online_canonical_counterfactual_sampling_temperature",
            1.0,
        )
    )
    if (
        not math.isfinite(
            online_canonical_counterfactual_sampling_temperature
        )
        or online_canonical_counterfactual_sampling_temperature <= 0
    ):
        raise ValueError(
            "online_canonical_counterfactual_sampling_temperature must be "
            "finite and positive"
        )
    if online_canonical_replay:
        if not bool(getattr(args, "verified_discovery_tracking", True)):
            raise ValueError(
                "online canonical replay requires verified discovery tracking"
            )
        if online_canonical_key_mode not in {
            "modebench_outcome",
            "math_verified_answer",
        }:
            raise ValueError(
                "online canonical replay currently requires "
                "online_canonical_key_mode=modebench_outcome or "
                "math_verified_answer"
            )
    if online_canonical_counterfactual_proposals:
        if not online_canonical_replay:
            raise ValueError(
                "counterfactual canonical proposals require canonical replay"
            )
        if online_canonical_key_mode != "modebench_outcome":
            raise ValueError(
                "counterfactual canonical proposals require executable "
                "ModeBench outcome keys"
            )
        if (
            online_canonical_bank_alpha != 0.0
            or online_canonical_novelty_beta != 0.0
        ) and not (
            online_canonical_counterfactual_separate_objective_support
        ):
            raise ValueError(
                "proposal support cannot feed an on-policy "
                "canonical-bank advantage"
            )
        if not bool(getattr(args, "online_evaluation", False)):
            raise ValueError(
                "counterfactual canonical proposals require online validation"
            )
    elif online_canonical_counterfactual_separate_objective_support:
        raise ValueError(
            "separate counterfactual objective support requires "
            "counterfactual proposals"
        )
    if online_canonical_counterfactual_singleton_entropy_gate:
        if not online_canonical_counterfactual_proposals:
            raise ValueError(
                "singleton entropy gate requires counterfactual proposals"
            )
        if not semantic_shannon_open_set_inverse_adaptation:
            raise ValueError(
                "singleton entropy gate requires model-derived open-set "
                "inverse entropy adaptation"
            )
    if (
        online_canonical_policy_entropy_adaptation
        and online_canonical_dual_target_ratio > 0
    ):
        raise ValueError(
            "canonical policy-entropy adaptation and Haarnoja dual control "
            "are separate treatments"
        )
    if (
        online_canonical_policy_entropy_adaptation
        and bool(getattr(args, "maxent_inverse_adaptation", False))
    ):
        raise ValueError(
            "direct inverse MaxEnt and canonical policy-entropy adaptation "
            "are separate treatments"
        )
    if online_canonical_policy_entropy_adaptation:
        if not online_canonical_bank_active or online_canonical_bank_alpha <= 0:
            raise ValueError(
                "canonical policy-entropy adaptation requires a positive "
                "bank alpha"
            )
    if online_canonical_dual_target_ratio > 0:
        if not online_canonical_bank_active or online_canonical_bank_alpha <= 0:
            raise ValueError(
                "online canonical dual control requires a positive bank alpha"
            )
        if online_canonical_dual_min_alpha > online_canonical_bank_alpha:
            raise ValueError(
                "online_canonical_dual_min_alpha must not exceed bank alpha"
            )
        if online_canonical_dual_max_alpha < online_canonical_bank_alpha:
            raise ValueError(
                "online_canonical_dual_max_alpha must be at least bank alpha"
            )
    if online_canonical_key_mode not in {
        "modebench_outcome",
        "math_verified_answer",
        "math_strategy_qwen72",
    }:
        raise ValueError(
            "online_canonical_key_mode must be modebench_outcome, "
            "math_verified_answer, or math_strategy_qwen72"
        )
    if math_strategy_gate_task_reward:
        if online_canonical_key_mode != "math_strategy_qwen72":
            raise ValueError(
                "math strategy task-reward gating requires "
                "online_canonical_key_mode=math_strategy_qwen72"
            )
        if not bool(getattr(args, "verified_discovery_tracking", True)):
            raise ValueError(
                "math strategy task-reward gating requires canonical "
                "tracking so every positive response is audited"
            )
    if math_strategy_allow_unstructured_inference and (
        online_canonical_key_mode != "math_strategy_qwen72"
    ):
        raise ValueError(
            "unstructured MATH strategy inference requires "
            "online_canonical_key_mode=math_strategy_qwen72"
        )
    if online_canonical_objective_active:
        if args.critic_type != "drgrpo":
            raise ValueError(
                "online canonical bank requires critic_type=drgrpo"
            )
        if online_canonical_key_mode == "modebench_outcome":
            if (
                args.prompt_template != "qwen_boxed"
                or args.verifier_version != "fast"
                or args.test_split != "multi_answer"
            ):
                raise ValueError(
                    "online canonical banks require executable ModeBench "
                    "validation with "
                    "prompt_template=qwen_boxed, verifier_version=fast, "
                    "and test_split=multi_answer"
                )
        elif online_canonical_key_mode == "math_verified_answer":
            if (
                args.prompt_template != "qwen_math"
                or args.verifier_version != "math_verify"
                or args.test_split != "math"
            ):
                raise ValueError(
                    "verified-answer MATH replay requires "
                    "prompt_template=qwen_math, "
                    "verifier_version=math_verify, and test_split=math"
                )
        elif (
            args.prompt_template != "qwen_math"
            or args.verifier_version != "math_verify"
            or args.test_split != "math"
        ):
            raise ValueError(
                "MATH strategy banks require validator-bound "
                "prompt_template=qwen_math, verifier_version=math_verify, "
                "and test_split=math"
            )
    if online_canonical_key_mode == "math_strategy_qwen72" and (
        online_canonical_bank_active
        or bool(getattr(args, "verified_discovery_tracking", True))
        or math_strategy_gate_task_reward
    ):
        if not str(getattr(args, "math_strategy_endpoint", "")).strip():
            raise ValueError(
                "math_strategy_qwen72 requires math_strategy_endpoint"
            )
        if int(getattr(args, "math_strategy_timeout_seconds", 600)) <= 0:
            raise ValueError("math_strategy_timeout_seconds must be positive")
        if int(getattr(args, "math_strategy_workers", 4)) <= 0:
            raise ValueError("math_strategy_workers must be positive")
        if int(getattr(args, "math_strategy_max_item_chars", 4000)) < 600:
            raise ValueError("math_strategy_max_item_chars must be at least 600")
    if not math.isfinite(semantic_shannon_coef) or semantic_shannon_coef < 0:
        raise ValueError(
            "semantic_shannon_coef must be finite and non-negative"
        )
    if (
        not math.isfinite(semantic_shannon_surprisal_clip)
        or semantic_shannon_surprisal_clip <= 0
    ):
        raise ValueError(
            "semantic_shannon_surprisal_clip must be finite and positive"
        )
    if (
        not math.isfinite(semantic_shannon_pseudocount)
        or semantic_shannon_pseudocount <= 0
    ):
        raise ValueError(
            "semantic_shannon_pseudocount must be finite and positive"
        )
    if (
        semantic_shannon_separate_advantage
        and semantic_shannon_coef <= 0
    ):
        raise ValueError(
            "semantic_shannon_separate_advantage requires a positive "
            "semantic_shannon_coef"
        )
    if semantic_shannon_separate_advantage and args.critic_type != "drgrpo":
        raise ValueError(
            "semantic_shannon_separate_advantage requires critic_type=drgrpo"
        )
    if (
        not math.isfinite(semantic_shannon_quality_gated_cap)
        or semantic_shannon_quality_gated_cap <= 0
    ):
        raise ValueError(
            "semantic_shannon_quality_gated_cap must be finite and positive"
        )
    if (
        semantic_shannon_quality_gated_advantage
        and not semantic_shannon_separate_advantage
    ):
        raise ValueError(
            "semantic_shannon_quality_gated_advantage requires "
            "semantic_shannon_separate_advantage"
        )
    if (
        not math.isfinite(semantic_shannon_success_conditioned_signed_cap)
        or semantic_shannon_success_conditioned_signed_cap <= 0
    ):
        raise ValueError(
            "semantic_shannon_success_conditioned_signed_cap must be finite "
            "and positive"
        )
    if (
        semantic_shannon_success_conditioned_signed_advantage
        and not semantic_shannon_separate_advantage
    ):
        raise ValueError(
            "semantic_shannon_success_conditioned_signed_advantage requires "
            "semantic_shannon_separate_advantage"
        )
    if (
        semantic_shannon_success_conditioned_signed_advantage
        and semantic_shannon_quality_gated_advantage
    ):
        raise ValueError(
            "semantic Shannon quality-gated and success-conditioned signed "
            "advantages are separate treatments"
        )
    if (
        semantic_shannon_open_set_inverse_adaptation
        and not semantic_shannon_success_conditioned_signed_advantage
    ):
        raise ValueError(
            "semantic_shannon_open_set_inverse_adaptation requires the "
            "success-conditioned signed semantic advantage"
        )
    if semantic_shannon_open_set_warmup_steps <= 0:
        raise ValueError(
            "semantic_shannon_open_set_warmup_steps must be positive"
        )
    if (
        not math.isfinite(semantic_shannon_open_set_ema_decay)
        or not 0 <= semantic_shannon_open_set_ema_decay < 1
    ):
        raise ValueError(
            "semantic_shannon_open_set_ema_decay must be finite and in [0, 1)"
        )
    xdr_task_advantage_weights = bool(
        getattr(args, "xdr_task_advantage_weights", False)
    )
    if xdr_task_advantage_weights:
        if (
            not math.isfinite(float(args.xdr_tau))
            or float(args.xdr_tau) <= 0
        ):
            raise ValueError(
                "xdr_task_advantage_weights requires a finite positive xdr_tau"
            )
        if not semantic_shannon_success_conditioned_signed_advantage:
            raise ValueError(
                "xdr_task_advantage_weights requires the success-conditioned "
                "signed semantic advantage"
            )
        if bool(getattr(args, "xdr_mode_adaptive", False)):
            raise ValueError(
                "xdr_task_advantage_weights and mode-adaptive xDr are "
                "separate treatments"
            )
    if outcome_collision_coef > 0:
        if semantic_shannon_coef > 0:
            raise ValueError(
                "outcome-collision and semantic Shannon shaping are "
                "separate treatments"
            )
        if diayn_num_options > 1:
            raise ValueError(
                "outcome-collision shaping and DIAYN answer options are "
                "separate treatments"
            )
        if maxent_alpha > 0:
            raise ValueError(
                "outcome-collision shaping and direct MaxEnt are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError(
                "outcome-collision shaping and SEED are separate treatments"
            )
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "outcome-collision shaping and token entropy are separate treatments"
            )
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "outcome-collision shaping and signed-surrogate xDr are "
                "separate treatments"
            )
    if semantic_shannon_coef > 0:
        if diayn_num_options > 1:
            raise ValueError(
                "semantic Shannon shaping and DIAYN answer options are "
                "separate treatments"
            )
        open_set_inverse_maxent_composition = (
            semantic_shannon_open_set_inverse_adaptation
            and bool(getattr(args, "maxent_inverse_adaptation", False))
            and str(getattr(args, "maxent_objective", "sequence"))
            == "conditional_token_mean"
        )
        if maxent_alpha > 0 and not open_set_inverse_maxent_composition:
            raise ValueError(
                "semantic Shannon shaping and direct MaxEnt are separate "
                "treatments except for open-set semantic inverse adaptation "
                "with inverse conditional-token MaxEnt"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError(
                "semantic Shannon shaping and SEED are separate treatments"
            )
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "semantic Shannon shaping and token entropy are separate treatments"
            )
        if (
            math.isfinite(float(args.xdr_tau))
            and not xdr_task_advantage_weights
        ):
            raise ValueError(
                "semantic Shannon shaping and signed-surrogate xDr are "
                "separate treatments"
            )
    if online_canonical_objective_active:
        open_set_split_replay_composition = (
            semantic_shannon_open_set_inverse_adaptation
            and online_canonical_replay
            and online_canonical_replay_objective
            == "split_mass_balance_per_rollout"
        )
        if (
            semantic_shannon_coef > 0
            and not open_set_split_replay_composition
        ):
            raise ValueError(
                "online canonical bank and semantic Shannon are separate "
                "treatments except for open-set semantic inverse adaptation "
                "with split mass/balance replay"
            )
        if outcome_collision_coef > 0:
            raise ValueError(
                "online canonical bank and outcome collision are separate treatments"
            )
        if diayn_num_options > 1:
            raise ValueError(
                "online canonical bank and DIAYN are separate treatments"
            )
        inverse_fixed_canonical_hybrid = (
            bool(getattr(args, "maxent_inverse_adaptation", False))
            and str(getattr(args, "maxent_objective", "sequence"))
            == "conditional_token_mean"
            and online_canonical_dual_target_ratio == 0
            and not online_canonical_policy_entropy_adaptation
        )
        if maxent_alpha > 0 and not inverse_fixed_canonical_hybrid:
            raise ValueError(
                "online canonical bank and token-policy MaxEnt are separate "
                "treatments except for inverse conditional-token MaxEnt with "
                "a fixed canonical coefficient"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError(
                "online canonical bank and SEED are separate treatments"
            )
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "online canonical bank and token entropy are separate treatments"
            )
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "online canonical bank and signed-surrogate xDr are separate treatments"
            )
    if diayn_num_options > 1:
        if int(args.num_samples) % diayn_num_options != 0:
            raise ValueError("num_samples must divide evenly across DIAYN options")
        if maxent_alpha > 0:
            raise ValueError(
                "DIAYN answer-option MI and direct MaxEnt are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError(
                "DIAYN answer-option MI and SEED are separate treatments"
            )
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "DIAYN answer-option MI and token-entropy control are separate treatments"
            )
        if math.isfinite(float(args.xdr_tau)):
            raise ValueError(
                "DIAYN answer-option MI and signed-surrogate xDr are separate treatments"
            )
        if float(args.seed_entropy_alpha) > 0:
            raise ValueError("DIAYN answer-option MI and SEED are separate treatments")
        if float(args.policy_entropy_coef) > 0:
            raise ValueError(
                "DIAYN answer-option MI and token entropy are separate treatments"
            )
        coverage_k = int(getattr(args, "eval_mode_coverage_k", 0) or 0)
        if coverage_k > 0 and coverage_k % diayn_num_options != 0:
            raise ValueError(
                "eval_mode_coverage_k must divide evenly across DIAYN options"
            )
        for name in (
            "diayn_mi_ema_decay",
            "diayn_mi_smoothing",
            "diayn_mi_bonus_clip",
        ):
            value = float(getattr(args, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= float(args.diayn_mi_ema_decay) < 1:
            raise ValueError("diayn_mi_ema_decay must be in [0, 1)")
        if float(args.diayn_mi_smoothing) <= 0:
            raise ValueError("diayn_mi_smoothing must be positive")
        if float(args.diayn_mi_bonus_clip) <= 0:
            raise ValueError("diayn_mi_bonus_clip must be positive")
    canonical_task = resolve_canonical_action_task(args)
    canonical_actions = canonical_task != "none"
    canonical_graph_actions = canonical_task == "graph_coloring"
    canonical_action_count = int(args.canonical_graph_action_count)
    canonical_learner_sampling = bool(args.canonical_graph_learner_sampling)
    canonical_fixed_shape_sampling = bool(
        args.canonical_graph_fixed_shape_sampling
    )
    replicated_freeform_sampling = bool(args.replicated_freeform_sampling)
    if replicated_freeform_sampling:
        if canonical_actions:
            raise ValueError(
                "replicated_freeform_sampling cannot use canonical actions"
            )
        if int(args.rollout_batch_size_per_device) != 1:
            raise ValueError(
                "replicated free-form sampling requires "
                "rollout_batch_size_per_device=1"
            )
        if int(args.num_gpus_per_actor) <= 1 and not bool(
            args.local_actor_weight_sync
        ):
            raise ValueError(
                "replicated free-form sampling requires a multi-GPU actor or "
                "local_actor_weight_sync"
            )
    if online_canonical_counterfactual_proposals:
        if not replicated_freeform_sampling:
            raise ValueError(
                "counterfactual canonical proposals require replicated "
                "free-form sampling"
            )
    if bool(args.local_actor_weight_sync):
        if not replicated_freeform_sampling:
            raise ValueError(
                "local_actor_weight_sync requires replicated_freeform_sampling"
            )
        if int(args.num_gpus_per_actor) != 1:
            raise ValueError(
                "local_actor_weight_sync requires num_gpus_per_actor=1"
            )
    if int(args.vllm_sleep_level) not in {1, 2}:
        raise ValueError("vllm_sleep_level must be 1 or 2")
    if int(args.vllm_sleep_level) == 2:
        if not bool(args.vllm_sleep):
            raise ValueError("vllm_sleep_level=2 requires vllm_sleep")
        if not bool(args.local_actor_weight_sync):
            raise ValueError(
                "vllm_sleep_level=2 requires local_actor_weight_sync"
            )
        if int(args.sync_params_every) != 1:
            raise ValueError(
                "vllm_sleep_level=2 requires sync_params_every=1"
            )
    if canonical_actions:
        if online_canonical_bank_active:
            raise ValueError(
                "online growing-support banks require free-form rollouts"
            )
        if outcome_collision_coef > 0:
            raise ValueError(
                "outcome-collision shaping currently requires free-form rollouts"
            )
        if semantic_shannon_coef > 0:
            raise ValueError(
                "semantic Shannon shaping currently requires free-form rollouts"
            )
        if diayn_num_options > 1:
            raise ValueError("DIAYN answer options currently require free-form rollouts")
        if maxent_objective != "sequence":
            raise ValueError(
                "canonical finite policies require maxent_objective=sequence"
            )
        required_template = (
            "qwen_graph_digits"
            if canonical_graph_actions
            else "qwen_countdown_digits"
        )
        if args.prompt_template != required_template:
            raise ValueError(
                f"canonical {canonical_task} actions require "
                f"prompt_template={required_template}"
            )
        if canonical_action_count != 3:
            raise ValueError(
                "canonical finite policies require canonical_graph_action_count=3"
            )
        if args.test_split != "multi_answer":
            raise ValueError(
                "canonical finite policies require test_split=multi_answer"
            )
        if float(args.maxent_length_target) > 0:
            raise ValueError(
                "canonical fixed-horizon actions cannot use the response-length controller"
            )
        if float(args.top_p) != 1.0:
            raise ValueError("canonical actions require top_p=1")
        if int(args.top_k) != -1:
            raise ValueError("canonical actions require top_k=-1")
        if not math.isfinite(float(args.temperature)) or float(args.temperature) <= 0:
            raise ValueError(
                "canonical actions require finite positive temperature"
            )
        if canonical_learner_sampling and int(args.rollout_batch_size) != 1:
            raise ValueError(
                "canonical learner-side sampling requires rollout_batch_size=1"
            )
        if canonical_learner_sampling and not canonical_fixed_shape_sampling:
            raise ValueError(
                "canonical learner-side sampling requires the frozen fixed-shape "
                "causal-placeholder path"
            )
    elif args.prompt_template in {"qwen_graph_digits", "qwen_countdown_digits"}:
        raise ValueError(
            "canonical digit prompt templates require canonical_action_task"
        )
    elif canonical_learner_sampling:
        raise ValueError(
            "canonical_graph_learner_sampling requires canonical_graph_actions "
            "or canonical_action_task"
        )
    elif canonical_fixed_shape_sampling:
        raise ValueError(
            "canonical_graph_fixed_shape_sampling requires canonical graph actions "
            "or canonical_action_task"
        )
    if canonical_fixed_shape_sampling and not canonical_learner_sampling:
        raise ValueError(
            "canonical_graph_fixed_shape_sampling requires learner-side sampling"
        )
    if math.isfinite(float(args.xdr_tau)) and getattr(args, "reinforce_update", False):
        raise ValueError(
            "xDr treatments require the maintained non-REINFORCE learner path"
        )
    if args.xdr_mode_adaptive and (
        not math.isfinite(float(args.xdr_tau)) or float(args.xdr_tau) <= 0
    ):
        raise ValueError("xdr_mode_adaptive requires a finite positive xdr_tau")
    target_ratio = float(args.xdr_tau_control_target_ratio)
    if not math.isfinite(target_ratio) or not 0 <= target_ratio <= 1:
        raise ValueError("xdr_tau_control_target_ratio must be in [0, 1]")
    controller_base_tau = float(args.xdr_tau)
    if target_ratio > 0:
        if not math.isfinite(controller_base_tau) or controller_base_tau <= 0:
            raise ValueError(
                "xDr tau control requires a finite positive base temperature"
            )
        if args.xdr_mode_adaptive:
            raise ValueError(
                "xdr_mode_adaptive and xDr tau control are separate treatments"
            )
        tau_min = float(args.xdr_tau_control_min)
        if not math.isfinite(tau_min) or tau_min <= 0 or tau_min > controller_base_tau:
            raise ValueError("xdr_tau_control_min must be in (0, xdr_tau]")
        if int(args.xdr_tau_control_warmup_steps) <= 0:
            raise ValueError("xdr_tau_control_warmup_steps must be positive")
        ema_decay = float(args.xdr_tau_control_ema_decay)
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("xdr_tau_control_ema_decay must be in [0, 1)")
        gain = float(args.xdr_tau_control_gain)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("xdr_tau_control_gain must be finite and positive")
    sac_target_ratio = float(args.xdr_sac_dual_target_ratio)
    if not math.isfinite(sac_target_ratio) or not 0 <= sac_target_ratio <= 1:
        raise ValueError("xdr_sac_dual_target_ratio must be in [0, 1]")
    if target_ratio > 0 and sac_target_ratio > 0:
        raise ValueError(
            "proportional and SAC-dual xDr controllers are separate treatments"
        )
    if sac_target_ratio > 0:
        if not math.isfinite(controller_base_tau) or controller_base_tau <= 0:
            raise ValueError(
                "xDr SAC-dual control requires a finite positive base temperature"
            )
        if args.xdr_mode_adaptive:
            raise ValueError(
                "xdr_mode_adaptive and xDr SAC-dual control are separate treatments"
            )
        min_tau = float(args.xdr_sac_dual_min_tau)
        max_tau = float(args.xdr_sac_dual_max_tau)
        if not math.isfinite(min_tau) or min_tau <= 0 or min_tau > controller_base_tau:
            raise ValueError("xdr_sac_dual_min_tau must be in (0, xdr_tau]")
        if not math.isfinite(max_tau) or max_tau < controller_base_tau:
            raise ValueError(
                "xdr_sac_dual_max_tau must be at least the base temperature"
            )
        if int(args.xdr_sac_dual_warmup_steps) <= 0:
            raise ValueError("xdr_sac_dual_warmup_steps must be positive")
        alpha_lr = float(args.xdr_sac_dual_alpha_lr)
        if not math.isfinite(alpha_lr) or alpha_lr <= 0:
            raise ValueError("xdr_sac_dual_alpha_lr must be finite and positive")

    maxent_control_ratio = float(args.maxent_control_target_ratio)
    maxent_dual_ratio = float(args.maxent_dual_target_ratio)
    maxent_inverse_adaptation = bool(args.maxent_inverse_adaptation)
    maxent_control_target_entropy = float(args.maxent_control_target_entropy)
    maxent_dual_target_entropy = float(args.maxent_dual_target_entropy)
    for name, value in (
        ("maxent_control_target_ratio", maxent_control_ratio),
        ("maxent_dual_target_ratio", maxent_dual_ratio),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name, value in (
        ("maxent_control_target_entropy", maxent_control_target_entropy),
        ("maxent_dual_target_entropy", maxent_dual_target_entropy),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    active_maxent_controllers = sum(
        (
            maxent_control_ratio > 0,
            maxent_dual_ratio > 0,
            maxent_inverse_adaptation,
        )
    )
    if active_maxent_controllers > 1:
        raise ValueError(
            "proportional, Haarnoja-dual, and inverse MaxEnt controllers are "
            "separate treatments"
        )
    if maxent_control_target_entropy > 0 and maxent_control_ratio == 0:
        raise ValueError(
            "maxent_control_target_entropy requires proportional MaxEnt control"
        )
    if maxent_dual_target_entropy > 0 and maxent_dual_ratio == 0:
        raise ValueError(
            "maxent_dual_target_entropy requires Haarnoja-dual MaxEnt control"
        )
    if canonical_task != "none" and maxent_control_ratio > 0:
        if maxent_control_target_entropy <= 0:
            raise ValueError(
                "canonical proportional MaxEnt control requires an explicit "
                "positive maxent_control_target_entropy"
            )
    if canonical_task != "none" and maxent_dual_ratio > 0:
        if maxent_dual_target_entropy <= 0:
            raise ValueError(
                "canonical Haarnoja-dual MaxEnt control requires an explicit "
                "positive maxent_dual_target_entropy"
            )
    if canonical_task != "none":
        canonical_max_entropy = math.log(
            27 if canonical_task == "graph_coloring" else 108
        )
        for name, value in (
            ("maxent_control_target_entropy", maxent_control_target_entropy),
            ("maxent_dual_target_entropy", maxent_dual_target_entropy),
        ):
            if value > canonical_max_entropy + 1e-9:
                raise ValueError(
                    f"{name} cannot exceed the canonical {canonical_task} "
                    f"maximum log-support entropy {canonical_max_entropy:.12g}"
                )
    if active_maxent_controllers and maxent_alpha <= 0:
        raise ValueError("MaxEnt control requires a finite positive maxent_alpha")
    if maxent_inverse_adaptation:
        if int(args.maxent_inverse_warmup_steps) <= 0:
            raise ValueError("maxent_inverse_warmup_steps must be positive")
        inverse_ema_decay = float(args.maxent_inverse_ema_decay)
        if (
            not math.isfinite(inverse_ema_decay)
            or not 0 <= inverse_ema_decay < 1
        ):
            raise ValueError(
                "maxent_inverse_ema_decay must be finite and in [0, 1)"
            )
        if bool(args.online_canonical_policy_entropy_adaptation):
            raise ValueError(
                "direct inverse MaxEnt and canonical policy-entropy "
                "adaptation are separate treatments"
            )
    if maxent_control_ratio > 0:
        if int(args.maxent_control_warmup_steps) <= 0:
            raise ValueError("maxent_control_warmup_steps must be positive")
        max_alpha = float(args.maxent_control_max_alpha)
        if not math.isfinite(max_alpha) or max_alpha < maxent_alpha:
            raise ValueError("maxent_control_max_alpha must be at least maxent_alpha")
        ema_decay = float(args.maxent_control_ema_decay)
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("maxent_control_ema_decay must be in [0, 1)")
        gain = float(args.maxent_control_gain)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("maxent_control_gain must be finite and positive")
    if maxent_dual_ratio > 0:
        if int(args.maxent_dual_warmup_steps) <= 0:
            raise ValueError("maxent_dual_warmup_steps must be positive")
        min_alpha = float(args.maxent_dual_min_alpha)
        max_alpha = float(args.maxent_dual_max_alpha)
        if not math.isfinite(min_alpha) or min_alpha <= 0 or min_alpha > maxent_alpha:
            raise ValueError("maxent_dual_min_alpha must be in (0, maxent_alpha]")
        if not math.isfinite(max_alpha) or max_alpha < maxent_alpha:
            raise ValueError("maxent_dual_max_alpha must be at least maxent_alpha")
        alpha_lr = float(args.maxent_dual_alpha_lr)
        if not math.isfinite(alpha_lr) or alpha_lr <= 0:
            raise ValueError("maxent_dual_alpha_lr must be finite and positive")
        ema_decay = float(args.maxent_dual_ema_decay)
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("maxent_dual_ema_decay must be in [0, 1)")
    maxent_length_target = float(args.maxent_length_target)
    if not math.isfinite(maxent_length_target) or maxent_length_target < 0:
        raise ValueError("maxent_length_target must be finite and non-negative")
    if maxent_length_target > 0:
        if maxent_alpha <= 0:
            raise ValueError("MaxEnt length control requires positive maxent_alpha")
        horizon = float(args.generate_max_length)
        if maxent_length_target < 1 or maxent_length_target > horizon:
            raise ValueError("maxent_length_target must be in [1, generate_max_length]")
        initial_lambda = float(args.maxent_length_lambda_init)
        max_lambda = float(args.maxent_length_lambda_max)
        if (
            not math.isfinite(initial_lambda)
            or initial_lambda < 0
            or not math.isfinite(max_lambda)
            or max_lambda <= 0
            or initial_lambda > max_lambda
        ):
            raise ValueError(
                "maxent_length_lambda_init must be in [0, maxent_length_lambda_max]"
            )
        length_ema_decay = float(args.maxent_length_ema_decay)
        if not math.isfinite(length_ema_decay) or not 0 <= length_ema_decay < 1:
            raise ValueError("maxent_length_ema_decay must be in [0, 1)")
        length_dual_lr = float(args.maxent_length_dual_lr)
        if not math.isfinite(length_dual_lr) or length_dual_lr <= 0:
            raise ValueError("maxent_length_dual_lr must be finite and positive")
        if bool(args.ignore_no_eos):
            raise ValueError(
                "MaxEnt length control must retain horizon-truncated no-EOS rows"
            )
    if float(args.policy_entropy_coef) < 0:
        raise ValueError("policy_entropy_coef must be non-negative")
    if float(args.seed_entropy_alpha) < 0:
        raise ValueError("seed_entropy_alpha must be non-negative")
    if math.isfinite(float(args.xdr_tau)) and float(args.seed_entropy_alpha) > 0:
        raise ValueError("xDr and SEED are separate arms; enable at most one")
    if int(args.eval_mode_coverage_k) < 0:
        raise ValueError("eval_mode_coverage_k must be non-negative")
    if float(args.eval_mode_coverage_temperature) < 0:
        raise ValueError("eval_mode_coverage_temperature must be non-negative")
    if int(args.eval_mode_coverage_draws) < 1:
        raise ValueError("eval_mode_coverage_draws must be positive")
    if int(args.eval_mode_coverage_seed) < 0:
        raise ValueError("eval_mode_coverage_seed must be non-negative")
    if int(args.baseline_zero_adv_response_tokens) < 0:
        raise ValueError("baseline_zero_adv_response_tokens must be non-negative")
    return args
