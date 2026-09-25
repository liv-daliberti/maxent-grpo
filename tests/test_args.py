import math
from types import SimpleNamespace

import pytest

from oat_drgrpo.args import ZeroMathArgs, validate_zero_math_args


def _args(**overrides):
    values = {
        "critic_type": "drgrpo",
        "num_samples": 32,
        "xdr_tau": 0.05,
        "reinforce_update": False,
        "xdr_mode_adaptive": False,
        "xdr_tau_control_target_ratio": 0.0,
        "xdr_tau_control_warmup_steps": 64,
        "xdr_tau_control_min": 0.005,
        "xdr_tau_control_ema_decay": 0.9,
        "xdr_tau_control_gain": 20.0,
        "xdr_sac_dual_target_ratio": 0.0,
        "xdr_sac_dual_warmup_steps": 64,
        "xdr_sac_dual_min_tau": 0.005,
        "xdr_sac_dual_max_tau": 0.5,
        "xdr_sac_dual_alpha_lr": 0.003,
        "maxent_alpha": 0.0,
        "maxent_objective": "sequence",
        "maxent_control_target_ratio": 0.0,
        "maxent_control_target_entropy": 0.0,
        "maxent_control_warmup_steps": 64,
        "maxent_control_max_alpha": 0.5,
        "maxent_control_ema_decay": 0.9,
        "maxent_control_gain": 1.0,
        "maxent_dual_target_ratio": 0.0,
        "maxent_dual_target_entropy": 0.0,
        "maxent_dual_warmup_steps": 64,
        "maxent_dual_min_alpha": 0.005,
        "maxent_dual_max_alpha": 0.5,
        "maxent_dual_alpha_lr": 0.003,
        "maxent_dual_ema_decay": 0.7,
        "maxent_inverse_adaptation": False,
        "maxent_inverse_warmup_steps": 64,
        "maxent_inverse_ema_decay": 0.9,
        "maxent_length_target": 0.0,
        "maxent_length_lambda_init": 0.0,
        "maxent_length_lambda_max": 0.02,
        "maxent_length_ema_decay": 0.9,
        "maxent_length_dual_lr": 0.0002,
        "diayn_num_options": 0,
        "diayn_mi_beta": 0.0,
        "diayn_mi_ema_decay": 0.9,
        "diayn_mi_smoothing": 1.0,
        "diayn_mi_bonus_clip": 5.0,
        "diayn_mi_correct_only": True,
        "diayn_mi_leave_one_out": False,
        "outcome_collision_coef": 0.0,
        "outcome_collision_outside_centering": False,
        "semantic_shannon_coef": 0.0,
        "semantic_shannon_surprisal_clip": 5.0,
        "semantic_shannon_pseudocount": 1.0,
        "semantic_shannon_separate_advantage": False,
        "semantic_shannon_quality_gated_advantage": False,
        "semantic_shannon_quality_gated_cap": 0.05,
        "semantic_shannon_success_conditioned_signed_advantage": False,
        "semantic_shannon_success_conditioned_signed_cap": 0.05,
        "semantic_shannon_success_conditioned_group_centered_advantage": False,
        "semantic_shannon_success_conditioned_verified_support_advantage": False,
        "semantic_shannon_verified_support_include_replay_bank": False,
        "online_canonical_bank_alpha": 0.0,
        "online_canonical_bank_pseudocount": 1.0,
        "online_canonical_bank_surprisal_clip": 5.0,
        "online_canonical_dual_target_ratio": 0.0,
        "online_canonical_dual_min_alpha": 0.005,
        "online_canonical_dual_max_alpha": 0.5,
        "online_canonical_dual_alpha_lr": 0.003,
        "online_canonical_dual_ema_decay": 0.9,
        "online_canonical_policy_entropy_adaptation": False,
        "online_canonical_policy_entropy_warmup_steps": 64,
        "online_canonical_policy_entropy_ema_decay": 0.9,
        "online_canonical_replay": False,
        "online_canonical_replay_alpha": 0.1,
        "online_canonical_replay_objective": "bank_balance",
        "online_canonical_replay_capacity": 16,
        "online_canonical_replay_global_groups_per_step": 0,
        "online_canonical_replay_global_bootstrap_steps": 0,
        "online_canonical_replay_mass_alpha": 0.1,
        "online_canonical_key_mode": "modebench_outcome",
        "verified_route_replay_capacity_per_route": 16,
        "verified_route_recurring_min_neutral_prompts": 2,
        "verified_route_proposal_max_mean_logprob_drop": 2.0,
        "verified_discovery_tracking": True,
        "math_strategy_gate_task_reward": False,
        "math_strategy_allow_unstructured_inference": False,
        "policy_entropy_coef": 0.0,
        "seed_entropy_alpha": 0.0,
        "eval_mode_coverage_k": 8,
        "eval_mode_coverage_temperature": 1.0,
        "eval_mode_coverage_draws": 4,
        "eval_mode_coverage_seed": 1001,
        "baseline_zero_adv_response_tokens": 8,
        "export_steps": 0,
        "export_from": 0,
        "resume_steps": 96,
        "resume_from": 96,
        "max_export_num": 1,
        "max_resume_num": 1,
        "max_export_mem": 64,
        "max_resume_mem": 256,
        "prune_resume_on_success": True,
        "generate_max_length": 192,
        "ignore_no_eos": False,
        "canonical_graph_actions": False,
        "canonical_graph_action_count": 3,
        "canonical_graph_learner_sampling": False,
        "canonical_graph_fixed_shape_sampling": False,
        "replicated_freeform_sampling": False,
        "local_actor_weight_sync": False,
        "num_gpus_per_actor": 1,
        "vllm_sleep_level": 1,
        "vllm_sleep": True,
        "sync_params_every": 1,
        "rollout_batch_size": 1,
        "rollout_batch_size_per_device": 1,
        "prompt_template": "qwen_boxed",
        "verifier_version": "fast",
        "test_split": "all",
        "top_p": 1.0,
        "top_k": -1,
        "temperature": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tau_controller_accepts_finite_xdr_base():
    args = _args(xdr_tau_control_target_ratio=0.8)

    assert validate_zero_math_args(args) is args


def test_haarnoja_dual_entropy_ema_is_default_and_responsive():
    fields = ZeroMathArgs.__dataclass_fields__

    assert fields["maxent_dual_ema_decay"].default == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_mode_coverage_draws": 0}, "draws must be positive"),
        ({"eval_mode_coverage_seed": -1}, "seed must be non-negative"),
    ],
)
def test_repeated_evaluation_rejects_invalid_draw_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


def test_haarnoja_dual_rejects_invalid_entropy_ema_decay():
    with pytest.raises(ValueError, match="maxent_dual_ema_decay"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                maxent_alpha=0.05,
                maxent_dual_target_ratio=0.8,
                maxent_dual_ema_decay=1.0,
            )
        )


def test_inverse_maxent_accepts_label_free_conditional_entropy_control():
    args = _args(
        xdr_tau=float("inf"),
    )

    assert validate_zero_math_args(args) is args


def test_counterfactual_proposals_accept_only_support_only_replicated_replay():
    fields = ZeroMathArgs.__dataclass_fields__
    assert (
        fields["online_canonical_counterfactual_separate_objective_support"].default
        is False
    )
    assert fields["online_canonical_counterfactual_max_attempts"].default == 3
    assert (
        fields[
            "online_canonical_counterfactual_exact_grammar_transforms"
        ].default
        is False
    )
    assert fields[
        "online_canonical_counterfactual_sampling_temperature"
    ].default == pytest.approx(1.0)
    assert (
        fields["online_canonical_counterfactual_starvation_fallback"].default
        is False
    )
    assert fields[
        "online_canonical_counterfactual_starvation_patience_updates"
    ].default == 64
    assert fields[
        "online_canonical_counterfactual_starvation_fallback_max_attempts"
    ].default == 4
    assert (
        fields["online_canonical_counterfactual_fixed_control_groups"].default
        == 0
    )
    assert (
        fields[
            "online_canonical_counterfactual_admission_compute_only"
        ].default
        is False
    )
    assert fields["online_canonical_replay_compute_only"].default is False
    assert (
        fields["online_canonical_replay_retention_safe_balance"].default
        is False
    )
    assert fields["online_canonical_proposal_replay_priority_visits"].default == 0
    assert fields[
        "online_canonical_proposal_replay_priority_multiplier"
    ].default == pytest.approx(1.0)
    assert (
        fields["online_canonical_proposal_retention_tracking"].default
        is False
    )
    assert (
        fields[
            "online_canonical_proposal_adaptive_retention_priority"
        ].default
        is False
    )
    assert fields[
        "online_canonical_proposal_retention_max_missed_rollout_opportunities"
    ].default == 2
    assert fields[
        "online_canonical_proposal_retention_max_mean_logprob_drop"
    ].default == pytest.approx(0.5)
    args = _args(
        xdr_tau=float("inf"),
        test_split="multi_answer",
        online_evaluation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        online_canonical_counterfactual_proposals=True,
        online_canonical_counterfactual_anchor_max_tokens=128,
        replicated_freeform_sampling=True,
        local_actor_weight_sync=True,
    )

    assert validate_zero_math_args(args) is args

    compute_only = _args(
        xdr_tau=float("inf"),
        test_split="multi_answer",
        online_evaluation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_counterfactual_proposals=True,
        online_canonical_counterfactual_separate_objective_support=True,
        online_canonical_counterfactual_admission_compute_only=True,
        replicated_freeform_sampling=True,
        local_actor_weight_sync=True,
    )
    assert validate_zero_math_args(compute_only) is compute_only

    with pytest.raises(ValueError, match="admission compute-only requires"):
        validate_zero_math_args(
            _args(
                online_canonical_counterfactual_admission_compute_only=True,
            )
        )

    fallback_args = _args(
        xdr_tau=float("inf"),
        test_split="multi_answer",
        online_evaluation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        online_canonical_counterfactual_proposals=True,
        online_canonical_counterfactual_separate_objective_support=True,
        online_canonical_counterfactual_max_attempts=1,
        online_canonical_counterfactual_starvation_fallback=True,
        online_canonical_counterfactual_starvation_patience_updates=64,
        online_canonical_counterfactual_starvation_fallback_max_attempts=4,
        online_canonical_counterfactual_starvation_burst_updates=16,
        online_canonical_counterfactual_starvation_cooldown_updates=48,
        replicated_freeform_sampling=True,
        local_actor_weight_sync=True,
    )
    assert validate_zero_math_args(fallback_args) is fallback_args

    with pytest.raises(ValueError, match="fallback requires counterfactual"):
        validate_zero_math_args(
            _args(
                online_canonical_counterfactual_starvation_fallback=True,
            )
        )

    with pytest.raises(ValueError, match="must exceed the base"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                test_split="multi_answer",
                online_evaluation=True,
                online_canonical_replay=True,
                online_canonical_replay_objective=(
                    "split_mass_balance_per_rollout"
                ),
                online_canonical_counterfactual_proposals=True,
                online_canonical_counterfactual_separate_objective_support=True,
                online_canonical_counterfactual_max_attempts=4,
                online_canonical_counterfactual_starvation_fallback=True,
                online_canonical_counterfactual_starvation_fallback_max_attempts=4,
                replicated_freeform_sampling=True,
                local_actor_weight_sync=True,
            )
        )

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        validate_zero_math_args(
            _args(
                online_canonical_counterfactual_max_attempts=0,
            )
        )

    with pytest.raises(ValueError, match="temperature must be finite"):
        validate_zero_math_args(
            _args(
                online_canonical_counterfactual_sampling_temperature=float("inf"),
            )
        )

    with pytest.raises(ValueError, match="fixed counterfactual control groups"):
        validate_zero_math_args(
            _args(
                online_canonical_counterfactual_fixed_control_groups=3,
            )
        )

    with pytest.raises(ValueError, match="compute_only requires canonical replay"):
        validate_zero_math_args(
            _args(
                online_canonical_replay_compute_only=True,
            )
        )

    with pytest.raises(ValueError, match="require canonical replay"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                test_split="multi_answer",
                online_evaluation=True,
                online_canonical_counterfactual_proposals=True,
                replicated_freeform_sampling=True,
                local_actor_weight_sync=True,
            )
        )

    with pytest.raises(ValueError, match="cannot feed an on-policy"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                test_split="multi_answer",
                online_evaluation=True,
                online_canonical_replay=True,
                online_canonical_bank_alpha=0.1,
                online_canonical_counterfactual_proposals=True,
                replicated_freeform_sampling=True,
                local_actor_weight_sync=True,
            )
        )

    separated = _args(
        xdr_tau=float("inf"),
        test_split="multi_answer",
        online_evaluation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        online_canonical_counterfactual_proposals=True,
        online_canonical_counterfactual_separate_objective_support=True,
        replicated_freeform_sampling=True,
        local_actor_weight_sync=True,
    )
    assert validate_zero_math_args(separated) is separated

    full = _args(
        xdr_tau=float("inf"),
        test_split="multi_answer",
        online_evaluation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        online_canonical_replay_mass_alpha=0.1,
        online_canonical_replay_retention_safe_balance=True,
        online_canonical_counterfactual_proposals=True,
        online_canonical_counterfactual_separate_objective_support=True,
        online_canonical_proposal_replay_priority_visits=4,
        online_canonical_proposal_replay_priority_multiplier=4.0,
        replicated_freeform_sampling=True,
        local_actor_weight_sync=True,
    )
    assert validate_zero_math_args(full) is full

    retention = _args(
        xdr_tau=float("inf"),
        test_split="multi_answer",
        online_evaluation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        online_canonical_counterfactual_proposals=True,
        online_canonical_counterfactual_separate_objective_support=True,
        online_canonical_proposal_replay_priority_visits=4,
        online_canonical_proposal_replay_priority_multiplier=4.0,
        online_canonical_proposal_retention_tracking=True,
        online_canonical_proposal_adaptive_retention_priority=True,
        online_canonical_proposal_retention_refresh_visits=4,
        replicated_freeform_sampling=True,
        local_actor_weight_sync=True,
    )
    assert validate_zero_math_args(retention) is retention

    with pytest.raises(ValueError, match="requires retention tracking"):
        validate_zero_math_args(
            _args(
                online_canonical_proposal_adaptive_retention_priority=True,
            )
        )

    with pytest.raises(ValueError, match="requires global replay"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                test_split="multi_answer",
                online_evaluation=True,
                online_canonical_replay=True,
                online_canonical_replay_objective=(
                    "split_mass_balance_per_rollout"
                ),
                online_canonical_counterfactual_proposals=True,
                online_canonical_counterfactual_separate_objective_support=True,
                online_canonical_proposal_replay_priority_visits=4,
                online_canonical_proposal_replay_priority_multiplier=4.0,
                online_canonical_proposal_retention_tracking=True,
                online_canonical_proposal_adaptive_retention_priority=True,
                replicated_freeform_sampling=True,
                local_actor_weight_sync=True,
            )
        )

    with pytest.raises(ValueError, match="retention-safe balance requires split"):
        validate_zero_math_args(
            _args(
                online_canonical_replay=True,
                online_canonical_replay_objective="verified_likelihood_per_rollout",
                online_canonical_replay_retention_safe_balance=True,
            )
        )

    with pytest.raises(ValueError, match="priority requires counterfactual"):
        validate_zero_math_args(
            _args(
                online_canonical_proposal_replay_priority_visits=4,
                online_canonical_proposal_replay_priority_multiplier=4.0,
            )
        )

    with pytest.raises(
        ValueError,
        match="separate counterfactual objective support requires",
    ):
        validate_zero_math_args(
            _args(
                online_canonical_counterfactual_separate_objective_support=True,
            )
        )


def test_inverse_conditional_maxent_accepts_fixed_canonical_hybrid():
    args = _args(
        xdr_tau=float("inf"),
        maxent_alpha=0.000075,
        maxent_objective="conditional_token_mean",
        maxent_inverse_adaptation=True,
        online_canonical_bank_alpha=0.1,
        test_split="multi_answer",
    )

    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    "overrides",
    [
        {"maxent_inverse_adaptation": False},
        {"maxent_objective": "sequence"},
        {"online_canonical_dual_target_ratio": 0.8},
    ],
)
def test_canonical_direct_maxent_composition_requires_e52_contract(overrides):
    settings = {
        "xdr_tau": float("inf"),
        "maxent_alpha": 0.000075,
        "maxent_objective": "conditional_token_mean",
        "maxent_inverse_adaptation": True,
        "online_canonical_bank_alpha": 0.1,
        "test_split": "multi_answer",
    }
    settings.update(overrides)

    with pytest.raises(ValueError, match="fixed canonical coefficient"):
        validate_zero_math_args(_args(**settings))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "xdr_tau": float("inf"),
                "maxent_inverse_adaptation": True,
            },
            "positive maxent_alpha",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "maxent_alpha": 0.000075,
                "maxent_inverse_adaptation": True,
                "maxent_inverse_warmup_steps": 0,
            },
            "warmup_steps",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "maxent_alpha": 0.000075,
                "maxent_inverse_adaptation": True,
                "maxent_inverse_ema_decay": 1.0,
            },
            "ema_decay",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "maxent_alpha": 0.000075,
                "maxent_inverse_adaptation": True,
                "maxent_control_target_ratio": 0.8,
            },
            "separate treatments",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "maxent_alpha": 0.000075,
                "maxent_inverse_adaptation": True,
                "online_canonical_bank_alpha": 0.1,
                "online_canonical_policy_entropy_adaptation": True,
            },
            "separate treatments",
        ),
    ],
)
def test_inverse_maxent_rejects_invalid_or_confounded_control(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"xdr_tau_control_target_ratio": 0.8, "xdr_tau": float("inf")},
            "finite positive base",
        ),
        (
            {"xdr_tau_control_target_ratio": 0.8, "xdr_mode_adaptive": True},
            "separate treatments",
        ),
        (
            {"xdr_tau_control_target_ratio": 0.8, "xdr_tau_control_min": 0.1},
            "must be in",
        ),
    ],
)
def test_tau_controller_rejects_confounded_or_invalid_arm(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


def test_sac_dual_controller_accepts_finite_xdr_base():
    args = _args(xdr_sac_dual_target_ratio=0.8)

    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"xdr_sac_dual_target_ratio": 0.8, "xdr_tau": float("inf")},
            "finite positive base",
        ),
        (
            {"xdr_sac_dual_target_ratio": 0.8, "xdr_mode_adaptive": True},
            "separate treatments",
        ),
        (
            {
                "xdr_sac_dual_target_ratio": 0.8,
                "xdr_tau_control_target_ratio": 0.8,
            },
            "separate treatments",
        ),
        (
            {"xdr_sac_dual_target_ratio": 0.8, "xdr_sac_dual_min_tau": 0.1},
            "must be in",
        ),
        (
            {"xdr_sac_dual_target_ratio": 0.8, "xdr_sac_dual_max_tau": 0.01},
            "at least",
        ),
    ],
)
def test_sac_dual_rejects_confounded_or_invalid_arm(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


@pytest.mark.parametrize("controller", ["fixed", "proportional", "dual"])
def test_direct_maxent_accepts_all_three_control_modes(controller):
    overrides = {
        "maxent_alpha": 0.05,
        "xdr_tau": float("inf"),
    }
    if controller == "proportional":
        overrides["maxent_control_target_ratio"] = 0.8
    elif controller == "dual":
        overrides["maxent_dual_target_ratio"] = 0.8

    args = _args(**overrides)

    assert validate_zero_math_args(args) is args


def test_length_neutral_conditional_token_maxent_is_free_form_only():
    args = _args(
        maxent_alpha=0.001,
        maxent_objective="conditional_token_mean",
        xdr_tau=float("inf"),
    )

    assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="canonical finite policies"):
        validate_zero_math_args(
            _args(
                maxent_alpha=0.001,
                maxent_objective="conditional_token_mean",
                xdr_tau=float("inf"),
                canonical_action_task="graph_coloring",
                canonical_graph_actions=True,
                prompt_template="qwen_graph_digits",
                test_split="multi_answer",
            )
        )


def test_diayn_answer_options_accept_freeform_reward_shaping():
    args = _args(
        xdr_tau=float("inf"),
        diayn_num_options=4,
        diayn_mi_beta=0.2,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_outcome_collision_accepts_freeform_drgrpo():
    args = _args(
        xdr_tau=float("inf"),
        outcome_collision_coef=0.1,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_outcome_collision_outside_centering_accepts_freeform_drgrpo():
    fields = ZeroMathArgs.__dataclass_fields__
    assert fields["outcome_collision_outside_centering"].default is False

    args = _args(
        xdr_tau=float("inf"),
        outcome_collision_coef=0.1,
        outcome_collision_outside_centering=True,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_semantic_shannon_accepts_frozen_freeform_defaults():
    fields = ZeroMathArgs.__dataclass_fields__
    assert fields["semantic_shannon_coef"].default == pytest.approx(0.0)
    assert fields["semantic_shannon_surprisal_clip"].default == pytest.approx(5.0)
    assert fields["semantic_shannon_pseudocount"].default == pytest.approx(1.0)
    assert fields["semantic_shannon_separate_advantage"].default is False
    assert fields["semantic_shannon_quality_gated_advantage"].default is False
    assert fields["semantic_shannon_quality_gated_cap"].default == pytest.approx(0.05)
    assert (
        fields["semantic_shannon_success_conditioned_signed_advantage"].default is False
    )
    assert fields[
        "semantic_shannon_success_conditioned_signed_cap"
    ].default == pytest.approx(0.05)
    assert (
        fields[
            "semantic_shannon_success_conditioned_group_centered_advantage"
        ].default
        is False
    )
    assert (
        fields[
            "semantic_shannon_success_conditioned_verified_support_advantage"
        ].default
        is False
    )

    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_surprisal_clip=5.0,
        semantic_shannon_pseudocount=1.0,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_online_canonical_bank_requires_executable_modebench_contract():
    assert (
        ZeroMathArgs.__dataclass_fields__["verified_discovery_tracking"].default is True
    )
    args = _args(
        xdr_tau=float("inf"),
        online_canonical_bank_alpha=0.1,
        test_split="multi_answer",
    )
    assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="executable ModeBench"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_bank_alpha=0.1,
                prompt_template="qwen_math",
                verifier_version="math_verify",
                test_split="math",
            )
        )


def test_online_canonical_bank_accepts_task_bound_pantry_contract():
    args = _args(
        xdr_tau=float("inf"),
        canonical_action_task="pantry_support_mask",
        canonical_graph_action_count=6,
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        prompt_template="qwen_pantry_support_mask",
        verifier_version="fast",
        test_split="multi_answer",
        semantic_shannon_coef=0.1,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_signed_advantage=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
    )

    assert validate_zero_math_args(args) is args


def test_full_open_bank_accepts_only_fixed_shape_pantry_canonical_explorer():
    pantry = _args(
        xdr_tau=float("inf"),
        canonical_action_task="pantry_support_mask",
        canonical_graph_action_count=6,
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        prompt_template="qwen_pantry_support_mask",
        verifier_version="fast",
        test_split="multi_answer",
        online_evaluation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        online_canonical_replay_mass_alpha=0.1,
        online_canonical_replay_retention_safe_balance=True,
        online_canonical_counterfactual_proposals=True,
        online_canonical_counterfactual_separate_objective_support=True,
        online_canonical_proposal_replay_priority_visits=4,
        online_canonical_proposal_replay_priority_multiplier=4.0,
    )
    assert validate_zero_math_args(pantry) is pantry

    with pytest.raises(ValueError, match="fixed-shape Pantry learner sampler"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                canonical_action_task="pantry_support_mask",
                canonical_graph_action_count=6,
                canonical_graph_learner_sampling=True,
                canonical_graph_fixed_shape_sampling=False,
                prompt_template="qwen_pantry_support_mask",
                verifier_version="fast",
                test_split="multi_answer",
                online_evaluation=True,
                online_canonical_replay=True,
                online_canonical_replay_objective="split_mass_balance_per_rollout",
                online_canonical_counterfactual_proposals=True,
                online_canonical_counterfactual_separate_objective_support=True,
            )
        )


def test_online_canonical_math_strategy_requires_validator_and_endpoint():
    args = _args(
        xdr_tau=float("inf"),
        online_canonical_bank_alpha=0.1,
        online_canonical_key_mode="math_strategy_qwen72",
        math_strategy_endpoint="http://node105:8766/v1",
        prompt_template="qwen_math",
        verifier_version="math_verify",
        test_split="math",
    )
    assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="math_strategy_endpoint"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_key_mode="math_strategy_qwen72",
                math_strategy_endpoint="",
                prompt_template="qwen_math",
                verifier_version="math_verify",
                test_split="math",
            )
        )


def test_online_canonical_verified_answer_math_contract_is_explicitly_single_mode():
    args = _args(
        xdr_tau=float("inf"),
        online_canonical_replay=True,
        online_canonical_key_mode="math_verified_answer",
        prompt_template="qwen_math",
        verifier_version="math_verify",
        test_split="math",
    )
    assert validate_zero_math_args(args) is args

    route_dev = _args(
        xdr_tau=float("inf"),
        online_canonical_replay=True,
        online_canonical_key_mode="math_verified_answer",
        prompt_template="qwen_math",
        verifier_version="math_verify",
        test_split="math_dev",
    )
    assert validate_zero_math_args(route_dev) is route_dev

    with pytest.raises(ValueError, match="verified-answer MATH replay"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_replay=True,
                online_canonical_key_mode="math_verified_answer",
                prompt_template="qwen_boxed",
                verifier_version="fast",
                test_split="multi_answer",
            )
        )


def test_verified_route_mode_accepts_only_frozen_modebench_or_math_dev_contract():
    modebench = _args(
        xdr_tau=float("inf"),
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        online_canonical_key_mode="verified_route",
        prompt_template="qwen_boxed",
        verifier_version="fast",
        test_split="multi_answer",
    )
    assert validate_zero_math_args(modebench) is modebench

    math_dev = _args(
        xdr_tau=float("inf"),
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        online_canonical_key_mode="verified_route",
        prompt_template="qwen_math_route",
        verifier_version="math_verify",
        test_split="math_dev",
    )
    assert validate_zero_math_args(math_dev) is math_dev

    with pytest.raises(ValueError, match="sealed route-development"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_key_mode="verified_route",
                prompt_template="qwen_math_route",
                verifier_version="math_verify",
                test_split="math",
            )
        )
    with pytest.raises(ValueError, match="exactly one fixed-budget"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_replay=True,
                online_canonical_replay_objective=("verified_likelihood_per_rollout"),
                online_canonical_replay_global_groups_per_step=2,
                online_canonical_key_mode="verified_route",
                test_split="multi_answer",
            )
        )
    with pytest.raises(ValueError, match="canonical advantages at zero"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_replay=True,
                online_canonical_replay_objective=("verified_likelihood_per_rollout"),
                online_canonical_replay_global_groups_per_step=1,
                online_canonical_key_mode="verified_route",
                online_canonical_bank_alpha=0.1,
                test_split="multi_answer",
            )
        )
    with pytest.raises(ValueError, match="separate support store"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_evaluation=True,
                online_canonical_replay=True,
                online_canonical_replay_objective=("verified_likelihood_per_rollout"),
                online_canonical_replay_global_groups_per_step=1,
                online_canonical_counterfactual_proposals=True,
                online_canonical_key_mode="verified_route",
                test_split="multi_answer",
                replicated_freeform_sampling=True,
                local_actor_weight_sync=True,
            )
        )
    with pytest.raises(ValueError, match="temperatures must match"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_evaluation=True,
                online_canonical_replay=True,
                online_canonical_replay_objective=("verified_likelihood_per_rollout"),
                online_canonical_replay_global_groups_per_step=1,
                online_canonical_counterfactual_proposals=True,
                online_canonical_counterfactual_separate_objective_support=True,
                online_canonical_counterfactual_sampling_temperature=1.2,
                online_canonical_key_mode="verified_route",
                test_split="multi_answer",
                replicated_freeform_sampling=True,
                local_actor_weight_sync=True,
            )
        )


def test_math_strategy_task_reward_gate_requires_audited_math_mode():
    args = _args(
        xdr_tau=float("inf"),
        online_canonical_key_mode="math_strategy_qwen72",
        math_strategy_endpoint="http://node105:8766/v1",
        math_strategy_gate_task_reward=True,
        prompt_template="qwen_math",
        verifier_version="math_verify",
        test_split="math",
    )
    assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="key_mode=math_strategy_qwen72"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                math_strategy_gate_task_reward=True,
                test_split="multi_answer",
            )
        )
    with pytest.raises(ValueError, match="requires canonical tracking"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_key_mode="math_strategy_qwen72",
                math_strategy_endpoint="http://node105:8766/v1",
                math_strategy_gate_task_reward=True,
                verified_discovery_tracking=False,
                prompt_template="qwen_math",
                verifier_version="math_verify",
                test_split="math",
            )
        )


def test_online_canonical_dual_requires_valid_bank_alpha_and_bounds():
    args = _args(
        xdr_tau=float("inf"),
        online_canonical_bank_alpha=0.1,
        online_canonical_dual_target_ratio=0.8,
        online_canonical_dual_min_alpha=0.1,
        online_canonical_dual_max_alpha=0.5,
        test_split="multi_answer",
    )
    assert validate_zero_math_args(args) is args
    assert validate_zero_math_args(
        _args(
            xdr_tau=float("inf"),
            online_canonical_bank_alpha=0.1,
            online_canonical_dual_target_ratio=0.8,
            online_canonical_dual_min_alpha=0.1,
            online_canonical_dual_max_alpha=float("inf"),
            test_split="multi_answer",
        )
    ).online_canonical_dual_max_alpha == float("inf")

    with pytest.raises(ValueError, match="positive bank alpha"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_dual_target_ratio=0.8,
                test_split="multi_answer",
            )
        )
    with pytest.raises(ValueError, match="min_alpha must not exceed"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_bank_alpha=0.1,
                online_canonical_dual_target_ratio=0.8,
                online_canonical_dual_min_alpha=0.2,
                test_split="multi_answer",
            )
        )
    for invalid_max in (float("-inf"), float("nan")):
        with pytest.raises(ValueError, match=r"positive or \+inf"):
            validate_zero_math_args(
                _args(
                    xdr_tau=float("inf"),
                    online_canonical_bank_alpha=0.1,
                    online_canonical_dual_target_ratio=0.8,
                    online_canonical_dual_max_alpha=invalid_max,
                    test_split="multi_answer",
                )
            )


def test_online_canonical_policy_entropy_adaptation_is_exclusive():
    args = _args(
        xdr_tau=float("inf"),
        online_canonical_bank_alpha=0.1,
        online_canonical_policy_entropy_adaptation=True,
        online_canonical_policy_entropy_warmup_steps=64,
        online_canonical_policy_entropy_ema_decay=0.9,
        test_split="multi_answer",
    )
    assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="positive bank alpha"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_policy_entropy_adaptation=True,
                test_split="multi_answer",
            )
        )
    with pytest.raises(ValueError, match="separate treatments"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_policy_entropy_adaptation=True,
                online_canonical_dual_target_ratio=0.8,
                online_canonical_dual_min_alpha=0.1,
                test_split="multi_answer",
            )
        )


def test_canonical_replay_capacity_floor_follows_the_objective():
    """A one-mode bank is legal for the mass objectives and not for KL balance.

    ``bank_balance`` scores ``KL(U_g || softmax(scores_g))``, which is
    undefined on a singleton group, so its floor stays at two. The mass
    objectives already score singleton groups in normal training -- any prompt
    that has found exactly one mode is replayed as one -- so a capacity of one
    is a configurable ablation for them rather than an invalid request.
    """

    for objective in (
        "verified_likelihood",
        "verified_likelihood_per_rollout",
        "split_mass_balance_per_rollout",
    ):
        args = _args(
            xdr_tau=float("inf"),
            online_canonical_replay=True,
            online_canonical_replay_objective=objective,
            online_canonical_replay_capacity=1,
            test_split="multi_answer",
        )
        assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="capacity"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_replay=True,
                online_canonical_replay_objective="bank_balance",
                online_canonical_replay_capacity=1,
                test_split="multi_answer",
            )
        )


def test_canonical_replay_accepts_target_free_inverse_direct_entropy_hybrid():
    args = _args(
        xdr_tau=float("inf"),
        maxent_alpha=0.000075,
        maxent_objective="conditional_token_mean",
        maxent_inverse_adaptation=True,
        online_canonical_replay=True,
        online_canonical_replay_alpha=0.1,
        online_canonical_replay_capacity=16,
        test_split="multi_answer",
    )

    assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="capacity"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_replay=True,
                online_canonical_replay_capacity=1,
                test_split="multi_answer",
            )
        )
    with pytest.raises(ValueError, match="replay_objective"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_replay=True,
                online_canonical_replay_objective="gold_support_target",
                test_split="multi_answer",
            )
        )

    split = _args(
        xdr_tau=float("inf"),
        maxent_alpha=0.000075,
        maxent_objective="conditional_token_mean",
        maxent_inverse_adaptation=True,
        online_canonical_replay=True,
        online_canonical_replay_objective=("split_mass_balance_per_rollout"),
        test_split="multi_answer",
    )
    assert validate_zero_math_args(split) is split

    with pytest.raises(ValueError, match="mass_alpha"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                online_canonical_replay=True,
                online_canonical_replay_mass_alpha=0.0,
                test_split="multi_answer",
            )
        )


def test_global_verified_replay_requires_replay_and_accepts_fixed_compute_budget():
    args = _args(
        xdr_tau=float("inf"),
        online_canonical_replay=True,
        online_canonical_replay_objective=("split_mass_balance_per_rollout"),
        online_canonical_replay_global_groups_per_step=1,
        test_split="multi_answer",
    )
    assert validate_zero_math_args(args) is args

    with pytest.raises(ValueError, match="non-negative"):
        validate_zero_math_args(
            _args(
                online_canonical_replay=True,
                online_canonical_replay_global_groups_per_step=-1,
            )
        )
    with pytest.raises(ValueError, match="requires replay"):
        validate_zero_math_args(
            _args(
                online_canonical_replay=False,
                online_canonical_replay_global_groups_per_step=1,
            )
        )

    bootstrap = _args(
        xdr_tau=float("inf"),
        online_canonical_replay=True,
        online_canonical_replay_objective=("split_mass_balance_per_rollout"),
        online_canonical_replay_global_groups_per_step=1,
        online_canonical_replay_global_bootstrap_steps=64,
        test_split="multi_answer",
    )
    assert validate_zero_math_args(bootstrap) is bootstrap

    with pytest.raises(ValueError, match="bootstrap_steps.*non-negative"):
        validate_zero_math_args(
            _args(
                online_canonical_replay=True,
                online_canonical_replay_global_groups_per_step=1,
                online_canonical_replay_global_bootstrap_steps=-1,
            )
        )
    with pytest.raises(ValueError, match="requires positive global groups"):
        validate_zero_math_args(
            _args(
                online_canonical_replay=True,
                online_canonical_replay_global_groups_per_step=0,
                online_canonical_replay_global_bootstrap_steps=64,
            )
        )


def test_semantic_shannon_separate_advantage_accepts_freeform_drgrpo():
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_surprisal_clip=5.0,
        semantic_shannon_pseudocount=1.0,
        semantic_shannon_separate_advantage=True,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_semantic_shannon_quality_gate_accepts_explicit_capped_extension():
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_surprisal_clip=5.0,
        semantic_shannon_pseudocount=1.0,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_quality_gated_advantage=True,
        semantic_shannon_quality_gated_cap=0.05,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_semantic_shannon_success_conditioned_signed_accepts_extension():
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_signed_advantage=True,
        semantic_shannon_success_conditioned_signed_cap=0.05,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_semantic_shannon_group_centered_accepts_replay_composition():
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_group_centered_advantage=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        test_split="multi_answer",
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_semantic_shannon_accepts_explicit_zero_coefficient_control():
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.0,
        semantic_shannon_allow_zero_coefficient_control=True,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_verified_support_advantage=True,
        semantic_shannon_verified_support_include_replay_bank=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        test_split="multi_answer",
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_semantic_shannon_zero_coefficient_control_is_explicit_and_scoped():
    with pytest.raises(ValueError, match="explicit zero-coefficient"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                semantic_shannon_coef=0.0,
                semantic_shannon_separate_advantage=True,
            )
        )
    with pytest.raises(ValueError, match="requires semantic_shannon_separate"):
        validate_zero_math_args(
            _args(
                semantic_shannon_allow_zero_coefficient_control=True,
            )
        )


def test_semantic_shannon_verified_support_accepts_replay_composition():
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_verified_support_advantage=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        test_split="multi_answer",
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_semantic_shannon_open_set_uses_the_fixed_signed_advantage_path():
    fields = ZeroMathArgs.__dataclass_fields__
    assert fields["semantic_shannon_success_conditioned_signed_advantage"].default is False
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_signed_advantage=True,
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


def test_open_set_split_canonical_composition_accepts_fixed_coefficients():
    args = _args(
        xdr_tau=float("inf"),
        maxent_alpha=0.000075,
        maxent_objective="conditional_token_mean",
        maxent_inverse_adaptation=True,
        semantic_shannon_coef=0.1,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_signed_advantage=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="split_mass_balance_per_rollout",
        test_split="multi_answer",
    )

    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"semantic_shannon_coef": -0.1}, "finite and non-negative"),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_surprisal_clip": 0.0,
            },
            "finite and positive",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_pseudocount": float("inf"),
            },
            "finite and positive",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "outcome_collision_coef": 0.1,
            },
            "separate treatments",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "maxent_alpha": 0.05,
            },
            "direct MaxEnt",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "diayn_num_options": 4,
            },
            "DIAYN",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "canonical_action_task": "graph_coloring",
                "canonical_graph_actions": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
            },
            "free-form",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_separate_advantage": True,
            },
            "requires a positive",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "critic_type": "grpo",
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_separate_advantage": True,
            },
            "critic_type=drgrpo",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_quality_gated_advantage": True,
            },
            "requires semantic_shannon_separate_advantage",
        ),
        (
            {
                "semantic_shannon_quality_gated_cap": 0.0,
            },
            "finite and positive",
        ),
        (
            {
                "semantic_shannon_quality_gated_cap": float("inf"),
            },
            "finite and positive",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_success_conditioned_signed_advantage": True,
            },
            "requires semantic_shannon_separate_advantage",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_success_conditioned_group_centered_advantage": True,
            },
            "requires semantic_shannon_separate_advantage",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_success_conditioned_verified_support_advantage": True,
            },
            "requires semantic_shannon_separate_advantage",
        ),
        (
            {
                "semantic_shannon_success_conditioned_signed_cap": 0.0,
            },
            "finite and positive",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_separate_advantage": True,
                "semantic_shannon_quality_gated_advantage": True,
                "semantic_shannon_success_conditioned_signed_advantage": True,
            },
            "separate treatments",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_separate_advantage": True,
                "semantic_shannon_success_conditioned_signed_advantage": True,
                "semantic_shannon_success_conditioned_group_centered_advantage": True,
            },
            "separate treatments",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_separate_advantage": True,
                "semantic_shannon_success_conditioned_group_centered_advantage": True,
                "semantic_shannon_success_conditioned_verified_support_advantage": True,
            },
            "separate treatments",
        ),
    ],
)
def test_semantic_shannon_rejects_invalid_or_confounded_arms(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"outcome_collision_coef": -0.1}, "finite and non-negative"),
        (
            {"outcome_collision_outside_centering": True},
            "requires a positive",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "outcome_collision_coef": 0.1,
                "maxent_alpha": 0.05,
            },
            "direct MaxEnt",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "outcome_collision_coef": 0.1,
                "diayn_num_options": 4,
            },
            "DIAYN",
        ),
        (
            {
                "xdr_tau": float("inf"),
                "outcome_collision_coef": 0.1,
                "canonical_action_task": "graph_coloring",
                "canonical_graph_actions": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
            },
            "free-form",
        ),
    ],
)
def test_outcome_collision_rejects_invalid_or_confounded_arms(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"diayn_num_options": 4, "num_samples": 10}, "divide evenly"),
        ({"diayn_num_options": 1, "diayn_mi_beta": 0.1}, "requires"),
        (
            {"diayn_num_options": 4, "diayn_mi_beta": 0.1, "maxent_alpha": 0.05},
            "direct MaxEnt",
        ),
        (
            {
                "diayn_num_options": 4,
                "diayn_mi_beta": 0.1,
                "canonical_action_task": "graph_coloring",
                "canonical_graph_actions": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
            },
            "free-form",
        ),
    ],
)
def test_diayn_answer_options_reject_confounded_or_invalid_runs(overrides, message):
    base = {"xdr_tau": float("inf")}
    base.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**base))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"maxent_alpha": 0.05, "xdr_tau": 0.05},
            "separate treatments",
        ),
        (
            {
                "maxent_alpha": -0.1,
                "xdr_tau": float("inf"),
            },
            "maxent_alpha",
        ),
        (
            {
                "maxent_alpha": 0.05,
                "xdr_tau": float("inf"),
                "maxent_control_target_ratio": 0.8,
                "maxent_control_max_alpha": 0.01,
            },
            "maxent_control_max_alpha",
        ),
        (
            {
                "maxent_alpha": 0.05,
                "xdr_tau": float("inf"),
                "seed_entropy_alpha": 1.0,
            },
            "separate treatments",
        ),
    ],
)
def test_direct_maxent_rejects_confounded_or_invalid_arm(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


@pytest.mark.parametrize("entropy_control", ["fixed", "proportional", "dual"])
def test_length_constrained_maxent_accepts_every_entropy_alpha_mode(
    entropy_control,
):
    overrides = dict(
        maxent_alpha=0.002,
        maxent_length_target=16,
        xdr_tau=float("inf"),
    )
    if entropy_control == "proportional":
        overrides["maxent_control_target_ratio"] = 0.8
    elif entropy_control == "dual":
        overrides["maxent_dual_target_ratio"] = 0.8
        overrides["maxent_dual_min_alpha"] = 0.0005
    args = _args(**overrides)

    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"maxent_length_target": 16}, "positive maxent_alpha"),
        (
            {
                "maxent_alpha": 0.002,
                "maxent_length_target": 193,
                "xdr_tau": float("inf"),
            },
            "generate_max_length",
        ),
        (
            {
                "maxent_alpha": 0.002,
                "maxent_length_target": 16,
                "maxent_length_lambda_init": 0.03,
                "xdr_tau": float("inf"),
            },
            "lambda_init",
        ),
        (
            {
                "maxent_alpha": 0.002,
                "maxent_length_target": 16,
                "ignore_no_eos": True,
                "xdr_tau": float("inf"),
            },
            "no-EOS",
        ),
    ],
)
def test_length_constrained_maxent_rejects_invalid_or_confounded_settings(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


def test_canonical_pantry_policy_uses_exact_six_bit_entropy_bound():
    accepted = _args(
        canonical_action_task="pantry_support_mask",
        canonical_graph_action_count=6,
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        prompt_template="qwen_pantry_support_mask",
        test_split="multi_answer",
        xdr_tau=float("inf"),
        maxent_alpha=0.05,
        maxent_control_target_ratio=1.0,
        maxent_control_target_entropy=math.log(64),
    )
    assert validate_zero_math_args(accepted) is accepted

    rejected = _args(
        canonical_action_task="pantry_support_mask",
        canonical_graph_action_count=6,
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        prompt_template="qwen_pantry_support_mask",
        test_split="multi_answer",
        xdr_tau=float("inf"),
        maxent_alpha=0.05,
        maxent_control_target_ratio=1.0,
        maxent_control_target_entropy=math.log(64) + 1e-6,
    )
    with pytest.raises(ValueError, match="maximum log-support entropy"):
        validate_zero_math_args(rejected)


def test_canonical_graph_policy_accepts_the_preregistered_fixed_horizon():
    args = _args(
        canonical_graph_actions=True,
        canonical_graph_learner_sampling=True,
        canonical_graph_fixed_shape_sampling=True,
        prompt_template="qwen_graph_digits",
        test_split="multi_answer",
        xdr_tau=float("inf"),
        maxent_alpha=0.05,
    )

    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "canonical_graph_actions": True,
                "test_split": "multi_answer",
            },
            "qwen_graph_digits",
        ),
        (
            {
                "canonical_graph_actions": True,
                "prompt_template": "qwen_graph_digits",
            },
            "multi_answer",
        ),
        (
            {
                "canonical_graph_actions": True,
                "canonical_graph_action_count": 2,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
            },
            "action_count=3",
        ),
        (
            {
                "canonical_graph_actions": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
                "maxent_length_target": 3,
            },
            "fixed-horizon",
        ),
        (
            {
                "canonical_graph_actions": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
                "top_k": 1,
            },
            "top_k=-1",
        ),
        (
            {
                "canonical_graph_actions": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
                "temperature": 0,
            },
            "positive temperature",
        ),
        (
            {"canonical_graph_learner_sampling": True},
            "requires canonical_graph_actions",
        ),
        (
            {
                "canonical_graph_actions": True,
                "canonical_graph_learner_sampling": True,
                "canonical_graph_fixed_shape_sampling": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
                "rollout_batch_size": 2,
            },
            "rollout_batch_size=1",
        ),
        (
            {
                "canonical_graph_actions": True,
                "canonical_graph_learner_sampling": True,
                "prompt_template": "qwen_graph_digits",
                "test_split": "multi_answer",
            },
            "fixed-shape",
        ),
        (
            {"canonical_graph_fixed_shape_sampling": True},
            "requires canonical graph actions",
        ),
    ],
)
def test_canonical_graph_policy_rejects_mismatched_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


@pytest.mark.parametrize("controller", ["proportional", "dual"])
def test_canonical_countdown_adaptive_control_accepts_explicit_exact_target(
    controller,
):
    overrides = {
        "canonical_action_task": "countdown",
        "canonical_graph_learner_sampling": True,
        "canonical_graph_fixed_shape_sampling": True,
        "prompt_template": "qwen_countdown_digits",
        "test_split": "multi_answer",
        "xdr_tau": float("inf"),
        "maxent_alpha": 0.05,
    }
    if controller == "proportional":
        overrides.update(
            maxent_control_target_ratio=0.8,
            maxent_control_target_entropy=3.9741470618167156,
        )
    else:
        overrides.update(
            maxent_dual_target_ratio=0.8,
            maxent_dual_target_entropy=3.9741470618167156,
        )

    args = _args(**overrides)
    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize("controller", ["proportional", "dual"])
def test_canonical_adaptive_control_requires_an_explicit_target(controller):
    overrides = {
        "canonical_action_task": "countdown",
        "canonical_graph_learner_sampling": True,
        "canonical_graph_fixed_shape_sampling": True,
        "prompt_template": "qwen_countdown_digits",
        "test_split": "multi_answer",
        "xdr_tau": float("inf"),
        "maxent_alpha": 0.05,
    }
    overrides[
        "maxent_control_target_ratio"
        if controller == "proportional"
        else "maxent_dual_target_ratio"
    ] = 0.8

    with pytest.raises(ValueError, match="explicit positive"):
        validate_zero_math_args(_args(**overrides))


@pytest.mark.parametrize(
    ("task", "template", "target"),
    [
        ("graph_coloring", "qwen_graph_digits", math.log(27) + 1e-4),
        ("countdown", "qwen_countdown_digits", math.log(108) + 1e-4),
    ],
)
def test_canonical_entropy_target_cannot_exceed_log_support(task, template, target):
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_zero_math_args(
            _args(
                canonical_action_task=task,
                canonical_graph_learner_sampling=True,
                canonical_graph_fixed_shape_sampling=True,
                prompt_template=template,
                test_split="multi_answer",
                xdr_tau=float("inf"),
                maxent_alpha=0.05,
                maxent_control_target_ratio=0.8,
                maxent_control_target_entropy=target,
            )
        )


@pytest.mark.parametrize("tau", [-0.1, 1.1, float("nan")])
def test_ucpo_rejects_invalid_tau(tau):
    with pytest.raises(ValueError, match="ucpo_tau"):
        validate_zero_math_args(_args(xdr_tau=float("inf"), ucpo_tau=tau))


def test_rlep_dr_accepts_one_fresh_16_row_group_and_two_replays():
    args = _args(
        xdr_tau=float("inf"),
        num_samples=16,
        rlep_experience_root="/frozen/pool",
        rlep_replay_count=2,
    )
    assert validate_zero_math_args(args) is args


def test_rlep_dr_accepts_sparse_prompt_fallback_when_pool_is_enabled():
    args = _args(
        xdr_tau=float("inf"),
        num_samples=16,
        rlep_experience_root="/frozen/pool",
        rlep_replay_count=2,
        rlep_sparse_fallback=True,
    )
    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rlep_experience_root": "/pool", "rlep_replay_count": 0}, "enabled together"),
        ({"rlep_experience_root": "", "rlep_replay_count": 2}, "enabled together"),
        ({"rlep_experience_root": "/pool", "rlep_replay_count": -1}, "non-negative"),
        (
            {
                "critic_type": "grpo",
                "rlep_experience_root": "/pool",
                "rlep_replay_count": 2,
            },
            "requires critic_type=drgrpo",
        ),
        (
            {
                "num_samples": 32,
                "rlep_experience_root": "/pool",
                "rlep_replay_count": 2,
            },
            "one fresh 16-row",
        ),
        (
            {
                "num_samples": 16,
                "ucpo_tau": 0.2,
                "rlep_experience_root": "/pool",
                "rlep_replay_count": 2,
            },
            "separate comparative baselines",
        ),
        (
            {
                "num_samples": 16,
                "online_canonical_replay": True,
                "rlep_experience_root": "/pool",
                "rlep_replay_count": 2,
            },
            "cannot be combined with canonical replay",
        ),
    ],
)
def test_rlep_rejects_mixed_or_malformed_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(xdr_tau=float("inf"), **overrides))


def test_rlep_online_pool_takes_a_count_and_no_experience_root():
    args = _args(
        xdr_tau=float("inf"),
        num_samples=16,
        rlep_replay_count=2,
        rlep_online_pool=True,
        rlep_sparse_fallback=True,
    )
    assert validate_zero_math_args(args) is args
    with pytest.raises(ValueError, match="no rlep_experience_root"):
        validate_zero_math_args(
            _args(
                xdr_tau=float("inf"),
                num_samples=16,
                rlep_experience_root="/pool",
                rlep_replay_count=2,
                rlep_online_pool=True,
            )
        )
    with pytest.raises(ValueError, match="positive rlep_replay_count"):
        validate_zero_math_args(
            _args(xdr_tau=float("inf"), num_samples=16, rlep_online_pool=True)
        )



def test_semantic_verified_support_can_include_replay_membership():
    args = _args(
        xdr_tau=float("inf"),
        semantic_shannon_coef=0.1,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_verified_support_advantage=True,
        semantic_shannon_verified_support_include_replay_bank=True,
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        test_split="multi_answer",
        num_samples=16,
    )

    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "semantic_shannon_verified_support_include_replay_bank": True,
            },
            "requires verified-support mode",
        ),
        (
            {
                "semantic_shannon_coef": 0.1,
                "semantic_shannon_separate_advantage": True,
                "semantic_shannon_success_conditioned_verified_support_advantage": True,
                "semantic_shannon_verified_support_include_replay_bank": True,
            },
            "requires online canonical replay",
        ),
    ],
)
def test_semantic_verified_support_replay_membership_fails_closed(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_args(**overrides))


def _dapo_args(**overrides):
    values = {
        "critic_type": "grpo",
        "num_samples": 16,
        "xdr_tau": float("inf"),
        "dapo_enabled": True,
        "dapo_clip_low": 0.20,
        "dapo_clip_high": 0.28,
        "dapo_max_num_gen_batches": 10,
        "dapo_overlong_buffer_ratio": 0.20,
        "dapo_overlong_penalty_factor": 1.0,
        "train_batch_size": 16,
        "train_batch_size_per_device": 16,
        "verified_discovery_tracking": False,
    }
    values.update(overrides)
    return _args(**values)


def test_dapo_accepts_complete_isolated_configuration():
    args = _dapo_args()

    assert validate_zero_math_args(args) is args


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"critic_type": "drgrpo"}, "requires critic_type=grpo"),
        ({"ucpo_tau": 0.2}, "separate comparative baselines"),
        ({"verified_discovery_tracking": True}, "incompatible components"),
        ({"online_canonical_replay": True}, "incompatible components"),
        ({"train_batch_size": 32}, "complete rollout group"),
        (
            {
                "rlep_experience_root": "/frozen/pool",
                "rlep_replay_count": 2,
            },
            "separate comparative baselines",
        ),
    ],
)
def test_dapo_rejects_mixed_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_dapo_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dapo_clip_low": 0.0}, "DAPO clipping"),
        ({"dapo_clip_high": 1.0}, "DAPO clipping"),
        ({"dapo_clip_low": 0.3, "dapo_clip_high": 0.2}, "DAPO clipping"),
        ({"dapo_max_num_gen_batches": 0}, "must be positive"),
        ({"dapo_overlong_buffer_ratio": 1.0}, "must be finite and in"),
        ({"dapo_overlong_penalty_factor": -1.0}, "must be finite and non-negative"),
    ],
)
def test_dapo_rejects_malformed_recipe(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_zero_math_args(_dapo_args(**overrides))
