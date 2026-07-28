"""Learner initialization for the single Dr.GRPO/xDr.GRPO path."""

from __future__ import annotations

import functools
import logging
import math
import os
from typing import List

from datasets import load_from_disk
from oat.actors.base import ActorBase
from oat.utils.ops import masked_sum

from ..args import ZeroMathArgs, resolve_canonical_action_task
from ..answer_options import AnswerOptionMITracker
from ..canonical_actions import resolve_canonical_action_space
from ..canonical_replay import (
    CanonicalReplayInverseController,
    CanonicalReplayLikelihoodController,
)
from ..maxent_controllers import (
    MaxEntDualController,
    MaxEntInverseController,
    MaxEntProportionalController,
)
from ..maxent_length_controller import MaxEntLengthController
from ..math_strategy_canonicalizer import MathStrategyCanonicalizer
from ..online_canonical_bank import OnlineCanonicalBank
from ..online_canonical_controller import (
    OnlineCanonicalDualController,
    OnlineCanonicalPolicyEntropyController,
)
from ..runtime import patch_oat_learner_datetime, resolve_fixed_oat_exp_suffix
from ..semantic_shannon import SemanticShannonTracker
from ..trajectory_dataset import ZeroMathTrajectoryDataset
from ..xdr_tau_controller import XdrTauController
from ..xdr_sac_dual_controller import XdrSacDualController


def build_maxent_controllers(
    args: ZeroMathArgs,
) -> tuple[
    MaxEntProportionalController
    | MaxEntDualController
    | MaxEntInverseController
    | None,
    MaxEntLengthController | None,
]:
    """Construct the independent entropy-alpha and expected-length controls."""

    canonical_task = resolve_canonical_action_task(args)
    canonical_max_entropy = (
        math.log(27 if canonical_task == "graph_coloring" else 108)
        if canonical_task != "none"
        else None
    )
    if canonical_task != "none":
        configured_target = (
            float(args.maxent_control_target_entropy)
            if float(args.maxent_control_target_ratio) > 0
            else float(args.maxent_dual_target_entropy)
            if float(args.maxent_dual_target_ratio) > 0
            else None
        )
        if configured_target is not None and configured_target <= 0:
            raise ValueError(
                "adaptive canonical MaxEnt requires an explicit positive "
                "configured entropy target"
            )
        if (
            configured_target is not None
            and canonical_max_entropy is not None
            and configured_target > canonical_max_entropy + 1e-9
        ):
            raise ValueError(
                "canonical configured entropy target exceeds exact log-support "
                f"maximum {canonical_max_entropy:.12g}"
            )
    if canonical_task != "none":
        controller_units = "canonical_action_nats_exact_v1"
        controller_metric = "canonical_exact_sequence_entropy"
    elif str(getattr(args, "maxent_objective", "sequence")) == (
        "conditional_token_mean"
    ):
        controller_units = "conditional_content_token_nats_mean_v1"
        controller_metric = "maxent_conditional_token_entropy"
    else:
        controller_units = "sequence_nats_v1"
        controller_metric = "maxent_sequence_entropy"
    alpha_controller: (
        MaxEntProportionalController
        | MaxEntDualController
        | MaxEntInverseController
        | None
    ) = None
    if float(args.maxent_control_target_ratio) > 0:
        alpha_controller = MaxEntProportionalController(
            base_alpha=float(args.maxent_alpha),
            max_alpha=float(args.maxent_control_max_alpha),
            target_ratio=float(args.maxent_control_target_ratio),
            warmup_steps=int(args.maxent_control_warmup_steps),
            ema_decay=float(args.maxent_control_ema_decay),
            gain=float(args.maxent_control_gain),
            configured_target_entropy=float(args.maxent_control_target_entropy),
            entropy_units=controller_units,
            observation_metric_key=controller_metric,
        )
    elif float(args.maxent_dual_target_ratio) > 0:
        alpha_controller = MaxEntDualController(
            base_alpha=float(args.maxent_alpha),
            min_alpha=float(args.maxent_dual_min_alpha),
            max_alpha=float(args.maxent_dual_max_alpha),
            target_ratio=float(args.maxent_dual_target_ratio),
            warmup_steps=int(args.maxent_dual_warmup_steps),
            alpha_lr=float(args.maxent_dual_alpha_lr),
            ema_decay=float(args.maxent_dual_ema_decay),
            configured_target_entropy=float(args.maxent_dual_target_entropy),
            entropy_units=controller_units,
            observation_metric_key=controller_metric,
        )
    elif bool(getattr(args, "maxent_inverse_adaptation", False)):
        alpha_controller = MaxEntInverseController(
            base_alpha=float(args.maxent_alpha),
            warmup_steps=int(args.maxent_inverse_warmup_steps),
            ema_decay=float(args.maxent_inverse_ema_decay),
            entropy_units=controller_units,
            observation_metric_key=controller_metric,
        )

    length_controller = None
    if float(args.maxent_length_target) > 0:
        length_controller = MaxEntLengthController(
            target_length=float(args.maxent_length_target),
            init_lambda=float(args.maxent_length_lambda_init),
            max_lambda=float(args.maxent_length_lambda_max),
            dual_lr=float(args.maxent_length_dual_lr),
            ema_decay=float(args.maxent_length_ema_decay),
        )
    return alpha_controller, length_controller


class ZeroMathInitMixin:
    """Initialize OAT state, exact-answer data, and Dr.GRPO normalization."""

    def _init(self, args: ZeroMathArgs, actors: List[ActorBase]) -> None:
        requested_use_wb = args.use_wb
        args.use_wb = False
        fixed_exp_suffix = resolve_fixed_oat_exp_suffix()
        if fixed_exp_suffix:
            logging.info("Using fixed OAT experiment suffix %s", fixed_exp_suffix)
        with patch_oat_learner_datetime(fixed_exp_suffix):
            super()._init(args, actors)

        self.dataset_builder = ZeroMathTrajectoryDataset
        args.use_wb = requested_use_wb
        if hasattr(self, "strategy") and hasattr(self.strategy, "args"):
            self.strategy.args.use_wb = requested_use_wb
        self.eval_dataset_dict = load_from_disk(args.eval_data)
        if args.test_split != "all":
            self.eval_dataset_dict = {
                key: value
                for key, value in self.eval_dataset_dict.items()
                if key in args.test_split
            }
        self.args = args
        self._requested_use_wb = requested_use_wb
        self._wandb = None
        self._wandb_run_id: str | None = None
        self._wandb_run_name = os.path.basename(self.save_path)
        self.masked_aggregator = functools.partial(
            masked_sum, constant_normalizer=args.generate_max_length
        )
        self._baseline_grad_norm_logging_disabled_warned = False
        self._invalid_scoring_token_ids_warned_contexts = set()
        self._invalid_logit_columns_warned_contexts = set()
        self._prompt_batches_consumed_total = 0
        self._canonical_action_space = None
        self._canonical_action_token_ids: tuple[int, ...] | None = None
        self._canonical_action_token_ids_by_position: (
            tuple[tuple[int, ...], ...] | None
        ) = None
        canonical_task = resolve_canonical_action_task(args)
        if canonical_task != "none":
            self._canonical_action_space = resolve_canonical_action_space(
                self.tokenizer, canonical_task
            )
            self._canonical_action_token_ids = (
                self._canonical_action_space.union_token_ids
            )
            self._canonical_action_token_ids_by_position = (
                self._canonical_action_space.token_ids_by_position
            )
            logging.info(
                "canonical %s policy: position_token_ids=%s sequence_count=%d "
                "max_entropy=%.9f horizon=%d "
                "tokenizer=%s tokenizer_class=%s vocab_size=%s",
                canonical_task,
                self._canonical_action_token_ids_by_position,
                self._canonical_action_space.sequence_count,
                self._canonical_action_space.max_sequence_entropy,
                self._canonical_action_space.horizon,
                getattr(self.tokenizer, "name_or_path", args.pretrain),
                type(self.tokenizer).__name__,
                len(self.tokenizer),
            )
        self._xdr_tau_controller: XdrTauController | None = None
        controller_base_tau = float(args.xdr_tau)
        if float(args.xdr_tau_control_target_ratio) > 0:
            self._xdr_tau_controller = XdrTauController(
                base_tau=controller_base_tau,
                min_tau=float(args.xdr_tau_control_min),
                target_ratio=float(args.xdr_tau_control_target_ratio),
                warmup_steps=int(args.xdr_tau_control_warmup_steps),
                ema_decay=float(args.xdr_tau_control_ema_decay),
                gain=float(args.xdr_tau_control_gain),
            )
        elif float(args.xdr_sac_dual_target_ratio) > 0:
            self._xdr_tau_controller = XdrSacDualController(
                base_tau=controller_base_tau,
                min_tau=float(args.xdr_sac_dual_min_tau),
                max_tau=float(args.xdr_sac_dual_max_tau),
                target_ratio=float(args.xdr_sac_dual_target_ratio),
                warmup_steps=int(args.xdr_sac_dual_warmup_steps),
                alpha_lr=float(args.xdr_sac_dual_alpha_lr),
            )
        (
            self._maxent_alpha_controller,
            self._maxent_length_controller,
        ) = build_maxent_controllers(args)
        if isinstance(self._maxent_alpha_controller, MaxEntInverseController):
            logging.info(
                "direct inverse MaxEnt control enabled: "
                "reference_alpha=%.9g warmup_steps=%d ema_decay=%.6g "
                "sensor=%s units=%s "
                "rule=unprojected_warmup_inverse_direct_entropy_v1",
                float(args.maxent_alpha),
                int(args.maxent_inverse_warmup_steps),
                float(args.maxent_inverse_ema_decay),
                self._maxent_alpha_controller.observation_metric_key,
                self._maxent_alpha_controller.entropy_units,
            )
        self._diayn_mi_tracker: AnswerOptionMITracker | None = None
        if int(getattr(args, "diayn_num_options", 0) or 0) > 1:
            self._diayn_mi_tracker = AnswerOptionMITracker(
                num_options=int(args.diayn_num_options),
                ema_decay=float(args.diayn_mi_ema_decay),
                smoothing=float(args.diayn_mi_smoothing),
                bonus_clip=float(args.diayn_mi_bonus_clip),
                leave_one_out=bool(args.diayn_mi_leave_one_out),
            )
            logging.info(
                "DIAYN answer-option MI enabled: options=%s beta=%.6g "
                "ema_decay=%.3f correct_only=%s leave_one_out=%s",
                int(args.diayn_num_options),
                float(args.diayn_mi_beta),
                float(args.diayn_mi_ema_decay),
                bool(args.diayn_mi_correct_only),
                bool(args.diayn_mi_leave_one_out),
            )
        self._semantic_shannon_tracker: SemanticShannonTracker | None = None
        semantic_shannon_coef = float(
            getattr(args, "semantic_shannon_coef", 0.0) or 0.0
        )
        if semantic_shannon_coef > 0:
            self._semantic_shannon_tracker = SemanticShannonTracker(
                coefficient=semantic_shannon_coef,
                surprisal_clip=float(args.semantic_shannon_surprisal_clip),
                pseudocount=float(args.semantic_shannon_pseudocount),
                quality_gated_advantage=bool(
                    getattr(
                        args,
                        "semantic_shannon_quality_gated_advantage",
                        False,
                    )
                ),
                quality_gated_cap=float(
                    getattr(args, "semantic_shannon_quality_gated_cap", 0.05)
                ),
                success_conditioned_signed_advantage=bool(
                    getattr(
                        args,
                        "semantic_shannon_success_conditioned_signed_advantage",
                        False,
                    )
                ),
                success_conditioned_signed_cap=float(
                    getattr(
                        args,
                        "semantic_shannon_success_conditioned_signed_cap",
                        0.05,
                    )
                ),
                open_set_inverse_adaptation=bool(
                    getattr(
                        args,
                        "semantic_shannon_open_set_inverse_adaptation",
                        False,
                    )
                ),
                open_set_warmup_steps=int(
                    getattr(
                        args,
                        "semantic_shannon_open_set_warmup_steps",
                        64,
                    )
                ),
                open_set_ema_decay=float(
                    getattr(
                        args,
                        "semantic_shannon_open_set_ema_decay",
                        0.9,
                    )
                ),
            )
            logging.info(
                "semantic Shannon shaping enabled: coefficient=%.6g "
                "surprisal_clip=%.6g pseudocount=%.6g "
                "separate_advantage=%s quality_gated_advantage=%s "
                "quality_gated_cap=%.6g "
                "success_conditioned_signed_advantage=%s "
                "success_conditioned_signed_cap=%.6g "
                "open_set_inverse_adaptation=%s "
                "open_set_warmup_steps=%s open_set_ema_decay=%.6g "
                "estimator=predictive_prompt_counts_loo_unseen_v1",
                semantic_shannon_coef,
                float(args.semantic_shannon_surprisal_clip),
                float(args.semantic_shannon_pseudocount),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_separate_advantage",
                        False,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_quality_gated_advantage",
                        False,
                    )
                ),
                float(
                    getattr(args, "semantic_shannon_quality_gated_cap", 0.05)
                ),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_success_conditioned_signed_advantage",
                        False,
                    )
                ),
                float(
                    getattr(
                        args,
                        "semantic_shannon_success_conditioned_signed_cap",
                        0.05,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "semantic_shannon_open_set_inverse_adaptation",
                        False,
                    )
                ),
                int(
                    getattr(
                        args,
                        "semantic_shannon_open_set_warmup_steps",
                        64,
                    )
                ),
                float(
                    getattr(
                        args,
                        "semantic_shannon_open_set_ema_decay",
                        0.9,
                    )
                ),
            )
        self._online_canonical_bank: OnlineCanonicalBank | None = None
        self._math_strategy_canonicalizer: (
            MathStrategyCanonicalizer | None
        ) = None
        self._online_canonical_alpha_controller: (
            OnlineCanonicalDualController
            | OnlineCanonicalPolicyEntropyController
            | None
        ) = None
        self._canonical_replay_controller: (
            CanonicalReplayInverseController | None
        ) = None
        self._canonical_replay_mass_controller: (
            CanonicalReplayLikelihoodController | None
        ) = None
        online_canonical_bank_alpha = float(
            getattr(args, "online_canonical_bank_alpha", 0.0) or 0.0
        )
        online_canonical_novelty_beta = float(
            getattr(args, "online_canonical_novelty_beta", 0.0) or 0.0
        )
        online_canonical_replay = bool(
            getattr(args, "online_canonical_replay", False)
        )
        online_canonical_objective_active = (
            online_canonical_bank_alpha > 0.0
            or online_canonical_novelty_beta > 0.0
            or online_canonical_replay
        )
        verified_discovery_tracking = bool(
            getattr(args, "verified_discovery_tracking", True)
        )
        if (
            online_canonical_objective_active
            or verified_discovery_tracking
        ):
            self._online_canonical_bank = OnlineCanonicalBank(
                entropy_alpha=online_canonical_bank_alpha,
                novelty_beta=online_canonical_novelty_beta,
                pseudocount=float(args.online_canonical_bank_pseudocount),
                surprisal_clip=float(
                    args.online_canonical_bank_surprisal_clip
                ),
                retain_exemplars=online_canonical_replay,
                replay_capacity=int(
                    args.online_canonical_replay_capacity
                ),
                global_replay_groups_per_step=int(
                    args.online_canonical_replay_global_groups_per_step
                ),
                global_replay_bootstrap_steps=int(
                    args.online_canonical_replay_global_bootstrap_steps
                ),
                separate_proposal_objective_support=bool(
                    getattr(
                        args,
                        "online_canonical_counterfactual_separate_objective_support",
                        False,
                    )
                ),
            )
            if online_canonical_objective_active:
                logging.info(
                    "online canonical bank enabled: alpha=%.6g beta=%.6g "
                    "pseudocount=%.6g surprisal_clip=%.6g key_mode=%s "
                    "estimator=verified_prompt_bank_loo_v1",
                    online_canonical_bank_alpha,
                    online_canonical_novelty_beta,
                    float(args.online_canonical_bank_pseudocount),
                    float(args.online_canonical_bank_surprisal_clip),
                    str(args.online_canonical_key_mode),
                )
            else:
                logging.info(
                    "passive verified-discovery tracking enabled for ordinary "
                    "Dr.GRPO: key_mode=%s objective_influence=zero",
                    str(args.online_canonical_key_mode),
                )
            if online_canonical_replay:
                self._canonical_replay_controller = (
                    CanonicalReplayInverseController(
                        base_alpha=float(
                            args.online_canonical_replay_alpha
                        ),
                        warmup_steps=int(
                            args.online_canonical_replay_warmup_steps
                        ),
                        ema_decay=float(
                            args.online_canonical_replay_ema_decay
                        ),
                    )
                )
                if str(args.online_canonical_replay_objective) == (
                    "split_mass_balance_per_rollout"
                ):
                    self._canonical_replay_mass_controller = (
                        CanonicalReplayLikelihoodController(
                            base_alpha=float(
                                args.online_canonical_replay_mass_alpha
                            ),
                            warmup_steps=int(
                                args.online_canonical_replay_mass_warmup_steps
                            ),
                            ema_decay=float(
                                args.online_canonical_replay_mass_ema_decay
                            ),
                        )
                    )
                logging.info(
                    "verified canonical replay enabled: "
                    "reference_alpha=%.6g capacity=%d "
                    "global_groups_per_step=%d "
                    "global_bootstrap_steps=%d "
                    "warmup_steps=%d ema_decay=%.6g "
                    "sensor=normalized_model_entropy_over_observed_bank_v1 "
                    "actuator=%s "
                    "projection=none gold_support_feedback=none",
                    float(args.online_canonical_replay_alpha),
                    int(args.online_canonical_replay_capacity),
                    int(
                        args.online_canonical_replay_global_groups_per_step
                    ),
                    int(
                        args.online_canonical_replay_global_bootstrap_steps
                    ),
                    int(args.online_canonical_replay_warmup_steps),
                    float(args.online_canonical_replay_ema_decay),
                    (
                        "KL_uniform_to_model_exemplar_scores_v1"
                        if str(args.online_canonical_replay_objective)
                        == "bank_balance"
                        else (
                            "uniform_verified_exemplar_likelihood_v1"
                            if str(args.online_canonical_replay_objective)
                            == "verified_likelihood"
                            else (
                                "split_verified_mass_and_bank_balance_"
                                "per_rollout_v1"
                                if str(
                                    args.online_canonical_replay_objective
                                )
                                == "split_mass_balance_per_rollout"
                                else (
                                    "uniform_verified_exemplar_likelihood_"
                                    "per_rollout_with_singleton_anchor_v1"
                                )
                            )
                        )
                    ),
                )
                if bool(
                    getattr(
                        args,
                        "online_canonical_counterfactual_proposals",
                        False,
                    )
                ):
                    logging.info(
                        "verified open-set proposals enabled: "
                        "proposal_groups_per_eligible_prompt=up_to_%d "
                        "proposal_width=num_samples "
                        "original_prompt_temperature_sweep=%.6g,+0.2/attempt "
                        "admission=novel_validator_and_task_positive_only "
                        "proposal_rows_to_ppo=0 "
                        "objective_support_separated=%s "
                        "gold_support_feedback=none "
                        "desired_mode_count_feedback=none "
                        "evaluation_feedback=none",
                        int(
                            args
                            .online_canonical_counterfactual_max_attempts
                        ),
                        float(
                            args
                            .online_canonical_counterfactual_sampling_temperature
                        ),
                        bool(
                            getattr(
                                args,
                                "online_canonical_counterfactual_separate_objective_support",
                                False,
                            )
                        ),
                    )
            if str(args.online_canonical_key_mode) == "math_strategy_qwen72":
                self._math_strategy_canonicalizer = MathStrategyCanonicalizer(
                    endpoint=str(args.math_strategy_endpoint),
                    model=str(args.math_strategy_model),
                    timeout_seconds=int(args.math_strategy_timeout_seconds),
                    max_workers=int(args.math_strategy_workers),
                    max_item_chars=int(args.math_strategy_max_item_chars),
                    allow_unstructured_menu_inference=bool(
                        getattr(
                            args,
                            "math_strategy_allow_unstructured_inference",
                            False,
                        )
                    ),
                )
                logging.info(
                    "validator-bound MATH strategy canonicalizer enabled: "
                    "model=%s endpoint=%s integrity_passes=2 "
                    "runtime_partition_passes=0 "
                    "runtime_pairwise_veto_passes=0 "
                    "seeds=470721,470722 "
                    "novelty_rule=exact_precalibrated_menu_combo_v18 "
                    "fail_closed task_reward_gate=%s "
                    "unstructured_menu_inference=%s",
                    str(args.math_strategy_model),
                    str(args.math_strategy_endpoint),
                    bool(
                        getattr(
                            args,
                            "math_strategy_gate_task_reward",
                            False,
                        )
                    ),
                    bool(
                        getattr(
                            args,
                            "math_strategy_allow_unstructured_inference",
                            False,
                        )
                    ),
                )
            online_canonical_dual_target = float(
                getattr(
                    args, "online_canonical_dual_target_ratio", 0.0
                )
                or 0.0
            )
            if online_canonical_dual_target > 0:
                self._online_canonical_alpha_controller = (
                    OnlineCanonicalDualController(
                        base_alpha=online_canonical_bank_alpha,
                        min_alpha=float(
                            args.online_canonical_dual_min_alpha
                        ),
                        max_alpha=float(
                            args.online_canonical_dual_max_alpha
                        ),
                        target_ratio=online_canonical_dual_target,
                        alpha_lr=float(
                            args.online_canonical_dual_alpha_lr
                        ),
                        ema_decay=float(
                            args.online_canonical_dual_ema_decay
                        ),
                    )
                )
                logging.info(
                    "online canonical Haarnoja control enabled: "
                    "normalized_target=%.6g base_alpha=%.6g "
                    "min_alpha=%.6g max_alpha=%.6g alpha_lr=%.6g "
                    "ema_decay=%.6g "
                    "sensor=exact_postupdate_H_over_log_support_v1",
                    online_canonical_dual_target,
                    online_canonical_bank_alpha,
                    float(args.online_canonical_dual_min_alpha),
                    float(args.online_canonical_dual_max_alpha),
                    float(args.online_canonical_dual_alpha_lr),
                    float(args.online_canonical_dual_ema_decay),
                )
            elif bool(
                getattr(
                    args,
                    "online_canonical_policy_entropy_adaptation",
                    False,
                )
            ):
                self._online_canonical_alpha_controller = (
                    OnlineCanonicalPolicyEntropyController(
                        base_alpha=online_canonical_bank_alpha,
                        warmup_steps=int(
                            args.online_canonical_policy_entropy_warmup_steps
                        ),
                        ema_decay=float(
                            args.online_canonical_policy_entropy_ema_decay
                        ),
                    )
                )
                logging.info(
                    "online canonical policy-entropy adaptation enabled: "
                    "reference_alpha=%.6g warmup_steps=%d "
                    "ema_decay=%.6g "
                    "sensor=masked_mean_policy_token_entropy_v1 "
                    "rule=unprojected_relative_to_own_warmup_v1",
                    online_canonical_bank_alpha,
                    int(
                        args.online_canonical_policy_entropy_warmup_steps
                    ),
                    float(
                        args.online_canonical_policy_entropy_ema_decay
                    ),
                )
