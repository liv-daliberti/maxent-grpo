"""Baseline GRPO learner helpers."""

from __future__ import annotations

import gc
import logging
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from oat.utils.ops import masked_mean

from ..args import resolve_canonical_action_task
from ..answer_options import coerce_option_id, conditional_answer_repr
from ..canonical_actions import (
    canonical_behavior_overlap_diagnostics,
    materialize_canonical_behavior_policy,
    materialize_position_canonical_behavior_policy,
    restricted_action_log_probs_entropy_and_distribution,
    restricted_position_action_log_probs_entropy_and_distribution,
)
from ..canonical_replay import (
    CanonicalReplayBatch,
    canonical_replay_split_mass_balance_loss,
    canonical_replay_uniform_loss,
    canonical_replay_uniform_verified_likelihood_loss,
    materialize_canonical_replay_batch,
)
from ..math_grader import (
    extract_normalized_final_answer,
    validated_exploration_identity,
    validated_modebench_outcome_key,
)
from ..math_strategy_canonicalizer import MathStrategyCanonicalizer
from ..on_policy_maxent import (
    mean_active_token_entropy_by_response,
    prefix_ratio_expected_length_surrogate,
    prefix_ratio_maxent_surrogate,
    standard_maxent_length_penalty_loss,
    standard_maxent_loss,
)
from ..online_canonical_bank import (
    OnlineCanonicalBank,
    VerifiedCanonicalReplayGroup,
)
from ..replicated_group import (
    replicated_group_permutation_seed,
    validate_replicated_group_layout,
)
from ..outcome_collision import (
    add_outcome_collision_outside_centering_advantage,
    compute_outcome_collision_bonuses,
)
from ..semantic_shannon import (
    SemanticShannonTracker,
    add_semantic_shannon_separate_advantage,
)
from ..seed_weights import compute_seed_row_weights
from ..tensor_utils import cap_last_valid_token_pos_for_zero_advantage
from ..verified_route_library import VerifiedRouteLibrary
from ..xdr import aggregation_group_diagnostics, compute_xdr_row_weights


MATH_VERIFIED_ANSWER_OUTCOME_KEY = "math_verified_answer:correct"


def math_verified_answer_outcome_keys(
    reward_positive: list[bool],
) -> list[str | None]:
    """Collapse validator-positive MATH answers to one prompt-local outcome."""

    return [
        MATH_VERIFIED_ANSWER_OUTCOME_KEY if positive else None
        for positive in reward_positive
    ]


def apply_math_strategy_task_reward_gate(
    final_rewards: torch.Tensor,
    task_final_rewards: torch.Tensor,
    admitted: list[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero both learning reward views for non-admitted strategy executions."""

    if final_rewards.shape != task_final_rewards.shape:
        raise ValueError("MATH strategy reward views differ in shape")
    if final_rewards.numel() != len(admitted):
        raise ValueError("MATH strategy admission mask differs in length")
    contract_mask = torch.tensor(
        admitted,
        dtype=final_rewards.dtype,
        device=final_rewards.device,
    ).reshape_as(final_rewards)
    return (
        final_rewards * contract_mask,
        task_final_rewards * contract_mask,
    )


@contextmanager
def _temporary_eval_mode(model: torch.nn.Module, *, enabled: bool):
    """Temporarily disable training-only model behavior and restore it exactly."""

    was_training = bool(model.training)
    if enabled:
        model.eval()
    try:
        yield
    finally:
        if enabled:
            model.train(was_training)


class ZeroMathGrpoMixin:
    """Dr.GRPO update with optional candidate or control-arm weighting."""

    def _use_instrumented_grpo_learning_step(self) -> bool:
        return int(getattr(self.args, "zero_stage", 0) or 0) >= 3 or bool(
            getattr(self.args, "adam_offload", False)
        )

    def _seed_answer_keys_grouped(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        references_grouped: list[list[Any | None]] | None = None,
    ) -> list[list[str | None]]:
        """Extract canonical final-answer keys for the SEED control."""

        label_ids = input_ids[:, 1:]
        rows = []
        for row_ids, row_mask in zip(label_ids, response_masks):
            token_ids = row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()
            rows.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        grouped_rows = [
            rows[index : index + group_size]
            for index in range(0, len(rows), group_size)
        ]
        keys_grouped = []
        for prompt_index, prompt_rows in enumerate(grouped_rows):
            refs = (
                [None] * len(prompt_rows)
                if references_grouped is None
                else references_grouped[prompt_index]
            )
            keys_grouped.append(
                [
                    extract_normalized_final_answer(
                        text,
                        template=str(self.args.prompt_template),
                        gt_answer=refs[row_index],
                    )
                    for row_index, text in enumerate(prompt_rows)
                ]
            )
        return keys_grouped

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

    def _materialize_canonical_replay(
        self,
        groups: list[VerifiedCanonicalReplayGroup],
        *,
        device: torch.device | int,
    ) -> CanonicalReplayBatch:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("canonical replay requires a tokenizer pad or EOS token")
        return materialize_canonical_replay_batch(
            groups,
            pad_token_id=int(pad_token_id),
            device=device,
        )

    def _score_canonical_replay_rows(
        self,
        replay: CanonicalReplayBatch,
        *,
        start: int,
        stop: int,
        policy_vocab_upper_bound: int,
    ) -> torch.Tensor:
        """Return length-neutral model scores for a memory-safe row chunk."""

        replay_input_ids = self._sanitize_scoring_token_ids(
            replay.input_ids[start:stop],
            upper_bound=policy_vocab_upper_bound,
            context="canonical_replay_policy_input",
        )
        replay_logits = self.model(
            replay_input_ids,
            attention_mask=replay.attention_mask[start:stop],
        )["logits"]
        if self.args.temperature != 1:
            replay_logits = replay_logits / self.args.temperature
        replay_logits = self._mask_invalid_scoring_logit_columns(
            replay_logits,
            valid_vocab_size=policy_vocab_upper_bound,
            context="canonical_replay_policy_logits",
        )
        replay_logps, _ = self._policy_logps_and_optional_entropy(
            replay_logits,
            replay_input_ids,
            replay.response_masks[start:stop],
            need_entropy=False,
        )
        replay_mask = replay.response_masks[start:stop].to(replay_logps.dtype)
        token_counts = replay_mask.sum(dim=1)
        if not bool(token_counts.gt(0).all()):
            raise RuntimeError("canonical replay materialized an empty response")
        return (replay_logps * replay_mask).sum(dim=1) / token_counts

    def _baseline_update_with_precomputed_advantages(
        self,
        *,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        prompt_id_lens,
        loss_masks: torch.Tensor,
        response_masks: torch.Tensor,
        logps: torch.Tensor,
        ref_logps: torch.Tensor | None,
        advantages: torch.Tensor,
        final_rewards: torch.Tensor,
        returns: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        policy_vocab_upper_bound: int | None = None,
        row_weights: torch.Tensor | None = None,
        extra_infos: dict[str, torch.Tensor] | None = None,
        canonical_replay_groups: (list[VerifiedCanonicalReplayGroup] | None) = None,
    ) -> dict[str, torch.Tensor]:
        args = self.args
        canonical_task = resolve_canonical_action_task(args)
        canonical_actions = canonical_task != "none"
        infos: dict[str, torch.Tensor] = {}
        if extra_infos:
            infos.update(extra_infos)
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if policy_vocab_upper_bound is None:
            policy_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(
                self.model
            )

        replicated_group = bool(
            getattr(args, "canonical_graph_learner_sampling", False)
            or getattr(args, "replicated_freeform_sampling", False)
        )
        learner_world_size = dist.get_world_size() if replicated_group else 1
        if replicated_group:
            if len(input_ids) != int(args.num_samples):
                raise RuntimeError(
                    "replicated update requires one complete candidate group"
                )
            try:
                replicated_layout = validate_replicated_group_layout(
                    num_samples=int(args.num_samples),
                    learner_world_size=learner_world_size,
                    train_batch_size=int(args.train_batch_size),
                    train_batch_size_per_device=int(args.train_batch_size_per_device),
                )
            except ValueError as error:
                raise RuntimeError(
                    f"invalid replicated update layout: {error}"
                ) from error
            local_candidate_count = replicated_layout.local_candidate_count
            if int(self.strategy.grad_acc_step) != int(
                replicated_layout.micro_batches_per_rank
            ):
                raise RuntimeError(
                    "replicated update accumulation width does not match the "
                    "exact logical candidate group"
                )
        else:
            local_candidate_count = len(input_ids)
        total_micro_batches = args.num_ppo_epochs * math.ceil(
            local_candidate_count / max(args.train_batch_size_per_device, 1)
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
        for ppo_epoch in range(args.num_ppo_epochs):
            if replicated_group:
                permutation_seed = replicated_group_permutation_seed(
                    experiment_seed=int(args.seed),
                    learner_step=int(getattr(self, "steps", 0)),
                    ppo_epoch=int(ppo_epoch),
                )
                permutation = np.random.RandomState(permutation_seed).permutation(
                    len(input_ids)
                )
                rank = dist.get_rank()
                shard_start = rank * local_candidate_count
                batch_inds = permutation[
                    shard_start : shard_start + local_candidate_count
                ]
            else:
                batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(batch_inds), args.train_batch_size_per_device):
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
                mb_row_weights = (
                    row_weights[mini_batch_inds] if row_weights is not None else None
                )

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
                    and float(getattr(args, "maxent_alpha", 0.0) or 0.0) <= 0
                    and self.args.critic_type in ["grpo", "drgrpo"]
                    and len(mb_advantage) == 1
                    and bool(torch.count_nonzero(mb_advantage).item() == 0)
                ):
                    prompt_len = int(prompt_id_lens[int(mini_batch_inds[0])])
                    mb_last_valid_token_pos = (
                        cap_last_valid_token_pos_for_zero_advantage(
                            prompt_len=prompt_len,
                            last_valid_token_pos=int(mb_last_valid_token_pos),
                            response_token_budget=int(
                                getattr(
                                    self.args, "baseline_zero_adv_response_tokens", 8
                                )
                            ),
                        )
                    )
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_logps = mb_logps[:, : mb_last_valid_token_pos - 1]

                if self.args.critic_type == "ppo":
                    if returns is None or values is None:
                        raise ValueError(
                            "ppo baseline updates require returns and values."
                        )
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
                maxent_objective = str(getattr(args, "maxent_objective", "sequence"))
                new_logps, policy_token_entropy = (
                    self._policy_logps_and_optional_entropy(
                        logits,
                        mb_input_ids,
                        mb_response_masks,
                        # ``train/entropy`` retains one full-policy definition
                        # across control and treatment arms. E21 computes its
                        # separately labeled conditional-content objective in
                        # addition to this shared diagnostic.
                        need_entropy=True,
                    )
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

                base_pg_loss = self.masked_aggregator(
                    pg_loss_max, mb_response_masks, axis=1
                )
                if mb_row_weights is None:
                    base_pg_loss = (base_pg_loss * mb_loss_masks).mean()
                else:
                    # xDr.GRPO: per-candidate tempered aggregation weights
                    # (G * softmax(U/tau) per prompt group, detached). Only
                    # the pg-loss aggregation is reweighted.
                    base_pg_loss = (
                        base_pg_loss * mb_loss_masks * mb_row_weights
                    ).mean()
                pg_loss = base_pg_loss
                infos["pg_loss"] = pg_loss.detach()
                loss = pg_loss
                maxent_alpha = float(getattr(args, "maxent_alpha", 0.0) or 0.0)
                maxent_controller = getattr(self, "_maxent_alpha_controller", None)
                if maxent_controller is not None:
                    maxent_alpha = float(maxent_controller.current_alpha)
                maxent_length_controller = getattr(
                    self, "_maxent_length_controller", None
                )
                maxent_length_lambda = (
                    float(maxent_length_controller.current_lambda)
                    if maxent_length_controller is not None
                    else 0.0
                )
                policy_entropy_coef = float(
                    getattr(args, "policy_entropy_coef", 0.0) or 0.0
                )
                entropy_for_loss = None
                token_entropy = policy_token_entropy
                if maxent_alpha > 0.0:
                    objective_token_entropy = token_entropy
                    if maxent_objective == "conditional_token_mean":
                        if canonical_actions:
                            raise RuntimeError(
                                "conditional-token MaxEnt cannot use a canonical policy"
                            )
                        objective_token_entropy = (
                            self._chunked_conditional_content_entropy_from_logits(
                                logits
                            )
                        )
                    if objective_token_entropy is None:
                        raise RuntimeError("policy entropy was not computed")
                    if objective_token_entropy.shape != mb_response_masks.shape:
                        objective_token_entropy = objective_token_entropy[
                            ..., : mb_response_masks.shape[-1]
                        ]
                    if maxent_objective == "conditional_token_mean":
                        # E21 deliberately does not importance-differentiate
                        # the sampled state distribution. Each response gets
                        # one equal-weight mean over active positions, and EOS
                        # has already been removed from the local categorical
                        # distribution. Thus neither extra states nor the EOS
                        # decision earn a direct entropy reward.
                        conditional_entropy_by_row = (
                            mean_active_token_entropy_by_response(
                                objective_token_entropy,
                                mb_response_masks,
                            )
                        )
                        entropy_surrogate_by_row = conditional_entropy_by_row
                        sequence_entropy_by_row = conditional_entropy_by_row.detach()
                        sampled_prefix_entropy_by_row = (
                            conditional_entropy_by_row.detach()
                        )
                        prefix_ratio_mean_by_row = torch.ones_like(
                            conditional_entropy_by_row
                        )
                        prefix_ratio_max_by_row = torch.ones_like(
                            conditional_entropy_by_row
                        )
                        prefix_ratio_clipfrac_by_row = torch.zeros_like(
                            conditional_entropy_by_row
                        )
                    else:
                        (
                            entropy_surrogate_by_row,
                            sequence_entropy_by_row,
                            sampled_prefix_entropy_by_row,
                            prefix_ratio_mean_by_row,
                            prefix_ratio_max_by_row,
                            prefix_ratio_clipfrac_by_row,
                        ) = prefix_ratio_maxent_surrogate(
                            objective_token_entropy,
                            new_logps,
                            mb_logps,
                            mb_response_masks,
                            # Standard MaxEnt uses raw completion-sequence entropy.
                            # Dr.GRPO's shared outer 1/T_max is applied below.
                            normalization_constant=1.0,
                            # The three-action canonical pilot is short enough to
                            # retain the exact direct-entropy identity. Legacy
                            # free-form smokes retain their preregistered clipped
                            # stability surrogate.
                            cliprange=(
                                None if canonical_actions else float(args.cliprange)
                            ),
                        )
                    active_rows = mb_loss_masks.sum().clamp_min(1.0)
                    entropy_surrogate = (
                        entropy_surrogate_by_row * mb_loss_masks
                    ).sum() / active_rows
                    sequence_entropy = (
                        sequence_entropy_by_row * mb_loss_masks
                    ).sum() / active_rows
                    sampled_prefix_entropy = (
                        sampled_prefix_entropy_by_row * mb_loss_masks
                    ).sum() / active_rows
                    prefix_ratio_mean = (
                        prefix_ratio_mean_by_row * mb_loss_masks
                    ).sum() / active_rows
                    prefix_ratio_max = (prefix_ratio_max_by_row * mb_loss_masks).max()
                    prefix_ratio_clipfrac = (
                        prefix_ratio_clipfrac_by_row * mb_loss_masks
                    ).sum() / active_rows
                    sequence_entropy_per_tmax = sequence_entropy / float(
                        args.generate_max_length
                    )
                    sampled_prefix_entropy_per_tmax = sampled_prefix_entropy / float(
                        args.generate_max_length
                    )
                    # Dr.GRPO divides its reward-policy gradient by T_max.
                    # The active objective is standard E[R] + alpha H, so raw
                    # sequence entropy receives exactly that one shared outer
                    # scale. Do not normalize H by T_max a second time.
                    # Its self-including group-mean reward baseline attenuates
                    # the expected reward gradient by (G-1)/G; applying the
                    # same factor here preserves alpha's stated objective
                    # units without changing the maintained Dr.GRPO update.
                    reward_estimator_scale = float(args.num_samples - 1) / float(
                        args.num_samples
                    )
                    entropy_loss = standard_maxent_loss(
                        entropy_surrogate,
                        alpha=maxent_alpha,
                        reward_estimator_scale=reward_estimator_scale,
                        update_normalizer=(
                            1.0
                            if maxent_objective == "conditional_token_mean"
                            else float(args.generate_max_length)
                        ),
                    )
                    loss = loss + entropy_loss
                    if maxent_length_controller is not None:
                        if not bool((mb_loss_masks > 0).all()):
                            raise RuntimeError(
                                "MaxEnt length control cannot exclude response rows"
                            )
                        (
                            length_surrogate_by_row,
                            expected_length_by_row,
                            sampled_prefix_length_by_row,
                            length_prefix_ratio_mean_by_row,
                            length_prefix_ratio_max_by_row,
                            length_prefix_ratio_clipfrac_by_row,
                        ) = prefix_ratio_expected_length_surrogate(
                            new_logps,
                            mb_logps,
                            mb_response_masks,
                            cliprange=float(args.cliprange),
                        )
                        length_surrogate = (
                            length_surrogate_by_row * mb_loss_masks
                        ).sum() / active_rows
                        expected_length = (
                            expected_length_by_row * mb_loss_masks
                        ).sum() / active_rows
                        sampled_prefix_length = (
                            sampled_prefix_length_by_row * mb_loss_masks
                        ).sum() / active_rows
                        length_prefix_ratio_mean = (
                            length_prefix_ratio_mean_by_row * mb_loss_masks
                        ).sum() / active_rows
                        length_prefix_ratio_max = (
                            length_prefix_ratio_max_by_row * mb_loss_masks
                        ).max()
                        length_prefix_ratio_clipfrac = (
                            length_prefix_ratio_clipfrac_by_row * mb_loss_masks
                        ).sum() / active_rows
                        length_loss = standard_maxent_length_penalty_loss(
                            length_surrogate,
                            length_lambda=maxent_length_lambda,
                            reward_estimator_scale=reward_estimator_scale,
                            update_normalizer=float(args.generate_max_length),
                        )
                        loss = loss + length_loss
                        infos["maxent_length_lambda_used"] = torch.tensor(
                            maxent_length_lambda, device=loss.device
                        )
                        infos["maxent_expected_length"] = expected_length.detach()
                        infos["maxent_sampled_prefix_length"] = (
                            sampled_prefix_length.detach()
                        )
                        infos["maxent_length_surrogate"] = length_surrogate.detach()
                        infos["maxent_length_loss"] = length_loss.detach()
                        infos["maxent_length_prefix_ratio_mean"] = (
                            length_prefix_ratio_mean.detach()
                        )
                        infos["maxent_length_prefix_ratio_max"] = (
                            length_prefix_ratio_max.detach()
                        )
                        infos["maxent_length_prefix_ratio_clipfrac"] = (
                            length_prefix_ratio_clipfrac.detach()
                        )
                        stats["maxent_length_lambda_used"].append(maxent_length_lambda)
                        stats["maxent_expected_length"].append(
                            float(expected_length.detach().cpu().item())
                        )
                        stats["maxent_sampled_prefix_length"].append(
                            float(sampled_prefix_length.detach().cpu().item())
                        )
                        stats["maxent_length_surrogate"].append(
                            float(length_surrogate.detach().cpu().item())
                        )
                        stats["maxent_length_loss"].append(
                            float(length_loss.detach().cpu().item())
                        )
                        stats["maxent_length_prefix_ratio_mean"].append(
                            float(length_prefix_ratio_mean.detach().cpu().item())
                        )
                        stats["maxent_length_prefix_ratio_max"].append(
                            float(length_prefix_ratio_max.detach().cpu().item())
                        )
                        stats["maxent_length_prefix_ratio_clipfrac"].append(
                            float(length_prefix_ratio_clipfrac.detach().cpu().item())
                        )
                    infos["maxent_alpha_used"] = torch.tensor(
                        maxent_alpha, device=loss.device
                    )
                    if maxent_objective == "conditional_token_mean":
                        infos["maxent_conditional_token_entropy"] = (
                            sequence_entropy.detach()
                        )
                        infos["maxent_state_distribution_detached"] = torch.tensor(
                            1.0, device=loss.device
                        )
                        infos["maxent_eos_excluded"] = torch.tensor(
                            1.0, device=loss.device
                        )
                        infos["maxent_response_equal_weight"] = torch.tensor(
                            1.0, device=loss.device
                        )
                    else:
                        infos["maxent_sequence_entropy"] = sequence_entropy.detach()
                        infos["maxent_sequence_entropy_per_tmax"] = (
                            sequence_entropy_per_tmax.detach()
                        )
                        infos["maxent_sampled_prefix_entropy"] = (
                            sampled_prefix_entropy.detach()
                        )
                        infos["maxent_sampled_prefix_entropy_per_tmax"] = (
                            sampled_prefix_entropy_per_tmax.detach()
                        )
                        infos["maxent_prefix_ratio_mean"] = prefix_ratio_mean.detach()
                        infos["maxent_prefix_ratio_max"] = prefix_ratio_max.detach()
                        infos["maxent_prefix_ratio_clipfrac"] = (
                            prefix_ratio_clipfrac.detach()
                        )
                    infos["maxent_entropy_surrogate"] = entropy_surrogate.detach()
                    infos["maxent_entropy_loss"] = entropy_loss.detach()
                    infos["maxent_reward_estimator_scale"] = torch.tensor(
                        reward_estimator_scale, device=loss.device
                    )
                    infos["maxent_valid_row_fraction"] = (
                        (mb_loss_masks > 0).float().mean().detach()
                    )
                    if maxent_objective == "conditional_token_mean":
                        stats["maxent_conditional_token_entropy"].append(
                            float(sequence_entropy.detach().cpu().item())
                        )
                    else:
                        stats["maxent_sequence_entropy"].append(
                            float(sequence_entropy.detach().cpu().item())
                        )
                        stats["maxent_sequence_entropy_per_tmax"].append(
                            float(sequence_entropy_per_tmax.detach().cpu().item())
                        )
                        stats["maxent_sampled_prefix_entropy"].append(
                            float(sampled_prefix_entropy.detach().cpu().item())
                        )
                        stats["maxent_sampled_prefix_entropy_per_tmax"].append(
                            float(sampled_prefix_entropy_per_tmax.detach().cpu().item())
                        )
                        stats["maxent_prefix_ratio_mean"].append(
                            float(prefix_ratio_mean.detach().cpu().item())
                        )
                        stats["maxent_prefix_ratio_max"].append(
                            float(prefix_ratio_max.detach().cpu().item())
                        )
                        stats["maxent_prefix_ratio_clipfrac"].append(
                            float(prefix_ratio_clipfrac.detach().cpu().item())
                        )
                    stats["maxent_entropy_surrogate"].append(
                        float(entropy_surrogate.detach().cpu().item())
                    )
                    stats["maxent_entropy_loss"].append(
                        float(entropy_loss.detach().cpu().item())
                    )
                elif policy_entropy_coef != 0.0:
                    if token_entropy is None:
                        raise RuntimeError("policy entropy was not computed")
                    if token_entropy.shape != mb_response_masks.shape:
                        token_entropy = token_entropy[
                            ..., : mb_response_masks.shape[-1]
                        ]
                    entropy_by_row = self.masked_aggregator(
                        token_entropy, mb_response_masks, axis=1
                    )
                    entropy_for_loss = (entropy_by_row * mb_loss_masks).mean()
                    entropy_loss = -policy_entropy_coef * entropy_for_loss
                    infos["policy_entropy_coef"] = torch.tensor(
                        policy_entropy_coef,
                        device=loss.device,
                    )
                    infos["policy_entropy_loss"] = entropy_loss.detach()
                    loss = loss + entropy_loss
                if args.beta > 0:
                    if ref_logps is None:
                        raise ValueError(
                            "beta > 0 baseline updates require reference log-probs."
                        )
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
                    # Always log token-level masked-mean entropy so the
                    # train/entropy key has identical semantics across arms
                    # regardless of whether the entropy bonus is active.
                    if token_entropy is None:
                        raise RuntimeError("policy entropy was not computed")
                    token_entropy_for_log = token_entropy.detach()
                    if token_entropy_for_log.shape != mb_response_masks.shape:
                        token_entropy_for_log = token_entropy_for_log[
                            ..., : mb_response_masks.shape[-1]
                        ]
                    entropy = masked_mean(token_entropy_for_log, mb_response_masks)
                    infos["entropy"] = entropy
                    if canonical_actions:
                        canonical_sequence_entropy_by_row = (
                            token_entropy_for_log * mb_response_masks
                        ).sum(dim=1)
                        canonical_sequence_entropy = (
                            canonical_sequence_entropy_by_row * mb_loss_masks
                        ).sum() / mb_loss_masks.sum().clamp_min(1.0)
                        if self._canonical_action_space is None:
                            raise RuntimeError("canonical action space is missing")
                        canonical_max_entropy = float(
                            self._canonical_action_space.max_sequence_entropy
                        )
                        canonical_sequence_entropy_value = float(
                            canonical_sequence_entropy.detach().cpu().item()
                        )
                        if (
                            not math.isfinite(canonical_sequence_entropy_value)
                            or canonical_sequence_entropy_value < -1e-6
                            or canonical_sequence_entropy_value
                            > canonical_max_entropy + 1e-5
                        ):
                            raise RuntimeError(
                                "canonical sampled-prefix entropy left its "
                                f"support bound: {canonical_sequence_entropy_value}"
                            )
                        infos["canonical_token_entropy_mean"] = entropy
                        # This is a behavior-prefix diagnostic. After a
                        # learner update it is not H(q_new) unless prefix
                        # ratios are applied; reserve that name for the IS
                        # estimator and exact 27-leaf endpoint audit.
                        infos["canonical_sampled_prefix_entropy_sum"] = (
                            canonical_sequence_entropy
                        )
                        infos["canonical_sampled_prefix_entropy_ratio"] = (
                            canonical_sequence_entropy / canonical_max_entropy
                        )
                        infos["canonical_action_count"] = torch.tensor(
                            int(self._canonical_action_space.horizon),
                            device=loss.device,
                        )
                        infos["canonical_action_vocab_size"] = torch.tensor(
                            len(self._canonical_action_token_ids or ()),
                            device=loss.device,
                        )

                self.strategy.backward(loss, self.model, self.optimizer)

                if (
                    canonical_replay_groups
                    and local_grad_step % self.strategy.grad_acc_step == 0
                ):
                    replay_controller = getattr(
                        self,
                        "_canonical_replay_controller",
                        None,
                    )
                    if replay_controller is None:
                        raise RuntimeError(
                            "canonical replay groups lack their inverse controller"
                        )
                    replay = self._materialize_canonical_replay(
                        canonical_replay_groups,
                        device=input_ids.device,
                    )
                    replay_row_count = int(replay.input_ids.size(0))
                    replay_chunk_size = max(
                        1,
                        int(args.train_batch_size_per_device),
                    )
                    with _temporary_eval_mode(self.model, enabled=True):
                        with torch.no_grad():
                            detached_scores = torch.cat(
                                [
                                    self._score_canonical_replay_rows(
                                        replay,
                                        start=start,
                                        stop=min(
                                            start + replay_chunk_size,
                                            replay_row_count,
                                        ),
                                        policy_vocab_upper_bound=(
                                            policy_vocab_upper_bound
                                        ),
                                    ).detach()
                                    for start in range(
                                        0,
                                        replay_row_count,
                                        replay_chunk_size,
                                    )
                                ]
                            )
                        replay_objective = str(args.online_canonical_replay_objective)
                        replay_alpha = float(replay_controller.current_alpha)
                        replay_mass_alpha = 0.0
                        split_result = None
                        if replay_objective == "bank_balance":
                            replay_result = canonical_replay_uniform_loss(
                                detached_scores,
                                replay.group_sizes,
                            )
                        elif replay_objective == "verified_likelihood":
                            replay_result = (
                                canonical_replay_uniform_verified_likelihood_loss(
                                    detached_scores,
                                    replay.group_sizes,
                                )
                            )
                        elif replay_objective == ("verified_likelihood_per_rollout"):
                            replay_result = (
                                canonical_replay_uniform_verified_likelihood_loss(
                                    detached_scores,
                                    replay.group_sizes,
                                )
                            )
                        elif replay_objective == ("split_mass_balance_per_rollout"):
                            split_result = canonical_replay_split_mass_balance_loss(
                                detached_scores,
                                replay.group_sizes,
                            )
                            mass_controller = getattr(
                                self,
                                "_canonical_replay_mass_controller",
                                None,
                            )
                            if mass_controller is None:
                                raise RuntimeError(
                                    "split canonical replay lacks its "
                                    "verified-mass controller"
                                )
                            replay_mass_alpha = float(mass_controller.current_alpha)
                            replay_result = None
                        else:
                            raise RuntimeError(
                                "unsupported canonical replay objective: "
                                f"{replay_objective}"
                            )
                        actuator_loss = (
                            split_result.mass_loss
                            if split_result is not None
                            else replay_result.loss
                        )
                        balance_loss = (
                            split_result.balance_loss
                            if split_result is not None
                            else replay_result.cross_entropy_excess
                        )
                        normalized_entropy = (
                            split_result.normalized_entropy
                            if split_result is not None
                            else replay_result.normalized_entropy
                        )
                        if not all(
                            bool(torch.isfinite(value))
                            for value in (
                                actuator_loss,
                                balance_loss,
                                normalized_entropy,
                            )
                        ):
                            raise RuntimeError(
                                "canonical replay produced a non-finite loss or sensor"
                            )
                        reward_estimator_scale = float(args.num_samples - 1) / float(
                            args.num_samples
                        )
                        replay_objective_scale = (
                            1.0 / float(args.num_samples)
                            if replay_objective
                            in {
                                "verified_likelihood_per_rollout",
                                "split_mass_balance_per_rollout",
                            }
                            else 1.0
                        )
                        weighted_replay_objective = (
                            (
                                actuator_loss * replay_mass_alpha
                                + balance_loss * replay_alpha
                            )
                            if split_result is not None
                            else actuator_loss * replay_alpha
                        )
                        replay_weighted_loss = (
                            weighted_replay_objective
                            * reward_estimator_scale
                            * replay_objective_scale
                        )
                        if not bool(torch.isfinite(replay_weighted_loss)):
                            raise RuntimeError(
                                "unprojected canonical replay coefficient "
                                "exceeded the live loss arithmetic range"
                            )
                        raw_score_gradients = (
                            (
                                (
                                    split_result.mass_score_gradients
                                    * replay_mass_alpha
                                    + split_result.balance_score_gradients
                                    * replay_alpha
                                )
                                if split_result is not None
                                else replay_result.score_gradients * replay_alpha
                            )
                            .to(input_ids.device)
                            .detach()
                        )
                        replay_compute_only = bool(
                            getattr(
                                args,
                                "online_canonical_replay_compute_only",
                                False,
                            )
                        )
                        score_gradients = (
                            torch.zeros_like(raw_score_gradients)
                            if replay_compute_only
                            else raw_score_gradients
                        )
                        replay_applied_weighted_loss = (
                            replay_weighted_loss.new_zeros(())
                            if replay_compute_only
                            else replay_weighted_loss
                        )
                        replay_backward_scale = (
                            reward_estimator_scale
                            * replay_objective_scale
                            * float(self.strategy.grad_acc_step)
                        )
                        for start in range(
                            0,
                            replay_row_count,
                            replay_chunk_size,
                        ):
                            stop = min(
                                start + replay_chunk_size,
                                replay_row_count,
                            )
                            live_scores = self._score_canonical_replay_rows(
                                replay,
                                start=start,
                                stop=stop,
                                policy_vocab_upper_bound=(policy_vocab_upper_bound),
                            )
                            exact_gradient_surrogate = (
                                live_scores * score_gradients[start:stop]
                            ).sum()
                            replay_backward_loss = (
                                exact_gradient_surrogate * replay_backward_scale
                            )
                            if not bool(torch.isfinite(replay_backward_loss)):
                                raise RuntimeError(
                                    "canonical replay produced a non-finite "
                                    "chunked backward scalar"
                                )
                            self.strategy.backward(
                                replay_backward_loss,
                                self.model,
                                self.optimizer,
                            )
                    # DeepSpeed divides every backward call by the configured
                    # accumulation width. Replay is evaluated once at the
                    # boundary (rather than repeating a full exemplar bank for
                    # every one-row policy microbatch), so cancel exactly that
                    # mechanical division. The detached q-minus-uniform weights
                    # above are the exact reverse-KL score derivative at this
                    # unchanged parameter snapshot; chunking changes peak
                    # memory, not the objective gradient.
                    infos.update(
                        {
                            "canonical_replay_actuator_loss": (actuator_loss.detach()),
                            "canonical_replay_balance_loss": (balance_loss.detach()),
                            "canonical_replay_weighted_loss": (
                                replay_applied_weighted_loss.detach()
                            ),
                            "canonical_replay_raw_weighted_loss": (
                                replay_weighted_loss.detach()
                            ),
                            "canonical_replay_compute_only": torch.tensor(
                                float(replay_compute_only),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_backward_scale": torch.tensor(
                                float(self.strategy.grad_acc_step),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_chunk_size": torch.tensor(
                                replay_chunk_size,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_score_passes": torch.tensor(
                                2.0,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_normalized_model_entropy": (
                                normalized_entropy.detach()
                            ),
                            "canonical_replay_cross_entropy_excess": (
                                balance_loss.detach()
                            ),
                            "canonical_replay_alpha_used": torch.tensor(
                                replay_alpha,
                                dtype=torch.float64,
                                device=input_ids.device,
                            ),
                            "canonical_replay_eligible_groups": torch.tensor(
                                (
                                    split_result.balance_eligible_groups
                                    if split_result is not None
                                    else replay_result.eligible_groups
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_retained_modes": torch.tensor(
                                (
                                    split_result.balance_retained_modes
                                    if split_result is not None
                                    else replay_result.retained_modes
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_actuator_groups": torch.tensor(
                                (
                                    split_result.actuator_groups
                                    if split_result is not None
                                    else replay_result.actuator_groups
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_actuator_modes": torch.tensor(
                                (
                                    split_result.actuator_modes
                                    if split_result is not None
                                    else replay_result.actuator_modes
                                ),
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_reward_estimator_scale": (
                                torch.tensor(
                                    reward_estimator_scale,
                                    dtype=torch.float32,
                                    device=input_ids.device,
                                )
                            ),
                            "canonical_replay_score_gradient_sum": (
                                raw_score_gradients.sum()
                            ),
                            "canonical_replay_objective_scale": torch.tensor(
                                replay_objective_scale,
                                dtype=torch.float32,
                                device=input_ids.device,
                            ),
                            "canonical_replay_applied_score_gradient_sum": (
                                score_gradients.sum()
                                * replay_objective_scale
                            ),
                            "canonical_replay_verified_likelihood_active": (
                                torch.tensor(
                                    float(
                                        replay_objective
                                        in {
                                            "verified_likelihood",
                                            "verified_likelihood_per_rollout",
                                            "split_mass_balance_per_rollout",
                                        }
                                    ),
                                    dtype=torch.float32,
                                    device=input_ids.device,
                                )
                            ),
                            "canonical_replay_mass_alpha_used": torch.tensor(
                                replay_mass_alpha,
                                dtype=torch.float64,
                                device=input_ids.device,
                            ),
                            "canonical_replay_balance_alpha_used": torch.tensor(
                                replay_alpha,
                                dtype=torch.float64,
                                device=input_ids.device,
                            ),
                            "canonical_replay_mass_score_gradient_sum": (
                                (
                                    split_result.mass_score_gradients.sum()
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_balance_score_gradient_sum": (
                                (
                                    split_result.balance_score_gradients.sum()
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_mass_score_gradient_l2": (
                                (
                                    torch.linalg.vector_norm(
                                        split_result.mass_score_gradients
                                    )
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_balance_score_gradient_l2": (
                                (
                                    torch.linalg.vector_norm(
                                        split_result.balance_score_gradients
                                    )
                                    if split_result is not None
                                    else detached_scores.new_tensor(0.0)
                                ).to(input_ids.device)
                            ),
                            "canonical_replay_applied_score_gradient_l2": (
                                torch.linalg.vector_norm(score_gradients)
                            ),
                        }
                    )
                    for key, value in (
                        (
                            "canonical_replay_actuator_loss",
                            actuator_loss,
                        ),
                        (
                            "canonical_replay_balance_loss",
                            balance_loss,
                        ),
                        (
                            "canonical_replay_weighted_loss",
                            replay_applied_weighted_loss,
                        ),
                        (
                            "canonical_replay_normalized_model_entropy",
                            normalized_entropy,
                        ),
                        (
                            "canonical_replay_cross_entropy_excess",
                            balance_loss,
                        ),
                    ):
                        stats[key].append(float(value.detach().cpu().item()))
                    stats["canonical_replay_alpha_used"].append(replay_alpha)
                    stats["canonical_replay_mass_alpha_used"].append(replay_mass_alpha)
                    stats["canonical_replay_eligible_groups"].append(
                        float(
                            split_result.balance_eligible_groups
                            if split_result is not None
                            else replay_result.eligible_groups
                        )
                    )
                    stats["canonical_replay_retained_modes"].append(
                        float(
                            split_result.balance_retained_modes
                            if split_result is not None
                            else replay_result.retained_modes
                        )
                    )

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
        for key in (
            "maxent_conditional_token_entropy",
            "maxent_sequence_entropy",
            "maxent_sequence_entropy_per_tmax",
            "maxent_entropy_surrogate",
            "maxent_sampled_prefix_entropy",
            "maxent_sampled_prefix_entropy_per_tmax",
            "maxent_prefix_ratio_mean",
            "maxent_prefix_ratio_clipfrac",
            "maxent_entropy_loss",
            "maxent_length_lambda_used",
            "maxent_expected_length",
            "maxent_sampled_prefix_length",
            "maxent_length_surrogate",
            "maxent_length_loss",
            "maxent_length_prefix_ratio_mean",
            "maxent_length_prefix_ratio_clipfrac",
            "canonical_replay_actuator_loss",
            "canonical_replay_balance_loss",
            "canonical_replay_weighted_loss",
            "canonical_replay_normalized_model_entropy",
            "canonical_replay_cross_entropy_excess",
            "canonical_replay_alpha_used",
            "canonical_replay_eligible_groups",
            "canonical_replay_retained_modes",
        ):
            if stats[key]:
                infos[key] = torch.tensor(stats[key]).mean()
        if stats["maxent_prefix_ratio_max"]:
            infos["maxent_prefix_ratio_max"] = torch.tensor(
                stats["maxent_prefix_ratio_max"]
            ).max()
        if stats["maxent_length_prefix_ratio_max"]:
            infos["maxent_length_prefix_ratio_max"] = torch.tensor(
                stats["maxent_length_prefix_ratio_max"]
            ).max()
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

    def _grpo_learning_step_with_progress(self, trajectory):
        args = self.args
        canonical_task = resolve_canonical_action_task(args)
        canonical_actions = canonical_task != "none"
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        task_final_rewards = final_rewards.detach().clone()
        raw_task_final_rewards = task_final_rewards.detach().clone()
        prompt_id_lens = trajectory["prompt_ids_lens"]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]
        diayn_infos: dict[str, torch.Tensor] = {}
        outcome_collision_infos: dict[str, torch.Tensor] = {}
        outcome_collision_outside_advantage: torch.Tensor | None = None
        semantic_shannon_infos: dict[str, torch.Tensor] = {}
        semantic_shannon_separate_advantage: torch.Tensor | None = None
        online_canonical_infos: dict[str, torch.Tensor] = {}
        online_canonical_advantage: torch.Tensor | None = None
        canonical_replay_groups: list[VerifiedCanonicalReplayGroup] = []
        canonical_behavior_infos: dict[str, torch.Tensor] = {}
        if canonical_actions:
            expected_count = int(args.canonical_graph_action_count)
            observed_counts = response_masks.sum(dim=1)
            if not bool(observed_counts.eq(expected_count).all()):
                raise RuntimeError(
                    "canonical rows must contain exactly "
                    f"{expected_count} actions; got "
                    f"{observed_counts.detach().cpu().tolist()}"
                )
        logging.info(f"learn data size {input_ids.shape}")

        mi_tracker = getattr(self, "_diayn_mi_tracker", None)
        if mi_tracker is not None:
            num_rows = int(input_ids.size(0))
            option_ids = [
                coerce_option_id(value)
                for value in list(trajectory.get("diayn_option_ids") or [])
            ]
            if len(option_ids) != num_rows or any(
                value is None for value in option_ids
            ):
                raise RuntimeError(
                    "DIAYN rollout is missing one valid option id per candidate"
                )
            num_options = int(args.diayn_num_options)
            for group_start in range(0, num_rows, int(args.num_samples)):
                group_options = option_ids[
                    group_start : group_start + int(args.num_samples)
                ]
                expected_per_option = int(args.num_samples) // num_options
                observed = [
                    sum(int(value == option) for value in group_options)
                    for option in range(num_options)
                ]
                if observed != [expected_per_option] * num_options:
                    raise RuntimeError(
                        "DIAYN candidate group is not balanced across options: "
                        f"observed={observed} expected={expected_per_option}"
                    )

            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            refs_grouped = [
                references[index : index + int(args.num_samples)]
                for index in range(0, num_rows, int(args.num_samples))
            ]
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                int(args.num_samples),
                refs_grouped,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            conditional_keys = [
                conditional_answer_repr(reference, key)
                for reference, key in zip(references, answer_keys)
            ]
            task_correct = (task_final_rewards.detach().reshape(-1) > 0).cpu().tolist()
            task_reward_mean = task_final_rewards.detach().mean()
            bonuses, diagnostics = mi_tracker.update_and_score(
                answer_reprs=conditional_keys,
                option_ids=option_ids,
                correct=[bool(value) for value in task_correct],
                loss_masks=loss_masks.detach().cpu().tolist(),
                beta=float(args.diayn_mi_beta),
                correct_only=bool(args.diayn_mi_correct_only),
            )
            bonus_tensor = torch.tensor(
                bonuses, dtype=final_rewards.dtype, device=final_rewards.device
            ).reshape_as(final_rewards)
            final_rewards = final_rewards + bonus_tensor
            diayn_infos = {
                "diayn_task_reward_mean": task_reward_mean,
                "diayn_augmented_reward_mean": final_rewards.detach().mean(),
                "diayn_mi_bonus_mean": torch.tensor(
                    diagnostics.bonus_mean, device=final_rewards.device
                ),
                "diayn_mi_bonus_min": torch.tensor(
                    diagnostics.bonus_min, device=final_rewards.device
                ),
                "diayn_mi_bonus_max": torch.tensor(
                    diagnostics.bonus_max, device=final_rewards.device
                ),
                "diayn_mi_eligible_fraction": torch.tensor(
                    diagnostics.eligible_fraction, device=final_rewards.device
                ),
                "diayn_classifier_accuracy": torch.tensor(
                    diagnostics.classifier_accuracy, device=final_rewards.device
                ),
                "diayn_mi_lower_bound_nats": torch.tensor(
                    diagnostics.lower_bound_nats, device=final_rewards.device
                ),
                "diayn_distinct_answer_reprs": torch.tensor(
                    diagnostics.distinct_answer_reprs, device=final_rewards.device
                ),
                "diayn_loo_supported_fraction": torch.tensor(
                    diagnostics.leave_one_out_supported_fraction,
                    device=final_rewards.device,
                ),
            }

        outcome_collision_coef = float(
            getattr(args, "outcome_collision_coef", 0.0) or 0.0
        )
        outcome_collision_outside_centering = bool(
            getattr(args, "outcome_collision_outside_centering", False)
        )
        if outcome_collision_coef > 0:
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            references_grouped = [
                references[index : index + int(args.num_samples)]
                for index in range(0, num_rows, int(args.num_samples))
            ]
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                int(args.num_samples),
                references_grouped,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            bonuses, diagnostics = compute_outcome_collision_bonuses(
                answer_keys,
                num_samples=int(args.num_samples),
                coefficient=outcome_collision_coef,
            )
            bonus_tensor = torch.tensor(
                bonuses, dtype=final_rewards.dtype, device=final_rewards.device
            ).reshape_as(final_rewards)
            bonus_groups = bonus_tensor.reshape(-1, int(args.num_samples))
            centered_bonus_groups = bonus_groups - bonus_groups.mean(
                dim=1, keepdim=True
            )
            bonus_group_ranges = (
                bonus_groups.max(dim=1).values - bonus_groups.min(dim=1).values
            )
            augmented_rewards = final_rewards + bonus_tensor
            if outcome_collision_outside_centering:
                # E40: preserve ordinary Dr.GRPO centering for the task
                # reward. The detached collision vector is applied exactly
                # once to the precomputed sequence advantage below.
                outcome_collision_outside_advantage = bonus_tensor.detach()
            else:
                # E37: retain the established reward-shaping behavior.
                final_rewards = augmented_rewards
            outcome_collision_infos = {
                "outcome_collision_task_reward_mean": (
                    task_final_rewards.detach().mean()
                ),
                "outcome_collision_augmented_reward_mean": (
                    augmented_rewards.detach().mean()
                ),
                "outcome_collision_reward_sent_to_centering_mean": (
                    final_rewards.detach().mean()
                ),
                "outcome_collision_outside_centering_active": torch.tensor(
                    float(outcome_collision_outside_centering),
                    device=final_rewards.device,
                ),
                "outcome_collision_rate": torch.tensor(
                    diagnostics.collision_rate, device=final_rewards.device
                ),
                "outcome_collision_bonus_mean": torch.tensor(
                    diagnostics.bonus_mean, device=final_rewards.device
                ),
                "outcome_collision_bonus_min": torch.tensor(
                    diagnostics.bonus_min, device=final_rewards.device
                ),
                "outcome_collision_bonus_max": torch.tensor(
                    diagnostics.bonus_max, device=final_rewards.device
                ),
                "outcome_collision_bonus_zero_spread_group_fraction": (
                    bonus_group_ranges.eq(0).float().mean()
                ),
                "outcome_collision_centered_bonus_abs_mean": (
                    centered_bonus_groups.abs().mean()
                ),
                "outcome_collision_centered_bonus_rms": torch.sqrt(
                    centered_bonus_groups.square().mean()
                ),
                "outcome_collision_distinct_outcomes_mean": torch.tensor(
                    diagnostics.distinct_outcomes_mean,
                    device=final_rewards.device,
                ),
                "outcome_collision_distinct_fraction": torch.tensor(
                    diagnostics.distinct_fraction, device=final_rewards.device
                ),
                "outcome_collision_invalid_fraction": torch.tensor(
                    diagnostics.invalid_fraction, device=final_rewards.device
                ),
                "outcome_collision_parseable_fraction": torch.tensor(
                    diagnostics.parseable_fraction, device=final_rewards.device
                ),
            }

        semantic_shannon_tracker = getattr(self, "_semantic_shannon_tracker", None)
        if semantic_shannon_tracker is not None:
            if not isinstance(semantic_shannon_tracker, SemanticShannonTracker):
                raise RuntimeError("invalid semantic Shannon tracker")
            semantic_shannon_use_separate_advantage = bool(
                getattr(args, "semantic_shannon_separate_advantage", False)
            )
            semantic_shannon_use_quality_gate = bool(
                getattr(
                    args,
                    "semantic_shannon_quality_gated_advantage",
                    False,
                )
            )
            semantic_shannon_use_success_conditioned_signed = bool(
                getattr(
                    args,
                    ("semantic_shannon_success_conditioned_signed_advantage"),
                    False,
                )
            )
            if (
                semantic_shannon_use_quality_gate
                or semantic_shannon_use_success_conditioned_signed
            ) and not semantic_shannon_use_separate_advantage:
                raise RuntimeError(
                    "semantic Shannon gated modes require the separate advantage path"
                )
            if (
                semantic_shannon_use_quality_gate
                and semantic_shannon_use_success_conditioned_signed
            ):
                raise RuntimeError(
                    "semantic Shannon quality-gated and "
                    "success-conditioned signed modes are mutually exclusive"
                )
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            references_grouped = [
                references[index : index + int(args.num_samples)]
                for index in range(0, num_rows, int(args.num_samples))
            ]
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                int(args.num_samples),
                references_grouped,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            semantic_task_rewards = task_final_rewards.detach().view(-1).cpu().tolist()
            if (
                str(
                    getattr(
                        args,
                        "online_canonical_key_mode",
                        "modebench_outcome",
                    )
                )
                == "math_verified_answer"
            ):
                # MATH-500 is a realistic single-answer generalization track,
                # not a reasoning-strategy benchmark. Collapse every
                # verifier-positive representation to the same prompt-local
                # outcome so formatting aliases cannot masquerade as modes.
                answer_keys = math_verified_answer_outcome_keys(
                    [float(reward) > 0.0 for reward in semantic_task_rewards]
                )
            prompt_token_ids = [
                input_ids[row_index, : int(prompt_id_lens[row_index])]
                .detach()
                .cpu()
                .tolist()
                for row_index in range(num_rows)
            ]
            diagnostics = None
            advantage_diagnostics = None
            quality_gated_diagnostics = None
            success_conditioned_signed_diagnostics = None
            if semantic_shannon_use_success_conditioned_signed:
                (
                    separate_advantages,
                    success_conditioned_signed_diagnostics,
                ) = semantic_shannon_tracker.score_success_conditioned_signed_advantages_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    task_rewards=semantic_task_rewards,
                    active_mask=loss_masks.detach().view(-1).cpu().tolist(),
                    num_samples=int(args.num_samples),
                )
                semantic_shannon_separate_advantage = (
                    torch.tensor(
                        separate_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            elif semantic_shannon_use_quality_gate:
                (
                    separate_advantages,
                    quality_gated_diagnostics,
                ) = semantic_shannon_tracker.score_quality_gated_advantages_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    task_rewards=semantic_task_rewards,
                    active_mask=loss_masks.detach().view(-1).cpu().tolist(),
                    num_samples=int(args.num_samples),
                )
                semantic_shannon_separate_advantage = (
                    torch.tensor(
                        separate_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            elif semantic_shannon_use_separate_advantage:
                (
                    separate_advantages,
                    diagnostics,
                    advantage_diagnostics,
                ) = semantic_shannon_tracker.score_separate_advantages_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    num_samples=int(args.num_samples),
                )
                semantic_shannon_separate_advantage = (
                    torch.tensor(
                        separate_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            else:
                bonuses, diagnostics = semantic_shannon_tracker.score_and_update(
                    prompt_token_ids=prompt_token_ids,
                    answer_keys=answer_keys,
                    num_samples=int(args.num_samples),
                )
                bonus_tensor = torch.tensor(
                    bonuses,
                    dtype=final_rewards.dtype,
                    device=final_rewards.device,
                ).reshape_as(final_rewards)
                final_rewards = final_rewards + bonus_tensor
            semantic_shannon_infos = {
                "semantic_shannon_task_reward_mean": (
                    task_final_rewards.detach().mean()
                ),
                "semantic_shannon_augmented_reward_mean": (
                    final_rewards.detach().mean()
                ),
                "semantic_shannon_reward_sent_to_centering_mean": (
                    final_rewards.detach().mean()
                ),
                "semantic_shannon_separate_advantage_active": torch.tensor(
                    float(semantic_shannon_use_separate_advantage),
                    device=final_rewards.device,
                ),
                "semantic_shannon_quality_gated_advantage_active": torch.tensor(
                    float(semantic_shannon_use_quality_gate),
                    device=final_rewards.device,
                ),
                "semantic_shannon_success_conditioned_signed_advantage_active": (
                    torch.tensor(
                        float(semantic_shannon_use_success_conditioned_signed),
                        device=final_rewards.device,
                    )
                ),
            }
            if diagnostics is not None:
                semantic_shannon_infos.update(
                    {
                        "semantic_shannon_bonus_mean": torch.tensor(
                            diagnostics.bonus_mean, device=final_rewards.device
                        ),
                        "semantic_shannon_bonus_min": torch.tensor(
                            diagnostics.bonus_min, device=final_rewards.device
                        ),
                        "semantic_shannon_bonus_max": torch.tensor(
                            diagnostics.bonus_max, device=final_rewards.device
                        ),
                        "semantic_shannon_surprisal_mean": torch.tensor(
                            diagnostics.surprisal_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_normalized_surprisal_mean": torch.tensor(
                            diagnostics.normalized_surprisal_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_entropy_mean": torch.tensor(
                            diagnostics.entropy_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_clipped_surprisal_mean": torch.tensor(
                            diagnostics.clipped_surprisal_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_clip_fraction": torch.tensor(
                            diagnostics.clip_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_predictive_probability_mean": torch.tensor(
                            diagnostics.predictive_probability_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_predictive_probability_min": torch.tensor(
                            diagnostics.predictive_probability_min,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_predictive_probability_max": torch.tensor(
                            diagnostics.predictive_probability_max,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_normalization_error_max": torch.tensor(
                            diagnostics.normalization_error_max,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_unseen_fraction": torch.tensor(
                            diagnostics.unseen_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_history_total_mean": torch.tensor(
                            diagnostics.history_total_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_distinct_outcomes_mean": torch.tensor(
                            diagnostics.distinct_outcomes_mean,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_distinct_fraction": torch.tensor(
                            diagnostics.distinct_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_invalid_fraction": torch.tensor(
                            diagnostics.invalid_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_parseable_fraction": torch.tensor(
                            diagnostics.parseable_fraction,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_tracked_prompts": torch.tensor(
                            diagnostics.tracked_prompts,
                            device=final_rewards.device,
                        ),
                        "semantic_shannon_tracked_outcomes": torch.tensor(
                            diagnostics.tracked_outcomes,
                            device=final_rewards.device,
                        ),
                    }
                )
            if quality_gated_diagnostics is not None:
                quality_values = {
                    field_name: getattr(quality_gated_diagnostics, field_name)
                    for field_name in (
                        "raw_all_row_advantage_mean",
                        "raw_all_row_advantage_min",
                        "raw_all_row_advantage_max",
                        "raw_all_row_advantage_abs_mean",
                        "raw_all_row_advantage_rms",
                        "raw_all_row_advantage_positive_fraction",
                        "raw_all_row_advantage_negative_fraction",
                        "raw_all_row_advantage_zero_fraction",
                        "effective_advantage_mean",
                        "effective_advantage_min",
                        "effective_advantage_max",
                        "effective_advantage_abs_mean",
                        "effective_advantage_rms",
                        "effective_advantage_positive_fraction",
                        "effective_advantage_zero_fraction",
                        "eligible_fraction",
                        "gated_fraction",
                        "active_fraction",
                        "reward_positive_fraction",
                        "parseable_fraction",
                        "positive_only_zeroed_fraction",
                        "cap_fraction",
                        "advantage_cap",
                        "predictive_baseline_mean",
                        "predictive_centering_error_max",
                        "predictive_probability_mean",
                        "predictive_probability_min",
                        "predictive_probability_max",
                        "normalization_error_max",
                        "history_total_before_mean",
                        "history_rows_added",
                        "history_groups_updated",
                        "history_groups_skipped",
                        "tracked_prompts",
                        "tracked_outcomes",
                    )
                }
                semantic_shannon_infos.update(
                    {
                        f"semantic_shannon_quality_gated_{name}": torch.tensor(
                            value, device=final_rewards.device
                        )
                        for name, value in quality_values.items()
                    }
                )
            if success_conditioned_signed_diagnostics is not None:
                signed_values = {
                    field_name: getattr(
                        success_conditioned_signed_diagnostics, field_name
                    )
                    for field_name in (
                        "raw_eligible_advantage_mean",
                        "raw_eligible_advantage_min",
                        "raw_eligible_advantage_max",
                        "raw_eligible_advantage_abs_mean",
                        "raw_eligible_advantage_rms",
                        "effective_advantage_mean",
                        "effective_advantage_min",
                        "effective_advantage_max",
                        "effective_advantage_abs_mean",
                        "effective_advantage_rms",
                        "effective_advantage_positive_fraction",
                        "effective_advantage_negative_fraction",
                        "effective_advantage_zero_fraction",
                        "eligible_fraction",
                        "gated_fraction",
                        "active_fraction",
                        "reward_positive_fraction",
                        "parseable_fraction",
                        "positive_cap_fraction",
                        "negative_cap_fraction",
                        "advantage_cap",
                        "predictive_baseline_mean",
                        "predictive_centering_error_max",
                        "predictive_probability_mean",
                        "predictive_probability_min",
                        "predictive_probability_max",
                        "normalization_error_max",
                        "history_total_before_mean",
                        "history_rows_added",
                        "history_groups_updated",
                        "history_groups_skipped",
                        "tracked_prompts",
                        "tracked_outcomes",
                        "open_set_inverse_adaptation_active",
                        "open_set_coefficient_used",
                        "open_set_observed_normalized_entropy",
                        "open_set_entropy_ema",
                        "open_set_reference_entropy",
                        "open_set_inverse_multiplier",
                        "open_set_next_coefficient",
                        "open_set_observations",
                        "open_set_warmup_complete",
                        "open_set_observation_skipped",
                        "open_set_projection_active",
                    )
                }
                semantic_shannon_infos.update(
                    {
                        (
                            f"semantic_shannon_success_conditioned_signed_{name}"
                        ): torch.tensor(value, device=final_rewards.device)
                        for name, value in signed_values.items()
                    }
                )
            if (
                semantic_shannon_use_separate_advantage
                and advantage_diagnostics is not None
            ):
                semantic_shannon_infos.update(
                    {
                        "semantic_shannon_separate_predictive_baseline_mean": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_mean,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_baseline_min": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_min,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_baseline_max": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_max,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_baseline_normalized_mean": (
                            torch.tensor(
                                advantage_diagnostics.predictive_baseline_normalized_mean,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_predictive_centering_error_max": (
                            torch.tensor(
                                advantage_diagnostics.predictive_centering_error_max,
                                device=final_rewards.device,
                            )
                        ),
                        "semantic_shannon_separate_advantage_scale": torch.tensor(
                            advantage_diagnostics.advantage_scale,
                            device=final_rewards.device,
                        ),
                    }
                )

        online_canonical_bank = getattr(self, "_online_canonical_bank", None)
        if online_canonical_bank is not None:
            if not isinstance(online_canonical_bank, OnlineCanonicalBank):
                raise RuntimeError("invalid online canonical bank")
            online_canonical_bank_objective_active = (
                online_canonical_bank.objective_active
            )
            online_canonical_replay_active = bool(
                getattr(args, "online_canonical_replay", False)
            )
            num_rows = int(input_ids.size(0))
            response_texts: list[str] = []
            response_token_ids: list[list[int]] = []
            label_ids = input_ids[:, 1:]
            for row_ids, row_mask in zip(label_ids, response_masks):
                token_ids = row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()
                response_token_ids.append(token_ids)
                response_texts.append(
                    self.tokenizer.decode(token_ids, skip_special_tokens=True)
                )
            key_mode = str(
                getattr(
                    args,
                    "online_canonical_key_mode",
                    "modebench_outcome",
                )
            )
            if key_mode in {"math_strategy_qwen72", "verified_route"}:
                response_texts = [
                    str(value) for value in list(trajectory.get("responses") or [])
                ]
                if len(response_texts) != num_rows:
                    raise RuntimeError(
                        "MATH strategy canonicalization requires the exact "
                        "validator-graded response for every row"
                    )
            prompt_token_ids = [
                input_ids[row_index, : int(prompt_id_lens[row_index])]
                .detach()
                .cpu()
                .tolist()
                for row_index in range(num_rows)
            ]
            task_reward_positive = (
                task_final_rewards.detach().view(-1).gt(0).cpu().tolist()
            )
            math_strategy_diagnostics = None
            route_verifier_ids: list[str | None] = [None] * num_rows
            route_signatures: list[str | None] = [None] * num_rows
            if key_mode == "modebench_outcome":
                references = list(trajectory.get("references") or [])
                references = (references + [None] * num_rows)[:num_rows]
                outcome_keys = [
                    validated_modebench_outcome_key(
                        text,
                        references[row_index],
                    )
                    for row_index, text in enumerate(response_texts)
                ]
            elif key_mode == "math_verified_answer":
                # The ordinary full MATH verifier has already produced
                # task_reward_positive. Use that validation as the complete
                # canonical contract and intentionally expose exactly one
                # accepted outcome per prompt. This enables verified-mass
                # replay while making multi-mode balance structurally
                # ineligible; it does not claim to verify reasoning routes.
                outcome_keys = math_verified_answer_outcome_keys(task_reward_positive)
            elif key_mode == "math_strategy_qwen72":
                canonicalizer = getattr(self, "_math_strategy_canonicalizer", None)
                if not isinstance(canonicalizer, MathStrategyCanonicalizer):
                    raise RuntimeError("MATH strategy key mode lacks its canonicalizer")
                prompt_texts = [
                    str(value) for value in list(trajectory.get("prompts") or [])
                ]
                if len(prompt_texts) != num_rows:
                    raise RuntimeError(
                        "MATH strategy canonicalization requires one raw "
                        "problem per trajectory row"
                    )
                outcome_keys, math_strategy_diagnostics = canonicalizer.canonicalize(
                    prompt_token_ids=prompt_token_ids,
                    prompt_texts=prompt_texts,
                    response_texts=response_texts,
                    task_reward_positive=task_reward_positive,
                    active_mask=(loss_masks.detach().view(-1).cpu().tolist()),
                    num_samples=int(args.num_samples),
                )
            elif key_mode == "verified_route":
                references = list(trajectory.get("references") or [])
                references = (references + [None] * num_rows)[:num_rows]
                prompt_texts = [
                    str(value) for value in list(trajectory.get("prompts") or [])
                ]
                if len(prompt_texts) != num_rows:
                    raise RuntimeError(
                        "verified-route canonicalization requires one raw "
                        "problem per trajectory row"
                    )
                identities = [
                    validated_exploration_identity(
                        response_texts[row_index],
                        prompt_texts[row_index],
                        references[row_index],
                        fast=(str(args.verifier_version) != "math_verify"),
                        task_verified=bool(task_reward_positive[row_index]),
                    )
                    for row_index in range(num_rows)
                ]
                outcome_keys = [
                    (identity.endpoint_key if identity is not None else None)
                    for identity in identities
                ]
                route_verifier_ids = [
                    (identity.verifier if identity is not None else None)
                    for identity in identities
                ]
                route_signatures = [
                    (identity.route_signature if identity is not None else None)
                    for identity in identities
                ]
            else:
                raise RuntimeError(
                    f"unsupported online canonical key mode: {key_mode!r}"
                )
            validator_admitted = [key is not None for key in outcome_keys]
            validator_positive_actor_negative_rows = [
                index
                for index, (verified, rewarded) in enumerate(
                    zip(validator_admitted, task_reward_positive)
                )
                if verified and not rewarded
            ]
            actor_positive_validator_negative_rows = [
                index
                for index, (verified, rewarded) in enumerate(
                    zip(validator_admitted, task_reward_positive)
                )
                if rewarded and not verified
            ]
            disagreement_rows = (
                validator_positive_actor_negative_rows
                + actor_positive_validator_negative_rows
            )
            if disagreement_rows:
                logging.warning(
                    "online canonical validator/task-reward disagreement "
                    "at rows %s; fail-closed intersection excludes them "
                    "from bank admission",
                    sorted(disagreement_rows),
                )
            # A canonical key is eligible only when both independent gates
            # agree: the executable canonicalizer validated the response and
            # the rollout received positive task reward.  In particular, a
            # parseable/valid-looking completion that was task-reward-zero
            # (for example because the rollout contract rejected truncation)
            # must not influence either the passive discovery tracker or the
            # active canonical objective.  Disagreement is telemetry, not a
            # fatal training condition.
            outcome_keys = [
                key if (key is not None and rewarded) else None
                for key, rewarded in zip(outcome_keys, task_reward_positive)
            ]
            if key_mode == "verified_route":
                verified_route_library = getattr(
                    self,
                    "_verified_route_library",
                    None,
                )
                if not isinstance(
                    verified_route_library,
                    VerifiedRouteLibrary,
                ):
                    raise RuntimeError(
                        "verified-route key mode lacks its route library"
                    )
                action_logprob_rows = list(trajectory.get("action_logprobs") or [])
                if len(action_logprob_rows) != num_rows:
                    raise RuntimeError(
                        "verified-route tracking requires actor log "
                        "probabilities for every row"
                    )
                model_mean_logprobs: list[float] = []
                for row_index, values in enumerate(action_logprob_rows):
                    row_values = [float(value) for value in list(values)]
                    if route_signatures[row_index] is not None and not row_values:
                        raise RuntimeError(
                            "verified neutral route lacks behavior log probabilities"
                        )
                    model_mean_logprobs.append(
                        (sum(row_values) / len(row_values) if row_values else 0.0)
                    )
                route_signatures = [
                    (route if endpoint_key is not None else None)
                    for route, endpoint_key in zip(
                        route_signatures,
                        outcome_keys,
                    )
                ]
                route_verifier_ids = [
                    (verifier if endpoint_key is not None else None)
                    for verifier, endpoint_key in zip(
                        route_verifier_ids,
                        outcome_keys,
                    )
                ]
                verified_route_library.observe_neutral(
                    prompt_token_ids=prompt_token_ids,
                    verifier_ids=route_verifier_ids,
                    endpoint_keys=outcome_keys,
                    route_signatures=route_signatures,
                    response_token_ids=response_token_ids,
                    model_mean_logprobs=model_mean_logprobs,
                    task_verified=task_reward_positive,
                    active_mask=(loss_masks.detach().view(-1).cpu().tolist()),
                )
            admitted = [key is not None for key in outcome_keys]
            if bool(getattr(args, "math_strategy_gate_task_reward", False)):
                if key_mode != "math_strategy_qwen72":
                    raise RuntimeError(
                        "MATH strategy reward gate reached a non-MATH key mode"
                    )
                final_rewards, task_final_rewards = (
                    apply_math_strategy_task_reward_gate(
                        final_rewards,
                        task_final_rewards,
                        admitted,
                    )
                )
            bank_advantages, bank_diagnostics = online_canonical_bank.score_and_update(
                prompt_token_ids=prompt_token_ids,
                outcome_keys=outcome_keys,
                task_rewards=(task_final_rewards.detach().view(-1).cpu().tolist()),
                active_mask=loss_masks.detach().view(-1).cpu().tolist(),
                num_samples=int(args.num_samples),
                entropy_alpha_override=(
                    getattr(
                        self,
                        "_online_canonical_alpha_controller",
                        None,
                    ).current_alpha
                    if getattr(
                        self,
                        "_online_canonical_alpha_controller",
                        None,
                    )
                    is not None
                    else None
                ),
                response_token_ids=(
                    response_token_ids if online_canonical_replay_active else None
                ),
            )
            if online_canonical_bank_objective_active:
                online_canonical_advantage = (
                    torch.tensor(
                        bank_advantages,
                        dtype=final_rewards.dtype,
                        device=final_rewards.device,
                    )
                    .reshape_as(final_rewards)
                    .detach()
                )
            canonical_replay_used_global_scheduler = False
            canonical_replay_used_prompt_local_scheduler = False
            verified_route_replay_used = False
            verified_route_endpoint_fallback_used = False
            if online_canonical_replay_active:
                replay_min_modes = (
                    1
                    if str(args.online_canonical_replay_objective)
                    in {
                        "verified_likelihood_per_rollout",
                        "split_mass_balance_per_rollout",
                    }
                    else 2
                )
                if key_mode == "verified_route":
                    verified_route_library = getattr(
                        self,
                        "_verified_route_library",
                        None,
                    )
                    if not isinstance(
                        verified_route_library,
                        VerifiedRouteLibrary,
                    ):
                        raise RuntimeError(
                            "verified-route replay lacks its route library"
                        )
                    canonical_replay_used_global_scheduler = True
                    canonical_replay_groups = (
                        verified_route_library.scheduled_cross_prompt_replay_groups(
                            prompt_token_ids,
                        )
                    )
                    verified_route_replay_used = bool(canonical_replay_groups)
                    if not canonical_replay_groups:
                        # Graph has no domain-independent executable route;
                        # early cold-start batches in other domains may not yet
                        # have a route recurring on two neutral prompts. Keep
                        # the exact fixed compute budget with the independently
                        # verified endpoint bank until route replay is eligible.
                        canonical_replay_groups = (
                            online_canonical_bank.scheduled_global_replay_groups(
                                min_modes=1,
                            )
                        )
                        verified_route_endpoint_fallback_used = bool(
                            canonical_replay_groups
                        )
                elif int(args.online_canonical_replay_global_groups_per_step) > 0:
                    if int(
                        args.online_canonical_replay_global_bootstrap_steps
                    ) > 0 and not (
                        online_canonical_bank.global_replay_bootstrap_active
                    ):
                        canonical_replay_used_prompt_local_scheduler = True
                        canonical_replay_groups = online_canonical_bank.replay_groups(
                            prompt_token_ids,
                            min_modes=(replay_min_modes),
                        )
                    else:
                        canonical_replay_used_global_scheduler = True
                        canonical_replay_groups = (
                            online_canonical_bank.scheduled_global_replay_groups(
                                min_modes=replay_min_modes,
                            )
                        )
                else:
                    canonical_replay_used_prompt_local_scheduler = True
                    canonical_replay_groups = online_canonical_bank.replay_groups(
                        prompt_token_ids,
                        min_modes=replay_min_modes,
                    )
            online_canonical_infos = {
                f"online_canonical_{name}": torch.tensor(
                    getattr(bank_diagnostics, name),
                    device=final_rewards.device,
                )
                for name in (
                    "entropy_estimate_mean",
                    "normalized_entropy_mean",
                    "normalized_entropy_ratio_mean",
                    "normalized_entropy_ratio_eligible_fraction",
                    "log_support_mean",
                    "entropy_alpha_used",
                    "entropy_advantage_mean",
                    "entropy_advantage_rms",
                    "novelty_advantage_mean",
                    "novelty_advantage_rms",
                    "combined_advantage_mean",
                    "combined_advantage_rms",
                    "eligible_fraction",
                    "reward_positive_fraction",
                    "canonicalizable_correct_fraction",
                    "new_outcome_count",
                    "new_outcome_row_fraction",
                    "bank_size_before_mean",
                    "bank_size_after_mean",
                    "tracked_prompts",
                    "tracked_outcomes",
                    "support_at_least_two_prompt_fraction",
                )
            }
            if key_mode == "verified_route":
                verified_route_library = getattr(
                    self,
                    "_verified_route_library",
                    None,
                )
                if not isinstance(
                    verified_route_library,
                    VerifiedRouteLibrary,
                ):
                    raise RuntimeError(
                        "verified-route telemetry lacks its route library"
                    )
                route_diagnostics = verified_route_library.diagnostics()
                online_canonical_infos.update(
                    {
                        f"verified_route_{name}": torch.tensor(
                            float(getattr(route_diagnostics, name)),
                            device=final_rewards.device,
                        )
                        for name in (
                            "neutral_rows_observed",
                            "neutral_routes_observed",
                            "proposal_rows_admitted",
                            "proposal_rows_rejected_trust",
                            "proposal_graduations",
                            "distinct_routes",
                            "recurring_routes",
                            "distinct_source_prompts",
                            "cross_prompt_neutral_reproductions",
                            "post_replay_cross_prompt_neutral_reproductions",
                            "cross_prompt_replay_updates",
                            "cross_prompt_replay_groups",
                            "cross_prompt_replay_rows",
                        )
                    }
                )
                online_canonical_infos.update(
                    {
                        "verified_route_replay_used": torch.tensor(
                            float(verified_route_replay_used),
                            device=final_rewards.device,
                        ),
                        "verified_route_endpoint_fallback_used": torch.tensor(
                            float(verified_route_endpoint_fallback_used),
                            device=final_rewards.device,
                        ),
                        "verified_route_proposal_rows_to_ppo": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                        "verified_route_gold_support_feedback": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                        "verified_route_eval_feedback": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                    }
                )
            if online_canonical_replay_active:
                online_canonical_infos.update(
                    {
                        "canonical_replay_available_groups": torch.tensor(
                            len(canonical_replay_groups),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_available_modes": torch.tensor(
                            sum(
                                len(group.outcome_keys)
                                for group in canonical_replay_groups
                            ),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_capacity": torch.tensor(
                            int(args.online_canonical_replay_capacity),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_compute_only_configured": torch.tensor(
                            float(
                                bool(
                                    getattr(
                                        args,
                                        "online_canonical_replay_compute_only",
                                        False,
                                    )
                                )
                            ),
                            device=final_rewards.device,
                        ),
                        "canonical_replay_realized_prompt_tokens": torch.tensor(
                            sum(
                                len(group.prompt_token_ids)
                                * len(group.response_token_ids)
                                for group in canonical_replay_groups
                            ),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_realized_response_tokens": torch.tensor(
                            sum(
                                len(response)
                                for group in canonical_replay_groups
                                for response in group.response_token_ids
                            ),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_charged_response_token_budget": torch.tensor(
                            int(args.online_canonical_replay_capacity)
                            * int(args.generate_max_length)
                            * int(
                                max(
                                    1,
                                    args.online_canonical_replay_global_groups_per_step,
                                )
                            ),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_gold_support_feedback": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_alpha_projection_active": torch.tensor(
                            0.0,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_global_scheduler_active": torch.tensor(
                            float(
                                int(args.online_canonical_replay_global_groups_per_step)
                                > 0
                            ),
                            device=final_rewards.device,
                        ),
                        "canonical_replay_global_groups_per_step": torch.tensor(
                            int(args.online_canonical_replay_global_groups_per_step),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_global_bootstrap_steps": torch.tensor(
                            int(args.online_canonical_replay_global_bootstrap_steps),
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_global_bootstrap_updates": torch.tensor(
                            online_canonical_bank.global_replay_updates,
                            dtype=torch.float32,
                            device=final_rewards.device,
                        ),
                        "canonical_replay_global_bootstrap_active": torch.tensor(
                            float(online_canonical_bank.global_replay_bootstrap_active),
                            device=final_rewards.device,
                        ),
                        "canonical_replay_prompt_local_phase_active": torch.tensor(
                            float(
                                int(args.online_canonical_replay_global_bootstrap_steps)
                                > 0
                                and not (
                                    online_canonical_bank.global_replay_bootstrap_active
                                )
                            ),
                            device=final_rewards.device,
                        ),
                        "canonical_replay_schedule_used_global": torch.tensor(
                            float(canonical_replay_used_global_scheduler),
                            device=final_rewards.device,
                        ),
                        "canonical_replay_schedule_used_prompt_local": torch.tensor(
                            float(canonical_replay_used_prompt_local_scheduler),
                            device=final_rewards.device,
                        ),
                    }
                )
            if math_strategy_diagnostics is not None:
                online_canonical_infos.update(
                    {
                        f"math_strategy_{name}": torch.tensor(
                            getattr(math_strategy_diagnostics, name),
                            device=final_rewards.device,
                        )
                        for name in (
                            "judge_calls",
                            "validator_positive_rows",
                            "accepted_rows",
                            "rejected_integrity_rows",
                            "rejected_ambiguous_rows",
                            "rejected_disagreement_rows",
                            "matched_existing_rows",
                            "new_strategy_rows",
                            "new_strategy_count",
                            "judge_format_failure_rows",
                            "rejected_contract_rows",
                            "inferred_unstructured_rows",
                            "rejected_strategy_inference_rows",
                        )
                    }
                )
            online_canonical_infos["math_strategy_raw_task_reward_mean"] = (
                raw_task_final_rewards.detach().mean()
            )
            online_canonical_infos["math_strategy_gated_task_reward_mean"] = (
                task_final_rewards.detach().mean()
            )
            online_canonical_infos["math_strategy_task_reward_gate_active"] = (
                torch.tensor(
                    float(
                        bool(
                            getattr(
                                args,
                                "math_strategy_gate_task_reward",
                                False,
                            )
                        )
                    ),
                    device=final_rewards.device,
                )
            )
            online_canonical_infos["online_canonical_task_reward_mean"] = (
                task_final_rewards.detach().mean()
            )
            online_canonical_infos["online_canonical_reward_sent_to_centering_mean"] = (
                final_rewards.detach().mean()
            )
            online_canonical_infos.update(
                {
                    "online_canonical_validator_positive_actor_negative_rows": (
                        torch.tensor(
                            len(validator_positive_actor_negative_rows),
                            device=final_rewards.device,
                        )
                    ),
                    "online_canonical_actor_positive_validator_negative_rows": (
                        torch.tensor(
                            len(actor_positive_validator_negative_rows),
                            device=final_rewards.device,
                        )
                    ),
                    "online_canonical_validator_task_disagreement_rows": (
                        torch.tensor(
                            len(disagreement_rows),
                            device=final_rewards.device,
                        )
                    ),
                    "verified_discovery_cumulative_outcomes": torch.tensor(
                        bank_diagnostics.tracked_outcomes,
                        device=final_rewards.device,
                    ),
                    "verified_discovery_tracked_prompts": torch.tensor(
                        bank_diagnostics.tracked_prompts,
                        device=final_rewards.device,
                    ),
                    "verified_discovery_mean_support_per_prompt": torch.tensor(
                        online_canonical_bank.mean_support_per_prompt,
                        device=final_rewards.device,
                    ),
                }
            )

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
        canonical_learner_full_logps: torch.Tensor | None = None
        canonical_learner_support_mask: torch.Tensor | None = None
        if canonical_actions:
            canonical_learner_full_logps = torch.zeros(
                (
                    input_ids.shape[0],
                    input_ids.shape[1] - 1,
                    len(self._canonical_action_token_ids or ()),
                ),
                dtype=torch.float32,
                device=input_ids.device,
            )
            canonical_learner_support_mask = torch.zeros_like(
                canonical_learner_full_logps, dtype=torch.bool
            )
        policy_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(self.model)
        # E14's behavior trace is produced in eval mode.  Score its frozen old
        # policy in the same mode as well; otherwise training-only behavior can
        # invalidate what should be a same-snapshot identity check.  This is
        # intentionally scoped to canonical actions so retained baselines are
        # unchanged.
        with (
            _temporary_eval_mode(self.model, enabled=canonical_actions),
            torch.no_grad(),
        ):
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
                if canonical_actions:
                    if canonical_task == "graph_coloring":
                        (
                            batch_logps,
                            _,
                            batch_full_logps,
                        ) = restricted_action_log_probs_entropy_and_distribution(
                            batch_logits,
                            mb_input_ids,
                            mb_response_masks,
                            allowed_token_ids=(self._canonical_action_token_ids or ()),
                        )
                        batch_support_mask = (
                            mb_response_masks.to(torch.bool)
                            .unsqueeze(-1)
                            .expand_as(batch_full_logps)
                        )
                    else:
                        (
                            batch_logps,
                            _,
                            batch_full_logps,
                            batch_support_mask,
                        ) = restricted_position_action_log_probs_entropy_and_distribution(
                            batch_logits,
                            mb_input_ids,
                            mb_response_masks,
                            allowed_token_ids_by_position=(
                                self._canonical_action_token_ids_by_position or ()
                            ),
                        )
                    assert canonical_learner_full_logps is not None
                    assert canonical_learner_support_mask is not None
                    canonical_learner_full_logps[
                        mini_batch_inds, : mb_last_valid_token_pos - 1
                    ] = batch_full_logps
                    canonical_learner_support_mask[
                        mini_batch_inds, : mb_last_valid_token_pos - 1
                    ] = batch_support_mask
                else:
                    batch_logps, _ = self._policy_logps_and_optional_entropy(
                        batch_logits,
                        mb_input_ids,
                        mb_response_masks,
                        need_entropy=False,
                    )
                logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps

        old_logps = logps
        if canonical_actions:
            canonical_support = tuple(self._canonical_action_token_ids or ())
            positional_supports = tuple(
                self._canonical_action_token_ids_by_position or ()
            )
            if len(positional_supports) != 3:
                raise RuntimeError("canonical learner did not retain three supports")
            behavior_support_mask = None
            if canonical_task == "graph_coloring":
                (
                    behavior_selected_logps,
                    behavior_full_logps,
                    behavior_norm_error,
                    behavior_selected_echo_diff,
                ) = materialize_canonical_behavior_policy(
                    input_ids[:, 1:],
                    response_masks,
                    action_ids=list(trajectory.get("action_ids") or []),
                    selected_log_probs=list(trajectory.get("action_logprobs") or []),
                    full_log_probs=list(
                        trajectory.get("canonical_behavior_action_logprobs") or []
                    ),
                    behavior_action_token_ids=list(
                        trajectory.get("canonical_behavior_action_token_ids") or []
                    ),
                    allowed_token_ids=canonical_support,
                    normalizer_atol=1e-6,
                )
            else:
                (
                    behavior_selected_logps,
                    behavior_full_logps,
                    behavior_support_mask,
                    behavior_norm_error,
                    behavior_selected_echo_diff,
                ) = materialize_position_canonical_behavior_policy(
                    input_ids[:, 1:],
                    response_masks,
                    action_ids=list(trajectory.get("action_ids") or []),
                    selected_log_probs=list(trajectory.get("action_logprobs") or []),
                    full_log_probs=list(
                        trajectory.get("canonical_behavior_action_logprobs") or []
                    ),
                    behavior_action_token_ids_by_position=list(
                        trajectory.get(
                            "canonical_behavior_action_token_ids_by_position"
                        )
                        or []
                    ),
                    allowed_token_ids_by_position=positional_supports,
                    normalizer_atol=1e-6,
                )
            if canonical_learner_full_logps is None:
                raise RuntimeError("canonical learner full policy trace is missing")
            if behavior_support_mask is not None and not torch.equal(
                behavior_support_mask, canonical_learner_support_mask
            ):
                raise RuntimeError(
                    "canonical behavior and learner positional supports disagree"
                )
            canonical_behavior_infos.update(
                canonical_behavior_overlap_diagnostics(
                    behavior_selected_logps,
                    behavior_full_logps,
                    logps,
                    canonical_learner_full_logps,
                    response_masks,
                    support_mask=behavior_support_mask,
                )
            )
            selected_diff = torch.abs(logps - behavior_selected_logps)[
                response_masks.to(torch.bool)
            ].max()
            canonical_behavior_infos.update(
                {
                    "canonical_behavior_denominator_actor": torch.tensor(
                        1.0, device=input_ids.device
                    ),
                    "canonical_behavior_q_row_count": response_masks.sum(),
                    "canonical_behavior_q_support_min": torch.tensor(
                        min(len(support) for support in positional_supports),
                        device=input_ids.device,
                    ),
                    "canonical_behavior_q_support_max": torch.tensor(
                        max(len(support) for support in positional_supports),
                        device=input_ids.device,
                    ),
                    "canonical_behavior_q_norm_error_max": torch.tensor(
                        behavior_norm_error,
                        dtype=torch.float32,
                        device=input_ids.device,
                    ),
                    "canonical_behavior_selected_echo_diff_max": torch.tensor(
                        behavior_selected_echo_diff,
                        dtype=torch.float32,
                        device=input_ids.device,
                    ),
                    # Retain the old name as a diagnostic only; the gate now
                    # checks full behavior overlap rather than equality.
                    "canonical_actor_logp_diff_max": selected_diff,
                }
            )
            old_logps = behavior_selected_logps
            logging.info(
                "canonical behavior overlap: ratio=[%.6f, %.6f] tv_max=%.6f "
                "kl_actor_learner_max=%.6f kl_learner_actor_max=%.6f "
                "sequence_ess=%.6f prefix_ess_min=%.6f selected_diff=%.6f",
                float(canonical_behavior_infos["canonical_behavior_ratio_min"]),
                float(canonical_behavior_infos["canonical_behavior_ratio_max"]),
                float(canonical_behavior_infos["canonical_behavior_tv_max"]),
                float(
                    canonical_behavior_infos["canonical_behavior_kl_actor_learner_max"]
                ),
                float(
                    canonical_behavior_infos["canonical_behavior_kl_learner_actor_max"]
                ),
                float(
                    canonical_behavior_infos["canonical_behavior_sequence_ess_fraction"]
                ),
                float(
                    canonical_behavior_infos[
                        "canonical_behavior_prefix_ess_fraction_min"
                    ]
                ),
                float(selected_diff),
            )

        if self.ref_model is not None:
            all_ref_logps = []
            ref_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(
                self.ref_model
            )
            with torch.no_grad():
                for i in range(0, len(input_ids), args.train_batch_size_per_device):
                    batch_end = min(
                        i + args.train_batch_size_per_device, len(input_ids)
                    )
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
                    batch_ref_logps, _ = self._policy_logps_and_optional_entropy(
                        batch_ref_logits,
                        batch_input_ids,
                        response_masks[batch_inds],
                        need_entropy=False,
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
        # Freeze the ordinary task-centered advantage before any separately
        # added semantic term. E44 uses this only to form detached xDr row
        # weights, while the actor below still receives the combined advantage.
        task_advantages_for_xdr = advantages.detach()
        if outcome_collision_outside_advantage is not None:
            base_advantages = advantages.detach()
            semantic_advantages = outcome_collision_outside_advantage
            advantages = add_outcome_collision_outside_centering_advantage(
                advantages,
                semantic_advantages,
            )
            combined_advantages = advantages.detach()
            base_advantage_rms = torch.sqrt(base_advantages.square().mean())
            semantic_advantage_rms = torch.sqrt(semantic_advantages.square().mean())
            combined_advantage_rms = torch.sqrt(combined_advantages.square().mean())
            outcome_collision_infos.update(
                {
                    "outcome_collision_outside_base_advantage_mean": (
                        base_advantages.mean()
                    ),
                    "outcome_collision_outside_base_advantage_abs_mean": (
                        base_advantages.abs().mean()
                    ),
                    "outcome_collision_outside_base_advantage_rms": (
                        base_advantage_rms
                    ),
                    "outcome_collision_outside_base_advantage_nonzero_fraction": (
                        base_advantages.ne(0).float().mean()
                    ),
                    "outcome_collision_outside_semantic_advantage_mean": (
                        semantic_advantages.mean()
                    ),
                    "outcome_collision_outside_semantic_advantage_min": (
                        semantic_advantages.min()
                    ),
                    "outcome_collision_outside_semantic_advantage_max": (
                        semantic_advantages.max()
                    ),
                    "outcome_collision_outside_semantic_advantage_abs_mean": (
                        semantic_advantages.abs().mean()
                    ),
                    "outcome_collision_outside_semantic_advantage_rms": (
                        semantic_advantage_rms
                    ),
                    "outcome_collision_outside_semantic_advantage_nonzero_fraction": (
                        semantic_advantages.ne(0).float().mean()
                    ),
                    "outcome_collision_outside_combined_advantage_mean": (
                        combined_advantages.mean()
                    ),
                    "outcome_collision_outside_combined_advantage_abs_mean": (
                        combined_advantages.abs().mean()
                    ),
                    "outcome_collision_outside_combined_advantage_rms": (
                        combined_advantage_rms
                    ),
                    "outcome_collision_outside_combined_advantage_nonzero_fraction": (
                        combined_advantages.ne(0).float().mean()
                    ),
                }
            )
        if semantic_shannon_separate_advantage is not None:
            base_advantages = advantages.detach()
            semantic_advantages = semantic_shannon_separate_advantage
            advantages = add_semantic_shannon_separate_advantage(
                advantages,
                semantic_advantages,
            )
            combined_advantages = advantages.detach()
            semantic_shannon_infos.update(
                {
                    "semantic_shannon_separate_base_advantage_mean": (
                        base_advantages.mean()
                    ),
                    "semantic_shannon_separate_base_advantage_abs_mean": (
                        base_advantages.abs().mean()
                    ),
                    "semantic_shannon_separate_base_advantage_rms": torch.sqrt(
                        base_advantages.square().mean()
                    ),
                    "semantic_shannon_separate_base_advantage_nonzero_fraction": (
                        base_advantages.ne(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_mean": (
                        semantic_advantages.mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_min": (
                        semantic_advantages.min()
                    ),
                    "semantic_shannon_separate_semantic_advantage_max": (
                        semantic_advantages.max()
                    ),
                    "semantic_shannon_separate_semantic_advantage_abs_mean": (
                        semantic_advantages.abs().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_rms": (
                        torch.sqrt(semantic_advantages.square().mean())
                    ),
                    "semantic_shannon_separate_semantic_advantage_positive_fraction": (
                        semantic_advantages.gt(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_negative_fraction": (
                        semantic_advantages.lt(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_zero_fraction": (
                        semantic_advantages.eq(0).float().mean()
                    ),
                    "semantic_shannon_separate_semantic_advantage_nonzero_fraction": (
                        semantic_advantages.ne(0).float().mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_mean": (
                        combined_advantages.mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_abs_mean": (
                        combined_advantages.abs().mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_rms": torch.sqrt(
                        combined_advantages.square().mean()
                    ),
                    "semantic_shannon_separate_combined_advantage_nonzero_fraction": (
                        combined_advantages.ne(0).float().mean()
                    ),
                }
            )
        if online_canonical_advantage is not None:
            base_advantages = advantages.detach()
            advantages = advantages + online_canonical_advantage
            combined_advantages = advantages.detach()
            online_canonical_infos.update(
                {
                    "online_canonical_separate_base_advantage_mean": (
                        base_advantages.mean()
                    ),
                    "online_canonical_separate_base_advantage_rms": torch.sqrt(
                        base_advantages.square().mean()
                    ),
                    "online_canonical_separate_combined_advantage_mean": (
                        combined_advantages.mean()
                    ),
                    "online_canonical_separate_combined_advantage_rms": (
                        torch.sqrt(combined_advantages.square().mean())
                    ),
                    "online_canonical_advantage_applied_after_task_centering": (
                        torch.tensor(1.0, device=final_rewards.device)
                    ),
                }
            )
        row_weights = None
        extra_infos: dict[str, torch.Tensor] = {
            **canonical_behavior_infos,
            **diayn_infos,
            **outcome_collision_infos,
            **semantic_shannon_infos,
            **online_canonical_infos,
        }
        configured_xdr_tau = float(getattr(args, "xdr_tau", math.inf))
        tau_controller = getattr(self, "_xdr_tau_controller", None)
        xdr_tau = (
            float(tau_controller.current_tau)
            if tau_controller is not None
            else configured_xdr_tau
        )
        seed_alpha = float(getattr(args, "seed_entropy_alpha", 0.0) or 0.0)
        if math.isfinite(xdr_tau) and self.args.critic_type == "drgrpo":
            # xDr.GRPO: per-candidate Dr.GRPO utilities at the rollout policy
            # (ratio=1): U_i = A_i * T_i / T_max. Weights are computed once per
            # rollout batch and frozen for the update, like the advantages.
            extra_infos["xdr_tau_used"] = torch.tensor(
                xdr_tau, dtype=torch.float32, device=final_rewards.device
            )
            per_group_tau = None
            if bool(getattr(args, "xdr_mode_adaptive", False)):
                # Mode-adaptive tempering: tau_x = tau0 / log(1 + kappa_x),
                # with kappa_x the number of distinct canonical answer modes
                # observed among the group's correct candidates. Prompts where
                # more modes are already in play get sharper attenuation of
                # negative-advantage gradients (they have the most to lose).
                num_rows = int(input_ids.size(0))
                refs = list(trajectory.get("references") or [])
                refs = (refs + [None] * num_rows)[:num_rows]
                refs_grouped = [
                    refs[i : i + args.num_samples]
                    for i in range(0, num_rows, args.num_samples)
                ]
                keys_grouped = self._seed_answer_keys_grouped(
                    input_ids, response_masks, args.num_samples, refs_grouped
                )
                correct = final_rewards.detach().reshape(-1, args.num_samples) > 0
                kappas = []
                for g, group_keys in enumerate(keys_grouped):
                    modes = {
                        key if key is not None else ("__u__", g, i)
                        for i, key in enumerate(group_keys)
                        if bool(correct[g, i])
                    }
                    kappas.append(max(len(modes), 1))
                per_group_tau = configured_xdr_tau / torch.log1p(
                    torch.tensor(
                        kappas, dtype=torch.float32, device=final_rewards.device
                    )
                )
                extra_infos["xdr_adaptive_tau_mean"] = per_group_tau.mean().detach()
            xdr_task_advantage_weights = bool(
                getattr(args, "xdr_task_advantage_weights", False)
            )
            xdr_weight_advantages = (
                task_advantages_for_xdr if xdr_task_advantage_weights else advantages
            )
            extra_infos["xdr_task_advantage_weights_active"] = torch.tensor(
                float(xdr_task_advantage_weights),
                dtype=torch.float32,
                device=final_rewards.device,
            )
            extra_infos["xdr_weight_advantage_mean"] = (
                xdr_weight_advantages.detach().mean()
            )
            extra_infos["xdr_weight_advantage_rms"] = torch.sqrt(
                xdr_weight_advantages.detach().square().mean()
            )
            row_weights = compute_xdr_row_weights(
                xdr_weight_advantages,
                response_masks.sum(dim=1),
                num_samples=args.num_samples,
                tau=xdr_tau,
                t_max=int(args.generate_max_length),
                loss_masks=loss_masks,
                per_group_tau=per_group_tau,
            )
        elif seed_alpha > 0 and self.args.critic_type == "drgrpo":
            # SEED-Dr.GRPO: per-prompt semantic-entropy scaling. Cluster the
            # group by canonical final answer, weight clusters by the policy's
            # length-normalized sequence likelihood, and scale the prompt's
            # rows by (1 + (alpha/log G) * H_sem)^{-1}. The per-row references
            # must reach the clusterer so modebench answers (countdown
            # expressions, colorings) reduce to canonical mode keys rather
            # than surface forms.
            num_rows = int(input_ids.size(0))
            references = list(trajectory.get("references") or [])
            references = (references + [None] * num_rows)[:num_rows]
            references_grouped = [
                references[i : i + args.num_samples]
                for i in range(0, num_rows, args.num_samples)
            ]
            seq_logp_sums = (logps * response_masks.float()).sum(dim=1)
            token_counts_raw = response_masks.sum(dim=1)
            token_counts = token_counts_raw.clamp(min=1).float()
            answer_keys_grouped = self._seed_answer_keys_grouped(
                input_ids,
                response_masks,
                args.num_samples,
                references_grouped,
            )
            answer_keys = [key for group in answer_keys_grouped for key in group]
            # Rows with no response tokens would otherwise take the maximal
            # normalized logp of exactly 0; exclude them from the cluster
            # softmax alongside loss-masked rows.
            effective_masks = loss_masks * (token_counts_raw > 0).float()
            row_weights = compute_seed_row_weights(
                seq_logp_sums / token_counts,
                answer_keys,
                num_samples=args.num_samples,
                alpha=seed_alpha,
                loss_masks=effective_masks,
            )
            extra_infos["seed_prompt_scale_mean"] = (
                row_weights.view(-1, args.num_samples)[:, 0].mean().detach()
            )
        if self.args.critic_type in ("grpo", "drgrpo"):
            # Aggregation diagnostics (effective active rollouts, incorrect-
            # mass share) are defined identically for every quartet arm from
            # the realized per-row aggregation weights: uniform for Dr.GRPO
            # and Token-MaxEnt, prompt-rescaled uniform for SEED, tempered
            # softmax for xDr.GRPO. Masking by loss_masks restricts each
            # group's distribution to the rows that actually train (a no-op
            # for xdr weights, which already zero masked rows).
            diag_weights = (
                row_weights if row_weights is not None else torch.ones_like(loss_masks)
            ) * loss_masks
            extra_infos.update(
                aggregation_group_diagnostics(
                    diag_weights, final_rewards, num_samples=args.num_samples
                )
            )
        return self._baseline_update_with_precomputed_advantages(
            input_ids=input_ids,
            att_mask=att_mask,
            prompt_id_lens=prompt_id_lens,
            loss_masks=loss_masks,
            response_masks=response_masks,
            logps=old_logps,
            ref_logps=ref_logps,
            advantages=advantages,
            final_rewards=task_final_rewards,
            returns=returns if self.args.critic_type == "ppo" else None,
            values=values if self.args.critic_type == "ppo" else None,
            policy_vocab_upper_bound=policy_vocab_upper_bound,
            row_weights=row_weights,
            extra_infos=extra_infos,
            canonical_replay_groups=canonical_replay_groups,
        )

    # Dr. GRPO Modification 2: remove difficulty bias by computing the MC
    # advantage without dividing by std, except for standard GRPO compatibility.
    def compute_monte_carlo_advantages(
        self,
        rewards: torch.Tensor,
        response_masks=None,
    ) -> torch.Tensor:
        del response_masks
        rewards = rewards.sum(-1)
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if getattr(self.args, "critic_type", "grpo") == "grpo":
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples,
                dim=0,
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages
