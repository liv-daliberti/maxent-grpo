"""Dr.X/listwise learner helpers."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from oat.utils.ops import masked_mean

from .. import semantic_clusters as _semantic_clusters
from ..args import ZeroMathArgs
from ..correctness_schedule import build_correctness_scheduled_settings
from ..drx_targets import (
    apply_neutral_tiebreak_to_advantages,
    build_drgrpo_token_active_row_mask,
    compute_drx_projection_sequence_coefficients,
)
from ..listwise import (
    aggregate_masked_row_values,
    build_drx_target_bundle,
    build_listwise_q_targets,
    build_padded_action_logprobs,
    collect_weight_entropy_stats,
    coerce_non_negative_float,
    compute_group_centered_advantages,
    compute_listwise_centered_advantages,
    compute_listwise_weights,
    compute_normalized_semantic_cluster_entropy,
    flatten_prompt_major_tensor,
    iter_grouped_minibatch_indices,
    mask_and_normalize_listwise_q_targets,
    masked_group_log_softmax,
    normalize_maxent_clip_mode,
    reshape_prompt_major_tensor,
)
from ..ppo_clip import (
    compute_listwise_clip_advantages,
    compute_listwise_sequence_coefficients,
    compute_token_level_clip_loss,
)
from ..semantic_features import (
    build_candidate_response_features,
)
from ..semantic_utility import compute_quality_centered_semantic_drx_utilities
from ..stats_utils import (
    compute_correctness_group_rate_infos,
    finalize_listwise_info_stats,
    finalize_row_sharded_info_stats,
)
from .drx_backward import ZeroMathDrxBackwardMixin
from .drx_logging import ZeroMathDrxLoggingMixin
from .drx_semantics import ZeroMathDrxSemanticMixin


_VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD = 0.999


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
    return _semantic_clusters.build_runtime_semantic_cluster_bundle(
        args=args,
        default_method=default_method,
        final_answer_keys_grouped=final_answer_keys_grouped,
        valid_row_mask_grouped=valid_row_mask_grouped,
        reasoning_signature_keys_grouped=reasoning_signature_keys_grouped,
        reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
        reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
    )


class ZeroMathDrxMixin(
    ZeroMathDrxLoggingMixin,
    ZeroMathDrxBackwardMixin,
    ZeroMathDrxSemanticMixin,
):
    """Dr.X/listwise path helpers that do not own the full optimizer step."""

    def _use_row_sharded_exact_drx_path(self) -> bool:
        return self.objective == "maxent_listwise" and 0 < int(
            getattr(self.args, "train_batch_size_per_device", 0) or 0
        ) < int(getattr(self.args, "num_samples", 0) or 0)

    def _listwise_token_clip_constant_normalizer(self) -> float | None:
        if (
            self.args.critic_type == "drgrpo"
            and str(
                getattr(
                    self.args,
                    "maxent_drgrpo_token_length_normalizer",
                    "max_length",
                )
            )
            == "response_length"
        ):
            return None
        if self.args.critic_type == "drgrpo":
            return float(max(int(self.args.generate_max_length), 1))
        return None

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
                "clip_mode=%s token_clip_primary=%s "
                "drgrpo_token_primary=%s "
                "drgrpo_token_advantage_source=%s "
                "drgrpo_token_length_normalizer=%s "
                "sequence_aux_coef=%s "
                "exact_drx_weighted_drgrpo=%s neutral_projection_coef=%s "
                "semantic_cluster_method=%s "
                "semantic_similarity_threshold=%s "
                "semantic_embedding_similarity_threshold=%s "
                "semantic_embedding_max_tokens=%s "
                "semantic_cluster_max_tokens=%s "
                "semantic_spectral_max_clusters=%s "
                "semantic_spectral_eigengap_min=%s "
                "semantic_correctness_target_frac=%s "
                "semantic_correctness_sharpness=%s "
                "semantic_remix_mode=%s "
                "neutral_tiebreak_alpha=%s "
                "tiebreak_anchor=%s tiebreak_clip_max=%s "
                "competitive_mode_tau=%s competitive_mode_gap=%s "
                "competitive_mode_top_k=%s competitive_mode_budget_max=%s "
                "competitive_mode_budget_scale=%s competitive_mode_intra_tau=%s "
                "prompt_select_min_alpha_frac=%s competitive_mode_positive_only=%s "
                "correctness_schedule_enabled=%s correctness_schedule_ema_decay=%s "
                "correctness_schedule_low=%s correctness_schedule_high=%s "
                "correctness_schedule_budget_max_early=%s correctness_schedule_budget_max_late=%s "
                "correctness_schedule_prompt_select_early=%s correctness_schedule_prompt_select_late=%s "
                "correctness_schedule_mode_tau_early=%s correctness_schedule_mode_tau_late=%s "
                "correctness_schedule_intra_tau_early=%s correctness_schedule_intra_tau_late=%s "
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
                bool(args.maxent_token_clip_primary),
                bool(args.maxent_drgrpo_token_primary),
                str(args.maxent_drgrpo_token_advantage_source),
                str(args.maxent_drgrpo_token_length_normalizer),
                float(args.maxent_sequence_aux_coef),
                bool(args.maxent_drgrpo_token_primary),
                float(args.maxent_neutral_projection_coef),
                str(args.maxent_semantic_cluster_method),
                float(args.maxent_semantic_similarity_threshold),
                float(args.maxent_semantic_embedding_similarity_threshold),
                int(args.maxent_semantic_embedding_max_tokens),
                int(args.maxent_semantic_cluster_max_tokens),
                int(args.maxent_semantic_spectral_max_clusters),
                float(args.maxent_semantic_spectral_eigengap_min),
                float(args.maxent_semantic_correctness_target_frac),
                float(args.maxent_semantic_correctness_sharpness),
                str(args.maxent_semantic_remix_mode),
                float(args.maxent_reward_shaping_alpha),
                str(args.maxent_tiebreak_anchor),
                float(args.maxent_tiebreak_clip_max),
                float(args.maxent_competitive_mode_tau),
                float(args.maxent_competitive_mode_gap),
                int(args.maxent_competitive_mode_top_k),
                float(args.maxent_competitive_mode_budget_max),
                float(args.maxent_competitive_mode_budget_scale),
                float(args.maxent_competitive_mode_intra_tau),
                float(args.maxent_prompt_select_min_alpha_frac),
                bool(args.maxent_competitive_mode_positive_only),
                bool(args.maxent_correctness_schedule_enabled),
                float(args.maxent_correctness_schedule_ema_decay),
                float(args.maxent_correctness_schedule_low),
                float(args.maxent_correctness_schedule_high),
                float(args.maxent_correctness_schedule_budget_max_early),
                float(args.maxent_correctness_schedule_budget_max_late),
                float(
                    args.maxent_correctness_schedule_prompt_select_min_alpha_frac_early
                ),
                float(
                    args.maxent_correctness_schedule_prompt_select_min_alpha_frac_late
                ),
                float(args.maxent_correctness_schedule_mode_tau_early),
                float(args.maxent_correctness_schedule_mode_tau_late),
                float(args.maxent_correctness_schedule_intra_tau_early),
                float(args.maxent_correctness_schedule_intra_tau_late),
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
        token_clip_primary = bool(args.maxent_token_clip_primary)
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
        correctness_schedule_enabled = bool(
            getattr(args, "maxent_correctness_schedule_enabled", True)
        )
        correctness_schedule_ema_decay = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_correctness_schedule_ema_decay", 0.997),
                    default=0.995,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_low = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_correctness_schedule_low", 0.45),
                    default=0.45,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_high = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_correctness_schedule_high", 0.90),
                    default=0.90,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_budget_max_early = coerce_non_negative_float(
            getattr(args, "maxent_correctness_schedule_budget_max_early", 0.18),
            default=0.18,
        )
        correctness_schedule_budget_max_late = coerce_non_negative_float(
            getattr(args, "maxent_correctness_schedule_budget_max_late", 0.06),
            default=0.05,
        )
        correctness_schedule_prompt_select_early = min(
            max(
                coerce_non_negative_float(
                    getattr(
                        args,
                        "maxent_correctness_schedule_prompt_select_min_alpha_frac_early",
                        0.20,
                    ),
                    default=0.20,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_prompt_select_late = min(
            max(
                coerce_non_negative_float(
                    getattr(
                        args,
                        "maxent_correctness_schedule_prompt_select_min_alpha_frac_late",
                        0.55,
                    ),
                    default=0.55,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_mode_tau_early = max(
            coerce_non_negative_float(
                getattr(args, "maxent_correctness_schedule_mode_tau_early", 0.08),
                default=0.08,
            ),
            1e-8,
        )
        correctness_schedule_mode_tau_late = max(
            coerce_non_negative_float(
                getattr(args, "maxent_correctness_schedule_mode_tau_late", 0.03),
                default=0.03,
            ),
            1e-8,
        )
        correctness_schedule_intra_tau_early = max(
            coerce_non_negative_float(
                getattr(args, "maxent_correctness_schedule_intra_tau_early", 0.03),
                default=0.03,
            ),
            1e-8,
        )
        correctness_schedule_intra_tau_late = max(
            coerce_non_negative_float(
                getattr(args, "maxent_correctness_schedule_intra_tau_late", 0.005),
                default=0.005,
            ),
            1e-8,
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
        _, formatted_values = build_candidate_response_features(
            trajectory["action_ids"],
            tokenizer=self.tokenizer,
            template=args.prompt_template,
            device=device,
            dtype=reward_values.dtype,
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
        grouped_correctness_values = (grouped_reward_values / reward_scale_denom).clamp(
            min=0.0, max=1.0
        )
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
        logging.info(
            "listwise prep ready: behavior_logps=%s", tuple(behavior_logps.shape)
        )

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

        if self._use_row_sharded_exact_drx_path():
            return self._listwise_learning_step_row_sharded_exact_drx(
                input_ids=input_ids,
                att_mask=att_mask,
                response_masks=response_masks,
                loss_masks=loss_masks,
                behavior_logps=behavior_logps,
                grouped_reward_values=grouped_reward_values,
                grouped_correctness_values=grouped_correctness_values,
                grouped_formatted_values=grouped_formatted_values,
                behavior_seq_logps_grouped=behavior_seq_logps_grouped,
                behavior_seq_logps_grouped_raw=behavior_seq_logps_grouped_raw,
                ref_seq_logps_grouped=ref_seq_logps_grouped,
                ref_seq_logps_grouped_raw=ref_seq_logps_grouped_raw,
            )

        total_rows = int(input_ids.size(0))
        grad_acc_step = max(int(self.strategy.grad_acc_step), 1)
        local_minibatches_per_epoch = max(
            math.ceil(total_rows / max(int(args.train_batch_size_per_device), 1)),
            1,
        )
        total_optimizer_updates = max(
            math.ceil(
                args.num_ppo_epochs * local_minibatches_per_epoch / grad_acc_step
            ),
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
                if (
                    drgrpo_token_primary
                    and exact_drx_weight_source == "sequence_clipped"
                ):
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
                mb_ref_seq_grouped_raw = ref_seq_logps_grouped_raw[
                    prompt_batch_inds
                ].to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_reward_grouped = grouped_reward_values[prompt_batch_inds].to(
                    device=policy_seq_logps_grouped.device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_correctness_grouped = grouped_correctness_values[
                    prompt_batch_inds
                ].to(
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
                        utility_dtype_info = torch.finfo(
                            probe_drgrpo_utility_grouped.dtype
                        )
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
                        neutral_group_mask = (valid_group_counts <= 1) | (
                            (valid_utility_max - valid_utility_min) <= 1e-8
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
                        neutral_group_mask = (valid_group_counts <= 1) | (
                            (valid_q_max - valid_q_min) <= 1e-8
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
                    dist.all_reduce(
                        contributing_group_count_tensor, op=dist.ReduceOp.SUM
                    )
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

                schedule_any_correct_grouped = (
                    (
                        (
                            mb_correctness_grouped
                            >= _VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD
                        )
                        & grouped_loss_masks
                    )
                    .any(dim=1)
                    .to(policy_seq_logps_grouped.dtype)
                )
                schedule_contributing_group_mask = grouped_loss_masks.any(dim=1)
                schedule_any_correct_sum = (
                    schedule_any_correct_grouped
                    * schedule_contributing_group_mask.to(
                        dtype=policy_seq_logps_grouped.dtype
                    )
                ).sum()
                schedule_contributing_count = schedule_contributing_group_mask.to(
                    dtype=policy_seq_logps_grouped.dtype
                ).sum()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(schedule_any_correct_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(schedule_contributing_count, op=dist.ReduceOp.SUM)
                if float(schedule_contributing_count.item()) > 0.0:
                    schedule_batch_any_correct_mean = float(
                        (
                            schedule_any_correct_sum
                            / schedule_contributing_count.clamp(min=1.0)
                        ).item()
                    )
                else:
                    schedule_batch_any_correct_mean = 0.0
                schedule_settings = build_correctness_scheduled_settings(
                    enabled=correctness_schedule_enabled,
                    previous_ema=self._maxent_correctness_schedule_any_correct_ema,
                    batch_any_correct_mean=schedule_batch_any_correct_mean,
                    ema_decay=correctness_schedule_ema_decay,
                    correctness_low=correctness_schedule_low,
                    correctness_high=correctness_schedule_high,
                    static_budget_max=competitive_mode_budget_max,
                    static_prompt_select_min_alpha_frac=prompt_select_min_alpha_frac,
                    static_mode_tau=competitive_mode_tau,
                    static_intra_tau=competitive_mode_intra_tau,
                    budget_max_early=correctness_schedule_budget_max_early,
                    budget_max_late=correctness_schedule_budget_max_late,
                    prompt_select_min_alpha_frac_early=correctness_schedule_prompt_select_early,
                    prompt_select_min_alpha_frac_late=correctness_schedule_prompt_select_late,
                    mode_tau_early=correctness_schedule_mode_tau_early,
                    mode_tau_late=correctness_schedule_mode_tau_late,
                    intra_tau_early=correctness_schedule_intra_tau_early,
                    intra_tau_late=correctness_schedule_intra_tau_late,
                )
                self._maxent_correctness_schedule_any_correct_ema = (
                    schedule_settings.any_correct_ema
                )
                effective_competitive_mode_budget_max = float(
                    schedule_settings.budget_max
                )
                effective_prompt_select_min_alpha_frac = float(
                    schedule_settings.prompt_select_min_alpha_frac
                )
                effective_competitive_mode_tau = float(schedule_settings.mode_tau)
                effective_competitive_mode_intra_tau = float(
                    schedule_settings.intra_tau
                )

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
                    (
                        reasoning_trace_embeddings_grouped,
                        reasoning_trace_valid_row_mask_grouped,
                        semantic_trace_truncated_frac,
                    ) = self._semantic_trace_embeddings_grouped(
                        reasoning_trace_texts_grouped=reasoning_trace_texts_grouped,
                        valid_row_mask_grouped=grouped_loss_masks.detach(),
                    )
                    semantic_cluster_bundle = build_runtime_semantic_cluster_bundle(
                        args=args,
                        default_method="greedy",
                        final_answer_keys_grouped=final_answer_keys_grouped,
                        valid_row_mask_grouped=grouped_loss_masks,
                        reasoning_signature_keys_grouped=reasoning_signature_keys_grouped,
                        reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                        reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
                    )
                    answer_key_extracted_mask_grouped = (
                        torch.tensor(
                            [
                                [answer_key is not None for answer_key in prompt_keys]
                                for prompt_keys in final_answer_keys_grouped
                            ],
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.bool,
                        )
                        & grouped_loss_masks
                    )
                    trace_extracted_mask_grouped = (
                        torch.tensor(
                            [
                                [trace_text is not None for trace_text in prompt_rows]
                                for prompt_rows in reasoning_trace_texts_grouped
                            ],
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.bool,
                        )
                        & grouped_loss_masks
                    )
                    signature_extracted_mask_grouped = (
                        torch.tensor(
                            [
                                [signature is not None for signature in prompt_rows]
                                for prompt_rows in reasoning_signature_keys_grouped
                            ],
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.bool,
                        )
                        & grouped_loss_masks
                    )
                    semantic_valid_row_mask_grouped = (
                        semantic_cluster_bundle.semantic_valid_row_mask_grouped.to(
                            device=policy_seq_logps_grouped.device
                        )
                    )
                    semantic_answer_key_extracted_count = (
                        answer_key_extracted_mask_grouped.to(torch.float32).sum()
                    )
                    semantic_trace_extracted_count = trace_extracted_mask_grouped.to(
                        torch.float32
                    ).sum()
                    semantic_signature_extracted_count = (
                        signature_extracted_mask_grouped.to(torch.float32).sum()
                    )
                    semantic_valid_row_count = semantic_valid_row_mask_grouped.to(
                        torch.float32
                    ).sum()
                    valid_row_count_for_semantics = (
                        grouped_loss_masks.to(torch.float32).sum().clamp(min=1.0)
                    )
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
                        (
                            semantic_valid_row_count / valid_row_count_for_semantics
                        ).detach()
                    )
                    stats["listwise_semantic_trace_truncated_frac"].append(
                        semantic_trace_truncated_frac.detach()
                    )
                    if behavior_seq_grouped_raw is None:
                        behavior_seq_grouped_raw, _ = self._sequence_logps_grouped(
                            mb_behavior_logps,
                            mb_response_masks,
                            group_size,
                            length_normalize=False,
                            context="behavior log-probs (raw)",
                        )
                    output_entropy_grouped = (
                        -behavior_seq_grouped_raw.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        )
                        / token_counts_grouped.to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        ).clamp(min=1.0)
                    ).clamp(min=0.0)
                    (
                        exact_quality_utility_grouped,
                        exact_quality_grouped,
                        exact_semantic_drx_diag,
                    ) = compute_quality_centered_semantic_drx_utilities(
                        reward_grouped=mb_reward_grouped.detach(),
                        output_entropy_grouped=output_entropy_grouped.detach(),
                        semantic_entropy_lambda=float(args.semantic_entropy_lambda),
                        candidate_correctness_grouped=mb_correctness_grouped.detach(),
                        valid_row_mask_grouped=grouped_loss_masks.detach(),
                        semantic_correctness_target_frac=float(
                            args.maxent_semantic_correctness_target_frac
                        ),
                        semantic_correctness_sharpness=float(
                            args.maxent_semantic_correctness_sharpness
                        ),
                    )
                    exact_centered_advantages_grouped = (
                        compute_group_centered_advantages(
                            reward_grouped=exact_quality_utility_grouped.detach().to(
                                device=policy_seq_logps_grouped.device,
                                dtype=policy_seq_logps_grouped.dtype,
                            ),
                            valid_row_mask_grouped=grouped_loss_masks.detach(),
                        )
                    )
                    exact_quality_for_adv_grouped = exact_quality_grouped.detach().to(
                        device=policy_seq_logps_grouped.device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    exact_semantic_piece_grouped = (
                        exact_quality_utility_grouped.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        )
                        - exact_quality_for_adv_grouped
                    )
                    exact_correctness_adv_grouped = compute_group_centered_advantages(
                        reward_grouped=exact_quality_for_adv_grouped,
                        valid_row_mask_grouped=grouped_loss_masks.detach(),
                    )
                    exact_semantic_adv_grouped = compute_group_centered_advantages(
                        reward_grouped=exact_semantic_piece_grouped,
                        valid_row_mask_grouped=grouped_loss_masks.detach(),
                    )
                    quality_valid_values = exact_quality_grouped[grouped_loss_masks]
                    semantic_surprisal_valid_values = (
                        exact_semantic_drx_diag.semantic_surprisal_grouped[
                            grouped_loss_masks
                        ]
                    )
                    semantic_gate_valid_values = (
                        exact_semantic_drx_diag.semantic_gate_grouped[
                            grouped_loss_masks.any(dim=1)
                        ]
                        if exact_semantic_drx_diag.semantic_gate_grouped is not None
                        else torch.empty(
                            (0,),
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    utility_valid_values = exact_quality_utility_grouped[
                        grouped_loss_masks
                    ]
                    semantic_piece_valid_values = exact_semantic_piece_grouped[
                        grouped_loss_masks
                    ]
                    correctness_adv_valid_values = exact_correctness_adv_grouped[
                        grouped_loss_masks
                    ]
                    semantic_adv_valid_values = exact_semantic_adv_grouped[
                        grouped_loss_masks
                    ]
                    if correctness_adv_valid_values.numel() > 0:
                        correctness_adv_abs_mean = (
                            correctness_adv_valid_values.abs()
                            .mean()
                            .detach()
                            .to(torch.float32)
                        )
                        semantic_adv_abs_mean = (
                            semantic_adv_valid_values.abs()
                            .mean()
                            .detach()
                            .to(torch.float32)
                        )
                    else:
                        correctness_adv_abs_mean = torch.tensor(
                            0.0,
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                        semantic_adv_abs_mean = torch.tensor(
                            0.0,
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    valid_group_mask = grouped_loss_masks.detach().any(dim=1)
                    valid_count_grouped = (
                        grouped_loss_masks.detach()
                        .to(torch.float32)
                        .sum(dim=1)
                        .clamp(min=1.0)
                    )
                    correctness_frac_grouped = (
                        torch.where(
                            grouped_loss_masks.detach(),
                            exact_quality_for_adv_grouped.to(torch.float32),
                            torch.zeros_like(exact_quality_for_adv_grouped).to(
                                torch.float32
                            ),
                        ).sum(dim=1)
                        / valid_count_grouped
                    )
                    semantic_adv_abs_grouped = (
                        torch.where(
                            grouped_loss_masks.detach(),
                            exact_semantic_adv_grouped.abs().to(torch.float32),
                            torch.zeros_like(exact_semantic_adv_grouped).to(
                                torch.float32
                            ),
                        ).sum(dim=1)
                        / valid_count_grouped
                    )
                    all_wrong_group_mask = valid_group_mask & (
                        correctness_frac_grouped <= 1e-8
                    )
                    all_correct_group_mask = valid_group_mask & (
                        correctness_frac_grouped >= 1.0 - 1e-8
                    )
                    mixed_group_mask = (
                        valid_group_mask
                        & ~all_wrong_group_mask
                        & ~all_correct_group_mask
                    )

                    def _mean_group_or_zero(mask: torch.Tensor) -> torch.Tensor:
                        if bool(mask.any().item()):
                            return (
                                semantic_adv_abs_grouped[mask]
                                .mean()
                                .detach()
                                .to(torch.float32)
                            )
                        return torch.tensor(
                            0.0,
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )

                    semantic_adv_abs_all_wrong = _mean_group_or_zero(
                        all_wrong_group_mask
                    )
                    semantic_adv_abs_mixed = _mean_group_or_zero(mixed_group_mask)
                    semantic_adv_abs_all_correct = _mean_group_or_zero(
                        all_correct_group_mask
                    )
                    if bool(valid_group_mask.any().item()):
                        semantic_effective_group_frac = (
                            (
                                valid_group_mask
                                & (semantic_adv_abs_grouped > 1e-8)
                            )
                            .to(torch.float32)
                            .sum()
                            / valid_group_mask.to(torch.float32).sum().clamp(min=1.0)
                        ).detach()
                    else:
                        semantic_effective_group_frac = torch.tensor(
                            0.0,
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    if correctness_adv_valid_values.numel() > 0:
                        cosine_denom = (
                            correctness_adv_valid_values.norm()
                            * semantic_adv_valid_values.norm()
                        )
                        if bool((cosine_denom > 1e-12).item()):
                            semantic_correctness_adv_cosine = (
                                (
                                    correctness_adv_valid_values
                                    * semantic_adv_valid_values
                                ).sum()
                                / cosine_denom
                            ).detach().to(torch.float32)
                        else:
                            semantic_correctness_adv_cosine = torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                    else:
                        semantic_correctness_adv_cosine = torch.tensor(
                            0.0,
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    stats["listwise_exact_quality_mean"].append(
                        (
                            quality_valid_values.mean().detach().to(torch.float32)
                            if quality_valid_values.numel() > 0
                            else torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                    )
                    stats["listwise_exact_semantic_surprisal_mean"].append(
                        (
                            semantic_surprisal_valid_values.mean()
                            .detach()
                            .to(torch.float32)
                            if semantic_surprisal_valid_values.numel() > 0
                            else torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                    )
                    stats["listwise_exact_semantic_gate_mean"].append(
                        (
                            semantic_gate_valid_values.mean().detach().to(torch.float32)
                            if semantic_gate_valid_values.numel() > 0
                            else torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                    )
                    stats["listwise_exact_semantic_piece_mean"].append(
                        (
                            semantic_piece_valid_values.mean().detach().to(torch.float32)
                            if semantic_piece_valid_values.numel() > 0
                            else torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                    )
                    stats["listwise_exact_correctness_adv_abs_mean"].append(
                        correctness_adv_abs_mean
                    )
                    stats["listwise_exact_semantic_adv_abs_mean"].append(
                        semantic_adv_abs_mean
                    )
                    stats[
                        "listwise_exact_semantic_to_correctness_adv_ratio"
                    ].append(
                        semantic_adv_abs_mean
                        / (correctness_adv_abs_mean + 1e-8)
                    )
                    stats["listwise_exact_semantic_adv_fraction"].append(
                        semantic_adv_abs_mean
                        / (semantic_adv_abs_mean + correctness_adv_abs_mean + 1e-8)
                    )
                    stats["listwise_exact_semantic_adv_abs_mean_all_wrong"].append(
                        semantic_adv_abs_all_wrong
                    )
                    stats["listwise_exact_semantic_adv_abs_mean_mixed"].append(
                        semantic_adv_abs_mixed
                    )
                    stats["listwise_exact_semantic_adv_abs_mean_all_correct"].append(
                        semantic_adv_abs_all_correct
                    )
                    stats["listwise_exact_semantic_effective_group_frac"].append(
                        semantic_effective_group_frac
                    )
                    stats["listwise_exact_semantic_correctness_adv_cosine"].append(
                        semantic_correctness_adv_cosine
                    )
                    stats["listwise_exact_utility_mean"].append(
                        (
                            utility_valid_values.mean().detach().to(torch.float32)
                            if utility_valid_values.numel() > 0
                            else torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                    )
                    (
                        tie_break_reward_grouped,
                        tie_break_diag,
                        tie_break_anchor_source,
                    ) = self._semantic_neutral_tiebreak_rewards_grouped(
                        behavior_seq_logps_grouped=behavior_seq_grouped,
                        reference_seq_logps_grouped=mb_ref_seq_grouped,
                        candidate_correctness_grouped=mb_correctness_grouped,
                        cluster_ids_grouped=semantic_cluster_bundle.cluster_ids_grouped,
                        valid_row_mask_grouped=grouped_loss_masks,
                    )
                    (
                        adjusted_probe_row_advantages_grouped,
                        raw_neutral_group_mask,
                        tiebreak_applied_group_mask,
                    ) = apply_neutral_tiebreak_to_advantages(
                        row_advantages_grouped=probe_row_advantages_grouped,
                        utility_grouped=probe_drgrpo_utility_grouped.detach(),
                        tiebreak_values_grouped=tie_break_reward_grouped.to(
                            device=policy_seq_logps_grouped.device,
                            dtype=probe_row_advantages_grouped.dtype,
                        ),
                        valid_row_mask_grouped=grouped_loss_masks,
                        enabled=float(args.maxent_reward_shaping_alpha) > 0.0,
                        neutral_eps=1e-8,
                    )
                    probe_row_advantages_grouped = (
                        adjusted_probe_row_advantages_grouped.to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        )
                    )
                    probe_row_advantages = flatten_prompt_major_tensor(
                        probe_row_advantages_grouped
                    ).to(device=new_logps.device, dtype=new_logps.dtype)
                    if exact_drx_weight_source == "sequence_clipped":
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
                        adjusted_probe_drgrpo_pg_row_loss, _, _, _ = (
                            compute_token_level_clip_loss(
                                new_logps=new_logps,
                                behavior_logps=mb_behavior_logps.to(new_logps.dtype),
                                response_masks=mb_response_masks,
                                row_advantages=probe_row_advantages,
                                clip_low=float(args.cliprange),
                                clip_high=float(args.cliprange),
                                constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                            )
                        )
                        if exact_drx_weight_source == "clipped":
                            adjusted_probe_drgrpo_weight_scores = (
                                -adjusted_probe_drgrpo_pg_row_loss
                            )
                        elif exact_drx_weight_source == "unclipped":
                            adjusted_probe_drgrpo_weight_scores = aggregate_masked_row_values(
                                probe_token_ratio * probe_row_advantages.unsqueeze(1),
                                mb_response_masks,
                                constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                            ).to(device=new_logps.device, dtype=new_logps.dtype)
                        elif exact_drx_weight_source == "local_linear":
                            adjusted_probe_drgrpo_weight_scores = aggregate_masked_row_values(
                                torch.ones_like(new_logps)
                                * probe_row_advantages.unsqueeze(1),
                                mb_response_masks,
                                constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                            ).to(device=new_logps.device, dtype=new_logps.dtype)
                        else:
                            raise ValueError(
                                "Unsupported maxent_exact_drx_weight_source: "
                                f"{exact_drx_weight_source}"
                            )
                        probe_drgrpo_utility_grouped = reshape_prompt_major_tensor(
                            adjusted_probe_drgrpo_weight_scores.to(
                                device=policy_seq_logps_grouped.device,
                                dtype=policy_seq_logps_grouped.dtype,
                            ),
                            group_size,
                        )
                        if probe_drgrpo_utility_grouped is None:
                            raise ValueError(
                                "Could not reshape the tie-broken Dr.GRPO per-candidate utilities into prompt groups."
                            )
                    stats["listwise_raw_neutral_group_frac"].append(
                        raw_neutral_group_mask.to(torch.float32).mean().detach()
                    )
                    stats["listwise_neutral_tiebreak_applied_group_frac"].append(
                        tiebreak_applied_group_mask.to(torch.float32).mean().detach()
                    )
                    stats["listwise_neutral_tiebreak_alpha"].append(
                        torch.tensor(
                            float(args.maxent_reward_shaping_alpha),
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    stats["listwise_neutral_tiebreak_anchor_source_behavior"].append(
                        torch.tensor(
                            1.0 if tie_break_anchor_source == "behavior" else 0.0,
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    stats["listwise_neutral_tiebreak_anchor_source_reference"].append(
                        torch.tensor(
                            1.0 if tie_break_anchor_source == "reference" else 0.0,
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
                    )
                    if tie_break_diag is not None:
                        valid_tiebreak_values = tie_break_reward_grouped[
                            grouped_loss_masks
                        ]
                        stats["listwise_neutral_tiebreak_prompt_alpha_mean"].append(
                            tie_break_diag.prompt_alpha_grouped.mean()
                            .detach()
                            .to(torch.float32)
                        )
                        stats[
                            "listwise_neutral_tiebreak_correct_anchor_mass_mean"
                        ].append(
                            tie_break_diag.correct_anchor_mass_grouped.mean()
                            .detach()
                            .to(torch.float32)
                        )
                        stats["listwise_neutral_tiebreak_reward_mean"].append(
                            (
                                valid_tiebreak_values.mean().detach().to(torch.float32)
                                if valid_tiebreak_values.numel() > 0
                                else torch.tensor(
                                    0.0,
                                    device=policy_seq_logps_grouped.device,
                                    dtype=torch.float32,
                                )
                            )
                        )
                    else:
                        stats["listwise_neutral_tiebreak_prompt_alpha_mean"].append(
                            torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                        stats[
                            "listwise_neutral_tiebreak_correct_anchor_mass_mean"
                        ].append(
                            torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                        stats["listwise_neutral_tiebreak_reward_mean"].append(
                            torch.tensor(
                                0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
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
                        float(effective_competitive_mode_budget_max)
                        * semantic_alpha_raw_grouped
                        * grouped_loss_masks.any(dim=1).to(
                            dtype=policy_seq_logps_grouped.dtype
                        )
                    )
                    semantic_mass_weights_grouped = None
                    if str(args.maxent_semantic_remix_mode) == "anchor_rare":
                        semantic_behavior_seq_grouped = (
                            behavior_seq_grouped_raw
                            if exact_drx_weight_source == "sequence_clipped"
                            else behavior_seq_grouped
                        )
                        (
                            anchor_semantic_utility_grouped,
                            semantic_mass_weights_grouped,
                            semantic_anchor_mass_source,
                        ) = self._anchor_relative_semantic_mass_weights_grouped(
                            behavior_seq_logps_grouped=semantic_behavior_seq_grouped,
                            reference_seq_logps_grouped=semantic_ref_seq_grouped,
                            valid_row_mask_grouped=grouped_loss_masks,
                            tau=current_tau,
                            candidate_kl_coef=candidate_kl_coef,
                        )
                        stats["listwise_semantic_anchor_mass_source_behavior"].append(
                            torch.tensor(
                                1.0
                                if semantic_anchor_mass_source == "behavior"
                                else 0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                        stats["listwise_semantic_anchor_mass_source_reference"].append(
                            torch.tensor(
                                1.0
                                if semantic_anchor_mass_source == "reference"
                                else 0.0,
                                device=policy_seq_logps_grouped.device,
                                dtype=torch.float32,
                            )
                        )
                        valid_anchor_utility = anchor_semantic_utility_grouped[
                            grouped_loss_masks
                        ]
                        stats["listwise_semantic_anchor_utility_mean"].append(
                            (
                                valid_anchor_utility.mean().detach().to(torch.float32)
                                if valid_anchor_utility.numel() > 0
                                else torch.tensor(
                                    0.0,
                                    device=policy_seq_logps_grouped.device,
                                    dtype=torch.float32,
                                )
                            )
                        )
                    cluster_ids_grouped = (
                        semantic_cluster_bundle.cluster_ids_grouped.to(
                            device=policy_seq_logps_grouped.device
                        )
                    )
                    drx_bundle = build_drx_target_bundle(
                        utility_grouped=exact_quality_utility_grouped.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=policy_seq_logps_grouped.dtype,
                        ),
                        ref_seq_logps_grouped=semantic_ref_seq_grouped,
                        valid_row_mask_grouped=grouped_loss_masks.detach(),
                        competitive_mode_tau=effective_competitive_mode_tau,
                        competitive_mode_gap=competitive_mode_gap,
                        competitive_mode_top_k=competitive_mode_top_k,
                        competitive_mode_budget_grouped=semantic_explore_budget_grouped.detach(),
                        competitive_mode_budget_max=effective_competitive_mode_budget_max,
                        competitive_mode_intra_tau=effective_competitive_mode_intra_tau,
                        prompt_select_min_alpha_frac=effective_prompt_select_min_alpha_frac,
                        competitive_mode_positive_only=competitive_mode_positive_only,
                        semantic_guard_max_expected_len_delta=semantic_guard_max_expected_len_delta,
                        semantic_guard_max_expected_format_drop=semantic_guard_max_expected_format_drop,
                        tau=current_tau,
                        candidate_kl_coef=candidate_kl_coef,
                        neutral_eps=1e-8,
                        neutral_projection_coef=neutral_projection_coef,
                        semantic_remix_mode=str(args.maxent_semantic_remix_mode),
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
                    active_group_count = int(
                        active_group_mask.to(torch.int64).sum().item()
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
                        global_active_group_count = int(
                            active_group_count_tensor.item()
                        )
                    global_contributing_group_count = contributing_group_count
                    if dist.is_available() and dist.is_initialized():
                        contributing_group_count_tensor = torch.tensor(
                            float(contributing_group_count),
                            device=policy_seq_logps_grouped.device,
                        )
                        dist.all_reduce(
                            contributing_group_count_tensor, op=dist.ReduceOp.SUM
                        )
                        global_contributing_group_count = int(
                            contributing_group_count_tensor.item()
                        )
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
                        projection_group_weight_tensor = (
                            local_projection_group_weight.detach().clone()
                        )
                        dist.all_reduce(
                            projection_group_weight_tensor, op=dist.ReduceOp.SUM
                        )
                        global_projection_group_weight = float(
                            projection_group_weight_tensor.item()
                        )
                        projection_scale_mass_tensor = (
                            local_projection_scale_mass.detach().clone()
                        )
                        dist.all_reduce(
                            projection_scale_mass_tensor, op=dist.ReduceOp.SUM
                        )
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
                        valid_group_denoms = (
                            grouped_loss_masks.sum(
                                dim=1,
                                keepdim=True,
                            )
                            .clamp(min=1)
                            .to(weights_grouped.dtype)
                        )
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
                    global_projection_group_weight = float(
                        max(global_active_group_count, 0)
                    )
                    global_projection_scale_mass = global_projection_group_weight

                    if drgrpo_token_primary:
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
                        stats[
                            "listwise_semantic_competitive_mode_eligible_count"
                        ].append(
                            semantic_diag.eligible_mode_count_grouped.mean().detach()
                        )
                        stats[
                            "listwise_semantic_competitive_mode_eligible_frac"
                        ].append(
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
                            )
                            .mean()
                            .detach()
                        )
                        stats["listwise_semantic_explore_applied_group_frac"].append(
                            semantic_diag.explore_applied_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            )
                            .mean()
                            .detach()
                        )
                        stats["listwise_semantic_prompt_selected_frac"].append(
                            semantic_diag.prompt_selected_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            )
                            .mean()
                            .detach()
                        )
                        stats["listwise_semantic_prompt_rejected_low_opp_frac"].append(
                            semantic_diag.prompt_rejected_low_opp_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            )
                            .mean()
                            .detach()
                        )
                        stats[
                            "listwise_semantic_prompt_rejected_nonpositive_frac"
                        ].append(
                            semantic_diag.prompt_rejected_nonpositive_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            )
                            .mean()
                            .detach()
                        )
                        stats[
                            "listwise_semantic_prompt_rejected_len_guard_frac"
                        ].append(
                            semantic_diag.prompt_rejected_len_guard_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            )
                            .mean()
                            .detach()
                        )
                        stats[
                            "listwise_semantic_prompt_rejected_format_guard_frac"
                        ].append(
                            semantic_diag.prompt_rejected_format_guard_group_mask.to(
                                semantic_diag.explore_budget_grouped.dtype
                            )
                            .mean()
                            .detach()
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
                        stats["listwise_semantic_expected_utility_q"].append(
                            semantic_diag.expected_utility_q_grouped.mean().detach()
                        )
                        stats[
                            "listwise_semantic_expected_utility_explore_target"
                        ].append(
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
                        stats[
                            "listwise_semantic_expected_format_explore_target"
                        ].append(
                            semantic_diag.expected_format_explore_target_grouped.mean().detach()
                        )
                        stats["listwise_semantic_expected_format_final_w"].append(
                            semantic_diag.expected_format_final_w_grouped.mean().detach()
                        )
                        stats[
                            "listwise_semantic_correctness_schedule_any_correct_batch_mean"
                        ].append(
                            torch.tensor(
                                schedule_settings.batch_any_correct_mean,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )
                        stats[
                            "listwise_semantic_correctness_schedule_any_correct_ema"
                        ].append(
                            torch.tensor(
                                schedule_settings.any_correct_ema,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )
                        stats[
                            "listwise_semantic_correctness_schedule_exploration_level"
                        ].append(
                            torch.tensor(
                                schedule_settings.exploration_level,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )
                        stats[
                            "listwise_semantic_correctness_schedule_consolidation_level"
                        ].append(
                            torch.tensor(
                                schedule_settings.consolidation_level,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )
                        stats[
                            "listwise_semantic_correctness_schedule_budget_max"
                        ].append(
                            torch.tensor(
                                schedule_settings.budget_max,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )
                        stats[
                            "listwise_semantic_correctness_schedule_prompt_select_min_alpha_frac"
                        ].append(
                            torch.tensor(
                                schedule_settings.prompt_select_min_alpha_frac,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )
                        stats["listwise_semantic_correctness_schedule_mode_tau"].append(
                            torch.tensor(
                                schedule_settings.mode_tau,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )
                        stats[
                            "listwise_semantic_correctness_schedule_intra_tau"
                        ].append(
                            torch.tensor(
                                schedule_settings.intra_tau,
                                device=weights_grouped.device,
                                dtype=weights_grouped.dtype,
                            )
                        )

                    correct_indicator_grouped = (
                        (
                            mb_correctness_grouped
                            >= _VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD
                        )
                        & grouped_loss_masks
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

                    gain_group_count = int(gain_group_mask.to(torch.int64).sum().item())
                    semantic_entropy_prompt_mean_local = self._masked_prompt_mean(
                        semantic_cluster_entropy_t.detach(),
                        gain_group_mask,
                    )
                    exploration_gain_any_correct_mean_local = self._masked_prompt_mean(
                        exploration_gain_any_correct_t.detach(),
                        gain_group_mask,
                    )
                    exploration_gain_drgrpo_mean_local = self._masked_prompt_mean(
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
                        ].append(exploration_gain_drgrpo_t.detach()[gain_group_mask])

                    stats["listwise_semantic_exploration_gain_any_correct"].append(
                        exploration_gain_any_correct_mean_local
                    )
                    stats["listwise_semantic_exploration_gain_drgrpo"].append(
                        exploration_gain_drgrpo_mean_local
                    )
                    ref_available = bool(candidate_kl_coef > 0.0)
                    ref_semantic_entropy_t = torch.zeros_like(
                        semantic_cluster_entropy_t
                    )
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
                        ref_semantic_entropy_t, _ = (
                            compute_normalized_semantic_cluster_entropy(
                                candidate_probs_grouped=ref_probs_grouped.detach(),
                                cluster_ids_grouped=cluster_ids_grouped,
                                valid_row_mask_grouped=grouped_loss_masks,
                                normalizer_group_size=group_size,
                            )
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
                        projection_ce_loss = per_group_projection_ce[
                            ce_group_mask
                        ].mean()
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
                infos["listwise_aux_loss_raw"] = zero_component
                infos["listwise_aux_loss_weighted"] = zero_component
                infos["listwise_aux_loss_effective"] = zero_component
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
                global_drgrpo_signal_row_count = 0
                weighted_drgrpo_pg_loss = None
                if drgrpo_token_primary:
                    if float(args.beta) > 0.0:
                        raise NotImplementedError(
                            "The exact DrX utility-lift path currently requires beta=0 "
                            "so the weighted objective remains the pure Dr.GRPO clip "
                            "surrogate. Candidate-level trust should use "
                            "maxent_candidate_kl_coef."
                        )
                    if (
                        str(args.maxent_drgrpo_token_advantage_source)
                        == "utility_centered"
                    ):
                        drgrpo_row_adv_flat = flatten_prompt_major_tensor(
                            exact_centered_advantages_grouped
                        ).to(device=new_logps.device, dtype=new_logps.dtype)
                    else:
                        drgrpo_row_adv_flat = probe_row_advantages
                    drgrpo_pg_loss = (probe_drgrpo_pg_row_loss * mb_loss_masks).mean()
                    infos["drgrpo_pg_loss"] = drgrpo_pg_loss.detach()
                    infos["drgrpo_primary_loss"] = drgrpo_pg_loss.detach()

                    token_active_row_mask_grouped = build_drgrpo_token_active_row_mask(
                        advantage_source=str(args.maxent_drgrpo_token_advantage_source),
                        informative_group_mask=informative_group_mask,
                        valid_row_mask_grouped=grouped_loss_masks,
                        utility_centered_advantages_grouped=(
                            exact_centered_advantages_grouped.to(
                                device=policy_seq_logps_grouped.device,
                                dtype=policy_seq_logps_grouped.dtype,
                            )
                            if str(args.maxent_drgrpo_token_advantage_source)
                            == "utility_centered"
                            else torch.zeros_like(
                                exact_centered_advantages_grouped,
                                device=policy_seq_logps_grouped.device,
                                dtype=policy_seq_logps_grouped.dtype,
                            )
                        ),
                    )
                    drgrpo_active_row_mask = flatten_prompt_major_tensor(
                        token_active_row_mask_grouped
                    ).to(torch.bool)
                    drgrpo_active_row_count = int(
                        drgrpo_active_row_mask.to(torch.int64).sum().item()
                    )
                    global_drgrpo_active_row_count = drgrpo_active_row_count
                    if dist.is_available() and dist.is_initialized():
                        drgrpo_active_row_count_tensor = torch.tensor(
                            float(drgrpo_active_row_count),
                            device=policy_seq_logps_grouped.device,
                        )
                        dist.all_reduce(
                            drgrpo_active_row_count_tensor, op=dist.ReduceOp.SUM
                        )
                        global_drgrpo_active_row_count = int(
                            drgrpo_active_row_count_tensor.item()
                        )

                    if (
                        str(args.maxent_drgrpo_token_advantage_source)
                        == "utility_centered"
                    ):
                        weighted_drgrpo_multiplier_flat = torch.ones_like(
                            drgrpo_row_adv_flat
                        )
                        weighted_drgrpo_row_adv_flat = drgrpo_row_adv_flat
                    else:
                        token_weight_row_flat = flatten_prompt_major_tensor(
                            token_target_weights_grouped
                        ).to(
                            device=new_logps.device,
                            dtype=new_logps.dtype,
                        )
                        weighted_drgrpo_multiplier_flat = (
                            float(group_size) * token_weight_row_flat
                        )
                        weighted_drgrpo_row_adv_flat = (
                            weighted_drgrpo_multiplier_flat * drgrpo_row_adv_flat
                        )
                    weighted_drgrpo_delta_adv_flat = (
                        weighted_drgrpo_row_adv_flat - drgrpo_row_adv_flat
                    )
                    drgrpo_signal_row_mask = (
                        drgrpo_active_row_mask
                        & torch.isfinite(weighted_drgrpo_row_adv_flat)
                        & (weighted_drgrpo_row_adv_flat.abs() > 1e-8)
                    )
                    drgrpo_signal_row_count = int(
                        drgrpo_signal_row_mask.to(torch.int64).sum().item()
                    )
                    global_drgrpo_signal_row_count = drgrpo_signal_row_count
                    if dist.is_available() and dist.is_initialized():
                        drgrpo_signal_row_count_tensor = torch.tensor(
                            float(drgrpo_signal_row_count),
                            device=policy_seq_logps_grouped.device,
                        )
                        dist.all_reduce(
                            drgrpo_signal_row_count_tensor, op=dist.ReduceOp.SUM
                        )
                        global_drgrpo_signal_row_count = int(
                            drgrpo_signal_row_count_tensor.item()
                        )
                    weighted_drgrpo_pg_row_loss, _, _, _ = (
                        compute_token_level_clip_loss(
                            new_logps=new_logps,
                            behavior_logps=mb_behavior_logps.to(new_logps.dtype),
                            response_masks=mb_response_masks,
                            row_advantages=weighted_drgrpo_row_adv_flat,
                            clip_low=float(args.cliprange),
                            clip_high=float(args.cliprange),
                            constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                        )
                    )
                    if bool(drgrpo_active_row_mask.any().item()):
                        weighted_drgrpo_pg_loss = (
                            weighted_drgrpo_pg_row_loss[drgrpo_active_row_mask]
                        ).mean()
                    else:
                        weighted_drgrpo_pg_loss = (
                            weighted_drgrpo_pg_row_loss * 0.0
                        ).sum()

                    pg_clipfrac = masked_mean(
                        (
                            torch.clamp(
                                torch.exp(
                                    (
                                        new_logps
                                        - mb_behavior_logps.to(new_logps.dtype)
                                    ).clamp(
                                        -40.0,
                                        40.0,
                                    )
                                ),
                                1.0 - args.cliprange,
                                1.0 + args.cliprange,
                            )
                            < torch.exp(
                                (
                                    new_logps - mb_behavior_logps.to(new_logps.dtype)
                                ).clamp(
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
                        drgrpo_adv_abs_mean = (
                            drgrpo_row_adv_flat[drgrpo_active_row_mask].abs().mean()
                        )
                        listwise_adv_abs_mean = (
                            listwise_centered_adv_grouped[contributing_group_mask]
                            .abs()
                            .mean()
                            if contributing_group_count > 0
                            else zero_component
                        )
                        listwise_delta_adv_abs_mean = (
                            weighted_drgrpo_delta_adv_flat[drgrpo_active_row_mask]
                            .abs()
                            .mean()
                        )
                        combined_adv_abs_mean = (
                            weighted_drgrpo_row_adv_flat[drgrpo_active_row_mask]
                            .abs()
                            .mean()
                        )
                        listwise_weight_deviation = (
                            (
                                weighted_drgrpo_multiplier_flat[drgrpo_active_row_mask]
                                - 1.0
                            )
                            .abs()
                            .mean()
                        )
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
                    infos["listwise_raw_to_drgrpo_ratio"] = (
                        listwise_weight_deviation.detach()
                    )
                    infos["listwise_post_scale_ratio"] = torch.abs(
                        listwise_delta_adv_abs_mean.detach()
                    ) / (torch.abs(drgrpo_adv_abs_mean.detach()) + 1e-8)
                    infos["listwise_aux_loss_raw"] = (
                        weighted_drgrpo_pg_loss.detach() - drgrpo_pg_loss.detach()
                    )
                    infos["listwise_aux_loss_weighted"] = infos["listwise_aux_loss_raw"]
                    infos["listwise_aux_loss_effective"] = infos[
                        "listwise_aux_loss_raw"
                    ]
                    infos["listwise_helpfulness_proxy"] = torch.abs(
                        infos["listwise_aux_loss_effective"]
                    ) / (torch.abs(infos["drgrpo_primary_loss"]) + 1e-8)
                    infos["listwise_helpfulness_proxy_valid"] = torch.tensor(
                        1.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["objective_effective_total_loss"] = loss.detach()
                elif not token_clip_primary:
                    infos["pg_loss"] = loss.detach()
                    infos["listwise_aux_loss_raw"] = loss.detach()
                    infos["listwise_aux_loss_weighted"] = loss.detach()
                    infos["listwise_aux_loss_effective"] = loss.detach()
                    infos["objective_effective_total_loss"] = loss.detach()

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
                            valid_group_denoms = (
                                grouped_loss_masks.sum(
                                    dim=1,
                                    keepdim=True,
                                )
                                .clamp(min=1)
                                .to(weights_grouped.dtype)
                            )
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
                                reward_mass_grouped[active_group_mask].mean().detach()
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
                            clip_loss = per_group_clip_loss[
                                active_clip_group_mask
                            ].mean()
                        else:
                            clip_loss = (per_group_clip_loss * 0.0).sum()
                        loss = loss + clip_coef * clip_loss
                        infos["clip_loss"] = clip_loss.detach()
                        infos["listwise_aux_loss_raw"] = clip_loss.detach()
                        infos["listwise_aux_loss_weighted"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["listwise_aux_loss_effective"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["objective_effective_total_loss"] = infos[
                            "listwise_aux_loss_effective"
                        ]

                        is_low_clipped = (seq_ratio < 1.0 - clip_low) & (clip_adv < 0.0)
                        is_high_clipped = (seq_ratio > 1.0 + clip_high) & (
                            clip_adv > 0.0
                        )
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
                        (
                            per_row_clip_loss,
                            seq_ratio,
                            is_low_clipped,
                            is_high_clipped,
                        ) = compute_token_level_clip_loss(
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
                        active_row_count = int(
                            active_row_mask.to(torch.int64).sum().item()
                        )
                        if active_row_count > 0:
                            clip_loss = (
                                per_row_clip_loss
                                * active_row_mask.to(per_row_clip_loss.dtype)
                            ).sum() / float(active_row_count)
                        else:
                            clip_loss = (per_row_clip_loss * 0.0).sum()
                        if token_clip_primary:
                            loss = clip_coef * clip_loss
                        else:
                            loss = loss + clip_coef * clip_loss
                        infos["clip_loss"] = clip_loss.detach()
                        infos["listwise_aux_loss_raw"] = clip_loss.detach()
                        infos["listwise_aux_loss_weighted"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["listwise_aux_loss_effective"] = (
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
                has_token_signal = global_drgrpo_signal_row_count > 0
                has_projection_signal = (
                    float(sequence_aux_coef) > 0.0
                    and global_projection_group_weight > 0.0
                    and global_projection_scale_mass > 0.0
                )
                if drgrpo_token_primary:
                    skip_zero_signal_update = not (
                        has_token_signal or has_projection_signal
                    )
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
                    if token_clip_primary:
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
                            seq_coeffs_grouped = (
                                float(sequence_aux_coef) * seq_coeffs_grouped
                            )
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
                            sequence_aux_coef=(float(sequence_aux_coef)),
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
                if local_grad_step % grad_acc_step == 0 and self.strategy.is_rank_0():
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
                    infos["listwise_policy_prob_mean"] = (
                        policy_probs_grouped.mean().detach()
                    )
                    infos["listwise_weight_mean"] = weights_grouped.mean().detach()
                    infos["listwise_weight_std"] = (
                        weights_grouped.to(torch.float32).std(unbiased=False).detach()
                    )
                    infos["listwise_weight_entropy"] = (
                        active_weight_entropy.detach().to(
                            device=policy_seq_logps_grouped.device,
                            dtype=torch.float32,
                        )
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
                    infos["listwise_neutral_group_frac"] = (
                        neutral_group_mask.to(torch.float32).mean().detach()
                    )
                    infos["listwise_informative_group_frac"] = (
                        active_group_mask.to(torch.float32).mean().detach()
                    )
                    infos["listwise_informative_group_count_global"] = torch.tensor(
                        float(global_active_group_count),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_contributing_group_frac"] = (
                        contributing_group_mask.to(torch.float32).mean().detach()
                    )
                    infos["listwise_contributing_group_count_global"] = torch.tensor(
                        float(global_contributing_group_count),
                        device=policy_seq_logps_grouped.device,
                    )
                    infos.update(
                        compute_correctness_group_rate_infos(
                            correctness_grouped=mb_correctness_grouped,
                            valid_row_mask_grouped=grouped_loss_masks,
                            threshold=_VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD,
                        )
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
                    infos["listwise_valid_row_frac"] = (
                        grouped_loss_masks.to(torch.float32).mean().detach()
                    )
                    infos["listwise_partial_group_frac"] = (
                        (
                            grouped_loss_masks.any(dim=1)
                            & (~grouped_loss_masks.all(dim=1))
                        )
                        .to(torch.float32)
                        .mean()
                        .detach()
                    )
                    infos["listwise_valid_weight_mass"] = (
                        target_weights_grouped.sum(dim=1).mean().detach()
                    )
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
                    infos["listwise_token_clip_primary"] = torch.tensor(
                        1.0 if token_clip_primary else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_drgrpo_token_primary"] = torch.tensor(
                        1.0 if drgrpo_token_primary else 0.0,
                        device=policy_seq_logps_grouped.device,
                    )
                    infos["listwise_drgrpo_token_active_row_frac"] = (
                        token_active_row_mask_grouped.to(torch.float32).mean().detach()
                    )
                    infos["listwise_drgrpo_token_active_row_count_global"] = (
                        torch.tensor(
                            float(global_drgrpo_active_row_count),
                            device=policy_seq_logps_grouped.device,
                        )
                    )
                    infos["listwise_drgrpo_token_advantage_source_utility_centered"] = (
                        torch.tensor(
                            1.0
                            if str(args.maxent_drgrpo_token_advantage_source)
                            == "utility_centered"
                            else 0.0,
                            device=policy_seq_logps_grouped.device,
                        )
                    )
                    infos["listwise_drgrpo_token_length_normalizer_response"] = (
                        torch.tensor(
                            1.0
                            if str(args.maxent_drgrpo_token_length_normalizer)
                            == "response_length"
                            else 0.0,
                            device=policy_seq_logps_grouped.device,
                        )
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
                    self._record_listwise_runtime_infos(
                        infos,
                        device=policy_seq_logps_grouped.device,
                        skip_zero_signal_update=skip_zero_signal_update,
                    )
                    if token_clip_primary:
                        infos["pg_loss"] = loss.detach()
                    elif not drgrpo_token_primary:
                        logprobs_diff = (
                            new_logps - mb_behavior_logps
                        ) * mb_response_masks
                        stats["logprobs_diff_max"].append(
                            torch.amax(logprobs_diff.detach())
                        )
                        stats["logprobs_diff_min"].append(
                            torch.amin(logprobs_diff.detach())
                        )
                        stats["zero_pg_loss_count"].append(
                            (per_group_projection_ce == 0)
                            .detach()
                            .sum()
                            .to(torch.float32)
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
                        reduced_semantic_entropy_mu = (
                            self._distributed_weighted_mean_scalar(
                                semantic_entropy_prompt_mean_local,
                                weight=gain_group_count,
                            )
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
                    self._apply_listwise_controller_updates(
                        infos,
                        device=policy_seq_logps_grouped.device,
                        skip_zero_signal_update=skip_zero_signal_update,
                        weight_entropy_controller=reduced_weight_entropy,
                        semantic_entropy_mu=reduced_semantic_entropy_mu,
                        exploration_gain_any_correct=(
                            reduced_exploration_gain_any_correct
                        ),
                        exploration_gain_drgrpo=reduced_exploration_gain_drgrpo,
                        measured_kl=reduced_kl_value,
                        active_tau_target_metric=active_tau_target_metric,
                    )

        return finalize_listwise_info_stats(
            infos,
            stats,
            prompt_diag_stats,
            advantages=advantages,
            grouped_reward_values=grouped_reward_values,
        )

    def _listwise_learning_step_row_sharded_exact_drx(
        self,
        *,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        loss_masks: torch.Tensor,
        behavior_logps: torch.Tensor,
        grouped_reward_values: torch.Tensor,
        grouped_correctness_values: torch.Tensor,
        grouped_formatted_values: torch.Tensor,
        behavior_seq_logps_grouped: torch.Tensor,
        behavior_seq_logps_grouped_raw: torch.Tensor,
        ref_seq_logps_grouped: torch.Tensor,
        ref_seq_logps_grouped_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        args: ZeroMathArgs = self.args
        infos: dict[str, torch.Tensor] = {}
        stats: dict[str, list[torch.Tensor]] = defaultdict(list)
        device = input_ids.device
        group_size = max(int(args.num_samples), 1)
        prompt_groups_per_step = max(int(args.train_batch_size_per_device), 1)

        if not dist.is_available() or not dist.is_initialized():
            raise ValueError(
                "The row-sharded exact DrX path requires torch.distributed to be initialized."
            )
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        drgrpo_token_primary = bool(args.maxent_drgrpo_token_primary)
        if world_size != group_size:
            raise ValueError(
                "The row-sharded exact DrX path requires world_size == num_samples "
                f"(got world_size={world_size}, num_samples={group_size})."
            )
        if prompt_groups_per_step >= group_size:
            raise ValueError(
                "The row-sharded exact DrX path requires "
                "0 < train_batch_size_per_device < num_samples."
            )

        gathered_prompt_counts = self._all_gather_same_shape_tensor(
            torch.tensor(
                [int(grouped_reward_values.size(0))],
                device=device,
                dtype=torch.int64,
            )
        ).reshape(-1)
        if gathered_prompt_counts.numel() <= 0:
            raise ValueError("Could not determine the row-sharded prompt-group counts.")
        if not bool((gathered_prompt_counts == gathered_prompt_counts[0]).all().item()):
            raise ValueError(
                "The row-sharded exact DrX path currently requires every rank to own "
                "the same number of prompt groups in the local rollout batch."
            )
        local_prompt_count = int(gathered_prompt_counts[0].item())

        gathered_input_widths = self._all_gather_same_shape_tensor(
            torch.tensor(
                [int(input_ids.size(1))],
                device=device,
                dtype=torch.int64,
            )
        ).reshape(-1)
        gathered_action_widths = self._all_gather_same_shape_tensor(
            torch.tensor(
                [int(response_masks.size(1))],
                device=device,
                dtype=torch.int64,
            )
        ).reshape(-1)
        global_input_width = int(gathered_input_widths.max().item())
        global_action_width = int(gathered_action_widths.max().item())
        local_input_width = int(input_ids.size(1))
        local_action_width = int(response_masks.size(1))
        if local_input_width < global_input_width:
            input_pad = global_input_width - local_input_width
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, input_pad),
                value=int(pad_token_id),
            )
            att_mask = torch.nn.functional.pad(att_mask, (0, input_pad), value=0)
        if local_action_width < global_action_width:
            action_pad = global_action_width - local_action_width
            response_masks = torch.nn.functional.pad(
                response_masks,
                (0, action_pad),
                value=0,
            )
            behavior_logps = torch.nn.functional.pad(
                behavior_logps,
                (0, action_pad),
                value=0.0,
            )
        if rank == 0:
            logging.info(
                "row-sharded exact DrX synchronized widths: input_widths=%s "
                "action_widths=%s global_input_width=%s global_action_width=%s",
                gathered_input_widths.tolist(),
                gathered_action_widths.tolist(),
                global_input_width,
                global_action_width,
            )

        grouped_input_ids = reshape_prompt_major_tensor(input_ids, group_size)
        grouped_att_mask = reshape_prompt_major_tensor(att_mask, group_size)
        grouped_response_masks = reshape_prompt_major_tensor(response_masks, group_size)
        grouped_behavior_logps = reshape_prompt_major_tensor(behavior_logps, group_size)
        grouped_loss_masks = reshape_prompt_major_tensor(
            loss_masks.to(torch.bool), group_size
        )
        if (
            grouped_input_ids is None
            or grouped_att_mask is None
            or grouped_response_masks is None
            or grouped_behavior_logps is None
            or grouped_loss_masks is None
        ):
            raise ValueError(
                "Could not reshape the local rollout tensors into prompt-major groups "
                "for the row-sharded exact DrX path."
            )

        exact_drx_weight_source = str(
            getattr(args, "maxent_exact_drx_weight_source", "sequence_clipped")
        )
        length_normalize_policy = bool(args.maxent_length_normalize_policy)
        sequence_aux_coef = coerce_non_negative_float(
            getattr(args, "maxent_sequence_aux_coef", 1.0),
            default=1.0,
        )
        candidate_kl_coef = coerce_non_negative_float(
            getattr(args, "maxent_candidate_kl_coef", 0.0),
            default=0.0,
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
                    getattr(args, "maxent_prompt_select_min_alpha_frac", 0.0),
                    default=0.0,
                ),
                0.0,
            ),
            1.0,
        )
        competitive_mode_positive_only = bool(
            getattr(args, "maxent_competitive_mode_positive_only", False)
        )
        correctness_schedule_enabled = bool(
            getattr(args, "maxent_correctness_schedule_enabled", False)
        )
        correctness_schedule_ema_decay = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_correctness_schedule_ema_decay", 0.9),
                    default=0.9,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_low = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_correctness_schedule_low", 0.05),
                    default=0.05,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_high = min(
            max(
                coerce_non_negative_float(
                    getattr(args, "maxent_correctness_schedule_high", 0.90),
                    default=0.90,
                ),
                correctness_schedule_low,
            ),
            1.0,
        )
        correctness_schedule_budget_max_early = coerce_non_negative_float(
            getattr(
                args,
                "maxent_correctness_schedule_budget_max_early",
                competitive_mode_budget_max,
            ),
            default=competitive_mode_budget_max,
        )
        correctness_schedule_budget_max_late = coerce_non_negative_float(
            getattr(
                args,
                "maxent_correctness_schedule_budget_max_late",
                competitive_mode_budget_max,
            ),
            default=competitive_mode_budget_max,
        )
        correctness_schedule_prompt_select_early = min(
            max(
                coerce_non_negative_float(
                    getattr(
                        args,
                        "maxent_correctness_schedule_prompt_select_min_alpha_frac_early",
                        prompt_select_min_alpha_frac,
                    ),
                    default=prompt_select_min_alpha_frac,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_prompt_select_late = min(
            max(
                coerce_non_negative_float(
                    getattr(
                        args,
                        "maxent_correctness_schedule_prompt_select_min_alpha_frac_late",
                        prompt_select_min_alpha_frac,
                    ),
                    default=prompt_select_min_alpha_frac,
                ),
                0.0,
            ),
            1.0,
        )
        correctness_schedule_mode_tau_early = max(
            coerce_non_negative_float(
                getattr(
                    args,
                    "maxent_correctness_schedule_mode_tau_early",
                    competitive_mode_tau,
                ),
                default=competitive_mode_tau,
            ),
            1e-8,
        )
        correctness_schedule_mode_tau_late = max(
            coerce_non_negative_float(
                getattr(
                    args,
                    "maxent_correctness_schedule_mode_tau_late",
                    competitive_mode_tau,
                ),
                default=competitive_mode_tau,
            ),
            1e-8,
        )
        correctness_schedule_intra_tau_early = max(
            coerce_non_negative_float(
                getattr(
                    args,
                    "maxent_correctness_schedule_intra_tau_early",
                    competitive_mode_intra_tau,
                ),
                default=competitive_mode_intra_tau,
            ),
            1e-8,
        )
        correctness_schedule_intra_tau_late = max(
            coerce_non_negative_float(
                getattr(
                    args,
                    "maxent_correctness_schedule_intra_tau_late",
                    competitive_mode_intra_tau,
                ),
                default=competitive_mode_intra_tau,
            ),
            1e-8,
        )
        semantic_guard_max_expected_len_delta = float(
            getattr(args, "maxent_semantic_guard_max_expected_len_delta", 24.0)
        )
        if not math.isfinite(semantic_guard_max_expected_len_delta):
            semantic_guard_max_expected_len_delta = float("inf")
        semantic_guard_max_expected_format_drop = coerce_non_negative_float(
            getattr(args, "maxent_semantic_guard_max_expected_format_drop", 0.0),
            default=0.0,
        )
        if (
            bool(getattr(args, "maxent_branch_grad_diagnostics", False))
            and not self._listwise_branch_grad_probe_warned
        ):
            logging.warning(
                "Disabling listwise branch gradient diagnostics for the row-sharded "
                "exact DrX path because the one-row-per-rank execution does not "
                "reuse the grouped probe implementation."
            )
            self._listwise_branch_grad_probe_warned = True

        grad_acc_step = max(int(self.strategy.grad_acc_step), 1)
        total_prompt_count = local_prompt_count * world_size
        local_minibatches_per_epoch = max(
            math.ceil(total_prompt_count / max(prompt_groups_per_step, 1)),
            1,
        )
        total_optimizer_updates = max(
            math.ceil(
                args.num_ppo_epochs * local_minibatches_per_epoch / grad_acc_step
            ),
            1,
        )
        local_grad_step = 0

        for _ in range(args.num_ppo_epochs):
            prompt_permutation = self._broadcast_global_prompt_permutation(
                local_prompt_count=local_prompt_count,
                device=device,
            )
            for prompt_batch_start in range(
                0,
                int(prompt_permutation.numel()),
                prompt_groups_per_step,
            ):
                prompt_batch_ids = prompt_permutation[
                    prompt_batch_start : prompt_batch_start + prompt_groups_per_step
                ]
                if int(prompt_batch_ids.numel()) <= 0:
                    continue
                local_grad_step += 1
                prompt_owner_ranks: list[int] = []
                prompt_owner_indices: list[int] = []
                prompt_input_batches: list[torch.Tensor] = []
                prompt_att_batches: list[torch.Tensor] = []
                prompt_response_batches: list[torch.Tensor] = []
                prompt_behavior_logp_batches: list[torch.Tensor] = []
                prompt_loss_mask_batches: list[torch.Tensor] = []
                prompt_reward_batches: list[torch.Tensor] = []
                prompt_correctness_batches: list[torch.Tensor] = []
                prompt_behavior_seq_batches: list[torch.Tensor] = []
                prompt_behavior_seq_raw_batches: list[torch.Tensor] = []
                prompt_ref_seq_batches: list[torch.Tensor] = []
                prompt_ref_seq_raw_batches: list[torch.Tensor] = []

                for global_prompt_id_t in prompt_batch_ids:
                    global_prompt_id = int(global_prompt_id_t.item())
                    owner_rank = global_prompt_id // local_prompt_count
                    owner_prompt_idx = global_prompt_id % local_prompt_count
                    prompt_owner_ranks.append(owner_rank)
                    prompt_owner_indices.append(owner_prompt_idx)

                    if rank == owner_rank:
                        owner_input_ids = grouped_input_ids[owner_prompt_idx]
                        owner_att_mask = grouped_att_mask[owner_prompt_idx]
                        owner_response_masks = grouped_response_masks[owner_prompt_idx]
                        owner_behavior_logps = grouped_behavior_logps[owner_prompt_idx]
                        owner_loss_masks = grouped_loss_masks[owner_prompt_idx].to(
                            torch.int64
                        )
                        owner_reward_values = grouped_reward_values[owner_prompt_idx]
                        owner_correctness_values = grouped_correctness_values[
                            owner_prompt_idx
                        ]
                        owner_behavior_seq = behavior_seq_logps_grouped[
                            owner_prompt_idx
                        ]
                        owner_behavior_seq_raw = behavior_seq_logps_grouped_raw[
                            owner_prompt_idx
                        ]
                        owner_ref_seq = ref_seq_logps_grouped[owner_prompt_idx]
                        owner_ref_seq_raw = ref_seq_logps_grouped_raw[owner_prompt_idx]
                    else:
                        owner_input_ids = None
                        owner_att_mask = None
                        owner_response_masks = None
                        owner_behavior_logps = None
                        owner_loss_masks = None
                        owner_reward_values = None
                        owner_correctness_values = None
                        owner_behavior_seq = None
                        owner_behavior_seq_raw = None
                        owner_ref_seq = None
                        owner_ref_seq_raw = None

                    prompt_input_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_input_ids,
                            src_rank=owner_rank,
                            shape=(group_size, input_ids.size(1)),
                            dtype=input_ids.dtype,
                            device=device,
                        )
                    )
                    prompt_att_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_att_mask,
                            src_rank=owner_rank,
                            shape=(group_size, att_mask.size(1)),
                            dtype=att_mask.dtype,
                            device=device,
                        )
                    )
                    prompt_response_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_response_masks,
                            src_rank=owner_rank,
                            shape=(group_size, response_masks.size(1)),
                            dtype=response_masks.dtype,
                            device=device,
                        )
                    )
                    prompt_behavior_logp_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_behavior_logps,
                            src_rank=owner_rank,
                            shape=(group_size, behavior_logps.size(1)),
                            dtype=behavior_logps.dtype,
                            device=device,
                        )
                    )
                    prompt_loss_mask_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_loss_masks,
                            src_rank=owner_rank,
                            shape=(group_size,),
                            dtype=torch.int64,
                            device=device,
                        ).to(torch.bool)
                    )
                    prompt_reward_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_reward_values,
                            src_rank=owner_rank,
                            shape=(group_size,),
                            dtype=grouped_reward_values.dtype,
                            device=device,
                        )
                    )
                    prompt_correctness_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_correctness_values,
                            src_rank=owner_rank,
                            shape=(group_size,),
                            dtype=grouped_correctness_values.dtype,
                            device=device,
                        )
                    )
                    prompt_behavior_seq_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_behavior_seq,
                            src_rank=owner_rank,
                            shape=(group_size,),
                            dtype=behavior_seq_logps_grouped.dtype,
                            device=device,
                        )
                    )
                    prompt_behavior_seq_raw_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_behavior_seq_raw,
                            src_rank=owner_rank,
                            shape=(group_size,),
                            dtype=behavior_seq_logps_grouped_raw.dtype,
                            device=device,
                        )
                    )
                    prompt_ref_seq_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_ref_seq,
                            src_rank=owner_rank,
                            shape=(group_size,),
                            dtype=ref_seq_logps_grouped.dtype,
                            device=device,
                        )
                    )
                    prompt_ref_seq_raw_batches.append(
                        self._broadcast_tensor_from_owner(
                            owner_ref_seq_raw,
                            src_rank=owner_rank,
                            shape=(group_size,),
                            dtype=ref_seq_logps_grouped_raw.dtype,
                            device=device,
                        )
                    )

                prompt_input_ids = torch.stack(prompt_input_batches, dim=0)
                prompt_att_mask = torch.stack(prompt_att_batches, dim=0)
                prompt_response_masks = torch.stack(prompt_response_batches, dim=0)
                prompt_behavior_logps = torch.stack(prompt_behavior_logp_batches, dim=0)
                prompt_loss_masks = torch.stack(prompt_loss_mask_batches, dim=0)
                prompt_reward_values = torch.stack(prompt_reward_batches, dim=0)
                prompt_correctness_values = torch.stack(
                    prompt_correctness_batches, dim=0
                )
                prompt_behavior_seq = torch.stack(prompt_behavior_seq_batches, dim=0)
                prompt_behavior_seq_raw = torch.stack(
                    prompt_behavior_seq_raw_batches, dim=0
                )
                prompt_ref_seq = torch.stack(prompt_ref_seq_batches, dim=0)
                prompt_ref_seq_raw = torch.stack(prompt_ref_seq_raw_batches, dim=0)

                (
                    _,
                    [
                        local_input_ids,
                        local_att_mask,
                        local_response_masks,
                        local_behavior_logps,
                    ],
                ) = self._trim_policy_batch(
                    prompt_input_ids[:, rank, :],
                    prompt_att_mask[:, rank, :],
                    prompt_response_masks[:, rank, :],
                    prompt_behavior_logps[:, rank, :],
                )
                if local_grad_step == 1:
                    logging.info(
                        "row-sharded exact DrX minibatch ready: input=%s att=%s "
                        "response=%s prompt_batch=%s owner_ranks=%s prompt_idxs=%s",
                        tuple(local_input_ids.shape),
                        tuple(local_att_mask.shape),
                        tuple(local_response_masks.shape),
                        int(prompt_batch_ids.numel()),
                        prompt_owner_ranks,
                        prompt_owner_indices,
                    )

                new_logps, entropy, policy_chunk_size = self._compute_policy_probe(
                    local_input_ids,
                    local_att_mask,
                    local_response_masks,
                )
                infos["listwise_policy_probe_chunk_size"] = torch.tensor(
                    float(policy_chunk_size),
                    device=device,
                )

                prompt_advantages = (
                    prompt_reward_values
                    - prompt_reward_values.mean(dim=1, keepdim=True)
                ).to(device=new_logps.device, dtype=new_logps.dtype)
                local_row_advantages = prompt_advantages[:, rank].reshape(-1)

                probe_drgrpo_pg_row_loss, _, _, _ = compute_token_level_clip_loss(
                    new_logps=new_logps,
                    behavior_logps=local_behavior_logps.to(new_logps.dtype),
                    response_masks=local_response_masks,
                    row_advantages=local_row_advantages,
                    clip_low=float(args.cliprange),
                    clip_high=float(args.cliprange),
                    constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                )
                probe_log_ratio = (
                    new_logps - local_behavior_logps.to(new_logps.dtype)
                ).clamp(-40.0, 40.0)
                probe_token_ratio = torch.exp(probe_log_ratio).to(new_logps.dtype)
                probe_token_advantages = local_row_advantages.unsqueeze(1)
                local_policy_seq_raw = (new_logps * local_response_masks).sum(dim=1)
                local_token_counts = (
                    local_response_masks.sum(dim=1).clamp(min=1).to(new_logps.dtype)
                )
                local_policy_seq = local_policy_seq_raw / local_token_counts
                if not length_normalize_policy:
                    local_policy_seq = local_policy_seq_raw
                if exact_drx_weight_source == "clipped":
                    local_utility_score = -probe_drgrpo_pg_row_loss.detach()
                elif exact_drx_weight_source == "unclipped":
                    local_utility_score = aggregate_masked_row_values(
                        probe_token_ratio * probe_token_advantages,
                        local_response_masks,
                        constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                    ).detach()
                elif exact_drx_weight_source == "local_linear":
                    local_utility_score = aggregate_masked_row_values(
                        torch.ones_like(new_logps) * probe_token_advantages,
                        local_response_masks,
                        constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                    ).detach()
                elif exact_drx_weight_source == "sequence_clipped":
                    local_utility_score = local_policy_seq_raw.detach()
                else:
                    raise ValueError(
                        "Unsupported maxent_exact_drx_weight_source for the "
                        f"row-sharded exact DrX path: {exact_drx_weight_source}"
                    )

                gathered_policy_seq = (
                    self._all_gather_prompt_values(
                        local_policy_seq.detach().to(device=device, dtype=torch.float32)
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .to(dtype=torch.float32)
                )
                gathered_policy_seq_raw = (
                    self._all_gather_prompt_values(
                        local_policy_seq_raw.detach().to(
                            device=device, dtype=torch.float32
                        )
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .to(dtype=torch.float32)
                )
                gathered_utility_scores = (
                    self._all_gather_prompt_values(
                        local_utility_score.detach().to(
                            device=device, dtype=torch.float32
                        )
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .to(dtype=torch.float32)
                )
                gathered_probe_row_loss = (
                    self._all_gather_prompt_values(
                        probe_drgrpo_pg_row_loss.detach().to(
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .to(dtype=torch.float32)
                )
                if entropy is not None:
                    gathered_entropy = (
                        self._all_gather_prompt_values(
                            masked_mean(entropy, local_response_masks)
                            .detach()
                            .to(device=device, dtype=torch.float32)
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )
                    stats["entropy"].append(gathered_entropy.mean().detach())
                    infos["entropy"] = gathered_entropy.mean().detach()

                local_logprobs_diff = (
                    new_logps - local_behavior_logps.to(new_logps.dtype)
                ).detach() * local_response_masks
                stats["logprobs_diff_max"].append(
                    self._all_gather_prompt_values(
                        torch.amax(local_logprobs_diff).to(
                            device=device,
                            dtype=torch.float32,
                        )
                    ).max()
                )
                stats["logprobs_diff_min"].append(
                    self._all_gather_prompt_values(
                        torch.amin(local_logprobs_diff).to(
                            device=device,
                            dtype=torch.float32,
                        )
                    ).min()
                )
                stats["zero_pg_loss_count"].append(
                    (gathered_probe_row_loss == 0).to(torch.float32).sum().detach()
                )
                base_pg_clipfrac_local = masked_mean(
                    (
                        torch.clamp(
                            torch.exp(probe_log_ratio),
                            1.0 - args.cliprange,
                            1.0 + args.cliprange,
                        )
                        < torch.exp(probe_log_ratio)
                    ).float(),
                    local_response_masks,
                    axis=1,
                ).to(torch.float32)
                stats["pg_clipfrac"].append(
                    self._all_gather_prompt_values(
                        base_pg_clipfrac_local.detach().to(
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    .mean()
                    .detach()
                )

                policy_seq_logps_grouped = gathered_policy_seq.to(
                    device=device,
                    dtype=prompt_ref_seq.dtype,
                )
                policy_seq_logps_grouped_raw = gathered_policy_seq_raw.to(
                    device=device,
                    dtype=prompt_ref_seq_raw.dtype,
                )
                behavior_seq_grouped = prompt_behavior_seq.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                behavior_seq_grouped_raw = prompt_behavior_seq_raw.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                grouped_loss_masks_prompt = prompt_loss_masks
                mb_ref_seq_grouped = prompt_ref_seq.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_ref_seq_grouped_raw = prompt_ref_seq_raw.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_reward_grouped = prompt_reward_values.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                mb_correctness_grouped = prompt_correctness_values.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                probe_row_advantages_grouped = prompt_advantages.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )

                if exact_drx_weight_source == "sequence_clipped":
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
                        grouped_loss_masks_prompt,
                        probe_drgrpo_utility_grouped,
                        torch.zeros_like(probe_drgrpo_utility_grouped),
                    )
                else:
                    probe_drgrpo_utility_grouped = gathered_utility_scores.unsqueeze(
                        0
                    ).to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    probe_drgrpo_utility_grouped = torch.where(
                        grouped_loss_masks_prompt,
                        probe_drgrpo_utility_grouped,
                        torch.zeros_like(probe_drgrpo_utility_grouped),
                    )

                valid_group_counts = grouped_loss_masks_prompt.to(torch.int64).sum(
                    dim=1
                )
                utility_dtype_info = torch.finfo(probe_drgrpo_utility_grouped.dtype)
                valid_utility_max = torch.where(
                    grouped_loss_masks_prompt,
                    probe_drgrpo_utility_grouped,
                    torch.full_like(
                        probe_drgrpo_utility_grouped, utility_dtype_info.min
                    ),
                ).amax(dim=1)
                valid_utility_min = torch.where(
                    grouped_loss_masks_prompt,
                    probe_drgrpo_utility_grouped,
                    torch.full_like(
                        probe_drgrpo_utility_grouped, utility_dtype_info.max
                    ),
                ).amin(dim=1)
                neutral_group_mask = (valid_group_counts <= 1) | (
                    (valid_utility_max - valid_utility_min) <= 1e-8
                    if bool(args.maxent_listwise_skip_zero_variance_groups)
                    else torch.zeros_like(valid_group_counts, dtype=torch.bool)
                )
                contributing_group_mask = grouped_loss_masks_prompt.any(dim=1)
                active_group_mask = (~neutral_group_mask) & contributing_group_mask
                active_group_count = int(active_group_mask.to(torch.int64).sum().item())
                contributing_group_count = int(
                    contributing_group_mask.to(torch.int64).sum().item()
                )
                schedule_any_correct_grouped = (
                    (
                        (
                            mb_correctness_grouped
                            >= _VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD
                        )
                        & grouped_loss_masks_prompt
                    )
                    .any(dim=1)
                    .to(policy_seq_logps_grouped.dtype)
                )
                schedule_batch_any_correct_mean = (
                    float(
                        schedule_any_correct_grouped[contributing_group_mask]
                        .mean()
                        .item()
                    )
                    if contributing_group_count > 0
                    else 0.0
                )
                schedule_settings = build_correctness_scheduled_settings(
                    enabled=correctness_schedule_enabled,
                    previous_ema=self._maxent_correctness_schedule_any_correct_ema,
                    batch_any_correct_mean=schedule_batch_any_correct_mean,
                    ema_decay=correctness_schedule_ema_decay,
                    correctness_low=correctness_schedule_low,
                    correctness_high=correctness_schedule_high,
                    static_budget_max=competitive_mode_budget_max,
                    static_prompt_select_min_alpha_frac=prompt_select_min_alpha_frac,
                    static_mode_tau=competitive_mode_tau,
                    static_intra_tau=competitive_mode_intra_tau,
                    budget_max_early=correctness_schedule_budget_max_early,
                    budget_max_late=correctness_schedule_budget_max_late,
                    prompt_select_min_alpha_frac_early=correctness_schedule_prompt_select_early,
                    prompt_select_min_alpha_frac_late=correctness_schedule_prompt_select_late,
                    mode_tau_early=correctness_schedule_mode_tau_early,
                    mode_tau_late=correctness_schedule_mode_tau_late,
                    intra_tau_early=correctness_schedule_intra_tau_early,
                    intra_tau_late=correctness_schedule_intra_tau_late,
                )
                self._maxent_correctness_schedule_any_correct_ema = (
                    schedule_settings.any_correct_ema
                )
                effective_competitive_mode_budget_max = float(
                    schedule_settings.budget_max
                )
                effective_prompt_select_min_alpha_frac = float(
                    schedule_settings.prompt_select_min_alpha_frac
                )
                effective_competitive_mode_tau = float(schedule_settings.mode_tau)
                effective_competitive_mode_intra_tau = float(
                    schedule_settings.intra_tau
                )

                current_tau = self._sync_maxent_tau_from_state()
                (
                    final_answer_keys_grouped,
                    reasoning_trace_texts_grouped,
                    reasoning_signature_keys_grouped,
                ) = self._semantic_cluster_inputs(
                    flatten_prompt_major_tensor(prompt_input_ids),
                    flatten_prompt_major_tensor(prompt_response_masks),
                    group_size,
                )
                (
                    reasoning_trace_embeddings_grouped,
                    reasoning_trace_valid_row_mask_grouped,
                    semantic_trace_truncated_frac,
                ) = self._semantic_trace_embeddings_grouped(
                    reasoning_trace_texts_grouped=reasoning_trace_texts_grouped,
                    valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                )
                semantic_cluster_bundle = build_runtime_semantic_cluster_bundle(
                    args=args,
                    default_method="greedy",
                    final_answer_keys_grouped=final_answer_keys_grouped,
                    valid_row_mask_grouped=grouped_loss_masks_prompt,
                    reasoning_signature_keys_grouped=reasoning_signature_keys_grouped,
                    reasoning_trace_embeddings_grouped=reasoning_trace_embeddings_grouped,
                    reasoning_trace_valid_row_mask_grouped=reasoning_trace_valid_row_mask_grouped,
                )
                answer_key_extracted_mask_grouped = (
                    torch.tensor(
                        [
                            [answer_key is not None for answer_key in prompt_keys]
                            for prompt_keys in final_answer_keys_grouped
                        ],
                        device=device,
                        dtype=torch.bool,
                    )
                    & grouped_loss_masks_prompt
                )
                trace_extracted_mask_grouped = (
                    torch.tensor(
                        [
                            [trace_text is not None for trace_text in prompt_rows]
                            for prompt_rows in reasoning_trace_texts_grouped
                        ],
                        device=device,
                        dtype=torch.bool,
                    )
                    & grouped_loss_masks_prompt
                )
                signature_extracted_mask_grouped = (
                    torch.tensor(
                        [
                            [signature is not None for signature in prompt_rows]
                            for prompt_rows in reasoning_signature_keys_grouped
                        ],
                        device=device,
                        dtype=torch.bool,
                    )
                    & grouped_loss_masks_prompt
                )
                semantic_valid_row_mask_grouped = (
                    semantic_cluster_bundle.semantic_valid_row_mask_grouped.to(
                        device=device
                    )
                )
                valid_row_count_for_semantics = (
                    grouped_loss_masks_prompt.to(torch.float32).sum().clamp(min=1.0)
                )
                stats["listwise_semantic_answer_key_extracted_frac"].append(
                    (
                        answer_key_extracted_mask_grouped.to(torch.float32).sum()
                        / valid_row_count_for_semantics
                    ).detach()
                )
                stats["listwise_semantic_trace_extracted_frac"].append(
                    (
                        trace_extracted_mask_grouped.to(torch.float32).sum()
                        / valid_row_count_for_semantics
                    ).detach()
                )
                stats["listwise_semantic_signature_extracted_frac"].append(
                    (
                        signature_extracted_mask_grouped.to(torch.float32).sum()
                        / valid_row_count_for_semantics
                    ).detach()
                )
                stats["listwise_semantic_cluster_valid_frac"].append(
                    (
                        semantic_valid_row_mask_grouped.to(torch.float32).sum()
                        / valid_row_count_for_semantics
                    ).detach()
                )
                stats["listwise_semantic_trace_truncated_frac"].append(
                    semantic_trace_truncated_frac.detach()
                )
                prompt_token_counts_grouped = (
                    prompt_response_masks.to(dtype=policy_seq_logps_grouped.dtype)
                    .sum(dim=2)
                    .clamp(min=1.0)
                )
                output_entropy_grouped = (
                    -prompt_behavior_seq_raw.detach().to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    / prompt_token_counts_grouped.to(device=device)
                ).clamp(min=0.0)
                (
                    exact_quality_utility_grouped,
                    exact_quality_grouped,
                    exact_semantic_drx_diag,
                ) = compute_quality_centered_semantic_drx_utilities(
                    reward_grouped=mb_reward_grouped.detach(),
                    output_entropy_grouped=output_entropy_grouped.detach(),
                    semantic_entropy_lambda=float(args.semantic_entropy_lambda),
                    candidate_correctness_grouped=mb_correctness_grouped.detach(),
                    valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                    semantic_correctness_target_frac=float(
                        args.maxent_semantic_correctness_target_frac
                    ),
                    semantic_correctness_sharpness=float(
                        args.maxent_semantic_correctness_sharpness
                    ),
                )
                exact_centered_advantages_grouped = compute_group_centered_advantages(
                    reward_grouped=exact_quality_utility_grouped.detach().to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    ),
                    valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                )
                exact_quality_for_adv_grouped = exact_quality_grouped.detach().to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                exact_semantic_piece_grouped = (
                    exact_quality_utility_grouped.detach().to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    - exact_quality_for_adv_grouped
                )
                exact_correctness_adv_grouped = compute_group_centered_advantages(
                    reward_grouped=exact_quality_for_adv_grouped,
                    valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                )
                exact_semantic_adv_grouped = compute_group_centered_advantages(
                    reward_grouped=exact_semantic_piece_grouped,
                    valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                )
                quality_valid_values = exact_quality_grouped[grouped_loss_masks_prompt]
                semantic_surprisal_valid_values = (
                    exact_semantic_drx_diag.semantic_surprisal_grouped[
                        grouped_loss_masks_prompt
                    ]
                )
                semantic_gate_valid_values = (
                    exact_semantic_drx_diag.semantic_gate_grouped[
                        grouped_loss_masks_prompt.any(dim=1)
                    ]
                    if exact_semantic_drx_diag.semantic_gate_grouped is not None
                    else torch.empty(
                        (0,),
                        device=device,
                        dtype=torch.float32,
                    )
                )
                utility_valid_values = exact_quality_utility_grouped[
                    grouped_loss_masks_prompt
                ]
                semantic_piece_valid_values = exact_semantic_piece_grouped[
                    grouped_loss_masks_prompt
                ]
                correctness_adv_valid_values = exact_correctness_adv_grouped[
                    grouped_loss_masks_prompt
                ]
                semantic_adv_valid_values = exact_semantic_adv_grouped[
                    grouped_loss_masks_prompt
                ]
                if correctness_adv_valid_values.numel() > 0:
                    correctness_adv_abs_mean = (
                        correctness_adv_valid_values.abs().mean().detach().to(torch.float32)
                    )
                    semantic_adv_abs_mean = (
                        semantic_adv_valid_values.abs().mean().detach().to(torch.float32)
                    )
                else:
                    correctness_adv_abs_mean = self._listwise_scalar(0.0).to(
                        torch.float32
                    )
                    semantic_adv_abs_mean = self._listwise_scalar(0.0).to(
                        torch.float32
                    )
                valid_group_mask = grouped_loss_masks_prompt.detach().any(dim=1)
                valid_count_grouped = (
                    grouped_loss_masks_prompt.detach()
                    .to(torch.float32)
                    .sum(dim=1)
                    .clamp(min=1.0)
                )
                correctness_frac_grouped = (
                    torch.where(
                        grouped_loss_masks_prompt.detach(),
                        exact_quality_for_adv_grouped.to(torch.float32),
                        torch.zeros_like(exact_quality_for_adv_grouped).to(
                            torch.float32
                        ),
                    ).sum(dim=1)
                    / valid_count_grouped
                )
                semantic_adv_abs_grouped = (
                    torch.where(
                        grouped_loss_masks_prompt.detach(),
                        exact_semantic_adv_grouped.abs().to(torch.float32),
                        torch.zeros_like(exact_semantic_adv_grouped).to(torch.float32),
                    ).sum(dim=1)
                    / valid_count_grouped
                )
                all_wrong_group_mask = valid_group_mask & (
                    correctness_frac_grouped <= 1e-8
                )
                all_correct_group_mask = valid_group_mask & (
                    correctness_frac_grouped >= 1.0 - 1e-8
                )
                mixed_group_mask = (
                    valid_group_mask & ~all_wrong_group_mask & ~all_correct_group_mask
                )

                def _mean_group_or_zero(mask: torch.Tensor) -> torch.Tensor:
                    if bool(mask.any().item()):
                        return semantic_adv_abs_grouped[mask].mean().detach().to(
                            torch.float32
                        )
                    return self._listwise_scalar(0.0).to(torch.float32)

                semantic_adv_abs_all_wrong = _mean_group_or_zero(all_wrong_group_mask)
                semantic_adv_abs_mixed = _mean_group_or_zero(mixed_group_mask)
                semantic_adv_abs_all_correct = _mean_group_or_zero(
                    all_correct_group_mask
                )
                if bool(valid_group_mask.any().item()):
                    semantic_effective_group_frac = (
                        (valid_group_mask & (semantic_adv_abs_grouped > 1e-8))
                        .to(torch.float32)
                        .sum()
                        / valid_group_mask.to(torch.float32).sum().clamp(min=1.0)
                    ).detach()
                else:
                    semantic_effective_group_frac = self._listwise_scalar(0.0).to(
                        torch.float32
                    )
                if correctness_adv_valid_values.numel() > 0:
                    cosine_denom = (
                        correctness_adv_valid_values.norm()
                        * semantic_adv_valid_values.norm()
                    )
                    if bool((cosine_denom > 1e-12).item()):
                        semantic_correctness_adv_cosine = (
                            (
                                correctness_adv_valid_values
                                * semantic_adv_valid_values
                            ).sum()
                            / cosine_denom
                        ).detach().to(torch.float32)
                    else:
                        semantic_correctness_adv_cosine = self._listwise_scalar(0.0).to(
                            torch.float32
                        )
                else:
                    semantic_correctness_adv_cosine = self._listwise_scalar(0.0).to(
                        torch.float32
                    )
                stats["listwise_exact_quality_mean"].append(
                    (
                        quality_valid_values.mean().detach().to(torch.float32)
                        if quality_valid_values.numel() > 0
                        else self._listwise_scalar(0.0)
                    )
                )
                stats["listwise_exact_semantic_surprisal_mean"].append(
                    (
                        semantic_surprisal_valid_values.mean()
                        .detach()
                        .to(torch.float32)
                        if semantic_surprisal_valid_values.numel() > 0
                        else self._listwise_scalar(0.0)
                    )
                )
                stats["listwise_exact_semantic_gate_mean"].append(
                    (
                        semantic_gate_valid_values.mean().detach().to(torch.float32)
                        if semantic_gate_valid_values.numel() > 0
                        else self._listwise_scalar(0.0)
                    )
                )
                stats["listwise_exact_semantic_piece_mean"].append(
                    (
                        semantic_piece_valid_values.mean().detach().to(torch.float32)
                        if semantic_piece_valid_values.numel() > 0
                        else self._listwise_scalar(0.0)
                    )
                )
                stats["listwise_exact_correctness_adv_abs_mean"].append(
                    correctness_adv_abs_mean
                )
                stats["listwise_exact_semantic_adv_abs_mean"].append(
                    semantic_adv_abs_mean
                )
                stats["listwise_exact_semantic_to_correctness_adv_ratio"].append(
                    semantic_adv_abs_mean / (correctness_adv_abs_mean + 1e-8)
                )
                stats["listwise_exact_semantic_adv_fraction"].append(
                    semantic_adv_abs_mean
                    / (semantic_adv_abs_mean + correctness_adv_abs_mean + 1e-8)
                )
                stats["listwise_exact_semantic_adv_abs_mean_all_wrong"].append(
                    semantic_adv_abs_all_wrong
                )
                stats["listwise_exact_semantic_adv_abs_mean_mixed"].append(
                    semantic_adv_abs_mixed
                )
                stats["listwise_exact_semantic_adv_abs_mean_all_correct"].append(
                    semantic_adv_abs_all_correct
                )
                stats["listwise_exact_semantic_effective_group_frac"].append(
                    semantic_effective_group_frac
                )
                stats["listwise_exact_semantic_correctness_adv_cosine"].append(
                    semantic_correctness_adv_cosine
                )
                stats["listwise_exact_utility_mean"].append(
                    (
                        utility_valid_values.mean().detach().to(torch.float32)
                        if utility_valid_values.numel() > 0
                        else self._listwise_scalar(0.0)
                    )
                )

                tie_break_reward_grouped, tie_break_diag, tie_break_anchor_source = (
                    self._semantic_neutral_tiebreak_rewards_grouped(
                        behavior_seq_logps_grouped=behavior_seq_grouped,
                        reference_seq_logps_grouped=mb_ref_seq_grouped,
                        candidate_correctness_grouped=mb_correctness_grouped,
                        cluster_ids_grouped=semantic_cluster_bundle.cluster_ids_grouped,
                        valid_row_mask_grouped=grouped_loss_masks_prompt,
                    )
                )
                (
                    adjusted_prompt_advantages_grouped,
                    raw_neutral_group_mask,
                    tiebreak_applied_group_mask,
                ) = apply_neutral_tiebreak_to_advantages(
                    row_advantages_grouped=probe_row_advantages_grouped,
                    utility_grouped=probe_drgrpo_utility_grouped.detach(),
                    tiebreak_values_grouped=tie_break_reward_grouped.to(
                        device=device,
                        dtype=probe_row_advantages_grouped.dtype,
                    ),
                    valid_row_mask_grouped=grouped_loss_masks_prompt,
                    enabled=float(args.maxent_reward_shaping_alpha) > 0.0,
                    neutral_eps=1e-8,
                )
                prompt_advantages = adjusted_prompt_advantages_grouped.to(
                    device=new_logps.device,
                    dtype=new_logps.dtype,
                )
                probe_row_advantages_grouped = adjusted_prompt_advantages_grouped.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                adjusted_local_row_advantages = prompt_advantages[:, rank].reshape(-1)
                if exact_drx_weight_source == "sequence_clipped":
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
                        grouped_loss_masks_prompt,
                        probe_drgrpo_utility_grouped,
                        torch.zeros_like(probe_drgrpo_utility_grouped),
                    )
                else:
                    adjusted_probe_drgrpo_pg_row_loss, _, _, _ = (
                        compute_token_level_clip_loss(
                            new_logps=new_logps,
                            behavior_logps=local_behavior_logps.to(new_logps.dtype),
                            response_masks=local_response_masks,
                            row_advantages=adjusted_local_row_advantages,
                            clip_low=float(args.cliprange),
                            clip_high=float(args.cliprange),
                            constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                        )
                    )
                    if exact_drx_weight_source == "clipped":
                        adjusted_local_utility_score = (
                            -adjusted_probe_drgrpo_pg_row_loss.detach()
                        )
                    elif exact_drx_weight_source == "unclipped":
                        adjusted_local_utility_score = aggregate_masked_row_values(
                            probe_token_ratio
                            * adjusted_local_row_advantages.unsqueeze(1),
                            local_response_masks,
                            constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                        ).detach()
                    elif exact_drx_weight_source == "local_linear":
                        adjusted_local_utility_score = aggregate_masked_row_values(
                            torch.ones_like(new_logps)
                            * adjusted_local_row_advantages.unsqueeze(1),
                            local_response_masks,
                            constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                        ).detach()
                    else:
                        raise ValueError(
                            "Unsupported maxent_exact_drx_weight_source for the "
                            f"row-sharded exact DrX path: {exact_drx_weight_source}"
                        )
                    probe_drgrpo_utility_grouped = (
                        self._all_gather_prompt_values(
                            adjusted_local_utility_score.to(
                                device=device,
                                dtype=torch.float32,
                            )
                        )
                        .transpose(0, 1)
                        .contiguous()
                        .to(
                            device=device,
                            dtype=policy_seq_logps_grouped.dtype,
                        )
                    )
                    probe_drgrpo_utility_grouped = torch.where(
                        grouped_loss_masks_prompt,
                        probe_drgrpo_utility_grouped,
                        torch.zeros_like(probe_drgrpo_utility_grouped),
                    )
                stats["listwise_raw_neutral_group_frac"].append(
                    raw_neutral_group_mask.to(torch.float32).mean().detach()
                )
                stats["listwise_neutral_tiebreak_applied_group_frac"].append(
                    tiebreak_applied_group_mask.to(torch.float32).mean().detach()
                )
                stats["listwise_neutral_tiebreak_alpha"].append(
                    self._listwise_scalar(float(args.maxent_reward_shaping_alpha))
                )
                stats["listwise_neutral_tiebreak_anchor_source_behavior"].append(
                    self._listwise_scalar(
                        1.0 if tie_break_anchor_source == "behavior" else 0.0
                    )
                )
                stats["listwise_neutral_tiebreak_anchor_source_reference"].append(
                    self._listwise_scalar(
                        1.0 if tie_break_anchor_source == "reference" else 0.0
                    )
                )
                if tie_break_diag is not None:
                    valid_tiebreak_values = tie_break_reward_grouped[
                        grouped_loss_masks_prompt
                    ]
                    stats["listwise_neutral_tiebreak_prompt_alpha_mean"].append(
                        tie_break_diag.prompt_alpha_grouped.mean()
                        .detach()
                        .to(torch.float32)
                    )
                    stats["listwise_neutral_tiebreak_correct_anchor_mass_mean"].append(
                        tie_break_diag.correct_anchor_mass_grouped.mean()
                        .detach()
                        .to(torch.float32)
                    )
                    stats["listwise_neutral_tiebreak_reward_mean"].append(
                        (
                            valid_tiebreak_values.mean().detach().to(torch.float32)
                            if valid_tiebreak_values.numel() > 0
                            else self._listwise_scalar(0.0)
                        )
                    )
                else:
                    stats["listwise_neutral_tiebreak_prompt_alpha_mean"].append(
                        self._listwise_scalar(0.0)
                    )
                    stats["listwise_neutral_tiebreak_correct_anchor_mass_mean"].append(
                        self._listwise_scalar(0.0)
                    )
                    stats["listwise_neutral_tiebreak_reward_mean"].append(
                        self._listwise_scalar(0.0)
                    )

                semantic_ref_seq_grouped = (
                    mb_ref_seq_grouped_raw
                    if exact_drx_weight_source == "sequence_clipped"
                    else mb_ref_seq_grouped
                ).detach()
                behavior_log_probs_grouped_raw = masked_group_log_softmax(
                    behavior_seq_grouped_raw.to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    ),
                    grouped_loss_masks_prompt,
                )
                behavior_probs_grouped_raw = torch.where(
                    grouped_loss_masks_prompt,
                    torch.exp(behavior_log_probs_grouped_raw),
                    torch.zeros_like(behavior_log_probs_grouped_raw),
                )
                best_drgrpo_utility_grouped = torch.where(
                    grouped_loss_masks_prompt,
                    probe_drgrpo_utility_grouped.detach(),
                    torch.full_like(
                        probe_drgrpo_utility_grouped,
                        utility_dtype_info.min,
                    ),
                ).amax(dim=1)
                best_drgrpo_utility_grouped = torch.where(
                    grouped_loss_masks_prompt.any(dim=1),
                    best_drgrpo_utility_grouped,
                    torch.zeros_like(best_drgrpo_utility_grouped),
                )
                behavior_expected_drgrpo_utility_grouped = (
                    behavior_probs_grouped_raw.detach()
                    * torch.where(
                        grouped_loss_masks_prompt,
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
                    float(effective_competitive_mode_budget_max)
                    * semantic_alpha_raw_grouped
                    * grouped_loss_masks_prompt.any(dim=1).to(
                        dtype=policy_seq_logps_grouped.dtype
                    )
                )
                semantic_mass_weights_grouped = None
                if str(args.maxent_semantic_remix_mode) == "anchor_rare":
                    semantic_behavior_seq_grouped = (
                        behavior_seq_grouped_raw
                        if exact_drx_weight_source == "sequence_clipped"
                        else behavior_seq_grouped
                    )
                    (
                        anchor_semantic_utility_grouped,
                        semantic_mass_weights_grouped,
                        semantic_anchor_mass_source,
                    ) = self._anchor_relative_semantic_mass_weights_grouped(
                        behavior_seq_logps_grouped=semantic_behavior_seq_grouped,
                        reference_seq_logps_grouped=semantic_ref_seq_grouped,
                        valid_row_mask_grouped=grouped_loss_masks_prompt,
                        tau=current_tau,
                        candidate_kl_coef=candidate_kl_coef,
                    )
                    stats["listwise_semantic_anchor_mass_source_behavior"].append(
                        self._listwise_scalar(
                            1.0 if semantic_anchor_mass_source == "behavior" else 0.0
                        )
                    )
                    stats["listwise_semantic_anchor_mass_source_reference"].append(
                        self._listwise_scalar(
                            1.0 if semantic_anchor_mass_source == "reference" else 0.0
                        )
                    )
                    valid_anchor_utility = anchor_semantic_utility_grouped[
                        grouped_loss_masks_prompt
                    ]
                    stats["listwise_semantic_anchor_utility_mean"].append(
                        (
                            valid_anchor_utility.mean().detach().to(torch.float32)
                            if valid_anchor_utility.numel() > 0
                            else self._listwise_scalar(0.0)
                        )
                    )
                cluster_ids_grouped = semantic_cluster_bundle.cluster_ids_grouped.to(
                    device=device
                )
                drx_bundle = build_drx_target_bundle(
                    utility_grouped=exact_quality_utility_grouped.detach().to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    ),
                    ref_seq_logps_grouped=semantic_ref_seq_grouped,
                    valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                    competitive_mode_tau=effective_competitive_mode_tau,
                    competitive_mode_gap=competitive_mode_gap,
                    competitive_mode_top_k=competitive_mode_top_k,
                    competitive_mode_budget_grouped=semantic_explore_budget_grouped.detach(),
                    competitive_mode_budget_max=effective_competitive_mode_budget_max,
                    competitive_mode_intra_tau=effective_competitive_mode_intra_tau,
                    prompt_select_min_alpha_frac=effective_prompt_select_min_alpha_frac,
                    competitive_mode_positive_only=competitive_mode_positive_only,
                    semantic_guard_max_expected_len_delta=semantic_guard_max_expected_len_delta,
                    semantic_guard_max_expected_format_drop=semantic_guard_max_expected_format_drop,
                    tau=current_tau,
                    candidate_kl_coef=candidate_kl_coef,
                    neutral_eps=1e-8,
                    neutral_projection_coef=neutral_projection_coef,
                    semantic_remix_mode=str(args.maxent_semantic_remix_mode),
                )
                weights_grouped = drx_bundle.w_star_grouped.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                informative_group_mask = drx_bundle.informative_group_mask.to(
                    device=device
                )
                neutral_group_mask = drx_bundle.neutral_group_mask.to(device=device)
                contributing_group_mask = drx_bundle.contributing_group_mask.to(
                    device=device
                )
                active_group_mask = informative_group_mask
                active_group_count = int(active_group_mask.to(torch.int64).sum().item())
                contributing_group_count = int(
                    contributing_group_mask.to(torch.int64).sum().item()
                )
                target_weights_grouped = torch.where(
                    grouped_loss_masks_prompt,
                    weights_grouped,
                    torch.zeros_like(weights_grouped),
                )
                token_target_weights_grouped = torch.where(
                    grouped_loss_masks_prompt,
                    drx_bundle.token_target_grouped.to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    ),
                    torch.zeros_like(weights_grouped),
                )
                projection_target_grouped = torch.where(
                    grouped_loss_masks_prompt,
                    drx_bundle.projection_target_grouped.to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    ),
                    torch.zeros_like(weights_grouped),
                )
                projection_group_scale = drx_bundle.projection_group_scale.to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                global_projection_group_weight = float(
                    contributing_group_mask.to(projection_group_scale.dtype)
                    .sum()
                    .item()
                )
                global_projection_scale_mass = float(
                    projection_group_scale.sum().item()
                )

                semantic_cluster_entropy_t, semantic_cluster_count_t = (
                    compute_normalized_semantic_cluster_entropy(
                        candidate_probs_grouped=behavior_probs_grouped_raw.detach(),
                        cluster_ids_grouped=cluster_ids_grouped,
                        valid_row_mask_grouped=grouped_loss_masks_prompt,
                        normalizer_group_size=group_size,
                    )
                )
                stats["listwise_semantic_cluster_entropy"].append(
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
                    stats["listwise_semantic_distinct_correct_mode_count"].append(
                        semantic_diag.distinct_correct_mode_count_grouped.mean().detach()
                    )
                    stats["listwise_semantic_competitive_mode_top_score"].append(
                        semantic_diag.top_score_grouped.mean().detach()
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
                        )
                        .mean()
                        .detach()
                    )
                    stats["listwise_semantic_explore_applied_group_frac"].append(
                        semantic_diag.explore_applied_group_mask.to(
                            semantic_diag.explore_budget_grouped.dtype
                        )
                        .mean()
                        .detach()
                    )
                    stats["listwise_semantic_prompt_selected_frac"].append(
                        semantic_diag.prompt_selected_group_mask.to(
                            semantic_diag.explore_budget_grouped.dtype
                        )
                        .mean()
                        .detach()
                    )
                    stats["listwise_semantic_prompt_rejected_low_opp_frac"].append(
                        semantic_diag.prompt_rejected_low_opp_group_mask.to(
                            semantic_diag.explore_budget_grouped.dtype
                        )
                        .mean()
                        .detach()
                    )
                    stats["listwise_semantic_prompt_rejected_nonpositive_frac"].append(
                        semantic_diag.prompt_rejected_nonpositive_group_mask.to(
                            semantic_diag.explore_budget_grouped.dtype
                        )
                        .mean()
                        .detach()
                    )
                    stats["listwise_semantic_prompt_rejected_len_guard_frac"].append(
                        semantic_diag.prompt_rejected_len_guard_group_mask.to(
                            semantic_diag.explore_budget_grouped.dtype
                        )
                        .mean()
                        .detach()
                    )
                    stats["listwise_semantic_prompt_rejected_format_guard_frac"].append(
                        semantic_diag.prompt_rejected_format_guard_group_mask.to(
                            semantic_diag.explore_budget_grouped.dtype
                        )
                        .mean()
                        .detach()
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
                    (mb_correctness_grouped >= _VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD)
                    & grouped_loss_masks_prompt
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
                gain_group_mask = semantic_cluster_count_t > 0.0
                gain_group_count = int(gain_group_mask.to(torch.int64).sum().item())
                semantic_entropy_prompt_mean_local = (
                    semantic_cluster_entropy_t[gain_group_mask].mean()
                    if gain_group_count > 0
                    else self._listwise_scalar(0.0)
                )
                exploration_gain_any_correct_mean_local = (
                    exploration_gain_any_correct_t[gain_group_mask].mean()
                    if gain_group_count > 0
                    else self._listwise_scalar(0.0)
                )
                exploration_gain_drgrpo_mean_local = (
                    exploration_gain_drgrpo_t[gain_group_mask].mean()
                    if gain_group_count > 0
                    else self._listwise_scalar(0.0)
                )
                stats["listwise_semantic_exploration_gain_any_correct"].append(
                    exploration_gain_any_correct_mean_local.detach()
                )
                stats["listwise_semantic_exploration_gain_drgrpo"].append(
                    exploration_gain_drgrpo_mean_local.detach()
                )
                ref_available = bool(candidate_kl_coef > 0.0)
                ref_semantic_entropy_t = torch.zeros_like(semantic_cluster_entropy_t)
                if ref_available:
                    ref_log_probs_grouped = masked_group_log_softmax(
                        semantic_ref_seq_grouped.to(
                            device=device,
                            dtype=weights_grouped.dtype,
                        ),
                        grouped_loss_masks_prompt,
                    )
                    ref_probs_grouped = torch.where(
                        grouped_loss_masks_prompt,
                        torch.exp(ref_log_probs_grouped),
                        torch.zeros_like(ref_log_probs_grouped),
                    )
                    ref_semantic_entropy_t, _ = (
                        compute_normalized_semantic_cluster_entropy(
                            candidate_probs_grouped=ref_probs_grouped.detach(),
                            cluster_ids_grouped=cluster_ids_grouped,
                            valid_row_mask_grouped=grouped_loss_masks_prompt,
                            normalizer_group_size=group_size,
                        )
                    )
                stats["listwise_semantic_cluster_entropy_ref"].append(
                    ref_semantic_entropy_t.mean().detach()
                )
                stats["listwise_semantic_cluster_entropy_gain"].append(
                    (semantic_cluster_entropy_t - ref_semantic_entropy_t)
                    .mean()
                    .detach()
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
                    policy_seq_logps_grouped.detach(),
                    grouped_loss_masks_prompt,
                )
                policy_probs_grouped = torch.where(
                    grouped_loss_masks_prompt,
                    torch.exp(policy_log_probs_grouped),
                    torch.zeros_like(policy_log_probs_grouped),
                )
                per_group_projection_ce = -(
                    target_weights_grouped * policy_log_probs_grouped
                ).sum(dim=1)
                projection_ce_loss = (per_group_projection_ce * 0.0).sum()
                if global_projection_group_weight > 0.0:
                    projection_ce_loss = (
                        projection_group_scale.detach() * per_group_projection_ce
                    ).sum() / max(global_projection_group_weight, 1e-8)
                projection_ce_loss_effective = (
                    float(sequence_aux_coef) * projection_ce_loss
                )
                listwise_centered_adv_grouped = compute_listwise_centered_advantages(
                    weights_grouped=target_weights_grouped,
                    behavior_seq_logps_grouped=behavior_seq_grouped,
                    valid_row_mask_grouped=grouped_loss_masks_prompt,
                ).to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                zero_component = projection_ce_loss.detach() * 0.0
                infos["listwise_ce_loss"] = projection_ce_loss.detach()
                infos["listwise_projection_ce_loss"] = projection_ce_loss.detach()
                infos["listwise_projection_ce_loss_effective"] = (
                    projection_ce_loss_effective.detach()
                )

                clip_loss = None
                clip_low = clip_high = None
                baseline_value = None
                baseline_grouped = None
                reward_mass_grouped = None
                clip_adv_grouped = None
                clip_coef = 0.0
                clip_mode = normalize_maxent_clip_mode(
                    getattr(args, "maxent_clip_mode", "sequence")
                )
                effective_clip_mode = "none"
                if drgrpo_token_primary:
                    clip_low = clip_high = float(args.cliprange)
                    effective_clip_mode = "token"
                elif bool(args.maxent_use_clip_objective):
                    clip_coef = coerce_non_negative_float(
                        args.maxent_clip_objective_coef,
                        default=1.0,
                    )
                    if clip_coef > 0.0:
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
                            valid_group_denoms = (
                                grouped_loss_masks_prompt.sum(
                                    dim=1,
                                    keepdim=True,
                                )
                                .clamp(min=1)
                                .to(weights_grouped.dtype)
                            )
                            baseline_grouped = torch.where(
                                grouped_loss_masks_prompt,
                                1.0 / valid_group_denoms,
                                torch.zeros_like(weights_grouped),
                            )
                        else:
                            baseline_value = float(baseline)
                            baseline_grouped = torch.where(
                                grouped_loss_masks_prompt,
                                torch.full_like(weights_grouped, baseline_value),
                                torch.zeros_like(weights_grouped),
                            )
                        if bool(args.maxent_clip_preserve_reward_mass):
                            reward_mass_grouped = torch.where(
                                grouped_loss_masks_prompt,
                                mb_reward_grouped,
                                torch.zeros_like(mb_reward_grouped),
                            ).sum(dim=1, keepdim=True)
                        clip_adv_grouped = compute_listwise_clip_advantages(
                            weights_grouped=weights_grouped,
                            valid_row_mask_grouped=grouped_loss_masks_prompt,
                            baseline_value=baseline_value,
                            baseline_grouped=baseline_grouped,
                            reward_mass_grouped=reward_mass_grouped,
                        )
                        if active_group_count > 0 and reward_mass_grouped is not None:
                            infos["listwise_clip_reward_mass_mean"] = (
                                reward_mass_grouped[active_group_mask].mean().detach()
                            )
                if not drgrpo_token_primary and effective_clip_mode == "token":
                    raise ValueError(
                        "The row-sharded exact DrX path currently supports "
                        "maxent_clip_mode=sequence or none when "
                        "maxent_drgrpo_token_primary=0."
                    )
                if not drgrpo_token_primary and effective_clip_mode != "none":
                    infos["listwise_clip_preserve_reward_mass"] = torch.tensor(
                        float(bool(args.maxent_clip_preserve_reward_mass)),
                        device=device,
                    )
                if float(args.beta) > 0.0:
                    raise NotImplementedError(
                        "The row-sharded exact DrX path currently requires beta=0 "
                        "so the local-row PPO clip term remains the pure Dr.GRPO surrogate."
                    )
                drgrpo_pg_loss = (
                    gathered_probe_row_loss.to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    * grouped_loss_masks_prompt.to(policy_seq_logps_grouped.dtype)
                ).mean()
                infos["drgrpo_pg_loss"] = drgrpo_pg_loss.detach()
                infos["drgrpo_primary_loss"] = drgrpo_pg_loss.detach()

                if str(args.maxent_drgrpo_token_advantage_source) == "utility_centered":
                    drgrpo_token_adv_grouped = exact_centered_advantages_grouped.to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    weighted_drgrpo_multiplier_grouped = torch.ones_like(
                        drgrpo_token_adv_grouped
                    )
                    weighted_drgrpo_row_adv_grouped = drgrpo_token_adv_grouped
                else:
                    drgrpo_token_adv_grouped = prompt_advantages.to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    weighted_drgrpo_multiplier_grouped = float(
                        group_size
                    ) * token_target_weights_grouped.to(
                        device=device,
                        dtype=policy_seq_logps_grouped.dtype,
                    )
                    weighted_drgrpo_row_adv_grouped = (
                        weighted_drgrpo_multiplier_grouped * drgrpo_token_adv_grouped
                    )
                token_active_row_mask_grouped = build_drgrpo_token_active_row_mask(
                    advantage_source=str(args.maxent_drgrpo_token_advantage_source),
                    informative_group_mask=informative_group_mask,
                    valid_row_mask_grouped=grouped_loss_masks_prompt,
                    utility_centered_advantages_grouped=drgrpo_token_adv_grouped,
                )
                global_drgrpo_active_row_count = int(
                    token_active_row_mask_grouped.to(torch.int64).sum().item()
                )
                token_signal_row_mask_grouped = (
                    token_active_row_mask_grouped
                    & torch.isfinite(weighted_drgrpo_row_adv_grouped)
                    & (weighted_drgrpo_row_adv_grouped.abs() > 1e-8)
                )
                global_drgrpo_signal_row_count = int(
                    token_signal_row_mask_grouped.to(torch.int64).sum().item()
                )
                weighted_drgrpo_delta_adv_grouped = (
                    weighted_drgrpo_row_adv_grouped - drgrpo_token_adv_grouped
                )
                local_weighted_drgrpo_row_adv = weighted_drgrpo_row_adv_grouped[
                    :, rank
                ].to(
                    device=device,
                    dtype=policy_seq_logps_grouped.dtype,
                )
                local_weighted_drgrpo_row_adv = local_weighted_drgrpo_row_adv.to(
                    device=new_logps.device,
                    dtype=new_logps.dtype,
                )
                weighted_drgrpo_pg_row_loss, _, _, _ = compute_token_level_clip_loss(
                    new_logps=new_logps,
                    behavior_logps=local_behavior_logps.to(new_logps.dtype),
                    response_masks=local_response_masks,
                    row_advantages=local_weighted_drgrpo_row_adv,
                    clip_low=float(args.cliprange),
                    clip_high=float(args.cliprange),
                    constant_normalizer=self._listwise_token_clip_constant_normalizer(),
                )
                gathered_weighted_drgrpo_row_loss = (
                    self._all_gather_prompt_values(
                        weighted_drgrpo_pg_row_loss.detach().to(
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .to(dtype=policy_seq_logps_grouped.dtype)
                )
                if bool(token_active_row_mask_grouped.any().item()):
                    weighted_drgrpo_pg_loss = gathered_weighted_drgrpo_row_loss[
                        token_active_row_mask_grouped
                    ].mean()
                    drgrpo_adv_abs_mean = (
                        drgrpo_token_adv_grouped[token_active_row_mask_grouped]
                        .abs()
                        .mean()
                    )
                    listwise_delta_adv_abs_mean = (
                        weighted_drgrpo_delta_adv_grouped[token_active_row_mask_grouped]
                        .abs()
                        .mean()
                    )
                    combined_adv_abs_mean = (
                        weighted_drgrpo_row_adv_grouped[token_active_row_mask_grouped]
                        .abs()
                        .mean()
                    )
                    listwise_weight_deviation = (
                        (
                            weighted_drgrpo_multiplier_grouped[
                                token_active_row_mask_grouped
                            ]
                            - 1.0
                        )
                        .abs()
                        .mean()
                    )
                else:
                    weighted_drgrpo_pg_loss = zero_component
                    drgrpo_adv_abs_mean = zero_component
                    listwise_delta_adv_abs_mean = zero_component
                    combined_adv_abs_mean = zero_component
                    listwise_weight_deviation = zero_component
                listwise_adv_abs_mean = (
                    listwise_centered_adv_grouped[contributing_group_mask].abs().mean()
                    if contributing_group_count > 0
                    else zero_component
                )
                if drgrpo_token_primary:
                    infos["pg_loss"] = weighted_drgrpo_pg_loss.detach()
                    infos["combined_token_pg_loss"] = weighted_drgrpo_pg_loss.detach()
                    infos["listwise_adv_abs_mean"] = listwise_adv_abs_mean.detach()
                    infos["listwise_adv_abs_mean_scaled"] = (
                        listwise_delta_adv_abs_mean.detach()
                    )
                    infos["drgrpo_adv_abs_mean"] = drgrpo_adv_abs_mean.detach()
                    infos["combined_adv_abs_mean"] = combined_adv_abs_mean.detach()
                    infos["listwise_raw_to_drgrpo_ratio"] = (
                        listwise_weight_deviation.detach()
                    )
                    infos["listwise_post_scale_ratio"] = torch.abs(
                        listwise_delta_adv_abs_mean.detach()
                    ) / (torch.abs(drgrpo_adv_abs_mean.detach()) + 1e-8)
                    infos["listwise_aux_loss_raw"] = (
                        weighted_drgrpo_pg_loss.detach() - drgrpo_pg_loss.detach()
                    )
                    infos["listwise_aux_loss_weighted"] = infos["listwise_aux_loss_raw"]
                    infos["listwise_aux_loss_effective"] = infos[
                        "listwise_aux_loss_raw"
                    ]
                    infos["listwise_helpfulness_proxy"] = torch.abs(
                        infos["listwise_aux_loss_effective"]
                    ) / (torch.abs(infos["drgrpo_primary_loss"]) + 1e-8)
                    infos["listwise_helpfulness_proxy_valid"] = torch.tensor(
                        1.0,
                        device=device,
                    )
                    infos["clip_loss"] = zero_component
                    infos["objective_effective_total_loss"] = (
                        weighted_drgrpo_pg_loss.detach()
                        + projection_ce_loss_effective.detach()
                    )
                else:
                    infos["pg_loss"] = projection_ce_loss.detach()
                    infos["combined_token_pg_loss"] = zero_component
                    infos["listwise_adv_abs_mean"] = listwise_adv_abs_mean.detach()
                    infos["listwise_adv_abs_mean_scaled"] = zero_component
                    infos["drgrpo_adv_abs_mean"] = drgrpo_adv_abs_mean.detach()
                    infos["combined_adv_abs_mean"] = zero_component
                    infos["listwise_raw_to_drgrpo_ratio"] = zero_component
                    infos["listwise_post_scale_ratio"] = zero_component
                    infos["listwise_helpfulness_proxy"] = zero_component
                    infos["listwise_helpfulness_proxy_valid"] = torch.tensor(
                        0.0,
                        device=device,
                    )
                    if (
                        effective_clip_mode == "sequence"
                        and clip_adv_grouped is not None
                    ):
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
                            seq_ratio * clip_adv_grouped,
                            seq_ratio_clipped * clip_adv_grouped,
                        )
                        per_group_clip_loss = -clip_objective.sum(dim=1)
                        if active_group_count > 0:
                            clip_loss = per_group_clip_loss[active_group_mask].mean()
                        else:
                            clip_loss = (per_group_clip_loss * 0.0).sum()
                        infos["clip_loss"] = clip_loss.detach()
                        infos["listwise_aux_loss_raw"] = clip_loss.detach()
                        infos["listwise_aux_loss_weighted"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["listwise_aux_loss_effective"] = (
                            float(clip_coef) * clip_loss.detach()
                        )
                        infos["objective_effective_total_loss"] = infos[
                            "listwise_aux_loss_effective"
                        ]

                        is_low_clipped = (seq_ratio < 1.0 - clip_low) & (
                            clip_adv_grouped < 0.0
                        )
                        is_high_clipped = (seq_ratio > 1.0 + clip_high) & (
                            clip_adv_grouped > 0.0
                        )
                        clip_region = is_low_clipped | is_high_clipped
                        active_valid_entries = (
                            grouped_loss_masks_prompt & active_group_mask.unsqueeze(1)
                        )
                        active_valid_entry_count = active_valid_entries.to(
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
                                device=device,
                            )
                            stats["clip_ratio_low"].append(zero_stat)
                            stats["clip_ratio_high"].append(zero_stat)
                            stats["clip_ratio_region"].append(zero_stat)
                    else:
                        infos["clip_loss"] = zero_component
                        infos["listwise_aux_loss_raw"] = projection_ce_loss.detach()
                        infos["listwise_aux_loss_weighted"] = (
                            projection_ce_loss.detach()
                        )
                        infos["listwise_aux_loss_effective"] = (
                            projection_ce_loss.detach()
                        )
                        infos["objective_effective_total_loss"] = (
                            projection_ce_loss.detach()
                        )

                has_token_signal = global_drgrpo_signal_row_count > 0
                has_projection_signal = (
                    float(sequence_aux_coef) > 0.0
                    and global_projection_group_weight > 0.0
                    and global_projection_scale_mass > 0.0
                )
                if drgrpo_token_primary:
                    zero_signal_cadence_update = (
                        global_drgrpo_active_row_count > 0
                        and not has_token_signal
                        and not has_projection_signal
                    )
                    skip_zero_signal_update = (
                        global_drgrpo_active_row_count <= 0
                        and not has_projection_signal
                    )
                else:
                    zero_signal_cadence_update = False
                    skip_zero_signal_update = active_group_count <= 0
                if local_grad_step == 1 and (
                    skip_zero_signal_update or zero_signal_cadence_update
                ):
                    if drgrpo_token_primary:
                        if skip_zero_signal_update:
                            logging.info(
                                "row-sharded exact DrX prompt batch has no valid token "
                                "denominator rows and no projection signal; skipping the "
                                "optimizer step."
                            )
                        else:
                            logging.info(
                                "row-sharded exact DrX prompt batch has valid token "
                                "denominator rows but zero token signal; applying a "
                                "zero-loss backward/optimizer step to match Dr.GRPO "
                                "optimizer cadence."
                            )
                    else:
                        logging.info(
                            "row-sharded exact DrX prompt batch has no active prompt "
                            "groups; skipping the optimizer step."
                        )

                if skip_zero_signal_update:
                    backward_chunk_size = 0
                    clip_backward_chunk_size = 0
                else:
                    if drgrpo_token_primary:
                        if (
                            float(sequence_aux_coef) > 0.0
                            and global_projection_group_weight > 0.0
                            and global_projection_scale_mass > 0.0
                        ):
                            seq_coeffs_grouped = compute_drx_projection_sequence_coefficients(
                                policy_seq_logps_grouped=policy_seq_logps_grouped.detach(),
                                projection_target_grouped=projection_target_grouped.detach(),
                                projection_group_scale=projection_group_scale.detach(),
                                valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                                normalizer_total_group_weight=global_projection_group_weight,
                            ).to(
                                device=device,
                                dtype=policy_seq_logps_grouped.dtype,
                            )
                            seq_coeffs_grouped = (
                                float(sequence_aux_coef) * seq_coeffs_grouped
                            )
                        else:
                            seq_coeffs_grouped = torch.zeros_like(
                                target_weights_grouped,
                                device=device,
                                dtype=policy_seq_logps_grouped.dtype,
                            )
                        token_clip_enabled = global_drgrpo_active_row_count > 0
                        local_active_row_mask = token_active_row_mask_grouped[:, rank]
                        local_row_advantages_for_backward = (
                            local_weighted_drgrpo_row_adv
                        )
                        token_clip_row_count_normalizer = global_drgrpo_active_row_count
                        token_clip_coef = (
                            float(world_size) if token_clip_enabled else 0.0
                        )
                    else:
                        seq_coeffs_grouped = compute_listwise_sequence_coefficients(
                            policy_seq_logps_grouped=policy_seq_logps_grouped.detach(),
                            weights_grouped=target_weights_grouped.detach(),
                            active_group_mask=active_group_mask.detach(),
                            normalizer_active_group_count=active_group_count,
                            valid_row_mask_grouped=grouped_loss_masks_prompt.detach(),
                            behavior_seq_logps_grouped=(
                                behavior_seq_grouped.detach()
                                if clip_coef > 0.0 and effective_clip_mode == "sequence"
                                else None
                            ),
                            clip_row_mask_grouped=(
                                grouped_loss_masks_prompt.detach()
                                if clip_coef > 0.0 and effective_clip_mode == "sequence"
                                else None
                            ),
                            reward_mass_grouped=(
                                reward_mass_grouped.detach()
                                if reward_mass_grouped is not None
                                and clip_coef > 0.0
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
                                and clip_coef > 0.0
                                and effective_clip_mode == "sequence"
                                else None
                            ),
                        ).to(
                            device=device,
                            dtype=policy_seq_logps_grouped.dtype,
                        )
                        token_clip_enabled = False
                        local_active_row_mask = None
                        local_row_advantages_for_backward = None
                        token_clip_row_count_normalizer = None
                        token_clip_coef = 0.0

                    local_seq_coeff = seq_coeffs_grouped[:, rank].to(
                        device=new_logps.device,
                        dtype=new_logps.dtype,
                    ) * float(world_size)
                    backward_chunk_size, clip_backward_chunk_size = (
                        self._backward_listwise_sequence_coefficients(
                            local_input_ids,
                            local_att_mask,
                            local_response_masks,
                            local_seq_coeff.reshape(-1),
                            length_normalize=length_normalize_policy,
                            behavior_logps=(
                                local_behavior_logps if token_clip_enabled else None
                            ),
                            row_advantages=(
                                local_row_advantages_for_backward
                                if token_clip_enabled
                                else None
                            ),
                            active_row_mask=(
                                local_active_row_mask if token_clip_enabled else None
                            ),
                            active_row_count_normalizer=(
                                token_clip_row_count_normalizer
                                if token_clip_enabled
                                else None
                            ),
                            clip_low=float(args.cliprange),
                            clip_high=float(args.cliprange),
                            clip_coef=token_clip_coef,
                        )
                    )

                infos["listwise_policy_backward_chunk_size"] = torch.tensor(
                    float(backward_chunk_size),
                    device=device,
                )
                infos["listwise_clip_backward_chunk_size"] = torch.tensor(
                    float(clip_backward_chunk_size),
                    device=device,
                )
                infos["listwise_policy_prob_mean"] = (
                    policy_probs_grouped.mean().detach()
                )
                infos["listwise_weight_mean"] = weights_grouped.mean().detach()
                infos["listwise_weight_std"] = (
                    weights_grouped.to(torch.float32).std(unbiased=False).detach()
                )
                infos["listwise_weight_entropy"] = active_weight_entropy.detach().to(
                    dtype=torch.float32
                )
                infos["weight_entropy"] = infos["listwise_weight_entropy"]
                infos["listwise_neutral_group_frac"] = (
                    neutral_group_mask.to(torch.float32).mean().detach()
                )
                infos["listwise_informative_group_frac"] = (
                    active_group_mask.to(torch.float32).mean().detach()
                )
                infos["listwise_informative_group_count_global"] = torch.tensor(
                    float(active_group_count),
                    device=device,
                )
                infos["listwise_contributing_group_frac"] = (
                    contributing_group_mask.to(torch.float32).mean().detach()
                )
                infos["listwise_contributing_group_count_global"] = torch.tensor(
                    float(contributing_group_count),
                    device=device,
                )
                infos.update(
                    compute_correctness_group_rate_infos(
                        correctness_grouped=mb_correctness_grouped,
                        valid_row_mask_grouped=grouped_loss_masks_prompt,
                        threshold=_VERIFIED_CORRECTNESS_SCHEDULE_THRESHOLD,
                    )
                )
                infos["listwise_active_group_frac"] = infos[
                    "listwise_informative_group_frac"
                ]
                infos["listwise_active_group_count_global"] = infos[
                    "listwise_informative_group_count_global"
                ]
                infos["listwise_valid_row_frac"] = (
                    grouped_loss_masks_prompt.to(torch.float32).mean().detach()
                )
                infos["listwise_partial_group_frac"] = (
                    (
                        grouped_loss_masks_prompt.any(dim=1)
                        & (~grouped_loss_masks_prompt.all(dim=1))
                    )
                    .to(torch.float32)
                    .mean()
                    .detach()
                )
                infos["listwise_valid_weight_mass"] = (
                    target_weights_grouped.sum(dim=1).mean().detach()
                )
                infos["listwise_clip_mode_sequence"] = torch.tensor(
                    1.0 if effective_clip_mode == "sequence" else 0.0,
                    device=device,
                )
                infos["listwise_clip_mode_token"] = torch.tensor(
                    1.0 if effective_clip_mode == "token" else 0.0,
                    device=device,
                )
                infos["listwise_clip_mode_none"] = torch.tensor(
                    1.0 if effective_clip_mode == "none" else 0.0,
                    device=device,
                )
                infos["listwise_token_clip_primary"] = torch.tensor(
                    0.0,
                    device=device,
                )
                infos["listwise_drgrpo_token_primary"] = torch.tensor(
                    1.0 if drgrpo_token_primary else 0.0,
                    device=device,
                )
                infos["listwise_drgrpo_token_active_row_frac"] = (
                    token_active_row_mask_grouped.to(torch.float32).mean().detach()
                )
                infos["listwise_drgrpo_token_active_row_count_global"] = torch.tensor(
                    float(global_drgrpo_active_row_count),
                    device=device,
                )
                infos["listwise_sequence_aux_coef"] = torch.tensor(
                    float(sequence_aux_coef),
                    device=device,
                )
                infos["listwise_zero_signal_cadence_update"] = torch.tensor(
                    1.0 if zero_signal_cadence_update else 0.0,
                    device=device,
                )
                infos["listwise_candidate_kl_coef"] = torch.tensor(
                    float(candidate_kl_coef),
                    device=device,
                )
                infos["listwise_branch_grad_diagnostics"] = torch.tensor(
                    0.0,
                    device=device,
                )
                infos["listwise_grad_probe_enabled"] = torch.tensor(
                    0.0,
                    device=device,
                )
                infos["listwise_grad_probe_valid"] = torch.tensor(
                    0.0,
                    device=device,
                )
                self._record_listwise_runtime_infos(
                    infos,
                    device=device,
                    skip_zero_signal_update=skip_zero_signal_update,
                    stats=stats,
                )
                for per_update_metric in (
                    "combined_adv_abs_mean",
                    "combined_token_pg_loss",
                    "drgrpo_adv_abs_mean",
                    "drgrpo_pg_loss",
                    "listwise_clip_backward_chunk_size",
                    "listwise_drgrpo_token_active_row_count_global",
                    "listwise_drgrpo_token_active_row_frac",
                    "listwise_policy_backward_chunk_size",
                    "objective_effective_total_loss",
                    "pg_loss",
                ):
                    stats[per_update_metric].append(infos[per_update_metric].detach())

                reduced_weight_entropy = (
                    float(active_weight_entropy.detach().item())
                    if active_group_count > 0
                    else None
                )
                reduced_semantic_entropy_mu = (
                    float(semantic_entropy_prompt_mean_local.detach().item())
                    if gain_group_count > 0
                    else None
                )
                reduced_exploration_gain_any_correct = (
                    float(exploration_gain_any_correct_mean_local.detach().item())
                    if gain_group_count > 0
                    else None
                )
                reduced_exploration_gain_drgrpo = (
                    float(exploration_gain_drgrpo_mean_local.detach().item())
                    if gain_group_count > 0
                    else None
                )
                self._apply_listwise_controller_updates(
                    infos,
                    device=device,
                    skip_zero_signal_update=skip_zero_signal_update,
                    weight_entropy_controller=reduced_weight_entropy,
                    semantic_entropy_mu=reduced_semantic_entropy_mu,
                    exploration_gain_any_correct=reduced_exploration_gain_any_correct,
                    exploration_gain_drgrpo=reduced_exploration_gain_drgrpo,
                    measured_kl=None,
                    update_beta_when_skipped=False,
                )

                if local_grad_step % grad_acc_step == 0:
                    if skip_zero_signal_update:
                        if not self._listwise_zero_signal_skip_warned:
                            logging.warning(
                                "Skipping a zero-signal row-sharded exact DrX optimizer "
                                "step because the prompt group has no projection signal "
                                "and no valid token denominator rows."
                            )
                            self._listwise_zero_signal_skip_warned = True
                    else:
                        if not self._listwise_grad_norm_logging_disabled_warned:
                            logging.warning(
                                "Skipping row-sharded exact DrX policy_grad_norm logging "
                                "because the chunked backward path uses per-sequence "
                                "passes and DeepSpeed gradient-norm collectives can hang "
                                "while ranks finish unevenly."
                            )
                            self._listwise_grad_norm_logging_disabled_warned = True
                    stats["policy_grad_norm"].append(self._listwise_scalar(0.0))
                    stats["get_grad_norm_time"].append(self._listwise_scalar(0.0))

                if not skip_zero_signal_update:
                    self.strategy.optimizer_step(
                        self.optimizer, self.model, self.scheduler
                    )

                if local_grad_step % grad_acc_step == 0 and self.strategy.is_rank_0():
                    logging.info(
                        "row-sharded exact DrX optimizer update %s/%s: status=%s "
                        "prompt_batch=%s owner_ranks=%s prompt_idxs=%s "
                        "active_groups=%s active_rows=%s "
                        "policy_probe_chunk=%s backward_chunk=%s clip_backward_chunk=%s",
                        max(local_grad_step // grad_acc_step, 1),
                        total_optimizer_updates,
                        (
                            "skipped_zero_signal"
                            if skip_zero_signal_update
                            else "applied_zero_signal_cadence"
                            if zero_signal_cadence_update
                            else "applied_exact_drx_weighted_drgrpo"
                        ),
                        int(prompt_batch_ids.numel()),
                        prompt_owner_ranks,
                        prompt_owner_indices,
                        active_group_count,
                        global_drgrpo_active_row_count,
                        policy_chunk_size,
                        backward_chunk_size,
                        clip_backward_chunk_size,
                    )

        return finalize_row_sharded_info_stats(
            infos,
            stats,
            device=input_ids.device if isinstance(input_ids, torch.Tensor) else device,
        )
