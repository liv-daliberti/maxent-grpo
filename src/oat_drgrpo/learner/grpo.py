"""Baseline GRPO learner helpers."""

from __future__ import annotations

import gc
import logging
import math
import time
from collections import defaultdict

import numpy as np
import torch
from oat.utils.ops import masked_mean

from ..listwise import cap_last_valid_token_pos_for_zero_advantage
from ..seed_weights import compute_seed_row_weights
from ..xdr import compute_xdr_row_weights, xdr_group_diagnostics


class ZeroMathGrpoMixin:
    """Small baseline GRPO helpers shared by public objective paths."""

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
    ) -> dict[str, torch.Tensor]:
        args = self.args
        infos: dict[str, torch.Tensor] = {}
        if extra_infos:
            infos.update(extra_infos)
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if policy_vocab_upper_bound is None:
            policy_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(
                self.model
            )

        total_micro_batches = args.num_ppo_epochs * math.ceil(
            len(input_ids) / max(args.train_batch_size_per_device, 1)
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

                base_pg_loss = self.masked_aggregator(
                    pg_loss_max, mb_response_masks, axis=1
                )
                if mb_row_weights is None:
                    base_pg_loss = (base_pg_loss * mb_loss_masks).mean()
                else:
                    # xDr.GRPO: per-candidate tempered aggregation weights
                    # (G * softmax(U/tau) per prompt group, detached). Only the
                    # pg-loss aggregation is reweighted; the entropy bonus and
                    # tokenwise KL penalty keep their uniform 1/G aggregation.
                    base_pg_loss = (
                        base_pg_loss * mb_loss_masks * mb_row_weights
                    ).mean()
                pg_loss = base_pg_loss
                infos["pg_loss"] = pg_loss.detach()
                loss = pg_loss
                policy_entropy_coef = float(
                    getattr(args, "policy_entropy_coef", 0.0) or 0.0
                )
                entropy_for_loss = None
                if policy_entropy_coef != 0.0:
                    token_entropy = self._chunked_entropy_from_logits(logits)
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
                    if entropy_for_loss is None:
                        token_entropy_for_log = self._chunked_entropy_from_logits(
                            logits
                        )
                    else:
                        token_entropy_for_log = token_entropy.detach()
                    if token_entropy_for_log.shape != mb_response_masks.shape:
                        token_entropy_for_log = token_entropy_for_log[
                            ..., : mb_response_masks.shape[-1]
                        ]
                    entropy = masked_mean(
                        token_entropy_for_log, mb_response_masks
                    )
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

    def _grpo_learning_step_with_progress(self, trajectory):
        args = self.args
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
        row_weights = None
        extra_infos: dict[str, torch.Tensor] = {}
        xdr_tau = float(getattr(args, "xdr_tau", math.inf))
        seed_alpha = float(getattr(args, "seed_entropy_alpha", 0.0) or 0.0)
        if math.isfinite(xdr_tau) and self.args.critic_type == "drgrpo":
            # xDr.GRPO: per-candidate Dr.GRPO utilities at the rollout policy
            # (ratio=1): U_i = A_i * T_i / T_max. Weights are computed once per
            # rollout batch and frozen for the update, like the advantages.
            row_weights = compute_xdr_row_weights(
                advantages,
                response_masks.sum(dim=1),
                num_samples=args.num_samples,
                tau=xdr_tau,
                t_max=int(args.generate_max_length),
                loss_masks=loss_masks,
            )
            extra_infos.update(
                xdr_group_diagnostics(
                    row_weights, final_rewards, num_samples=args.num_samples
                )
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
            answer_keys = [
                key for group in answer_keys_grouped for key in group
            ]
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
            extra_infos["seed_prompt_scale_mean"] = row_weights.view(
                -1, args.num_samples
            )[:, 0].mean().detach()
        return self._baseline_update_with_precomputed_advantages(
            input_ids=input_ids,
            att_mask=att_mask,
            prompt_id_lens=prompt_id_lens,
            loss_masks=loss_masks,
            response_masks=response_masks,
            logps=logps,
            ref_logps=ref_logps,
            advantages=advantages,
            final_rewards=final_rewards,
            returns=returns if self.args.critic_type == "ppo" else None,
            values=values if self.args.critic_type == "ppo" else None,
            policy_vocab_upper_bound=policy_vocab_upper_bound,
            row_weights=row_weights,
            extra_infos=extra_infos,
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
