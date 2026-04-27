"""Backward and gradient-probe helpers for Dr.X/listwise learning."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from ..listwise import (
    coerce_non_negative_float,
    flatten_prompt_major_tensor,
    iter_budgeted_row_chunks,
    iter_fixed_row_chunks,
)
from ..ppo_clip import compute_token_level_clip_loss


class ZeroMathDrxBackwardMixin:
    """Chunked backward and diagnostic gradient helpers for Dr.X/listwise paths."""

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
                cosine_denominator = (2.0 * token_grad_norm * listwise_grad_norm).clamp(
                    min=1e-12
                )
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
