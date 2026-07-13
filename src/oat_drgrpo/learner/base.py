"""Shared learner utilities for scoring, batching, and distributed helpers."""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist
from oat.utils.ops import entropy_from_logits

from ..scoring import (
    mask_invalid_logit_columns,
    resolve_token_id_upper_bound,
    sanitize_scoring_token_ids,
)
from ..listwise import gather_selected_logps_chunked


class ZeroMathLearnerBaseMixin:
    """Common utility methods used by baseline and Dr.X learner paths."""

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
            synchronized_token_counts = self._synchronized_backward_token_counts(
                att_mask
            )
            return configured_budget, synchronized_token_counts

        safe_chunk_size = max(int(configured_chunk_size), 1)
        synchronized_token_counts = self._synchronized_backward_token_counts(att_mask)
        max_synchronized_tokens = max(synchronized_token_counts, default=1)
        safety_budget = 4096
        if (
            safe_chunk_size > 1
            and max_synchronized_tokens * safe_chunk_size > safety_budget
        ):
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
        """Return the deepest safe module for hidden-state extraction."""

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
        self,
        model: torch.nn.Module,
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
                "Sanitized %d scoring token ids for %s outside upper_bound=%d "
                "using replacement_id=%s (min=%s max=%s)",
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
                "Masking %d tokenizer-inaccessible logit columns for %s "
                "(valid_vocab_size=%d, logits_width=%d).",
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
                    numerator = (
                        value.detach().to(
                            device=device,
                            dtype=torch.float32,
                        )
                        * safe_weight
                    )
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

    def _all_gather_same_shape_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        payload = tensor.contiguous()
        if not dist.is_available() or not dist.is_initialized():
            return payload.unsqueeze(0)
        gathered = [torch.zeros_like(payload) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, payload)
        return torch.stack(gathered, dim=0)

    def _broadcast_tensor_from_owner(
        self,
        tensor: torch.Tensor | None,
        *,
        src_rank: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            if tensor is None:
                raise ValueError(
                    "tensor must be provided when broadcasting without distributed initialization."
                )
            return tensor.to(device=device, dtype=dtype).contiguous()
        if dist.get_rank() == src_rank:
            if tensor is None:
                raise ValueError(
                    "Owner rank must provide a tensor for prompt-group broadcast."
                )
            payload = tensor.to(device=device, dtype=dtype).contiguous()
        else:
            payload = torch.empty(shape, device=device, dtype=dtype)
        dist.broadcast(payload, src=src_rank)
        return payload

    def _broadcast_global_prompt_permutation(
        self,
        *,
        local_prompt_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return torch.randperm(local_prompt_count, device=device, dtype=torch.long)
        world_size = dist.get_world_size()
        total_prompt_count = max(int(local_prompt_count), 0) * max(int(world_size), 1)
        if dist.get_rank() == 0:
            permutation = torch.randperm(
                total_prompt_count,
                device=device,
                dtype=torch.long,
            )
        else:
            permutation = torch.empty(
                (total_prompt_count,),
                device=device,
                dtype=torch.long,
            )
        dist.broadcast(permutation, src=0)
        return permutation

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
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            synced = torch.tensor(
                [safe_last_valid_token_pos],
                device=device,
                dtype=torch.int64,
            )
            dist.all_reduce(synced, op=dist.ReduceOp.MAX)
            safe_last_valid_token_pos = max(int(synced.item()), 1)
        return safe_last_valid_token_pos

    def _logprob_token_chunk_size(self) -> int:
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
            input_ids.shape[0],
            input_ids.shape[1] - 1,
            device=input_ids.device,
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
        batch_chunk_size = min(8, max(int(shifted_logits.size(0)), 1))
        batch_entropy_chunks = []
        for batch_start in range(0, int(shifted_logits.size(0)), batch_chunk_size):
            batch_stop = min(
                batch_start + batch_chunk_size,
                int(shifted_logits.size(0)),
            )
            token_entropy_chunks = []
            for start in range(0, int(shifted_logits.size(1)), token_chunk_size):
                stop = min(start + token_chunk_size, int(shifted_logits.size(1)))
                chunk_logits = shifted_logits[batch_start:batch_stop, start:stop, :]
                chunk_logits_fp32 = (
                    chunk_logits
                    if chunk_logits.dtype == torch.float32
                    else chunk_logits.float()
                )
                token_entropy_chunks.append(entropy_from_logits(chunk_logits_fp32))
            batch_entropy_chunks.append(torch.cat(token_entropy_chunks, dim=1))
        return torch.cat(batch_entropy_chunks, dim=0)
