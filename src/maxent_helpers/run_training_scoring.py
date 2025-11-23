# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scoring helpers extracted from the MaxEnt-GRPO training loop."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, List, Optional, Sequence, Tuple

from .run_helpers import (
    _prepare_labels_for_ce,
    require_torch,
    require_transformer_base_classes,
)
from .run_training_loss import SequenceScores
from .run_training_types import (
    BatchingSettings,
    GenerationSettings,
    LengthStats,
    PromptCacheEntry,
    ReferenceLogprobs,
    RewardComputation,
    RuntimeHandles,
    ScoreBatch,
)
from .scoring import _chunked_sequence_logprobs

torch = require_torch("training_scoring")
Tensor = torch.Tensor
PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes("training_scoring")


def _autocast_context(accelerator: Any, device: torch.device) -> Any:
    """Prefer Accelerator.autocast with a torch.autocast fallback."""
    accel_autocast = getattr(accelerator, "autocast", None)
    if callable(accel_autocast):
        return accel_autocast()
    device_type = getattr(device, "type", None) or "cuda"
    try:
        return torch.autocast(device_type=device_type)
    except (RuntimeError, ValueError):
        return nullcontext()


@dataclass
class CompletionTensors:
    """Completion token IDs and masks."""

    ids: Tensor
    mask: Tensor


@dataclass
class _SliceState:
    """Cached tensors and metadata required for batch slicing."""

    total_sequences: int
    slice_size: int
    completion_ids: Tensor
    completion_mask: Tensor
    prompt_entries: List[PromptCacheEntry]
    pad_token_id: int
    max_prompt_len: int

    @classmethod
    def from_score_batch(cls, score_batch: ScoreBatch) -> "_SliceState":
        """Build a state snapshot derived from a score batch."""
        total_sequences = score_batch.total_sequences
        if total_sequences == 0:
            slice_size = 0
        else:
            slice_size = score_batch.slice_size if score_batch.slice_size > 0 else total_sequences
        return cls(
            total_sequences=total_sequences,
            slice_size=slice_size,
            completion_ids=score_batch.completion_ids,
            completion_mask=score_batch.completion_attention_mask,
            prompt_entries=score_batch.prompt_entries,
            pad_token_id=score_batch.pad_token_id,
            max_prompt_len=max(1, score_batch.max_prompt_len),
        )


def _collect_prompt_entries(
    prompt_batch: List[str],
    batching_cfg: BatchingSettings,
) -> Optional[List[PromptCacheEntry]]:
    prompt_entries = [
        batching_cfg.prompt_length_cache_get(prompt) for prompt in prompt_batch
    ]
    if not prompt_entries:
        return None
    return prompt_entries


def _tokenize_completions(
    completion_batch: List[str],
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
) -> CompletionTensors:
    completion_enc = tokenizer(
        completion_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=generation_cfg.max_completion_len,
        add_special_tokens=False,
    )
    ids = completion_enc["input_ids"].long()
    mask = completion_enc["attention_mask"].long()
    return CompletionTensors(
        ids=ids,
        mask=mask,
    )


def _prepare_prompt_slice(
    prompt_slice: List[PromptCacheEntry],
    max_prompt_len: int,
    pad_token_id: int,
    ids_dtype: torch.dtype,
    mask_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, List[int]]:
    prompt_lengths = [
        min(entry.length, max_prompt_len) for entry in prompt_slice
    ]
    max_prompt_tokens = max(prompt_lengths) if prompt_lengths else 0
    batch_size = len(prompt_slice)
    if max_prompt_tokens > 0:
        prompt_ids = torch.full(
            (batch_size, max_prompt_tokens),
            pad_token_id,
            dtype=ids_dtype,
        )
        prompt_mask = torch.zeros(
            (batch_size, max_prompt_tokens),
            dtype=mask_dtype,
        )
        for row, (entry, length) in enumerate(zip(prompt_slice, prompt_lengths)):
            if length == 0:
                continue
            prompt_ids[row, :length] = torch.tensor(
                entry.input_ids[:length],
                dtype=ids_dtype,
            )
            prompt_mask[row, :length] = torch.tensor(
                entry.attention_mask[:length],
                dtype=mask_dtype,
            )
    else:
        prompt_ids = torch.empty((batch_size, 0), dtype=ids_dtype)
        prompt_mask = torch.empty((batch_size, 0), dtype=mask_dtype)
    return prompt_ids, prompt_mask, prompt_lengths


def iter_batch_slices(
    score_batch: ScoreBatch,
    device: torch.device,
):
    """Yield scoring slices for a batch, assembling prompt tensors on demand."""
    state = _SliceState.from_score_batch(score_batch)
    if state.total_sequences == 0 or state.slice_size <= 0:
        return
    for start in range(0, state.total_sequences, state.slice_size):
        end = min(start + state.slice_size, state.total_sequences)
        prompt_slice = state.prompt_entries[start:end]
        comp_ids_slice = state.completion_ids[start:end]
        comp_mask_slice = state.completion_mask[start:end]
        batch_size = len(prompt_slice)
        if batch_size == 0:
            continue
        prompt_ids, prompt_mask, prompt_lengths = _prepare_prompt_slice(
            prompt_slice,
            state.max_prompt_len,
            state.pad_token_id,
            comp_ids_slice.dtype,
            comp_mask_slice.dtype,
        )
        input_ids = torch.cat([prompt_ids, comp_ids_slice], dim=1)
        attention_mask = torch.cat([prompt_mask, comp_mask_slice], dim=1)
        labels = _prepare_labels_for_ce(
            input_ids.clone(),
            prompt_lengths,
        )
        yield (
            input_ids.to(device, non_blocking=True),
            attention_mask.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )


def build_score_batch(
    reward_comp: RewardComputation,
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
    batching_cfg: BatchingSettings,
) -> Optional[ScoreBatch]:
    """Tokenize prompt+completion pairs and prepare masks/labels."""
    prompt_batch = reward_comp.pairs.prompts
    completion_batch = reward_comp.pairs.completions
    total_sequences = len(prompt_batch)
    if total_sequences == 0:
        return None
    prompt_entries = _collect_prompt_entries(
        prompt_batch,
        batching_cfg,
    )
    if prompt_entries is None:
        return None
    completion_tensors = _tokenize_completions(
        completion_batch,
        tokenizer,
        generation_cfg,
    )
    slice_size = batching_cfg.score_slice if batching_cfg.score_slice > 0 else total_sequences
    slice_size = max(1, slice_size)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    return ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_tensors.ids,
        completion_attention_mask=completion_tensors.mask,
        pad_token_id=pad_token_id,
        max_prompt_len=generation_cfg.max_prompt_len,
        slice_size=slice_size,
        total_sequences=total_sequences,
    )


def reference_from_model(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run the frozen reference model to compute log-probs."""
    ref_model = runtime.get_ref_model()
    ref_logp_chunks: List[Tensor] = []
    ref_tok_chunks: List[Tensor] = []
    slice_iter = iter_batch_slices(score_batch, runtime.device)
    for slice_inputs, slice_mask, slice_labels in slice_iter:
        with torch.no_grad():
            ref_logp_slice, ref_tok_slice = _chunked_sequence_logprobs(
                ref_model,
                input_ids=slice_inputs,
                attention_mask=slice_mask,
                labels=slice_labels,
                chunk_size=batching_cfg.logprob_chunk_size,
                gather_full_params=True,
            )
        if ref_logp_slice.numel() == 0 or ref_tok_slice.numel() == 0:
            return None
        ref_logp_chunks.append(ref_logp_slice.detach().cpu())
        ref_tok_chunks.append(ref_tok_slice.detach().cpu())
    if not ref_logp_chunks:
        return None
    return (
        torch.cat(ref_logp_chunks, dim=0).to(runtime.device),
        torch.cat(ref_tok_chunks, dim=0).to(runtime.device),
    )


def finalize_reference_stats(
    ref_logp_sum: Tensor,
    ref_tok_counts: Tensor,
) -> ReferenceLogprobs:
    """Build a ReferenceLogprobs object and derived scalars."""
    ref_logp_sum_raw = ref_logp_sum.detach().clone()
    ref_tok_counts_tensor = ref_tok_counts.detach().clone()
    safe_tok_counts = ref_tok_counts_tensor.clamp(min=1).to(ref_logp_sum_raw.dtype)
    ref_logp_per_token = ref_logp_sum_raw / safe_tok_counts
    ref_logp_mean = float(ref_logp_sum_raw.mean().detach().cpu())
    avg_completion_tokens = float(ref_tok_counts_tensor.float().mean().detach().cpu())
    return ReferenceLogprobs(
        ref_logp_sum=ref_logp_per_token,
        ref_tok_counts=ref_tok_counts_tensor,
        ref_logp_sum_raw=ref_logp_sum_raw,
        ref_logp_mean=ref_logp_mean,
        avg_completion_tokens=avg_completion_tokens,
    )


def gather_reference_logprobs(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Optional[ReferenceLogprobs]:
    """Compute log-probabilities by running the frozen reference model."""
    tensors = reference_from_model(score_batch, runtime, batching_cfg)
    if tensors is None:
        return None
    return finalize_reference_stats(*tensors)


def reference_from_vllm_meta(
    flat_meta: Sequence[Optional[Any]],
    total_sequences: int,
    device: torch.device,
) -> Optional[ReferenceLogprobs]:
    """Convert flattened vLLM log-prob metadata into ReferenceLogprobs."""
    if not flat_meta:
        return None
    if len(flat_meta) != total_sequences:
        return None
    logp_vals: List[float] = []
    tok_counts: List[int] = []
    for entry in flat_meta:
        if entry is None:
            return None
        logprob_sum = getattr(entry, "logprob_sum", None)
        token_count = getattr(entry, "token_count", None)
        if logprob_sum is None and isinstance(entry, dict):
            logprob_sum = entry.get("logprob_sum")
        if token_count is None and isinstance(entry, dict):
            token_count = entry.get("token_count")
        if logprob_sum is None or token_count is None:
            return None
        logp_vals.append(float(logprob_sum))
        tok_counts.append(max(1, int(token_count)))
    ref_logp_sum = torch.tensor(logp_vals, dtype=torch.float32, device=device)
    ref_tok_counts = torch.tensor(tok_counts, dtype=torch.float32, device=device)
    return finalize_reference_stats(ref_logp_sum, ref_tok_counts)


def score_model_outputs(
    model: PreTrainedModel,
    score_batch: ScoreBatch,
    batching_cfg: BatchingSettings,
    runtime: RuntimeHandles,
) -> Optional[Tensor]:
    """Compute current model log-probs for the batch."""
    cur_logp_slices: List[Tensor] = []
    slice_iter = iter_batch_slices(score_batch, runtime.device)
    with _autocast_context(runtime.accelerator, runtime.device):
        for slice_inputs, slice_mask, slice_labels in slice_iter:
            cur_logp_slice, _ = _chunked_sequence_logprobs(
                model,
                input_ids=slice_inputs,
                attention_mask=slice_mask,
                labels=slice_labels,
                chunk_size=batching_cfg.logprob_chunk_size,
            )
            cur_logp_slices.append(cur_logp_slice)
    if not cur_logp_slices:
        return None
    return torch.cat(cur_logp_slices, dim=0)


def summarize_completion_lengths(
    ref_stats: ReferenceLogprobs,
    max_completion_len: int,
) -> Tuple[Tensor, LengthStats, float]:
    """Summarize completion lengths for metrics."""
    completion_lengths = ref_stats.ref_tok_counts.detach().float()
    num_completion_tokens = float(completion_lengths.sum().item())
    clipped_mask = completion_lengths >= float(max_completion_len)
    completion_lengths_cpu = completion_lengths.cpu()
    if completion_lengths.numel() > 0:
        min_length = float(completion_lengths_cpu.min().item())
        mean_length = float(completion_lengths_cpu.mean().item())
        max_length = float(completion_lengths_cpu.max().item())
        clipped_ratio = float(clipped_mask.float().mean().item())
    else:
        min_length = mean_length = max_length = 0.0
        clipped_ratio = 0.0
    terminated_lengths_cpu = completion_lengths_cpu[~clipped_mask.cpu()]
    if terminated_lengths_cpu.numel() > 0:
        min_terminated = float(terminated_lengths_cpu.min().item())
        mean_terminated = float(terminated_lengths_cpu.mean().item())
        max_terminated = float(terminated_lengths_cpu.max().item())
    else:
        min_terminated = mean_terminated = max_terminated = 0.0
    return completion_lengths, LengthStats(
        min_length=min_length,
        mean_length=mean_length,
        max_length=max_length,
        clipped_ratio=clipped_ratio,
        min_terminated=min_terminated,
        mean_terminated=mean_terminated,
        max_terminated=max_terminated,
    ), num_completion_tokens


def build_sequence_scores(
    cur_logp_sum: Tensor,
    ref_stats: ReferenceLogprobs,
) -> SequenceScores:
    """Return SequenceScores built from current and reference log-probs."""
    behavior_logp_sum = ref_stats.ref_logp_sum_raw.to(cur_logp_sum.device)
    log_ratio_train = cur_logp_sum - behavior_logp_sum
    denom_tok_tensor = ref_stats.ref_tok_counts.detach().clamp(min=1).to(cur_logp_sum.dtype)
    if denom_tok_tensor.numel() == 0:
        denom_tok_tensor = torch.ones_like(cur_logp_sum, dtype=cur_logp_sum.dtype)
    return SequenceScores(
        cur_logp_sum=cur_logp_sum,
        behavior_logp_sum=behavior_logp_sum,
        log_ratio_train=log_ratio_train,
        denom_tok_tensor=denom_tok_tensor,
    )
