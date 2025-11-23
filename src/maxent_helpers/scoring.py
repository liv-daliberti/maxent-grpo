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

"""Pytorch-based utilities for scoring batches of sequences."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import List, Tuple

from .zero_utils import _maybe_zero_gather_embedding, _maybe_zero_gather_params

try:
    torch = import_module("torch")
    F = import_module("torch.nn.functional")
    Tensor = torch.Tensor
except ModuleNotFoundError as import_error:  # pragma: no cover - import side effect
    raise RuntimeError(
        "Torch is required for maxent scoring. "
        "Install it with `pip install torch`."
    ) from import_error

try:
    PreTrainedModel = getattr(import_module("transformers"), "PreTrainedModel")
except ModuleNotFoundError as import_error:  # pragma: no cover - import side effect
    raise RuntimeError(
        "Transformers is required for maxent scoring. "
        "Install it with `pip install transformers`."
    ) from import_error

LOG = logging.getLogger(__name__)


def _warn_if_flattened_embedding(model: PreTrainedModel) -> None:
    """Log diagnostic info when embeddings are unexpectedly flattened."""
    embedding_fn = getattr(model, "get_input_embeddings", lambda: None)
    embedding = embedding_fn() if callable(embedding_fn) else None
    weight = getattr(embedding, "weight", None)
    if weight is None or getattr(weight, "ndim", 2) == 2:
        return
    LOG.error(
        "Reference embedding weight ndim=%s shape=%s ds_id=%s ds_status=%s",
        getattr(weight, "ndim", None),
        getattr(weight, "shape", None),
        getattr(weight, "ds_id", None),
        getattr(weight, "ds_status", None),
    )


def _sequence_logprobs(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute per-sequence log-prob sums and token counts."""
    # Some ZeRO/DeepSpeed setups temporarily flatten embedding weights,
    # which breaks ``torch.nn.functional.linear`` unless we gather them.
    with _maybe_zero_gather_embedding(model):
        _warn_if_flattened_embedding(model)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100).float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gather_labels = shift_labels.clamp(min=0).unsqueeze(-1)
    token_logp = log_probs.gather(dim=-1, index=gather_labels).squeeze(-1)
    token_logp = token_logp * mask
    seq_logp = token_logp.sum(dim=1)
    seq_counts = mask.sum(dim=1)
    return seq_logp, seq_counts


def _normalize_chunk_size(chunk_size: int, total_size: int) -> int:
    """Clamp the requested chunk size to a sane value."""
    if chunk_size <= 0:
        return total_size
    return min(max(1, chunk_size), total_size)


BatchTensors = Tuple[Tensor, Tensor, Tensor]


def _score_chunk_with_retry(
    model: PreTrainedModel,
    tensors: BatchTensors,
    start: int,
    max_chunk: int,
) -> Tuple[Tensor, Tensor, int]:
    """Score a slice of the batch, shrinking on OOM until it fits."""
    input_ids, attention_mask, labels = tensors
    total_size = input_ids.size(0)
    cur_chunk = min(max(1, max_chunk), total_size - start)
    while True:
        stop = start + cur_chunk
        try:
            return (
                *_sequence_logprobs(
                    model=model,
                    input_ids=input_ids[start:stop],
                    attention_mask=attention_mask[start:stop],
                    labels=labels[start:stop],
                ),
                cur_chunk,
            )
        except RuntimeError as runtime_error:
            if "out of memory" not in str(runtime_error).lower() or cur_chunk <= 1:
                raise
            next_chunk = max(1, cur_chunk // 2)
            LOG.warning(
                "OOM while scoring sequences (chunk=%d, total=%d);"
                " retrying with chunk=%d",
                cur_chunk,
                total_size,
                next_chunk,
            )
            torch.cuda.empty_cache()
            cur_chunk = next_chunk


def _chunked_sequence_logprobs(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    chunk_size: int,
    *,
    gather_full_params: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Evaluate _sequence_logprobs in mini-batches to bound activation memory."""
    total_size = input_ids.size(0)
    tensors: BatchTensors = (input_ids, attention_mask, labels)
    chunk_size = _normalize_chunk_size(chunk_size, total_size)
    logp_chunks: List[Tensor] = []
    count_chunks: List[Tensor] = []
    start = 0
    with _maybe_zero_gather_params(model, gather_full_params):
        while start < total_size:
            chunk_logp, chunk_counts, used_chunk = _score_chunk_with_retry(
                model=model,
                tensors=tensors,
                start=start,
                max_chunk=chunk_size,
            )
            logp_chunks.append(chunk_logp)
            count_chunks.append(chunk_counts)
            start += used_chunk
            chunk_size = used_chunk
    return torch.cat(logp_chunks, dim=0), torch.cat(count_chunks, dim=0)


def _iter_score_slices(
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    slice_size: int,
    device: torch.device,
):
    """Yield prompt+completion tensors already moved to the training device."""
    total = input_ids.size(0)
    if slice_size <= 0:
        slice_size = total
    slice_size = max(1, slice_size)
    for start in range(0, total, slice_size):
        end = min(start + slice_size, total)
        inputs = input_ids[start:end].to(device, non_blocking=True)
        mask = attention_mask[start:end].to(device, non_blocking=True)
        label_slice = labels[start:end].to(device, non_blocking=True)
        yield inputs, mask, label_slice


__all__ = [
    "_chunked_sequence_logprobs",
    "_iter_score_slices",
]
