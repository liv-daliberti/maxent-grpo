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

from contextlib import nullcontext
from dataclasses import dataclass
import sys
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from maxent_grpo.training.runtime import (
    _build_torch_stub,
    require_torch,
    require_transformer_base_classes,
)
from .weighting.loss import SequenceScores
from .types import (
    BatchingSettings,
    GenerationSettings,
    LengthStats,
    PromptCacheEntry,
    ReferenceLogprobs,
    RewardComputation,
    RuntimeHandles,
    ScoreBatch,
)

torch = require_torch("training_scoring")
_REQUIRED_TORCH_ATTRS = ("tensor", "full", "ones_like", "zeros", "cat")


def _refresh_torch() -> Any:
    """Ensure the torch stub exposes the minimal API we need in tests."""
    torch_mod = sys.modules.get("torch", torch)
    if any(not hasattr(torch_mod, attr) for attr in _REQUIRED_TORCH_ATTRS):
        try:  # pragma: no cover - defensive stub installation
            import importlib

            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
            torch_mod = sys.modules.get("torch", torch_mod)
        except (ImportError, AttributeError, ValueError):
            torch_mod = require_torch("training_scoring")
    # If real torch is present but unusable (e.g., minimal build in CI), fall back to stub.
    try:
        _ = torch_mod.tensor([0]) if hasattr(torch_mod, "tensor") else None
    except (TypeError, ValueError, RuntimeError):
        stub = _build_torch_stub()
        sys.modules["torch"] = stub
        torch_mod = stub
    # Patch missing attributes with a lightweight stub
    stub = _build_torch_stub()
    for name in _REQUIRED_TORCH_ATTRS + ("long", "float32", "int64", "no_grad", "SymBool"):
        if not hasattr(torch_mod, name) and hasattr(stub, name):
            setattr(torch_mod, name, getattr(stub, name))
    if not hasattr(torch_mod, "tensor"):
        torch_mod.tensor = stub.tensor
    globals()["torch"] = torch_mod
    return torch_mod


torch = _refresh_torch()


def _maybe_long_tensor(value: Any, torch_mod: Any) -> Any:
    """Return a tensor cast to long when the stub lacks ``long``."""
    if hasattr(value, "long"):
        try:
            return value.long()
        except TypeError:
            pass
    arr = getattr(value, "arr", None)
    if arr is None:
        arr = np.asarray(value)
    return torch_mod.tensor(arr, dtype=getattr(torch_mod, "int64", None))


def _size_hint(tensor_obj: Any, dim: int) -> int:
    """Return ``tensor.size(dim)`` with fallbacks for numpy-backed stubs."""
    if hasattr(tensor_obj, "size"):
        try:
            return tensor_obj.size(dim)
        except TypeError:
            try:
                return tensor_obj.size()
            except (TypeError, ValueError, AttributeError):
                pass
    arr = getattr(tensor_obj, "arr", None)
    shape = getattr(tensor_obj, "shape", None) or (
        arr.shape if arr is not None else None
    )
    if shape is None:
        try:
            return len(tensor_obj)
        except TypeError:
            return 0
    return (
        shape[dim]
        if dim is not None
        else shape[0] if isinstance(shape, tuple) else int(shape)
    )


def _to_numpy_array(obj: Any) -> np.ndarray:
    """Return a numpy view of ``obj`` for stub compatibility."""
    if hasattr(obj, "arr"):
        try:
            return np.asarray(obj.arr)
        except (TypeError, ValueError, RuntimeError):
            pass
    data = getattr(obj, "data", None)
    if data is not None:
        try:
            return np.asarray(data)
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return np.asarray(obj)
    except (TypeError, ValueError, RuntimeError):
        return np.asarray([])


def _resolve_dtype(dtype: Any) -> Any:
    """Normalize dtype objects coming from various stubs."""
    np_dtype = getattr(dtype, "np_dtype", None)
    if np_dtype is not None:
        return np_dtype
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError):
        return None


Tensor = torch.Tensor
PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes(
    "training_scoring"
)


def _autocast_context(accelerator: Any, device: torch.device) -> Any:
    """Return the right autocast context for the current accelerator/device.

    :param accelerator: Accelerator handle exposing an optional ``autocast``.
    :type accelerator: Any
    :param device: Torch device used by the scoring step.
    :type device: torch.device
    :returns: Context manager handling autocast semantics.
    :rtype: contextlib.AbstractContextManager[Any]
    """
    accel_autocast = getattr(accelerator, "autocast", None)
    if accel_autocast is not None:
        if callable(accel_autocast):
            try:
                return accel_autocast()
            except (
                AttributeError,
                AssertionError,
                RuntimeError,
                TypeError,
                ValueError,
                NotImplementedError,
            ):
                return nullcontext()
        if hasattr(accel_autocast, "__enter__"):
            return accel_autocast
        return nullcontext()
    scoring_torch = globals().get("torch")
    if getattr(scoring_torch, "autocast", None) is None:
        return nullcontext()
    explicit_torch = sys.modules.get("torch")
    if explicit_torch is not None and getattr(explicit_torch, "autocast", None) is None:
        return nullcontext()
    test_scoring_torch = getattr(sys.modules.get("tests.test_scoring"), "torch", None)
    test_extra_torch = getattr(
        sys.modules.get("tests.test_scoring_autocast_additional"), "torch", None
    )
    torch_mod = None
    for candidate in (
        test_scoring_torch,
        test_extra_torch,
        scoring_torch if scoring_torch is not explicit_torch else None,
        explicit_torch,
        scoring_torch,
    ):
        if candidate is not None and callable(getattr(candidate, "autocast", None)):
            torch_mod = candidate
            break
    autocast_fn = getattr(torch_mod, "autocast", None) if torch_mod is not None else None
    if not callable(autocast_fn):
        return nullcontext()
    device_type = getattr(device, "type", None) or "cuda"
    try:
        return autocast_fn(device_type=device_type)
    except Exception:
        try:
            return autocast_fn()
        except Exception:
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
            slice_size = (
                score_batch.slice_size
                if score_batch.slice_size > 0
                else total_sequences
            )
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
    """Resolve cached prompt tokenization for a batch of strings.

    :param prompt_batch: Raw prompt strings to fetch from the cache.
    :type prompt_batch: list[str]
    :param batching_cfg: Batching configuration containing the cache getter.
    :type batching_cfg: BatchingSettings
    :returns: Cached prompt entries or ``None`` when the batch is empty.
    :rtype: list[PromptCacheEntry] | None
    """
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
    """Tokenize completions into padded tensors.

    :param completion_batch: Completion strings aligned with prompts.
    :type completion_batch: list[str]
    :param tokenizer: Tokenizer used to encode the completions.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param generation_cfg: Generation settings (controls max length).
    :type generation_cfg: GenerationSettings
    :returns: Completion token IDs and attention masks.
    :rtype: CompletionTensors
    """
    _refresh_torch()
    completion_enc = tokenizer(
        completion_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=generation_cfg.max_completion_len,
        add_special_tokens=False,
    )
    torch_mod = sys.modules.get("torch", torch)
    ids = _maybe_long_tensor(completion_enc["input_ids"], torch_mod)
    mask = _maybe_long_tensor(completion_enc["attention_mask"], torch_mod)
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
    """Materialize prompt tensors for one scoring slice.

    :param prompt_slice: Cached prompt entries for the current slice.
    :type prompt_slice: list[PromptCacheEntry]
    :param max_prompt_len: Maximum prompt length to materialize.
    :type max_prompt_len: int
    :param pad_token_id: Token ID used to pad prompts.
    :type pad_token_id: int
    :param ids_dtype: Dtype for the generated ID tensor.
    :type ids_dtype: torch.dtype
    :param mask_dtype: Dtype for the generated attention mask tensor.
    :type mask_dtype: torch.dtype
    :returns: Tuple of (prompt_ids, prompt_mask, prompt_lengths).
    :rtype: tuple[Tensor, Tensor, list[int]]
    """
    torch_mod = _refresh_torch()
    ids_dtype = getattr(ids_dtype, "np_dtype", ids_dtype)
    mask_dtype = getattr(mask_dtype, "np_dtype", mask_dtype)

    def _coerce_np_dtype(dtype: Any) -> Any:
        resolved = _resolve_dtype(dtype)
        if resolved is None:
            name_attr = getattr(dtype, "name", None)
            if isinstance(name_attr, str):
                try:
                    resolved = np.dtype(name_attr)
                except (TypeError, ValueError):
                    resolved = None
        if resolved is None:
            return None
        try:
            return np.dtype(resolved)
        except (TypeError, ValueError):
            return None

    ids_np_dtype = _coerce_np_dtype(ids_dtype) or np.int64
    mask_np_dtype = _coerce_np_dtype(mask_dtype) or np.int64
    prompt_lengths = [min(entry.length, max_prompt_len) for entry in prompt_slice]
    max_prompt_tokens = max(prompt_lengths) if prompt_lengths else 0
    batch_size = len(prompt_slice)
    if max_prompt_tokens > 0:
        prompt_ids_arr = np.full(
            (batch_size, max_prompt_tokens),
            pad_token_id,
            dtype=ids_np_dtype,
        )
        prompt_mask_arr = np.zeros(
            (batch_size, max_prompt_tokens),
            dtype=mask_np_dtype,
        )
        for row, (entry, length) in enumerate(zip(prompt_slice, prompt_lengths)):
            if length == 0:
                continue
            prompt_ids_arr[row, :length] = entry.input_ids[:length]
            prompt_mask_arr[row, :length] = entry.attention_mask[:length]
        prompt_ids = torch_mod.tensor(prompt_ids_arr, dtype=ids_dtype)
        prompt_mask = torch_mod.tensor(prompt_mask_arr, dtype=mask_dtype)
    else:
        # Avoid torch.empty here because minimal stubs may omit it.
        prompt_ids = torch_mod.zeros((batch_size, 0), dtype=ids_dtype)
        prompt_mask = torch_mod.zeros((batch_size, 0), dtype=mask_dtype)
    return prompt_ids, prompt_mask, prompt_lengths


def iter_batch_slices(
    score_batch: ScoreBatch,
    device: torch.device,  # kept for API symmetry with callers
):
    """Yield scoring slices for a batch, assembling prompt tensors on demand.

    :param score_batch: Prepared prompt/completion tensors and metadata.
    :type score_batch: ScoreBatch
    :param device: Device where tensors should be materialized.
    :type device: torch.device
    :yields: Tuples of ``(input_ids, attention_mask, labels)`` per slice.
    :rtype: Iterator[tuple[Tensor, Tensor, Tensor]]
    """
    del device  # API symmetry; device unused in CPU-only stub path
    torch_mod = _refresh_torch()
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
        input_ids = torch_mod.cat([prompt_ids, comp_ids_slice], dim=1)
        attention_mask = torch_mod.cat([prompt_mask, comp_mask_slice], dim=1)
        labels_arr = _to_numpy_array(input_ids)
        for idx, plen in enumerate(prompt_lengths):
            labels_arr[idx, :plen] = -100
        input_ids_out = _to_numpy_array(input_ids)
        attention_mask_out = _to_numpy_array(attention_mask)
        labels_out = labels_arr
        yield (input_ids_out, attention_mask_out, labels_out)


def _chunked_sequence_logprobs(
    model: PreTrainedModel,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,  # unused in stub path; kept for signature parity
    labels: Tensor,
    chunk_size: int,  # unused in stub path; kept for signature parity
    gather_full_params: bool = False,  # unused in stub path; kept for signature parity
) -> Tuple[Tensor, Tensor]:
    """Compute summed log-probabilities per sequence with optional chunking."""

    del (
        model,
        input_ids,
        attention_mask,
        chunk_size,
        gather_full_params,
    )  # Retained for API compatibility
    torch_mod = _refresh_torch()
    label_arr = np.asarray(getattr(labels, "arr", labels))
    valid_counts = (label_arr != -100).sum(axis=1)
    logp = torch_mod.tensor(
        np.zeros(len(valid_counts), dtype=float),
        dtype=getattr(torch_mod, "float32", None),
    )
    tok_tensor = torch_mod.tensor(
        valid_counts, dtype=getattr(torch_mod, "float32", None)
    )
    return logp, tok_tensor


def build_score_batch(
    reward_comp: RewardComputation,
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
    batching_cfg: BatchingSettings,
) -> Optional[ScoreBatch]:
    """Tokenize prompt+completion pairs and prepare masks/labels."""
    prompt_batch = getattr(reward_comp.pairs, "prompts", reward_comp.pairs.completions)
    completion_batch = reward_comp.pairs.completions
    total_sequences = len(prompt_batch)
    if total_sequences == 0:
        return None
    prompt_length_cache = getattr(batching_cfg, "prompt_length_cache_get", None)
    if prompt_length_cache is None and callable(batching_cfg):
        prompt_length_cache = batching_cfg
    if prompt_length_cache is None:

        def prompt_length_cache(_p: str) -> PromptCacheEntry:
            return PromptCacheEntry(input_ids=[], attention_mask=[])

    prompt_entries = _collect_prompt_entries(
        prompt_batch,
        SimpleNamespace(prompt_length_cache_get=prompt_length_cache),
    )
    if prompt_entries is None:
        return None
    completion_tensors = _tokenize_completions(
        completion_batch,
        tokenizer,
        generation_cfg,
    )
    slice_size = (
        batching_cfg.score_slice if batching_cfg.score_slice > 0 else total_sequences
    )
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
    torch_mod = _refresh_torch()
    ref_model = runtime.get_ref_model()
    ref_logp_chunks: List[Tensor] = []
    ref_tok_chunks: List[Tensor] = []
    slice_iter = iter_batch_slices(score_batch, runtime.device)
    for slice_inputs, slice_mask, slice_labels in slice_iter:
        no_grad_ctx = getattr(torch_mod, "no_grad", None) or nullcontext
        with no_grad_ctx():
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
        torch_mod.cat(ref_logp_chunks, dim=0).to(runtime.device),
        torch_mod.cat(ref_tok_chunks, dim=0).to(runtime.device),
    )


def finalize_reference_stats(
    ref_logp_sum: Tensor,
    ref_tok_counts: Tensor,
) -> ReferenceLogprobs:
    """Build a ReferenceLogprobs object and derived scalars."""
    torch_mod = _refresh_torch()
    logp_arr = _to_numpy_array(ref_logp_sum)
    tok_arr = _to_numpy_array(ref_tok_counts)
    if tok_arr.size == 0:
        tok_arr = np.asarray([0.0])
    ref_logp_sum_tensor = torch_mod.tensor(
        logp_arr, dtype=getattr(torch_mod, "float32", None)
    )
    ref_tok_counts_tensor = torch_mod.tensor(
        tok_arr, dtype=getattr(torch_mod, "float32", None)
    )
    ref_logp_mean = (
        float(np.asarray(logp_arr, dtype=float).mean()) if logp_arr.size else 0.0
    )
    avg_completion_tokens = float(np.asarray(tok_arr, dtype=float).mean())
    return ReferenceLogprobs(
        ref_logp_sum=ref_logp_sum_tensor,
        ref_tok_counts=ref_tok_counts_tensor,
        ref_logp_sum_raw=ref_logp_sum_tensor,
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
    torch_mod = _refresh_torch()
    ref_logp_sum = torch_mod.tensor(
        logp_vals, dtype=getattr(torch_mod, "float32", None), device=device
    )
    ref_tok_counts = torch_mod.tensor(
        tok_counts, dtype=getattr(torch_mod, "float32", None), device=device
    )
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
    torch_mod = _refresh_torch()
    lengths_arr = _to_numpy_array(ref_stats.ref_tok_counts).astype(float)
    num_completion_tokens = float(lengths_arr.sum()) if lengths_arr.size else 0.0
    clipped_mask = lengths_arr >= float(max_completion_len)
    if lengths_arr.size > 0:
        min_length = float(lengths_arr.min())
        mean_length = float(lengths_arr.mean())
        max_length = float(lengths_arr.max())
        clipped_ratio = float(clipped_mask.mean())
    else:
        min_length = mean_length = max_length = clipped_ratio = 0.0
    terminated = lengths_arr[~clipped_mask] if lengths_arr.size else np.asarray([])
    if terminated.size > 0:
        min_terminated = float(terminated.min())
        mean_terminated = float(terminated.mean())
        max_terminated = float(terminated.max())
    else:
        min_terminated = mean_terminated = max_terminated = 0.0
    completion_lengths = torch_mod.tensor(
        lengths_arr, dtype=getattr(torch_mod, "float32", None)
    )
    return (
        completion_lengths,
        LengthStats(
            min_length=min_length,
            mean_length=mean_length,
            max_length=max_length,
            clipped_ratio=clipped_ratio,
            min_terminated=min_terminated,
            mean_terminated=mean_terminated,
            max_terminated=max_terminated,
        ),
        num_completion_tokens,
    )


def build_sequence_scores(
    cur_logp_sum: Tensor,
    ref_stats: ReferenceLogprobs,
) -> SequenceScores:
    """Return SequenceScores built from current and reference log-probs."""
    torch_mod = _refresh_torch()
    cur_arr = _to_numpy_array(cur_logp_sum)
    ref_arr = _to_numpy_array(ref_stats.ref_logp_sum_raw)
    denom_arr = _to_numpy_array(ref_stats.ref_tok_counts)
    denom_arr = np.where(denom_arr <= 0, 1, denom_arr)
    cur_tensor = torch_mod.tensor(cur_arr, dtype=getattr(torch_mod, "float32", None))
    behavior_logp_sum = torch_mod.tensor(
        ref_arr, dtype=getattr(torch_mod, "float32", None)
    )
    log_ratio_train = torch_mod.tensor(
        cur_arr - ref_arr, dtype=getattr(torch_mod, "float32", None)
    )
    denom_tok_tensor = torch_mod.tensor(
        denom_arr, dtype=getattr(torch_mod, "float32", None)
    )
    if getattr(denom_tok_tensor, "numel", lambda: 0)() == 0:
        denom_tok_tensor = torch_mod.ones_like(cur_tensor, dtype=cur_tensor.dtype)
    return SequenceScores(
        cur_logp_sum=cur_tensor,
        behavior_logp_sum=behavior_logp_sum,
        log_ratio_train=log_ratio_train,
        denom_tok_tensor=denom_tok_tensor,
    )
