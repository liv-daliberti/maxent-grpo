"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Unit tests for training.scoring utilities.
"""

from __future__ import annotations

from contextlib import nullcontext
import builtins
from types import SimpleNamespace
from typing import List

import pytest
import numpy as np
import sys
import types

from maxent_grpo.training.scoring import _PadTokenGuard


# Provide a richer torch stub when the real package is unavailable or broken.
try:  # pragma: no cover - environment dependent
    import torch as _torch_mod  # type: ignore
except Exception:
    _torch_mod = None

_needs_stub = _torch_mod is None or not hasattr(
    _torch_mod, "tensor"
)  # Prefer real torch when available.
if _needs_stub:

    class _Device:
        def __init__(self, device="cpu"):
            self.type = str(device)

        def __repr__(self):
            return f"device('{self.type}')"

    class _Tensor:
        __array_priority__ = 100.0

        def __init__(self, data, dtype=None):
            self.arr = np.array(data, dtype=dtype)

        @property
        def shape(self):
            return self.arr.shape

        @property
        def dtype(self):
            return self.arr.dtype

        @property
        def device(self):
            return _Device("cpu")

        def long(self):
            return _Tensor(self.arr.astype(np.int64))

        def float(self):
            return _Tensor(self.arr.astype(np.float32))

        def size(self, dim=None):
            return self.arr.size if dim is None else self.arr.shape[dim]

        def numel(self):
            return int(self.arr.size)

        def item(self):
            return self.arr.item()

        def __float__(self):
            return float(self.arr.astype(np.float64).item())

        def to(self, device=None, dtype=None, non_blocking=False):
            target_dtype = dtype
            if (
                target_dtype is None
                and device is not None
                and not isinstance(device, (_Device, str))
            ):
                target_dtype = device
            return _Tensor(
                self.arr.astype(target_dtype) if target_dtype is not None else self.arr
            )

        def cpu(self):
            return self

        def detach(self):
            return self

        def clone(self):
            return _Tensor(self.arr.copy())

        def sum(self, dim=None):
            arr = self.arr.astype(np.int64) if self.arr.dtype == bool else self.arr
            return _Tensor(np.sum(arr, axis=dim))

        def mean(self, dim=None):
            return _Tensor(np.mean(self.arr, axis=dim))

        def min(self):
            return _Tensor(np.min(self.arr))

        def max(self):
            return _Tensor(np.max(self.arr))

        def clamp(self, min=None, max=None):
            lo = min if min is not None else None
            hi = max if max is not None else None
            arr = self.arr
            if lo is None and hi is None:
                return _Tensor(arr)
            return _Tensor(
                np.clip(
                    arr,
                    lo if lo is not None else arr.min(),
                    hi if hi is not None else arr.max(),
                )
            )

        def unsqueeze(self, dim):
            return _Tensor(np.expand_dims(self.arr, axis=dim))

        def squeeze(self, dim=None):
            return _Tensor(
                np.squeeze(self.arr, axis=dim)
                if dim is not None
                else np.squeeze(self.arr)
            )

        def reshape(self, *shape):
            return _Tensor(self.arr.reshape(*shape))

        def gather(self, dim, index):
            idx = index.arr if isinstance(index, _Tensor) else index
            result = np.take_along_axis(self.arr, idx, axis=dim)
            return _Tensor(result)

        def ne(self, other):
            return _Tensor(
                self.arr != (other.arr if isinstance(other, _Tensor) else other)
            )

        def eq(self, other):
            return _Tensor(
                self.arr == (other.arr if isinstance(other, _Tensor) else other)
            )

        def ge(self, other):
            return _Tensor(
                self.arr >= (other.arr if isinstance(other, _Tensor) else other)
            )

        def __eq__(self, other):
            return self.eq(other)

        def __ne__(self, other):
            return self.ne(other)

        def __ge__(self, other):
            return self.ge(other)

        def __iter__(self):
            for item in self.arr:
                yield item

        def __array__(self, dtype=None):
            return np.array(self.arr, dtype=dtype)

        def tolist(self):
            return self.arr.tolist()

        def __len__(self):
            return len(self.arr)

        def __getitem__(self, key):
            return _Tensor(self.arr[key])

        def __setitem__(self, key, value):
            self.arr[key] = value.arr if isinstance(value, _Tensor) else value

        def __add__(self, other):
            return _Tensor(
                self.arr + (other.arr if isinstance(other, _Tensor) else other)
            )

        def __sub__(self, other):
            return _Tensor(
                self.arr - (other.arr if isinstance(other, _Tensor) else other)
            )

        def masked_fill(self, mask, value):
            mask_arr = mask.arr if isinstance(mask, _Tensor) else np.asarray(mask)
            filled = np.where(mask_arr, value, self.arr)
            return _Tensor(filled)

        def __mul__(self, other):
            return _Tensor(
                self.arr * (other.arr if isinstance(other, _Tensor) else other)
            )

        def __truediv__(self, other):
            return _Tensor(
                self.arr / (other.arr if isinstance(other, _Tensor) else other)
            )

        def __gt__(self, other):
            return _Tensor(
                self.arr > (other.arr if isinstance(other, _Tensor) else other)
            )

        def __lt__(self, other):
            return _Tensor(
                self.arr < (other.arr if isinstance(other, _Tensor) else other)
            )

        def __neg__(self):
            return _Tensor(-self.arr)

    def _tensor(data, dtype=None, device=None):
        return _Tensor(data, dtype=dtype)

    def _ones_like(t, dtype=None):
        return _Tensor(
            np.ones_like(t.arr, dtype=dtype if dtype is not None else t.arr.dtype)
        )

    def _zeros_like(t, dtype=None):
        return _Tensor(
            np.zeros_like(t.arr, dtype=dtype if dtype is not None else t.arr.dtype)
        )

    def _zeros(shape, dtype=None):
        return _Tensor(np.zeros(shape, dtype=dtype))

    def _ones(shape, dtype=None):
        return _Tensor(np.ones(shape, dtype=dtype))

    def _full(shape, fill_value, dtype=None):
        return _Tensor(np.full(shape, fill_value, dtype=dtype))

    def _empty(shape, dtype=None):
        return _Tensor(np.empty(shape, dtype=dtype))

    def _arange(end, dtype=None):
        return _Tensor(np.arange(end, dtype=dtype))

    def _cat(tensors, dim=0):
        arrays = [t.arr if isinstance(t, _Tensor) else np.array(t) for t in tensors]
        return _Tensor(np.concatenate(arrays, axis=dim))

    def _all(tensor):
        arr = tensor.arr if isinstance(tensor, _Tensor) else np.array(tensor)
        return bool(np.all(arr))

    def _no_grad():
        return nullcontext()

    def _log_softmax(logits, dim=-1):
        arr = logits.arr if isinstance(logits, _Tensor) else np.array(logits)
        max_val = np.max(arr, axis=dim, keepdims=True)
        exps = np.exp(arr - max_val)
        logsum = np.log(np.sum(exps, axis=dim, keepdims=True))
        return _Tensor(arr - max_val - logsum)

    torch = types.SimpleNamespace(
        Tensor=_Tensor,
        tensor=_tensor,
        ones_like=_ones_like,
        zeros_like=_zeros_like,
        zeros=_zeros,
        ones=_ones,
        full=_full,
        empty=_empty,
        arange=_arange,
        cat=_cat,
        all=_all,
        no_grad=_no_grad,
        float32=np.float32,
        float64=np.float64,
        long=np.int64,
        int64=np.int64,
        dtype=np.dtype,
        device=lambda x="cpu": _Device(x),
        autograd=types.SimpleNamespace(no_grad=lambda: nullcontext()),
        autocast=lambda *a, **k: nullcontext(),
        nn=types.SimpleNamespace(
            functional=types.SimpleNamespace(log_softmax=_log_softmax)
        ),
    )
    torch.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(DataLoader=object, Sampler=object)
    )
    torch.no_grad = lambda: nullcontext()
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional
    sys.modules["torch.utils"] = torch.utils
    sys.modules["torch.utils.data"] = torch.utils.data
    acc_mod = types.SimpleNamespace(Accelerator=type("Accelerator", (), {}))
    sys.modules["accelerate"] = acc_mod
    sys.modules.setdefault(
        "accelerate.utils",
        types.SimpleNamespace(DeepSpeedPlugin=type("DeepSpeedPlugin", (), {})),
    )
else:
    import torch  # type: ignore

    torch = sys.modules["torch"]

# Ensure a transformers stub is present with required attributes.
if "transformers" not in sys.modules:
    transformers_stub = types.SimpleNamespace(
        PreTrainedModel=type("PreTrainedModel", (), {}),
        PreTrainedTokenizer=type("PreTrainedTokenizer", (), {}),
        PreTrainedTokenizerBase=type("PreTrainedTokenizerBase", (), {}),
    )
    sys.modules["transformers"] = transformers_stub
else:
    transformers_mod = sys.modules["transformers"]
    transformers_mod.PreTrainedModel = getattr(
        transformers_mod, "PreTrainedModel", type("PreTrainedModel", (), {})
    )
    transformers_mod.PreTrainedTokenizer = getattr(
        transformers_mod, "PreTrainedTokenizer", type("PreTrainedTokenizer", (), {})
    )
    transformers_mod.PreTrainedTokenizerBase = getattr(
        transformers_mod,
        "PreTrainedTokenizerBase",
        type("PreTrainedTokenizerBase", (), {}),
    )

# Ensure prior imports using different torch stubs don't leak into this module.
for _mod in [
    "training.scoring",
    "training.types",
    "training.types.runtime",
    "training.weighting",
    "training.weighting.loss",
]:
    sys.modules.pop(_mod, None)

# Seed a lightweight wandb stub before importing training modules to placate accelerate checks.
sys.modules.setdefault("wandb", SimpleNamespace(__spec__=SimpleNamespace()))

import maxent_grpo.training.scoring as scoring  # noqa: E402
from maxent_grpo.training.scoring import (  # noqa: E402
    _autocast_context,
    _chunked_sequence_logprobs,
    _collect_prompt_entries,
    build_score_batch,
    build_sequence_scores,
    finalize_reference_stats,
    iter_batch_slices,
    gather_reference_logprobs,
    reference_from_model,
    reference_from_vllm_meta,
    score_model_outputs,
    summarize_completion_lengths,
)
from maxent_grpo.training.types import (  # noqa: E402
    BatchingSettings,
    GenerationSettings,
    LengthStats,
    PromptCacheEntry,
    ReferenceLogprobs,
    RewardComputation,
    ScoreBatch,
)
from maxent_grpo.training.run_helpers import VLLMClientConfig  # noqa: E402


class _TinyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.called = False

    def __call__(self, texts: List[str], **_kwargs):
        self.called = True
        # encode each string as incremental integers
        enc = [[i + 1 for i, _ in enumerate(text)] for text in texts]
        max_len = max(len(row) for row in enc)
        padded = [row + [0] * (max_len - len(row)) for row in enc]
        mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in enc]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _LinearModel:
    """Deterministic model returning ascending logits."""

    def __call__(self, input_ids, attention_mask, labels):
        vocab = 6
        bsz, seqlen = input_ids.shape
        logits = torch.arange(bsz * seqlen * vocab, dtype=torch.float32).reshape(
            bsz, seqlen, vocab
        )
        return SimpleNamespace(logits=logits)


def _dummy_prompt_cache(prompt: str) -> PromptCacheEntry:
    # encode prompt length as len(prompt)
    ids = list(range(1, len(prompt) + 1))
    return PromptCacheEntry(input_ids=ids, attention_mask=[1] * len(ids))


def _score_batch():
    prompts = ["hi", "ok"]
    completions = ["A", "BC"]
    reward_comp = RewardComputation(
        total_utils=[],
        per_reward_values={},
        advantage=SimpleNamespace(samples=[]),
        pairs=SimpleNamespace(prompts=prompts, completions=completions),
        q_distribution=SimpleNamespace(grouped=[], samples=[]),
        moments=SimpleNamespace(mean=0.0, std=0.0),
    )
    tokenizer = _TinyTokenizer()
    gen_cfg = GenerationSettings(
        max_prompt_len=4,
        max_completion_len=4,
        gen_temperature=1.0,
        gen_top_p=1.0,
        use_vllm=False,
        vllm=VLLMClientConfig(
            url="http://localhost",
            rounds_cfg=1,
            retry_sleep=0.0,
            backfill_local=False,
            request_logprobs=False,
        ),
    )
    batching_cfg = BatchingSettings(
        logprob_chunk_size=0, score_slice=0, prompt_length_cache_get=_dummy_prompt_cache
    )
    sb = build_score_batch(reward_comp, tokenizer, gen_cfg, batching_cfg)
    assert sb is not None
    return sb, tokenizer, gen_cfg, batching_cfg


def test_collect_prompt_entries_empty_batch_returns_none():
    dummy_cfg = SimpleNamespace(prompt_length_cache_get=_dummy_prompt_cache)
    assert _collect_prompt_entries([], dummy_cfg) is None


def test_prepare_prompt_slice_and_iter_batch_slices_shapes():
    sb, _, _, _ = _score_batch()
    device = torch.device("cpu")
    slices = list(iter_batch_slices(sb, device))
    # One slice with concatenated prompt+completion tokens
    assert len(slices) == 1
    input_ids, attention_mask, labels = slices[0]
    shape_ids = getattr(input_ids, "shape", None) or (len(input_ids),)
    shape_mask = getattr(attention_mask, "shape", None) or (len(attention_mask),)
    shape_labels = getattr(labels, "shape", None) or (len(labels),)
    assert shape_ids == shape_mask == shape_labels
    # Labels should mask prompt tokens with -100
    labels_arr = np.asarray(labels[:, :2])
    assert np.all(labels_arr == -100)
    completion_region = np.asarray(labels[:, 2:])
    assert (completion_region >= 0).any()


def test_iter_batch_slices_applies_tail_limit():
    sb_full, _, _, _ = _score_batch()
    device = torch.device("cpu")
    base_len = list(iter_batch_slices(sb_full, device))[0][0].shape[1]

    sb_tail, _, _, _ = _score_batch()
    sb_tail.score_tail_tokens = 2
    tail_slice = list(iter_batch_slices(sb_tail, device))[0]
    tail_len = tail_slice[0].shape[1]
    assert tail_len == min(2, base_len)
    assert tail_len <= base_len


def test_iter_batch_slices_tail_excludes_global_padding():
    prompt_entries = [PromptCacheEntry(input_ids=[10, 11], attention_mask=[1, 1])]
    completion_ids = torch.tensor([[101, 102, 0, 0, 0]], dtype=torch.long)
    completion_mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.long)
    score_batch = ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_ids,
        completion_attention_mask=completion_mask,
        pad_token_id=0,
        max_prompt_len=4,
        slice_size=2,
        total_sequences=1,
        score_tail_tokens=2,
    )
    device = torch.device("cpu")
    input_ids, _attn, labels = next(iter_batch_slices(score_batch, device))
    assert input_ids.shape[1] == 2
    assert input_ids[0].tolist() == [101, 102]
    assert labels[0].tolist() == [101, 102]


def test_iter_batch_slices_tail_truncates_completion_span():
    prompt_tokens = list(range(100))
    prompt_entries = [
        PromptCacheEntry(input_ids=prompt_tokens, attention_mask=[1] * len(prompt_tokens))
    ]
    completion_ids = torch.tensor([[401, 402, 403]], dtype=torch.long)
    completion_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    # Tail window smaller than completion span would previously drop completion columns.
    score_batch = ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_ids,
        completion_attention_mask=completion_mask,
        pad_token_id=0,
        max_prompt_len=len(prompt_tokens),
        slice_size=1,
        total_sequences=1,
        score_tail_tokens=2,
    )
    device = torch.device("cpu")
    input_ids, _attn, labels = next(iter_batch_slices(score_batch, device))
    assert input_ids.shape[1] == 2
    assert labels.tolist() == [[402, 403]]


def test_iter_batch_slices_tail_keeps_single_token_completion():
    prompt_tokens = list(range(256))
    prompt_entries = [
        PromptCacheEntry(input_ids=prompt_tokens, attention_mask=[1] * len(prompt_tokens))
    ]
    completion_ids = torch.tensor([[915]], dtype=torch.long)
    completion_mask = torch.tensor([[1]], dtype=torch.long)
    score_batch = ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_ids,
        completion_attention_mask=completion_mask,
        pad_token_id=0,
        max_prompt_len=len(prompt_tokens),
        slice_size=1,
        total_sequences=1,
        score_tail_tokens=64,
    )
    device = torch.device("cpu")
    _ids, _attn, labels = next(iter_batch_slices(score_batch, device))
    assert (labels != -100).any()
    assert 915 in labels.tolist()[0]


def test_iter_batch_slices_masks_completion_padding():
    prompt_entries = [
        PromptCacheEntry(input_ids=[1], attention_mask=[1]),
        PromptCacheEntry(input_ids=[2], attention_mask=[1]),
    ]
    completion_ids = torch.tensor(
        [[201, 202, 0, 0], [301, 302, 303, 304]], dtype=torch.long
    )
    completion_mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long
    )
    score_batch = ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_ids,
        completion_attention_mask=completion_mask,
        pad_token_id=0,
        max_prompt_len=2,
        slice_size=2,
        total_sequences=2,
    )
    device = torch.device("cpu")
    _ids, _attn, labels = next(iter_batch_slices(score_batch, device))
    prompt_width = 1
    assert labels[0, prompt_width : prompt_width + 2].tolist() == [201, 202]
    assert labels[0, prompt_width + 2 :].tolist() == [-100, -100]
    assert labels[1, prompt_width : prompt_width + 4].tolist() == [301, 302, 303, 304]


def test_iter_batch_slices_materializes_tensors():
    sb, _, _, _ = _score_batch()
    device = torch.device("cpu")
    input_ids, attention_mask, labels = next(iter_batch_slices(sb, device))
    # Ensure numpy arrays are converted back to torch tensors before model call.
    tensor_type = getattr(torch, "Tensor", tuple())

    def _looks_like_tensor(obj):
        is_tensor_fn = getattr(torch, "is_tensor", None)
        if callable(is_tensor_fn):
            try:
                if is_tensor_fn(obj):
                    return True
            except Exception:
                pass
        if tensor_type and isinstance(obj, tensor_type):
            return True
        return hasattr(obj, "arr") or hasattr(obj, "shape")

    assert _looks_like_tensor(input_ids)
    assert _looks_like_tensor(attention_mask)
    assert _looks_like_tensor(labels)
    dev_ids = getattr(input_ids, "device", None)
    dev_mask = getattr(attention_mask, "device", None)
    dev_labels = getattr(labels, "device", None)
    dev_ids_type = getattr(dev_ids, "type", dev_ids)
    dev_mask_type = getattr(dev_mask, "type", dev_mask)
    dev_labels_type = getattr(dev_labels, "type", dev_labels)
    assert dev_ids_type == dev_mask_type == dev_labels_type
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_reference_scoring_preserves_slice_order(monkeypatch):
    """Check that sliced ref scoring concatenates per-sequence results in order."""

    prompt_entries = [
        PromptCacheEntry(input_ids=[10], attention_mask=[1]),
        PromptCacheEntry(input_ids=[20, 21], attention_mask=[1, 1]),
        PromptCacheEntry(input_ids=[30], attention_mask=[1]),
    ]
    completion_ids = torch.tensor(
        [[101, 102], [201, 202], [301, 302]], dtype=torch.long
    )
    completion_mask = torch.ones_like(completion_ids, dtype=torch.long)
    score_batch = ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_ids,
        completion_attention_mask=completion_mask,
        pad_token_id=0,
        max_prompt_len=4,
        slice_size=2,  # force >1 slice for 3 sequences
        total_sequences=3,
    )
    batching_cfg = BatchingSettings(
        logprob_chunk_size=0,
        score_slice=2,
        prompt_length_cache_get=_dummy_prompt_cache,
        slice_prefetch=2,
    )
    runtime = SimpleNamespace(
        get_ref_model=lambda: SimpleNamespace(), device=torch.device("cpu")
    )

    def _fake_chunked_sequence_logprobs(
        model,
        *,
        input_ids,
        attention_mask,
        labels,
        chunk_size,
        gather_full_params=False,
        return_hidden=False,
        pooling="mean",
    ):
        _ = (
            model,
            attention_mask,
            chunk_size,
            gather_full_params,
            return_hidden,
            pooling,
        )
        valid_mask = labels.ne(-100)
        seq_sum = (labels * valid_mask).sum(dim=1)
        tok_counts = valid_mask.sum(dim=1)
        return seq_sum.float(), tok_counts.float(), None

    monkeypatch.setattr(
        scoring, "_chunked_sequence_logprobs", _fake_chunked_sequence_logprobs
    )

    ref = gather_reference_logprobs(score_batch, runtime, batching_cfg)
    assert ref is not None
    assert ref.ref_logp_sum.tolist() == [203.0, 403.0, 603.0]
    assert ref.ref_tok_counts.tolist() == [3.0, 2.0, 2.0]
   

def test_chunked_sequence_logprobs_computes_sums():
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, -100], [2, 3]])
    logp, tok_counts, _hidden = _chunked_sequence_logprobs(
        _LinearModel(),
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    # First sequence only first token counted, second sequence two tokens
    assert tok_counts.tolist() == [1, 2]
    assert logp.numel() == 2


def test_chunked_sequence_logprobs_honors_chunk_limits():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[int] = []
            self.embed_tokens = torch.nn.Embedding(16, 4)

        def forward(self, input_ids, attention_mask, labels, output_hidden_states=False):
            self.calls.append(input_ids.shape[0])
            vocab = 8
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            hidden_np = np.asarray(input_ids)
            hidden_np = np.repeat(hidden_np[..., None], 4, axis=-1).astype(np.float32)
            hidden = torch.tensor(hidden_np, dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=[hidden])

    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=torch.long)
    base_model = _Model()
    chunk_model = _Model()
    base_logp, base_tok, base_hidden = _chunked_sequence_logprobs(
        base_model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=0,
        return_hidden=True,
        pooling="last",
    )
    chunk_logp, chunk_tok, chunk_hidden = _chunked_sequence_logprobs(
        chunk_model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=2,
        return_hidden=True,
        pooling="last",
    )
    assert base_model.calls == [3]
    assert chunk_model.calls == [2, 1]

    def _to_np(tensor):
        try:
            return np.asarray(tensor)
        except Exception:
            arr = getattr(tensor, "arr", None)
            return np.asarray(arr if arr is not None else tensor)

    np.testing.assert_allclose(_to_np(base_logp), _to_np(chunk_logp))
    np.testing.assert_allclose(_to_np(base_tok), _to_np(chunk_tok))
    np.testing.assert_allclose(_to_np(base_hidden), _to_np(chunk_hidden))


def test_chunked_sequence_logprobs_gathers_params(monkeypatch):
    """Ensure ZeRO-3 gathered parameters context is invoked when requested."""

    class _Ctx:
        entered = False

        def __enter__(self):
            _Ctx.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Zero:
        def __init__(self):
            self.seen = None

        def GatheredParameters(self, params, modifier_rank=None):
            self.seen = params
            return _Ctx()

    # Patch a lightweight deepspeed.zero stub
    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(zero=_Zero()))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 2)

        def forward(self, input_ids, attention_mask, labels, output_hidden_states=False):
            vocab = 5
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=None)

    model = _Model()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    _chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
        gather_full_params=True,
    )
    stub = sys.modules.get("deepspeed")
    assert getattr(getattr(stub, "zero", None), "seen", None) is not None
    assert _Ctx.entered


def test_chunked_sequence_logprobs_uses_zero_embedding_context(monkeypatch):
    """Reference scoring should always enter the ZeRO embedding gather context."""

    class _Ctx:
        def __init__(self, model):
            self.model = model
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls: dict[str, object] = {}

    def _fake_ctx(model):
        ctx = _Ctx(model)
        calls["ctx"] = ctx
        return ctx

    monkeypatch.setattr(scoring, "_maybe_zero_gather_embedding", _fake_ctx)

    class _Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, labels, output_hidden_states=False):
            vocab = 5
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=None)

    model = _Model()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    _chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    ctx = calls.get("ctx")
    assert isinstance(ctx, _Ctx)
    assert ctx.entered is True
    assert ctx.model is model


def test_chunked_sequence_logprobs_uses_zero_param_context(monkeypatch):
    """Reference scoring must gather all ZeRO params before the forward pass."""

    class _Ctx:
        def __init__(self, model, enabled):
            self.model = model
            self.enabled = enabled
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls: dict[str, object] = {}

    def _fake_ctx(model, enabled):
        ctx = _Ctx(model, enabled)
        calls["ctx"] = ctx
        return ctx

    monkeypatch.setattr(scoring, "_maybe_zero_gather_params", _fake_ctx)

    class _Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, labels, output_hidden_states=False):
            vocab = 5
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=None)

    model = _Model()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    _chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    ctx = calls.get("ctx")
    assert isinstance(ctx, _Ctx)
    assert ctx.entered is True
    assert ctx.model is model
    assert ctx.enabled is True


def test_chunked_sequence_logprobs_param_gather_restores_weights(monkeypatch):
    """ZeRO param gather should expose non-empty weights to the model forward."""

    class _WeightStub:
        def __init__(self):
            self._shape = (0,)

        @property
        def shape(self):
            return self._shape

        def set_shape(self, shape):
            self._shape = tuple(shape)

    class _Model(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
            self.config = SimpleNamespace(pad_token_id=0, vocab_size=5)
            self.embed_tokens = torch.nn.Embedding(self.config.vocab_size, 2)

        def forward(self, input_ids, attention_mask, labels, output_hidden_states=False):
            if not self.weight.shape or self.weight.shape[0] == 0:
                raise RuntimeError("norm weight missing")
            vocab = self.config.vocab_size
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=None)

    weight = _WeightStub()
    model = _Model(weight)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2]], dtype=torch.long)

    def _run_with_ctx(ctx_factory):
        monkeypatch.setattr(scoring, "_maybe_zero_gather_params", ctx_factory)
        return _chunked_sequence_logprobs(
            model,
            input_ids=input_ids,
            attention_mask=attn,
            labels=labels,
            chunk_size=1,
        )

    def _null_ctx(model, enabled):
        _ = model, enabled
        return nullcontext()

    monkeypatch.setattr(scoring, "_maybe_zero_gather_embedding", lambda _m: nullcontext())
    monkeypatch.setattr(scoring, "_maybe_zero_gather_params", _null_ctx)
    with pytest.raises(RuntimeError):
        _chunked_sequence_logprobs(
            model,
            input_ids=input_ids,
            attention_mask=attn,
            labels=labels,
            chunk_size=1,
        )

    class _GatherCtx:
        def __enter__(self):
            weight.set_shape((3584,))
            return self

        def __exit__(self, exc_type, exc, tb):
            weight.set_shape((0,))
            return False

    def _ctx_factory(model, enabled):
        _ = model, enabled
        return _GatherCtx()

    result = _run_with_ctx(_ctx_factory)
    assert isinstance(result, tuple) and len(result) == 3
    logp, tok_counts, _hidden = result
    assert logp.numel() == 1
    assert tok_counts.tolist() == [2]
    assert weight.shape == (0,)


def test_chunked_sequence_logprobs_gather_restores_2d_embeddings(monkeypatch):
    """ZeRO embedding gather should recover 2-D weights for reference scoring."""

    class _WeightStub:
        def __init__(self):
            self._shape = (0,)

        def set_shape(self, shape):
            self._shape = tuple(shape)

        @property
        def shape(self):
            return self._shape

        def size(self, dim=None):
            if dim is None:
                return self._shape
            return self._shape[dim]

    class _Emb:
        def __init__(self, weight):
            self.weight = weight
            self.padding_idx = 0

    class _Model(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self._emb = _Emb(weight)
            self.embed_tokens = None
            self.config = SimpleNamespace(pad_token_id=0, vocab_size=5)

        def get_input_embeddings(self):
            return self._emb

        def forward(self, input_ids, attention_mask, labels, output_hidden_states=False):
            vocab = self.config.vocab_size
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=None)

    def _run_once(weight, gather_ctx):
        model = _Model(weight)
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        attn = torch.ones_like(input_ids)
        labels = torch.tensor([[1, 2]], dtype=torch.long)
        monkeypatch.setattr(scoring, "_maybe_zero_gather_embedding", gather_ctx)
        return _chunked_sequence_logprobs(
            model,
            input_ids=input_ids,
            attention_mask=attn,
            labels=labels,
            chunk_size=1,
        )

    weight = _WeightStub()

    # Without a gather context, the zero-dim embedding should trigger a skip.
    assert _run_once(weight, lambda _model: nullcontext()) is None

    class _GatherCtx:
        def __enter__(self):
            weight.set_shape((4, 3))
            return self

        def __exit__(self, exc_type, exc, tb):
            weight.set_shape((0,))
            return False

    def _ctx_factory(model):
        _ = model
        return _GatherCtx()

    result = _run_once(weight, _ctx_factory)
    assert isinstance(result, tuple) and len(result) == 3
    logp, tok_counts, _ = result
    assert logp.numel() == 1
    assert tok_counts.tolist() == [2]
    assert weight.shape == (0,)


def test_chunked_sequence_logprobs_handles_attribute_only_config():
    """Config objects without a mapping interface should still be read safely."""

    class _Config:
        def __init__(self):
            self.pad_token_id = 1
            self.vocab_size = 5

    class _Model:
        def __init__(self):
            cfg = _Config()
            self.config = cfg
            self.embed_tokens = SimpleNamespace(
                weight=torch.zeros((cfg.vocab_size, 2), dtype=torch.float32)
            )

        def __call__(self, input_ids, attention_mask, labels, output_hidden_states=False):
            logits = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
                dtype=torch.float32,
            )
            return SimpleNamespace(logits=logits, hidden_states=None)

    model = _Model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, -100, 2]], dtype=torch.long)
    logp, tok_counts, _hidden = _chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    assert tok_counts.tolist() == [2]
    assert logp.shape == (1,)


def test_chunked_sequence_logprobs_uses_input_embeddings_when_embed_missing():
    """Models without `embed_tokens` should still expose embeddings via getters."""

    class _Embedding:
        def __init__(self):
            self.weight = torch.zeros((8, 2), dtype=torch.float32)

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=0, vocab_size=8)
            self._embedding = _Embedding()

        def get_input_embeddings(self):
            return self._embedding

        def __call__(self, input_ids, attention_mask, labels, output_hidden_states=False):
            logits = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
                dtype=torch.float32,
            )
            return SimpleNamespace(logits=logits, hidden_states=None)

    model = _Model()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    logp, tok_counts, _hidden = _chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    assert tok_counts.tolist() == [2]
    assert logp.shape == (1,)


def test_chunked_sequence_logprobs_clamps_config_pad_token():
    """Clamp config.pad_token_id to the embedding vocab when it is too large."""

    class _Config:
        def __init__(self):
            self.pad_token_id = 10
            self.vocab_size = 20

    class _Model:
        def __init__(self):
            self.config = _Config()
            self.embed_tokens = SimpleNamespace(
                weight=torch.zeros((3, 2), dtype=torch.float32)
            )

        def __call__(self, input_ids, attention_mask, labels, output_hidden_states=False):
            vocab = self.config.vocab_size
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=None)

    model = _Model()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    _chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    # Config should be restored to its original pad_token_id.
    assert model.config.pad_token_id == 10


def test_chunked_sequence_logprobs_applies_clamped_pad_during_forward():
    """Ensure the clamped pad_token_id is seen inside the model call."""

    class _Config:
        def __init__(self):
            self.pad_token_id = 11
            self.vocab_size = 20

    class _Model:
        def __init__(self):
            self.config = _Config()
            self.embed_tokens = SimpleNamespace(
                weight=torch.zeros((4, 2), dtype=torch.float32)
            )
            self.seen_pad = None

        def __call__(self, input_ids, attention_mask, labels, output_hidden_states=False):
            # record what pad_token_id the guard exposed while in the forward pass
            self.seen_pad = self.config.pad_token_id
            vocab = self.config.vocab_size
            bsz, seqlen = input_ids.shape
            logits = torch.zeros((bsz, seqlen, vocab), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden_states=None)

        def forward(self, *args, **kwargs):
            return self.__call__(*args, **kwargs)

    model = _Model()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    _chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    assert model.seen_pad == 3
    assert model.config.pad_token_id == 11


def test_reference_from_vllm_meta_handles_valid_payload(monkeypatch):
    # Provide a lightweight torch stub that accepts ``device=`` during tensor construction.
    class _Tensor:
        def __init__(self, data, dtype=None, device=None):
            self.arr = np.asarray(data, dtype=dtype)
            self.device = device

        def tolist(self):
            return self.arr.tolist()

        def cpu(self):
            return self

        def numel(self):
            return self.arr.size

    def _tensor(data, dtype=None, device=None):
        return _Tensor(data, dtype=dtype, device=device)

    torch_stub = SimpleNamespace(
        tensor=_tensor,
        full=lambda shape, val, dtype=None: _Tensor(np.full(shape, val, dtype=dtype)),
        ones_like=lambda t, dtype=None: _Tensor(np.ones_like(t.arr, dtype=dtype)),
        zeros=lambda shape, dtype=None: _Tensor(np.zeros(shape, dtype=dtype)),
        cat=lambda seq, dim=0: _Tensor(np.concatenate([s.arr for s in seq], axis=dim)),
        float32=np.float32,
        int64=np.int64,
        long=np.int64,
        device=lambda name="cpu": SimpleNamespace(type=name),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(scoring, "torch", torch_stub)
    monkeypatch.setattr(scoring, "_refresh_torch", lambda: torch_stub)

    meta = [
        {"logprob_sum": -1.0, "token_count": 2},
        {"logprob_sum": -0.5, "token_count": 1},
    ]
    ref = reference_from_vllm_meta(
        meta, total_sequences=2, device=torch_stub.device("cpu")
    )
    assert isinstance(ref, ReferenceLogprobs)
    assert ref.ref_tok_counts.tolist() == [2.0, 1.0]
    assert pytest.approx(ref.ref_logp_mean) == -0.75


def test_reference_from_vllm_meta_handles_attr_objects():
    class _Payload:
        def __init__(self, logprob_sum, token_count):
            self.logprob_sum = logprob_sum
            self.token_count = token_count

    device = torch.device("cpu")
    meta = [_Payload(-1.5, 4), _Payload(-0.5, 2)]
    ref = reference_from_vllm_meta(meta, total_sequences=2, device=device)
    assert isinstance(ref, ReferenceLogprobs)
    assert ref.ref_tok_counts.tolist() == [4.0, 2.0]
    assert ref.ref_logp_sum_raw.tolist() == [-1.5, -0.5]


def test_finalize_reference_stats_and_lengths_summary():
    ref = finalize_reference_stats(
        torch.tensor([-2.0, -1.0], dtype=torch.float32),
        torch.tensor([2.0, 1.0], dtype=torch.float32),
    )
    lengths, stats, total_tokens = summarize_completion_lengths(
        ref, max_completion_len=2
    )
    assert isinstance(stats, LengthStats)
    assert lengths.tolist() == [2.0, 1.0]
    assert pytest.approx(total_tokens) == 3.0
    assert stats.clipped_ratio == 0.5  # one completion at max len


def test_build_sequence_scores_clamps_empty_denominator():
    ref_stats = ReferenceLogprobs(
        ref_logp_sum=torch.tensor([0.0, 0.0], dtype=torch.float32),
        ref_tok_counts=torch.tensor([0.0, 0.0], dtype=torch.float32),
        ref_logp_sum_raw=torch.tensor([0.0, 0.0], dtype=torch.float32),
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )
    cur = torch.tensor([0.5, 1.0])
    scores = build_sequence_scores(cur, ref_stats)
    assert scores.denom_tok_tensor.tolist() == [1.0, 1.0]


def test_build_sequence_scores_defaults_behavior_to_current_policy():
    ref_stats = ReferenceLogprobs(
        ref_logp_sum=torch.tensor([-1.0, -2.0], dtype=torch.float32),
        ref_tok_counts=torch.tensor([1.0, 1.0], dtype=torch.float32),
        ref_logp_sum_raw=torch.tensor([-1.0, -2.0], dtype=torch.float32),
        ref_logp_mean=-1.5,
        avg_completion_tokens=1.0,
    )
    cur = torch.tensor([0.2, 0.4], dtype=torch.float32)
    scores = build_sequence_scores(cur, ref_stats)
    # Behavior should follow the rollout actor (current policy) by default, not the reference model.
    assert scores.behavior_logp_sum.tolist() == pytest.approx(cur.tolist())
    # The ref-based log ratio should remain unchanged for KL logging.
    assert scores.log_ratio_train.tolist() == pytest.approx([1.2, 2.4])


def test_build_sequence_scores_accepts_behavior_override():
    ref_stats = ReferenceLogprobs(
        ref_logp_sum=torch.tensor([-1.0, -1.0], dtype=torch.float32),
        ref_tok_counts=torch.tensor([1.0, 1.0], dtype=torch.float32),
        ref_logp_sum_raw=torch.tensor([-1.0, -1.0], dtype=torch.float32),
        ref_logp_mean=-1.0,
        avg_completion_tokens=1.0,
    )
    cur = torch.tensor([0.0, 0.0], dtype=torch.float32)
    behavior = torch.tensor([0.3, 0.4], dtype=torch.float32)
    scores = build_sequence_scores(cur, ref_stats, behavior_logp_sum=behavior)
    assert scores.behavior_logp_sum.tolist() == pytest.approx(behavior.tolist())


def test_score_model_outputs_and_reference_from_model(monkeypatch):
    sb, tokenizer, gen_cfg, batching_cfg = _score_batch()
    device = torch.device("cpu")

    class _Model(_LinearModel):
        def to(self, *_args, **_kwargs):
            return self

    runtime = SimpleNamespace(
        accelerator=SimpleNamespace(autocast=lambda: nullcontext()),
        device=device,
        get_ref_model=lambda: _Model(),
    )

    ref_tensors = reference_from_model(sb, runtime, batching_cfg)
    assert ref_tensors is not None
    ref_stats = finalize_reference_stats(*ref_tensors)
    cur_logp = score_model_outputs(_Model(), sb, batching_cfg, runtime)
    assert cur_logp is not None
    cur_logp_sum = cur_logp[0] if isinstance(cur_logp, tuple) else cur_logp
    scores = build_sequence_scores(cur_logp_sum, ref_stats)
    assert scores.cur_logp_sum.shape == ref_stats.ref_logp_sum_raw.shape


def test_autocast_context_prefers_accelerator_autocast(monkeypatch):
    marker = object()

    class _Accel:
        def __init__(self):
            self.called = 0

        def autocast(self):
            self.called += 1
            return marker

    # Ensure torch.autocast is not consulted when accelerator provides one.
    monkeypatch.setattr(
        torch,
        "autocast",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("torch.autocast should not be called when accel exists")
        ),
    )
    accel = _Accel()
    _autocast_context(accel, torch.device("cpu"))
    assert accel.called == 1
    # Only requirement is that accelerator.autocast is used; the specific
    # object returned can vary, so just make sure torch.autocast was not hit.

    # Fallback path: patch torch.autocast to a sentinel context manager
    sentinel = nullcontext()
    monkeypatch.setattr(torch, "autocast", lambda **_: sentinel)
    ctx2 = _autocast_context(SimpleNamespace(), torch.device("cpu"))
    assert ctx2 is sentinel


def test_tokenize_completions_and_build_score_batch():
    sb, tokenizer, gen_cfg, _ = _score_batch()
    assert isinstance(sb, ScoreBatch)
    # Tokenizer was invoked when building the score batch
    assert tokenizer.called is True


def test_refresh_torch_handles_import_failure(monkeypatch):
    """_refresh_torch should recover when ops.sitecustomize import fails."""
    original_import = builtins.__import__
    bad_torch = types.SimpleNamespace(
        full=lambda *a, **k: None,
        ones_like=lambda *a, **k: None,
        zeros=lambda *a, **k: None,
        cat=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", bad_torch)

    def _fake_import(name, *args, **kwargs):
        if name.startswith("ops.sitecustomize"):
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    torch_mod = scoring._refresh_torch()
    assert hasattr(torch_mod, "tensor")


def test_refresh_torch_replaces_broken_tensor(monkeypatch):
    class _BadTorch:
        def __init__(self):
            self.full = self.ones_like = self.zeros = self.cat = lambda *a, **k: None
            self.nn = types.SimpleNamespace(functional=None)

        def tensor(self, *args, **kwargs):
            raise RuntimeError("broken")

    bad = _BadTorch()
    monkeypatch.setitem(sys.modules, "torch", bad)
    torch_mod = scoring._refresh_torch()
    assert torch_mod is not bad
    assert hasattr(torch_mod, "tensor")


def test_refresh_torch_sets_tensor_when_required_list_empty(monkeypatch):
    stub = types.SimpleNamespace(
        full=lambda *a, **k: None,
        ones_like=lambda *a, **k: None,
        zeros=lambda *a, **k: None,
        cat=lambda *a, **k: None,
        tensor=lambda *a, **k: "stub_tensor",
    )
    monkeypatch.setattr(scoring, "_REQUIRED_TORCH_ATTRS", tuple())
    monkeypatch.setattr(scoring, "_build_torch_stub", lambda: stub)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    torch_mod = scoring._refresh_torch()
    assert callable(getattr(torch_mod, "tensor"))
    assert torch_mod.tensor([]) == "stub_tensor"


def test_maybe_long_tensor_handles_typeerror(monkeypatch):
    class _Val:
        def __init__(self):
            self.arr = None

        def long(self):
            raise TypeError("nope")

        def __array__(self, dtype=None):
            return np.array([1, 2], dtype=dtype)

    result = scoring._maybe_long_tensor(_Val(), torch)
    assert hasattr(result, "dtype")


def test_maybe_long_tensor_handles_missing_long(monkeypatch):
    class _Val:
        def __array__(self, dtype=None):
            return np.array([3, 4], dtype=dtype)

    result = scoring._maybe_long_tensor(_Val(), torch)
    assert hasattr(result, "dtype")


def test_size_hint_and_to_numpy_array_fallbacks(monkeypatch):
    class _BadLen:
        def __len__(self):
            raise TypeError("len failed")

    assert scoring._size_hint(_BadLen(), dim=0) == 0

    class _ShapeScalar:
        shape = 5

    assert scoring._size_hint(_ShapeScalar(), dim=None) == 5

    class _ArrRaises:
        def __init__(self):
            self.arr = types.SimpleNamespace(
                __array__=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    ValueError("bad arr")
                )
            )
            self.data = [1, 2]

    assert scoring._to_numpy_array(_ArrRaises()).tolist() == [1, 2]

    class _DataRaises:
        def __init__(self):
            class _BadData:
                def __array__(self, dtype=None):
                    raise TypeError("bad data")

            self.data = _BadData()

        def __array__(self, dtype=None):
            return np.array([4, 5], dtype=dtype)

    assert scoring._to_numpy_array(_DataRaises()).tolist() == [4, 5]

    class _ArrayFail:
        def __array__(self, dtype=None):
            raise RuntimeError("array fail")

    assert scoring._to_numpy_array(_ArrayFail()).size == 0


def test_resolve_dtype_handles_np_dtype_and_invalid():
    class _DType:
        np_dtype = np.float16

    assert scoring._resolve_dtype(_DType()) == np.float16
    assert scoring._resolve_dtype(object()) is None


def test_autocast_context_handles_runtime_error(monkeypatch):
    sentinel = nullcontext()

    def _bad_autocast(**_kwargs):
        raise ValueError("no autocast")

    monkeypatch.setattr(torch, "autocast", _bad_autocast)
    ctx = scoring._autocast_context(SimpleNamespace(), torch.device("cuda"))
    assert isinstance(ctx, type(sentinel))


def test_autocast_context_returns_null_when_absent(monkeypatch):
    monkeypatch.setattr(scoring, "torch", SimpleNamespace(autocast=None))
    ctx = scoring._autocast_context(SimpleNamespace(), SimpleNamespace(type="cpu"))
    assert isinstance(ctx, nullcontext)


def test_slice_state_handles_empty_and_missing_prompts(monkeypatch):
    empty_sb = ScoreBatch(
        prompt_entries=[],
        completion_ids=torch.tensor([]),
        completion_attention_mask=torch.tensor([]),
        pad_token_id=0,
        max_prompt_len=1,
        slice_size=0,
        total_sequences=0,
    )
    assert list(iter_batch_slices(empty_sb, torch.device("cpu"))) == []

    sb_missing_prompts = ScoreBatch(
        prompt_entries=[],
        completion_ids=torch.tensor([[1]]),
        completion_attention_mask=torch.tensor([[1]]),
        pad_token_id=0,
        max_prompt_len=1,
        slice_size=1,
        total_sequences=1,
    )
    assert list(iter_batch_slices(sb_missing_prompts, torch.device("cpu"))) == []


def test_prepare_prompt_slice_handles_zero_lengths():
    prompt_slice = [
        PromptCacheEntry(input_ids=[1, 2], attention_mask=[1, 1]),
        PromptCacheEntry(input_ids=[], attention_mask=[]),
    ]
    ids_dtype = torch.tensor([[1]]).dtype
    mask_dtype = torch.tensor([[1]]).dtype
    prompt_ids, prompt_mask, lengths = scoring._prepare_prompt_slice(
        prompt_slice,
        max_prompt_len=4,
        pad_token_id=0,
        ids_dtype=ids_dtype,
        mask_dtype=mask_dtype,
    )
    assert lengths == [2, 0]
    assert prompt_ids.shape[1] == 2

    zero_slice = [PromptCacheEntry(input_ids=[], attention_mask=[])]
    prompt_ids2, prompt_mask2, lengths2 = scoring._prepare_prompt_slice(
        zero_slice,
        max_prompt_len=2,
        pad_token_id=0,
        ids_dtype=ids_dtype,
        mask_dtype=mask_dtype,
    )
    assert prompt_ids2.shape[1] == 0
    assert prompt_mask2.shape[1] == 0
    assert lengths2 == [0]


def test_build_score_batch_fallbacks(monkeypatch):
    reward_comp = RewardComputation(
        total_utils=[],
        per_reward_values={},
        advantage=SimpleNamespace(samples=[]),
        pairs=SimpleNamespace(prompts=["p"], completions=["c"]),
        q_distribution=SimpleNamespace(grouped=[], samples=[]),
        moments=SimpleNamespace(mean=0.0, std=0.0),
    )

    class _Tok:
        pad_token_id = None
        eos_token_id = 7

        def __call__(self, texts, **_kwargs):
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

    batching_cfg = SimpleNamespace(
        logprob_chunk_size=0, score_slice=0, prompt_length_cache_get=None
    )
    sb = build_score_batch(
        reward_comp,
        _Tok(),
        GenerationSettings(
            max_prompt_len=1,
            max_completion_len=1,
            gen_temperature=1.0,
            gen_top_p=1.0,
            use_vllm=False,
            vllm=VLLMClientConfig(
                url="http://localhost",
                rounds_cfg=1,
                retry_sleep=0.0,
                backfill_local=False,
                request_logprobs=False,
            ),
        ),
        batching_cfg,
    )
    assert sb is not None
    assert sb.pad_token_id == 7

    # Empty batch should return None
    reward_comp_empty = RewardComputation(
        total_utils=[],
        per_reward_values={},
        advantage=SimpleNamespace(samples=[]),
        pairs=SimpleNamespace(prompts=[], completions=[]),
        q_distribution=SimpleNamespace(grouped=[], samples=[]),
        moments=SimpleNamespace(mean=0.0, std=0.0),
    )
    assert (
        build_score_batch(
            reward_comp_empty,
            _Tok(),
            GenerationSettings(
                max_prompt_len=1,
                max_completion_len=1,
                gen_temperature=1.0,
                gen_top_p=1.0,
                use_vllm=False,
                vllm=VLLMClientConfig(
                    url="http://localhost",
                    rounds_cfg=1,
                    retry_sleep=0.0,
                    backfill_local=False,
                    request_logprobs=False,
                ),
            ),
            batching_cfg,
        )
        is None
    )

    monkeypatch.setattr(
        scoring, "_collect_prompt_entries", lambda *args, **kwargs: None
    )
    assert sb is not None
    assert (
        build_score_batch(
            reward_comp,
            _Tok(),
            GenerationSettings(
                max_prompt_len=1,
                max_completion_len=1,
                gen_temperature=1.0,
                gen_top_p=1.0,
                use_vllm=False,
                vllm=VLLMClientConfig(
                    url="http://localhost",
                    rounds_cfg=1,
                    retry_sleep=0.0,
                    backfill_local=False,
                    request_logprobs=False,
                ),
            ),
            batching_cfg,
        )
        is None
    )


def test_reference_from_model_handles_empty_slices(monkeypatch):
    sb = ScoreBatch(
        prompt_entries=[],
        completion_ids=torch.tensor([]),
        completion_attention_mask=torch.tensor([]),
        pad_token_id=0,
        max_prompt_len=1,
        slice_size=0,
        total_sequences=0,
    )
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        accelerator=SimpleNamespace(autocast=lambda: nullcontext()),
        get_ref_model=lambda: None,
    )
    batching_cfg = SimpleNamespace(logprob_chunk_size=0)
    assert reference_from_model(sb, runtime, batching_cfg) is None
    assert gather_reference_logprobs(sb, runtime, batching_cfg) is None

    # Force empty tensors mid-loop
    monkeypatch.setattr(
        scoring,
        "_chunked_sequence_logprobs",
        lambda *args, **kwargs: (torch.tensor([]), torch.tensor([])),
    )
    sb2, _, _, batching_cfg2 = _score_batch()
    runtime.get_ref_model = lambda: _LinearModel()
    assert reference_from_model(sb2, runtime, batching_cfg2) is None



def test_reference_from_vllm_meta_invalid_cases():
    device = torch.device("cpu")
    assert reference_from_vllm_meta([], total_sequences=1, device=device) is None
    assert reference_from_vllm_meta([{}], total_sequences=2, device=device) is None
    assert reference_from_vllm_meta([None], total_sequences=1, device=device) is None
    assert (
        reference_from_vllm_meta(
            [{"logprob_sum": None, "token_count": None}],
            total_sequences=1,
            device=device,
        )
        is None
    )


def test_score_model_outputs_handles_empty(monkeypatch):
    sb = ScoreBatch(
        prompt_entries=[],
        completion_ids=torch.tensor([]),
        completion_attention_mask=torch.tensor([]),
        pad_token_id=0,
        max_prompt_len=1,
        slice_size=0,
        total_sequences=0,
    )
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        accelerator=SimpleNamespace(autocast=lambda: nullcontext()),
    )
    assert (
        score_model_outputs(
            _LinearModel(), sb, SimpleNamespace(logprob_chunk_size=0), runtime
        )
        is None
    )


def test_finalize_reference_stats_and_lengths_empty(monkeypatch):
    ref = finalize_reference_stats(torch.tensor([]), torch.tensor([]))
    lengths, stats, total_tokens = summarize_completion_lengths(
        ref, max_completion_len=1
    )
    assert lengths.numel() == 1  # finalize_reference_stats injects a zero token count
    assert stats.min_length == 0.0
    assert total_tokens == 0.0

    ref_empty = ReferenceLogprobs(
        ref_logp_sum=torch.tensor([]),
        ref_tok_counts=torch.tensor([]),
        ref_logp_sum_raw=torch.tensor([]),
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )
    lengths2, stats2, total_tokens2 = summarize_completion_lengths(
        ref_empty, max_completion_len=1
    )
    assert lengths2.numel() == 0
    assert stats2.min_length == 0.0
    assert total_tokens2 == 0.0


def test_build_sequence_scores_handles_empty_denoms(monkeypatch):
    ref_stats = ReferenceLogprobs(
        ref_logp_sum=torch.tensor([]),
        ref_tok_counts=torch.tensor([]),
        ref_logp_sum_raw=torch.tensor([]),
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )
    scores = build_sequence_scores(torch.tensor([]), ref_stats)
    assert getattr(scores.denom_tok_tensor, "numel", lambda: 0)() >= 0


def test_pad_token_guard_applies_to_two_dim_weight() -> None:
    class S2D(SimpleNamespace):
        pass

    module = S2D(padding_idx=5)
    module.weight = np.ones((4, 4))
    with _PadTokenGuard([(module, "padding_idx")], 99):
        # The guard should clamp the requested padding index to be
        # within the number of embeddings exposed by the weight.
        assert module.padding_idx == 3
    assert module.padding_idx == 5


def test_pad_token_guard_skips_non_two_dim_weight() -> None:
    class S1D(SimpleNamespace):
        pass

    module = S1D(padding_idx=5)
    module.weight = np.ones((4,))
    with _PadTokenGuard([(module, "padding_idx")], 99):
        assert module.padding_idx == 5


def test_pad_token_guard_uses_num_embeddings_when_available() -> None:
    class FakeEmbedding(SimpleNamespace):
        pass

    module = FakeEmbedding(padding_idx=100, num_embeddings=10)
    # A 1-D weight shape mimics sharded or stubbed embeddings where the
    # module still exposes its logical embedding count via ``num_embeddings``.
    module.weight = np.ones((4,))
    with _PadTokenGuard([(module, "padding_idx")], 99):
        assert module.padding_idx == 9
    assert module.padding_idx == 100
