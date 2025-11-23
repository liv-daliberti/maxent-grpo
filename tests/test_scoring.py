"""Unit tests for training.scoring utilities."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import List

import pytest
import numpy as np
import sys
import types


# Provide a richer torch stub when the real package is unavailable or broken.
try:  # pragma: no cover - environment dependent
    import torch as _torch_mod  # type: ignore
except Exception:
    _torch_mod = None

_needs_stub = _torch_mod is None  # Prefer real torch when available.
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
            if target_dtype is None and device is not None and not isinstance(device, (_Device, str)):
                target_dtype = device
            return _Tensor(self.arr.astype(target_dtype) if target_dtype is not None else self.arr)

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
            return _Tensor(np.clip(arr, lo if lo is not None else arr.min(), hi if hi is not None else arr.max()))

        def unsqueeze(self, dim):
            return _Tensor(np.expand_dims(self.arr, axis=dim))

        def squeeze(self, dim=None):
            return _Tensor(np.squeeze(self.arr, axis=dim) if dim is not None else np.squeeze(self.arr))

        def reshape(self, *shape):
            return _Tensor(self.arr.reshape(*shape))

        def gather(self, dim, index):
            idx = index.arr if isinstance(index, _Tensor) else index
            result = np.take_along_axis(self.arr, idx, axis=dim)
            return _Tensor(result)

        def ne(self, other):
            return _Tensor(self.arr != (other.arr if isinstance(other, _Tensor) else other))

        def eq(self, other):
            return _Tensor(self.arr == (other.arr if isinstance(other, _Tensor) else other))

        def ge(self, other):
            return _Tensor(self.arr >= (other.arr if isinstance(other, _Tensor) else other))

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
            return _Tensor(self.arr + (other.arr if isinstance(other, _Tensor) else other))

        def __sub__(self, other):
            return _Tensor(self.arr - (other.arr if isinstance(other, _Tensor) else other))

        def __mul__(self, other):
            return _Tensor(self.arr * (other.arr if isinstance(other, _Tensor) else other))

        def __truediv__(self, other):
            return _Tensor(self.arr / (other.arr if isinstance(other, _Tensor) else other))

        def __gt__(self, other):
            return _Tensor(self.arr > (other.arr if isinstance(other, _Tensor) else other))

        def __lt__(self, other):
            return _Tensor(self.arr < (other.arr if isinstance(other, _Tensor) else other))

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

from training.scoring import (  # noqa: E402
    _autocast_context,
    _chunked_sequence_logprobs,
    _collect_prompt_entries,
    build_score_batch,
    build_sequence_scores,
    finalize_reference_stats,
    iter_batch_slices,
    reference_from_model,
    reference_from_vllm_meta,
    score_model_outputs,
    summarize_completion_lengths,
)
from training.types import (  # noqa: E402
    BatchingSettings,
    GenerationSettings,
    LengthStats,
    PromptCacheEntry,
    ReferenceLogprobs,
    RewardComputation,
    ScoreBatch,
)
from training.run_helpers import VLLMClientConfig  # noqa: E402


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
    assert np.all(np.asarray(labels[:, 2:]) >= 0)


def test_chunked_sequence_logprobs_computes_sums():
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attn = torch.ones_like(input_ids)
    labels = torch.tensor([[1, -100], [2, 3]])
    logp, tok_counts = _chunked_sequence_logprobs(
        _LinearModel(),
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        chunk_size=1,
    )
    # First sequence only first token counted, second sequence two tokens
    assert tok_counts.tolist() == [1, 2]
    assert logp.numel() == 2


def test_reference_from_vllm_meta_handles_valid_payload():
    meta = [
        {"logprob_sum": -1.0, "token_count": 2},
        {"logprob_sum": -0.5, "token_count": 1},
    ]
    ref = reference_from_vllm_meta(meta, total_sequences=2, device=torch.device("cpu"))
    assert isinstance(ref, ReferenceLogprobs)
    assert ref.ref_tok_counts.tolist() == [2.0, 1.0]
    assert pytest.approx(ref.ref_logp_mean) == -0.75


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
    scores = build_sequence_scores(cur_logp, ref_stats)
    assert scores.cur_logp_sum.shape == ref_stats.ref_logp_sum_raw.shape


def test_autocast_context_prefers_accelerator_autocast(monkeypatch):
    marker = object()

    class _Accel:
        def autocast(self):
            return marker

    ctx = _autocast_context(_Accel(), torch.device("cpu"))
    assert ctx is marker

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
