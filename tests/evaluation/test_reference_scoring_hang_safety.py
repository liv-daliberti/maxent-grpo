from __future__ import annotations

from types import SimpleNamespace
import sys
from types import ModuleType

import pytest


def test_chunked_sequence_logprobs_deepspeed_embed_gather_uses_rank_any(monkeypatch):
    """Ensure ZeRO embed gather decision is collective to avoid deadlocks."""
    from maxent_grpo.training import scoring as scoring_mod

    torch = scoring_mod._refresh_torch()

    gathered_params_calls: list[list[object]] = []

    class _GatheredParameters:
        def __init__(self, params, modifier_rank=None):
            gathered_params_calls.append(list(params))

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    deepspeed_stub = ModuleType("deepspeed")
    deepspeed_stub.zero = SimpleNamespace(
        is_enabled=lambda: True, GatheredParameters=_GatheredParameters
    )
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_stub)

    class _FakeDist:
        def __init__(self):
            self._other_rank_values = iter(
                [
                    True,  # input_present_all (other rank has weight)
                    False,  # output_present_all (other rank also has no output weight)
                    True,  # needs_input_any (other rank needs gather even if local doesn't)
                    False,  # needs_output_any
                ]
            )

        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def all_gather_object(self, output_list, input_obj):
            output_list[0] = bool(input_obj)
            output_list[1] = bool(next(self._other_rank_values))

    # Patch the torch module used by scoring_mod (refresh may swap stubs).
    monkeypatch.setattr(torch, "distributed", _FakeDist(), raising=False)

    weight = torch.zeros((8, 4), dtype=getattr(torch, "float32", None))

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=0, vocab_size=16)
            self._emb = SimpleNamespace(weight=weight, padding_idx=0)

        def get_input_embeddings(self):
            return self._emb

        def get_output_embeddings(self):
            return None

        def forward(self, **_kwargs):
            raise RuntimeError("stop after gather ctx")

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

    model = _Model()
    input_ids = torch.zeros((1, 4), dtype=getattr(torch, "long", None))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full((1, 4), -100, dtype=getattr(torch, "long", None))

    with pytest.raises(RuntimeError, match="stop after gather ctx"):
        scoring_mod._chunked_sequence_logprobs(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            chunk_size=0,
            gather_full_params=False,
        )

    assert gathered_params_calls, "Expected DeepSpeed GatheredParameters to be used."
    assert len(gathered_params_calls[0]) == 1


def test_chunked_sequence_logprobs_avoids_nested_zero_gather_assert(monkeypatch):
    """Single gather strategy should avoid nested active_sub_modules failures."""
    from maxent_grpo.training import scoring as scoring_mod
    import maxent_grpo.training.zero_utils as zero_utils_mod

    torch = scoring_mod._refresh_torch()

    active_by_param: dict[int, set[int]] = {}
    gather_calls: list[list[object]] = []
    token_counter = {"value": 0}

    class _BadActive:
        def __bool__(self):
            raise TypeError("cannot coerce active state")

    class _FakeWeight:
        def __init__(self):
            self.shape = (8,)
            self.ds_id = 5
            self.ds_status = "NOT_AVAILABLE"
            self.ds_active_sub_modules = _BadActive()

    class _GatheredParameters:
        def __init__(self, params, modifier_rank=None):
            del modifier_rank
            self.params = list(params)
            gather_calls.append(self.params)
            token_counter["value"] += 1
            self.token = token_counter["value"]

        def __enter__(self):
            for param in self.params:
                key = id(param)
                active_by_param.setdefault(key, set()).add(self.token)
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            for param in self.params:
                key = id(param)
                active = active_by_param.get(key, set())
                if self.token in active:
                    active.remove(self.token)
                if active:
                    raise AssertionError({"active_sub_modules": set(active)})
                active_by_param[key] = active
            return False

    deepspeed_stub = ModuleType("deepspeed")
    deepspeed_stub.zero = SimpleNamespace(
        is_enabled=lambda: True, GatheredParameters=_GatheredParameters
    )
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_stub)

    # Keep zero_utils on the same gather implementation so nested contexts would
    # reproduce the DeepSpeed-style assertion without the lock/strategy fix.
    monkeypatch.setattr(zero_utils_mod, "_ensure_deepspeed_ready", lambda: True)
    monkeypatch.setattr(
        zero_utils_mod,
        "ds_zero",
        SimpleNamespace(GatheredParameters=_GatheredParameters),
    )
    monkeypatch.setattr(zero_utils_mod, "_is_deepspeed_engine", lambda _m: False)
    monkeypatch.setattr(zero_utils_mod, "_zero_stage", lambda _m: 0)
    monkeypatch.setattr(zero_utils_mod, "_gather_callable", lambda: _GatheredParameters)

    weight = _FakeWeight()

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=0, vocab_size=16)
            self._emb = SimpleNamespace(weight=weight, padding_idx=0)

        def parameters(self):
            return [weight]

        def get_input_embeddings(self):
            return self._emb

        def get_output_embeddings(self):
            return None

        def forward(self, **_kwargs):
            raise RuntimeError("stop after gather ctx")

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

    model = _Model()
    input_ids = torch.zeros((1, 4), dtype=getattr(torch, "long", None))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full((1, 4), -100, dtype=getattr(torch, "long", None))

    result = scoring_mod._chunked_sequence_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        chunk_size=0,
        gather_full_params=False,
    )

    # Previously this path could invoke nested GatheredParameters on the same
    # param (manual gather + helper gather), which triggers active_sub_modules
    # assertions at context exit.
    assert result is None
    assert len(gather_calls) == 1
    assert gather_calls[0] == [weight]


def test_sanitize_ref_logprob_meta_requires_complete_entries():
    """Keep partial metadata so behavior-logprob fallbacks can still run."""
    from maxent_grpo.training import rewards as rewards_mod

    complete = [
        {"logprob_sum": -1.0, "token_count": 2},
        {"logprob_sum": -0.5, "token_count": 1},
    ]
    assert (
        rewards_mod._sanitize_ref_logprob_meta(list(complete), total_sequences=2)
        == complete
    )
    missing = [{"logprob_sum": -1.0, "token_count": 2}, None]
    assert (
        rewards_mod._sanitize_ref_logprob_meta(list(missing), total_sequences=2)
        == missing
    )
    missing_fields = [{"logprob_sum": None, "token_count": 2}]
    assert (
        rewards_mod._sanitize_ref_logprob_meta(list(missing_fields), total_sequences=1)
        == missing_fields
    )


def test_gather_reference_logprobs_returns_none_if_any_rank_fails(monkeypatch):
    """If any rank fails to compute ref logprobs, all ranks must skip."""
    from maxent_grpo.training import scoring as scoring_mod
    from maxent_grpo.training import scoring_reference as scoring_ref_mod

    torch = scoring_mod._refresh_torch()

    def _fake_reference_from_model(_score_batch, _runtime, _batching_cfg):
        base = torch.zeros((2,))
        return base, torch.ones_like(base)

    class _Accel:
        num_processes = 2

        @staticmethod
        def gather_object(_value):
            return [True, False]

    monkeypatch.setattr(
        scoring_ref_mod,
        "reference_from_model",
        _fake_reference_from_model,
    )
    monkeypatch.setattr(
        scoring_ref_mod,
        "finalize_reference_stats",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    runtime = SimpleNamespace(accelerator=_Accel(), device="cpu")
    score_batch = SimpleNamespace(slice_size=4)
    batching_cfg = SimpleNamespace(logprob_chunk_size=0)

    assert (
        scoring_mod.gather_reference_logprobs(score_batch, runtime, batching_cfg)
        is None
    )


def test_gather_reference_logprobs_preflight_skips_when_any_rank_empty(monkeypatch):
    """Preflight must skip reference scoring when any rank would run zero slices."""
    from maxent_grpo.training import scoring as scoring_mod
    from maxent_grpo.training import scoring_reference as scoring_ref_mod

    torch = scoring_mod._refresh_torch()

    class _Accel:
        num_processes = 2

        @staticmethod
        def gather_object(value):
            # Local rank reports 1 row; other rank reports 0 rows -> must skip before forward.
            return [value, 0]

    monkeypatch.setattr(
        scoring_ref_mod,
        "reference_from_model",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("reference_from_model should not run")
        ),
    )

    runtime = SimpleNamespace(accelerator=_Accel(), device="cpu")
    score_batch = SimpleNamespace(
        total_sequences=1,
        slice_size=1,
        prompt_entries=[SimpleNamespace(length=1)],
        completion_ids=torch.zeros((1, 1)),
        completion_attention_mask=torch.ones_like(torch.zeros((1, 1))),
    )
    batching_cfg = SimpleNamespace(logprob_chunk_size=0)

    assert (
        scoring_mod.gather_reference_logprobs(score_batch, runtime, batching_cfg)
        is None
    )


def test_collect_batch_stats_skips_when_any_rank_missing_score_batch_under_zero(
    monkeypatch,
):
    """Avoid ZeRO deadlocks when ScoreBatch construction diverges across ranks."""
    from maxent_grpo.training import pipeline as pipeline_mod
    from maxent_grpo.training.runtime import require_torch

    torch = require_torch("test_collect_batch_stats_zero_guard")

    class _FakeDist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def all_gather_object(self, output_list, _input_obj):
            # Simulate a different rank reporting score_batch is None.
            output_list[0] = False
            output_list[1] = True

    monkeypatch.setattr(torch, "distributed", _FakeDist(), raising=False)

    accelerator = SimpleNamespace(
        num_processes=2,
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=2)),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", tokenizer="tok", accelerator=accelerator),
        scoring=SimpleNamespace(
            batching=SimpleNamespace(prompt_length_cache_get=None),
            weighting=SimpleNamespace(),
        ),
        generation=SimpleNamespace(max_completion_len=4),
    )
    gen_batch = SimpleNamespace(grouped_completions=[["a"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=[],
        pairs=SimpleNamespace(prompts=["p"], completions=["c"]),
    )

    monkeypatch.setattr(
        pipeline_mod,
        "build_score_batch",
        lambda *_a, **_k: SimpleNamespace(
            total_sequences=1, prompt_entries=[], max_prompt_len=4
        ),
    )

    assert pipeline_mod._collect_batch_stats(ctx, gen_batch, reward_comp) is None


def test_collect_batch_stats_runs_reference_gather_for_zero_alignment(monkeypatch):
    """When any rank needs ref-model scoring under ZeRO, all ranks must call it."""
    from maxent_grpo.training import pipeline as pipeline_mod
    from maxent_grpo.training.runtime import require_torch

    torch = require_torch("test_collect_batch_stats_zero_ref_alignment")

    class _FakeDist:
        def __init__(self):
            self._calls = 0

        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def all_gather_object(self, output_list, input_obj):
            # First call: ScoreBatch is present on all ranks -> do not skip.
            # Second call: Another rank needs reference-model scoring -> align by calling gather on all.
            self._calls += 1
            output_list[0] = bool(input_obj)
            output_list[1] = False if self._calls == 1 else True

    monkeypatch.setattr(torch, "distributed", _FakeDist(), raising=False)

    accelerator = SimpleNamespace(
        num_processes=2,
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=2)),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", tokenizer="tok", accelerator=accelerator),
        scoring=SimpleNamespace(
            batching=SimpleNamespace(prompt_length_cache_get=None),
            weighting=SimpleNamespace(),
            reference_logprobs_source="auto",
        ),
        generation=SimpleNamespace(max_completion_len=4),
    )
    gen_batch = SimpleNamespace(grouped_completions=[["a", "b"]])
    reward_comp = SimpleNamespace(
        ref_logprob_meta=[{"meta": 1}, {"meta": 2}],
        pairs=SimpleNamespace(prompts=["p"], completions=["c1", "c2"]),
    )
    score_batch = SimpleNamespace(
        total_sequences=2,
        prompt_entries=[SimpleNamespace(length=2)],
        max_prompt_len=4,
        slice_size=2,
        completion_ids=None,
        completion_attention_mask=None,
        pad_token_id=0,
        score_tail_tokens=None,
    )

    ref_stats = SimpleNamespace(
        ref_logp_sum_raw=[-1.0, -1.0],
        ref_logp_sum=[-1.0, -1.0],
        ref_tok_counts=[1.0, 1.0],
        avg_completion_tokens=1.0,
    )
    gather_calls = {"count": 0}

    monkeypatch.setattr(
        pipeline_mod, "build_score_batch", lambda *_a, **_k: score_batch
    )
    monkeypatch.setattr(
        pipeline_mod, "_reference_stats_from_meta", lambda *_a, **_k: ref_stats
    )

    def _fake_gather(*_a, **_k):
        gather_calls["count"] += 1
        return None

    def _fake_weight_stats(_grouped, _reward, candidate_ref, _cfg):
        assert candidate_ref is ref_stats
        return SimpleNamespace(flat_weights=[1.0], weight_entropy=0.0)

    monkeypatch.setattr(pipeline_mod, "gather_reference_logprobs", _fake_gather)
    monkeypatch.setattr(pipeline_mod, "compute_weight_stats", _fake_weight_stats)
    monkeypatch.setattr(
        pipeline_mod,
        "summarize_completion_lengths",
        lambda *_a, **_k: (None, SimpleNamespace(), 2.0),
    )

    result = pipeline_mod._collect_batch_stats(ctx, gen_batch, reward_comp)
    assert result is not None
    assert gather_calls["count"] == 1
    assert result.ref_stats is ref_stats
