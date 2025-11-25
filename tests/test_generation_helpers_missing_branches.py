"""
Exercise additional branches in training.generation.helpers.
"""

from __future__ import annotations

from types import SimpleNamespace


from maxent_grpo.training.generation import helpers
from maxent_grpo.training.run_helpers import GenerationPenaltyConfig, VLLMClientConfig


def _make_generator(**overrides):
    vllm_cfg = overrides.get(
        "vllm",
        VLLMClientConfig(
            url="http://host",
            rounds_cfg=1,
            retry_sleep=0.0,
            backfill_local=False,
            request_logprobs=False,
        ),
    )
    penalty = overrides.get("penalty", GenerationPenaltyConfig())
    ctx = helpers.GenerationContext(
        max_prompt_len=overrides.get("max_prompt_len", 2),
        max_completion_len=overrides.get("max_completion_len", 4),
        gen_temperature=0.1,
        gen_top_p=0.9,
        use_vllm=overrides.get("use_vllm", False),
        vllm=vllm_cfg,
        accelerator=SimpleNamespace(
            num_processes=1,
            is_main_process=True,
            unwrap_model=lambda m: m,
            autocast=None,
            process_index=0,
        ),
        model=SimpleNamespace(generate=lambda **_k: [[0, 1]]),
        tokenizer=overrides.get(
            "tokenizer",
            SimpleNamespace(
                decode=lambda ids, skip_special_tokens=True: "".join(
                    str(i) for i in ids
                )
            ),
        ),
        generation_stats=overrides.get("generation_stats", {}),
        device=SimpleNamespace(type="cpu"),
        penalty=penalty,
    )
    gen = helpers.CompletionGenerator(ctx)
    if "helper" in overrides:
        gen._vllm_helper = overrides["helper"]
    return gen


def test_vllm_helper_passthroughs(monkeypatch):
    class _Helper:
        def __init__(self):
            self.synced = False
            self.synced_model = None

        _fsdp_cls = object()

        def maybe_sync_weights(self):
            self.synced = True

        def _sync_model_params_to_vllm(self, model):
            self.synced_model = model

        def _vllm_base_url(self, url):
            return url.rstrip("/generate")

    helper = _Helper()
    gen = _make_generator(helper=helper)
    gen._maybe_sync_vllm_weights()
    assert helper.synced is True
    gen._sync_model_params_to_vllm("model", SimpleNamespace())
    assert helper.synced_model == "model"
    assert gen._vllm_base_url("http://x/generate") == "http://x"
    assert gen._fsdp_cls is helper._fsdp_cls


def test_prompt_char_limit_handles_defaults(monkeypatch):
    gen = _make_generator(max_prompt_len=0)
    monkeypatch.setattr(helpers, "PROMPT_CHAR_LIMIT", -1)
    assert gen._prompt_char_limit() == 0
    monkeypatch.setattr(helpers, "PROMPT_CHAR_LIMIT", 4)
    gen.ctx.max_prompt_len = 1
    assert gen._prompt_char_limit() == 4


def test_generate_local_with_counts(monkeypatch):
    called = {}

    class _Tok:
        class _Enc(dict):
            def __init__(self, prompts):
                mask = _Mask(prompts)
                super().__init__(attention_mask=mask)
                self.prompts = prompts

            def to(self, device):
                self["device"] = device
                return self

        def __call__(self, prompts, **_k):
            called["prompts"] = prompts
            return self._Enc(prompts)

        def decode(self, ids, skip_special_tokens=True):
            return "|".join(str(i) for i in ids)

    class _Mask:
        def __init__(self, prompts):
            self.prompts = prompts

        def sum(self, dim):
            assert dim == 1
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [len(p) for p in self.prompts]

    gen = _make_generator(tokenizer=_Tok())
    gen.ctx.model = SimpleNamespace(generate=lambda **_k: [[0, 1], [0, 2], [0, 3]])
    grouped, meta = gen._generate_local(["a", "bb"], 1, per_prompt_counts=[1, 2])
    assert len(grouped[0]) == 1 and len(grouped[1]) == 2
    assert meta is None


def test_build_vllm_request_kwargs_defaults():
    gen = _make_generator(use_vllm=True)
    kwargs = gen._build_vllm_request_kwargs(["p"], 2)
    assert kwargs["url"] == gen.ctx.vllm_url
    assert kwargs["max_tokens"] == gen.ctx.max_completion_len
    assert kwargs["top_k"] is None
    assert kwargs["best_of"] is None


def test_merge_group_chunk_truncates():
    chunk = [["a"], ["b", "c"]]
    meta_chunk = [[1], [2, 3]]
    merged, merged_meta = helpers.CompletionGenerator._merge_group_chunk(
        chunk, meta_chunk, requested_n=1
    )
    assert merged == ["a"]
    assert merged_meta == [1]


def test_expand_dedup_results_with_mapping():
    gen = _make_generator()
    grouped = [["x"], ["y"]]
    meta = [[None], [1]]
    expanded, expanded_meta = gen._vllm_helper._expand_dedup_results(
        grouped, meta, [1, 0, 1]
    )
    assert expanded == [["y"], ["x"], ["y"]]
    assert expanded_meta == [[1], [None], [1]]
