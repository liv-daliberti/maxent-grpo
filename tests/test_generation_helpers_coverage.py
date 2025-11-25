"""Extra coverage for generation helpers fallback paths."""

from __future__ import annotations

from types import SimpleNamespace


import maxent_grpo.training.generation.helpers as helpers
from maxent_grpo.training.run_helpers import GenerationPenaltyConfig


def _make_gen(**overrides) -> helpers.CompletionGenerator:
    return helpers.CompletionGenerator(
        helpers.GenerationContext(
            max_prompt_len=overrides.get("max_prompt_len", 8),
            max_completion_len=4,
            gen_temperature=0.5,
            gen_top_p=0.9,
            use_vllm=overrides.get("use_vllm", False),
            vllm=overrides.get(
                "vllm",
                SimpleNamespace(
                    url="http://vllm",
                    rounds_cfg=1,
                    retry_sleep=0.0,
                    backfill_local=False,
                    request_logprobs=False,
                    sync_weights=False,
                ),
            ),
            accelerator=overrides.get("accelerator", SimpleNamespace()),
            model=overrides.get("model", SimpleNamespace()),
            tokenizer=overrides.get("tokenizer", SimpleNamespace()),
            generation_stats=overrides.get("generation_stats", {}),
            device=overrides.get("device", "cpu"),
            penalty=overrides.get("penalty", GenerationPenaltyConfig()),
        )
    )


def test_describe_delegates_to_context_dict():
    gen = _make_gen(device="cuda:0")
    desc = gen.describe()
    assert desc["device"] == "cuda:0"
    assert "max_prompt_len" in desc


def test_tokenize_expanded_prompts_fallback_path():
    gen = _make_gen(tokenizer=SimpleNamespace())  # non-callable -> fallback branch
    inputs, lengths = gen._tokenize_expanded_prompts(["abc", "de"])
    assert lengths == [3, 2]
    assert inputs["attention_mask"].tolist() == lengths


def test_run_local_model_uses_fallback_generate(monkeypatch):
    # Model without generate should hit the fallback path.
    tok = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: "|".join(str(i) for i in ids)
    )

    class _Accel:
        def unwrap_model(self, model):
            return model

    gen = _make_gen(
        accelerator=_Accel(),
        model=SimpleNamespace(),  # lacks generate
        tokenizer=tok,
    )
    sequences, lengths = [["x", "y"], ["a"]], [1, 1]
    out = gen._run_local_model(sequences, lengths)
    assert out == ["y", ""]


def test_run_vllm_rounds_replaces_hooks(monkeypatch):
    gen = _make_gen(use_vllm=True, generation_stats={"vllm_retry_rounds": 0})
    helper = gen._vllm_helper
    # Reset to base implementations so replacement logic runs.
    orig_exec = helpers.VLLMGenerationHelper._execute_vllm_request.__get__(
        helper, helpers.VLLMGenerationHelper
    )
    orig_batch = helpers.VLLMGenerationHelper._request_vllm_batch.__get__(
        helper, helpers.VLLMGenerationHelper
    )
    helper._execute_vllm_request = orig_exec
    helper._request_vllm_batch = orig_batch
    markers = {}
    monkeypatch.setattr(
        helper,
        "_run_vllm_rounds",
        lambda _state: markers.setdefault(
            "replaced",
            (
                helper._execute_vllm_request is not orig_exec,
                helper._request_vllm_batch is not orig_batch,
                helper._fallback_generate is gen._generate_local,
            ),
        ),
    )
    state = helpers._VLLMGenerationState(
        prompts=["p"],
        target_counts=[0],
        requested_n=1,
        round_limit=0,
        track_logprobs=False,
    )
    gen._run_vllm_rounds(state)
    assert helper._execute_vllm_request is not orig_exec
    assert helper._request_vllm_batch is not orig_batch
    assert callable(helper._fallback_generate)
    assert getattr(helper._fallback_generate, "__self__", None) is gen


def test_generate_with_vllm_calls_helper(monkeypatch):
    gen = _make_gen(use_vllm=True)
    captured = {}

    def _fake_generate(
        prompts, num_samples, counts, ensure_client=None, sync_model=None
    ):
        captured["args"] = (
            list(prompts),
            num_samples,
            counts,
            ensure_client,
            sync_model,
        )
        return [["ok"]], None

    monkeypatch.setattr(gen._vllm_helper, "generate", _fake_generate)
    grouped, meta = gen._generate_with_vllm(["p1"], 2, [2])
    assert grouped == [["ok"]] and meta is None
    assert captured["args"][0] == ["p1"]
    assert callable(captured["args"][3]) and callable(captured["args"][4])


def test_generate_vllm_collective_distributed_path(monkeypatch):
    class _Accel:
        def __init__(self):
            self.num_processes = 2
            self.process_index = 0
            self.is_main_process = True

    gen = _make_gen(use_vllm=True, accelerator=_Accel())
    monkeypatch.setattr(
        gen,
        "_flatten_prompts_for_broadcast",
        lambda prompts, counts: (["p0", "p1"], [0, 1], counts),
    )
    monkeypatch.setattr(
        gen,
        "_generate_with_vllm",
        lambda prompts, num_samples, counts: ([["g0"], ["g1"]], [["m0"], ["m1"]]),
    )
    monkeypatch.setattr(
        gen,
        "_scatter_vllm_payload",
        lambda flat_prompts, offsets, grouped_all, meta_all: (grouped_all, meta_all),
    )
    grouped, meta = gen._generate_vllm_collective(["p0", "p1"], 1, [1, 1])
    assert grouped == [["g0"], ["g1"]] and meta == [["m0"], ["m1"]]


def test_generate_falls_back_to_local_when_disabled(monkeypatch):
    gen = _make_gen(use_vllm=False)
    monkeypatch.setattr(
        gen, "_generate_local", lambda prompts, n, counts: (["local"], None)
    )
    grouped, meta = gen.generate(["p"], 1)
    assert grouped == ["local"] and meta is None


def test_scatter_object_fallbacks():
    accel_single = SimpleNamespace(num_processes=1, process_index=0)
    assert helpers._scatter_object(accel_single, None) is None
    assert helpers._scatter_object(accel_single, ["x"]) == "x"

    accel_multi = SimpleNamespace(num_processes=2, process_index=1, scatter_object=None)
    # No dist backend in tests; should pick index from input_list.
    assert helpers._scatter_object(accel_multi, ["r0", "r1"]) == "r1"
