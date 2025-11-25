"""
Additional edge coverage for core generation helpers.
"""

from __future__ import annotations

from argparse import Namespace
from types import ModuleType, SimpleNamespace
import importlib
import sys

import pytest

import maxent_grpo.generation.helpers as helpers


def test_append_completion_group_extends_meta_with_nones():
    grouped_comps = [["a"]]
    grouped_meta = [["m1"]]
    updated_meta = helpers.append_completion_group(
        grouped_comps, grouped_meta, 0, ["b", "c"], None
    )
    assert grouped_comps[0] == ["a", "b", "c"]
    assert updated_meta[0][-2:] == [None, None]


def test_truncate_to_expected_counts_returns_early_when_disabled():
    comps = [["a", "b"]]
    meta = [["m1", "m2"]]
    trimmed, trimmed_meta, partial = helpers.truncate_to_expected_counts(comps, meta, 0)
    assert trimmed == comps and trimmed_meta == meta
    assert partial == 0


def test_flatten_ref_metadata_handles_payload_and_type_error():
    class _MetaGood:
        def __init__(self, value):
            self.value = value

        def to_trl_payload(self):
            return {"v": self.value}

    class _MetaBad:
        def to_trl_payload(self):
            raise TypeError("nope")

    grouped = [["c1"], ["c2", "c3"]]
    meta = [[_MetaGood("ok")], [_MetaBad(), None]]
    flat = helpers.flatten_ref_metadata(grouped, meta)
    assert flat[0] == {"v": "ok"}
    assert isinstance(flat[1], _MetaBad)
    assert flat[2] is None


def test_flatten_prompt_completions_returns_empty_when_no_pairs(monkeypatch):
    pc_calls = {}

    class _PCB:
        def __init__(self, prompts, completions):
            pc_calls["prompts"] = prompts
            pc_calls["completions"] = completions

    mod = ModuleType("maxent_grpo.training.types")
    mod.PromptCompletionBatch = _PCB
    mod.__spec__ = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "maxent_grpo.training.types", mod)
    gen_batch = SimpleNamespace(prompts=[], answers=[], grouped_completions=[])
    batch, answers = helpers.flatten_prompt_completions(gen_batch)
    assert pc_calls["prompts"] == []
    assert answers == []
    assert isinstance(batch, _PCB)


def test_build_distilabel_pipeline_uses_stubbed_modules(monkeypatch):
    captured = {}

    class _Pipeline:
        def __init__(self):
            captured["pipeline_created"] = True

        def ray(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _StepResources:
        def __init__(self, replicas):
            captured["replicas"] = replicas

    class _TextGeneration:
        def __init__(
            self,
            llm,
            template,
            input_mappings,
            input_batch_size,
            num_generations,
            group_generations,
            resources,
        ):
            captured["llm"] = llm
            captured["template"] = template
            captured["input_mappings"] = input_mappings
            captured["input_batch_size"] = input_batch_size
            captured["num_generations"] = num_generations
            captured["group_generations"] = group_generations
            captured["resources"] = resources

    def _openai_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return kwargs

    dist_mod = ModuleType("distilabel")
    dist_mod.pipeline = SimpleNamespace(Pipeline=_Pipeline)
    dist_mod.steps = SimpleNamespace(
        StepResources=_StepResources,
        tasks=SimpleNamespace(TextGeneration=_TextGeneration),
    )
    dist_mod.llms = SimpleNamespace(OpenAILLM=_openai_llm)
    monkeypatch.setitem(sys.modules, "distilabel", dist_mod)

    cfg = helpers.DistilabelPipelineConfig(
        model="model-x",
        base_url="http://base",
        prompt_column="instr",
        prompt_template="T:{{ instruction }}",
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=10,
        num_generations=2,
        input_batch_size=3,
        client_replicas=4,
        timeout=50,
        retries=5,
    )
    pipe = helpers.build_distilabel_pipeline(cfg)
    assert isinstance(pipe, _Pipeline)
    assert captured["llm_kwargs"]["model"] == "model-x"
    assert captured["template"] == cfg.prompt_template
    assert captured["input_mappings"] == {"instruction": "instr"}
    assert captured["input_batch_size"] == cfg.input_batch_size
    assert captured["num_generations"] == cfg.num_generations
    assert captured["group_generations"] is True
    assert captured["replicas"] == cfg.client_replicas


def test_build_distilabel_pipeline_raises_when_import_fails(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(RuntimeError):
        helpers.build_distilabel_pipeline(helpers.DistilabelPipelineConfig(model="m"))


def test_run_distilabel_cli_pushes_and_raises_on_missing_push(monkeypatch):
    # Install datasets stub
    datasets_mod = ModuleType("datasets")
    datasets_mod.__spec__ = SimpleNamespace()
    datasets_mod.load_dataset = lambda *a, **k: ["data"]
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    class _Pipeline:
        def __init__(self):
            self.run_calls = []

        def run(self, dataset, dataset_batch_size, use_cache):
            self.run_calls.append((dataset, dataset_batch_size, use_cache))
            return _DistisetWithPush()

    class _DistisetWithPush:
        def __init__(self):
            self.pushed = []

        def push_to_hub(self, name, private):
            self.pushed.append((name, private))

    built = {}

    def _builder(cfg):
        built["cfg"] = cfg
        return _Pipeline()

    args = Namespace(
        hf_dataset="ds",
        hf_dataset_config="cfg",
        hf_dataset_split="train",
        model="m",
        vllm_server_url="http://server",
        prompt_template="{{ instruction }}",
        prompt_column=None,
        temperature=0.1,
        top_p=0.8,
        max_new_tokens=5,
        num_generations=2,
        input_batch_size=1,
        client_replicas=1,
        timeout=30,
        retries=0,
        hf_output_dataset="out",
        private=True,
    )
    helpers.run_distilabel_cli(args, pipeline_builder=_builder)
    assert isinstance(built["cfg"], helpers.DistilabelPipelineConfig)

    class _PipelineNoPush:
        def run(self, *_a, **_k):
            return SimpleNamespace()

    with pytest.raises(RuntimeError):
        helpers.run_distilabel_cli(
            args, pipeline_builder=lambda _cfg: _PipelineNoPush()
        )
