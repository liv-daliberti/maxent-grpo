"""
Tests for the distilabel helpers in generation.helpers.
"""

from types import ModuleType, SimpleNamespace
import sys

import maxent_grpo.generation.helpers as helpers


def _install_distilabel_stub(monkeypatch):
    """Register a minimal distilabel stub that records constructor calls."""
    call_log = {}

    class _OpenAILLM:
        def __init__(self, **kwargs):
            call_log["llm_kwargs"] = dict(kwargs)

    class _StepResources:
        def __init__(self, **kwargs):
            call_log["resources_kwargs"] = dict(kwargs)

    class _TextGeneration:
        def __init__(self, **kwargs):
            call_log["text_kwargs"] = dict(kwargs)

    class _Pipeline:
        def __init__(self):
            self.ray_called = False

        def ray(self):
            self.ray_called = True
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    disti = ModuleType("distilabel")
    disti.pipeline = SimpleNamespace(Pipeline=_Pipeline)
    disti.steps = SimpleNamespace(
        tasks=SimpleNamespace(TextGeneration=_TextGeneration),
        StepResources=_StepResources,
    )
    disti.llms = SimpleNamespace(OpenAILLM=_OpenAILLM)
    # Ensure importlib finds the stub
    monkeypatch.setitem(sys.modules, "distilabel", disti)
    monkeypatch.setitem(sys.modules, "distilabel.pipeline", disti.pipeline)
    monkeypatch.setitem(sys.modules, "distilabel.steps", disti.steps)
    monkeypatch.setitem(sys.modules, "distilabel.steps.tasks", disti.steps.tasks)
    monkeypatch.setitem(sys.modules, "distilabel.llms", disti.llms)
    return call_log, _Pipeline


def test_build_distilabel_pipeline_with_kwargs(monkeypatch):
    """cfg=None path should build config from kwargs and wire LLM/TextGeneration."""
    # Remove any cached distilabel modules first
    for mod in list(sys.modules):
        if mod.startswith("distilabel"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    call_log, pipeline_cls = _install_distilabel_stub(monkeypatch)

    pipe = helpers.build_distilabel_pipeline(
        model="stub-model",
        base_url="http://stub",
        prompt_column="question",
        prompt_template="{{ instruction }}",
        temperature=0.4,
        top_p=0.8,
        max_new_tokens=32,
        num_generations=2,
        input_batch_size=5,
        client_replicas=3,
        timeout=10,
        retries=2,
    )

    assert isinstance(pipe, pipeline_cls)
    assert pipe.ray_called is True

    llm_kwargs = call_log["llm_kwargs"]
    assert llm_kwargs["base_url"] == "http://stub"
    assert llm_kwargs["model"] == "stub-model"
    assert llm_kwargs["generation_kwargs"]["max_new_tokens"] == 32
    assert llm_kwargs["generation_kwargs"]["temperature"] == 0.4
    assert llm_kwargs["generation_kwargs"]["top_p"] == 0.8

    text_kwargs = call_log["text_kwargs"]
    assert text_kwargs["template"] == "{{ instruction }}"
    assert text_kwargs["input_batch_size"] == 5
    assert text_kwargs["num_generations"] == 2
    assert text_kwargs["group_generations"] is True
    assert text_kwargs["input_mappings"] == {"instruction": "question"}

    resources_kwargs = call_log["resources_kwargs"]
    assert resources_kwargs == {"replicas": 3}
