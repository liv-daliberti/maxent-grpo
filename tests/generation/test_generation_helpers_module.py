"""Minimal tests for distilabel pipeline helper."""

from __future__ import annotations

import types
import sys

import pytest

import maxent_grpo.generation.helpers as helpers


def test_build_distilabel_pipeline_raises_without_dep(monkeypatch):
    # Ensure distilabel import fails, triggering the runtime error
    monkeypatch.setitem(sys.modules, "distilabel", None)
    cfg = helpers.DistilabelPipelineConfig(model="m")
    with pytest.raises(RuntimeError):
        helpers.build_distilabel_pipeline(cfg)


def test_build_distilabel_pipeline_uses_kwargs_defaults(monkeypatch):
    # Provide a fake distilabel with expected attributes to exercise happy path
    holder = {}

    class _PipelineCtx:
        def __init__(self):
            self.added = None
            self.ran = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_step(self, step):
            self.added = step

        def run(self):
            self.ran = True

    class _Pipeline:
        def ray(self):
            ctx = _PipelineCtx()
            holder["ctx"] = ctx
            return ctx

    class _StepResources:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _TextGeneration:
        def __init__(self, *args, **kwargs):
            self.llm = kwargs.get("llm") if kwargs else None
            self.generation_kwargs = kwargs.get("generation_kwargs")
            self.kwargs = kwargs
            holder["created"] = self

    class _LLM:
        def __init__(
            self,
            base_url=None,
            api_key=None,
            model=None,
            timeout=None,
            max_retries=None,
            generation_kwargs=None,
            **_k,
        ):
            self.base_url = base_url
            self.timeout = timeout
            self.retries = max_retries

        def with_options(self, **kwargs):
            return types.SimpleNamespace(options=kwargs)

    fake_distilabel = types.SimpleNamespace(
        pipeline=types.SimpleNamespace(Pipeline=_Pipeline),
        steps=types.SimpleNamespace(
            StepResources=_StepResources,
            tasks=types.SimpleNamespace(TextGeneration=_TextGeneration),
        ),
        llms=types.SimpleNamespace(OpenAILLM=_LLM),
    )
    monkeypatch.setitem(sys.modules, "distilabel", fake_distilabel)

    cfg = helpers.DistilabelPipelineConfig(
        model="m",
        base_url="http://host",
        prompt_column="prompt",
        prompt_template="{{ instruction }}",
        temperature=0.5,
        top_p=0.8,
        max_new_tokens=10,
        num_generations=2,
        input_batch_size=3,
        client_replicas=1,
        timeout=5,
        retries=2,
    )
    helpers.build_distilabel_pipeline(cfg)
    created = holder.get("created")
    assert created is not None and isinstance(created.llm, _LLM)
    assert created.llm.base_url == "http://host"
