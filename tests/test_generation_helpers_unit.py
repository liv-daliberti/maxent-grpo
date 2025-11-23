"""Unit tests for lightweight helpers in ``generation.helpers``."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional
import sys

import pytest

from generation import helpers


def test_append_completion_group_extends_existing_meta_when_missing_values():
    grouped = [["base"]]
    grouped_meta = [["orig"]]
    updated = helpers.append_completion_group(
        grouped,
        grouped_meta,
        prompt_idx=0,
        completions=["extra1", "extra2"],
        meta_group=None,
    )
    assert grouped[0] == ["base", "extra1", "extra2"]
    assert updated == [["orig", None, None]]


def test_append_completion_group_expands_short_metadata_lists():
    grouped = [["seed"]]
    updated = helpers.append_completion_group(
        grouped,
        grouped_meta=None,
        prompt_idx=0,
        completions=["c1", "c2"],
        meta_group=["only_one"],
    )
    assert updated == [[None, "only_one", None]]


def test_pending_generation_indices_skip_nonpositive_expected():
    assert helpers.pending_generation_indices([["a"]], expected_generations=0) == []


def test_determine_retry_limit_prefers_override_and_default():
    assert (
        helpers.determine_retry_limit(expected_generations=2, max_retry_rounds=5) == 5
    )
    assert (
        helpers.determine_retry_limit(expected_generations=0, max_retry_rounds=None)
        == helpers._DEFAULT_RETRY_LIMIT  # type: ignore[attr-defined]
    )


def test_truncate_to_expected_counts_noop_when_expected_nonpositive():
    comps = [["a", "b"]]
    trimmed_comps, trimmed_meta, partial = helpers.truncate_to_expected_counts(
        comps, None, expected_generations=0
    )
    assert trimmed_comps == comps
    assert trimmed_meta is None
    assert partial == 0


def test_flatten_ref_metadata_handles_none_and_payload_errors():
    assert helpers.flatten_ref_metadata([["a"]], None) is None

    class _Payload:
        def __init__(self, value: str) -> None:
            self.value = value

        def to_trl_payload(self) -> str:
            raise TypeError("bad payload")

    flat = helpers.flatten_ref_metadata([["a"]], [[_Payload("x")]])
    assert len(flat) == 1
    assert isinstance(flat[0], _Payload)
    assert flat[0].value == "x"


def test_flatten_prompt_completions_returns_empty_when_min_len_zero(monkeypatch):
    @dataclass
    class _PromptCompletionBatch:
        prompts: List[str]
        completions: List[str]

    training_types = ModuleType("training.types")
    training_types.PromptCompletionBatch = _PromptCompletionBatch
    monkeypatch.setitem(sys.modules, "training.types", training_types)
    batch = SimpleNamespace(prompts=[], answers=[], grouped_completions=[])
    prompt_batch, answers = helpers.flatten_prompt_completions(batch)
    assert prompt_batch.prompts == []
    assert answers == []


def _install_distilabel_stub(monkeypatch):
    distilabel_mod = ModuleType("distilabel")

    class _Pipeline:
        def __init__(self) -> None:
            self.resources = None

        class _Ctx:
            def __init__(self, pipe):
                self._pipe = pipe

            def __enter__(self):
                return self._pipe

            def __exit__(self, exc_type, exc, tb):
                return False

        def ray(self):
            return self._Ctx(self)

    class _StepResources:
        def __init__(self, replicas: int):
            self.replicas = replicas

    class _TextGeneration:
        last_kwargs: Optional[dict[str, Any]] = None

        def __init__(self, **kwargs):
            _TextGeneration.last_kwargs = kwargs

    class _OpenAILLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    distilabel_mod.pipeline = SimpleNamespace(Pipeline=_Pipeline)
    distilabel_mod.steps = SimpleNamespace(
        StepResources=_StepResources,
        tasks=SimpleNamespace(TextGeneration=_TextGeneration),
    )
    distilabel_mod.llms = SimpleNamespace(OpenAILLM=_OpenAILLM)
    monkeypatch.setitem(sys.modules, "distilabel", distilabel_mod)
    return distilabel_mod


def test_build_distilabel_pipeline_applies_sampling_knobs(monkeypatch):
    distilabel_mod = _install_distilabel_stub(monkeypatch)
    cfg = helpers.DistilabelPipelineConfig(
        model="demo",
        prompt_column="instruction",
        temperature=0.2,
        top_p=0.9,
    )
    pipe = helpers.build_distilabel_pipeline(cfg)
    assert isinstance(pipe, distilabel_mod.pipeline.Pipeline)
    llm_kwargs = distilabel_mod.steps.tasks.TextGeneration.last_kwargs["llm"].kwargs
    assert llm_kwargs["generation_kwargs"]["temperature"] == 0.2
    assert llm_kwargs["generation_kwargs"]["top_p"] == 0.9


def test_run_distilabel_cli_raises_when_push_missing(monkeypatch):
    dataset = object()
    datasets_mod = ModuleType("datasets")
    datasets_mod.load_dataset = lambda *_args, **_kwargs: dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    class _Pipeline:
        def run(self, **_kwargs):
            return SimpleNamespace()

    args = Namespace(
        hf_dataset="org/repo",
        hf_dataset_config=None,
        hf_dataset_split="train",
        prompt_column="prompt",
        prompt_template="{{ instruction }}",
        model="demo",
        vllm_server_url="http://localhost",
        temperature=None,
        top_p=None,
        max_new_tokens=16,
        num_generations=1,
        input_batch_size=2,
        client_replicas=1,
        timeout=10,
        retries=0,
        hf_output_dataset="dest/repo",
        private=False,
    )

    with pytest.raises(RuntimeError, match="does not expose push_to_hub"):
        helpers.run_distilabel_cli(args, pipeline_builder=lambda _cfg: _Pipeline())
