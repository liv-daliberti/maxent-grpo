"""
Additional edge coverage for core generation helpers.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import maxent_grpo.training.generation.helpers as helpers


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


def test_flatten_ref_metadata_returns_none_when_meta_empty():
    grouped = [["c1", "c2"]]
    assert helpers.flatten_ref_metadata(grouped, [[]]) is None
    assert helpers.flatten_ref_metadata(grouped, None) is None


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
