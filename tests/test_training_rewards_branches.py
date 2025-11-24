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

Additional branch coverage for :mod:`maxent_grpo.training.rewards`.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


class _FakeTensor:
    def __init__(self, data, dtype=None, device=None):
        self.data = list(data)
        self.dtype = dtype
        self.device = device

    def mean(self):
        return SimpleNamespace(item=lambda: sum(self.data) / len(self.data))

    def std(self, unbiased=False):
        mean = sum(self.data) / len(self.data)
        var = sum((v - mean) ** 2 for v in self.data) / len(self.data)
        return SimpleNamespace(item=lambda: var**0.5)

    def numel(self):
        return len(self.data)

    def tolist(self):
        return list(self.data)


@pytest.fixture()
def rewards_mod(monkeypatch):
    """Reload rewards with a lightweight torch stub."""
    import maxent_grpo.training.run_helpers as run_helpers
    torch_stub = SimpleNamespace(
        tensor=lambda data, dtype=None, device=None: _FakeTensor(
            data, dtype=dtype, device=device
        ),
        float32="float32",
        int64=int,
        long=int,
        device=lambda kind="cpu": SimpleNamespace(type=kind),
    )
    monkeypatch.setattr(run_helpers, "require_torch", lambda *_args, **_kwargs: torch_stub)
    sys.modules.pop("maxent_grpo.training.rewards", None)
    mod = importlib.import_module("maxent_grpo.training.rewards")
    yield mod


def test_reward_moments_and_weighted_totals(rewards_mod):
    spec = SimpleNamespace(
        reward_funcs=[lambda comps, ans: [1.0, 2.0]],
        reward_weights=[2.0],
    )
    totals, per_reward = rewards_mod.compute_reward_totals(
        spec, ["a", "b"], ["x", "y"]
    )
    assert totals == [2.0, 4.0]
    assert per_reward["reward_0"] == [1.0, 2.0]

    device = SimpleNamespace(type="cuda")
    mean, std = rewards_mod.reward_moments([1.0, 3.0], device)
    assert pytest.approx(mean) == 2.0
    assert pytest.approx(std) == 1.0


def test_prepare_generation_batch_branches(rewards_mod, monkeypatch):
    # Empty prompts and None generation short-circuit
    assert (
        rewards_mod.prepare_generation_batch(
            {"prompt": [], "answer": []}, lambda *_a, **_k: None, {}, 1
        )
        is None
    )
    assert (
        rewards_mod.prepare_generation_batch(
            {"prompt": ["p"], "answer": ["a"]}, lambda *_a, **_k: None, {}, 1
        )
        is None
    )

    # Generator returning GenerationBatch follows the dataclass branch
    gb = rewards_mod.GenerationBatch(
        prompts=["p1"],
        answers=["a1"],
        grouped_completions=[["c1"]],
        grouped_ref_meta=None,
    )
    monkeypatch.setattr(
        rewards_mod, "AggregatedGenerationState", lambda comps, meta: SimpleNamespace(completions=comps, metadata=meta)
    )
    monkeypatch.setattr(
        rewards_mod, "retry_incomplete_prompts", lambda *args, **kwargs: args[3]
    )
    monkeypatch.setattr(
        rewards_mod,
        "seed_generation_groups",
        lambda prompt_count, comps, meta: (comps, meta),
    )
    monkeypatch.setattr(
        rewards_mod,
        "drop_empty_prompt_groups",
        lambda prompts, answers, comps, meta, stats: (prompts, answers, comps, meta),
    )
    monkeypatch.setattr(
        rewards_mod,
        "truncate_to_expected_counts",
        lambda comps, meta, expected: (comps, meta, 0),
    )
    out = rewards_mod.prepare_generation_batch(
        {"prompt": ["p1"], "answer": ["a1"]},
        lambda *_args: gb,
        {},
        1,
    )
    assert isinstance(out, rewards_mod.GenerationBatch)

    # Structured object with grouped_completions attribute uses the hasattr branch
    out2 = rewards_mod.prepare_generation_batch(
        {"prompt": ["p2"], "answer": ["a2"]},
        lambda *_a: SimpleNamespace(grouped_completions=[["c2"]], grouped_ref_meta=None),
        {},
        1,
    )
    assert isinstance(out2, rewards_mod.GenerationBatch)

    # Dropping all completions returns None
    monkeypatch.setattr(
        rewards_mod,
        "drop_empty_prompt_groups",
        lambda prompts, answers, comps, meta, stats: (prompts, answers, [], []),
    )
    assert (
        rewards_mod.prepare_generation_batch(
            {"prompt": ["p3"], "answer": ["a3"]},
            lambda *_a: SimpleNamespace(grouped_completions=[["c3"]], grouped_ref_meta=None),
            {},
            1,
        )
        is None
    )


def test_group_q_and_compute_reward_statistics_guards(rewards_mod, monkeypatch):
    q_grouped, q_samples = rewards_mod._group_q_distribution([[], []], [], 1.0, 1e-6)
    assert q_grouped == [[], []] and q_samples == []

    empty_batch = rewards_mod.GenerationBatch(
        prompts=[],
        answers=[],
        grouped_completions=[],
        grouped_ref_meta=None,
    )
    assert (
        rewards_mod.compute_reward_statistics(
            empty_batch, SimpleNamespace(), SimpleNamespace(type="cpu"), 1.0, 1e-6
        )
        is None
    )

    nonempty = rewards_mod.GenerationBatch(
        prompts=["p"],
        answers=["a"],
        grouped_completions=[["c"]],
        grouped_ref_meta=None,
    )
    monkeypatch.setattr(
        rewards_mod,
        "_flatten_prompt_completions",
        lambda _gb: (SimpleNamespace(completions=[]), []),
    )
    assert (
        rewards_mod.compute_reward_statistics(
            nonempty, SimpleNamespace(), SimpleNamespace(type="cpu"), 1.0, 1e-6
        )
        is None
    )
