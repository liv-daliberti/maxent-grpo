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
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxent_grpo.training import rewards as rw
from maxent_grpo.training.types import GenerationBatch, RewardComputation, RewardSpec


def test_reward_moments_handles_empty():
    mean, std = rw.reward_moments([], SimpleNamespace(type="cpu"))
    assert mean == 0.0 and std == 0.0


def test_group_q_distribution_handles_empty_group(monkeypatch):
    monkeypatch.setattr(
        rw,
        "_group_softmax",
        lambda vals, temperature, eps: [1.0 / len(vals) for _ in vals] if vals else [],
    )
    q_grouped, q_samples = rw._group_q_distribution([[], ["a"]], [0.0, 1.0], 1.0, 1e-6)
    assert q_grouped[0] == []
    assert sum(q_grouped[1]) == pytest.approx(1.0)
    assert len(q_samples) == 1  # only non-empty groups contribute samples


def test_prepare_generation_batch_retries_and_tracks_partials():
    calls = {"rounds": []}

    def _generator(prompts, expected, missing=None):
        calls["rounds"].append((list(prompts), expected, missing))
        if missing is None:
            # Return one completion per prompt.
            return [[f"{p}-0"] for p in prompts], [[None] for _ in prompts]
        # Retry path: fill one completion per pending prompt.
        return [[f"{p}-retry"] for p in prompts], [[None] for _ in prompts]

    batch = {"prompt": ["p1", "p2"], "answer": ["a1", "a2"]}
    stats: dict[str, int] = {}
    out = rw.prepare_generation_batch(batch, _generator, stats, expected_generations=2)
    assert isinstance(out, GenerationBatch)
    assert len(out.grouped_completions) == 2
    # No partials because retries filled in missing completions.
    assert stats.get("partial_prompts", 0) == 0
    # Ensure at least one retry round occurred.
    assert len(calls["rounds"]) >= 2


def test_compute_reward_statistics_none_on_empty_group():
    gen_batch = GenerationBatch(prompts=[], answers=[], grouped_completions=[], grouped_ref_meta=None)
    reward_spec = RewardSpec(reward_funcs=[], reward_weights=[])
    assert rw.compute_reward_statistics(gen_batch, reward_spec, SimpleNamespace(type="cpu"), 1.0, 1e-6) is None


def test_compute_reward_statistics_produces_components(monkeypatch):
    def _reward_fn(completions, answers, **_):
        return [1.0 if c == a else 0.0 for c, a in zip(completions, answers)]

    class _FakeTensor(list):
        def mean(self):
            return _FakeTensor([sum(self) / len(self)])

        def std(self, unbiased=False):
            mean = sum(self) / len(self)
            var = sum((x - mean) ** 2 for x in self) / len(self)
            return _FakeTensor([var**0.5])

        def item(self):
            return float(self[0])

        def numel(self):
            return len(self)

    class _FakeTorch:
        float32 = "f32"

        @staticmethod
        def tensor(vals, dtype=None, device=None):
            return _FakeTensor(vals)

        @staticmethod
        def device(_val="cpu"):
            return SimpleNamespace(type="cpu")

    monkeypatch.setattr(rw, "torch", _FakeTorch())
    monkeypatch.setattr(
        rw,
        "_group_softmax",
        lambda vals, temperature, eps: [1.0 / len(vals) for _ in vals],
    )

    gen_batch = GenerationBatch(
        prompts=["p"],
        answers=["c1"],
        grouped_completions=[["c1", "c2"]],
        grouped_ref_meta=None,
    )
    reward_spec = RewardSpec(reward_funcs=[_reward_fn], reward_weights=[1.0])
    comp = rw.compute_reward_statistics(
        gen_batch,
        reward_spec,
        SimpleNamespace(type="cpu"),
        q_temperature=1.0,
        q_epsilon=1e-6,
    )
    assert isinstance(comp, RewardComputation)
    assert comp.total_utils and comp.per_reward_values
    assert len(comp.q_distribution.samples) == len(comp.total_utils)
