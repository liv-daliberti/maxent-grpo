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

Unit tests for training.rewards edge cases and branches.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is required for rewards unit tests", allow_module_level=True)


class _FakeTensor:
    def __init__(self, data, device=None):
        self._data = list(data)
        self.device = device

    def mean(self):
        return SimpleNamespace(item=lambda: sum(self._data) / len(self._data))

    def std(self, unbiased=False):
        mean = sum(self._data) / len(self._data)
        var = sum((v - mean) ** 2 for v in self._data) / len(self._data)
        return SimpleNamespace(item=lambda: var**0.5)

    def numel(self):
        return len(self._data)


class _FakeDevice:
    def __init__(self, kind: str):
        self.type = kind


@pytest.fixture()
def rewards_mod(monkeypatch):
    """Reload training.rewards while stubbing heavyweight dependency gates."""
    torch_mod = importlib.import_module("torch")
    monkeypatch.setattr(
        "training.run_helpers.require_torch", lambda *_args, **_kwargs: torch_mod
    )
    monkeypatch.setattr(
        "training.run_helpers.require_dataloader",
        lambda *_a, **_k: SimpleNamespace(DataLoader=object),
    )
    monkeypatch.setattr(
        "training.run_helpers.require_accelerator",
        lambda *_a, **_k: SimpleNamespace(Accelerator=object),
    )
    mod = importlib.reload(importlib.import_module("training.rewards"))
    yield mod
    importlib.reload(importlib.import_module("training.run_helpers"))
    importlib.reload(importlib.import_module("training.rewards"))


def test_compute_reward_totals_applies_weights(rewards_mod):
    reward_spec = SimpleNamespace(
        reward_funcs=[lambda comps, ans: [1.0, 2.0]],
        reward_weights=[0.5],
    )
    totals, per_reward = rewards_mod.compute_reward_totals(
        reward_spec, ["a", "b"], ["x", "y"]
    )
    assert totals == [0.5, 1.0]
    assert per_reward["reward_0"] == [1.0, 2.0]


def test_reward_moments_handles_empty_and_nonempty(rewards_mod):
    device = _FakeDevice("cpu")
    mean, std = rewards_mod.reward_moments([], device)
    assert mean == 0.0 and std == 0.0

    mean, std = rewards_mod.reward_moments([1.0, 3.0], device)
    assert pytest.approx(mean) == 2.0
    assert pytest.approx(std) == pytest.approx(1.0)


def test_reward_moments_single_entry_zero_std(rewards_mod):
    mean, std = rewards_mod.reward_moments([5.0], _FakeDevice("cpu"))
    assert mean == 5.0 and std == 0.0


def test_reward_moments_tensor_fallback(monkeypatch, rewards_mod):
    original_sum = builtins.sum
    monkeypatch.setattr(
        builtins,
        "sum",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad sum")),
    )

    class _FakeTensor:
        def __init__(self, data, dtype=None, device=None):
            self.data = list(data)
            self.dtype = dtype
            self.device = device
            self._value = None

        def mean(self, *a, **k):
            self._value = original_sum(self.data) / len(self.data)
            return self

        def std(self, unbiased=False):
            self.unbiased = unbiased
            mean_val = original_sum(self.data) / len(self.data)
            var = original_sum((v - mean_val) ** 2 for v in self.data) / len(self.data)
            self._value = var**0.5
            return self

        def item(self):
            return self._value

        def numel(self):
            return len(self.data)

    class _FakeTorch:
        float32 = "f32"

        def tensor(self, data, dtype=None, device=None):
            return _FakeTensor(data, dtype=dtype, device=device)

        @staticmethod
        def device(kind):
            return f"dev:{kind}"

    fake_tensors = []

    class _FakeTorch:
        float32 = "f32"

        def tensor(self, data, dtype=None, device=None):
            tensor = _FakeTensor(data, dtype=dtype, device=device)
            fake_tensors.append(tensor)
            return tensor

        @staticmethod
        def device(kind):
            return f"dev:{kind}"

    fake_torch = _FakeTorch()
    monkeypatch.setattr(rewards_mod, "require_torch", lambda *_a, **_k: fake_torch)
    mean, std = rewards_mod.reward_moments([2.0, 6.0], _FakeDevice("cuda"))
    assert mean == 4.0
    assert std == pytest.approx(2.0)
    assert fake_tensors[0].dtype == "f32"
    assert fake_tensors[0].device.type == "cuda"
    assert getattr(fake_tensors[0], "unbiased", None) is False


def test_reward_moments_tensor_fallback_single_sample(monkeypatch, rewards_mod):
    original_sum = builtins.sum
    monkeypatch.setattr(
        builtins,
        "sum",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("sum fail")),
    )

    class _FakeTensor:
        def __init__(self, data, dtype=None, device=None):
            self.data = list(data)
            self.dtype = dtype
            self.device = device
            self._value = None

        def mean(self, *a, **k):
            self._value = original_sum(self.data) / len(self.data)
            return self

        def std(self, unbiased=False):
            self.unbiased = unbiased
            self._value = 0.0
            return self

        def item(self):
            return self._value

        def numel(self):
            return len(self.data)

    class _FakeTorch:
        float32 = "f32"

        def tensor(self, data, dtype=None, device=None):
            return _FakeTensor(data, dtype=dtype, device=device)

        @staticmethod
        def device(kind):
            return f"dev:{kind}"

    monkeypatch.setattr(rewards_mod, "require_torch", lambda *_a, **_k: _FakeTorch())
    mean, std = rewards_mod.reward_moments([10.0], _FakeDevice("cpu"))
    assert mean == 10.0 and std == 0.0


def test_group_advantages_handles_empty_groups(rewards_mod):
    grouped, flat = rewards_mod.group_advantages([[], ["a"]], [5.0, 7.0])
    assert grouped[0] == []
    assert grouped[1] == [0.0]  # single element baseline subtraction
    assert flat == [0.0]


def test_prepare_generation_batch_branches(rewards_mod, monkeypatch):
    # Empty prompts short-circuit
    assert (
        rewards_mod.prepare_generation_batch(
            {"prompt": [], "answer": []}, lambda *_a, **_k: None, {}, 1
        )
        is None
    )

    # Generator returns None
    assert (
        rewards_mod.prepare_generation_batch(
            {"prompt": ["p"], "answer": ["a"]}, lambda *_a, **_k: None, {}, 1
        )
        is None
    )

    calls = {}

    def _seed(prompt_count, comps, meta):
        calls["seed"] = True
        return comps, meta

    class _AggState:
        def __init__(self, comps, meta):
            self.completions = comps
            self.metadata = meta

    monkeypatch.setattr(rewards_mod, "seed_generation_groups", _seed)
    monkeypatch.setattr(
        rewards_mod,
        "AggregatedGenerationState",
        lambda comps, meta: _AggState(comps, meta),
    )
    monkeypatch.setattr(
        rewards_mod,
        "retry_incomplete_prompts",
        lambda *args, **kwargs: _AggState(args[3].completions, args[3].metadata),
    )

    # Drop everything to trigger the empty guard
    monkeypatch.setattr(
        rewards_mod,
        "drop_empty_prompt_groups",
        lambda prompts, answers, comps, meta, stats: (prompts, answers, [], []),
    )
    assert (
        rewards_mod.prepare_generation_batch(
            {"prompt": ["p"], "answer": ["a"]},
            lambda p, n: SimpleNamespace(
                grouped_completions=[["ok"]], grouped_ref_meta=None
            ),
            {},
            1,
        )
        is None
    )

    # Keep completions and register partial counts
    monkeypatch.setattr(
        rewards_mod,
        "drop_empty_prompt_groups",
        lambda prompts, answers, comps, meta, stats: (prompts, answers, comps, meta),
    )
    monkeypatch.setattr(
        rewards_mod,
        "truncate_to_expected_counts",
        lambda comps, meta, expected: (comps, meta, 2),
    )

    stats = {}
    batch = rewards_mod.prepare_generation_batch(
        {"prompt": ["p"], "answer": ["a"]},
        lambda p, n: SimpleNamespace(
            grouped_completions=[["ok"]], grouped_ref_meta=None
        ),
        stats,
        1,
    )
    assert isinstance(batch, rewards_mod.GenerationBatch)
    assert stats["partial_prompts"] == 2
    assert calls["seed"] is True


def test_group_q_distribution_handles_empty_groups(rewards_mod):
    q_grouped, q_samples = rewards_mod._group_q_distribution([[], []], [], 1.0, 1e-6)
    assert q_grouped == [[], []]
    assert q_samples == []


def test_compute_reward_statistics_empty_guards(rewards_mod, monkeypatch):
    empty_batch = rewards_mod.GenerationBatch(
        prompts=[],
        answers=[],
        grouped_completions=[],
        grouped_ref_meta=None,
    )
    assert (
        rewards_mod.compute_reward_statistics(
            empty_batch, SimpleNamespace(), _FakeDevice("cpu"), 1.0, 1e-6
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
            nonempty, SimpleNamespace(), _FakeDevice("cpu"), 1.0, 1e-6
        )
        is None
    )
