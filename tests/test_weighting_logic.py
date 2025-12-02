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

Unit tests for training.weighting.logic helpers.
"""

from __future__ import annotations

import json
import math
from importlib import import_module, reload
from types import SimpleNamespace

import numpy as np
import pytest

from maxent_grpo.training.types import (
    AdvantageStats,
    PromptCompletionBatch,
    QDistribution,
    ReferenceLogprobs,
    RewardComputation,
    RewardMoments,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightStats,
    WeightingSettings,
)


class _TorchTensor:
    """Minimal tensor stub with the ops exercised by weighting.logic."""

    def __init__(self, data, dtype=None):
        self.arr = np.array(data, dtype=dtype)

    def clamp(self, min=None, max=None):
        return _TorchTensor(np.clip(self.arr, a_min=min, a_max=max))

    def log(self):
        return _TorchTensor(np.log(self.arr))

    def __add__(self, other):
        other_arr = other.arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(self.arr + other_arr)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_arr = other.arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(self.arr - other_arr)

    def __rsub__(self, other):
        other_arr = other.arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(other_arr - self.arr)

    def __mul__(self, other):
        other_arr = other.arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(self.arr * other_arr)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_arr = other.arr if isinstance(other, _TorchTensor) else other
        return _TorchTensor(self.arr / other_arr)

    def __neg__(self):
        return _TorchTensor(-self.arr)

    def sum(self, dim=None):
        return _TorchTensor(self.arr.sum(axis=dim))

    def item(self):
        return float(np.array(self.arr).item())

    def tolist(self):
        return self.arr.tolist()

    def __iter__(self):
        if self.arr.shape == ():
            return iter([self.arr.item()])
        return iter(self.arr)

    def __len__(self):
        try:
            return len(self.arr)
        except TypeError:
            return 1

    def __getitem__(self, key):
        return _TorchTensor(self.arr[key])

    @property
    def shape(self):
        return self.arr.shape


class _TorchStub:
    Tensor = _TorchTensor
    float32 = np.float32

    def tensor(self, data, dtype=None, **_kwargs):
        return _TorchTensor(data, dtype=dtype)

    def log(self, tensor):
        arr = tensor.arr if isinstance(tensor, _TorchTensor) else tensor
        return _TorchTensor(np.log(arr))

    def softmax(self, tensor, dim=0):
        arr = tensor.arr if isinstance(tensor, _TorchTensor) else np.array(tensor)
        shifted = arr - np.max(arr, axis=dim, keepdims=True)
        exps = np.exp(shifted)
        denom = np.sum(exps, axis=dim, keepdims=True)
        return _TorchTensor(exps / np.maximum(denom, 1e-12))


def _build_weighting(
    *,
    beta: float = 0.3,
    tau: float = 0.2,
    len_norm_ref: bool = True,
    denom: float | None = None,
    tau_target: float | None = None,
    tau_lr: float = 0.1,
    tau_min: float = 0.0,
    tau_max: float = 10.0,
    tau_warmup: int = 0,
    kl_target: float = 0.1,
    kl_horizon: int = 10,
    kl_step: float = 0.5,
    train_grpo_objective: bool = False,
) -> WeightingSettings:
    normalization = WeightNormalizationSettings(
        denom=(
            denom
            if denom is not None
            else (
                1.0
                if train_grpo_objective
                else (beta + tau if (beta + tau) > 0 else 1.0)
            )
        ),
        len_norm_ref=len_norm_ref,
    )
    return WeightingSettings(
        tau=tau,
        beta=beta,
        normalization=normalization,
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=tau_target,
            learning_rate=tau_lr,
            minimum_value=tau_min,
            maximum_value=tau_max,
            warmup_steps=tau_warmup,
        ),
        kl_controller=KlControllerSettings(
            target=kl_target, horizon=kl_horizon, step_size=kl_step
        ),
        train_grpo_objective=train_grpo_objective,
    )


@pytest.fixture
def weighting_logic(monkeypatch):
    """Reload the weighting.logic module with a local torch stub."""

    logic = reload(import_module("training.weighting.logic"))
    stub = _TorchStub()
    monkeypatch.setattr(logic, "torch", stub)
    return logic, stub


def test_split_reference_helpers_handle_offsets(weighting_logic):
    logic, stub = weighting_logic
    grouped = [["a", "b"], ["c"]]
    ref_stats = ReferenceLogprobs(
        ref_logp_sum=stub.tensor([0.1, 0.2, 0.3]),
        ref_tok_counts=stub.tensor([2.0, 3.0, 1.0]),
        ref_logp_sum_raw=stub.tensor([1.0, 2.0, 3.0]),
        ref_logp_mean=0.0,
        avg_completion_tokens=2.0,
    )

    assert logic.split_reference_logprobs(grouped, ref_stats, True) == [
        [0.1, 0.2],
        [0.3],
    ]
    assert logic.split_reference_logprobs(grouped, ref_stats, False) == [
        [1.0, 2.0],
        [3.0],
    ]
    assert logic.split_reference_token_counts(grouped, ref_stats) == [
        [2.0, 3.0],
        [1.0],
    ]
    per_token = logic._split_ref_logprobs_per_token(grouped, ref_stats)
    assert per_token[0] == pytest.approx([0.5, 0.6666667])
    assert per_token[1] == [3.0]


def test_weight_vector_from_q_normalizes_with_reference_and_tokens(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(beta=0.3, tau=0.2, denom=0.0)
    q_values = [0.2, 0.8]
    logp_values = [-1.0, -0.5]
    token_counts = [2.0, 1.0]

    weights = logic.weight_vector_from_q(
        q_values, logp_values, token_counts, weighting_cfg
    )

    safe_denom = 0.5  # tau + beta fallback since denom was 0
    log_terms = np.log(q_values) / safe_denom + (0.3 / safe_denom) * np.array(
        logp_values
    )
    expected = np.exp(log_terms - np.log(np.exp(log_terms).sum()))
    expected = expected * np.array(token_counts)
    expected = expected / expected.sum()
    assert weights == pytest.approx(expected.tolist())


def test_weight_vector_from_q_returns_empty_for_missing_inputs(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting()
    assert logic.weight_vector_from_q([], [1.0], [1.0], weighting_cfg) == []
    assert logic.weight_vector_from_q([0.5], [], None, weighting_cfg) == []


def test_weight_vector_from_q_uses_safe_denom_floor(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(beta=0.0, tau=0.0, denom=-1.0)
    weights = logic.weight_vector_from_q(
        [0.5, 0.5],
        [0.0, 0.0],
        None,
        weighting_cfg,
    )
    assert weights == pytest.approx([0.5, 0.5])


def test_maybe_update_beta_clips_and_updates_grpo_denom(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(
        beta=1.0,
        tau=0.0,
        kl_target=0.5,
        kl_horizon=1,
        kl_step=2.0,
        train_grpo_objective=True,
    )

    logic.maybe_update_beta(weighting_cfg, measured_kl=0.0)

    assert weighting_cfg.beta == pytest.approx(1e-6)
    assert weighting_cfg.denom == pytest.approx(weighting_cfg.beta)


def test_maybe_update_beta_ignores_invalid_inputs(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(beta=1.0, tau=0.2, kl_target=0.5, kl_step=0.5)

    logic.maybe_update_beta(weighting_cfg, measured_kl=float("nan"))
    logic.maybe_update_beta(weighting_cfg, measured_kl="bad")
    logic.maybe_update_beta(weighting_cfg, measured_kl=weighting_cfg.kl_target)

    assert weighting_cfg.beta == pytest.approx(1.0)
    assert weighting_cfg.denom == pytest.approx(weighting_cfg.beta + weighting_cfg.tau)


def test_maybe_update_tau_respects_guards(weighting_logic):
    logic, _ = weighting_logic

    no_target_cfg = _build_weighting(tau_target=None)
    logic.maybe_update_tau(
        no_target_cfg, weight_stats=SimpleNamespace(weight_entropy=1.0), global_step=5
    )
    assert no_target_cfg.tau == pytest.approx(0.2)

    warmup_cfg = _build_weighting(tau_target=0.5, tau_warmup=10)
    logic.maybe_update_tau(
        warmup_cfg, weight_stats=SimpleNamespace(weight_entropy=1.0), global_step=5
    )
    assert warmup_cfg.tau == pytest.approx(0.2)

    non_finite_cfg = _build_weighting(tau_target=0.5)
    logic.maybe_update_tau(
        non_finite_cfg,
        weight_stats=SimpleNamespace(weight_entropy=float("inf")),
        global_step=20,
    )
    assert non_finite_cfg.tau == pytest.approx(0.2)


def test_maybe_update_tau_applies_ema_and_updates_denom(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(
        tau=0.2,
        beta=0.5,
        tau_target=0.7,
        tau_lr=0.5,
        tau_min=0.1,
        tau_max=1.0,
    )
    setattr(weighting_cfg, "_tau_entropy_ema", 0.6)

    logic.maybe_update_tau(
        weighting_cfg,
        weight_stats=SimpleNamespace(weight_entropy=0.5),
        global_step=5,
    )

    assert weighting_cfg.tau > 0.2
    assert weighting_cfg.denom == pytest.approx(weighting_cfg.tau + weighting_cfg.beta)
    assert getattr(weighting_cfg, "_tau_log") == pytest.approx(
        math.log(weighting_cfg.tau)
    )


def test_maybe_update_tau_accepts_logging_view_entropy(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(
        tau=0.2, beta=0.3, tau_target=0.5, tau_lr=0.1, tau_warmup=0
    )

    logic.maybe_update_tau(
        weighting_cfg,
        weight_stats=SimpleNamespace(entropy=0.1),
        global_step=1,
    )

    assert getattr(weighting_cfg, "_tau_entropy_ema") == pytest.approx(0.1)
    assert weighting_cfg.tau != pytest.approx(0.2)


def test_maybe_update_tau_returns_on_zero_error(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(tau=0.4, beta=0.3, tau_target=1.0, tau_lr=0.2)

    logic.maybe_update_tau(
        weighting_cfg,
        weight_stats=WeightStats(
            weights_grouped=[[1.0]],
            flat_weights=[1.0],
            weight_entropy=1.0,
            weight_entropy_min=1.0,
            weight_entropy_max=1.0,
            advantage_entropy=[0.0],
        ),
        global_step=1,
    )

    assert weighting_cfg.tau == pytest.approx(0.4)
    assert weighting_cfg.denom == pytest.approx(0.7)
    assert getattr(weighting_cfg, "_tau_entropy_ema") == pytest.approx(1.0)


class _FakeAccelerator:
    """Simple broadcast stub capturing the last payload from the source rank."""

    def __init__(self, process_index: int, shared: dict):
        self.process_index = process_index
        self._shared = shared

    def broadcast_object_list(self, obj_list, src=0):
        if self.process_index == src:
            # Store a deepcopy-like copy to mimic transfer.
            self._shared["payload"] = [list(obj_list[0])]
            return obj_list
        return self._shared.get("payload", obj_list)


def test_broadcast_controller_state_syncs_across_ranks(weighting_logic):
    logic, _ = weighting_logic
    shared = {}
    accel0 = _FakeAccelerator(process_index=0, shared=shared)
    accel1 = _FakeAccelerator(process_index=1, shared=shared)
    weight_stats = SimpleNamespace(weight_entropy=0.2)
    cfg0 = _build_weighting(tau=0.2, beta=0.1, tau_target=0.5, tau_lr=0.3, tau_warmup=0)
    cfg1 = _build_weighting(tau=0.9, beta=0.6, tau_target=0.5, tau_lr=0.3, tau_warmup=0)

    logic.maybe_update_tau(cfg0, weight_stats=weight_stats, global_step=2)
    logic.broadcast_controller_state(accel0, cfg0)
    assert "payload" in shared
    source_tau = shared["payload"][0][1]
    assert source_tau != cfg1.tau
    logic.broadcast_controller_state(accel1, cfg1)

    assert cfg1.tau == pytest.approx(cfg0.tau)
    assert getattr(cfg1, "_tau_entropy_ema") == pytest.approx(
        getattr(cfg0, "_tau_entropy_ema")
    )
    assert cfg1.beta == pytest.approx(cfg0.beta)
    assert cfg1.denom == pytest.approx(cfg0.denom)


def test_maybe_update_tau_scales_learning_rate(weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(
        tau=0.2, beta=0.3, tau_target=0.8, tau_lr=0.1, tau_warmup=0
    )
    setattr(weighting_cfg, "_tau_entropy_ema", 0.1)
    logic.maybe_update_tau(
        weighting_cfg,
        weight_stats=SimpleNamespace(weight_entropy=0.1),
        global_step=1,
        lr_scale=0.5,
    )
    assert getattr(weighting_cfg, "_tau_lr_effective") == pytest.approx(0.05)
    assert weighting_cfg.tau != pytest.approx(0.2)  # update should apply


def test_controller_state_roundtrip_and_missing_tau_log(tmp_path, weighting_logic):
    logic, _ = weighting_logic
    weighting_cfg = _build_weighting(beta=0.4, tau=0.3)
    setattr(weighting_cfg, "_tau_entropy_ema", 0.25)
    setattr(weighting_cfg, "_tau_log", math.log(weighting_cfg.tau))
    path = tmp_path / "state" / "ctl.json"

    logic.save_controller_state(str(path), weighting_cfg)
    assert path.is_file()

    loaded_cfg = _build_weighting(train_grpo_objective=True)
    assert logic.load_controller_state(str(path), loaded_cfg)
    assert loaded_cfg.beta == pytest.approx(weighting_cfg.beta)
    assert loaded_cfg.tau == pytest.approx(weighting_cfg.tau)
    assert loaded_cfg.denom == pytest.approx(1.0)
    assert getattr(loaded_cfg, "_tau_log") == pytest.approx(math.log(weighting_cfg.tau))

    state = json.loads(path.read_text())
    state.pop("tau_log", None)
    path.write_text(json.dumps(state))
    fallback_cfg = _build_weighting(beta=0.1, tau=0.6)
    assert logic.load_controller_state(str(path), fallback_cfg)
    assert getattr(fallback_cfg, "_tau_log") == pytest.approx(
        math.log(fallback_cfg.tau)
    )
    assert not logic.load_controller_state(None, fallback_cfg)


def test_save_controller_state_noops_without_path(tmp_path, weighting_logic):
    logic, _ = weighting_logic
    cfg = _build_weighting()
    logic.save_controller_state(None, cfg)
    assert not list(tmp_path.iterdir())


def test_load_controller_state_handles_bad_json(tmp_path, weighting_logic):
    logic, _ = weighting_logic
    cfg = _build_weighting(beta=0.1, tau=0.2)
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{")
    assert logic.load_controller_state(str(bad_path), cfg) is False
    assert cfg.beta == pytest.approx(0.1)
    assert cfg.tau == pytest.approx(0.2)


def test_load_controller_state_rejects_invalid_types(tmp_path, weighting_logic):
    logic, _ = weighting_logic
    cfg = _build_weighting(beta=0.3, tau=0.4)
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"beta": "oops", "tau": 0.5, "tau_log": 0.0}))
    assert logic.load_controller_state(str(path), cfg) is False
    assert cfg.beta == pytest.approx(0.3)
    assert cfg.tau == pytest.approx(0.4)


def test_collect_weight_entropy_handles_empty_and_values(weighting_logic):
    logic, _ = weighting_logic

    empty_entropy = logic.collect_weight_entropy([[], []])
    assert empty_entropy == (0.0, 0.0, 0.0, [])

    mean, ent_min, ent_max, advantage = logic.collect_weight_entropy(
        [[0.25, 0.75], [1.0]]
    )
    expected_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    assert mean == pytest.approx((expected_entropy + 0.0) / 2.0)
    assert ent_min == pytest.approx(0.0)
    assert ent_max == pytest.approx(expected_entropy)
    assert advantage == pytest.approx([-0.25, 0.25, 0.0])


def test_compute_weight_stats_uses_per_token_reference(weighting_logic):
    logic, stub = weighting_logic
    grouped = [["a", "b"], ["c"]]
    ref_stats = ReferenceLogprobs(
        ref_logp_sum=stub.tensor([0.0, 0.0, 0.0]),
        ref_tok_counts=stub.tensor([2.0, 0.0, 1.0]),
        ref_logp_sum_raw=stub.tensor([2.0, -1.0, 0.0]),
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )
    reward_comp = RewardComputation(
        total_utils=[0.0, 0.0, 0.0],
        per_reward_values={},
        advantage=AdvantageStats(grouped=[[], []], samples=[]),
        pairs=PromptCompletionBatch(
            prompts=["p1", "p1", "p2"], completions=["a", "b", "c"]
        ),
        q_distribution=QDistribution(
            grouped=[[0.2, 0.8], [0.6]], samples=[0.2, 0.8, 0.6]
        ),
        moments=RewardMoments(mean=0.0, std=1.0),
    )
    weighting_cfg = _build_weighting(beta=0.4, tau=0.5, len_norm_ref=False)

    stats = logic.compute_weight_stats(grouped, reward_comp, ref_stats, weighting_cfg)

    assert stats is not None
    expected_log_terms = np.log([0.2, 0.8]) / 0.9 + (0.4 / 0.9) * np.array([1.0, -1.0])
    expected_group = np.exp(
        expected_log_terms - np.log(np.exp(expected_log_terms).sum())
    )
    expected_group = expected_group * np.array([2.0, 1.0])
    expected_group = expected_group / expected_group.sum()
    assert stats.weights_grouped[0] == pytest.approx(expected_group.tolist(), rel=1e-4)
    assert stats.weights_grouped[1] == pytest.approx([1.0])
    assert len(stats.flat_weights) == 3
    assert len(stats.advantage_entropy) == 3
    assert stats.weight_entropy_max >= stats.weight_entropy_min


def test_compute_weight_stats_returns_none_for_empty_inputs(weighting_logic):
    logic, stub = weighting_logic
    weighting_cfg = _build_weighting()
    ref_stats = ReferenceLogprobs(
        ref_logp_sum=stub.tensor([]),
        ref_tok_counts=stub.tensor([]),
        ref_logp_sum_raw=stub.tensor([]),
        ref_logp_mean=0.0,
        avg_completion_tokens=0.0,
    )
    reward_comp = RewardComputation(
        total_utils=[],
        per_reward_values={},
        advantage=AdvantageStats(grouped=[], samples=[]),
        pairs=PromptCompletionBatch(prompts=[], completions=[]),
        q_distribution=QDistribution(grouped=[], samples=[]),
        moments=RewardMoments(mean=0.0, std=0.0),
    )

    assert logic.compute_weight_stats([], reward_comp, ref_stats, weighting_cfg) is None
