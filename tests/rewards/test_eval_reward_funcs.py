from types import SimpleNamespace

import pytest

import maxent_grpo.training.rewards as rewards_mod
from maxent_grpo.training.rewards import (
    load_eval_reward_functions,
    load_reward_functions,
)


def test_load_reward_functions_prefers_script_args():
    script_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[0.25]
    )
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[1.0]
    )

    funcs, weights = load_reward_functions(script_args, None, training_args)

    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [0.25]


def test_load_reward_functions_falls_back_to_training_when_script_empty():
    script_args = SimpleNamespace(reward_funcs=[], reward_weights=None)
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[0.75]
    )

    funcs, weights = load_reward_functions(script_args, None, training_args)

    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [0.75]


def test_load_reward_functions_handles_single_string():
    script_args = SimpleNamespace(reward_funcs="pure_accuracy_math", reward_weights=None)
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[0.9]
    )

    funcs, weights = load_reward_functions(script_args, None, training_args)

    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [1.0]


def test_load_reward_functions_prefers_training_when_script_not_recipe(monkeypatch):
    calls = []

    def _fake_get_reward_funcs(proxy, *_args, **_kwargs):
        names = list(getattr(proxy, "reward_funcs", []))
        calls.append(names)
        return [lambda *_, _name=name, **__: _name for name in names]

    monkeypatch.setattr(rewards_mod, "get_reward_funcs", _fake_get_reward_funcs)
    script_args = SimpleNamespace(reward_funcs=["script_reward"])
    training_args = SimpleNamespace(reward_funcs=["train_reward"])

    funcs, weights = load_reward_functions(script_args, None, training_args)

    assert calls[-1] == ["train_reward"]
    assert [fn(None) for fn in funcs] == ["train_reward"]
    assert weights == [1.0]


def test_load_reward_functions_keeps_recipe_rewards(monkeypatch):
    calls = []

    def _fake_get_reward_funcs(proxy, *_args, **_kwargs):
        names = list(getattr(proxy, "reward_funcs", []))
        calls.append(names)
        return [lambda *_, _name=name, **__: _name for name in names]

    monkeypatch.setattr(rewards_mod, "get_reward_funcs", _fake_get_reward_funcs)
    script_args = SimpleNamespace(
        reward_funcs=["script_reward"], recipe_path="configs/recipe.yaml"
    )
    training_args = SimpleNamespace(reward_funcs=["script_reward"])

    funcs, weights = load_reward_functions(script_args, None, training_args)

    assert calls[-1] == ["script_reward"]
    assert [fn(None) for fn in funcs] == ["script_reward"]
    assert weights == [1.0]


def test_load_reward_functions_respects_training_override_with_recipe(monkeypatch):
    calls = []

    def _fake_get_reward_funcs(proxy, *_args, **_kwargs):
        names = list(getattr(proxy, "reward_funcs", []))
        calls.append(names)
        return [lambda *_, _name=name, **__: _name for name in names]

    monkeypatch.setattr(rewards_mod, "get_reward_funcs", _fake_get_reward_funcs)
    script_args = SimpleNamespace(
        reward_funcs=["recipe_reward"], recipe_path="configs/recipe.yaml"
    )
    training_args = SimpleNamespace(reward_funcs=["train_reward"])

    funcs, weights = load_reward_functions(script_args, None, training_args)
    assert calls[-1] == ["train_reward"]
    assert [fn(None) for fn in funcs] == ["train_reward"]
    assert weights == [1.0]


def test_eval_reward_funcs_override_and_fallback():
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[1.0]
    )
    script_args = SimpleNamespace(
        eval_reward_funcs=["pure_accuracy_math"],
        eval_reward_weights=[0.5],
        reward_funcs=[],
    )

    funcs, weights = load_eval_reward_functions(script_args, None, training_args)
    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [0.5]


def test_eval_reward_funcs_fallback_to_script_rewards_when_eval_missing():
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[1.0]
    )
    script_args = SimpleNamespace(
        eval_reward_funcs=[],
        eval_reward_weights=None,
        reward_funcs=["pure_accuracy_math"],
        reward_weights=[0.3],
    )

    funcs, weights = load_eval_reward_functions(script_args, None, training_args)
    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [0.3]


def test_eval_reward_funcs_fallback_to_training_when_script_missing():
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[0.8]
    )
    script_args = SimpleNamespace(
        eval_reward_funcs=[], eval_reward_weights=None, reward_funcs=[]
    )

    funcs, weights = load_eval_reward_functions(script_args, None, training_args)
    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [0.8]


def test_eval_reward_funcs_handle_single_string():
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=[0.8]
    )
    script_args = SimpleNamespace(
        eval_reward_funcs="pure_accuracy_math", eval_reward_weights=None
    )

    funcs, weights = load_eval_reward_functions(script_args, None, training_args)
    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [1.0]


def test_eval_reward_funcs_use_training_when_no_recipe(monkeypatch):
    calls = []

    def _fake_get_reward_funcs(proxy, *_args, **_kwargs):
        names = list(getattr(proxy, "reward_funcs", []))
        calls.append(names)
        return [lambda *_, _name=name, **__: _name for name in names]

    monkeypatch.setattr(rewards_mod, "get_reward_funcs", _fake_get_reward_funcs)
    script_args = SimpleNamespace(
        eval_reward_funcs=[],
        reward_funcs=["script_reward"],
    )
    training_args = SimpleNamespace(reward_funcs=["train_reward"])

    funcs, weights = load_eval_reward_functions(script_args, None, training_args)
    assert calls[-1] == ["train_reward"]
    assert [fn(None) for fn in funcs] == ["train_reward"]
    assert weights == [1.0]


def test_eval_reward_funcs_honor_recipe_overrides(monkeypatch):
    calls = []

    def _fake_get_reward_funcs(proxy, *_args, **_kwargs):
        names = list(getattr(proxy, "reward_funcs", []))
        calls.append(names)
        return [lambda *_, _name=name, **__: _name for name in names]

    monkeypatch.setattr(rewards_mod, "get_reward_funcs", _fake_get_reward_funcs)
    script_args = SimpleNamespace(
        reward_funcs=["script_reward"], recipe_path="configs/recipe.yaml"
    )
    training_args = SimpleNamespace(reward_funcs=["script_reward"])

    funcs, weights = load_eval_reward_functions(script_args, None, training_args)
    assert calls[-1] == ["script_reward"]
    assert [fn(None) for fn in funcs] == ["script_reward"]
    assert weights == [1.0]


def test_eval_reward_funcs_respect_training_override_with_recipe(monkeypatch):
    calls = []

    def _fake_get_reward_funcs(proxy, *_args, **_kwargs):
        names = list(getattr(proxy, "reward_funcs", []))
        calls.append(names)
        return [lambda *_, _name=name, **__: _name for name in names]

    monkeypatch.setattr(rewards_mod, "get_reward_funcs", _fake_get_reward_funcs)
    script_args = SimpleNamespace(
        eval_reward_funcs=[],
        reward_funcs=["recipe_reward"],
        recipe_path="configs/recipe.yaml",
    )
    training_args = SimpleNamespace(reward_funcs=["train_reward"])

    funcs, weights = load_eval_reward_functions(script_args, None, training_args)
    assert calls[-1] == ["train_reward"]
    assert [fn(None) for fn in funcs] == ["train_reward"]
    assert weights == [1.0]
