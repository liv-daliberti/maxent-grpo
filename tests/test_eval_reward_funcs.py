from types import SimpleNamespace

from maxent_grpo.training.rewards import (
    load_eval_reward_functions,
    load_reward_functions,
)


def test_load_reward_functions_prefers_training_args():
    script_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=None
    )
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=None
    )

    funcs, weights = load_reward_functions(script_args, None, training_args)

    assert len(funcs) == 1
    assert callable(funcs[0])
    assert weights == [1.0]


def test_eval_reward_funcs_override_and_fallback():
    training_args = SimpleNamespace(
        reward_funcs=["pure_accuracy_math"], reward_weights=None
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

    script_args_empty = SimpleNamespace(
        eval_reward_funcs=[], eval_reward_weights=None, reward_funcs=[]
    )
    funcs_fallback, weights_fallback = load_eval_reward_functions(
        script_args_empty, None, training_args
    )
    assert len(funcs_fallback) == 1
    assert weights_fallback == [1.0]
