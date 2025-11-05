# coding=utf-8

"""Reward registry with only pure_accuracy_math."""

from typing import Callable, List
import transformers

from .rewards_core import pure_accuracy_reward_math


def get_reward_funcs(
    script_args,
    ref_model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> List[Callable]:
    registry = {
        "pure_accuracy_math": pure_accuracy_reward_math,
    }
    return [registry[name] for name in script_args.reward_funcs]

