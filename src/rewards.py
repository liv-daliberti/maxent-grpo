# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

"""Reward registry with only pure_accuracy_math."""

from typing import Callable, List
import transformers

from rewards_core import pure_accuracy_reward_math


def get_reward_funcs(
    script_args,
    ref_model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> List[Callable]:
    registry = {
        "pure_accuracy_math": pure_accuracy_reward_math,
    }
    return [registry[name] for name in script_args.reward_funcs]
