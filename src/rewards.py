"""
Reward functions and a small name→callable registry.

This module focuses on simple, dependency‑light rewards for GRPO and
MaxEnt‑GRPO experiments. At the moment it implements a strict correctness
reward for math problems where completions are formatted as
``<think>...</think><answer>...</answer>``. The registry helper returns a list
of functions matching names provided in script/config arguments.

Key functions
- ``pure_accuracy_reward_math``: Binary exact‑match on canonicalized
  ``<answer>`` vs. gold.
- ``get_reward_funcs``: Resolve a list of reward callables from names.

License
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the specific language governing permissions and
limitations under the License.
"""

# See the module docstring above for a quick overview and current contents.

from __future__ import annotations
# coding=utf-8

from typing import (
    Any, Callable, Dict, List, Optional, Union,
    Protocol, runtime_checkable,
)
import re
from re import Pattern as RePattern
import transformers

RewardFunction = Callable[[str, str], float]


# -------------------------------
# Core reward implementations
# -------------------------------

_format_pat: RePattern[str] = re.compile(r"(?si)<think>.*?</think>.*?<answer>.*?</answer>")
_answer_pat: RePattern[str] = re.compile(r"(?si)<answer>\s*(.*?)\s*</answer>")


CompletionType = Union[str, List[Dict[str, str]], Dict[str, Any]]

def _extract_content(comp: CompletionType) -> str:
    """Extract assistant text from common completion shapes.

    Accepts a variety of shapes typically returned by APIs, e.g. a bare string
    or a list with an object containing a ``content`` field.

    :param comp: Completion object or string to normalize.
    :type comp: Any
    :returns: Extracted text content (may be empty).
    :rtype: str
    """
    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        return str(comp[0].get("content", ""))
    return str(comp)


def _canon_math(s: str) -> str:
    """Canonicalize simple math answers for exact‑match comparison.

    Heuristics remove superficial wrappers like braces/parentheses, spaces, and
    normalize signed zeros and integer forms (e.g. ``3.0`` → ``3``).

    :param s: Raw answer string.
    :type s: str
    :returns: Canonicalized answer string.
    :rtype: str
    """
    if s is None:
        return ""
    s = s.strip()
    # Drop surrounding braces if they wrap a single expression
    if re.fullmatch(r"\{[^{}]+\}", s):
        s = s[1:-1]
    # Drop surrounding parentheses if they enclose only a simple numeric/root expression
    if re.fullmatch(r"\([^()]+\)", s):
        s_inner = s[1:-1].strip()
        if re.match(r"^[\d\.\-\\sqrt]+$", s_inner):
            s = s_inner
    # Remove spaces
    s = s.replace(" ", "")
    # Normalize signed zero
    if s in ("-0", "+0"):
        s = "0"
    # Remove trailing .0 from integers
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def pure_accuracy_reward_math(
    completions: List[CompletionType],
    answer: List[str],
    **_kwargs: Any
) -> List[float]:
    """Binary reward for exact match on a tagged math template.

    Expects completions formatted with ``<think>...</think><answer>...</answer>``.
    Extracts the ``<answer>`` payload and computes an exact match against the
    canonicalized gold ``answer`` list.

    :param completions: Generated completions (strings or provider objects).
    :type completions: list[Any]
    :param answer: Gold answers aligned with ``completions``.
    :type answer: list[str]
    :returns: Per‑completion rewards in {0.0, 1.0}.
    :rtype: list[float]
    """
    outs: List[float] = []
    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)
        if not _format_pat.match(txt):
            outs.append(0.0)
            continue
        m = _answer_pat.search(txt)
        if not m:
            outs.append(0.0)
            continue
        pred = m.group(1)
        ok = (_canon_math(pred) == _canon_math(gold))
        outs.append(1.0 if ok else 0.0)
    return outs


@runtime_checkable
class RewardConfig(Protocol):
    """Protocol for objects with reward function configuration."""
    reward_funcs: List[str]

def get_reward_funcs(
    script_args: RewardConfig,
    _ref_model: Optional[transformers.PreTrainedModel] = None,
    _tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
) -> List[RewardFunction]:
    """Resolve reward function callables from names.

    :param script_args: Script/config args providing ``reward_funcs`` names.
    :type script_args: Any
    :param _ref_model: Optional reference model (unused placeholder).
    :type _ref_model: transformers.PreTrainedModel | None
    :param _tokenizer: Optional tokenizer (unused placeholder).
    :type _tokenizer: transformers.PreTrainedTokenizerBase | None
    :returns: List of reward callables.
    :rtype: list[Callable]
    :raises KeyError: If an unknown reward name is requested.
    """
    registry = {
        "pure_accuracy_math": pure_accuracy_reward_math,
    }
    return [registry[name] for name in script_args.reward_funcs]
