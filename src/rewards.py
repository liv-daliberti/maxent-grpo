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

"""
Reward helpers for lightweight GRPO/MaxEnt-GRPO experiments.

The module intentionally keeps dependencies small so that reward shaping can be
tested quickly in standalone unit tests or notebooks.  Most consumers interact
with two entry points:

``pure_accuracy_reward_math``
    Implements a binary reward that evaluates math completions formatted as
    ``<think>...</think><answer>...</answer>``.  The helper trims decorator
    tokens, canonicalizes simple numeric expressions, and emits ``1.0`` for
    exact matches and ``0.0`` otherwise.

``get_reward_funcs``
    Resolve a list of reward callables from configuration supplied through CLI
    arguments or recipe YAML files.  The registry is intentionally tiny; it is
    expected that research experiments will register their own functions in the
    future.

Examples
--------

Creating a reward function from a training config::

    >>> from rewards import get_reward_funcs
    >>> cfg = SimpleNamespace(reward_funcs=[\"pure_accuracy_math\"])
    >>> reward_fn = get_reward_funcs(cfg)[0]
    >>> completions = [\"<think/> <answer>42</answer>\"]
    >>> reward_fn(completions, [\"42\"])
    [1.0]

The module is licensed under the Apache License, Version 2.0.  See the
repository's LICENSE file for the full text.
"""

# See the module docstring above for a quick overview and current contents.

from __future__ import annotations
# coding=utf-8

from typing import Any, Callable, Dict, List, Optional, Union, Protocol, runtime_checkable
import re
from re import Pattern as RePattern
import transformers

# Reward functions accept a single completion string and a reference answer.
RewardFunction = Callable[[str, str], float]


# -------------------------------
# Core reward implementations
# -------------------------------

_format_pat: RePattern[str] = re.compile(r"(?si)<think>.*?</think>.*?<answer>.*?</answer>")
_answer_pat: RePattern[str] = re.compile(r"(?si)<answer>\s*(.*?)\s*</answer>")


CompletionType = Union[str, List[Dict[str, str]], Dict[str, Any]]

def _extract_content(comp: CompletionType) -> str:
    """Extract assistant text from common completion shapes.

    Parameters
    ----------
    comp:
        Completion object or string to normalize.  The helper understands the
        shapes returned by most hosted inference APIs, including bare strings
        and OpenAI-style ``[{\"content\": ...}]`` payloads.

    Returns
    -------
    str
        Extracted text content (may be empty when no recognized payload is
        present).
    """
    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        return str(comp[0].get("content", ""))
    return str(comp)


def _canon_math(s: str) -> str:
    """Canonicalize simple math answers for exact-match comparison.

    The helper aggressively removes superficial wrappers such as braces,
    parentheses around lone numbers, whitespace, signed zeros, and trailing
    ``.0`` patterns.  The goal is to normalize cosmetic variance without
    attempting to symbolically simplify expressions.

    Parameters
    ----------
    s:
        Raw answer string extracted from a completion.

    Returns
    -------
    str
        Canonicalized answer string suitable for exact-match comparison.
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

    Parameters
    ----------
    completions:
        Generated completions (strings or provider objects).  Each entry must
        contain ``<think>``/``<answer>`` tags for the reward to consider it a
        valid candidate.
    answer:
        Gold answers aligned with ``completions``.
    **_kwargs:
        Unused keyword arguments; accepted for compatibility with more complex
        registry signatures.

    Returns
    -------
    list[float]
        Per-completion rewards in ``{0.0, 1.0}``.
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
    """Protocol describing the reward configuration consumed by :func:`get_reward_funcs`.

    Attributes
    ----------
    reward_funcs:
        List of string identifiers that should be resolved to callables via the
        registry.
    """
    reward_funcs: List[str]

def get_reward_funcs(
    script_args: RewardConfig,
    _ref_model: Optional[transformers.PreTrainedModel] = None,
    _tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
) -> List[RewardFunction]:
    """Resolve reward function callables from names.

    Parameters
    ----------
    script_args:
        Script/config args providing ``reward_funcs`` names.  The object must
        satisfy :class:`RewardConfig`.
    _ref_model:
        Optional reference model.  Present for forward compatibility with
        reward functions that may need to score relative to a baseline model.
    _tokenizer:
        Optional tokenizer that may be consumed by more advanced rewards.

    Returns
    -------
    list[RewardFunction]
        Resolved callables ready to be invoked by the training loop.

    Raises
    ------
    KeyError
        If an unknown reward name is requested.
    """
    registry = {
        "pure_accuracy_math": pure_accuracy_reward_math,
    }
    return [registry[name] for name in script_args.reward_funcs]
