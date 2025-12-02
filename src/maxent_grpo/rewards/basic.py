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

Baseline-friendly reward registry used by GRPO training and inference.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
    Union,
    runtime_checkable,
)
import re
from re import Pattern as RePattern

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Reward functions accept a single completion string and a reference answer.
RewardFunction = Callable[[str, str], float]

# -------------------------------
# Core reward implementations
# -------------------------------

_format_pat: RePattern[str] = re.compile(
    r"(?si)<think>.*?</think>.*?<answer>.*?</answer>"
)
_answer_pat: RePattern[str] = re.compile(r"(?si)<answer>\s*(.*?)\s*</answer>")


CompletionType = Union[str, List[Dict[str, str]], Dict[str, Any]]


def _extract_content(comp: CompletionType) -> str:
    """Extract assistant text from common completion shapes."""

    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        return str(comp[0].get("content", ""))
    return str(comp)


def _canon_math(s: str) -> str:
    """Canonicalize simple math answers for exact-match comparison."""

    if s is None:
        return ""
    s = s.strip()
    if re.fullmatch(r"\{[^{}]+\}", s):
        s = s[1:-1]
    if re.fullmatch(r"\([^()]+\)", s):
        s_inner = s[1:-1].strip()
        if re.match(r"^[\d\.\-\\sqrt]+$", s_inner):
            s = s_inner
    s = s.replace(" ", "")
    if s in ("-0", "+0"):
        s = "0"
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def pure_accuracy_reward_math(
    completions: List[CompletionType], answer: List[str], **_kwargs: Any
) -> List[float]:
    """Binary reward for exact match on a tagged math template.

    Strict formatting (``<think>…</think><answer>…</answer>``) is enforced by
    default. When called with ``is_eval=True`` or ``relaxed_format=True``,
    the ``<think>`` block is optional and only the ``<answer>`` tag is
    required. This keeps training strict while avoiding format false-negatives
    during eval.
    """

    outs: List[float] = []
    relaxed = bool(
        _kwargs.get("is_eval")
        or _kwargs.get("relaxed_format")
        or _kwargs.get("split") in {"eval", "validation", "test"}
    )
    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)
        if not relaxed and not _format_pat.match(txt):
            outs.append(0.0)
            continue
        m = _answer_pat.search(txt)
        if not m:
            outs.append(0.0)
            continue
        pred = m.group(1)
        ok = _canon_math(pred) == _canon_math(gold)
        outs.append(1.0 if ok else 0.0)
    return outs


@runtime_checkable
class RewardConfig(Protocol):
    """Minimal protocol describing the reward configuration interface."""

    reward_funcs: List[str]


def get_reward_funcs(
    script_args: RewardConfig,
    _ref_model: Optional[PreTrainedModel] = None,
    _tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> List[RewardFunction]:
    """Resolve reward function callables from names."""

    registry = {
        "pure_accuracy_math": pure_accuracy_reward_math,
    }
    return [registry[name] for name in script_args.reward_funcs]


__all__ = [
    "RewardFunction",
    "RewardConfig",
    "get_reward_funcs",
    "pure_accuracy_reward_math",
]
