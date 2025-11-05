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

from __future__ import annotations

import re
from typing import Any, List

_format_pat = re.compile(r"(?si)<think>.*?</think>.*?<answer>.*?</answer>")
_answer_pat = re.compile(r"(?si)<answer>\s*(.*?)\s*</answer>")


def _extract_content(comp: Any) -> str:
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
    completions: List[Any],
    answer: List[str],
    **kw,
) -> List[float]:
    """Binary reward: exact match of canonicalized math answers inside the tag template."""
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
