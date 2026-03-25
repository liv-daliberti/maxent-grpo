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

Baseline-friendly reward registry used by GRPO training.
"""

from __future__ import annotations
# pylint: disable=broad-exception-caught

import ast
import importlib
import json
import logging
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Union,
    runtime_checkable,
)
import re
from re import Pattern as RePattern

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# -------------------------------
# Core reward implementations
# -------------------------------

_format_pat: RePattern[str] = re.compile(
    r"(?si)<think>.*?</think>.*?<answer>.*?</answer>"
)
_answer_pat: RePattern[str] = re.compile(r"(?si)<answer>\s*(.*?)\s*</answer>")
_tag_pat: RePattern[str] = re.compile(r"(?i)</?think>|</?answer>")
_boxed_pat: RePattern[str] = re.compile(r"\\(?:boxed|fbox)\s*")
_python_block_pat: RePattern[str] = re.compile(
    r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL
)
_code_block_pat: RePattern[str] = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_def_name_pat: RePattern[str] = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

_LOG = logging.getLogger(__name__)


CompletionType = Union[str, List[Dict[str, str]], Dict[str, Any]]


class RewardFunction(Protocol):
    """Protocol describing batch reward functions."""

    def __call__(
        self,
        completions: List[CompletionType],
        *args: Any,
        **kwargs: Any,
    ) -> List[float]: ...


def _extract_content(comp: CompletionType) -> str:
    """Extract assistant text from common completion shapes."""

    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        return str(comp[0].get("content", ""))
    return str(comp)


def _extract_python_code(text: str) -> str:
    """Extract a Python snippet from answer/code fences or return raw text."""

    if not text:
        return ""
    python_blocks = _python_block_pat.findall(text)
    if python_blocks:
        return str(python_blocks[-1]).strip()
    generic_blocks = _code_block_pat.findall(text)
    if generic_blocks:
        return str(generic_blocks[-1]).strip()
    answer_match = _answer_pat.search(text)
    if answer_match:
        return str(answer_match.group(1)).strip()
    return str(text).strip()


def _extract_prompt_text(prompt: Any) -> str:
    """Return user-facing prompt text from string/chat prompt shapes."""

    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() == "user":
                return str(message.get("content", ""))
        for message in reversed(prompt):
            if isinstance(message, dict) and "content" in message:
                return str(message.get("content", ""))
    if isinstance(prompt, dict):
        for key in ("prompt", "question", "text"):
            if key in prompt:
                return str(prompt.get(key, ""))
    return str(prompt) if prompt is not None else ""


def _parse_answer_payload(raw: Any) -> Any:
    """Best-effort parse for list/dict payloads serialized as strings."""

    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _normalize_text_lines(value: Any) -> List[str]:
    """Normalize strings/lists into comparable stripped lines."""

    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_normalize_text_lines(item))
        return out
    text = str(value).strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _iter_boxed_answers(text: str) -> List[tuple[str, int]]:
    """Return parsed boxed/fbox payloads and their exclusive end offsets."""

    if not text:
        return []
    matches: List[tuple[str, int]] = []
    cursor = 0
    while True:
        match = _boxed_pat.search(text, cursor)
        if match is None:
            break
        idx = match.end()
        while idx < len(text) and text[idx].isspace():
            idx += 1
        candidate: Optional[str] = None
        next_cursor = idx + 1
        end_offset = idx
        if idx < len(text) and text[idx] == "{":
            depth = 0
            end = idx
            while end < len(text):
                char = text[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx + 1 : end].strip()
                        next_cursor = end + 1
                        end_offset = end + 1
                        break
                end += 1
        else:
            end = idx
            while end < len(text) and not text[end].isspace():
                if text[end] in ",;.!?":
                    break
                end += 1
            candidate = text[idx:end].strip()
            next_cursor = max(end, idx + 1)
            end_offset = end
        if candidate:
            matches.append((candidate, end_offset))
        cursor = next_cursor
    return matches


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Return the last ``\\boxed{...}``/``\\fbox{...}`` payload when present."""

    matches = _iter_boxed_answers(text)
    if not matches:
        return None
    return matches[-1][0]


def truncate_after_first_boxed_answer(text: str) -> str:
    """Trim a completion immediately after the first valid boxed answer."""

    matches = _iter_boxed_answers(text)
    if not matches:
        return text
    _, end_offset = matches[0]
    return text[:end_offset].rstrip()


def _gold_math_candidates(gold: Any) -> List[str]:
    """Return canonical gold-answer candidates from scalar/list payloads."""

    seen: set[str] = set()
    normalized: List[str] = []
    flat_candidates: List[str] = []
    work: List[Any] = [_parse_answer_payload(gold)]
    while work:
        item = work.pop(0)
        if item is None:
            continue
        if isinstance(item, list):
            work.extend(item)
            continue
        if isinstance(item, dict):
            for key in (
                "answer",
                "answers",
                "final_answer",
                "final_answers",
                "solution",
            ):
                if key in item:
                    work.append(item.get(key))
                    break
            else:
                flat_candidates.extend(_normalize_text_lines(item))
            continue
        flat_candidates.extend(_normalize_text_lines(item))
    for candidate in flat_candidates:
        canon = _canon_math(candidate)
        if canon and canon not in seen:
            seen.add(canon)
            normalized.append(canon)
    return normalized


def _outputs_match(predicted: Any, expected: Any) -> bool:
    """Return whether predicted and expected outputs match after normalization."""

    pred_lines = _normalize_text_lines(predicted)
    exp_lines = _normalize_text_lines(expected)
    return pred_lines == exp_lines


def _extract_mbpp_tests(payload: Any) -> Optional[List[str]]:
    """Extract MBPP-style ``assert`` tests from payload when present."""

    if isinstance(payload, dict):
        if "test_list" in payload:
            return _extract_mbpp_tests(payload.get("test_list"))
        return None
    if isinstance(payload, list):
        tests = [str(item).strip() for item in payload if str(item).strip()]
        if tests and all("assert" in test for test in tests):
            return tests
        return None
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and all("assert" in line for line in lines):
            return lines
    return None


def _extract_humaneval_test(payload: Any) -> Optional[str]:
    """Extract HumanEval-style test program body containing ``check``."""

    if isinstance(payload, dict):
        test_field = payload.get("test")
        if isinstance(test_field, str) and "def check" in test_field:
            return test_field
        return None
    if isinstance(payload, str):
        text = payload.strip()
        if "def check" in text:
            return text
    return None


def _extract_apps_cases(payload: Any) -> Optional[tuple[List[Any], List[Any]]]:
    """Extract APPS-style IO pairs from an ``input_output`` payload."""

    data = payload
    if isinstance(data, dict) and "input_output" in data:
        data = data.get("input_output")
    if isinstance(data, str):
        parsed = _parse_answer_payload(data)
        data = parsed if parsed is not None else data
    if not isinstance(data, dict):
        return None
    inputs = data.get("inputs")
    outputs = data.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if not inputs or len(inputs) != len(outputs):
        return None
    return inputs, outputs


def _parse_entry_point(prompt_text: str, explicit: Optional[str]) -> Optional[str]:
    """Return function name for HumanEval checks from explicit value or prompt."""

    if explicit:
        return explicit.strip() or None
    if not prompt_text:
        return None
    match = _def_name_pat.search(prompt_text)
    if match is None:
        return None
    return str(match.group(1))


def _run_script(script: str, timeout_s: float) -> Optional[str]:
    """Execute a Python snippet in isolated mode and return stdout."""

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout_s),
            check=False,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    stdout = proc.stdout.strip()
    if not stdout:
        return None
    return stdout.splitlines()[-1].strip()


def _score_mbpp_code(code: str, tests: List[str], timeout_s: float) -> float:
    """Execute MBPP assertions and return pass fraction in ``[0, 1]``."""

    if not code or not tests:
        return 0.0
    script = (
        "code = "
        + json.dumps(code)
        + "\n"
        + "tests = "
        + json.dumps(tests)
        + "\n"
        + "ns = {}\n"
        + "passed = 0\n"
        + "total = len(tests)\n"
        + "try:\n"
        + "    exec(code, ns, ns)\n"
        + "except BaseException:\n"
        + "    print('0.0')\n"
        + "    raise SystemExit(0)\n"
        + "for test in tests:\n"
        + "    try:\n"
        + "        exec(test, ns, ns)\n"
        + "        passed += 1\n"
        + "    except BaseException:\n"
        + "        pass\n"
        + "print(str((passed / total) if total else 0.0))\n"
    )
    raw = _run_script(script, timeout_s)
    if raw is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def _score_humaneval_code(
    code: str,
    test_program: str,
    entry_point: Optional[str],
    timeout_s: float,
) -> float:
    """Execute HumanEval checks and return binary pass/fail score."""

    if not code or not test_program or not entry_point:
        return 0.0
    script = (
        "code = "
        + json.dumps(code)
        + "\n"
        + "test_program = "
        + json.dumps(test_program)
        + "\n"
        + "entry_point = "
        + json.dumps(entry_point)
        + "\n"
        + "ns = {}\n"
        + "try:\n"
        + "    exec(code, ns, ns)\n"
        + "    exec(test_program, ns, ns)\n"
        + "    candidate = ns.get(entry_point)\n"
        + "    checker = ns.get('check')\n"
        + "    if candidate is None or not callable(checker):\n"
        + "        raise RuntimeError('missing entry point/check')\n"
        + "    checker(candidate)\n"
        + "    print('1.0')\n"
        + "except BaseException:\n"
        + "    print('0.0')\n"
    )
    raw = _run_script(script, timeout_s)
    return 1.0 if raw == "1.0" else 0.0


def _score_apps_code(
    code: str,
    cases: tuple[List[Any], List[Any]],
    timeout_s: float,
) -> float:
    """Execute APPS stdin/stdout tests and return pass fraction."""

    if not code:
        return 0.0
    inputs, outputs = cases
    total = len(inputs)
    if total <= 0:
        return 0.0
    passed = 0
    per_case_timeout = max(0.1, timeout_s / max(1, total))
    for case_input, expected in zip(inputs, outputs):
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                input=str(case_input),
                capture_output=True,
                text=True,
                timeout=per_case_timeout,
                check=False,
            )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        if _outputs_match(proc.stdout, expected):
            passed += 1
    return float(passed) / float(total)


def _score_python_unit_tests_sample(
    completion: CompletionType,
    answer_payload: Any,
    prompt_text: str,
    explicit_entry_point: Optional[str],
    timeout_s: float,
) -> float:
    """Score one completion against MBPP/HumanEval/APPS payloads."""

    text = _extract_content(completion)
    code = _extract_python_code(text)
    if not code:
        return 0.0

    parsed_payload = _parse_answer_payload(answer_payload)
    mbpp_tests = _extract_mbpp_tests(parsed_payload)
    if mbpp_tests:
        return _score_mbpp_code(code, mbpp_tests, timeout_s)

    test_program = _extract_humaneval_test(parsed_payload)
    if test_program:
        entry = _parse_entry_point(prompt_text, explicit_entry_point)
        return _score_humaneval_code(code, test_program, entry, timeout_s)

    apps_cases = _extract_apps_cases(parsed_payload)
    if apps_cases is not None:
        return _score_apps_code(code, apps_cases, timeout_s)

    return 0.0


def python_unit_test_reward(
    completions: List[CompletionType],
    answer: List[Any],
    *,
    prompts: Optional[List[Any]] = None,
    entry_point: Optional[List[Any]] = None,
    **_kwargs: Any,
) -> List[float]:
    """Run local Python-unit-test rewards for MBPP/HumanEval/APPS payloads.

    Supports:
    - MBPP: ``answer`` is ``test_list`` (list of ``assert`` strings)
    - HumanEval: ``answer`` is ``test`` code containing ``def check(...)``
    - APPS: ``answer`` is ``input_output`` payload with ``inputs``/``outputs``
    """

    batch_size = min(len(completions), len(answer))
    if batch_size <= 0:
        return []

    timeout_raw = os.environ.get("MAXENT_CODE_REWARD_TIMEOUT_S", "6.0")
    try:
        timeout_s = float(timeout_raw)
    except (TypeError, ValueError):
        timeout_s = 6.0
    timeout_s = max(0.5, timeout_s)

    prompts_list = prompts if isinstance(prompts, list) else []
    entry_points = entry_point if isinstance(entry_point, list) else []

    default_workers = min(8, max(1, (os.cpu_count() or 1)))
    workers_raw = os.environ.get("MAXENT_CODE_REWARD_WORKERS")
    if workers_raw is None:
        workers = default_workers
    else:
        try:
            workers = int(workers_raw)
        except (TypeError, ValueError):
            workers = default_workers
    workers = max(1, min(workers, batch_size))

    def _eval_one(idx: int) -> float:
        prompt_text = (
            _extract_prompt_text(prompts_list[idx]) if idx < len(prompts_list) else ""
        )
        explicit_entry: Optional[str] = None
        if idx < len(entry_points):
            candidate = entry_points[idx]
            if candidate is not None:
                explicit_entry = str(candidate)
        return _score_python_unit_tests_sample(
            completions[idx],
            answer[idx],
            prompt_text,
            explicit_entry,
            timeout_s,
        )

    if workers <= 1:
        return [_eval_one(i) for i in range(batch_size)]

    rewards = [0.0] * batch_size
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(_eval_one, i): i for i in range(batch_size)}
        for future, idx in future_to_idx.items():
            try:
                rewards[idx] = float(future.result())
            except (TypeError, ValueError, RuntimeError):
                rewards[idx] = 0.0
    return rewards


def _canon_math(s: Any) -> str:
    """Canonicalize simple math answers for exact-match comparison."""

    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    s = s.strip()
    replacements = {
        "π": r"\pi",
        "Π": r"\Pi",
        "τ": r"\tau",
        "θ": r"\theta",
        "Θ": r"\Theta",
        "φ": r"\phi",
        "Φ": r"\Phi",
        "∞": r"\infty",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    boxed = _extract_boxed_answer(s)
    if boxed is not None:
        s = boxed

    def _normalize_frac(match: re.Match[str]) -> str:
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f"({num})/({den})"

    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", _normalize_frac, s)
    # Strip common LaTeX wrappers that should not affect equality.
    s = (
        s.replace("\\left", "")
        .replace("\\right", "")
        .replace("$", "")
        .replace("\\,", "")
    )
    s = re.sub(r"\(([A-Za-z0-9_\\]+)\)", r"\1", s)
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


def _count_format_tags(text: str) -> tuple[int, int]:
    """Return (total tag count, unique tag count) for think/answer tags."""

    if not text:
        return 0, 0
    try:
        tags = [tag.lower() for tag in _tag_pat.findall(text)]
    except re.error:
        return 0, 0
    return len(tags), len(set(tags))


def _tag_multiplier(tag_total: int, tag_unique: int) -> float:
    """Return the reward multiplier for the observed tag counts."""

    if tag_total > 4:
        # Extra tags fall back to the 2-tag reward scale.
        return 0.5
    if tag_unique <= 0:
        return 0.1
    if tag_unique == 1:
        return 0.25
    if tag_unique == 2:
        return 0.5
    if tag_unique == 3:
        return 0.75
    if tag_unique == 4:
        return 1.0
    return 0.5


def _pure_accuracy_math_match_flags(
    text: str,
    gold: Any,
) -> tuple[bool, bool]:
    """Return (answer_tag_match, fallback_last_line_match)."""

    gold_canons = _gold_math_candidates(gold)
    if not gold_canons:
        return False, False
    pred_candidates: List[str] = []
    match = _answer_pat.search(text)
    if match is not None:
        pred_candidates.append(str(match.group(1)))
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        pred_candidates.append(boxed)
    pred_ok = any(_canon_math(pred) in gold_canons for pred in pred_candidates)
    last_line_match = False
    if not pred_ok:
        for line in reversed(text.splitlines()):
            last = line.strip()
            if last:
                last_canon = _canon_math(last)
                if last_canon not in gold_canons:
                    # Allow partial/open tags in fallback mode by stripping
                    # recognized format tags before canonicalizing.
                    last_canon = _canon_math(_tag_pat.sub("", last))
                last_line_match = last_canon in gold_canons
                break
    return pred_ok, last_line_match


def pure_accuracy_math_correctness(
    completions: List[CompletionType],
    answer: List[Any],
    *,
    allow_last_line_fallback: bool = False,
) -> List[bool]:
    """Return binary correctness aligned with ``pure_accuracy_reward_math``.

    A completion is considered correct when either:

    1. ``<answer>...</answer>`` canonicalizes to the gold answer, or
    2. (optional) no extracted answer matched but the final non-empty line
       matches.
    """

    outs: List[bool] = []
    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)
        pred_ok, last_line_match = _pure_accuracy_math_match_flags(txt, gold)
        if pred_ok:
            outs.append(True)
            continue
        outs.append(bool(allow_last_line_fallback and last_line_match))
    return outs


def accuracy_reward(
    completions: List[CompletionType],
    answer: List[Any],
    **_kwargs: Any,
) -> List[float]:
    """Open-R1-style accuracy reward (1.0 exact math match else 0.0).

    This keeps compatibility with Open-R1 reward names while using the same
    canonicalization/extraction logic as ``pure_accuracy_reward_math``,
    including boxed answers and list-valued gold labels.
    """

    outs: List[float] = []
    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)
        pred_ok, last_line_match = _pure_accuracy_math_match_flags(txt, gold)
        outs.append(1.0 if (pred_ok or last_line_match) else 0.0)
    return outs


def boxed_accuracy_reward_math(
    completions: List[CompletionType],
    answer: List[Any],
    **_kwargs: Any,
) -> List[float]:
    """Dr.GRPO-style binary reward based on boxed final answers.

    A completion is rewarded when its final ``\\boxed{...}`` (or ``<answer>`` for
    compatibility) matches one of the canonical gold answers. Plain unboxed
    answers do not receive credit.
    """

    outs: List[float] = []
    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)
        gold_canons = _gold_math_candidates(gold)
        pred = _extract_boxed_answer(txt)
        if pred is None:
            match = _answer_pat.search(txt)
            pred = str(match.group(1)) if match is not None else None
        outs.append(
            1.0
            if pred is not None and _canon_math(pred) in gold_canons
            else 0.0
        )
    return outs


def get_missing_boxed_answer_penalty_reward(
    penalty: float = -0.05,
) -> RewardFunction:
    """Return a fixed penalty when no boxed answer is present in the completion."""

    penalty_value = float(min(penalty, 0.0))

    def _reward(
        completions: List[CompletionType],
        **_kwargs: Any,
    ) -> List[float]:
        outs: List[float] = []
        for comp in completions:
            txt = _extract_content(comp)
            outs.append(0.0 if _extract_boxed_answer(txt) is not None else penalty_value)
        return outs

    return _reward


@lru_cache(maxsize=1)
def _seed_paper_boxed_reward_fn() -> Any:
    """Load the official SEED paper boxed reward function from the repo-local checkout."""

    repo_dir = Path(
        os.environ.get(
            "MAXENT_SEED_PAPER_REPO_DIR",
            str(
                Path(__file__).resolve().parents[3]
                / "var"
                / "seed_paper_eval"
                / "external"
                / "SEED-GRPO"
            ),
        )
    )
    if not repo_dir.exists():
        raise FileNotFoundError(
            "Official SEED-GRPO checkout not found for reward parity at "
            f"{repo_dir}. Run the repo-local SEED paper eval preparation first."
        )
    paper_site_packages = (
        Path(__file__).resolve().parents[3]
        / "var"
        / "seed_paper_eval"
        / "paper_venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if paper_site_packages.exists():
        paper_site_packages_str = str(paper_site_packages)
        if paper_site_packages_str not in sys.path:
            sys.path.insert(0, paper_site_packages_str)
        # Keep training reward parity on the same symbolic stack as the official
        # paper eval. A newer sympy/antlr runtime in e2e-venv produced listwise-
        # only comparison failures and noisier zero rewards.
        for prefix in ("antlr4", "math_verify", "sympy", "understand_r1_zero"):
            for module_name in tuple(sys.modules):
                if module_name != prefix and not module_name.startswith(f"{prefix}."):
                    continue
                module = sys.modules.get(module_name)
                module_file = getattr(module, "__file__", None)
                if module_file and str(module_file).startswith(paper_site_packages_str):
                    continue
                sys.modules.pop(module_name, None)
    else:
        _LOG.warning(
            "SEED paper reward parity requested, but paper site-packages were not found at %s.",
            paper_site_packages,
        )
    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
    if os.environ.get("MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }:
        for logger_name in ("math_verify", "math_verify.grader"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
    grader = importlib.import_module("understand_r1_zero.math_grader")
    reward_fn = getattr(grader, "boxed_reward_fn", None)
    if not callable(reward_fn):
        raise RuntimeError(
            f"Official SEED boxed_reward_fn is unavailable in {repo_dir}."
        )
    return reward_fn


def seed_paper_boxed_accuracy_reward_math(
    completions: List[CompletionType],
    answer: List[Any],
    **_kwargs: Any,
) -> List[float]:
    """Official SEED paper boxed accuracy reward with the same grader as eval."""

    reward_fn = _seed_paper_boxed_reward_fn()
    outs: List[float] = []
    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)
        try:
            _info, reward = reward_fn(txt, gold, fast=False)
            outs.append(float(reward))
        except Exception:
            outs.append(0.0)
    return outs


def format_reward(completions: List[CompletionType], **_kwargs: Any) -> List[float]:
    """Open-R1-compatible strict think/answer formatting reward."""

    pattern = re.compile(
        r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$",
        re.DOTALL | re.MULTILINE,
    )
    contents = [_extract_content(comp) for comp in completions]
    return [1.0 if pattern.match(content) else 0.0 for content in contents]


def tag_count_reward(completions: List[CompletionType], **_kwargs: Any) -> List[float]:
    """Open-R1-compatible partial credit based on expected tag counts."""

    def _count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    return [_count_tags(_extract_content(comp)) for comp in completions]


def reasoning_steps_reward(
    completions: List[CompletionType], **_kwargs: Any
) -> List[float]:
    """Reward explicit step-by-step structure in natural language."""

    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    contents = [_extract_content(comp) for comp in completions]
    matches = [len(re.findall(pattern, content)) for content in contents]
    return [min(1.0, count / 3.0) for count in matches]


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
) -> RewardFunction:
    """Return a length-scaled reward closure compatible with Open-R1 configs."""

    def _reward(
        completions: List[CompletionType],
        answer: List[str],
        **_kwargs: Any,
    ) -> List[float]:
        contents = [_extract_content(comp) for comp in completions]
        correctness = [bool(val) for val in accuracy_reward(completions, answer)]
        denom = max(float(max_len), 1.0)
        outs: List[float] = []
        for content, is_correct in zip(contents, correctness):
            progress = float(len(content)) / denom
            cosine = math.cos(progress * math.pi)
            if is_correct:
                lo, hi = min_value_correct, max_value_correct
            else:
                # Mirror Open-R1 behavior for incorrect responses.
                lo, hi = max_value_wrong, min_value_wrong
            reward = lo + 0.5 * (hi - lo) * (1.0 + cosine)
            outs.append(float(reward))
        return outs

    return _reward


def get_repetition_penalty_reward(
    ngram_size: int,
    max_penalty: float,
) -> RewardFunction:
    """Return an Open-R1-style repetition penalty reward closure."""

    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")
    n = max(int(ngram_size), 1)

    def _reward(completions: List[CompletionType], **_kwargs: Any) -> List[float]:
        outs: List[float] = []
        for comp in completions:
            text = _extract_content(comp).strip()
            if not text:
                outs.append(0.0)
                continue
            words = text.lower().split()
            if len(words) < n:
                outs.append(0.0)
                continue
            total = len(words) - n + 1
            ngrams = {tuple(words[i : i + n]) for i in range(total)}
            scaling = 1.0 - (len(ngrams) / float(total))
            outs.append(float(scaling * max_penalty))
        return outs

    return _reward


def len_reward(
    completions: List[CompletionType],
    answer: List[str],
    **_kwargs: Any,
) -> List[float]:
    """Length-based reward that discourages verbose incorrect outputs."""

    contents = [_extract_content(comp) for comp in completions]
    correctness = [bool(val) for val in accuracy_reward(completions, answer)]
    lengths = [len(content) for content in contents]
    if not lengths:
        return []
    min_len = min(lengths)
    max_len = max(lengths)
    if max_len == min_len:
        return [0.0] * len(lengths)
    outs: List[float] = []
    denom = float(max_len - min_len)
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - ((float(length) - float(min_len)) / denom)
        if is_correct:
            outs.append(float(lambda_val))
        else:
            outs.append(float(min(0.0, lambda_val)))
    return outs


def get_code_format_reward(language: str = "python") -> RewardFunction:
    """Return Open-R1-compatible code-format reward closure."""

    def _reward(completions: List[CompletionType], **_kwargs: Any) -> List[float]:
        pattern = re.compile(
            rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{re.escape(language)}.*?```.*?\n</answer>$",
            re.DOTALL | re.MULTILINE,
        )
        contents = [_extract_content(comp) for comp in completions]
        return [1.0 if pattern.match(content) else 0.0 for content in contents]

    return _reward


def binary_code_reward(
    completions: List[CompletionType],
    answer: List[Any],
    **kwargs: Any,
) -> List[float]:
    """Binary wrapper around ``python_unit_test_reward`` for compatibility."""

    scores = python_unit_test_reward(completions, answer, **kwargs)
    threshold = 0.99
    return [1.0 if score > threshold else 0.0 for score in scores]


def uses_pure_accuracy_math_reward(reward_funcs: Sequence[Any]) -> bool:
    """Return ``True`` when any configured reward resolves to pure math reward."""
    seen: set[int] = set()
    pending: List[Any] = list(reward_funcs)
    while pending:
        reward_fn = pending.pop()
        if reward_fn is None:
            continue
        fn_id = id(reward_fn)
        if fn_id in seen:
            continue
        seen.add(fn_id)
        if reward_fn is pure_accuracy_reward_math:
            return True
        name = str(getattr(reward_fn, "__name__", "") or "")
        qualname = str(getattr(reward_fn, "__qualname__", "") or "")
        if name == "pure_accuracy_reward_math" or qualname.endswith(
            ".pure_accuracy_reward_math"
        ):
            return True
        wrapped = getattr(reward_fn, "__wrapped__", None)
        if wrapped is not None and wrapped is not reward_fn:
            pending.append(wrapped)
        func = getattr(reward_fn, "func", None)
        if func is not None and func is not reward_fn:
            pending.append(func)
    return False


def pure_accuracy_reward_math(
    completions: List[CompletionType], answer: List[Any], **_kwargs: Any
) -> List[float]:
    """Reward exact math matches with a small formatting bonus when wrong.

    Correctness is detected from ``<answer>...</answer>`` and falls back to the
    last non-empty line (with known format tags stripped) when needed.

    Reward scale for correct outputs:
    - full ``<think>...</think><answer>...</answer>`` (4 distinct tags): ``1.0``
    - otherwise: ``0.5 * _tag_multiplier(tag_total, tag_unique)``

    Reward scale for incorrect outputs:
    - full ``<think>...</think><answer>...</answer>`` (4 distinct tags): ``0.05``
    - otherwise: ``0.0``
    """

    outs: List[float] = []
    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)
        # Allow harmless leading/trailing whitespace around the tag payload.
        format_ok = bool(_format_pat.fullmatch(txt.strip()))
        pred_ok, last_line_match = _pure_accuracy_math_match_flags(txt, gold)
        tag_total, tag_unique = _count_format_tags(txt)
        if format_ok and pred_ok and tag_total == 4 and tag_unique == 4:
            outs.append(1.0)
            continue
        if not (pred_ok or last_line_match):
            if format_ok and tag_total == 4 and tag_unique == 4:
                outs.append(0.05)
                continue
            outs.append(0.0)
            continue
        outs.append(0.5 * _tag_multiplier(tag_total, tag_unique))
    return outs


@runtime_checkable
class RewardConfig(Protocol):
    """Minimal protocol describing the reward configuration interface."""

    reward_funcs: List[str]


def get_reward_funcs(
    script_args: RewardConfig,
    _ref_model: Optional["PreTrainedModel"] = None,
    _tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> List["RewardFunction"]:
    """Resolve reward function callables from names."""

    cosine_reward = get_cosine_scaled_reward(
        min_value_wrong=float(getattr(script_args, "cosine_min_value_wrong", -1.0)),
        max_value_wrong=float(getattr(script_args, "cosine_max_value_wrong", -0.5)),
        min_value_correct=float(
            getattr(script_args, "cosine_min_value_correct", 0.5)
        ),
        max_value_correct=float(getattr(script_args, "cosine_max_value_correct", 1.0)),
        max_len=int(getattr(script_args, "cosine_max_len", 1000)),
    )
    repetition_reward = get_repetition_penalty_reward(
        ngram_size=int(getattr(script_args, "repetition_n_grams", 3)),
        max_penalty=float(getattr(script_args, "repetition_max_penalty", -1.0)),
    )
    code_format_reward = get_code_format_reward(
        language=str(getattr(script_args, "code_language", "python"))
    )
    missing_boxed_answer_penalty_reward = get_missing_boxed_answer_penalty_reward(
        penalty=float(getattr(script_args, "missing_boxed_answer_penalty", -0.05))
    )
    registry = {
        # Open-R1-compatible aliases.
        "accuracy": accuracy_reward,
        "format": format_reward,
        "tag_count": tag_count_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": cosine_reward,
        "repetition_penalty": repetition_reward,
        "length": len_reward,
        "code": python_unit_test_reward,
        "binary_code": binary_code_reward,
        "code_format": code_format_reward,
        # Native/extended names.
        "boxed_accuracy_math": boxed_accuracy_reward_math,
        "seed_paper_boxed_accuracy_math": seed_paper_boxed_accuracy_reward_math,
        "missing_boxed_answer_penalty_math": missing_boxed_answer_penalty_reward,
        "pure_accuracy_math": pure_accuracy_reward_math,
        "python_unit_tests": python_unit_test_reward,
        "mbpp_python_tests": python_unit_test_reward,
    }
    return [registry[name] for name in script_args.reward_funcs]


__all__ = [
    "RewardFunction",
    "RewardConfig",
    "accuracy_reward",
    "format_reward",
    "get_reward_funcs",
    "get_missing_boxed_answer_penalty_reward",
    "boxed_accuracy_reward_math",
    "seed_paper_boxed_accuracy_reward_math",
    "tag_count_reward",
    "reasoning_steps_reward",
    "get_cosine_scaled_reward",
    "get_repetition_penalty_reward",
    "len_reward",
    "get_code_format_reward",
    "pure_accuracy_math_correctness",
    "pure_accuracy_reward_math",
    "python_unit_test_reward",
    "binary_code_reward",
    "truncate_after_first_boxed_answer",
    "uses_pure_accuracy_math_reward",
]
