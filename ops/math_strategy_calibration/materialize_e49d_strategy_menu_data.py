#!/usr/bin/env python3
"""Generate, double-audit, and materialize finite strategy-menu MATH data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.math_strategy_menu import (  # noqa: E402
    MENU_END,
    MENU_SCHEMA,
    MENU_START,
    MathStrategyMenu,
    parse_strategy_menu,
    strategy_menu_response_instructions,
)


SYSTEM = (
    "You design conservative, executable mathematical proof plans. "
    "Return valid JSON only."
)
GENERATION_SEED_BASE = 491701
AUDIT_SEEDS = (491711, 491712)
AUDIT_ROLES = ("soundness_execution", "equivalence_attack")
AUDIT_CONTRACT_VERSION = "maximal_certified_support_v4"
AUDIT_TRANSPORT_VERSION = "local_string_bounds_v1"
SUPPORT_RECALL_VERSION = "one_extra_route_rescue_v1"
MAX_GENERATION_ATTEMPTS = 1
MAX_RESCUE_ATTEMPTS = 2


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_jsonl(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tree_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _endpoint(record_path: pathlib.Path) -> tuple[str, str]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    required = {
        "model": "qwen2.5-72b",
        "node": "node105",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise RuntimeError("unexpected frozen Qwen72 endpoint identity")
    host = os.environ.get("E49D_QWEN_HOST_OVERRIDE", "").strip()
    if not host:
        host = str(record["node"])
    return (f"http://{host}:{int(record['port'])}/v1", str(record["model"]))


def _post(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    error: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read()
            decoded = json.loads(body.decode("utf-8"))
            choices = decoded.get("choices") or []
            if not choices:
                raise ValueError("judge returned no choices")
            content = str(
                (choices[0].get("message") or {}).get("content") or ""
            )
            return decoded, content
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            error = RuntimeError(
                f"HTTP {exc.code} {exc.reason}: {detail[:2000]}"
            )
            if attempt < 2:
                time.sleep(2**attempt)
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Qwen72 request failed after 3 attempts: {error}")


def _menu_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "actions", "strategies"],
        "properties": {
            "schema": {"type": "string", "enum": [MENU_SCHEMA]},
            "actions": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action_id", "operation"],
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "enum": [f"A{index}" for index in range(1, 9)],
                        },
                        "operation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 320,
                        },
                    },
                },
            },
            "strategies": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["strategy_id", "action_ids", "plan"],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": ["S1", "S2", "S3"],
                        },
                        "action_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "string",
                                "enum": [
                                    f"A{index}" for index in range(1, 9)
                                ],
                            },
                        },
                        "plan": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 640,
                        },
                    },
                },
            },
        },
    }


def _generate_menu(
    *,
    endpoint: str,
    model: str,
    problem: str,
    attempt: int,
    feedback: str,
    route_ideas: list[dict[str, str]] | None = None,
    timeout: int,
) -> tuple[MathStrategyMenu, dict[str, Any]]:
    user = f"""Create a finite action menu for the MATH problem below.

Design preferably THREE, but at least TWO, genuinely different and
mathematically sound solution routes that a small language model could
execute. Each route must be sufficient to derive the answer from the problem,
not merely verify a guessed answer. Distinct routes must differ in a central
identity, theorem, substitution, construction, counting argument, or proof
method—not wording, notation, routine algebra, or order of arithmetic.

Build one shared action vocabulary A1..An. Each action is a concise,
problem-specific mathematical operation. Then define S1..Sk as distinct
ordered action-ID sequences. IDs must be consecutive in the order returned.
No action may repeat inside one strategy. Plans must explain how those exact
actions solve the problem, but MUST NOT state or leak the final numerical or
symbolic answer. Avoid vague actions such as "solve the problem", "reason",
"simplify as needed", or "check the answer".

PROBLEM:
{problem}
"""
    if feedback:
        user += (
            "\nA prior menu failed conservative audits. Repair these issues "
            "without mentioning the final answer:\n" + feedback[:3000]
        )
    if route_ideas:
        user += f"""

The following two route families were produced by a separate route-ideation
pass. You MUST implement both as S1 and S2 respectively. Give each route at
least one route-specific decisive action so their action-ID sequences differ.
Do not collapse one into the other, and do not add a redundant afterthought.

ROUTE FAMILIES:
{json.dumps(route_ideas, sort_keys=True)}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 2048,
        "seed": GENERATION_SEED_BASE + attempt,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_strategy_action_menu",
                "strict": True,
                "schema": _menu_schema(),
            },
        },
    }
    response, content = _post(endpoint, payload, timeout=timeout)
    parsed = json.loads(content)
    raw_strategies = (
        parsed.get("strategies") if isinstance(parsed, dict) else None
    )
    collapsed_duplicate_strategy_count = 0
    if isinstance(raw_strategies, list):
        seen_combos: set[tuple[str, ...]] = set()
        unique_strategies = []
        for strategy in raw_strategies:
            if not isinstance(strategy, dict):
                continue
            combo = tuple(strategy.get("action_ids") or ())
            if combo in seen_combos:
                collapsed_duplicate_strategy_count += 1
                continue
            seen_combos.add(combo)
            normalized = dict(strategy)
            normalized["strategy_id"] = f"S{len(unique_strategies) + 1}"
            unique_strategies.append(normalized)
        parsed["strategies"] = unique_strategies
    embedded = (
        f"x\n{MENU_START}\n"
        + json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        + f"\n{MENU_END}"
    )
    menu = parse_strategy_menu(embedded)
    if menu is None:
        raise RuntimeError("generated menu parser returned none")
    return menu, {
        "attempt": attempt,
        "request_seed": GENERATION_SEED_BASE + attempt,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "menu_sha256": menu.sha256,
        "route_rescue": bool(route_ideas),
        "collapsed_duplicate_strategy_count": (
            collapsed_duplicate_strategy_count
        ),
    }


def _generate_route_ideas(
    *,
    endpoint: str,
    model: str,
    problem: str,
    rescue_attempt: int,
    feedback: str,
    timeout: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["routes"],
        "properties": {
            "routes": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "route_id",
                        "decisive_method",
                        "outline",
                    ],
                    "properties": {
                        "route_id": {"type": "string"},
                        "decisive_method": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 320,
                        },
                        "outline": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 640,
                        },
                    },
                },
            }
        },
    }
    user = f"""Find exactly TWO genuinely different, mathematically sound
solution routes for the problem below. This is a rescue pass because a
one-shot generator proposed equivalent or unsound routes.

The two routes must differ in a decisive method, not prose or routine algebra.
Actively search across applicable method families: direct algebra versus
factoring or bounding; quadratic formula versus integer factor-pair search;
Chinese remaindering versus progression enumeration; direct versus
complementary counting; recurrence versus a closed form; synthetic versus
coordinate geometry; arc-length versus area relations; substitution versus a
special identity; constructive versus contradiction/case analysis. These are
general examples only—use only routes that are actually sound for this
problem.

Each outline must be sufficient to derive the requested answer. Do not state
or leak the final numerical or symbolic answer. Return R1 then R2.

PROBLEM:
{problem}

PRIOR FAILURE RATIONALES:
{feedback[:4000]}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1536,
        "seed": GENERATION_SEED_BASE + 100 + rescue_attempt,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_distinct_route_ideas",
                "strict": True,
                "schema": schema,
            },
        },
    }
    response, content = _post(endpoint, payload, timeout=timeout)
    parsed = json.loads(content)
    routes = parsed.get("routes") if isinstance(parsed, dict) else None
    if (
        not isinstance(routes, list)
        or len(routes) != 2
        or [route.get("route_id") for route in routes] != ["R1", "R2"]
    ):
        raise ValueError("route rescue must return R1 and R2 in order")
    normalized = [
        {
            "route_id": str(route["route_id"]),
            "decisive_method": str(route["decisive_method"]).strip(),
            "outline": str(route["outline"]).strip(),
        }
        for route in routes
    ]
    if (
        not all(route["decisive_method"] and route["outline"] for route in normalized)
        or len(
            {
                route["decisive_method"].casefold()
                for route in normalized
            }
        )
        != 2
    ):
        raise ValueError("route rescue returned duplicate or empty methods")
    return normalized, {
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "request_seed": GENERATION_SEED_BASE + 100 + rescue_attempt,
    }


def _audit_schema(menu: MathStrategyMenu) -> dict[str, Any]:
    strategy_ids = [strategy.strategy_id for strategy in menu.strategies]
    pair_ids = [
        f"{left}__{right}"
        for left, right in combinations(strategy_ids, 2)
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["strategy_assessments", "pair_assessments"],
        "properties": {
            "strategy_assessments": {
                "type": "array",
                "minItems": len(strategy_ids),
                "maxItems": len(strategy_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "strategy_id",
                        "status",
                        "failure_code",
                        "derived_answer",
                        "matches_reference_answer",
                        "brief_derivation",
                    ],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": strategy_ids,
                        },
                        "status": {
                            "type": "string",
                            "enum": ["sound", "unsound", "ambiguous"],
                        },
                        "failure_code": {
                            "type": "string",
                            "enum": [
                                "none",
                                "arithmetic_or_algebra_error",
                                "invalid_identity_or_theorem",
                                "invalid_geometric_relation",
                                "domain_or_case_error",
                                "missing_decisive_step",
                                "vague_or_circular",
                                "answer_mismatch",
                                "other",
                            ],
                        },
                        "derived_answer": {
                            "type": "string",
                        },
                        "matches_reference_answer": {"type": "boolean"},
                        "brief_derivation": {
                            "type": "string",
                        },
                    },
                },
            },
            "pair_assessments": {
                "type": "array",
                "minItems": len(pair_ids),
                "maxItems": len(pair_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "pair_id",
                        "relation",
                        "shared_core",
                        "left_decisive_operation",
                        "right_decisive_operation",
                        "brief_check",
                    ],
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "enum": pair_ids,
                        },
                        "relation": {
                            "type": "string",
                            "enum": ["distinct", "equivalent", "ambiguous"],
                        },
                        "shared_core": {
                            "type": "string",
                        },
                        "left_decisive_operation": {
                            "type": "string",
                        },
                        "right_decisive_operation": {
                            "type": "string",
                        },
                        "brief_check": {
                            "type": "string",
                        },
                    },
                },
            },
        },
    }


def _bounded_text(
    row: dict[str, Any],
    key: str,
    *,
    maximum: int | None,
) -> bool:
    value = row.get(key)
    return (
        isinstance(value, str)
        and bool(value.strip())
        and (maximum is None or len(value) <= maximum)
    )


def _audit_assessment_is_well_formed(
    menu: MathStrategyMenu,
    result: Any,
    *,
    enforce_string_bounds: bool = True,
) -> bool:
    """Enforce locally the exact audit schema, including text bounds."""

    if not isinstance(result, dict) or set(result) != {
        "strategy_assessments",
        "pair_assessments",
    }:
        return False
    strategy_ids = [strategy.strategy_id for strategy in menu.strategies]
    strategy_rows = result.get("strategy_assessments")
    if not isinstance(strategy_rows, list) or len(strategy_rows) != len(
        strategy_ids
    ):
        return False
    strategy_required = {
        "strategy_id",
        "status",
        "failure_code",
        "derived_answer",
        "matches_reference_answer",
        "brief_derivation",
    }
    failure_codes = {
        "none",
        "arithmetic_or_algebra_error",
        "invalid_identity_or_theorem",
        "invalid_geometric_relation",
        "domain_or_case_error",
        "missing_decisive_step",
        "vague_or_circular",
        "answer_mismatch",
        "other",
    }
    if any(
        not isinstance(row, dict)
        or set(row) != strategy_required
        or row.get("status") not in {"sound", "unsound", "ambiguous"}
        or row.get("failure_code") not in failure_codes
        or type(row.get("matches_reference_answer")) is not bool
        or not _bounded_text(
            row,
            "derived_answer",
            maximum=128 if enforce_string_bounds else None,
        )
        or not _bounded_text(
            row,
            "brief_derivation",
            maximum=512 if enforce_string_bounds else None,
        )
        for row in strategy_rows
    ):
        return False
    if {row.get("strategy_id") for row in strategy_rows} != set(strategy_ids):
        return False

    pair_ids = [
        f"{left}__{right}"
        for left, right in combinations(strategy_ids, 2)
    ]
    pair_rows = result.get("pair_assessments")
    if not isinstance(pair_rows, list) or len(pair_rows) != len(pair_ids):
        return False
    pair_required = {
        "pair_id",
        "relation",
        "shared_core",
        "left_decisive_operation",
        "right_decisive_operation",
        "brief_check",
    }
    if any(
        not isinstance(row, dict)
        or set(row) != pair_required
        or row.get("relation") not in {
            "distinct",
            "equivalent",
            "ambiguous",
        }
        or not _bounded_text(
            row,
            "shared_core",
            maximum=256 if enforce_string_bounds else None,
        )
        or not _bounded_text(
            row,
            "left_decisive_operation",
            maximum=256 if enforce_string_bounds else None,
        )
        or not _bounded_text(
            row,
            "right_decisive_operation",
            maximum=256 if enforce_string_bounds else None,
        )
        or not _bounded_text(
            row,
            "brief_check",
            maximum=512 if enforce_string_bounds else None,
        )
        for row in pair_rows
    ):
        return False
    return {row.get("pair_id") for row in pair_rows} == set(pair_ids)


def _audit_menu(
    *,
    endpoint: str,
    model: str,
    problem: str,
    reference_answer: str,
    menu: MathStrategyMenu,
    seed: int,
    audit_role: str,
    timeout: int,
) -> dict[str, Any]:
    if audit_role not in AUDIT_ROLES:
        raise ValueError(f"unknown audit role: {audit_role}")
    role_instruction = {
        "soundness_execution": """
Your primary role is SOUNDNESS EXECUTION. Independently solve the problem,
then literally simulate each strategy's exact ordered actions. A route is
sound only if those actions, without silently adding a missing theorem or
relation, derive an answer mathematically equivalent to the auditor-only
reference answer.""",
        "equivalence_attack": """
Your primary role is ADVERSARIAL EQUIVALENCE ATTACK. For each pair, first try
to map both plans to the same decisive equation, identity, search space, or
invariant using routine algebra. Distinct is allowed only if this attempted
merge fails and you can name one genuinely route-specific decisive operation
on each side. You must still execute every route and check its answer.""",
    }[audit_role]
    user = f"""Conservatively audit this proposed finite strategy menu.

{role_instruction}

For each strategy, independently work through its exact ordered actions and
decide whether they form a mathematically sound, sufficient route to derive
the requested answer. Record the answer actually derived, whether it matches
the auditor-only reference answer, and a concise derivation. Mark vague,
circular, missing-case, domain-invalid, relation-invalid, or incomplete routes
unsound or ambiguous. Do not repair a route, import a missing operation, or
infer soundness merely because the reference answer is visible.

For every required pair, act as an asymmetric NOVELTY VETO. Equivalent is the
default. Return distinct only when the brief_check can name different decisive
central operations, theorems, invariants, constructions, substitutions, or
proof structures that are actually present in the two plans. Matching
different words is never sufficient.

The following are equivalent, not distinct: directly computing a remainder
by division versus writing the division algorithm n=q m+r; factoring the same
prime powers versus applying exponent laws to those factors; solving the same
quadratic after routine rearrangement; writing a probability complement in
words versus symbols; changing variable names, units, order of arithmetic, or
level of detail; subtracting percentages before multiplying by a total versus
multiplying each percentage first and subtracting the resulting counts; and a
fixed list of repeated multiplications versus writing that same product with
an exponent. Adding an irrelevant check, an unnecessary negative root, or a
redundant afterthought does not create novelty. A recurrence versus closed
form is distinct only when the recurrence carries a genuine state relation or
induction argument—not when it is merely expanded arithmetic.

The following can be distinct when genuinely executed: CRT construction
versus explicit progression enumeration; quadratic-formula solving versus an
integer factor-pair search; direct versus complementary counting; synthetic
versus coordinate geometry; deriving a geometric quantity from an independent
area invariant versus a length/circumference invariant. Direct versus
complementary counting requires genuinely different counted configuration
sets; distributive arithmetic on the same percentages is not such a case. If
the claimed distinction reduces to the same decisive equation, identity, or
finite arithmetic by routine algebra, mark equivalent. If unsure, mark
ambiguous. Either result vetoes novelty.

In every pair assessment, state the shared mathematical core and the claimed
decisive operation exclusive to each side. If either exclusive operation is
absent, cosmetic, invalid, or recoverable from the shared core by routine
algebra, the relation cannot be distinct.

Keep every free-text field concise: derived_answer is only the final derived
value; each operation/shared-core field is one short phrase; and each
derivation or pair check is at most four compact sentences. Do not include
extended scratch work.

PROBLEM:
{problem}

AUDITOR-ONLY REFERENCE ANSWER:
{reference_answer}

MENU:
{menu.canonical_json}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_strategy_menu_audit",
                "strict": True,
                "schema": _audit_schema(menu),
            },
        },
    }
    response, content = _post(endpoint, payload, timeout=timeout)
    result = json.loads(content)
    if not _audit_assessment_is_well_formed(menu, result):
        passed = False
        strategy_rows = []
        pair_rows = []
    else:
        strategy_rows = result["strategy_assessments"]
        pair_rows = result["pair_assessments"]
    strategy_expected = {
        strategy.strategy_id for strategy in menu.strategies
    }
    if (
        not isinstance(strategy_rows, list)
        or {row.get("strategy_id") for row in strategy_rows}
        != strategy_expected
        or any(
            row.get("status") != "sound"
            or row.get("failure_code") != "none"
            or row.get("matches_reference_answer") is not True
            or not str(row.get("derived_answer") or "").strip()
            for row in strategy_rows
        )
    ):
        passed = False
    else:
        passed = True
    pair_expected = {
        f"{left}__{right}"
        for left, right in combinations(sorted(strategy_expected), 2)
    }
    if (
        not isinstance(pair_rows, list)
        or {row.get("pair_id") for row in pair_rows} != pair_expected
        or any(
            row.get("relation") != "distinct"
            or not str(row.get("left_decisive_operation") or "").strip()
            or not str(row.get("right_decisive_operation") or "").strip()
            or str(row.get("left_decisive_operation") or "").strip().casefold()
            == str(row.get("right_decisive_operation") or "").strip().casefold()
            for row in pair_rows
        )
    ):
        passed = False
    return {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "audit_transport_version": AUDIT_TRANSPORT_VERSION,
        "audit_role": audit_role,
        "seed": seed,
        "pass": passed,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "assessment": result,
    }


def _prune_menu(
    menu: MathStrategyMenu,
    retained_ids: tuple[str, ...],
) -> MathStrategyMenu:
    retained = [
        strategy
        for strategy in menu.strategies
        if strategy.strategy_id in retained_ids
    ]
    used_action_ids = {
        action_id
        for strategy in retained
        for action_id in strategy.action_ids
    }
    retained_actions = [
        action for action in menu.actions if action.action_id in used_action_ids
    ]
    action_map = {
        action.action_id: f"A{index}"
        for index, action in enumerate(retained_actions, start=1)
    }
    payload = {
        "schema": MENU_SCHEMA,
        "actions": [
            {
                "action_id": action_map[action.action_id],
                "operation": action.operation,
            }
            for action in retained_actions
        ],
        "strategies": [
            {
                "strategy_id": f"S{index}",
                "action_ids": [
                    action_map[action_id] for action_id in strategy.action_ids
                ],
                "plan": strategy.plan,
            }
            for index, strategy in enumerate(retained, start=1)
        ],
    }
    embedded = (
        f"x\n{MENU_START}\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + f"\n{MENU_END}"
    )
    pruned = parse_strategy_menu(embedded)
    if pruned is None:
        raise RuntimeError("certified menu disappeared during pruning")
    return pruned


def _maximal_certified_subset(
    menu: MathStrategyMenu,
    audits: list[dict[str, Any]],
) -> tuple[MathStrategyMenu, dict[str, Any]] | None:
    """Return the deterministic maximum pairwise-certified strategy subset."""

    expected_audits = set(zip(AUDIT_SEEDS, AUDIT_ROLES, strict=True))
    if (
        len(audits) != len(expected_audits)
        or {
            (audit.get("seed"), audit.get("audit_role"))
            for audit in audits
        }
        != expected_audits
        or any(
            audit.get("audit_contract_version") != AUDIT_CONTRACT_VERSION
            or audit.get("finish_reason") != "stop"
            or not str(audit.get("response_id") or "")
            or not isinstance(audit.get("assessment"), dict)
            for audit in audits
        )
        or len(
            {str(audit.get("response_id") or "") for audit in audits}
        )
        != len(audits)
    ):
        return None

    strategy_ids = tuple(
        strategy.strategy_id for strategy in menu.strategies
    )
    individually_sound: set[str] = set(strategy_ids)
    distinct_edges: set[tuple[str, str]] | None = None
    for audit in audits:
        assessment = audit["assessment"]
        transport_version = audit.get("audit_transport_version")
        if transport_version not in {None, AUDIT_TRANSPORT_VERSION}:
            return None
        if not _audit_assessment_is_well_formed(
            menu,
            assessment,
            enforce_string_bounds=(
                transport_version == AUDIT_TRANSPORT_VERSION
            ),
        ):
            return None
        strategy_rows = assessment.get("strategy_assessments")
        if not isinstance(strategy_rows, list):
            return None
        by_strategy = {
            str(row.get("strategy_id")): row
            for row in strategy_rows
            if isinstance(row, dict)
        }
        if set(by_strategy) != set(strategy_ids):
            return None
        individually_sound &= {
            strategy_id
            for strategy_id, row in by_strategy.items()
            if row.get("status") == "sound"
            and row.get("failure_code") == "none"
            and row.get("matches_reference_answer") is True
            and bool(str(row.get("derived_answer") or "").strip())
        }

        pair_rows = assessment.get("pair_assessments")
        if not isinstance(pair_rows, list):
            return None
        audit_edges: set[tuple[str, str]] = set()
        expected_pair_ids = {
            f"{left}__{right}"
            for left, right in combinations(strategy_ids, 2)
        }
        if {str(row.get("pair_id")) for row in pair_rows} != expected_pair_ids:
            return None
        for row in pair_rows:
            left, right = str(row["pair_id"]).split("__", maxsplit=1)
            left_operation = str(
                row.get("left_decisive_operation") or ""
            ).strip()
            right_operation = str(
                row.get("right_decisive_operation") or ""
            ).strip()
            if (
                row.get("relation") == "distinct"
                and left_operation
                and right_operation
                and left_operation.casefold() != right_operation.casefold()
            ):
                audit_edges.add((left, right))
        distinct_edges = (
            audit_edges
            if distinct_edges is None
            else distinct_edges & audit_edges
        )

    if not individually_sound:
        return None
    shared_edges = distinct_edges or set()
    retained_ids: tuple[str, ...] | None = None
    for size in range(len(strategy_ids), 0, -1):
        for candidate in combinations(strategy_ids, size):
            if not set(candidate) <= individually_sound:
                continue
            if all(
                (left, right) in shared_edges
                for left, right in combinations(candidate, 2)
            ):
                retained_ids = candidate
                break
        if retained_ids is not None:
            break
    if retained_ids is None:
        return None

    pruned = _prune_menu(menu, retained_ids)
    certification = {
        "schema": "e49d_maximal_certified_support_v1",
        "proposal_menu": json.loads(menu.canonical_json),
        "proposal_menu_sha256": menu.sha256,
        "audits": audits,
        "retained_original_strategy_ids": list(retained_ids),
        "retained_strategy_count": len(retained_ids),
        "retained_menu_sha256": pruned.sha256,
    }
    return pruned, certification


def _feedback(audits: list[dict[str, Any]]) -> str:
    """Return answer-blind repair directives; never relay audit derivations."""
    issues = []
    for audit in audits:
        if audit["pass"]:
            continue
        assessment = audit["assessment"]
        for row in assessment.get("strategy_assessments", []):
            if (
                row.get("status") != "sound"
                or row.get("failure_code") != "none"
                or row.get("matches_reference_answer") is not True
            ):
                issues.append(
                    f"{row.get('strategy_id')} {row.get('status')}: "
                    f"{row.get('failure_code')}; discard or replace this "
                    "route without using any auditor-derived result"
                )
        for row in assessment.get("pair_assessments", []):
            left = str(row.get("left_decisive_operation") or "").strip()
            right = str(row.get("right_decisive_operation") or "").strip()
            if (
                row.get("relation") != "distinct"
                or not left
                or not right
                or left.casefold() == right.casefold()
            ):
                issues.append(
                    f"{row.get('pair_id')} {row.get('relation')}: replace "
                    "one route with a genuinely different theorem, "
                    "construction, invariant, or search space"
                )
    return "\n".join(issues)


def _build_one(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    problem: str,
    reference_answer: str,
    timeout: int,
    prior_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback = ""
    attempts = []
    best_singleton: tuple[MathStrategyMenu, dict[str, Any]] | None = None

    def accepted_record(
        retained_menu: MathStrategyMenu,
        certification: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "schema": "e49d_strategy_menu_record_v1",
            "row_id": row_id,
            "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
            "reference_answer_sha256": _sha256_bytes(
                reference_answer.encode("utf-8")
            ),
            "problem": problem,
            "menu": json.loads(retained_menu.canonical_json),
            "menu_sha256": retained_menu.sha256,
            "attempts": attempts,
            "certification": certification,
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "pass": True,
        }
        if (
            len(retained_menu.strategies) == 1
            and any(
                attempt.get("phase") == "route_rescue"
                and attempt.get("rescue_attempt") == MAX_RESCUE_ATTEMPTS
                and isinstance(attempt.get("proposal_menu"), dict)
                and isinstance(attempt.get("audits"), list)
                for attempt in attempts
            )
        ):
            record["support_recall_version"] = SUPPORT_RECALL_VERSION
            record["support_recall_complete"] = True
        return record

    if (
        isinstance(prior_record, dict)
        and _record_passes_contract(prior_record)
        and len(prior_record["menu"]["strategies"]) == 1
        and not _record_has_support_recall(prior_record)
    ):
        return _recall_existing_singleton(
            endpoint=endpoint,
            model=model,
            row_id=row_id,
            problem=problem,
            reference_answer=reference_answer,
            timeout=timeout,
            prior_record=prior_record,
        )

    if (
        isinstance(prior_record, dict)
        and prior_record.get("pass") is True
        and isinstance(prior_record.get("menu"), dict)
    ):
        try:
            menu_json = json.dumps(
                prior_record["menu"],
                sort_keys=True,
                separators=(",", ":"),
            )
            menu = parse_strategy_menu(
                f"x\n{MENU_START}\n{menu_json}\n{MENU_END}"
            )
            if menu is None:
                raise ValueError("prior menu disappeared during v4 re-audit")
            audits = [
                _audit_menu(
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    seed=seed,
                    audit_role=audit_role,
                    timeout=timeout,
                )
                for seed, audit_role in zip(
                    AUDIT_SEEDS, AUDIT_ROLES, strict=True
                )
            ]
            attempts.append(
                {
                    "phase": "v4_reaudit",
                    "prior_menu_sha256": prior_record.get("menu_sha256"),
                    "audits": audits,
                }
            )
            selected = _maximal_certified_subset(menu, audits)
            if selected is not None:
                retained_menu, certification = selected
                if len(retained_menu.strategies) >= 2:
                    return accepted_record(retained_menu, certification)
                best_singleton = selected
            feedback = _feedback(audits)
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            attempts.append(
                {
                    "phase": "v4_reaudit",
                    "error": str(exc),
                }
            )
            feedback = str(exc)
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            menu, generation = _generate_menu(
                endpoint=endpoint,
                model=model,
                problem=problem,
                attempt=attempt,
                feedback=feedback,
                route_ideas=None,
                timeout=timeout,
            )
            audits = [
                _audit_menu(
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    seed=seed,
                    audit_role=audit_role,
                    timeout=timeout,
                )
                for seed, audit_role in zip(
                    AUDIT_SEEDS, AUDIT_ROLES, strict=True
                )
            ]
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            attempts.append({"attempt": attempt, "error": str(exc)})
            feedback = str(exc)
            continue
        attempts.append(
            {
                **generation,
                "proposal_menu": json.loads(menu.canonical_json),
                "audits": audits,
            }
        )
        selected = _maximal_certified_subset(menu, audits)
        if selected is not None:
            retained_menu, certification = selected
            if len(retained_menu.strategies) >= 2:
                return accepted_record(retained_menu, certification)
            if best_singleton is None:
                best_singleton = selected
        feedback = _feedback(audits)
    for rescue_attempt in range(1, MAX_RESCUE_ATTEMPTS + 1):
        try:
            route_ideas, route_generation = _generate_route_ideas(
                endpoint=endpoint,
                model=model,
                problem=problem,
                rescue_attempt=rescue_attempt,
                feedback=feedback,
                timeout=timeout,
            )
            menu, generation = _generate_menu(
                endpoint=endpoint,
                model=model,
                problem=problem,
                attempt=100 + rescue_attempt,
                feedback=feedback,
                route_ideas=route_ideas,
                timeout=timeout,
            )
            audits = [
                _audit_menu(
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    seed=seed,
                    audit_role=audit_role,
                    timeout=timeout,
                )
                for seed, audit_role in zip(
                    AUDIT_SEEDS, AUDIT_ROLES, strict=True
                )
            ]
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            failed_attempt = {
                "phase": "route_rescue",
                "rescue_attempt": rescue_attempt,
                "error": str(exc),
            }
            if rescue_attempt == MAX_RESCUE_ATTEMPTS:
                failed_attempt["support_recall_version"] = (
                    SUPPORT_RECALL_VERSION
                )
            attempts.append(failed_attempt)
            feedback = str(exc)
            continue
        completed_attempt = {
            "phase": "route_rescue",
            "rescue_attempt": rescue_attempt,
            "route_generation": route_generation,
            "route_ideas": route_ideas,
            **generation,
            "proposal_menu": json.loads(menu.canonical_json),
            "audits": audits,
        }
        if rescue_attempt == MAX_RESCUE_ATTEMPTS:
            completed_attempt["support_recall_version"] = (
                SUPPORT_RECALL_VERSION
            )
        attempts.append(completed_attempt)
        selected = _maximal_certified_subset(menu, audits)
        if selected is not None:
            retained_menu, certification = selected
            if len(retained_menu.strategies) >= 2:
                return accepted_record(retained_menu, certification)
            if best_singleton is None:
                best_singleton = selected
        feedback = _feedback(audits)
    if best_singleton is not None:
        return accepted_record(*best_singleton)
    return {
        "schema": "e49d_strategy_menu_record_v1",
        "row_id": row_id,
        "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": _sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "problem": problem,
        "attempts": attempts,
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "pass": False,
    }


def _recall_existing_singleton(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    problem: str,
    reference_answer: str,
    timeout: int,
    prior_record: dict[str, Any],
) -> dict[str, Any]:
    """Run only the one frozen supplemental rescue for a valid singleton."""

    attempts = list(prior_record.get("attempts") or [])
    feedback = _feedback(prior_record["certification"]["audits"])
    rescue_attempt = MAX_RESCUE_ATTEMPTS
    try:
        route_ideas, route_generation = _generate_route_ideas(
            endpoint=endpoint,
            model=model,
            problem=problem,
            rescue_attempt=rescue_attempt,
            feedback=feedback,
            timeout=timeout,
        )
        menu, generation = _generate_menu(
            endpoint=endpoint,
            model=model,
            problem=problem,
            attempt=100 + rescue_attempt,
            feedback=feedback,
            route_ideas=route_ideas,
            timeout=timeout,
        )
        audits = [
            _audit_menu(
                endpoint=endpoint,
                model=model,
                problem=problem,
                reference_answer=reference_answer,
                menu=menu,
                seed=seed,
                audit_role=audit_role,
                timeout=timeout,
            )
            for seed, audit_role in zip(
                AUDIT_SEEDS, AUDIT_ROLES, strict=True
            )
        ]
    except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
        attempts.append(
            {
                "phase": "route_rescue",
                "rescue_attempt": rescue_attempt,
                "support_recall_version": SUPPORT_RECALL_VERSION,
                "error": str(exc),
            }
        )
        fallback = dict(prior_record)
        fallback["attempts"] = attempts
        fallback["support_recall_version"] = SUPPORT_RECALL_VERSION
        fallback["support_recall_complete"] = False
        return fallback

    recall_attempt = {
        "phase": "route_rescue",
        "rescue_attempt": rescue_attempt,
        "support_recall_version": SUPPORT_RECALL_VERSION,
        "route_generation": route_generation,
        "route_ideas": route_ideas,
        **generation,
        "proposal_menu": json.loads(menu.canonical_json),
        "audits": audits,
    }
    attempts.append(recall_attempt)
    selected = _maximal_certified_subset(menu, audits)
    if selected is not None and len(selected[0].strategies) >= 2:
        retained_menu, certification = selected
        return {
            "schema": "e49d_strategy_menu_record_v1",
            "row_id": row_id,
            "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
            "reference_answer_sha256": _sha256_bytes(
                reference_answer.encode("utf-8")
            ),
            "problem": problem,
            "menu": json.loads(retained_menu.canonical_json),
            "menu_sha256": retained_menu.sha256,
            "attempts": attempts,
            "certification": certification,
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "support_recall_version": SUPPORT_RECALL_VERSION,
            "support_recall_complete": True,
            "pass": True,
        }

    fallback = dict(prior_record)
    fallback["attempts"] = attempts
    fallback["support_recall_version"] = SUPPORT_RECALL_VERSION
    fallback["support_recall_complete"] = True
    return fallback


def _row_id(split: str, index: int, row: dict[str, Any]) -> str:
    stable = str(row.get("unique_id") or "")
    if not stable:
        stable = _sha256_bytes(
            (
                str(row.get("problem", ""))
                + "\0"
                + str(row.get("answer", ""))
            ).encode("utf-8")
        )[:20]
    return f"{split}:{index:04d}:{stable}"


def _load_existing(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[str(record["row_id"])] = record
    return records


def _seed_prior_menus(
    *,
    source: pathlib.Path,
    destination: pathlib.Path,
    evidence_root: pathlib.Path,
) -> None:
    if destination.exists() or not source.is_file():
        return
    reusable: dict[str, dict[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if (
            record.get("pass") is True
            and isinstance(record.get("menu"), dict)
            and str(record.get("menu_sha256") or "")
        ):
            reusable[str(record["row_id"])] = record
    for row_id in sorted(reusable):
        _append_jsonl(destination, reusable[row_id])
    _write_json(
        evidence_root / "seed_evidence.json",
        {
            "schema": "e49d_seed_menu_evidence_v1",
            "source": str(source),
            "source_sha256": _sha256_bytes(source.read_bytes()),
            "imported_latest_pass_menu_rows": len(reusable),
            "note": (
                "Prior menus are proposals only; every imported row must "
                "receive fresh v4 answer-bound maximal-subset certification."
            ),
        },
    )


def _record_passes_contract(record: dict[str, Any]) -> bool:
    if (
        record.get("pass") is not True
        or record.get("audit_contract_version") != AUDIT_CONTRACT_VERSION
        or not isinstance(record.get("menu"), dict)
        or not str(record.get("menu_sha256") or "")
        or not str(record.get("reference_answer_sha256") or "")
    ):
        return False
    certification = record.get("certification")
    if not isinstance(certification, dict):
        return False
    proposal_payload = certification.get("proposal_menu")
    if not isinstance(proposal_payload, dict):
        return False
    try:
        proposal_json = json.dumps(
            proposal_payload, sort_keys=True, separators=(",", ":")
        )
        proposal = parse_strategy_menu(
            f"x\n{MENU_START}\n{proposal_json}\n{MENU_END}"
        )
        if proposal is None:
            return False
        recomputed = _maximal_certified_subset(
            proposal, certification.get("audits")
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    if recomputed is None:
        return False
    retained_menu, recomputed_certification = recomputed
    return (
        proposal.sha256 == certification.get("proposal_menu_sha256")
        and retained_menu.sha256 == record.get("menu_sha256")
        and retained_menu.sha256
        == certification.get("retained_menu_sha256")
        and recomputed_certification["retained_original_strategy_ids"]
        == certification.get("retained_original_strategy_ids")
        and len(retained_menu.strategies)
        == certification.get("retained_strategy_count")
    )


def _record_has_support_recall(record: dict[str, Any]) -> bool:
    if (
        record.get("support_recall_version") != SUPPORT_RECALL_VERSION
        or record.get("support_recall_complete") is not True
    ):
        return False
    recall_attempts = [
        attempt
        for attempt in record.get("attempts") or []
        if isinstance(attempt, dict)
        and attempt.get("phase") == "route_rescue"
        and attempt.get("rescue_attempt") == MAX_RESCUE_ATTEMPTS
        and attempt.get("support_recall_version")
        == SUPPORT_RECALL_VERSION
    ]
    if not recall_attempts:
        return False
    completed_recall_attempts = []
    for attempt in recall_attempts:
        if attempt.get("error"):
            if (
                isinstance(attempt.get("proposal_menu"), dict)
                or isinstance(attempt.get("audits"), list)
            ):
                return False
            continue
        if not (
            isinstance(attempt.get("proposal_menu"), dict)
            and isinstance(attempt.get("audits"), list)
        ):
            return False
        completed_recall_attempts.append(attempt)
    # Transport errors do not consume the frozen scientific attempt. Exactly
    # one completed request may do so; a second completed request fails closed.
    if len(completed_recall_attempts) != 1:
        return False
    recall = completed_recall_attempts[0]
    try:
        proposal_json = json.dumps(
            recall["proposal_menu"],
            sort_keys=True,
            separators=(",", ":"),
        )
        proposal = parse_strategy_menu(
            f"x\n{MENU_START}\n{proposal_json}\n{MENU_END}"
        )
        if proposal is None:
            return False
        selected = _maximal_certified_subset(proposal, recall["audits"])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    if selected is not None and len(selected[0].strategies) >= 2:
        return (
            len(record["menu"]["strategies"]) >= 2
            and record["menu_sha256"] == selected[0].sha256
        )
    return len(record["menu"]["strategies"]) == 1


def _record_is_materialization_ready(record: dict[str, Any]) -> bool:
    if not _record_passes_contract(record):
        return False
    if len(record["menu"]["strategies"]) >= 2:
        return True
    return _record_has_support_recall(record)


def _embed(problem: str, menu_payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        menu_payload, sort_keys=True, separators=(",", ":")
    )
    provisional = f"{problem}\n\n{MENU_START}\n{canonical}\n{MENU_END}"
    menu = parse_strategy_menu(provisional)
    if menu is None:
        raise RuntimeError("approved menu disappeared during embedding")
    return provisional + strategy_menu_response_instructions(menu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed-evidence", type=pathlib.Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    record_path = args.evidence / "menu_records.jsonl"
    if not args.audit_only and args.seed_evidence is not None:
        _seed_prior_menus(
            source=args.seed_evidence,
            destination=record_path,
            evidence_root=args.evidence,
        )
    if args.audit_only:
        manifest = json.loads(
            (args.output / "MATERIALIZATION_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            manifest.get("audit_contract_version")
            != AUDIT_CONTRACT_VERSION
            or manifest.get("audit_seeds") != list(AUDIT_SEEDS)
            or manifest.get("audit_roles") != list(AUDIT_ROLES)
            or manifest.get("support_recall_version")
            != SUPPORT_RECALL_VERSION
            or manifest.get("all_singletons_support_recall_complete")
            is not True
        ):
            raise RuntimeError("menu audit contract identity changed")
        if manifest["train_tree_sha256"] != _tree_hash(args.output / "train"):
            raise RuntimeError("menu training data changed")
        if manifest["eval_tree_sha256"] != _tree_hash(args.output / "eval"):
            raise RuntimeError("menu evaluation data changed")
        if manifest["source_train_tree_sha256"] != _tree_hash(
            args.source / "train"
        ):
            raise RuntimeError("menu source training data changed")
        if manifest["source_eval_tree_sha256"] != _tree_hash(
            args.source / "eval"
        ):
            raise RuntimeError("menu source evaluation data changed")
        if manifest["menu_records_sha256"] != _sha256_bytes(
            record_path.read_bytes()
        ):
            raise RuntimeError("menu evidence bytes changed")
        records = _load_existing(record_path)
        if len(records) != manifest["menu_count"] or not all(
            _record_is_materialization_ready(record)
            for record in records.values()
        ):
            raise RuntimeError("menu evidence is incomplete")
        source_train = load_from_disk(str(args.source / "train"))
        source_eval = load_from_disk(str(args.source / "eval"))
        output_train = load_from_disk(str(args.output / "train"))
        output_eval = load_from_disk(str(args.output / "eval"))
        split_pairs = (
            (
                "train",
                source_train[manifest["train_split"]],
                output_train[manifest["train_split"]],
            ),
            (
                "eval",
                source_eval[manifest["eval_split"]],
                output_eval[manifest["eval_split"]],
            ),
        )
        observed_ids = set()
        for split, source_dataset, output_dataset in split_pairs:
            if len(source_dataset) != len(output_dataset):
                raise RuntimeError(f"menu {split} row count changed")
            for index, (source_row, output_row) in enumerate(
                zip(source_dataset, output_dataset, strict=True)
            ):
                row_id = _row_id(split, index, source_row)
                observed_ids.add(row_id)
                record = records.get(row_id)
                if record is None:
                    raise RuntimeError(f"menu evidence missing {row_id}")
                original = str(source_row["problem"])
                if (
                    str(output_row["original_problem"]) != original
                    or record["problem_sha256"]
                    != _sha256_bytes(original.encode("utf-8"))
                    or record.get("reference_answer_sha256")
                    != _sha256_bytes(
                        str(source_row["answer"]).encode("utf-8")
                    )
                ):
                    raise RuntimeError(f"menu source binding failed for {row_id}")
                menu = parse_strategy_menu(str(output_row["problem"]))
                if (
                    menu is None
                    or menu.sha256 != record["menu_sha256"]
                    or str(output_row["strategy_menu_sha256"])
                    != record["menu_sha256"]
                ):
                    raise RuntimeError(f"menu binding failed for {row_id}")
        if observed_ids != set(records):
            raise RuntimeError("menu evidence contains unexpected row IDs")
        print(args.output)
        return

    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    endpoint, model = _endpoint(args.endpoint)
    train_dict = load_from_disk(str(args.source / "train"))
    eval_dict = load_from_disk(str(args.source / "eval"))
    train_name = next(iter(train_dict))
    eval_name = next(iter(eval_dict))
    splits = {
        "train": train_dict[train_name],
        "eval": eval_dict[eval_name],
    }
    work = []
    for split, dataset in splits.items():
        for index, row in enumerate(dataset):
            work.append(
                (
                    _row_id(split, index, row),
                    str(row["problem"]),
                    str(row["answer"]),
                )
            )

    existing = _load_existing(record_path)
    for row_id, problem, reference_answer in work:
        record = existing.get(row_id)
        if record is not None and (
            record.get("problem_sha256")
            != _sha256_bytes(problem.encode("utf-8"))
        ):
            raise RuntimeError(f"existing menu evidence drift for {row_id}")
        if (
            record is not None
            and record.get("audit_contract_version")
            == AUDIT_CONTRACT_VERSION
            and record.get("reference_answer_sha256")
            != _sha256_bytes(reference_answer.encode("utf-8"))
        ):
            raise RuntimeError(
                f"existing reference-answer evidence drift for {row_id}"
            )
    pending = [
        (row_id, problem, reference_answer)
        for row_id, problem, reference_answer in work
        if not (
            _record_is_materialization_ready(existing.get(row_id, {}))
        )
    ]
    print(
        f"strategy menus: total={len(work)} complete={len(work)-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _build_one,
                endpoint=endpoint,
                model=model,
                row_id=row_id,
                problem=problem,
                reference_answer=reference_answer,
                timeout=args.timeout,
                prior_record=existing.get(row_id),
            ): row_id
            for row_id, problem, reference_answer in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            _append_jsonl(record_path, record)
            existing[record["row_id"]] = record
            print(
                f"[{completed}/{len(pending)}] {record['row_id']} "
                f"pass={record['pass']}",
                flush=True,
            )

    failures = [
        row_id
        for row_id, _, _ in work
        if not (
            _record_is_materialization_ready(existing[row_id])
        )
    ]
    if failures:
        _write_json(
            args.evidence / "generation_summary.json",
            {
                "schema": "e49d_strategy_menu_generation_summary_v1",
                "pass": False,
                "failures": failures,
            },
        )
        raise RuntimeError(
            f"{len(failures)} strategy menus failed: {failures[:10]}"
        )

    output_splits = {}
    for split, dataset in splits.items():
        data = dataset.to_dict()
        augmented = []
        menu_hashes = []
        for index, row in enumerate(dataset):
            row_id = _row_id(split, index, row)
            record = existing[row_id]
            augmented.append(_embed(str(row["problem"]), record["menu"]))
            menu_hashes.append(record["menu_sha256"])
        data["original_problem"] = list(data["problem"])
        data["problem"] = augmented
        data["strategy_menu_sha256"] = menu_hashes
        output_splits[split] = Dataset.from_dict(data)

    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    try:
        DatasetDict({train_name: output_splits["train"]}).save_to_disk(
            str(staging / "train")
        )
        DatasetDict({eval_name: output_splits["eval"]}).save_to_disk(
            str(staging / "eval")
        )
        retained_counts = [
            len(existing[row_id]["menu"]["strategies"])
            for row_id, _, _ in work
        ]
        manifest = {
            "schema": "e49d_strategy_menu_materialization_v1",
            "source": str(args.source),
            "source_train_tree_sha256": _tree_hash(args.source / "train"),
            "source_eval_tree_sha256": _tree_hash(args.source / "eval"),
            "train_split": train_name,
            "eval_split": eval_name,
            "train_rows": len(splits["train"]),
            "eval_rows": len(splits["eval"]),
            "menu_count": len(work),
            "all_retained_strategies_double_audited": True,
            "singleton_menu_count": sum(
                count == 1 for count in retained_counts
            ),
            "multi_strategy_menu_count": sum(
                count >= 2 for count in retained_counts
            ),
            "multi_strategy_menu_fraction": (
                sum(count >= 2 for count in retained_counts)
                / len(retained_counts)
            ),
            "retained_strategy_count_mean": (
                sum(retained_counts) / len(retained_counts)
            ),
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "audit_seeds": list(AUDIT_SEEDS),
            "audit_roles": list(AUDIT_ROLES),
            "support_recall_version": SUPPORT_RECALL_VERSION,
            "all_singletons_support_recall_complete": True,
            "menu_records_sha256": _sha256_bytes(record_path.read_bytes()),
            "train_tree_sha256": _tree_hash(staging / "train"),
            "eval_tree_sha256": _tree_hash(staging / "eval"),
        }
        _write_json(staging / "MATERIALIZATION_MANIFEST.json", manifest)
        os.replace(staging, args.output)
    finally:
        if staging.exists():
            for item in sorted(staging.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                else:
                    item.rmdir()
            staging.rmdir()
    _write_json(
        args.evidence / "generation_summary.json",
        {
            "schema": "e49d_strategy_menu_generation_summary_v1",
            "pass": True,
            "menu_count": len(work),
            "all_retained_strategies_double_audited": True,
            "singleton_menu_count": sum(
                len(existing[row_id]["menu"]["strategies"]) == 1
                for row_id, _, _ in work
            ),
            "multi_strategy_menu_count": sum(
                len(existing[row_id]["menu"]["strategies"]) >= 2
                for row_id, _, _ in work
            ),
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "support_recall_version": SUPPORT_RECALL_VERSION,
            "all_singletons_support_recall_complete": True,
            "materialization_manifest": str(
                args.output / "MATERIALIZATION_MANIFEST.json"
            ),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
