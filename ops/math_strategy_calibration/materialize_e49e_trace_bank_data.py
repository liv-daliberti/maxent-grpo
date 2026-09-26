#!/usr/bin/env python3
"""Materialize a trace-executed, cross-proposal canonical strategy bank.

E49D audited each generated menu in isolation.  E49E instead unions one
individually certified route from each completed proposal into a prompt-local
candidate bank, requires two new action-by-action soundness traces for every
candidate, and then runs two new equivalence attacks over the surviving bank.
Only the deterministic maximum clique is exposed to the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import pathlib
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.math_answer_normalization import (  # noqa: E402
    NORMALIZATION_VERSION as ANSWER_NORMALIZATION_VERSION,
    audited_answer_matches,
)
from oat_drgrpo.math_strategy_menu import (  # noqa: E402
    MENU_END,
    MENU_SCHEMA,
    MENU_START,
    MathStrategyMenu,
    parse_strategy_menu,
    strategy_menu_response_instructions,
)


_BASE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/materialize_e49d_strategy_menu_data.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location(
    "e49d_materializer_for_e49e",
    _BASE_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError("cannot load the frozen E49D evidence interpreter")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)


TRACE_CONTRACT_VERSION = "cross_proposal_action_trace_v1"
SOUNDNESS_ROLES = ("literal_action_executor", "adversarial_action_checker")
SOUNDNESS_SEEDS = (492111, 492112)
PAIR_ROLES = ("trace_equivalence_attack_a", "trace_equivalence_attack_b")
PAIR_SEEDS = (492121, 492122)
MAX_BANK_STRATEGIES = 3
MAX_BANK_ACTIONS = 12
ACTION_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_])A[1-9][0-9]*(?![A-Za-z0-9_])"
)
SYSTEM = (
    "You are a conservative mathematical proof executor and adversarial "
    "auditor. Return valid JSON only."
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


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


def _tree_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _load_latest(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["row_id"])] = record
    return records


def _menu_from_payload(payload: dict[str, Any]) -> MathStrategyMenu:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    menu = parse_strategy_menu(
        f"x\n{MENU_START}\n{canonical}\n{MENU_END}"
    )
    if menu is None:
        raise ValueError("strategy menu disappeared")
    return menu


def _rewrite_action_references(text: str, mapping: dict[str, str]) -> str:
    return ACTION_REFERENCE_RE.sub(
        lambda match: mapping.get(match.group(0), match.group(0)),
        str(text),
    )


def _extract_closed_candidate(
    proposal: MathStrategyMenu,
    strategy_id: str,
) -> MathStrategyMenu | None:
    """Return one route only when all explicit action references are closed."""

    strategy = proposal.strategy(strategy_id)
    if strategy is None:
        return None
    action_by_id = {
        action.action_id: action.operation for action in proposal.actions
    }
    allowed = set(strategy.action_ids)
    referenced = set(ACTION_REFERENCE_RE.findall(strategy.plan))
    for action_id in strategy.action_ids:
        referenced.update(
            ACTION_REFERENCE_RE.findall(action_by_id[action_id])
        )
    if not referenced <= allowed:
        return None

    retained_actions = [
        action
        for action in proposal.actions
        if action.action_id in allowed
    ]
    mapping = {
        action.action_id: f"A{index}"
        for index, action in enumerate(retained_actions, start=1)
    }
    payload = {
        "schema": MENU_SCHEMA,
        "actions": [
            {
                "action_id": mapping[action.action_id],
                "operation": _rewrite_action_references(
                    action.operation, mapping
                ),
            }
            for action in retained_actions
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": [
                    mapping[action_id] for action_id in strategy.action_ids
                ],
                "plan": _rewrite_action_references(strategy.plan, mapping),
            }
        ],
    }
    return _menu_from_payload(payload)


def _candidate_routes(
    input_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recover deterministic, individually double-sound routes by proposal."""

    candidates: list[dict[str, Any]] = []
    observed: set[str] = set()
    entries: list[tuple[int, str, dict[str, Any], list[dict[str, Any]]]] = []
    durable = input_record.get("certification")
    if (
        isinstance(durable, dict)
        and isinstance(durable.get("proposal_menu"), dict)
        and isinstance(durable.get("audits"), list)
    ):
        entries.append(
            (
                -1,
                "durable_retained_certification",
                durable["proposal_menu"],
                durable["audits"],
            )
        )
    for attempt_index, attempt in enumerate(input_record.get("attempts") or []):
        proposal_payload = attempt.get("proposal_menu")
        audits = attempt.get("audits")
        if not isinstance(proposal_payload, dict) or not isinstance(audits, list):
            continue
        entries.append(
            (
                attempt_index,
                str(
                    attempt.get("phase")
                    or (
                        "route_rescue"
                        if attempt.get("route_rescue")
                        else "direct"
                    )
                ),
                proposal_payload,
                audits,
            )
        )
    for attempt_index, phase, proposal_payload, audits in entries:
        try:
            proposal = _menu_from_payload(proposal_payload)
            selected = base._maximal_certified_subset(proposal, audits)
        except (KeyError, TypeError, ValueError, RuntimeError):
            continue
        if selected is None:
            continue
        certification = selected[1]
        for original_strategy_id in certification[
            "retained_original_strategy_ids"
        ]:
            candidate = _extract_closed_candidate(
                proposal, original_strategy_id
            )
            if candidate is None or candidate.sha256 in observed:
                continue
            observed.add(candidate.sha256)
            candidates.append(
                {
                    "menu": candidate,
                    "source": {
                        "attempt_index": attempt_index,
                        "phase": phase,
                        "proposal_menu_sha256": proposal.sha256,
                        "original_strategy_id": original_strategy_id,
                        "candidate_menu_sha256": candidate.sha256,
                    },
                }
            )
    return candidates


def _choose_candidate_subset(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose a fixed bounded bank, favoring proposal diversity then brevity."""

    best: tuple[tuple[Any, ...], tuple[dict[str, Any], ...]] | None = None
    limit = min(MAX_BANK_STRATEGIES, len(candidates))
    for size in range(1, limit + 1):
        for indices in itertools.combinations(range(len(candidates)), size):
            selected = tuple(candidates[index] for index in indices)
            total_actions = sum(
                len(candidate["menu"].actions) for candidate in selected
            )
            if total_actions > MAX_BANK_ACTIONS:
                continue
            proposal_count = len(
                {
                    candidate["source"]["attempt_index"]
                    for candidate in selected
                }
            )
            score = (
                size,
                proposal_count,
                -total_actions,
                tuple(-index for index in indices),
            )
            if best is None or score > best[0]:
                best = (score, selected)
    return list(best[1]) if best is not None else []


def _merge_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[MathStrategyMenu, list[dict[str, Any]]]:
    actions = []
    strategies = []
    sources = []
    next_action = 1
    for strategy_index, candidate in enumerate(candidates, start=1):
        menu: MathStrategyMenu = candidate["menu"]
        strategy = menu.strategies[0]
        mapping = {
            action.action_id: f"A{next_action + offset}"
            for offset, action in enumerate(menu.actions)
        }
        actions.extend(
            {
                "action_id": mapping[action.action_id],
                "operation": _rewrite_action_references(
                    action.operation, mapping
                ),
            }
            for action in menu.actions
        )
        strategies.append(
            {
                "strategy_id": f"S{strategy_index}",
                "action_ids": [
                    mapping[action_id] for action_id in strategy.action_ids
                ],
                "plan": _rewrite_action_references(
                    strategy.plan, mapping
                ),
            }
        )
        source = dict(candidate["source"])
        source["bank_strategy_id"] = f"S{strategy_index}"
        sources.append(source)
        next_action += len(menu.actions)
    return _menu_from_payload(
        {
            "schema": MENU_SCHEMA,
            "actions": actions,
            "strategies": strategies,
        }
    ), sources


def _candidate_bank(
    input_record: dict[str, Any],
) -> tuple[MathStrategyMenu, list[dict[str, Any]]] | None:
    selected = _choose_candidate_subset(_candidate_routes(input_record))
    if not selected:
        return None
    return _merge_candidates(selected)


def _prune_menu_closed(
    menu: MathStrategyMenu,
    retained_ids: tuple[str, ...],
) -> MathStrategyMenu:
    retained = [
        strategy
        for strategy in menu.strategies
        if strategy.strategy_id in retained_ids
    ]
    if len(retained) != len(retained_ids):
        raise RuntimeError("retained strategy disappeared during pruning")
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
    return _menu_from_payload(
        {
            "schema": MENU_SCHEMA,
            "actions": [
                {
                    "action_id": action_map[action.action_id],
                    "operation": _rewrite_action_references(
                        action.operation,
                        action_map,
                    ),
                }
                for action in retained_actions
            ],
            "strategies": [
                {
                    "strategy_id": f"S{index}",
                    "action_ids": [
                        action_map[action_id]
                        for action_id in strategy.action_ids
                    ],
                    "plan": _rewrite_action_references(
                        strategy.plan,
                        action_map,
                    ),
                }
                for index, strategy in enumerate(retained, start=1)
            ],
        }
    )


def _load_known_invalid_controls(
    path: pathlib.Path,
    input_records: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "stage", "controls"}
        or payload.get("schema") != "e49e_known_invalid_controls_v1"
        or payload.get("stage") not in {"toy", "full"}
        or not isinstance(payload.get("controls"), list)
        or not payload["controls"]
    ):
        raise RuntimeError("invalid E49E known-invalid control manifest")
    required = {
        "control_id",
        "row_id",
        "candidate_menu_sha256",
        "strategy_id",
        "failure_class",
        "reason",
    }
    controls = []
    observed_ids = set()
    for raw in payload["controls"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != required
            or any(
                not isinstance(raw[key], str) or not raw[key].strip()
                for key in required
            )
            or raw["control_id"] in observed_ids
            or raw["row_id"] not in input_records
        ):
            raise RuntimeError("invalid E49E known-invalid control row")
        bank = _candidate_bank(input_records[raw["row_id"]])
        if bank is None:
            raise RuntimeError("known-invalid control bank disappeared")
        menu, _ = bank
        if (
            menu.sha256 != raw["candidate_menu_sha256"]
            or menu.strategy(raw["strategy_id"]) is None
        ):
            raise RuntimeError(
                f"known-invalid control binding changed: {raw['control_id']}"
            )
        observed_ids.add(raw["control_id"])
        controls.append(dict(raw))
    return controls


def _load_known_equivalent_controls(
    path: pathlib.Path,
    input_records: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "stage", "controls"}
        or payload.get("schema")
        != "e49e_known_equivalent_controls_v1"
        or payload.get("stage") not in {"toy", "full"}
        or not isinstance(payload.get("controls"), list)
        or not payload["controls"]
    ):
        raise RuntimeError("invalid E49E known-equivalent control manifest")
    required = {
        "control_id",
        "row_id",
        "candidate_menu_sha256",
        "left_strategy_id",
        "right_strategy_id",
        "reason",
    }
    controls = []
    observed_ids = set()
    for raw in payload["controls"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != required
            or any(
                not isinstance(raw[key], str) or not raw[key].strip()
                for key in required
            )
            or raw["control_id"] in observed_ids
            or raw["row_id"] not in input_records
            or raw["left_strategy_id"] == raw["right_strategy_id"]
        ):
            raise RuntimeError("invalid E49E known-equivalent control row")
        bank = _candidate_bank(input_records[raw["row_id"]])
        if bank is None:
            raise RuntimeError("known-equivalent control bank disappeared")
        menu, _ = bank
        if (
            menu.sha256 != raw["candidate_menu_sha256"]
            or menu.strategy(raw["left_strategy_id"]) is None
            or menu.strategy(raw["right_strategy_id"]) is None
        ):
            raise RuntimeError(
                f"known-equivalent control binding changed: "
                f"{raw['control_id']}"
            )
        observed_ids.add(raw["control_id"])
        controls.append(dict(raw))
    return controls


def _bounded_text(
    row: dict[str, Any],
    key: str,
    maximum: int,
) -> bool:
    value = row.get(key)
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _sound_schema(action_ids: list[str], strategy_id: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "strategy_id",
            "action_executions",
            "uses_only_declared_actions",
            "self_contained_without_other_strategy",
            "derived_answer",
            "matches_reference_answer",
            "status",
            "decisive_check",
        ],
        "properties": {
            "strategy_id": {"type": "string", "enum": [strategy_id]},
            "action_executions": {
                "type": "array",
                "minItems": len(action_ids),
                "maxItems": len(action_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "action_id",
                        "executed_calculation",
                        "output_fact",
                        "status",
                        "brief_check",
                    ],
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "enum": action_ids,
                        },
                        "executed_calculation": {"type": "string"},
                        "output_fact": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["valid", "invalid", "ambiguous"],
                        },
                        "brief_check": {"type": "string"},
                    },
                },
            },
            "uses_only_declared_actions": {"type": "boolean"},
            "self_contained_without_other_strategy": {"type": "boolean"},
            "derived_answer": {"type": "string"},
            "matches_reference_answer": {"type": "boolean"},
            "status": {
                "type": "string",
                "enum": ["sound", "unsound", "ambiguous"],
            },
            "decisive_check": {"type": "string"},
        },
    }


def _sound_assessment_is_well_formed(
    assessment: Any,
    *,
    action_ids: list[str],
    strategy_id: str,
) -> bool:
    required = {
        "strategy_id",
        "action_executions",
        "uses_only_declared_actions",
        "self_contained_without_other_strategy",
        "derived_answer",
        "matches_reference_answer",
        "status",
        "decisive_check",
    }
    if not isinstance(assessment, dict) or set(assessment) != required:
        return False
    rows = assessment.get("action_executions")
    row_required = {
        "action_id",
        "executed_calculation",
        "output_fact",
        "status",
        "brief_check",
    }
    return (
        assessment.get("strategy_id") == strategy_id
        and isinstance(rows, list)
        and [row.get("action_id") for row in rows] == action_ids
        and all(
            isinstance(row, dict)
            and set(row) == row_required
            and row.get("status") in {"valid", "invalid", "ambiguous"}
            and _bounded_text(row, "executed_calculation", 1600)
            and _bounded_text(row, "output_fact", 600)
            and _bounded_text(row, "brief_check", 600)
            for row in rows
        )
        and type(assessment.get("uses_only_declared_actions")) is bool
        and type(
            assessment.get("self_contained_without_other_strategy")
        )
        is bool
        and type(assessment.get("matches_reference_answer")) is bool
        and assessment.get("status") in {
            "sound",
            "unsound",
            "ambiguous",
        }
        and _bounded_text(assessment, "derived_answer", 256)
        and _bounded_text(assessment, "decisive_check", 1000)
    )


def _sound_request(
    *,
    endpoint: str,
    model: str,
    problem: str,
    reference_answer: str,
    menu: MathStrategyMenu,
    strategy_id: str,
    role: str,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    strategy = menu.strategy(strategy_id)
    if strategy is None:
        raise ValueError("unknown candidate strategy")
    action_map = {
        action.action_id: action.operation for action in menu.actions
    }
    actions = "\n".join(
        f"- {action_id}: {action_map[action_id]}"
        for action_id in strategy.action_ids
    )
    role_instruction = {
        "literal_action_executor": (
            "Literally execute each declared action yourself in order. "
            "Do not import an operation absent from the sequence."
        ),
        "adversarial_action_checker": (
            "Try to falsify the route. Check every arithmetic step, theorem "
            "condition, domain/case, sufficiency claim, and hidden dependency."
        ),
    }[role]
    prompt = f"""Audit exactly one proposed solution route.

{role_instruction}

For each action, show the concrete calculation or proof operation it performs
and the resulting fact. The next action must start from facts actually
established earlier. A vague instruction, unsupported identity, invalid
geometric relation, missing bound/case, circular verification, or reference
to another strategy makes the route unsound. A route is not self-contained if
it verifies a result obtained by an omitted route, silently performs an
undeclared decisive operation, or derives only a residue class/bound when an
exact answer is requested. Do not repair or supplement the route.

PROBLEM:
{problem}

AUDITOR-ONLY REFERENCE ANSWER:
{reference_answer}

TARGET STRATEGY:
{strategy.strategy_id} = {strategy.action_combo}
PLAN: {strategy.plan}
ACTIONS:
{actions}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_action_trace_soundness",
                "strict": True,
                "schema": _sound_schema(
                    list(strategy.action_ids), strategy_id
                ),
            },
        },
    }
    response, content = base._post(endpoint, payload, timeout=timeout)
    common = {
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "kind": "soundness",
        "role": role,
        "seed": seed,
        "candidate_menu_sha256": menu.sha256,
        "strategy_id": strategy_id,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "content_sha256": _sha256_bytes(content.encode("utf-8")),
    }
    try:
        assessment = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            **common,
            "assessment": None,
            "completed_invalid": True,
            "error": str(exc),
            "pass": False,
        }
    well_formed = _sound_assessment_is_well_formed(
        assessment,
        action_ids=list(strategy.action_ids),
        strategy_id=strategy_id,
    )
    answer_pass = False
    if well_formed:
        answer_pass = audited_answer_matches(
            assessment["derived_answer"],
            reference_answer,
        )
    passed = bool(
        well_formed
        and common["finish_reason"] == "stop"
        and str(common["response_id"] or "")
        and all(
            row["status"] == "valid"
            for row in assessment["action_executions"]
        )
        and assessment["uses_only_declared_actions"] is True
        and assessment["self_contained_without_other_strategy"] is True
        and assessment["matches_reference_answer"] is True
        and assessment["status"] == "sound"
        and answer_pass
    )
    return {
        **common,
        "assessment": assessment,
        "answer_validator_pass": answer_pass,
        "completed_invalid": not well_formed,
        "pass": passed,
    }


def _pair_schema(pair_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["pair_assessments"],
        "properties": {
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
                        "reducible_by_routine_algebra",
                        "both_routes_self_contained",
                        "brief_check",
                    ],
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "enum": pair_ids,
                        },
                        "relation": {
                            "type": "string",
                            "enum": [
                                "distinct",
                                "equivalent",
                                "ambiguous",
                            ],
                        },
                        "shared_core": {"type": "string"},
                        "left_decisive_operation": {"type": "string"},
                        "right_decisive_operation": {"type": "string"},
                        "reducible_by_routine_algebra": {
                            "type": "boolean"
                        },
                        "both_routes_self_contained": {"type": "boolean"},
                        "brief_check": {"type": "string"},
                    },
                },
            }
        },
    }


def _pair_assessment_is_well_formed(
    assessment: Any,
    pair_ids: list[str],
) -> bool:
    if (
        not isinstance(assessment, dict)
        or set(assessment) != {"pair_assessments"}
    ):
        return False
    rows = assessment["pair_assessments"]
    required = {
        "pair_id",
        "relation",
        "shared_core",
        "left_decisive_operation",
        "right_decisive_operation",
        "reducible_by_routine_algebra",
        "both_routes_self_contained",
        "brief_check",
    }
    return (
        isinstance(rows, list)
        and [row.get("pair_id") for row in rows] == pair_ids
        and all(
            isinstance(row, dict)
            and set(row) == required
            and row.get("relation")
            in {"distinct", "equivalent", "ambiguous"}
            and type(row.get("reducible_by_routine_algebra")) is bool
            and type(row.get("both_routes_self_contained")) is bool
            and _bounded_text(row, "shared_core", 600)
            and _bounded_text(row, "left_decisive_operation", 600)
            and _bounded_text(row, "right_decisive_operation", 600)
            and _bounded_text(row, "brief_check", 1200)
            for row in rows
        )
    )


def _pair_request(
    *,
    endpoint: str,
    model: str,
    problem: str,
    reference_answer: str,
    menu: MathStrategyMenu,
    sound_audits: dict[str, list[dict[str, Any]]],
    eligible_ids: list[str],
    role: str,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    pair_ids = [
        f"{left}__{right}"
        for left, right in itertools.combinations(eligible_ids, 2)
    ]
    strategy_payload = []
    action_map = {
        action.action_id: action.operation for action in menu.actions
    }
    for strategy_id in eligible_ids:
        strategy = menu.strategy(strategy_id)
        assert strategy is not None
        strategy_payload.append(
            {
                "strategy_id": strategy_id,
                "action_combo": strategy.action_combo,
                "plan": strategy.plan,
                "actions": [
                    {
                        "action_id": action_id,
                        "operation": action_map[action_id],
                    }
                    for action_id in strategy.action_ids
                ],
                "independent_execution_traces": [
                    audit["assessment"] for audit in sound_audits[strategy_id]
                ],
            }
        )
    prompt = f"""Adversarially compare every required pair of already
action-traced solution routes.

Equivalent is the default. Call a pair distinct only if both routes are
self-contained and use genuinely different decisive theorems, invariants,
constructions, substitutions, counting spaces, or proof structures. Routine
algebra, different notation, a redundant check, adding an unnecessary step,
enumerating the same arithmetic, or verifying one route with the other is
equivalent. In particular, decimals versus fractions, dollars versus cents,
unit conversion before versus after the same calculation, direct expansion
versus writing the identical formula, reordered equations, and naming the
same operation with different action or kernel labels are equivalent.
Different declared kernel labels are not evidence of novelty. A distinct pair
must establish different route-exclusive intermediate facts through
non-routine operations; translating one trace to the other by representation
conversion or ordinary algebra must not suffice. If either trace contains a
hidden dependency or mathematical defect missed earlier, mark the pair
ambiguous and set both_routes_self_contained=false. Name the shared core and
one exclusive decisive operation actually executed by each route.

ROLE: {role}

PROBLEM:
{problem}

AUDITOR-ONLY REFERENCE ANSWER:
{reference_answer}

TRACE-CERTIFIED CANDIDATES:
{json.dumps(strategy_payload, sort_keys=True)}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_trace_pair_equivalence",
                "strict": True,
                "schema": _pair_schema(pair_ids),
            },
        },
    }
    response, content = base._post(endpoint, payload, timeout=timeout)
    common = {
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "kind": "pair_equivalence",
        "role": role,
        "seed": seed,
        "candidate_menu_sha256": menu.sha256,
        "eligible_strategy_ids": eligible_ids,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "content_sha256": _sha256_bytes(content.encode("utf-8")),
    }
    try:
        assessment = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            **common,
            "assessment": None,
            "completed_invalid": True,
            "error": str(exc),
            "pass": False,
        }
    well_formed = _pair_assessment_is_well_formed(
        assessment, pair_ids
    )
    return {
        **common,
        "assessment": assessment,
        "completed_invalid": not well_formed,
        "pass": bool(well_formed and common["finish_reason"] == "stop"),
    }


def _cache_path(
    cache_root: pathlib.Path,
    *,
    row_id: str,
    kind: str,
    strategy_id: str,
    seed: int,
    menu_sha256: str,
) -> pathlib.Path:
    row_hash = _sha256_bytes(row_id.encode("utf-8"))[:20]
    return (
        cache_root
        / row_hash
        / f"{kind}-{strategy_id}-{seed}-{menu_sha256[:16]}.json"
    )


def _cached_request(
    cache_path: pathlib.Path,
    request_fn,
    **kwargs,
) -> dict[str, Any]:
    if cache_path.is_file():
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        if record.get("trace_contract_version") != TRACE_CONTRACT_VERSION:
            raise RuntimeError("cached trace contract changed")
        return record
    record = request_fn(**kwargs)
    _write_json(cache_path, record)
    return record


def _sound_record_passes(
    menu: MathStrategyMenu,
    strategy_id: str,
    record: dict[str, Any],
    *,
    reference_answer: str | None,
) -> bool:
    strategy = menu.strategy(strategy_id)
    if strategy is None:
        return False
    if (
        record.get("trace_contract_version") != TRACE_CONTRACT_VERSION
        or record.get("kind") != "soundness"
        or record.get("role") not in SOUNDNESS_ROLES
        or record.get("seed") not in SOUNDNESS_SEEDS
        or record.get("candidate_menu_sha256") != menu.sha256
        or record.get("strategy_id") != strategy_id
        or record.get("finish_reason") != "stop"
        or not str(record.get("response_id") or "")
        or not str(record.get("content_sha256") or "")
        or not _sound_assessment_is_well_formed(
            record.get("assessment"),
            action_ids=list(strategy.action_ids),
            strategy_id=strategy_id,
        )
    ):
        return False
    assessment = record["assessment"]
    answer_pass = record.get("answer_validator_pass") is True
    if reference_answer is not None:
        answer_pass = audited_answer_matches(
            assessment["derived_answer"],
            reference_answer,
        )
    return bool(
        answer_pass
        and all(
            row["status"] == "valid"
            for row in assessment["action_executions"]
        )
        and assessment["uses_only_declared_actions"] is True
        and assessment["self_contained_without_other_strategy"] is True
        and assessment["matches_reference_answer"] is True
        and assessment["status"] == "sound"
    )


def _pair_record_well_formed(
    menu: MathStrategyMenu,
    eligible_ids: list[str],
    record: dict[str, Any],
) -> bool:
    pair_ids = [
        f"{left}__{right}"
        for left, right in itertools.combinations(eligible_ids, 2)
    ]
    return bool(
        record.get("trace_contract_version") == TRACE_CONTRACT_VERSION
        and record.get("kind") == "pair_equivalence"
        and record.get("role") in PAIR_ROLES
        and record.get("seed") in PAIR_SEEDS
        and record.get("candidate_menu_sha256") == menu.sha256
        and record.get("eligible_strategy_ids") == eligible_ids
        and record.get("finish_reason") == "stop"
        and str(record.get("response_id") or "")
        and record.get("pass") is True
        and _pair_assessment_is_well_formed(
            record.get("assessment"), pair_ids
        )
    )


def _maximal_trace_certified_subset(
    menu: MathStrategyMenu,
    sound_audits: dict[str, list[dict[str, Any]]],
    pair_audits: list[dict[str, Any]],
    *,
    reference_answer: str | None,
) -> tuple[MathStrategyMenu, dict[str, Any]] | None:
    eligible_ids = [
        strategy.strategy_id
        for strategy in menu.strategies
        if len(sound_audits.get(strategy.strategy_id) or []) == 2
        and {
            (audit.get("seed"), audit.get("role"))
            for audit in sound_audits[strategy.strategy_id]
        }
        == set(zip(SOUNDNESS_SEEDS, SOUNDNESS_ROLES, strict=True))
        and all(
            _sound_record_passes(
                menu,
                strategy.strategy_id,
                audit,
                reference_answer=reference_answer,
            )
            for audit in sound_audits[strategy.strategy_id]
        )
    ]
    if not eligible_ids:
        return None
    distinct_edges: set[tuple[str, str]] = set()
    if len(eligible_ids) >= 2:
        if (
            len(pair_audits) != 2
            or {
                (audit.get("seed"), audit.get("role"))
                for audit in pair_audits
            }
            != set(zip(PAIR_SEEDS, PAIR_ROLES, strict=True))
            or not all(
                _pair_record_well_formed(menu, eligible_ids, audit)
                for audit in pair_audits
            )
        ):
            # Sound candidates remain eligible, but no pair may manufacture
            # novelty when either pair audit is missing or malformed.
            pair_audits = []
        if pair_audits:
            per_audit_edges = []
            for audit in pair_audits:
                edges = set()
                for row in audit["assessment"]["pair_assessments"]:
                    left, right = row["pair_id"].split("__", maxsplit=1)
                    if (
                        row["relation"] == "distinct"
                        and row["reducible_by_routine_algebra"] is False
                        and row["both_routes_self_contained"] is True
                        and row["left_decisive_operation"].strip().casefold()
                        != row["right_decisive_operation"].strip().casefold()
                    ):
                        edges.add((left, right))
                per_audit_edges.append(edges)
            distinct_edges = set.intersection(*per_audit_edges)

    retained: tuple[str, ...] | None = None
    for size in range(len(eligible_ids), 0, -1):
        for candidate in itertools.combinations(eligible_ids, size):
            if all(
                (left, right) in distinct_edges
                for left, right in itertools.combinations(candidate, 2)
            ):
                retained = candidate
                break
        if retained is not None:
            break
    if retained is None:
        return None
    pruned = _prune_menu_closed(menu, retained)
    certification = {
        "schema": "e49e_trace_certified_bank_v1",
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "candidate_menu": json.loads(menu.canonical_json),
        "candidate_menu_sha256": menu.sha256,
        "sound_audits": sound_audits,
        "pair_audits": pair_audits,
        "eligible_strategy_ids": eligible_ids,
        "retained_original_strategy_ids": list(retained),
        "retained_menu_sha256": pruned.sha256,
        "retained_strategy_count": len(retained),
    }
    return pruned, certification


def _build_one(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    problem: str,
    reference_answer: str,
    input_record: dict[str, Any],
    cache_root: pathlib.Path,
    timeout: int,
) -> dict[str, Any]:
    bank = _candidate_bank(input_record)
    common = {
        "schema": "e49e_trace_bank_record_v1",
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "row_id": row_id,
        "problem": problem,
        "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": _sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "input_record_sha256": _canonical_sha256(input_record),
        "input_record_menu_sha256": input_record.get("menu_sha256"),
    }
    if bank is None:
        return {**common, "error": "no structurally closed candidate", "pass": False}
    menu, sources = bank
    sound_audits: dict[str, list[dict[str, Any]]] = {}
    for strategy in menu.strategies:
        sound_audits[strategy.strategy_id] = []
        for seed, role in zip(
            SOUNDNESS_SEEDS, SOUNDNESS_ROLES, strict=True
        ):
            cache_path = _cache_path(
                cache_root,
                row_id=row_id,
                kind="soundness",
                strategy_id=strategy.strategy_id,
                seed=seed,
                menu_sha256=menu.sha256,
            )
            sound_audits[strategy.strategy_id].append(
                _cached_request(
                    cache_path,
                    _sound_request,
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    strategy_id=strategy.strategy_id,
                    role=role,
                    seed=seed,
                    timeout=timeout,
                )
            )

    eligible_ids = [
        strategy.strategy_id
        for strategy in menu.strategies
        if all(
            _sound_record_passes(
                menu,
                strategy.strategy_id,
                audit,
                reference_answer=reference_answer,
            )
            for audit in sound_audits[strategy.strategy_id]
        )
    ]
    pair_audits = []
    if len(eligible_ids) >= 2:
        for seed, role in zip(PAIR_SEEDS, PAIR_ROLES, strict=True):
            cache_path = _cache_path(
                cache_root,
                row_id=row_id,
                kind="pair",
                strategy_id="-".join(eligible_ids),
                seed=seed,
                menu_sha256=menu.sha256,
            )
            pair_audits.append(
                _cached_request(
                    cache_path,
                    _pair_request,
                    endpoint=endpoint,
                    model=model,
                    problem=problem,
                    reference_answer=reference_answer,
                    menu=menu,
                    sound_audits=sound_audits,
                    eligible_ids=eligible_ids,
                    role=role,
                    seed=seed,
                    timeout=timeout,
                )
            )
    selected = _maximal_trace_certified_subset(
        menu,
        sound_audits,
        pair_audits,
        reference_answer=reference_answer,
    )
    if selected is None:
        return {
            **common,
            "candidate_sources": sources,
            "candidate_menu": json.loads(menu.canonical_json),
            "candidate_menu_sha256": menu.sha256,
            "sound_audits": sound_audits,
            "pair_audits": pair_audits,
            "error": "no trace-certified route",
            "pass": False,
        }
    retained, certification = selected
    return {
        **common,
        "candidate_sources": sources,
        "menu": json.loads(retained.canonical_json),
        "menu_sha256": retained.sha256,
        "certification": certification,
        "pass": True,
    }


def _record_passes_contract(
    record: dict[str, Any],
    *,
    reference_answer: str | None = None,
    input_record: dict[str, Any] | None = None,
) -> bool:
    if (
        record.get("pass") is not True
        or record.get("trace_contract_version") != TRACE_CONTRACT_VERSION
        or not isinstance(record.get("menu"), dict)
        or not str(record.get("menu_sha256") or "")
    ):
        return False
    certification = record.get("certification")
    if not isinstance(certification, dict):
        return False
    try:
        candidate = _menu_from_payload(certification["candidate_menu"])
        if input_record is not None:
            expected_bank = _candidate_bank(input_record)
            if expected_bank is None:
                return False
            expected_candidate, expected_sources = expected_bank
            if (
                record.get("input_record_sha256")
                != _canonical_sha256(input_record)
                or candidate.sha256 != expected_candidate.sha256
                or record.get("candidate_sources") != expected_sources
            ):
                return False
        selected = _maximal_trace_certified_subset(
            candidate,
            certification["sound_audits"],
            certification["pair_audits"],
            reference_answer=reference_answer,
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    if selected is None:
        return False
    retained, recomputed = selected
    return bool(
        candidate.sha256 == certification.get("candidate_menu_sha256")
        and retained.sha256 == record["menu_sha256"]
        and retained.sha256 == certification.get("retained_menu_sha256")
        and recomputed["retained_original_strategy_ids"]
        == certification.get("retained_original_strategy_ids")
        and len(retained.strategies)
        == certification.get("retained_strategy_count")
    )


def _known_invalid_control_results(
    records: dict[str, dict[str, Any]],
    controls: list[dict[str, str]],
    input_records: dict[str, dict[str, Any]],
    references: dict[str, str],
) -> list[dict[str, Any]]:
    results = []
    expected_auditors = set(
        zip(SOUNDNESS_SEEDS, SOUNDNESS_ROLES, strict=True)
    )
    for control in controls:
        row_id = control["row_id"]
        bank = _candidate_bank(input_records[row_id])
        assert bank is not None
        menu, _ = bank
        strategy_id = control["strategy_id"]
        record = records.get(row_id) or {}
        certification = record.get("certification")
        if isinstance(certification, dict):
            all_sound = certification.get("sound_audits")
        else:
            all_sound = record.get("sound_audits")
        audits = (
            all_sound.get(strategy_id)
            if isinstance(all_sound, dict)
            and isinstance(all_sound.get(strategy_id), list)
            else []
        )
        completed = bool(
            len(audits) == 2
            and {
                (audit.get("seed"), audit.get("role"))
                for audit in audits
            }
            == expected_auditors
            and all(
                str(audit.get("response_id") or "")
                and str(audit.get("content_sha256") or "")
                for audit in audits
            )
        )
        double_sound = bool(
            completed
            and all(
                _sound_record_passes(
                    menu,
                    strategy_id,
                    audit,
                    reference_answer=references[row_id],
                )
                for audit in audits
            )
        )
        results.append(
            {
                **control,
                "audits_complete": completed,
                "double_sound_accept": double_sound,
                "rejected_by_soundness": completed and not double_sound,
                "audit_decisions": [
                    {
                        "role": audit.get("role"),
                        "seed": audit.get("seed"),
                        "pass": audit.get("pass") is True,
                        "completed_invalid": (
                            audit.get("completed_invalid") is True
                        ),
                        "content_sha256": audit.get("content_sha256"),
                    }
                    for audit in audits
                ],
            }
        )
    return results


def _known_equivalent_control_results(
    records: dict[str, dict[str, Any]],
    controls: list[dict[str, str]],
    input_records: dict[str, dict[str, Any]],
    references: dict[str, str],
) -> list[dict[str, Any]]:
    results = []
    expected_sound = set(
        zip(SOUNDNESS_SEEDS, SOUNDNESS_ROLES, strict=True)
    )
    expected_pairs = set(zip(PAIR_SEEDS, PAIR_ROLES, strict=True))
    for control in controls:
        row_id = control["row_id"]
        bank = _candidate_bank(input_records[row_id])
        assert bank is not None
        menu, _ = bank
        record = records.get(row_id) or {}
        certification = record.get("certification")
        container = (
            certification if isinstance(certification, dict) else record
        )
        all_sound = container.get("sound_audits")
        all_pairs = container.get("pair_audits")
        all_sound = all_sound if isinstance(all_sound, dict) else {}
        pair_audits = all_pairs if isinstance(all_pairs, list) else []

        ids = [
            control["left_strategy_id"],
            control["right_strategy_id"],
        ]
        sound_complete = True
        sound_pass = True
        for strategy_id in ids:
            audits = all_sound.get(strategy_id)
            complete = bool(
                isinstance(audits, list)
                and len(audits) == 2
                and {
                    (audit.get("seed"), audit.get("role"))
                    for audit in audits
                }
                == expected_sound
                and all(
                    str(audit.get("response_id") or "")
                    and str(audit.get("content_sha256") or "")
                    for audit in audits
                )
            )
            sound_complete = sound_complete and complete
            sound_pass = sound_pass and bool(
                complete
                and all(
                    _sound_record_passes(
                        menu,
                        strategy_id,
                        audit,
                        reference_answer=references[row_id],
                    )
                    for audit in audits
                )
            )

        eligible_ids = [
            strategy.strategy_id
            for strategy in menu.strategies
            if len(all_sound.get(strategy.strategy_id) or []) == 2
            and all(
                _sound_record_passes(
                    menu,
                    strategy.strategy_id,
                    audit,
                    reference_answer=references[row_id],
                )
                for audit in all_sound[strategy.strategy_id]
            )
        ]
        pair_complete = bool(
            sound_pass
            and set(ids) <= set(eligible_ids)
            and len(pair_audits) == 2
            and {
                (audit.get("seed"), audit.get("role"))
                for audit in pair_audits
            }
            == expected_pairs
            and all(
                _pair_record_well_formed(menu, eligible_ids, audit)
                for audit in pair_audits
            )
        )
        pair_id = "__".join(
            strategy_id
            for strategy_id in eligible_ids
            if strategy_id in set(ids)
        )
        decisions = []
        if pair_complete:
            for audit in pair_audits:
                row = next(
                    item
                    for item in audit["assessment"]["pair_assessments"]
                    if item["pair_id"] == pair_id
                )
                decisions.append(
                    {
                        "role": audit["role"],
                        "seed": audit["seed"],
                        "relation": row["relation"],
                        "reducible_by_routine_algebra": row[
                            "reducible_by_routine_algebra"
                        ],
                        "both_routes_self_contained": row[
                            "both_routes_self_contained"
                        ],
                        "brief_check": row["brief_check"],
                    }
                )
        double_distinct = bool(
            pair_complete
            and all(
                decision["relation"] == "distinct"
                and decision["reducible_by_routine_algebra"] is False
                and decision["both_routes_self_contained"] is True
                for decision in decisions
            )
        )
        results.append(
            {
                **control,
                "sound_audits_complete": sound_complete,
                "both_routes_sound": sound_pass,
                "pair_audits_complete": pair_complete,
                "double_distinct_false_new": double_distinct,
                # A control is safely rejected either when a completed
                # soundness audit vetoes one of its routes, or when both
                # sound routes receive complete pair audits that do not
                # unanimously certify a distinct edge.
                "rejected_as_new": sound_complete
                and (
                    not sound_pass
                    or (pair_complete and not double_distinct)
                ),
                "audit_decisions": decisions,
            }
        )
    return results


def _embed(problem: str, menu_payload: dict[str, Any]) -> str:
    menu = _menu_from_payload(menu_payload)
    return (
        f"{problem}\n\n{MENU_START}\n{menu.canonical_json}\n{MENU_END}"
        + strategy_menu_response_instructions(menu)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--input-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--known-invalid-controls",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--known-equivalent-controls",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only and args.preflight_only:
        raise ValueError("--audit-only and --preflight-only are exclusive")

    input_path = args.input_evidence / "menu_records.jsonl"
    record_path = args.evidence / "trace_bank_records.jsonl"
    cache_root = args.evidence / "request_cache"
    input_records = _load_latest(input_path)

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
            row_id = base._row_id(split, index, row)
            work.append(
                (
                    split,
                    index,
                    row_id,
                    str(row["problem"]),
                    str(row["answer"]),
                )
            )
    expected_ids = {row_id for _, _, row_id, _, _ in work}
    if set(input_records) != expected_ids:
        raise RuntimeError("E49D input evidence row IDs do not match source")
    controls = _load_known_invalid_controls(
        args.known_invalid_controls,
        input_records,
    )
    equivalent_controls = _load_known_equivalent_controls(
        args.known_equivalent_controls,
        input_records,
    )
    references = {
        row_id: reference for _, _, row_id, _, reference in work
    }

    if args.preflight_only:
        bank_rows = []
        for split, _, row_id, _, _ in work:
            bank = _candidate_bank(input_records[row_id])
            if bank is None:
                bank_rows.append(
                    {
                        "row_id": row_id,
                        "split": split,
                        "candidate_count": 0,
                        "candidate_menu_sha256": None,
                    }
                )
                continue
            candidate, _ = bank
            bank_rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "candidate_count": len(candidate.strategies),
                    "candidate_menu_sha256": candidate.sha256,
                }
            )
        counts = {
            str(count): sum(
                row["candidate_count"] == count for row in bank_rows
            )
            for count in range(MAX_BANK_STRATEGIES + 1)
        }
        payload = {
            "schema": "e49e_trace_bank_preflight_v1",
            "trace_contract_version": TRACE_CONTRACT_VERSION,
            "answer_normalization_version": ANSWER_NORMALIZATION_VERSION,
            "source_train_tree_sha256": _tree_hash(args.source / "train"),
            "source_eval_tree_sha256": _tree_hash(args.source / "eval"),
            "input_evidence_sha256": _sha256_bytes(input_path.read_bytes()),
            "known_invalid_controls_sha256": _sha256_bytes(
                args.known_invalid_controls.read_bytes()
            ),
            "known_invalid_control_count": len(controls),
            "known_equivalent_controls_sha256": _sha256_bytes(
                args.known_equivalent_controls.read_bytes()
            ),
            "known_equivalent_control_count": len(equivalent_controls),
            "row_count": len(bank_rows),
            "candidate_count_histogram": counts,
            "rows_without_candidate": [
                row["row_id"]
                for row in bank_rows
                if row["candidate_count"] == 0
            ],
            "rows": bank_rows,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    existing = _load_latest(record_path)
    if args.audit_only:
        manifest = json.loads(
            (args.output / "MATERIALIZATION_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            manifest.get("trace_contract_version")
            != TRACE_CONTRACT_VERSION
            or manifest.get("input_evidence_sha256")
            != _sha256_bytes(input_path.read_bytes())
            or manifest.get("known_invalid_controls_sha256")
            != _sha256_bytes(args.known_invalid_controls.read_bytes())
            or manifest.get("known_equivalent_controls_sha256")
            != _sha256_bytes(args.known_equivalent_controls.read_bytes())
            or manifest.get("trace_bank_records_sha256")
            != _sha256_bytes(record_path.read_bytes())
            or manifest.get("train_tree_sha256")
            != _tree_hash(args.output / "train")
            or manifest.get("eval_tree_sha256")
            != _tree_hash(args.output / "eval")
        ):
            raise RuntimeError("E49E materialization identity changed")
        for _, _, row_id, problem, reference in work:
            record = existing.get(row_id)
            if (
                record is None
                or record.get("problem_sha256")
                != _sha256_bytes(problem.encode("utf-8"))
                or record.get("reference_answer_sha256")
                != _sha256_bytes(reference.encode("utf-8"))
                or not _record_passes_contract(
                    record,
                    reference_answer=reference,
                    input_record=input_records[row_id],
                )
            ):
                raise RuntimeError(f"E49E evidence failed for {row_id}")
        control_results = _known_invalid_control_results(
            existing,
            controls,
            input_records,
            references,
        )
        if not all(
            result["rejected_by_soundness"] for result in control_results
        ):
            raise RuntimeError("E49E known-invalid control gate failed")
        equivalent_results = _known_equivalent_control_results(
            existing,
            equivalent_controls,
            input_records,
            references,
        )
        if not all(
            result["rejected_as_new"] for result in equivalent_results
        ):
            raise RuntimeError("E49E known-equivalent control gate failed")
        print(args.output)
        return

    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    endpoint, model = base._endpoint(args.endpoint)
    pending = [
        (row_id, problem, reference)
        for _, _, row_id, problem, reference in work
        if not _record_passes_contract(
            existing.get(row_id, {}),
            reference_answer=reference,
            input_record=input_records[row_id],
        )
    ]
    print(
        f"trace banks: total={len(work)} "
        f"complete={len(work)-len(pending)} pending={len(pending)}",
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
                reference_answer=reference,
                input_record=input_records[row_id],
                cache_root=cache_root,
                timeout=args.timeout,
            ): row_id
            for row_id, problem, reference in pending
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
        for _, _, row_id, _, reference in work
        if not _record_passes_contract(
            existing[row_id],
            reference_answer=reference,
            input_record=input_records[row_id],
        )
    ]
    control_results = _known_invalid_control_results(
        existing,
        controls,
        input_records,
        references,
    )
    controls_pass = all(
        result["rejected_by_soundness"] for result in control_results
    )
    equivalent_results = _known_equivalent_control_results(
        existing,
        equivalent_controls,
        input_records,
        references,
    )
    equivalent_controls_pass = all(
        result["rejected_as_new"] for result in equivalent_results
    )
    if failures or not controls_pass or not equivalent_controls_pass:
        _write_json(
            args.evidence / "generation_summary.json",
            {
                "schema": "e49e_trace_bank_generation_summary_v1",
                "pass": False,
                "failures": failures,
                "known_invalid_controls_pass": controls_pass,
                "known_invalid_control_results": control_results,
                "known_equivalent_controls_pass": (
                    equivalent_controls_pass
                ),
                "known_equivalent_control_results": equivalent_results,
            },
        )
        if not controls_pass:
            raise RuntimeError("E49E known-invalid control gate failed")
        if not equivalent_controls_pass:
            raise RuntimeError("E49E known-equivalent control gate failed")
        raise RuntimeError(
            f"{len(failures)} trace banks failed: {failures[:10]}"
        )

    output_splits = {}
    for split, dataset in splits.items():
        data = dataset.to_dict()
        augmented = []
        menu_hashes = []
        for index, row in enumerate(dataset):
            row_id = base._row_id(split, index, row)
            record = existing[row_id]
            augmented.append(_embed(str(row["problem"]), record["menu"]))
            menu_hashes.append(record["menu_sha256"])
        data["original_problem"] = list(data["problem"])
        data["problem"] = augmented
        data["strategy_menu_sha256"] = menu_hashes
        output_splits[split] = Dataset.from_dict(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
            for _, _, row_id, _, _ in work
        ]
        manifest = {
            "schema": "e49e_trace_bank_materialization_v1",
            "trace_contract_version": TRACE_CONTRACT_VERSION,
            "answer_normalization_version": ANSWER_NORMALIZATION_VERSION,
            "source": str(args.source),
            "source_train_tree_sha256": _tree_hash(args.source / "train"),
            "source_eval_tree_sha256": _tree_hash(args.source / "eval"),
            "input_evidence": str(input_path),
            "input_evidence_sha256": _sha256_bytes(input_path.read_bytes()),
            "known_invalid_controls_sha256": _sha256_bytes(
                args.known_invalid_controls.read_bytes()
            ),
            "known_invalid_control_results": control_results,
            "all_known_invalid_controls_rejected": controls_pass,
            "known_equivalent_controls_sha256": _sha256_bytes(
                args.known_equivalent_controls.read_bytes()
            ),
            "known_equivalent_control_results": equivalent_results,
            "all_known_equivalent_controls_rejected_as_new": (
                equivalent_controls_pass
            ),
            "trace_bank_records_sha256": _sha256_bytes(
                record_path.read_bytes()
            ),
            "train_split": train_name,
            "eval_split": eval_name,
            "train_rows": len(splits["train"]),
            "eval_rows": len(splits["eval"]),
            "menu_count": len(work),
            "singleton_menu_count": sum(
                count == 1 for count in retained_counts
            ),
            "multi_strategy_menu_count": sum(
                count >= 2 for count in retained_counts
            ),
            "known_invalid_controls_pass": controls_pass,
            "known_equivalent_controls_pass": equivalent_controls_pass,
            "multi_strategy_menu_fraction": (
                sum(count >= 2 for count in retained_counts)
                / len(retained_counts)
            ),
            "retained_strategy_count_mean": (
                sum(retained_counts) / len(retained_counts)
            ),
            "all_retained_routes_action_trace_double_audited": True,
            "all_retained_pairs_double_equivalence_attacked": True,
            "all_explicit_action_references_closed": True,
            "soundness_roles": list(SOUNDNESS_ROLES),
            "soundness_seeds": list(SOUNDNESS_SEEDS),
            "pair_roles": list(PAIR_ROLES),
            "pair_seeds": list(PAIR_SEEDS),
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
            "schema": "e49e_trace_bank_generation_summary_v1",
            "pass": True,
            "menu_count": len(work),
            "singleton_menu_count": sum(
                count == 1 for count in retained_counts
            ),
            "multi_strategy_menu_count": sum(
                count >= 2 for count in retained_counts
            ),
            "known_invalid_controls_pass": controls_pass,
            "known_invalid_control_results": control_results,
            "known_equivalent_controls_pass": equivalent_controls_pass,
            "known_equivalent_control_results": equivalent_results,
            "materialization_manifest": str(
                args.output / "MATERIALIZATION_MANIFEST.json"
            ),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
