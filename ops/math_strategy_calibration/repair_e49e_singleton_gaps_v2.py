#!/usr/bin/env python3
"""Answer-blind symbolic singleton fallback for unresolved E49E toy rows.

V1 exposed auditor-only reference material to the proposal model and then
correctly rejected policy-visible derived numerals.  V2 removes that failure
mode: the proposal request contains the problem and fixed route-role
instruction only.  Reference answers remain available solely to the two
unchanged execution/soundness auditors.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
BASE_PATH = HERE / "repair_e49e_singleton_gaps.py"
SPEC = importlib.util.spec_from_file_location(
    "e49e_singleton_repair_v2_base",
    BASE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load E49E singleton-repair v1 implementation")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

REPAIR_VERSION = "e49e_answer_blind_symbolic_singleton_repair_v2"
REPAIR_ORIGIN = "answer_blind_symbolic_singleton_repair"
PROPOSAL_SEEDS = (492241, 492242, 492243, 492244)
PROPOSAL_ROLES = (
    "symbolic_direct_route",
    "symbolic_checked_route",
    "natural_representation_route",
    "constraint_first_route",
)
PRIOR_REPAIR_VERSION = "e49e_reference_bound_singleton_repair_v1"
_prior_records: dict[str, dict[str, Any]] = {}

PRIOR_SPEC = importlib.util.spec_from_file_location(
    "e49e_singleton_repair_v1_validator",
    BASE_PATH,
)
if PRIOR_SPEC is None or PRIOR_SPEC.loader is None:
    raise RuntimeError("cannot load frozen E49E singleton-repair validator")
prior_validator = importlib.util.module_from_spec(PRIOR_SPEC)
PRIOR_SPEC.loader.exec_module(prior_validator)

_base_repair_one = base._repair_one
_base_repair_record_passes = base._repair_record_passes

NUMBER_WORD_VALUES = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
}
WORD_RE = re.compile(r"[a-z]+")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _singleton_schema() -> dict[str, Any]:
    action_ids = [f"A{index}" for index in range(1, 6)]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "actions", "strategies"],
        "properties": {
            "schema": {
                "type": "string",
                "enum": [base.pipeline.MENU_SCHEMA],
            },
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action_id", "operation"],
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "enum": action_ids,
                        },
                        "operation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 220,
                        },
                    },
                },
            },
            "strategies": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["strategy_id", "action_ids", "plan"],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": ["S1"],
                        },
                        "action_ids": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 5,
                            "items": {
                                "type": "string",
                                "enum": action_ids,
                            },
                        },
                        "plan": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 420,
                        },
                    },
                },
            },
        },
    }


def _role_instruction(role: str) -> str:
    return {
        "symbolic_direct_route": (
            "Use the shortest complete theorem-and-transformation route."
        ),
        "symbolic_checked_route": (
            "Include the decisive domain, boundary, integrality, or "
            "sufficiency check."
        ),
        "natural_representation_route": (
            "Use the problem's most natural algebraic, geometric, "
            "combinatorial, or complex-number representation."
        ),
        "constraint_first_route": (
            "Derive the binding structural constraints first, then state "
            "the exact symbolic reduction that determines the target."
        ),
    }[role]


def _menu_text(menu: Any) -> str:
    return "\n".join(
        [action.operation for action in menu.actions]
        + [strategy.plan for strategy in menu.strategies]
    ).casefold()


def _integer_reference_word(reference_answer: str) -> str | None:
    compact = str(reference_answer).casefold().strip()
    for token in ("\\boxed", "\\$", "$", "\\", "{", "}", "(", ")"):
        compact = compact.replace(token, "")
    compact = re.sub(r"\s+", "", compact)
    match = re.fullmatch(r"(?:[a-z]+(?:=|:))?([-+]?\d+)", compact)
    if match is None:
        return None
    integer = str(int(match.group(1)))
    for word, value in NUMBER_WORD_VALUES.items():
        if value == integer and word not in {
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "eleventh",
            "twelfth",
            "thirteenth",
            "fourteenth",
            "fifteenth",
            "sixteenth",
            "seventeenth",
            "eighteenth",
            "nineteenth",
            "twentieth",
        }:
            return word
    return None


def _proposal_is_nonleaking_v2(
    menu: Any,
    *,
    problem: str,
    reference_answer: str,
) -> bool:
    if not base._proposal_is_nonleaking(
        menu,
        problem=problem,
        reference_answer=reference_answer,
    ):
        return False
    text_words = set(WORD_RE.findall(_menu_text(menu)))
    problem_words = set(WORD_RE.findall(str(problem).casefold()))
    problem_numbers = {
        value.lstrip("+-")
        for value in base._numeric_literals(problem)
    }
    allowed_structural = {"0", "1", "2"}
    for word in text_words & NUMBER_WORD_VALUES.keys():
        if (
            word not in problem_words
            and NUMBER_WORD_VALUES[word] not in problem_numbers
            and NUMBER_WORD_VALUES[word] not in allowed_structural
        ):
            return False
    reference_word = _integer_reference_word(reference_answer)
    return reference_word is None or reference_word not in text_words


def _proposal_request(
    *,
    endpoint: str,
    model: str,
    problem: str,
    reference_answer: str,
    gold_solution: str,
    role: str,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    # Deliberately unused in the serialized request.
    del gold_solution
    prompt = f"""Write one concise symbolic action contract for this problem.

{_role_instruction(role)}

Return directives for a solver, not a worked solution.  Give two to five
ordered operations that name the actual theorem, transformation, recurrence,
construction, counting event, or case split to execute.  Do not calculate or
state the final answer.  Do not state any evaluated intermediate result.
Do state the decisive symbolic identity, equation, invariant, or event
definition whenever it can be written under the numeric rule below; a
specific symbolic reduction is not considered an evaluated result.
Numeric literals may appear only when copied verbatim from the problem or
when they are the structural constants 0, 1, and 2.  Before returning JSON,
scan every operation and the plan and replace any other derived numeral with
a symbolic phrase such as "the resulting residue", "the derived factor", or
"the constrained parameter".

The strategy must be S1.  It must use every declared action exactly once in
the declared order.  The contract must be self-contained, specific enough to
execute literally, and must not mention an alternative route.

PROBLEM:
{problem}
"""
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": base.pipeline.SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_answer_blind_symbolic_contract",
                "strict": True,
                "schema": _singleton_schema(),
            },
        },
    }
    response, content = base.pipeline.base._post(
        endpoint,
        request,
        timeout=timeout,
    )
    common = {
        "repair_version": REPAIR_VERSION,
        "kind": "singleton_proposal",
        "proposal_input_scope": "problem_only",
        "request_sha256": _canonical_sha256(request),
        "role": role,
        "seed": seed,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "content_sha256": _sha256_bytes(content.encode("utf-8")),
    }
    try:
        payload = json.loads(content)
        menu = base.pipeline._menu_from_payload(payload)
        closed = base.pipeline._extract_closed_candidate(menu, "S1")
        if closed is None:
            raise ValueError("singleton proposal failed action closure")
        if (
            len(menu.strategies) != 1
            or len(closed.strategies) != 1
            or list(menu.strategies[0].action_ids)
            != [action.action_id for action in menu.actions]
        ):
            raise ValueError("singleton proposal failed ordered-use gate")
        if not _proposal_is_nonleaking_v2(
            menu,
            problem=problem,
            reference_answer=reference_answer,
        ):
            raise ValueError(
                "singleton proposal failed numeric-or-word leakage gate"
            )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            **common,
            "menu": None,
            "completed_invalid": True,
            "error": str(exc),
            "pass": False,
        }
    return {
        **common,
        "menu": json.loads(closed.canonical_json),
        "menu_sha256": closed.sha256,
        "completed_invalid": False,
        "pass": bool(
            common["finish_reason"] == "stop"
            and str(common["response_id"] or "")
        ),
    }


def _carried_prior(
    *,
    row_id: str,
    problem: str,
    reference_answer: str,
) -> dict[str, Any] | None:
    prior = _prior_records.get(row_id)
    if not isinstance(prior, dict) or prior.get("pass") is not True:
        return None
    if not prior_validator._repair_record_passes(
        prior,
        row_id=row_id,
        problem=problem,
        reference_answer=reference_answer,
    ):
        raise RuntimeError(f"prior singleton repair changed: {row_id}")
    return {
        "schema": "e49e_singleton_repair_record_v1",
        "repair_version": REPAIR_VERSION,
        "row_id": row_id,
        "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": _sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "attempts": [],
        "carry_origin": PRIOR_REPAIR_VERSION,
        "prior_repair_record_sha256": _canonical_sha256(prior),
        "prior_repair_record": prior,
        "menu": prior["menu"],
        "menu_sha256": prior["menu_sha256"],
        "pass": True,
    }


def _repair_one(**kwargs) -> dict[str, Any]:
    carried = _carried_prior(
        row_id=kwargs["row_id"],
        problem=kwargs["problem"],
        reference_answer=kwargs["reference_answer"],
    )
    if carried is not None:
        return carried
    return _base_repair_one(**kwargs)


def _repair_record_passes(
    record: dict[str, Any],
    *,
    row_id: str,
    problem: str,
    reference_answer: str,
) -> bool:
    if record.get("carry_origin") == PRIOR_REPAIR_VERSION:
        prior = record.get("prior_repair_record")
        return bool(
            isinstance(prior, dict)
            and record.get("repair_version") == REPAIR_VERSION
            and record.get("row_id") == row_id
            and record.get("attempts") == []
            and record.get("prior_repair_record_sha256")
            == _canonical_sha256(prior)
            and prior_validator._repair_record_passes(
                prior,
                row_id=row_id,
                problem=problem,
                reference_answer=reference_answer,
            )
            and record.get("menu_sha256") == prior.get("menu_sha256")
            and record.get("menu") == prior.get("menu")
            and record.get("pass") is True
        )
    if not _base_repair_record_passes(
        record,
        row_id=row_id,
        problem=problem,
        reference_answer=reference_answer,
    ):
        return False
    for attempt in record["attempts"]:
        proposal = attempt["proposal"]
        if (
            proposal.get("proposal_input_scope") != "problem_only"
            or not isinstance(proposal.get("request_sha256"), str)
            or len(proposal["request_sha256"]) != 64
        ):
            return False
        if proposal.get("pass") is True:
            try:
                menu = base.pipeline._menu_from_payload(proposal["menu"])
            except (TypeError, ValueError):
                return False
            if not _proposal_is_nonleaking_v2(
                menu,
                problem=problem,
                reference_answer=reference_answer,
            ):
                return False
    return True


def _configure_base() -> None:
    base.REPAIR_VERSION = REPAIR_VERSION
    base.REPAIR_ORIGIN = REPAIR_ORIGIN
    base.PROPOSAL_SEEDS = PROPOSAL_SEEDS
    base.PROPOSAL_ROLES = PROPOSAL_ROLES
    base._singleton_schema = _singleton_schema
    base._proposal_request = _proposal_request
    base._repair_one = _repair_one
    base._repair_record_passes = _repair_record_passes


def main() -> None:
    prior_path = pathlib.Path(os.environ.get("E49E_PRIOR_REPAIR_RECORDS", ""))
    if prior_path.is_file():
        _prior_records.update(base.pipeline._load_latest(prior_path))
    _configure_base()
    base.main()


if __name__ == "__main__":
    main()
