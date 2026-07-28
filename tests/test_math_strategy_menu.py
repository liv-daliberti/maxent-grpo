from __future__ import annotations

import json
import re

import pytest

from oat_drgrpo.math_strategy_canonicalizer import MathStrategyCanonicalizer
from oat_drgrpo.math_strategy_menu import (
    MENU_END,
    MENU_SCHEMA,
    MENU_START,
    parse_strategy_declaration,
    parse_strategy_execution,
    parse_strategy_menu,
    strategy_menu_natural_response_instructions,
    strategy_menu_response_instructions,
)


def _menu_payload():
    return {
        "schema": MENU_SCHEMA,
        "actions": [
            {"action_id": "A1", "operation": "factor the expression"},
            {"action_id": "A2", "operation": "solve the resulting factors"},
            {"action_id": "A3", "operation": "differentiate the expression"},
            {"action_id": "A4", "operation": "check critical points"},
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": ["A1", "A2"],
                "plan": "Factor directly, then solve each factor.",
            },
            {
                "strategy_id": "S2",
                "action_ids": ["A3", "A4"],
                "plan": "Use calculus and compare the critical points.",
            },
        ],
    }


def _problem() -> str:
    payload = json.dumps(_menu_payload(), sort_keys=True, separators=(",", ":"))
    menu = parse_strategy_menu(
        f"Find the minimum.\n{MENU_START}\n{payload}\n{MENU_END}"
    )
    assert menu is not None
    return (
        "Find the minimum.\n"
        f"{MENU_START}\n{payload}\n{MENU_END}"
        + strategy_menu_response_instructions(menu)
    )


def _integrity_items(payload):
    prompt = payload["messages"][1]["content"]
    rendered = re.split(
        r"ANSWER-MATCHED RESPONSES "
        r"\(opaque IDs, randomly permuted\):\n",
        prompt,
        maxsplit=1,
    )[1]
    return re.findall(
        r"### ID (CAND_\d+)\n(.*?)(?=\n\n### ID |\Z)",
        rendered,
        flags=re.DOTALL,
    )


def _contract_transport(payload):
    assessments = []
    for item_id, text in _integrity_items(payload):
        header = re.search(
            r"<strategy_id>(S\d+)</strategy_id>", text
        )
        declared = header.group(1) if header else ""
        if "NAME_DROP_ONLY" in text:
            status = "invalid"
        elif "EXECUTES_S1" in text and declared != "S1":
            status = "invalid"
        elif "EXECUTES_S2" in text and declared != "S2":
            status = "invalid"
        else:
            status = "valid"
        assessments.append(
            {
                "item_id": item_id,
                "brief_check": "synthetic execution-contract audit",
                "status": status,
            }
        )
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"assessments": assessments})
                }
            }
        ]
    }


def _inference_items(payload):
    prompt = payload["messages"][1]["content"]
    rendered = prompt.split("ANSWER-MATCHED RESPONSES:\n", maxsplit=1)[1]
    return re.findall(
        r"### ID (CAND_\d+)\n(.*?)(?=\n\n### ID |\Z)",
        rendered,
        flags=re.DOTALL,
    )


def _inference_transport(payload):
    assessments = []
    for item_id, text in _inference_items(payload):
        if "FACTOR_ROUTE" in text:
            status, strategy = "valid", "S1"
        elif "CALCULUS_ROUTE" in text:
            status, strategy = "valid", "S2"
        else:
            status, strategy = "invalid", "NONE"
        assessments.append(
            {
                "item_id": item_id,
                "brief_check": "synthetic finite-menu inference",
                "status": status,
                "strategy_id": strategy,
            }
        )
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"assessments": assessments})
                }
            }
        ]
    }


def _response(strategy: str, combo: str, body: str) -> str:
    steps = "\n".join(
        f'<action_step id="{action_id}">\n'
        f"{body} {action_id}\n"
        "</action_step>"
        for action_id in combo.split(">")
    )
    return (
        f"<strategy_id>{strategy}</strategy_id>\n"
        f"<action_combo>{combo}</action_combo>\n"
        "<action_trace>\n"
        f"{steps}\n"
        "</action_trace>\n"
        "Therefore \\boxed{1}."
    )


def test_menu_and_declaration_are_strict_and_round_trip():
    problem = _problem()
    menu = parse_strategy_menu(problem)
    assert menu is not None
    assert [option.action_combo for option in menu.strategies] == [
        "A1>A2",
        "A3>A4",
    ]
    declaration = parse_strategy_declaration(
        _response("S1", "A1>A2", "EXECUTES_S1"),
        menu,
    )
    assert declaration is not None
    assert declaration.strategy_id == "S1"
    execution = parse_strategy_execution(
        _response("S1", "A1>A2", "EXECUTES_S1"),
        menu,
    )
    assert execution is not None
    assert [action_id for action_id, _ in execution.action_steps] == [
        "A1",
        "A2",
    ]
    assert parse_strategy_declaration(
        "\n" + _response("S1", "A1>A2", "EXECUTES_S1"),
        menu,
    ) is None
    assert parse_strategy_declaration(
        _response("S1", "A3>A4", "EXECUTES_S1"),
        menu,
    ) is None
    assert parse_strategy_declaration(
        _response("S9", "A1>A2", "EXECUTES_S1"),
        menu,
    ) is None


def test_execution_trace_rejects_missing_extra_or_reordered_action_blocks():
    menu = parse_strategy_menu(_problem())
    assert menu is not None
    valid = _response("S1", "A1>A2", "EXECUTES_S1")
    assert parse_strategy_execution(valid, menu) is not None
    assert parse_strategy_execution(
        valid.replace(
            '<action_step id="A2">',
            '<action_step id="A3">',
        ),
        menu,
    ) is None
    second = (
        '<action_step id="A2">\nEXECUTES_S1 A2\n</action_step>'
    )
    assert parse_strategy_execution(
        valid.replace("\n" + second, ""),
        menu,
    ) is None
    first = (
        '<action_step id="A1">\nEXECUTES_S1 A1\n</action_step>'
    )
    assert parse_strategy_execution(
        valid.replace(first + "\n" + second, second + "\n" + first),
        menu,
    ) is None


def test_menu_rejects_duplicate_combos():
    payload = _menu_payload()
    payload["strategies"][1]["action_ids"] = ["A1", "A2"]
    problem = f"x\n{MENU_START}{json.dumps(payload)}{MENU_END}"
    with pytest.raises(ValueError, match="combos must be distinct"):
        parse_strategy_menu(problem)


def test_singleton_certified_menu_and_declaration_round_trip():
    payload = _menu_payload()
    payload["actions"] = payload["actions"][:2]
    payload["strategies"] = [payload["strategies"][0]]
    problem = f"x\n{MENU_START}{json.dumps(payload)}{MENU_END}"
    menu = parse_strategy_menu(problem)
    assert menu is not None
    assert len(menu.strategies) == 1
    assert menu.strategies[0].action_combo == "A1>A2"
    assert parse_strategy_declaration(
        _response("S1", "A1>A2", "EXECUTES_S1"),
        menu,
    ) is not None
    assert parse_strategy_declaration(
        _response("S2", "A1>A2", "EXECUTES_S1"),
        menu,
    ) is None

    rendered_problem = (
        problem + strategy_menu_response_instructions(menu)
    )
    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_contract_transport,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[71], [71]],
        prompt_texts=[rendered_problem, rendered_problem],
        response_texts=[
            _response("S1", "A1>A2", "EXECUTES_S1"),
            _response("S1", "A1>A2", "EXECUTES_S1 paraphrase"),
        ],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys[0] == keys[1]
    assert keys[0] is not None
    assert diagnostics.accepted_rows == 2
    assert diagnostics.new_strategy_count == 1


def test_menu_bound_keys_are_finite_and_execution_validated():
    problem = _problem()
    responses = [
        _response("S1", "A1>A2", "EXECUTES_S1"),
        _response("S1", "A1>A2", "EXECUTES_S1 paraphrase"),
        _response("S2", "A3>A4", "EXECUTES_S2"),
        "missing declaration but \\boxed{1}",
        _response("S1", "A3>A4", "wrong combo"),
        _response("S1", "A1>A2", "EXECUTES_S2"),
        _response("S2", "A3>A4", "NAME_DROP_ONLY"),
    ]
    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_contract_transport,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[41, 42]] * len(responses),
        prompt_texts=[problem] * len(responses),
        response_texts=responses,
        task_reward_positive=[True] * len(responses),
        active_mask=[True] * len(responses),
        num_samples=len(responses),
    )
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert keys[3:] == [None, None, None, None]
    assert diagnostics.judge_calls == 2
    assert diagnostics.accepted_rows == 3
    assert diagnostics.rejected_contract_rows == 2
    assert diagnostics.rejected_integrity_rows == 2
    assert diagnostics.new_strategy_count == 2

    later, later_diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[41, 42], [41, 42]],
        prompt_texts=[problem, problem],
        response_texts=[
            _response("S1", "A1>A2", "EXECUTES_S1 again"),
            _response("S2", "A3>A4", "EXECUTES_S2 again"),
        ],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert later == [keys[0], keys[2]]
    assert later_diagnostics.matched_existing_rows == 2
    assert later_diagnostics.new_strategy_count == 0


def test_unstructured_menu_inference_requires_two_unanimous_finite_votes():
    seen_prompts = []

    def capturing_transport(payload):
        seen_prompts.append(payload["messages"][1]["content"])
        return _inference_transport(payload)

    responses = [
        "Natural derivation that fully executes FACTOR_ROUTE. \\boxed{1}",
        "Natural derivation that fully executes CALCULUS_ROUTE. \\boxed{1}",
        "Answer only. \\boxed{1}",
    ]
    default = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_inference_transport,
    )
    default_keys, default_diagnostics = default.canonicalize(
        prompt_token_ids=[[81]] * 3,
        prompt_texts=[_problem()] * 3,
        response_texts=responses,
        task_reward_positive=[True] * 3,
        active_mask=[True] * 3,
        num_samples=3,
    )
    assert default_keys == [None, None, None]
    assert default_diagnostics.rejected_contract_rows == 3

    inferred = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        allow_unstructured_menu_inference=True,
        transport=capturing_transport,
    )
    keys, diagnostics = inferred.canonicalize(
        prompt_token_ids=[[81]] * 3,
        prompt_texts=[_problem()] * 3,
        response_texts=responses,
        task_reward_positive=[True] * 3,
        active_mask=[True] * 3,
        num_samples=3,
    )
    assert keys[0] is not None
    assert keys[1] is not None
    assert keys[0] != keys[1]
    assert keys[2] is None
    assert diagnostics.accepted_rows == 2
    assert diagnostics.inferred_unstructured_rows == 2
    assert diagnostics.rejected_strategy_inference_rows == 1
    assert diagnostics.rejected_contract_rows == 1
    assert len(seen_prompts) == 2
    assert all(
        "declaration must match the route" in prompt
        for prompt in seen_prompts
    )


def test_unstructured_menu_inference_disagreement_fails_closed():
    def disagreeing_transport(payload):
        response = _inference_transport(payload)
        if payload["seed"] == 470722:
            assessment = json.loads(
                response["choices"][0]["message"]["content"]
            )["assessments"][0]
            assessment["strategy_id"] = "S2"
            response["choices"][0]["message"]["content"] = json.dumps(
                {"assessments": [assessment]}
            )
        return response

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        allow_unstructured_menu_inference=True,
        transport=disagreeing_transport,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[91]],
        prompt_texts=[_problem()],
        response_texts=["Natural FACTOR_ROUTE derivation. \\boxed{1}"],
        task_reward_positive=[True],
        active_mask=[True],
        num_samples=1,
    )
    assert keys == [None]
    assert diagnostics.inferred_unstructured_rows == 0
    assert diagnostics.rejected_strategy_inference_rows == 1


def test_natural_response_instructions_are_execution_strict_without_xml():
    menu = parse_strategy_menu(_problem())
    assert menu is not None
    instructions = strategy_menu_natural_response_instructions(menu)
    assert "S1: A1>A2" in instructions
    assert "executes every action" in instructions
    assert "declaration by itself proves nothing" in instructions
    assert "zero reward" in instructions
    assert "<action_trace>" not in instructions
    assert "<action_step" not in instructions
