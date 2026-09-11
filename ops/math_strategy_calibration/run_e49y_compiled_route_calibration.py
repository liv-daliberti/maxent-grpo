#!/usr/bin/env python3
"""Run prospective E49Y with deterministically compiled disjoint combos."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49w_bottom_up_route_calibration.py"
)
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49y_compiled_bottom_up_route_calibration_20260726.md"
)
SCHEMA = "e49y_compiled_bottom_up_route_calibration_v1"


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("e49y_e49w_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E49W base: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compiled_schema(sample_ids: list[str]) -> dict[str, Any]:
    route = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "strategy_id",
            "source_kind",
            "exemplar_ids",
            "actions",
            "plan",
        ],
        "properties": {
            "strategy_id": {
                "type": "string",
                "enum": ["S1", "S2"],
            },
            "source_kind": {
                "type": "string",
                "enum": ["observed", "proposed"],
            },
            "exemplar_ids": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "string",
                    "enum": sample_ids,
                },
            },
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 320,
                },
            },
            "plan": {
                "type": "string",
                "minLength": 1,
                "maxLength": 640,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["routes"],
        "properties": {
            "routes": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": route,
            }
        },
    }


def _compile_proposal(
    raw: dict[str, Any],
    sample_ids: set[str],
) -> dict[str, Any]:
    routes = raw.get("routes")
    if not isinstance(routes, list) or len(routes) != 2:
        raise ValueError("E49Y requires exactly two generated routes")
    by_id = {
        str(route.get("strategy_id")): route
        for route in routes
        if isinstance(route, dict)
    }
    if set(by_id) != {"S1", "S2"}:
        raise ValueError("E49Y route IDs must be exactly S1 and S2")
    if by_id["S1"].get("source_kind") != "observed":
        raise ValueError("E49Y S1 must bind to an observed route")

    actions = []
    strategies = []
    route_sources = []
    next_action = 1
    for strategy_id in ("S1", "S2"):
        route = by_id[strategy_id]
        operations = route.get("actions")
        exemplars = route.get("exemplar_ids")
        source_kind = route.get("source_kind")
        plan = route.get("plan")
        if (
            not isinstance(operations, list)
            or not 2 <= len(operations) <= 4
            or any(
                not isinstance(operation, str)
                or not operation.strip()
                or len(operation.strip()) > 320
                for operation in operations
            )
        ):
            raise ValueError("E49Y route operations violate the compiler")
        if (
            not isinstance(plan, str)
            or not plan.strip()
            or len(plan.strip()) > 640
        ):
            raise ValueError("E49Y route plan violates the compiler")
        if (
            not isinstance(exemplars, list)
            or len(exemplars) > 3
            or len(set(exemplars)) != len(exemplars)
            or not set(exemplars) <= sample_ids
        ):
            raise ValueError("E49Y route exemplar binding is malformed")
        if source_kind == "observed" and not exemplars:
            raise ValueError("E49Y observed route lacks an exemplar")
        if source_kind == "proposed" and exemplars:
            raise ValueError("E49Y proposed route cites an exemplar")

        combo = []
        for operation in operations:
            action_id = f"A{next_action}"
            actions.append(
                {
                    "action_id": action_id,
                    "operation": operation.strip(),
                }
            )
            combo.append(action_id)
            next_action += 1
        strategies.append(
            {
                "strategy_id": strategy_id,
                "action_ids": combo,
                "plan": plan.strip(),
            }
        )
        route_sources.append(
            {
                "strategy_id": strategy_id,
                "source_kind": source_kind,
                "exemplar_ids": exemplars,
            }
        )
    return {
        "menu": {
            "schema": "math_strategy_action_menu_v1",
            "actions": actions,
            "strategies": strategies,
        },
        "route_sources": route_sources,
    }


def _install_compiled_proposer(base: Any) -> None:
    def propose(
        *,
        endpoint: str,
        model: str,
        candidate: dict[str, Any],
        timeout: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rendered = "\n\n".join(
            f"EXEMPLAR {row['sample_id']}:\n{row['text']}"
            for row in candidate["exemplars"]
        )
        prompt = f"""Construct exactly two concise solution routes for this
hard MATH problem, working bottom-up from correct Qwen2.5-0.5B derivations.

Return S1 and S2. S1 must be materially executed by at least one supplied
exemplar. S2 should be a genuinely distinct supplied route when one exists;
otherwise it may be newly proposed only if concise, sound, and realistically
executable by a 0.5B model when requested.

For each route, list two to four concrete, problem-specific mathematical
operations in their exact execution order. Include every decisive step needed
to derive the answer. Do not use vague operations such as "solve", "reason",
"simplify as needed", or "apply a theorem"; name the actual substitution,
identity, construction, counted set, or transformation. Distinct routes must
differ in a central mathematical operation, not wording, routine algebra, or
a decorative check. Do not include, encode, or hint the final answer in any
operation or plan. Cite up to three exemplar IDs for an observed route. A
proposed route must cite none.

PROBLEM:
{candidate['problem']}

VALIDATOR-POSITIVE 0.5B EXEMPLARS:
{rendered}
"""
        sample_ids = [
            str(row["sample_id"]) for row in candidate["exemplars"]
        ]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": base.SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 3072,
            "seed": base.SEED + candidate["candidate_rank"],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "e49y_compiled_bottom_up_routes",
                    "strict": True,
                    "schema": _compiled_schema(sample_ids),
                },
            },
        }
        response, raw = base._post(endpoint, payload, timeout=timeout)
        proposal = _compile_proposal(raw, set(sample_ids))
        if not base._proposal_passes_local_contract(
            proposal, set(sample_ids)
        ):
            raise ValueError(
                "E49Y compiled proposal failed the finite-menu contract: "
                + ",".join(
                    base._proposal_local_contract_failures(
                        proposal, set(sample_ids)
                    )
                )
            )
        menu = base._parse_menu(proposal["menu"])
        return proposal, {
            "response_id": response.get("id"),
            "finish_reason": (
                ((response.get("choices") or [{}])[0]).get("finish_reason")
            ),
            "seed": payload["seed"],
            "representation": "two_route_operation_lists",
            "compiler": "disjoint_contiguous_action_ids_v1",
            "compiled_menu_sha256": menu.sha256,
        }

    base._propose = propose


def main() -> None:
    base = _load_base()
    base.CALIBRATION_SCHEMA = SCHEMA
    base.PROTOCOL = PROTOCOL
    base.SCRIPT = SCRIPT
    base.BASE_SCRIPT = BASE_PATH
    base.SEED = 490772
    _install_compiled_proposer(base)
    base.main()


if __name__ == "__main__":
    main()
