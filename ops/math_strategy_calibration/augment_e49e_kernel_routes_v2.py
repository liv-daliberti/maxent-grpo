#!/usr/bin/env python3
"""Route-wise fail-closed finite-kernel augmentation for E49E.

V1 rejected an entire proposal bank when one route leaked a derived numeral
or duplicated another route's kernel/action-code signature. V2 preserves the
same strict per-route leakage and execution contract, but deterministically
drops the offending route before audit. It never turns an inadmissible route
into policy-visible support.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
BASE_PATH = HERE / "augment_e49e_kernel_routes.py"
SPEC = importlib.util.spec_from_file_location(
    "e49e_kernel_augmentation_v2_base",
    BASE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load E49E finite-kernel v1 implementation")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

AUGMENTATION_VERSION = "e49e_finite_kernel_augmentation_v2"
PROPOSAL_SEED = 492202
PROPOSAL_SCHEMA = "math_finite_kernel_bank_v2"
_base_proposal_schema = base._proposal_schema
_base_build_one = base._build_one
_prior_records: dict[str, dict[str, Any]] = {}


def _proposal_schema() -> dict[str, Any]:
    schema = _base_proposal_schema()
    schema["properties"]["schema"]["enum"] = [PROPOSAL_SCHEMA]
    return schema


def _validated_single_route(
    route: Any,
    *,
    problem: str,
    reference_answer: str,
):
    if (
        not isinstance(route, dict)
        or set(route) != {"kernel_id", "actions", "plan"}
        or route.get("kernel_id") not in base.KERNELS
        or not isinstance(route.get("actions"), list)
        or not 2 <= len(route["actions"]) <= 6
        or not isinstance(route.get("plan"), str)
        or not route["plan"].strip()
        or len(route["plan"]) > 560
        or base.pipeline.ACTION_REFERENCE_RE.search(route["plan"])
    ):
        return None
    actions = []
    combo = []
    for index, action in enumerate(route["actions"], start=1):
        if (
            not isinstance(action, dict)
            or set(action) != {"op_code", "operation"}
            or action.get("op_code") not in base.OP_CODES
            or not isinstance(action.get("operation"), str)
            or not action["operation"].strip()
            or len(action["operation"]) > 260
            or base.pipeline.ACTION_REFERENCE_RE.search(
                action["operation"]
            )
        ):
            return None
        combo.append(action["op_code"])
        actions.append(
            {
                "action_id": f"A{index}",
                "operation": (
                    f"[OP:{action['op_code']}] "
                    f"{action['operation'].strip()}"
                ),
            }
        )
    menu = base.pipeline._menu_from_payload(
        {
            "schema": base.pipeline.MENU_SCHEMA,
            "actions": actions,
            "strategies": [
                {
                    "strategy_id": "S1",
                    "action_ids": [
                        action["action_id"] for action in actions
                    ],
                    "plan": (
                        f"[KERNEL:{route['kernel_id']}] "
                        f"{route['plan'].strip()}"
                    ),
                }
            ],
        }
    )
    if not base.repair._proposal_is_nonleaking(
        menu,
        problem=problem,
        reference_answer=reference_answer,
    ):
        return None
    return route, tuple(combo)


def _payload_to_menu(
    payload: dict[str, Any],
    reference_answer: str,
    *,
    problem: str,
):
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "routes"}
        or payload.get("schema") != PROPOSAL_SCHEMA
        or not isinstance(payload.get("routes"), list)
        or not 2 <= len(payload["routes"]) <= 3
    ):
        raise ValueError("invalid finite-kernel v2 bank")

    accepted = []
    kernels = set()
    combos = set()
    action_count = 0
    for proposed in payload["routes"]:
        validated = _validated_single_route(
            proposed,
            problem=problem,
            reference_answer=reference_answer,
        )
        if validated is None:
            continue
        route, combo = validated
        kernel = route["kernel_id"]
        if kernel in kernels or combo in combos:
            continue
        if action_count + len(route["actions"]) > 12:
            continue
        accepted.append(route)
        kernels.add(kernel)
        combos.add(combo)
        action_count += len(route["actions"])
    if not accepted:
        raise ValueError("no individually admissible finite-kernel route")

    actions = []
    strategies = []
    next_action = 1
    for strategy_index, route in enumerate(accepted, start=1):
        ids = []
        for action in route["actions"]:
            action_id = f"A{next_action}"
            next_action += 1
            ids.append(action_id)
            actions.append(
                {
                    "action_id": action_id,
                    "operation": (
                        f"[OP:{action['op_code']}] "
                        f"{action['operation'].strip()}"
                    ),
                }
            )
        strategies.append(
            {
                "strategy_id": f"S{strategy_index}",
                "action_ids": ids,
                "plan": (
                    f"[KERNEL:{route['kernel_id']}] "
                    f"{route['plan'].strip()}"
                ),
            }
        )
    menu = base.pipeline._menu_from_payload(
        {
            "schema": base.pipeline.MENU_SCHEMA,
            "actions": actions,
            "strategies": strategies,
        }
    )
    if not base.repair._proposal_is_nonleaking(
        menu,
        problem=problem,
        reference_answer=reference_answer,
    ):
        raise RuntimeError("route-wise leakage closure failed")
    return menu


def _build_one(**kwargs):
    current = _base_build_one(**kwargs)
    prior = _prior_records.get(kwargs["row_id"])
    if (
        not isinstance(prior, dict)
        or prior.get("origin") != "finite_kernel_augmentation"
    ):
        return current
    prior_selected = base.repair._augmentation_selected(
        prior,
        raw_record=kwargs["raw_record"],
        input_record=kwargs["input_record"],
        problem=kwargs["problem"],
        reference_answer=kwargs["reference_answer"],
    )
    if prior_selected is None:
        return current
    current_support = len(
        (current.get("menu") or {}).get("strategies") or []
    )
    prior_menu, prior_certification = prior_selected
    if len(prior_menu.strategies) <= current_support:
        return {
            **current,
            "prior_augmentation_record_sha256": (
                base._canonical_sha256(prior)
            ),
        }
    carried = dict(current)
    carried.pop("error", None)
    return {
        **carried,
        "origin": "prior_v1_finite_kernel_augmentation",
        "menu": base.json.loads(prior_menu.canonical_json),
        "menu_sha256": prior_menu.sha256,
        "certification": prior_certification,
        "prior_augmentation_record_sha256": (
            base._canonical_sha256(prior)
        ),
        "pass": True,
    }


def main() -> None:
    prior_path = pathlib.Path(
        os.environ.get("E49E_KERNEL_PRIOR_RECORDS", "")
    )
    if prior_path.is_file():
        _prior_records.update(base.pipeline._load_latest(prior_path))
    base.AUGMENTATION_VERSION = AUGMENTATION_VERSION
    base.PROPOSAL_SEED = PROPOSAL_SEED
    base.PROPOSAL_EXTRA_RULES = """
Before returning JSON, scan every operation and plan for numeric literals.
If a literal is not copied verbatim from the problem and is not 0, 1, or 2,
rewrite the step symbolically (for example, "the resulting factor" or "the
derived residue") without stating its value. Do not copy any intermediate
numeric result from the gold derivation. Each proposed route is checked
independently; an inadmissible route is discarded rather than repaired."""
    base._proposal_schema = _proposal_schema
    base._payload_to_menu = _payload_to_menu
    base._build_one = _build_one
    base.main()


if __name__ == "__main__":
    main()
