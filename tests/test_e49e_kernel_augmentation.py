from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
AUGMENT = (
    ROOT
    / "ops/math_strategy_calibration/augment_e49e_kernel_routes.py"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49e_kernel_augmentation_job.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_kernel_augmentation_test",
        AUGMENT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload():
    return {
        "schema": "math_finite_kernel_bank_v1",
        "routes": [
            {
                "kernel_id": "direct_algebra",
                "actions": [
                    {
                        "op_code": "READ_GIVENS",
                        "operation": "Form the equation from the givens.",
                    },
                    {
                        "op_code": "SOLVE_EQUATION",
                        "operation": "Solve that equation for the target.",
                    },
                ],
                "plan": "Form and solve the governing equation.",
            },
            {
                "kernel_id": "geometric_similarity",
                "actions": [
                    {
                        "op_code": "CONSTRUCT_OBJECT",
                        "operation": "Construct the relevant similar figures.",
                    },
                    {
                        "op_code": "APPLY_THEOREM",
                        "operation": "Apply similarity ratios to the target.",
                    },
                ],
                "plan": "Use a similarity construction and its ratios.",
            },
        ],
    }


def test_kernel_bank_materializes_limited_machine_visible_actions():
    module = _load()
    menu = module._payload_to_menu(
        _payload(),
        "17",
        problem="A problem without numeric givens.",
    )
    assert len(menu.strategies) == 2
    assert len(menu.actions) == 4
    assert menu.actions[0].operation.startswith("[OP:READ_GIVENS]")
    assert menu.strategies[0].plan.startswith("[KERNEL:direct_algebra]")
    assert menu.strategies[1].plan.startswith(
        "[KERNEL:geometric_similarity]"
    )


def test_duplicate_kernel_or_action_reference_fails_closed():
    module = _load()
    duplicate = _payload()
    duplicate["routes"][1]["kernel_id"] = "direct_algebra"
    with pytest.raises(ValueError):
        module._payload_to_menu(
            duplicate,
            "17",
            problem="A problem without numeric givens.",
        )

    referenced = _payload()
    referenced["routes"][0]["actions"][0]["operation"] = (
        "Import A9 and form the equation."
    )
    with pytest.raises(ValueError):
        module._payload_to_menu(
            referenced,
            "17",
            problem="A problem without numeric givens.",
        )

    duplicate_combo = _payload()
    duplicate_combo["routes"][1]["actions"][0][
        "op_code"
    ] = "READ_GIVENS"
    duplicate_combo["routes"][1]["actions"][1][
        "op_code"
    ] = "SOLVE_EQUATION"
    with pytest.raises(ValueError):
        module._payload_to_menu(
            duplicate_combo,
            "17",
            problem="A problem without numeric givens.",
        )

    with pytest.raises(ValueError):
        module._payload_to_menu(
            {
                "schema": "math_finite_kernel_bank_v1",
                "routes": ["not-an-object", _payload()["routes"][0]],
            },
            "17",
            problem="A problem without numeric givens.",
        )


def test_reference_answer_leak_fails_closed():
    module = _load()
    leaking = _payload()
    leaking["routes"][0]["actions"][1]["operation"] = (
        "State that the final answer is 17."
    )
    with pytest.raises(ValueError):
        module._payload_to_menu(
            leaking,
            "17",
            problem="A problem without numeric givens.",
        )

    derived = _payload()
    derived["routes"][0]["actions"][1]["operation"] = (
        "Use the equation to obtain the derived intermediate 9."
    )
    with pytest.raises(ValueError):
        module._payload_to_menu(
            derived,
            "17",
            problem="A problem with 5 objects.",
        )


def test_completed_kernel_proposal_is_bound_to_fixed_seed(monkeypatch):
    module = _load()
    content = json.dumps(_payload())

    def fake_post(_endpoint, request, *, timeout):
        assert request["seed"] == module.PROPOSAL_SEED
        assert timeout == 30
        return (
            {
                "id": "kernel-proposal",
                "choices": [{"finish_reason": "stop"}],
            },
            content,
        )

    monkeypatch.setattr(module.pipeline.base, "_post", fake_post)
    record = module._proposal_request(
        endpoint="http://judge/v1",
        model="qwen2.5-72b",
        problem="A problem admitting algebra and geometry.",
        reference_answer="17",
        gold_solution="",
        timeout=30,
    )
    assert record["pass"] is True
    assert record["seed"] == 492201
    assert len(record["menu"]["strategies"]) == 2
    assert module._proposal_record_passes(
        record,
        problem="A problem admitting algebra and geometry.",
        reference_answer="17",
    )
    record["menu"]["actions"][0]["operation"] = (
        "[OP:READ_GIVENS] Introduce the derived intermediate 9."
    )
    assert not module._proposal_record_passes(
        record,
        problem="A problem admitting algebra and geometry.",
        reference_answer="17",
    )


def test_kernel_launcher_binds_hardening_and_gap_rule():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "e49e_finite_kernel_hardening_amendment_20260724.md",
        "e49e_kernel_gap_eligibility_amendment_20260724.md",
        "e49e_kernel_cache_replay_amendment_20260724.md",
        '"kernel_hardening_amendment_sha256"',
        '"kernel_gap_eligibility_amendment_sha256"',
        '"kernel_cache_replay_amendment_sha256"',
        "augmentation_requests_at_freeze",
    ):
        assert required in launcher
