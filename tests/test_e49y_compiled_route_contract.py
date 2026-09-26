from __future__ import annotations

import importlib.util
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49y_compiled_route_calibration.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("e49y_contract_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw():
    return {
        "routes": [
            {
                "strategy_id": "S2",
                "source_kind": "proposed",
                "exemplar_ids": [],
                "actions": ["substitute the alternate variable", "factor"],
                "plan": "Use the alternate substitution and factor.",
            },
            {
                "strategy_id": "S1",
                "source_kind": "observed",
                "exemplar_ids": ["sample-1"],
                "actions": ["apply the first identity", "solve the equation"],
                "plan": "Use the first identity and solve.",
            },
        ]
    }


def test_compiler_preserves_route_order_and_assigns_disjoint_combos():
    module = _module()
    proposal = module._compile_proposal(_raw(), {"sample-1"})
    menu = proposal["menu"]

    assert [row["operation"] for row in menu["actions"]] == [
        "apply the first identity",
        "solve the equation",
        "substitute the alternate variable",
        "factor",
    ]
    assert menu["strategies"][0]["action_ids"] == ["A1", "A2"]
    assert menu["strategies"][1]["action_ids"] == ["A3", "A4"]
    assert set(menu["strategies"][0]["action_ids"]).isdisjoint(
        menu["strategies"][1]["action_ids"]
    )
    assert proposal["route_sources"] == [
        {
            "strategy_id": "S1",
            "source_kind": "observed",
            "exemplar_ids": ["sample-1"],
        },
        {
            "strategy_id": "S2",
            "source_kind": "proposed",
            "exemplar_ids": [],
        },
    ]


def test_compiler_rejects_proposed_route_with_exemplar_binding():
    module = _module()
    raw = _raw()
    raw["routes"][0]["exemplar_ids"] = ["sample-1"]
    with pytest.raises(ValueError, match="proposed route cites"):
        module._compile_proposal(raw, {"sample-1"})


def test_compiler_rejects_duplicate_strategy_ids():
    module = _module()
    raw = _raw()
    raw["routes"][0]["strategy_id"] = "S1"
    with pytest.raises(ValueError, match="exactly S1 and S2"):
        module._compile_proposal(raw, {"sample-1"})
