from __future__ import annotations

import importlib.util
import pathlib

from oat_drgrpo.math_strategy_menu import (
    MENU_END,
    MENU_START,
    parse_strategy_menu,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "materialize_e49x_accessible_math_toy.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("materialize_e49x", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _menu():
    problem = f"""test
{MENU_START}
{{"schema":"math_strategy_action_menu_v1","actions":[
{{"action_id":"A1","operation":"factor"}},
{{"action_id":"A2","operation":"sign chart"}},
{{"action_id":"A3","operation":"complete square"}},
{{"action_id":"A4","operation":"vertex analysis"}}],
"strategies":[
{{"strategy_id":"S1","action_ids":["A1","A2"],"plan":"factor route"}},
{{"strategy_id":"S2","action_ids":["A3","A4"],"plan":"vertex route"}}]}}
{MENU_END}"""
    menu = parse_strategy_menu(problem)
    assert menu is not None
    return menu


def test_e49x_pruning_retains_exact_operations_and_renumbers():
    module = _load()
    source = _menu()
    pruned = module._prune(source, "S2")
    assert len(pruned.strategies) == 1
    assert pruned.strategies[0].strategy_id == "S1"
    assert pruned.strategies[0].action_ids == ("A1", "A2")
    assert [action.operation for action in pruned.actions] == [
        "complete square",
        "vertex analysis",
    ]


def test_e49x_protocol_fixes_matched_e46_settings_and_ten_duals():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e49x_accessible_bottom_up_math_toy_20260726.md"
    ).read_text(encoding="utf-8")
    assert "ten dual training menus" in protocol
    assert "exactly three prompt epochs" in protocol
    assert "target 0.80" in protocol
    assert "Policy-entropy adaptation remains disabled" in protocol
