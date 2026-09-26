"""Guard-state preservation and exact scope checks for a live display migration."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import handoff_campaign_display_e112r1_20260909 as handoff


def test_only_reviewed_set_entry_allowed():
    before = (handoff.BEFORE if handoff.BEFORE.exists() else handoff.CAMPAIGN).read_text()
    after = handoff.candidate(before)
    assert handoff.presentation_audit(before, after)["all_other_ast_equal"]
    with pytest.raises(AssertionError):
        handoff.presentation_audit(before, after + "\nUNREVIEWED = True\n")
    with pytest.raises(AssertionError):
        handoff.presentation_audit(before, after.replace('int(row["running"]) == 0', 'True'))


def test_dynamic_science_and_history_redisplay():
    import campaign_stats as campaign
    before = (handoff.BEFORE if handoff.BEFORE.exists() else handoff.CAMPAIGN).read_text()
    after = handoff.candidate(before)
    tree = ast.parse(after)
    render_node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "render")
    scope = dict(campaign.__dict__)
    exec(compile(ast.Module(body=[render_node], type_ignores=[]), "reviewed_render", "exec"), scope)
    row = {"label": campaign.registry.by_tag("e112r1").label, "cells": 25,
           "terminal": 3, "running": 0, "pending": 0, "failed": 0,
           "realized": 12768, "total": 76800, "passes": 8, "depth": 1.33,
           "scale_breakdown": {"qwen3b": {"cells": 25, "terminal": 3, "running": 0,
                                          "pending": 0, "failed": 0, "realized": 12768,
                                          "total": 76800, "passes": 8, "depth": 1.33}}}
    for markdown in (False, True):
        normal = scope["render"]([row], markdown=markdown)
        history = scope["render"]([row], markdown=markdown, include_history=True)
        assert "qwen3b" not in normal
        assert "qwen3b" in history
        running = copy.deepcopy(row)
        running["running"] = 1
        running["scale_breakdown"]["qwen3b"]["running"] = 1
        assert "qwen3b" in scope["render"]([running], markdown=markdown)


def test_both_guard_transaction_shapes_pass_without_mutation():
    for tx in ({"jobs": {"1": {"attempts": [{"released": True}], "status": "monitoring"}}},
               {"attempts": [{"released": True}], "status": "watching"}):
        before = copy.deepcopy(tx)
        handoff.safe_transition(tx)
        assert tx == before


@pytest.mark.parametrize("partial", [
    {"attempts": [{"released": False}]},
    {"retirement_in_progress": True},
    {"retirement_pending": True},
    {"fallback_pending_reason": "deadline"},
    {"status": "fallback_in_progress"},
    {"status": "deadline_hold_handoff"},
    {"status": "holding"},
])
def test_unfinished_transition_blocks_cancellation(partial):
    with pytest.raises(AssertionError):
        handoff.safe_transition({"jobs": {"1": partial}})


def test_stopped_e119_guard_cannot_restart():
    with pytest.raises(AssertionError):
        handoff.safe_transition({"attempts": [], "status": "manual_stop"})


def test_nested_e119_lock_resolves():
    expected = ROOT / "var/artifacts/e119_lowprio_completion_20260909/guard/watch.lock"
    assert handoff.singleton(ROOT / "ops/exp_scaling/guard_e119_lowprio_20260909.py") == str(expected)
