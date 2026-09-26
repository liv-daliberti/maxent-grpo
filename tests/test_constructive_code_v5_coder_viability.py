from __future__ import annotations

import copy

import pytest

import evaluate_constructive_code_v5_coder_viability as v5
from evaluate_constructive_code_v3_coder_viability import (
    DEVELOPMENT_PROBLEMS,
    OVERLAY_SUITE,
    PLUS_SUITE,
)


def passing_gate():
    keys = {problem: f"key-{problem}" for problem in DEVELOPMENT_PROBLEMS}
    task_results = [
        {"problem_key": key, "status": "pass"} for key in keys.values()
    ] + [
        {"problem_key": f"unused-{index}", "status": "pass"}
        for index in range(8)
    ]
    suite_results = []
    for index, key in enumerate(keys.values()):
        suite_results.extend([
            {"problem_key": key, "suite_id": OVERLAY_SUITE, "status": "pass" if index != 1 else "fail"},
            {"problem_key": key, "suite_id": PLUS_SUITE, "status": "pass"},
        ])
    return {
        "status": "pass", "expected_replay_count": 2304,
        "observed_replay_count": 2304, "violations": [],
        "checker_equivalence_violations": [], "task_results": task_results,
        "suite_results": suite_results,
    }, keys


def test_v5_gate_requires_exact_2304_and_preserves_suite_selection() -> None:
    gate, keys = passing_gate()
    selected = v5._validate_v5_gate_and_choose_suites(gate, keys)
    assert selected["359_B"] == OVERLAY_SUITE
    assert selected["988_A"] == PLUS_SUITE
    bad = copy.deepcopy(gate)
    bad["observed_replay_count"] = 2303
    with pytest.raises(ValueError, match="not an exact pass"):
        v5._validate_v5_gate_and_choose_suites(bad, keys)


def test_v5_uses_v5_replay_modules_and_receipt_schema() -> None:
    replay, _base, materialize = v5._replay_modules()
    assert replay.__name__ == "replay_constructive_code_v5"
    assert materialize.V3_TASKS == materialize.V5_TASKS
    assert materialize.V3_SPLIT_ASSIGNMENT == materialize.V5_SPLIT_ASSIGNMENT
    assert v5.EXPECTED_REPLAY_COUNT == 2304
    assert "constructive-code-v5-coder-05b-viability-v1" in open(v5.__file__).read()
