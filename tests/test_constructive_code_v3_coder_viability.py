from __future__ import annotations

import copy

import pytest

from evaluate_constructive_code_v3_coder_viability import (
    DEVELOPMENT_PROBLEMS,
    EVALUATION_PROBLEMS,
    OVERLAY_SUITE,
    PLUS_SUITE,
    SYSTEM_MESSAGE,
    _hard_replay_violations,
    _prompt,
    _request_seed,
    _strip_exact_surrounding_fence,
    _validate_gate_and_choose_suites,
)


def _passing_gate():
    keys = {problem: f"key-{problem}" for problem in DEVELOPMENT_PROBLEMS}
    task_results = [
        {"problem_key": key, "status": "pass"} for key in keys.values()
    ] + [
        {"problem_key": f"unused-{index}", "status": "pass"}
        for index in range(8)
    ]
    suite_results = []
    for index, key in enumerate(keys.values()):
        suite_results.extend(
            [
                {
                    "problem_key": key,
                    "suite_id": OVERLAY_SUITE,
                    "status": "pass" if index != 1 else "fail",
                },
                {
                    "problem_key": key,
                    "suite_id": PLUS_SUITE,
                    "status": "pass",
                },
            ]
        )
    return (
        {
            "status": "pass",
            "expected_replay_count": 4800,
            "observed_replay_count": 4800,
            "violations": [],
            "checker_equivalence_violations": [],
            "task_results": task_results,
            "suite_results": suite_results,
        },
        keys,
    )


def test_frozen_prompt_contains_only_system_and_unchanged_statement():
    statement = "Public problem.\n\nInput\n1\n\nOutput\n2"
    prompt = _prompt(statement)
    assert SYSTEM_MESSAGE in prompt
    assert statement in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    for secret in ("canonical", "checker.cpp", "accepted behavior", "reward"):
        assert secret not in prompt.lower()


def test_only_exact_surrounding_python_or_bare_fence_is_removed():
    assert _strip_exact_surrounding_fence("```python\nprint(1)\n```") == (
        "print(1)",
        True,
    )
    assert _strip_exact_surrounding_fence("```\nprint(1)\n```") == (
        "print(1)",
        True,
    )
    for text in (
        "before\n```python\nprint(1)\n```",
        "```Python\nprint(1)\n```",
        "```python\nprint(1)\n```\nafter",
        "print(1)",
    ):
        assert _strip_exact_surrounding_fence(text) == (text, False)


def test_request_seeds_use_zero_based_frozen_formula():
    assert _request_seed(0, 0, 77101) == 77101
    assert _request_seed(3, 63, 77101) == 107164


def test_gate_prefers_overlay_then_falls_back_to_plus_and_fails_closed():
    gate, keys = _passing_gate()
    selected = _validate_gate_and_choose_suites(gate, keys)
    assert tuple(selected) == DEVELOPMENT_PROBLEMS
    assert selected["359_B"] == OVERLAY_SUITE
    assert selected["988_A"] == PLUS_SUITE

    bad = copy.deepcopy(gate)
    bad["observed_replay_count"] = 4799
    with pytest.raises(ValueError, match="not an exact pass"):
        _validate_gate_and_choose_suites(bad, keys)


def test_replay_classifier_separates_wrong_programs_from_hard_failures():
    ordinary_wrong = {
        "released_checker_accepted": False,
        "wrapper_accepted": False,
        "behavior_key": None,
        "execution": {
            "candidate_invocation_wall_seconds": [0.1],
            "checker_wall_seconds": 0.0,
            "first_failure": {
                "stage": "candidate",
                "returncode": 1,
                "timed_out": False,
                "output_limited": False,
                "sandbox_violation": False,
            },
        },
    }
    assert _hard_replay_violations(ordinary_wrong) == []

    timeout = copy.deepcopy(ordinary_wrong)
    timeout["execution"]["first_failure"]["timed_out"] = True
    assert _hard_replay_violations(timeout) == [
        "candidate execution-bound violation"
    ]

    disagreement = copy.deepcopy(ordinary_wrong)
    disagreement["released_checker_accepted"] = True
    assert "checker-wrapper disagreement" in _hard_replay_violations(disagreement)


def test_split_constants_are_problem_disjoint_and_exact():
    assert DEVELOPMENT_PROBLEMS == ("359_B", "988_A", "1283_C", "1399_D")
    assert EVALUATION_PROBLEMS == ("361_B", "1294_C", "1408_A", "149_C")
    assert set(DEVELOPMENT_PROBLEMS).isdisjoint(EVALUATION_PROBLEMS)
