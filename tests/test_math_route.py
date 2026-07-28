from __future__ import annotations

import json
from fractions import Fraction

from oat_drgrpo.math_route import (
    extract_math_route_block,
    problem_number_inventory,
    validate_math_route_response,
    validate_math_route_trace,
)
from oat_drgrpo.math_grader import validated_math_route_signature


def _trace(steps, final):
    return {
        "version": "math-route-v1",
        "steps": steps,
        "final": final,
    }


def test_grounded_pythagorean_trace_executes_without_declared_results():
    problem = "A right triangle has legs 5 and 12. Find the hypotenuse."
    trace = _trace(
        [
            {"id": "s1", "op": "source", "value": "5"},
            {"id": "s2", "op": "source", "value": "12"},
            {"id": "s3", "op": "square", "args": ["s1"]},
            {"id": "s4", "op": "square", "args": ["s2"]},
            {"id": "s5", "op": "add", "args": ["s3", "s4"]},
            {"id": "s6", "op": "sqrt", "args": ["s5"]},
        ],
        "s6",
    )

    validation = validate_math_route_trace(trace, problem)

    assert validation is not None
    assert validation.terminal_value == 13
    assert validation.source_count == 2
    assert "5" not in validation.route_signature
    assert "12" not in validation.route_signature


def test_route_signature_recurs_across_prompts_but_preserves_structure():
    first = validate_math_route_trace(
        _trace(
            [
                {"id": "s1", "op": "source", "value": "3"},
                {"id": "s2", "op": "source", "value": "4"},
                {"id": "s3", "op": "add", "args": ["s1", "s2"]},
            ],
            "s3",
        ),
        "Use 3 and 4.",
    )
    second = validate_math_route_trace(
        _trace(
            [
                {"id": "s1", "op": "source", "value": "20"},
                {"id": "s2", "op": "source", "value": "22"},
                {"id": "s3", "op": "add", "args": ["s2", "s1"]},
            ],
            "s3",
        ),
        "Use 20 and 22.",
    )
    different = validate_math_route_trace(
        _trace(
            [
                {"id": "s1", "op": "source", "value": "20"},
                {"id": "s2", "op": "source", "value": "22"},
                {"id": "s3", "op": "mul", "args": ["s1", "s2"]},
            ],
            "s3",
        ),
        "Use 20 and 22.",
    )

    assert first is not None and second is not None and different is not None
    assert first.route_signature == second.route_signature
    assert first.route_signature != different.route_signature


def test_fraction_inventory_is_exact_and_multiplicity_bounded():
    inventory = problem_number_inventory(
        r"Use \frac{1}{2}, 3/4, 1,000, and -2.5; then reuse 3/4."
    )

    assert inventory[Fraction(1, 2)] == 1
    assert inventory[Fraction(3, 4)] == 2
    assert inventory[Fraction(1000)] == 1
    assert inventory[Fraction(-5, 2)] == 1


def test_answer_literal_injection_unused_nodes_and_declared_results_fail_closed():
    problem = "Add 5 and 7."
    injected = _trace(
        [
            {"id": "s1", "op": "source", "value": "12"},
            {"id": "s2", "op": "square", "args": ["s1"]},
        ],
        "s2",
    )
    unused = _trace(
        [
            {"id": "s1", "op": "source", "value": "5"},
            {"id": "s2", "op": "source", "value": "7"},
            {"id": "s3", "op": "add", "args": ["s1", "s2"]},
            {"id": "s4", "op": "square", "args": ["s1"]},
        ],
        "s3",
    )
    declared = _trace(
        [
            {"id": "s1", "op": "source", "value": "5"},
            {"id": "s2", "op": "square", "args": ["s1"], "result": "25"},
        ],
        "s2",
    )

    assert validate_math_route_trace(injected, problem) is None
    assert validate_math_route_trace(unused, problem) is None
    assert validate_math_route_trace(declared, problem) is None


def test_route_block_must_be_unique_and_valid_json():
    trace = json.dumps(
        _trace(
            [
                {"id": "s1", "op": "source", "value": "10"},
                {"id": "s2", "op": "percent", "args": ["s1"]},
            ],
            "s2",
        )
    )
    response = f"Reasoning. <route>{trace}</route> Therefore \\\\boxed{{0.1}}"

    assert extract_math_route_block(response) == trace
    validation = validate_math_route_response(response, "What is 10 percent?")
    assert validation is not None
    assert validation.terminal_value == Fraction(1, 10)
    assert validate_math_route_response(
        response + f"<route>{trace}</route>",
        "What is 10 percent?",
    ) is None


def test_route_admission_requires_task_reward_and_terminal_answer_agreement():
    problem = "A right triangle has legs 5 and 12. Find the hypotenuse."
    correct_trace = json.dumps(
        _trace(
            [
                {"id": "s1", "op": "source", "value": "5"},
                {"id": "s2", "op": "source", "value": "12"},
                {"id": "s3", "op": "square", "args": ["s1"]},
                {"id": "s4", "op": "square", "args": ["s2"]},
                {"id": "s5", "op": "add", "args": ["s3", "s4"]},
                {"id": "s6", "op": "sqrt", "args": ["s5"]},
            ],
            "s6",
        )
    )
    wrong_terminal_trace = json.dumps(
        _trace(
            [
                {"id": "s1", "op": "source", "value": "5"},
                {"id": "s2", "op": "source", "value": "12"},
                {"id": "s3", "op": "add", "args": ["s1", "s2"]},
            ],
            "s3",
        )
    )

    accepted = (
        f"<route>{correct_trace}</route> Therefore the answer is \\\\boxed{{13}}"
    )
    mismatched = (
        f"<route>{wrong_terminal_trace}</route> Therefore the answer is \\\\boxed{{13}}"
    )
    task_wrong = (
        f"<route>{correct_trace}</route> Therefore the answer is \\\\boxed{{12}}"
    )

    signature = validated_math_route_signature(accepted, problem, "13")
    assert signature is not None
    assert validated_math_route_signature(mismatched, problem, "13") is None
    assert validated_math_route_signature(task_wrong, problem, "13") is None
