#!/usr/bin/env python3
"""Run the frozen development-only ConstructiveCode v6 Coder viability gate."""

from __future__ import annotations

from typing import Any, Mapping

import audit_constructive_code_v6_gate as v6_gate
import evaluate_constructive_code_v3_coder_viability as base_eval
import replay_constructive_code_v5 as replay_v5


EXPECTED_REPLAY_COUNT = 960
DEVELOPMENT_PROBLEMS = ("359_B", "988_A", "1399_D")
EVALUATION_PROBLEMS = ("361_B", "1294_C", "149_C")


def _validate_v6_gate_and_choose_suites(
    gate: Mapping[str, Any], problem_keys: Mapping[str, str]
) -> dict[str, str]:
    if (
        gate.get("status") != "pass"
        or gate.get("expected_replay_count") != EXPECTED_REPLAY_COUNT
        or gate.get("observed_replay_count") != EXPECTED_REPLAY_COUNT
        or gate.get("violations") not in ([], None)
        or gate.get("checker_equivalence_violations") not in ([], None)
        or gate.get("evaluation_rows_loaded") is not False
        or gate.get("language_model_sampling") is not False
    ):
        raise ValueError("ConstructiveCode v6 executable gate is not an exact pass")
    task_rows = gate.get("task_results")
    suite_rows = gate.get("suite_results")
    if not isinstance(task_rows, list) or len(task_rows) != 10:
        raise ValueError("ConstructiveCode v6 gate lacks ten task results")
    if not isinstance(suite_rows, list) or len(suite_rows) != 10:
        raise ValueError("ConstructiveCode v6 gate lacks ten suite results")
    task_by_key = {
        str(row.get("problem_key")): row
        for row in task_rows
        if isinstance(row, Mapping)
    }
    suite_status = {
        (str(row.get("problem_key")), str(row.get("suite_id"))): row.get("status")
        for row in suite_rows
        if isinstance(row, Mapping)
    }
    selected = {}
    for problem_id in DEVELOPMENT_PROBLEMS:
        problem_key = problem_keys[problem_id]
        row = task_by_key.get(problem_key)
        expected_suite = v6_gate.SELECTED_SUITES[problem_id]
        if (
            not isinstance(row, Mapping)
            or row.get("source_problem_id") != problem_id
            or row.get("status") != "pass"
            or row.get("selected_suite_id") != expected_suite
            or suite_status.get((problem_key, expected_suite)) != "pass"
        ):
            raise ValueError(f"v6 development task did not pass: {problem_id}")
        selected[problem_id] = expected_suite
    return selected


def _replay_modules():
    materialize = replay_v5.materialize_v5
    materialize.V3_TASKS = materialize.V5_TASKS
    materialize.V3_SPLIT_ASSIGNMENT = v6_gate.SPLIT_ASSIGNMENT
    materialize.V5_SPLIT_ASSIGNMENT = v6_gate.SPLIT_ASSIGNMENT
    return replay_v5, replay_v5.base, materialize


def main() -> None:
    base_eval.DEVELOPMENT_PROBLEMS = DEVELOPMENT_PROBLEMS
    base_eval.EVALUATION_PROBLEMS = EVALUATION_PROBLEMS
    base_eval.RECEIPT_SCHEMA = "constructive-code-v6-coder-05b-viability-v1"
    base_eval._validate_gate_and_choose_suites = _validate_v6_gate_and_choose_suites
    base_eval._replay_modules = _replay_modules
    base_eval.main()


if __name__ == "__main__":
    main()
