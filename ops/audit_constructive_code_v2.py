#!/usr/bin/env python3
"""Audit the frozen ConstructiveCode v2 dual-suite executable gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from audit_constructive_code_checker_equivalence import (
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "var/artifacts/constructive_code_v2_replay_manifest.json"
DEFAULT_REPLAYS = ROOT / "var/artifacts/constructive_code_v2_replays.jsonl"
DEFAULT_EQUIVALENCE = (
    ROOT / "var/artifacts/constructive_code_v2_checker_equivalence.json"
)
DEFAULT_OUTPUT = ROOT / "var/artifacts/constructive_code_v2_gate_audit.json"
SCHEMA_VERSION = "constructive-code-v2-gate-audit-v1"
EXPECTED_TASKS = {
    "327_B": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "1294_C": ("unordered_set", "multi_case_status_integer_set_v1"),
    "1283_C": ("assignment", "implicit_assignment_v1"),
    "1102_B": ("unordered_partition", "status_label_partition_v1"),
}
REQUIRED_SUITE_IDS = frozenset(
    {"codecontests_o_corner_cases_v2", "codecontests_plus_5x_v2"}
)
REQUIRED_LANGUAGES = ["py3", "pypy3", "python3"]
REQUIRED_PER_LABEL = 100
VERIFIED_THRESHOLD = 0.9
MEDIAN_LIMIT_SECONDS = 0.5
P95_LIMIT_SECONDS = 1.0


def _violation(
    violations: list[dict[str, Any]],
    code: str,
    **context: Any,
) -> None:
    violations.append({"code": code, **context})


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        records.append(value)
    return records


def build_v2_gate_audit(
    manifest: Mapping[str, Any],
    replays: Sequence[Mapping[str, Any]],
    equivalence_audit: Mapping[str, Any],
    *,
    expected_tasks: Mapping[str, tuple[str, str]] | None = None,
    required_suite_ids: frozenset[str] | None = None,
    suite_policy: str = "all",
    schema_version: str = SCHEMA_VERSION,
    decision_boundary: str = (
        "Pass admits only ConstructiveCode v2 split construction and "
        "development-only Qwen2.5-Coder-0.5B viability sampling."
    ),
) -> dict[str, Any]:
    """Return pass only when every frozen task passes both complete suites."""

    expected_tasks = EXPECTED_TASKS if expected_tasks is None else expected_tasks
    required_suite_ids = (
        REQUIRED_SUITE_IDS if required_suite_ids is None else required_suite_ids
    )
    if suite_policy not in {"all", "at_least_one"}:
        raise ValueError(f"unsupported suite policy: {suite_policy}")
    if not expected_tasks or not required_suite_ids:
        raise ValueError("expected tasks and required suites must be nonempty")
    violations: list[dict[str, Any]] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        _violation(violations, "manifest_schema_mismatch")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or (
        selection.get("languages") != REQUIRED_LANGUAGES
        or selection.get("correct_per_task") != REQUIRED_PER_LABEL
        or selection.get("incorrect_per_task") != REQUIRED_PER_LABEL
        or selection.get("order")
        != "exact code SHA-256 ascending within known label"
    ):
        _violation(violations, "frozen_selection_contract_mismatch")
    preflight = manifest.get("preflight")
    if not isinstance(preflight, Mapping) or (
        preflight.get("identity_and_hash_checks") != "pass"
    ):
        _violation(violations, "identity_and_hash_preflight_failed")

    contracts: dict[tuple[str, str], dict[str, Any]] = {}
    problem_keys: dict[str, str] = {}
    tasks = manifest.get("tasks")
    seen_problem_ids: set[str] = set()
    if not isinstance(tasks, list):
        tasks = []
        _violation(violations, "manifest_tasks_missing")
    for task in tasks:
        if not isinstance(task, Mapping):
            _violation(violations, "manifest_task_malformed")
            continue
        problem_id = str(task.get("source_problem_id") or "")
        problem_key = str(task.get("problem_key") or "")
        expected = expected_tasks.get(problem_id)
        if (
            not problem_key
            or expected is None
            or task.get("witness_family") != expected[0]
            or task.get("task_adapter") != expected[1]
            or problem_id in seen_problem_ids
        ):
            _violation(
                violations,
                "manifest_task_identity_mismatch",
                source_problem_id=problem_id,
            )
            continue
        seen_problem_ids.add(problem_id)
        problem_keys[problem_id] = problem_key
        suites = task.get("suites")
        if not isinstance(suites, list) or len(suites) != len(required_suite_ids) or {
            suite.get("suite_id")
            for suite in suites
            if isinstance(suite, Mapping)
        } != required_suite_ids:
            _violation(
                violations,
                "manifest_dual_suite_contract_mismatch",
                source_problem_id=problem_id,
            )
            continue
        for suite in suites:
            if not isinstance(suite, Mapping):
                continue
            suite_id = str(suite.get("suite_id") or "")
            contract = {
                **suite,
                "source_problem_id": problem_id,
                "task_adapter": expected[1],
                "witness_family": expected[0],
            }
            if (
                suite.get("required_correct_replays") != REQUIRED_PER_LABEL
                or suite.get("required_incorrect_replays") != REQUIRED_PER_LABEL
                or suite.get("verified_threshold") != VERIFIED_THRESHOLD
                or not isinstance(suite.get("test_count"), int)
                or isinstance(suite.get("test_count"), bool)
                or suite["test_count"] < 1
            ):
                _violation(
                    violations,
                    "manifest_suite_threshold_mismatch",
                    source_problem_id=problem_id,
                    suite_id=suite_id,
                )
            contracts[(problem_key, suite_id)] = contract
    if seen_problem_ids != set(expected_tasks):
        _violation(
            violations,
            "frozen_task_set_mismatch",
            observed=sorted(seen_problem_ids),
        )
    if len(contracts) != len(expected_tasks) * len(required_suite_ids):
        _violation(violations, "frozen_problem_suite_count_mismatch")

    equivalence_by_suite = {
        (str(row.get("problem_key") or ""), str(row.get("suite_id") or "")): row
        for row in equivalence_audit.get("suite_results", [])
        if isinstance(row, Mapping)
    }
    if suite_policy == "all" and equivalence_audit.get("status") != "pass":
        _violation(violations, "checker_equivalence_audit_failed")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen_replays: set[tuple[str, str, str]] = set()
    for index, replay in enumerate(replays):
        problem_key = str(replay.get("problem_key") or "")
        suite_id = str(replay.get("suite_id") or "")
        submission_sha = str(replay.get("submission_sha256") or "")
        key = (problem_key, suite_id)
        identity = (*key, submission_sha)
        if replay.get("schema_version") != REPLAY_SCHEMA or key not in contracts:
            _violation(violations, "unexpected_replay_identity", replay_index=index)
            continue
        contract = contracts[key]
        if (
            replay.get("source_problem_id") != contract["source_problem_id"]
            or replay.get("task_adapter") != contract["task_adapter"]
            or replay.get("witness_family") != contract["witness_family"]
            or replay.get("suite_sha256") != contract.get("suite_sha256")
            or replay.get("checker_sha256") != contract.get("checker_sha256")
        ):
            _violation(
                violations,
                "replay_identity_or_hash_mismatch",
                replay_index=index,
                problem_key=problem_key,
                suite_id=suite_id,
            )
        if identity in seen_replays:
            _violation(
                violations,
                "duplicate_replay_identity",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
            )
            continue
        seen_replays.add(identity)
        grouped[key].append(replay)

    suite_results = []
    selections: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(dict)
    for key, contract in sorted(contracts.items()):
        problem_key, suite_id = key
        records = grouped.get(key, [])
        correct = [row for row in records if row.get("known_label") == "correct"]
        incorrect = [
            row for row in records if row.get("known_label") == "incorrect"
        ]
        selections[problem_key][suite_id] = {
            "correct": {str(row.get("submission_sha256") or "") for row in correct},
            "incorrect": {
                str(row.get("submission_sha256") or "") for row in incorrect
            },
        }
        timings: list[float] = []
        execution_records_valid = True
        timeout_count = 0
        isolation_count = 0
        output_bound_count = 0
        for replay in records:
            execution = replay.get("execution")
            if not isinstance(execution, Mapping):
                execution_records_valid = False
                continue
            raw_timings = execution.get("candidate_invocation_wall_seconds")
            executed_tests = execution.get("executed_tests")
            valid_timings = (
                isinstance(raw_timings, list)
                and isinstance(executed_tests, int)
                and not isinstance(executed_tests, bool)
                and executed_tests > 0
                and executed_tests <= contract["test_count"]
                and len(raw_timings) == executed_tests
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and value >= 0
                    for value in raw_timings
                )
            )
            if not valid_timings:
                execution_records_valid = False
                continue
            numeric_timings = [float(value) for value in raw_timings]
            timings.extend(numeric_timings)
            aggregate = execution.get("candidate_wall_seconds")
            checker_seconds = execution.get("checker_wall_seconds")
            if (
                not isinstance(aggregate, (int, float))
                or isinstance(aggregate, bool)
                or not math.isclose(
                    float(aggregate),
                    sum(numeric_timings),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not isinstance(checker_seconds, (int, float))
                or isinstance(checker_seconds, bool)
                or not math.isfinite(float(checker_seconds))
                or checker_seconds < 0
                or execution.get("suite_tests") != contract["test_count"]
            ):
                execution_records_valid = False
            failure = execution.get("first_failure")
            if (
                replay.get("released_checker_accepted") is False
                or replay.get("wrapper_accepted") is False
            ) and not isinstance(failure, Mapping):
                execution_records_valid = False
            if isinstance(failure, Mapping):
                if failure.get("stage") == "candidate":
                    timeout_count += int(failure.get("timed_out") is True)
                    isolation_count += int(
                        failure.get("sandbox_violation") is True
                    )
                    output_bound_count += int(
                        failure.get("output_limited") is True
                    )
                elif failure.get("stage") == "released_checker":
                    timeout_count += int(failure.get("timed_out") is True)
        median_seconds = _percentile(timings, 0.5) if timings else math.inf
        p95_seconds = _percentile(timings, 0.95) if timings else math.inf
        equivalence = equivalence_by_suite.get(key)
        checks = {
            "exact_correct_replays": len(correct) == REQUIRED_PER_LABEL,
            "exact_incorrect_replays": len(incorrect) == REQUIRED_PER_LABEL,
            "wrapper_released_decision_equality": all(
                row.get("wrapper_accepted")
                == row.get("released_checker_accepted")
                for row in records
            ),
            "checker_equivalence": (
                isinstance(equivalence, Mapping)
                and equivalence.get("status") == "pass"
            ),
            "execution_records_valid": execution_records_valid and bool(timings),
            "zero_timeout_violations": timeout_count == 0,
            "zero_isolation_violations": isolation_count == 0,
            "zero_output_bound_violations": output_bound_count == 0,
            "median_candidate_execution": median_seconds <= MEDIAN_LIMIT_SECONDS,
            "p95_candidate_execution": p95_seconds <= P95_LIMIT_SECONDS,
        }
        hard_failures = sorted(
            name
            for name, passed in checks.items()
            if name != "checker_equivalence" and not passed
        )
        if hard_failures or (suite_policy == "all" and not all(checks.values())):
            _violation(
                violations,
                "suite_executable_gate_failed",
                problem_key=problem_key,
                suite_id=suite_id,
                failed_checks=sorted(name for name, passed in checks.items() if not passed),
            )
        suite_results.append(
            {
                "problem_key": problem_key,
                "suite_id": suite_id,
                "counts": {
                    "correct": len(correct),
                    "incorrect": len(incorrect),
                    "candidate_executions": len(timings),
                    "timeouts": timeout_count,
                    "isolation_violations": isolation_count,
                    "output_bound_violations": output_bound_count,
                },
                "latency_seconds": {
                    "median": median_seconds,
                    "p95": p95_seconds,
                },
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            }
        )

    task_results = []
    for problem_id, problem_key in sorted(problem_keys.items()):
        by_suite = selections.get(problem_key, {})
        suite_ids = sorted(required_suite_ids)
        same_programs = len(by_suite) == len(suite_ids) and all(
            by_suite.get(suite_id, {}).get(label)
            == by_suite.get(suite_ids[0], {}).get(label)
            for suite_id in suite_ids[1:]
            for label in ("correct", "incorrect")
        )
        problem_suite_results = [
            row for row in suite_results if row["problem_key"] == problem_key
        ]
        passing_suite_ids = sorted(
            row["suite_id"]
            for row in problem_suite_results
            if row["status"] == "pass"
        )
        suite_pass = (
            len(passing_suite_ids) == len(required_suite_ids)
            if suite_policy == "all"
            else bool(passing_suite_ids)
        )
        status = "pass" if same_programs and suite_pass else "fail"
        if not same_programs:
            _violation(
                violations,
                "cross_suite_submission_set_mismatch",
                source_problem_id=problem_id,
                problem_key=problem_key,
            )
        if not suite_pass:
            _violation(
                violations,
                "task_suite_admission_failed",
                source_problem_id=problem_id,
                problem_key=problem_key,
                suite_policy=suite_policy,
            )
        selected_suite_id = None
        if passing_suite_ids:
            overlay = "codecontests_o_corner_cases_v2"
            selected_suite_id = (
                overlay if overlay in passing_suite_ids else passing_suite_ids[0]
            )
        task_results.append(
            {
                "source_problem_id": problem_id,
                "problem_key": problem_key,
                "same_programs_across_suites": same_programs,
                "passing_suite_ids": passing_suite_ids,
                "selected_suite_id": selected_suite_id,
                "status": status,
            }
        )

    status = "pass" if contracts and not violations else "fail"
    return {
        "schema_version": schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision_boundary": decision_boundary,
        "suite_policy": suite_policy,
        "observed_replay_count": len(replays),
        "expected_replay_count": (
            len(expected_tasks) * len(required_suite_ids) * 2 * REQUIRED_PER_LABEL
        ),
        "suite_results": suite_results,
        "task_results": task_results,
        "checker_equivalence_violations": equivalence_audit.get("violations"),
        "violations": violations,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--equivalence", type=Path, default=DEFAULT_EQUIVALENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_v2_gate_audit(
        _load_json(args.manifest),
        _load_jsonl(args.replays),
        _load_json(args.equivalence),
    )
    _write_json(args.output, audit)
    print(
        f"[constructive-code-v2-gate] status={audit['status']} "
        f"replays={audit['observed_replay_count']} "
        f"violations={len(audit['violations'])} output={args.output}",
        flush=True,
    )
    raise SystemExit(0 if audit["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
