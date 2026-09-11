#!/usr/bin/env python3
"""Audit released-checker/wrapper equivalence from a frozen replay ledger."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "var/artifacts/constructive_code_checker_equivalence_audit.json"
MANIFEST_SCHEMA = "constructive-code-admission-manifest-v1"
REPLAY_SCHEMA = "constructive-code-checker-replay-v1"
AUDIT_SCHEMA = "constructive-code-checker-equivalence-audit-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _violation(
    violations: list[dict[str, Any]],
    code: str,
    **context: Any,
) -> None:
    violations.append({"code": code, **context})


def _manifest_suites(
    manifest: Mapping[str, Any],
    violations: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        _violation(violations, "manifest_schema_mismatch")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        _violation(violations, "manifest_tasks_missing")
        return {}
    suites: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            _violation(violations, "manifest_task_malformed")
            continue
        problem_key = str(task.get("problem_key") or "")
        raw_suites = task.get("suites")
        if not problem_key or not isinstance(raw_suites, list) or not raw_suites:
            _violation(
                violations,
                "manifest_task_identity_missing",
                problem_key=problem_key,
            )
            continue
        for suite in raw_suites:
            if not isinstance(suite, Mapping):
                _violation(
                    violations,
                    "manifest_suite_malformed",
                    problem_key=problem_key,
                )
                continue
            suite_id = str(suite.get("suite_id") or "")
            key = (problem_key, suite_id)
            checker_sha = str(suite.get("checker_sha256") or "")
            required_correct = suite.get("required_correct_replays")
            required_incorrect = suite.get("required_incorrect_replays")
            threshold = suite.get("verified_threshold", 0.9)
            if (
                not suite_id
                or _SHA256.fullmatch(checker_sha) is None
                or isinstance(required_correct, bool)
                or not isinstance(required_correct, int)
                or required_correct < 1
                or isinstance(required_incorrect, bool)
                or not isinstance(required_incorrect, int)
                or required_incorrect < 1
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0.0 <= float(threshold) <= 1.0
            ):
                _violation(
                    violations,
                    "manifest_suite_contract_invalid",
                    problem_key=problem_key,
                    suite_id=suite_id,
                )
                continue
            if key in suites:
                _violation(
                    violations,
                    "manifest_suite_duplicate",
                    problem_key=problem_key,
                    suite_id=suite_id,
                )
                continue
            suites[key] = {
                "checker_sha256": checker_sha,
                "required_correct_replays": required_correct,
                "required_incorrect_replays": required_incorrect,
                "verified_threshold": float(threshold),
            }
    return suites


def build_checker_equivalence_audit(
    manifest: Mapping[str, Any],
    replays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a fail-closed equivalence and identity audit."""

    violations: list[dict[str, Any]] = []
    suites = _manifest_suites(manifest, violations)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()

    for index, replay in enumerate(replays):
        if replay.get("schema_version") != REPLAY_SCHEMA:
            _violation(violations, "replay_schema_mismatch", replay_index=index)
            continue
        problem_key = str(replay.get("problem_key") or "")
        suite_id = str(replay.get("suite_id") or "")
        submission_sha = str(replay.get("submission_sha256") or "")
        key = (problem_key, suite_id)
        identity = (*key, submission_sha)
        if key not in suites:
            _violation(
                violations,
                "unexpected_problem_suite",
                replay_index=index,
                problem_key=problem_key,
                suite_id=suite_id,
            )
            continue
        if _SHA256.fullmatch(submission_sha) is None:
            _violation(
                violations,
                "submission_sha256_invalid",
                replay_index=index,
                problem_key=problem_key,
                suite_id=suite_id,
            )
            continue
        if identity in seen:
            _violation(
                violations,
                "duplicate_submission_replay",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
            )
            continue
        seen.add(identity)

        expected_checker = suites[key]["checker_sha256"]
        if replay.get("checker_sha256") != expected_checker:
            _violation(
                violations,
                "checker_sha256_mismatch",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
            )
        label = replay.get("known_label")
        released = replay.get("released_checker_accepted")
        wrapper = replay.get("wrapper_accepted")
        behavior_key = replay.get("behavior_key")
        if label not in {"correct", "incorrect"}:
            _violation(
                violations,
                "known_label_invalid",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
            )
            continue
        if not isinstance(released, bool) or not isinstance(wrapper, bool):
            _violation(
                violations,
                "checker_decision_invalid",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
            )
            continue
        if wrapper != released:
            _violation(
                violations,
                "wrapper_released_checker_mismatch",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
                released_checker_accepted=released,
                wrapper_accepted=wrapper,
            )
        valid_key = (
            isinstance(behavior_key, str)
            and behavior_key.startswith("constructive_behavior:v1:")
            and _SHA256.fullmatch(
                behavior_key.removeprefix("constructive_behavior:v1:")
            )
            is not None
        )
        if wrapper and not valid_key:
            _violation(
                violations,
                "accepted_wrapper_missing_behavior_key",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
            )
        if not wrapper and behavior_key is not None:
            _violation(
                violations,
                "rejected_wrapper_has_behavior_key",
                problem_key=problem_key,
                suite_id=suite_id,
                submission_sha256=submission_sha,
            )
        grouped[key].append(replay)

    suite_results: list[dict[str, Any]] = []
    for (problem_key, suite_id), contract in sorted(suites.items()):
        records = grouped.get((problem_key, suite_id), [])
        correct = [row for row in records if row.get("known_label") == "correct"]
        incorrect = [row for row in records if row.get("known_label") == "incorrect"]
        true_positive = sum(
            row.get("released_checker_accepted") is True for row in correct
        )
        true_negative = sum(
            row.get("released_checker_accepted") is False for row in incorrect
        )
        tpr = true_positive / len(correct) if correct else 0.0
        tnr = true_negative / len(incorrect) if incorrect else 0.0
        distinct_keys = {
            row.get("behavior_key")
            for row in correct
            if row.get("released_checker_accepted") is True
            and isinstance(row.get("behavior_key"), str)
        }
        checks = {
            "correct_replay_count": (
                len(correct) >= contract["required_correct_replays"]
            ),
            "incorrect_replay_count": (
                len(incorrect) >= contract["required_incorrect_replays"]
            ),
            "verified_true_positive_rate": (tpr >= contract["verified_threshold"]),
            "verified_true_negative_rate": (tnr >= contract["verified_threshold"]),
            "multiple_behavior_keys": len(distinct_keys) >= 2,
        }
        for check, passed in checks.items():
            if not passed:
                _violation(
                    violations,
                    f"suite_{check}_failed",
                    problem_key=problem_key,
                    suite_id=suite_id,
                )
        suite_results.append(
            {
                "problem_key": problem_key,
                "suite_id": suite_id,
                "checker_sha256": contract["checker_sha256"],
                "counts": {
                    "correct": len(correct),
                    "incorrect": len(incorrect),
                    "true_positive": true_positive,
                    "true_negative": true_negative,
                    "distinct_correct_behavior_keys": len(distinct_keys),
                },
                "rates": {
                    "true_positive_rate": tpr,
                    "true_negative_rate": tnr,
                },
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            }
        )

    status = "pass" if suites and not violations else "fail"
    return {
        "schema_version": AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision_boundary": (
            "A passing replay ledger establishes wrapper equivalence to the "
            "released checker and stable multi-witness identity only for the "
            "frozen problem/suite manifest. It does not establish model viability."
        ),
        "manifest_schema_version": manifest.get("schema_version"),
        "expected_suite_count": len(suites),
        "observed_replay_count": len(replays),
        "suite_results": suite_results,
        "violations": violations,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: replay must be an object")
        records.append(record)
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    audit = build_checker_equivalence_audit(
        manifest,
        _load_jsonl(args.replays),
    )
    _write_json(args.output, audit)
    print(
        "[constructive-code-checker-equivalence] "
        f"status={audit['status']} "
        f"suites={audit['expected_suite_count']} "
        f"replays={audit['observed_replay_count']} "
        f"violations={len(audit['violations'])} "
        f"output={args.output}",
        flush=True,
    )
    raise SystemExit(0 if audit["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
