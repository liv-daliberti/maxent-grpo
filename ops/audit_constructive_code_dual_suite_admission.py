#!/usr/bin/env python3
"""Join frozen overlay and Plus replay ledgers into task-level admission."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from audit_constructive_code_checker_equivalence import (
    MANIFEST_SCHEMA,
    build_checker_equivalence_audit,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAY_MANIFEST = (
    ROOT / "var/artifacts/constructive_code_admission_manifest_overlay_20x.json"
)
DEFAULT_OVERLAY_REPLAYS = (
    ROOT
    / "var/artifacts/constructive_code_checker_replays_admission_overlay_20x.jsonl"
)
DEFAULT_PLUS_MANIFEST = (
    ROOT / "var/artifacts/constructive_code_admission_manifest_plus_5x_20x.json"
)
DEFAULT_PLUS_REPLAYS = (
    ROOT
    / "var/artifacts/constructive_code_checker_replays_admission_plus_5x_20x.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "var/artifacts/constructive_code_dual_suite_admission_20x.json"
)
SCHEMA_VERSION = "constructive-code-dual-suite-admission-v1"
REQUIRED_SUITE_IDS = frozenset(
    {"codecontests_o_corner_cases_v1", "codecontests_plus_5x_v1"}
)
TASK_FIELDS = (
    "problem_key",
    "source_problem_id",
    "task_adapter",
    "witness_family",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(record)
    return records


def _task_identity(task: Mapping[str, Any]) -> tuple[str, str, str, str]:
    identity = tuple(str(task.get(field) or "") for field in TASK_FIELDS)
    if any(not value for value in identity):
        raise ValueError("task identity is incomplete")
    return identity  # type: ignore[return-value]


def _selection_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("manifest selection is missing")
    contract = {
        "language": selection.get("language"),
        "order": selection.get("order"),
        "correct_per_task": selection.get("correct_per_task"),
        "incorrect_per_task": selection.get("incorrect_per_task"),
    }
    if (
        contract["language"] != "py3"
        or contract["correct_per_task"] != 20
        or contract["incorrect_per_task"] != 20
        or not isinstance(contract["order"], str)
        or not contract["order"]
    ):
        raise ValueError("manifest does not carry the frozen 20+20 selection")
    return contract


def _replay_selection(
    replays: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, set[str]]]:
    selected: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"correct": set(), "incorrect": set()}
    )
    for replay in replays:
        problem_key = str(replay.get("problem_key") or "")
        label = str(replay.get("known_label") or "")
        submission = str(replay.get("submission_sha256") or "")
        if not problem_key or label not in {"correct", "incorrect"} or not submission:
            raise ValueError("replay selection identity is malformed")
        selected[problem_key][label].add(submission)
    return selected


def build_dual_suite_admission(
    overlay_manifest: Mapping[str, Any],
    overlay_replays: Sequence[Mapping[str, Any]],
    plus_manifest: Mapping[str, Any],
    plus_replays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return task eligibility only when both suite contracts agree exactly."""

    manifests = (overlay_manifest, plus_manifest)
    if any(manifest.get("schema_version") != MANIFEST_SCHEMA for manifest in manifests):
        raise ValueError("admission manifest schema mismatch")
    selections = [_selection_contract(manifest) for manifest in manifests]
    if selections[0] != selections[1]:
        raise ValueError("cross-suite replay selection contract drift")
    for field in ("execution_limits", "runtime", "testlib"):
        if _canonical_bytes(overlay_manifest.get(field)) != _canonical_bytes(
            plus_manifest.get(field)
        ):
            raise ValueError(f"cross-suite {field} contract drift")

    tasks_by_problem: dict[str, dict[str, Any]] = {}
    suite_sources: dict[str, str] = {}
    for source, manifest in zip(("overlay", "plus_5x"), manifests):
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"{source} manifest has no tasks")
        for task in tasks:
            if not isinstance(task, Mapping):
                raise ValueError(f"{source} manifest task is malformed")
            identity = _task_identity(task)
            problem_key = identity[0]
            raw_suites = task.get("suites")
            if not isinstance(raw_suites, list) or len(raw_suites) != 1:
                raise ValueError(f"{source} task must carry exactly one suite")
            suite = dict(raw_suites[0])
            suite_id = str(suite.get("suite_id") or "")
            prior_source = suite_sources.setdefault(suite_id, source)
            if prior_source != source:
                raise ValueError("suite ID appears in both source manifests")
            existing = tasks_by_problem.get(problem_key)
            if existing is None:
                existing = {
                    field: identity[index]
                    for index, field in enumerate(TASK_FIELDS)
                }
                existing["suites"] = []
                tasks_by_problem[problem_key] = existing
            elif _task_identity(existing) != identity:
                raise ValueError(f"cross-suite task identity drift: {problem_key}")
            existing["suites"].append(suite)

    if set(suite_sources) != REQUIRED_SUITE_IDS:
        raise ValueError("required suite IDs are incomplete or unexpected")
    for problem_key, task in tasks_by_problem.items():
        suite_ids = {str(suite.get("suite_id") or "") for suite in task["suites"]}
        if suite_ids != REQUIRED_SUITE_IDS:
            raise ValueError(f"task lacks both required suites: {problem_key}")
        task["suites"].sort(key=lambda suite: str(suite["suite_id"]))

    overlay_selection = _replay_selection(overlay_replays)
    plus_selection = _replay_selection(plus_replays)
    if overlay_selection != plus_selection:
        raise ValueError("cross-suite exact submission selection drift")
    if set(overlay_selection) != set(tasks_by_problem):
        raise ValueError("replay problem set differs from the task manifest")

    merged_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "tasks": sorted(
            tasks_by_problem.values(),
            key=lambda task: str(task["problem_key"]),
        ),
    }
    merged_replays = [*overlay_replays, *plus_replays]
    audit = build_checker_equivalence_audit(merged_manifest, merged_replays)
    structural_violations = [
        violation
        for violation in audit["violations"]
        if not (
            str(violation.get("code") or "").startswith("suite_")
            and str(violation.get("code") or "").endswith("_failed")
        )
    ]
    if structural_violations:
        raise ValueError("merged checker audit has structural violations")

    result_by_problem: dict[str, dict[str, Any]] = defaultdict(dict)
    for suite in audit["suite_results"]:
        result_by_problem[str(suite["problem_key"])][str(suite["suite_id"])] = suite
    task_results: list[dict[str, Any]] = []
    for problem_key, task in sorted(tasks_by_problem.items()):
        suites = result_by_problem.get(problem_key, {})
        eligible = (
            set(suites) == REQUIRED_SUITE_IDS
            and all(suite["status"] == "pass" for suite in suites.values())
        )
        failures = [
            {
                "suite_id": suite_id,
                "failed_checks": sorted(
                    check for check, passed in suite["checks"].items() if not passed
                ),
            }
            for suite_id, suite in sorted(suites.items())
            if suite["status"] != "pass"
        ]
        task_results.append(
            {
                **{field: task[field] for field in TASK_FIELDS},
                "status": "eligible" if eligible else "ineligible",
                "failures": failures,
                "suite_results": [suites[key] for key in sorted(suites)],
            }
        )

    eligible_count = sum(task["status"] == "eligible" for task in task_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision_boundary": (
            "Only tasks marked eligible passed the frozen 20+20 released-checker "
            "and semantic-key gate on both required suites. This report does not "
            "establish base-model viability or authorize evaluation sampling."
        ),
        "selection": selections[0],
        "required_suite_ids": sorted(REQUIRED_SUITE_IDS),
        "task_count": len(task_results),
        "eligible_task_count": eligible_count,
        "ineligible_task_count": len(task_results) - eligible_count,
        "all_slate_tasks_eligible": eligible_count == len(task_results),
        "tasks": task_results,
        "merged_checker_audit": audit,
    }


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
    parser.add_argument("--overlay-manifest", type=Path, default=DEFAULT_OVERLAY_MANIFEST)
    parser.add_argument("--overlay-replays", type=Path, default=DEFAULT_OVERLAY_REPLAYS)
    parser.add_argument("--plus-manifest", type=Path, default=DEFAULT_PLUS_MANIFEST)
    parser.add_argument("--plus-replays", type=Path, default=DEFAULT_PLUS_REPLAYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dual_suite_admission(
        _load_json(args.overlay_manifest),
        _load_jsonl(args.overlay_replays),
        _load_json(args.plus_manifest),
        _load_jsonl(args.plus_replays),
    )
    report["source_artifacts"] = {
        "overlay_manifest_sha256": _sha256_path(args.overlay_manifest),
        "overlay_replays_sha256": _sha256_path(args.overlay_replays),
        "plus_manifest_sha256": _sha256_path(args.plus_manifest),
        "plus_replays_sha256": _sha256_path(args.plus_replays),
    }
    _write_json(args.output, report)
    print(
        "[constructive-code-dual-suite] "
        f"status={report['status']} eligible={report['eligible_task_count']}/"
        f"{report['task_count']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
