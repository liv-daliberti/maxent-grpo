#!/usr/bin/env python3
"""Build the pre-model ConstructiveCode v6 curation manifest and gate audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import audit_constructive_code_v2 as base_audit


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/preregistration/constructive_code_executable_slate_v6_20260730.md"
V5_SOURCE_MANIFEST = ROOT / "var/data/constructive_code_v5/manifest.json"
V5_REPLAY_MANIFEST = ROOT / "var/artifacts/constructive_code_v5_replay_manifest.json"
V5_REPLAYS = ROOT / "var/artifacts/constructive_code_v5_replays.jsonl"
V5_EQUIVALENCE = ROOT / "var/artifacts/constructive_code_v5_checker_equivalence.json"
V5_AUDIT = ROOT / "var/artifacts/constructive_code_v5_gate_audit.json"
V5_IDENTITY = ROOT / "var/artifacts/constructive_code_v5_gate_identity.json"
DEFAULT_MANIFEST = ROOT / "var/data/constructive_code_v6/manifest.json"
DEFAULT_AUDIT = ROOT / "var/artifacts/constructive_code_v6_gate_audit.json"
DEFAULT_IDENTITY = ROOT / "var/artifacts/constructive_code_v6_gate_identity.json"

OVERLAY = "codecontests_o_corner_cases_v2"
PLUS = "codecontests_plus_5x_v2"
SPLIT_ASSIGNMENT = {
    "327_B": "train",
    "659_C": "train",
    "1283_C": "train",
    "1102_B": "train",
    "359_B": "development",
    "988_A": "development",
    "1399_D": "development",
    "361_B": "evaluation",
    "1294_C": "evaluation",
    "149_C": "evaluation",
}
SELECTED_SUITES = {
    "327_B": OVERLAY,
    "659_C": OVERLAY,
    "1283_C": PLUS,
    "1102_B": OVERLAY,
    "359_B": OVERLAY,
    "988_A": OVERLAY,
    "1399_D": PLUS,
    "361_B": OVERLAY,
    "1294_C": OVERLAY,
    "149_C": OVERLAY,
}
EXCLUDED_TASKS = ("1208_C", "1408_A")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    return rows


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _assert_pre_model_boundary(outputs: tuple[Path, ...]) -> None:
    forbidden = (
        ROOT / "var/artifacts/constructive_code_v5_coder_05b_viability.json",
        ROOT / "var/artifacts/constructive_code_v6_coder_05b_viability.json",
        ROOT / "var/artifacts/constructive_code_v6_paired_smoke_identity.json",
        ROOT / "var/artifacts/constructive_code_v6_stage_b_identity.json",
    )
    present = [path for path in forbidden if path.exists()]
    if present:
        raise FileExistsError(f"v6 gate must precede model sampling: {present}")
    present_outputs = [path for path in outputs if path.exists()]
    if present_outputs:
        raise FileExistsError(f"fresh v6 gate outputs required: {present_outputs}")


def _validate_failed_v5(v5_audit: Mapping[str, Any]) -> None:
    task_rows = v5_audit.get("task_results")
    if (
        v5_audit.get("status") != "fail"
        or v5_audit.get("expected_replay_count") != 2304
        or v5_audit.get("observed_replay_count") != 2304
        or not isinstance(task_rows, list)
        or len(task_rows) != 12
    ):
        raise ValueError("v5 antecedent is not the exact failed 2,304-record gate")
    failed = sorted(
        str(row.get("source_problem_id"))
        for row in task_rows
        if isinstance(row, Mapping) and row.get("status") != "pass"
    )
    passed = {
        str(row.get("source_problem_id"))
        for row in task_rows
        if isinstance(row, Mapping) and row.get("status") == "pass"
    }
    if failed != sorted(EXCLUDED_TASKS) or passed != set(SELECTED_SUITES):
        raise ValueError(f"v5 task outcome set drift: failed={failed} passed={sorted(passed)}")
    selected_by_v5 = {
        str(row.get("source_problem_id")): str(row.get("selected_suite_id"))
        for row in task_rows
        if isinstance(row, Mapping) and row.get("status") == "pass"
    }
    if selected_by_v5 != SELECTED_SUITES:
        raise ValueError("v6 frozen suite map differs from complete v5 outcomes")


def build() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_manifest = read_json(V5_SOURCE_MANIFEST)
    replay_manifest = read_json(V5_REPLAY_MANIFEST)
    replays = read_jsonl(V5_REPLAYS)
    equivalence = read_json(V5_EQUIVALENCE)
    v5_audit = read_json(V5_AUDIT)
    v5_identity = read_json(V5_IDENTITY)
    _validate_failed_v5(v5_audit)
    if len(replays) != 2304:
        raise ValueError("v5 raw replay ledger is not exactly 2,304 records")

    source_tasks = {
        str(row.get("source_problem_id")): row
        for row in source_manifest.get("tasks", [])
        if isinstance(row, Mapping)
    }
    replay_tasks = {
        str(row.get("source_problem_id")): row
        for row in replay_manifest.get("tasks", [])
        if isinstance(row, Mapping)
    }
    if set(source_tasks) != set(replay_tasks) or len(source_tasks) != 12:
        raise ValueError("v5 source/replay task manifests disagree")

    base_audit.REQUIRED_PER_LABEL = 48
    per_task_audits = []
    selected_replays: list[dict[str, Any]] = []
    for problem_id, selected_suite in SELECTED_SUITES.items():
        replay_task = replay_tasks[problem_id]
        problem_key = str(replay_task.get("problem_key"))
        expected_tasks = {
            problem_id: (
                str(replay_task.get("witness_family")),
                str(replay_task.get("task_adapter")),
            )
        }
        selected_contracts = [
            suite
            for suite in replay_task.get("suites", [])
            if isinstance(suite, Mapping) and suite.get("suite_id") == selected_suite
        ]
        task_manifest = {
            **replay_manifest,
            "tasks": [{**replay_task, "suites": selected_contracts}],
        }
        task_replays = [
            row
            for row in replays
            if row.get("source_problem_id") == problem_id
            and row.get("suite_id") == selected_suite
        ]
        suite_results = [
            row
            for row in equivalence.get("suite_results", [])
            if isinstance(row, Mapping)
            and row.get("problem_key") == problem_key
            and row.get("suite_id") == selected_suite
        ]
        task_equivalence = {
            **equivalence,
            "status": "pass" if len(suite_results) == 1 and suite_results[0].get("status") == "pass" else "fail",
            "expected_suite_count": 1,
            "observed_replay_count": len(task_replays),
            "suite_results": suite_results,
            "violations": [],
        }
        task_audit = base_audit.build_v2_gate_audit(
            task_manifest,
            task_replays,
            task_equivalence,
            expected_tasks=expected_tasks,
            required_suite_ids=frozenset({selected_suite}),
            suite_policy="all",
            schema_version="constructive-code-v6-task-suite-gate-audit-v1",
            decision_boundary="All ten frozen task-suite audits must pass before v6 viability.",
        )
        if (
            task_audit.get("status") != "pass"
            or task_audit.get("expected_replay_count") != 96
            or task_audit.get("observed_replay_count") != 96
            or task_audit.get("violations") not in ([], None)
            or task_audit.get("checker_equivalence_violations") not in ([], None)
        ):
            raise ValueError(f"v6 selected task-suite did not independently pass: {problem_id}")
        per_task_audits.append(task_audit)
        selected_replays.extend(task_replays)

    if len(selected_replays) != 960 or len({
        (row.get("problem_key"), row.get("suite_id"), row.get("submission_sha256"))
        for row in selected_replays
    }) != 960:
        raise ValueError("v6 selected replay set is not 960 unique records")

    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "constructive-code-slate-v6-curation-v1",
        "generated_at": generated,
        "status": "admitted_pre_model",
        "preregistration_sha256": sha256_file(PROTOCOL),
        "underlying_source_slate": "var/data/constructive_code_v5",
        "underlying_source_manifest_sha256": sha256_file(V5_SOURCE_MANIFEST),
        "underlying_replay_manifest_sha256": sha256_file(V5_REPLAY_MANIFEST),
        "underlying_replays_sha256": sha256_file(V5_REPLAYS),
        "underlying_checker_equivalence_sha256": sha256_file(V5_EQUIVALENCE),
        "failed_v5_gate_audit_sha256": sha256_file(V5_AUDIT),
        "failed_v5_gate_identity_sha256": sha256_file(V5_IDENTITY),
        "split_assignment": SPLIT_ASSIGNMENT,
        "selected_suites": SELECTED_SUITES,
        "excluded_v5_tasks": list(EXCLUDED_TASKS),
        "tasks": [
            {
                **source_tasks[problem_id],
                "split": SPLIT_ASSIGNMENT[problem_id],
                "selected_suite_id": SELECTED_SUITES[problem_id],
            }
            for problem_id in SPLIT_ASSIGNMENT
        ],
        "evaluation_rows_loaded": False,
        "language_model_sampling": False,
    }
    manifest["tasks_sha256"] = canonical_sha256(manifest["tasks"])
    audit = {
        "schema_version": "constructive-code-v6-gate-audit-v1",
        "generated_at": generated,
        "status": "pass",
        "decision_boundary": "Pass admits only the frozen development-only Qwen2.5-Coder-0.5B viability probe.",
        "suite_policy": "one pre-model v5-admissible official suite frozen per task",
        "observed_replay_count": len(selected_replays),
        "expected_replay_count": 960,
        "task_results": [task_audit["task_results"][0] for task_audit in per_task_audits],
        "suite_results": [task_audit["suite_results"][0] for task_audit in per_task_audits],
        "checker_equivalence_violations": [],
        "violations": [],
        "excluded_v5_tasks": list(EXCLUDED_TASKS),
        "failed_v5_status": v5_audit.get("status"),
        "failed_v5_violations": v5_audit.get("violations"),
        "failed_v5_checker_equivalence_violations": v5_audit.get("checker_equivalence_violations"),
        "selected_replays_sha256": canonical_sha256(selected_replays),
        "per_task_audits_sha256": canonical_sha256(per_task_audits),
        "evaluation_rows_loaded": False,
        "language_model_sampling": False,
    }
    identity = {
        "schema_version": "constructive-code-v6-gate-identity-v1",
        "generated_at": generated,
        "source_hash": v5_identity.get("source_hash"),
        "execution_hash": v5_identity.get("execution_hash"),
        "protocol_sha256": sha256_file(PROTOCOL),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "v6_manifest_sha256": canonical_sha256(manifest),
        "v5_source_manifest_sha256": sha256_file(V5_SOURCE_MANIFEST),
        "v5_replay_manifest_sha256": sha256_file(V5_REPLAY_MANIFEST),
        "v5_replays_sha256": sha256_file(V5_REPLAYS),
        "v5_equivalence_sha256": sha256_file(V5_EQUIVALENCE),
        "failed_v5_audit_sha256": sha256_file(V5_AUDIT),
        "failed_v5_identity_sha256": sha256_file(V5_IDENTITY),
        "expected_selected_replays": 960,
        "frozen_task_count": 10,
        "frozen_split_counts": {"train": 4, "development": 3, "evaluation": 3},
        "evaluation_split_loaded": False,
        "language_model_sampling": False,
        "scientific_change": "new ten-task benchmark version curated from complete pre-model v5 executable evidence",
    }
    return manifest, audit, identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _assert_pre_model_boundary((args.manifest, args.audit, args.identity))
    manifest, audit, identity = build()
    atomic_json(args.manifest, manifest)
    atomic_json(args.audit, audit)
    identity = {**identity, "v6_manifest_file_sha256": sha256_file(args.manifest), "v6_gate_audit_sha256": sha256_file(args.audit)}
    atomic_json(args.identity, identity)
    print(
        f"[constructive-v6-gate] status={audit['status']} tasks={len(audit['task_results'])}/10 "
        f"replays={audit['observed_replay_count']}/960 output={args.audit}",
        flush=True,
    )


if __name__ == "__main__":
    main()
