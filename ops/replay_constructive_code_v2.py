#!/usr/bin/env python3
"""Run the frozen ConstructiveCode v2 slate against both complete suites."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
TESTLIB_ROOT = Path(os.environ.get("OAT_ZERO_TESTLIB_ROOT", ROOT / "third_party/testlib")).resolve()
SANDBOX_SOURCE = Path(os.environ.get("OAT_ZERO_SANDBOX_SOURCE", ROOT / "ops/constructive_code_sandbox.c")).resolve()
PROTOCOL = Path(os.environ.get("OAT_ZERO_CONSTRUCTIVE_V2_PROTOCOL", ROOT / "paper/preregistration/constructive_code_executable_slate_v2_20260729.md")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_constructive_code_checker_equivalence import (  # noqa: E402
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
    build_checker_equivalence_audit,
)
from audit_constructive_code_v2 import (  # noqa: E402
    REQUIRED_LANGUAGES,
    REQUIRED_PER_LABEL,
    REQUIRED_SUITE_IDS,
    VERIFIED_THRESHOLD,
    build_v2_gate_audit,
)
from materialize_constructive_code_v2 import (  # noqa: E402
    DEFAULT_INDEX,
    OVERLAY_REVISION,
    PLUS_REVISION,
    PYTHON3_LABELS,
    V2_TASKS,
)
from oat_drgrpo.constructive_code import (  # noqa: E402
    CheckedWitness,
    ConstructiveCodeError,
    ReleasedCheckerDecision,
    canonicalize_checked_behavior,
    sha256_bytes,
)
from oat_drgrpo.constructive_code_adapters import (  # noqa: E402
    canonicalize_task_witness,
    registered_task_adapters,
)
from oat_drgrpo.constructive_code_sandbox import (  # noqa: E402
    PINNED_IMAGE_SHA256,
    SandboxLimits,
    build_launcher,
    percentile,
    prepare_runtime,
    run_candidate,
    sha256_file,
)
from replay_constructive_code_review_slate import (  # noqa: E402
    CheckerResult,
    Submission,
    Task,
    _canonical_json_bytes,
    _load_json,
    _load_jsonl,
    _load_tests,
    _run_checker,
    _write_json,
    _write_jsonl,
)


DEFAULT_SLATE = ROOT / "var/data/constructive_code_v2"
DEFAULT_V1 = ROOT / "var/data/constructive_code_review_slate_v1"
DEFAULT_IMAGE = ROOT / "var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
DEFAULT_REPLAYS = ROOT / "var/artifacts/constructive_code_v2_replays.jsonl"
DEFAULT_MANIFEST = ROOT / "var/artifacts/constructive_code_v2_replay_manifest.json"
DEFAULT_EQUIVALENCE = (
    ROOT / "var/artifacts/constructive_code_v2_checker_equivalence.json"
)
DEFAULT_GATE_AUDIT = ROOT / "var/artifacts/constructive_code_v2_gate_audit.json"
DEFAULT_RUN_AUDIT = ROOT / "var/artifacts/constructive_code_v2_run_audit.json"
TESTLIB_REVISION = "1e4e8a24c79c6bad3becbdb5a332ffc352b7d5dd"
TESTLIB_SHA256 = "bb323e3c89285214966076e0d23d5a295c5f6126da7ff198c1276ddb95ecb1a0"
PYTHON_CPU_MULTIPLIER = 3
OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024
CHECKER_WALL_SECONDS = 2
SUITE_FILES = {
    "codecontests_o_corner_cases_v2": "overlay_inputs.jsonl.gz",
    "codecontests_plus_5x_v2": "plus_5x_inputs.jsonl.gz",
}
SOURCE_MANIFEST_SCHEMA = "constructive-code-slate-v2"
TASK_RECORD_SCHEMA = "constructive-code-task-v2"
RUN_STATUS = "v2_frozen_dual_suite_replay"
RUN_AUDIT_SCHEMA = "constructive-code-v2-run-audit-v1"
VERSION_LABEL = "constructive-code-v2-replay"
GATE_AUDIT_HASH_FIELD = "v2_gate_audit_sha256"
REQUIRE_V1_DISJOINT = True
V1_LEDGER_COUNT_FIELD = "excluded_v1_hash_count"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_submission_records(
    records: Sequence[Mapping[str, Any]],
    excluded_v1_hashes: set[str],
) -> tuple[Submission, ...]:
    expected_order = sorted(
        records,
        key=lambda row: (
            str(row.get("known_label") or ""),
            str(row.get("submission_sha256") or ""),
        ),
    )
    if list(records) != expected_order:
        raise ValueError("v2 replay records are not hash-sorted within known label")
    by_label: dict[str, list[Submission]] = {"correct": [], "incorrect": []}
    seen_hashes: set[str] = set()
    for record in records:
        label = str(record.get("known_label") or "")
        language = str(record.get("language") or "")
        code = record.get("code")
        digest = str(record.get("submission_sha256") or "")
        if (
            label not in by_label
            or language not in PYTHON3_LABELS
            or not isinstance(code, str)
            or not code.strip()
            or sha256_bytes(code.encode("utf-8")) != digest
            or digest in seen_hashes
            or (REQUIRE_V1_DISJOINT and digest in excluded_v1_hashes)
        ):
            raise ValueError("malformed, duplicate, or non-held-out v2 replay record")
        seen_hashes.add(digest)
        by_label[label].append(Submission(code, label, digest))
    if any(len(by_label[label]) != REQUIRED_PER_LABEL for label in by_label):
        raise ValueError("v2 requires exactly 100 replays per known label")
    return tuple([*by_label["correct"], *by_label["incorrect"]])


def _v1_hashes(task_dir: Path, expected_problem_id: str) -> set[str]:
    task = _load_json(task_dir / "task.json")
    stored_task_sha = task.get("task_record_sha256")
    unsigned_task = {
        key: value for key, value in task.items() if key != "task_record_sha256"
    }
    replay_path = task_dir / "python_replays.jsonl"
    if (
        task.get("schema_version") != "constructive-code-review-task-v1"
        or task.get("source_problem_id") != expected_problem_id
        or stored_task_sha != _canonical_sha256(unsigned_task)
        or _sha256_path(replay_path)
        != task.get("replays", {}).get("python_replays_jsonl_sha256")
    ):
        raise ValueError(f"sealed v1 exclusion ledger drift: {expected_problem_id}")
    hashes = set()
    for row in _load_jsonl(replay_path):
        code = row.get("code")
        digest = str(row.get("submission_sha256") or "")
        if not isinstance(code, str) or sha256_bytes(code.encode("utf-8")) != digest:
            raise ValueError(f"malformed v1 exclusion record: {expected_problem_id}")
        hashes.add(digest)
    return hashes


def _validate_source_manifest(slate_root: Path) -> dict[str, Any]:
    path = slate_root / "manifest.json"
    manifest = _load_json(path)
    if (
        manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("status") != "pending_executable_replay"
        or manifest.get("tasks_sha256") != _canonical_sha256(manifest.get("tasks"))
    ):
        raise ValueError("invalid ConstructiveCode v2 source manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or (
        sources.get("codecontests_plus", {}).get("revision") != PLUS_REVISION
        or sources.get("codecontests_o", {}).get("revision") != OVERLAY_REVISION
    ):
        raise ValueError("ConstructiveCode v2 source revision drift")
    if sources.get("candidate_index_sha256") != _sha256_path(DEFAULT_INDEX):
        raise ValueError("ConstructiveCode v2 candidate index hash drift")
    summaries = manifest.get("tasks")
    if not isinstance(summaries, list) or {
        summary.get("source_problem_id")
        for summary in summaries
        if isinstance(summary, Mapping)
    } != set(V2_TASKS):
        raise ValueError("ConstructiveCode v2 frozen task set drift")
    if manifest.get("preregistration_sha256") != _sha256_path(PROTOCOL):
        raise ValueError("ConstructiveCode v2 preregistration hash drift")
    return manifest


def _validate_v1_logical_root(source_manifest: Mapping[str, Any]) -> None:
    """Validate the materializer's logical ledger identity, not its copied path."""

    sources = source_manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("ConstructiveCode v2 v1 exclusion root drift")
    declared = PurePosixPath(str(sources.get("v1_root") or ""))
    if declared.is_absolute() or declared.as_posix() != (
        "var/data/constructive_code_review_slate_v1"
    ):
        raise ValueError("ConstructiveCode v2 v1 exclusion root drift")


def _compile_checker(source: Path, output: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"released checker source hash mismatch: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        shutil.which("g++") or "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        "-I",
        str(TESTLIB_ROOT),
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True, timeout=120)
    output.chmod(0o500)
    return {
        "source_sha256": expected_sha256,
        "binary_sha256": sha256_file(output),
        "compile_command": command,
    }


def _load_tasks(
    *,
    slate_root: Path,
    v1_root: Path,
    build_root: Path,
) -> tuple[list[Task], list[dict[str, Any]], dict[str, Any]]:
    source_manifest = _validate_source_manifest(slate_root)
    _validate_v1_logical_root(source_manifest)
    if sha256_file(TESTLIB_ROOT / "testlib.h") != TESTLIB_SHA256:
        raise ValueError("pinned testlib.h hash drift")
    registered = set(registered_task_adapters())
    tasks: list[Task] = []
    builds = []
    for summary in source_manifest["tasks"]:
        problem_id = str(summary["source_problem_id"])
        task_dir = slate_root / str(summary["relative_path"])
        record = _load_json(task_dir / "task.json")
        stored_record_sha = record.get("task_record_sha256")
        unsigned_record = {
            key: value for key, value in record.items() if key != "task_record_sha256"
        }
        expected_family, expected_adapter = V2_TASKS[problem_id]
        if (
            record.get("schema_version") != TASK_RECORD_SCHEMA
            or record.get("source_problem_id") != problem_id
            or record.get("witness_family") != expected_family
            or record.get("task_adapter") != expected_adapter
            or stored_record_sha != _canonical_sha256(unsigned_record)
            or stored_record_sha != summary.get("task_record_sha256")
            or (problem_id, expected_adapter) not in registered
        ):
            raise ValueError(f"v2 task identity or record hash drift: {problem_id}")
        language_contract = record.get("language_contract")
        if not isinstance(language_contract, Mapping) or (
            language_contract.get("accepted_labels") != sorted(PYTHON3_LABELS)
            or language_contract.get("runtime") != "Python 3.10.20"
        ):
            raise ValueError(f"v2 language contract drift: {problem_id}")
        excluded = _v1_hashes(
            v1_root / str(summary["relative_path"]),
            problem_id,
        )
        if language_contract.get(V1_LEDGER_COUNT_FIELD) != len(excluded):
            raise ValueError(f"v1 exclusion ledger drift: {problem_id}")
        replay_path = task_dir / "py3_replays.jsonl"
        if _sha256_path(replay_path) != record.get("replays", {}).get("jsonl_sha256"):
            raise ValueError(f"v2 replay ledger hash drift: {problem_id}")
        submissions = _validate_submission_records(_load_jsonl(replay_path), excluded)
        if (
            record["replays"]["correct"].get("selected") != REQUIRED_PER_LABEL
            or record["replays"]["incorrect"].get("selected")
            != REQUIRED_PER_LABEL
        ):
            raise ValueError(f"v2 replay selection count drift: {problem_id}")
        checker_sha = str(record.get("checker_sha256") or "")
        checker_binary = build_root / problem_id.lower() / "checker"
        build = _compile_checker(task_dir / "checker.cpp", checker_binary, checker_sha)
        build["source_problem_id"] = problem_id
        builds.append(build)
        raw_limits = record.get("limits")
        if not isinstance(raw_limits, Mapping):
            raise ValueError(f"v2 limits missing: {problem_id}")
        time_ms = raw_limits.get("time_milliseconds")
        memory_mb = raw_limits.get("memory_megabytes")
        if (
            isinstance(time_ms, bool)
            or not isinstance(time_ms, int)
            or time_ms <= 0
            or isinstance(memory_mb, bool)
            or not isinstance(memory_mb, int)
            or memory_mb <= 0
        ):
            raise ValueError(f"v2 limits malformed: {problem_id}")
        cpu_seconds = max(1, math.ceil(time_ms * PYTHON_CPU_MULTIPLIER / 1000))
        limits = SandboxLimits(
            cpu_seconds=cpu_seconds,
            wall_seconds=float(cpu_seconds + 2),
            memory_bytes=memory_mb * 1024 * 1024,
            output_bytes=OUTPUT_LIMIT_BYTES,
            file_count=32,
            source_bytes=256 * 1024,
        )
        suites = record.get("suites")
        if not isinstance(suites, Mapping) or set(suites) != REQUIRED_SUITE_IDS:
            raise ValueError(f"v2 suite set drift: {problem_id}")
        for suite_id in sorted(REQUIRED_SUITE_IDS):
            suite = suites[suite_id]
            if not isinstance(suite, Mapping):
                raise ValueError(f"v2 suite metadata malformed: {problem_id}")
            tasks.append(
                Task(
                    problem_id=problem_id,
                    problem_key=str(record["problem_key"]),
                    adapter_id=expected_adapter,
                    witness_family=expected_family,
                    suite_id=suite_id,
                    suite_sha256=str(suite["suite_sha256"]),
                    checker_sha256=checker_sha,
                    checker_binary=checker_binary,
                    tests=_load_tests(task_dir / SUITE_FILES[suite_id], suite),
                    submissions=submissions,
                    limits=limits,
                )
            )
    if (
        len(tasks) != len(V2_TASKS) * len(REQUIRED_SUITE_IDS)
        or len(builds) != len(V2_TASKS)
    ):
        raise ValueError("frozen task or problem-suite count mismatch")
    return tasks, builds, source_manifest


def _replay_submission(
    *,
    task: Task,
    submission: Submission,
    launcher: Path,
    runtime_root: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    observations: list[CheckedWitness] = []
    released_accepted = True
    wrapper_accepted = True
    first_failure: dict[str, Any] | None = None
    candidate_timings = []
    checker_wall_seconds = 0.0
    executed_tests = 0
    for test in task.tests:
        executed_tests += 1
        candidate = run_candidate(
            launcher=launcher,
            runtime_root=runtime_root,
            source=submission.code,
            stdin=test.stdin,
            limits=task.limits,
            scratch_root=scratch_root,
            runtime_is_preverified=True,
        )
        candidate_timings.append(candidate.wall_seconds)
        if not candidate.completed_cleanly:
            released_accepted = False
            wrapper_accepted = False
            first_failure = {
                "stage": "candidate",
                "test_index": test.test_index,
                "returncode": candidate.returncode,
                "timed_out": candidate.timed_out,
                "output_limited": candidate.output_limited,
                "sandbox_violation": candidate.sandbox_violation,
                "stderr": candidate.stderr[-2000:].decode("utf-8", errors="replace"),
            }
            break
        checker: CheckerResult = _run_checker(
            task.checker_binary,
            test.stdin,
            candidate.stdout,
            scratch_root,
        )
        checker_wall_seconds += checker.wall_seconds
        decision = ReleasedCheckerDecision(
            checker_sha256=task.checker_sha256,
            input_sha256=test.input_sha256,
            output_sha256=sha256_bytes(candidate.stdout),
            accepted=checker.accepted,
            exit_code=checker.returncode,
            timed_out=checker.timed_out,
        )
        if not checker.accepted:
            released_accepted = False
            wrapper_accepted = False
            if first_failure is None:
                first_failure = {
                    "stage": "released_checker",
                    "test_index": test.test_index,
                    "returncode": checker.returncode,
                    "timed_out": checker.timed_out,
                    "message": checker.message,
                }
            break
        try:
            witness = canonicalize_task_witness(
                problem_id=task.problem_id,
                adapter_id=task.adapter_id,
                input_data=test.stdin,
                output=candidate.stdout,
                decision=decision,
            )
        except ConstructiveCodeError as error:
            wrapper_accepted = False
            if first_failure is None:
                first_failure = {
                    "stage": "task_adapter",
                    "test_index": test.test_index,
                    "message": str(error),
                }
            continue
        observations.append(CheckedWitness(decision=decision, witness=witness))

    behavior_key = None
    if released_accepted and wrapper_accepted:
        behavior_key = canonicalize_checked_behavior(
            task.problem_key,
            task.suite_sha256,
            observations,
        ).canonical_key
    return {
        "schema_version": REPLAY_SCHEMA,
        "problem_key": task.problem_key,
        "source_problem_id": task.problem_id,
        "task_adapter": task.adapter_id,
        "witness_family": task.witness_family,
        "suite_id": task.suite_id,
        "suite_sha256": task.suite_sha256,
        "checker_sha256": task.checker_sha256,
        "submission_sha256": submission.submission_sha256,
        "known_label": submission.known_label,
        "released_checker_accepted": released_accepted,
        "wrapper_accepted": wrapper_accepted,
        "behavior_key": behavior_key,
        "execution": {
            "candidate_invocation_wall_seconds": candidate_timings,
            "candidate_wall_seconds": sum(candidate_timings),
            "checker_wall_seconds": checker_wall_seconds,
            "executed_tests": executed_tests,
            "suite_tests": len(task.tests),
            "first_failure": first_failure,
        },
    }


def _replay_manifest(
    tasks: Sequence[Task],
    builds: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    slate_root: Path,
) -> dict[str, Any]:
    by_problem: dict[str, list[Task]] = {}
    for task in tasks:
        by_problem.setdefault(task.problem_id, []).append(task)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": RUN_STATUS,
        "selection": {
            "languages": REQUIRED_LANGUAGES,
            "order": "exact code SHA-256 ascending within known label",
            "correct_per_task": REQUIRED_PER_LABEL,
            "incorrect_per_task": REQUIRED_PER_LABEL,
        },
        "preflight": {
            "identity_and_hash_checks": "pass",
            "v1_hash_exclusion": "pass",
        },
        "execution_limits": {
            "python_cpu_multiplier": PYTHON_CPU_MULTIPLIER,
            "output_bytes": OUTPUT_LIMIT_BYTES,
            "released_checker_wall_seconds": CHECKER_WALL_SECONDS,
        },
        "runtime": {
            "image_sha256": PINNED_IMAGE_SHA256,
            "launcher_source_sha256": sha256_file(
                SANDBOX_SOURCE
            ),
        },
        "source_slate": {
            "schema_version": source_manifest["schema_version"],
            "manifest_sha256": _sha256_path(slate_root / "manifest.json"),
            "tasks_sha256": source_manifest["tasks_sha256"],
            "preregistration_sha256": source_manifest["preregistration_sha256"],
        },
        "checker_builds": list(builds),
        "testlib": {
            "revision": TESTLIB_REVISION,
            "testlib_h_sha256": TESTLIB_SHA256,
        },
        "tasks": [
            {
                "problem_key": problem_tasks[0].problem_key,
                "source_problem_id": problem_id,
                "task_adapter": problem_tasks[0].adapter_id,
                "witness_family": problem_tasks[0].witness_family,
                "suites": [
                    {
                        "suite_id": task.suite_id,
                        "suite_sha256": task.suite_sha256,
                        "checker_sha256": task.checker_sha256,
                        "test_count": len(task.tests),
                        "required_correct_replays": REQUIRED_PER_LABEL,
                        "required_incorrect_replays": REQUIRED_PER_LABEL,
                        "verified_threshold": VERIFIED_THRESHOLD,
                    }
                    for task in sorted(problem_tasks, key=lambda value: value.suite_id)
                ],
            }
            for problem_id, problem_tasks in sorted(by_problem.items())
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate-root", type=Path, default=DEFAULT_SLATE)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--replays-output", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--equivalence-output", type=Path, default=DEFAULT_EQUIVALENCE)
    parser.add_argument("--gate-audit-output", type=Path, default=DEFAULT_GATE_AUDIT)
    parser.add_argument("--run-audit-output", type=Path, default=DEFAULT_RUN_AUDIT)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    launcher_sha = build_launcher(SANDBOX_SOURCE, args.launcher)
    runtime_identity = prepare_runtime(args.image, args.runtime_root)
    tasks, builds, source_manifest = _load_tasks(
        slate_root=args.slate_root,
        v1_root=args.v1_root,
        build_root=args.build_root,
    )
    manifest = _replay_manifest(tasks, builds, source_manifest, args.slate_root)
    _write_json(args.manifest_output, manifest)
    jobs = [
        (task, submission)
        for task in tasks
        for submission in task.submissions
    ]
    started = time.monotonic()
    records = []
    completed_count = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _replay_submission,
                task=task,
                submission=submission,
                launcher=args.launcher,
                runtime_root=args.runtime_root,
                scratch_root=args.scratch_root,
            ): (task.problem_id, task.suite_id, submission.submission_sha256)
            for task, submission in jobs
        }
        for future in as_completed(futures):
            records.append(future.result())
            with lock:
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == len(jobs):
                    print(
                        f"[{VERSION_LABEL}] "
                        f"completed={completed_count}/{len(jobs)}",
                        flush=True,
                    )
    elapsed = time.monotonic() - started
    records.sort(
        key=lambda row: (
            row["source_problem_id"],
            row["suite_id"],
            row["known_label"],
            row["submission_sha256"],
        )
    )
    _write_jsonl(args.replays_output, records)
    equivalence = build_checker_equivalence_audit(manifest, records)
    _write_json(args.equivalence_output, equivalence)
    gate_audit = build_v2_gate_audit(manifest, records, equivalence)
    _write_json(args.gate_audit_output, gate_audit)
    candidate_timings = [
        float(value)
        for row in records
        for value in row["execution"]["candidate_invocation_wall_seconds"]
    ]
    total_checker_seconds = sum(
        row["execution"]["checker_wall_seconds"] for row in records
    )
    total_tests = len(candidate_timings)
    run_audit = {
        "schema_version": RUN_AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": gate_audit["status"],
        "task_count": len(V2_TASKS),
        "problem_suite_count": len(tasks),
        "submission_suite_replay_count": len(records),
        "executed_test_count": total_tests,
        "wall_seconds": elapsed,
        "aggregate_candidate_wall_seconds": sum(candidate_timings),
        "aggregate_checker_wall_seconds": total_checker_seconds,
        "candidate_latency_seconds": {
            "median": percentile(candidate_timings, 0.5),
            "p95": percentile(candidate_timings, 0.95),
        },
        "effective_tests_per_wall_second": total_tests / elapsed,
        "launcher_binary_sha256": launcher_sha,
        "runtime_identity": asdict(runtime_identity),
        "manifest_sha256": _sha256_path(args.manifest_output),
        "replays_sha256": _sha256_path(args.replays_output),
        "checker_equivalence_audit_sha256": _sha256_path(
            args.equivalence_output
        ),
    }
    run_audit[GATE_AUDIT_HASH_FIELD] = _sha256_path(args.gate_audit_output)
    _write_json(args.run_audit_output, run_audit)
    print(
        f"[{VERSION_LABEL}] status={gate_audit['status']} "
        f"replays={len(records)} tests={total_tests} "
        f"wall_seconds={elapsed:.3f} violations={len(gate_audit['violations'])}",
        flush=True,
    )
    raise SystemExit(0 if gate_audit["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
