#!/usr/bin/env python3
"""Replay the frozen ConstructiveCode slate in isolation against released checkers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_constructive_code_checker_equivalence import (  # noqa: E402
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
    build_checker_equivalence_audit,
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
    prepare_runtime,
    run_candidate,
    sha256_file,
)


DEFAULT_SLATE = ROOT / "var/data/constructive_code_review_slate_v1"
DEFAULT_PLUS_SUITES = ROOT / "var/data/constructive_code_plus_5x_suites_v1"
DEFAULT_IMAGE = ROOT / "var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
DEFAULT_REPLAYS = ROOT / "var/artifacts/constructive_code_checker_replays.jsonl"
DEFAULT_MANIFEST = ROOT / "var/artifacts/constructive_code_admission_manifest.json"
DEFAULT_AUDIT = ROOT / "var/artifacts/constructive_code_checker_equivalence_audit.json"
DEFAULT_RUN_AUDIT = ROOT / "var/artifacts/constructive_code_replay_run_audit.json"
TESTLIB_REVISION = "1e4e8a24c79c6bad3becbdb5a332ffc352b7d5dd"
TESTLIB_SHA256 = "bb323e3c89285214966076e0d23d5a295c5f6126da7ff198c1276ddb95ecb1a0"
PYTHON_CPU_MULTIPLIER = 3
OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024
CHECKER_WALL_SECONDS = 2


@dataclass(frozen=True)
class TestCase:
    test_index: int
    stdin: bytes
    input_sha256: str


@dataclass(frozen=True)
class Submission:
    code: str
    known_label: str
    submission_sha256: str


@dataclass(frozen=True)
class Task:
    problem_id: str
    problem_key: str
    adapter_id: str
    witness_family: str
    suite_id: str
    suite_sha256: str
    checker_sha256: str
    checker_binary: Path
    tests: tuple[TestCase, ...]
    submissions: tuple[Submission, ...]
    limits: SandboxLimits


@dataclass(frozen=True)
class CheckerResult:
    accepted: bool
    returncode: int
    timed_out: bool
    message: str
    wall_seconds: float


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
                raise ValueError(f"{path}:{line_number} must be an object")
            records.append(record)
    return records


def _load_tests(
    path: Path,
    suite: Mapping[str, Any],
) -> tuple[TestCase, ...]:
    if _sha256_path(path) != suite["compressed_jsonl_sha256"]:
        raise ValueError(f"compressed suite hash mismatch: {path}")
    tests: list[TestCase] = []
    identity: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            record = json.loads(line)
            index = len(tests)
            stdin = str(record["stdin"]).encode("utf-8")
            digest = sha256_bytes(stdin)
            if (
                record["test_index"] != index
                or record["input_bytes"] != len(stdin)
                or record["input_sha256"] != digest
            ):
                raise ValueError(f"suite test identity mismatch: {path}:{index}")
            tests.append(TestCase(index, stdin, digest))
            identity.append(
                {
                    "input_bytes": len(stdin),
                    "input_sha256": digest,
                    "test_index": index,
                }
            )
    if len(tests) != suite["test_count"]:
        raise ValueError(f"suite test count mismatch: {path}")
    if sha256_bytes(_canonical_json_bytes(identity)) != suite["suite_sha256"]:
        raise ValueError(f"suite identity hash mismatch: {path}")
    return tuple(tests)


def _select_submissions(
    path: Path,
    task: Mapping[str, Any],
    correct_limit: int,
    incorrect_limit: int,
) -> tuple[Submission, ...]:
    if _sha256_path(path) != task["replays"]["python_replays_jsonl_sha256"]:
        raise ValueError(f"replay JSONL hash mismatch: {path}")
    by_label: dict[str, list[Submission]] = {"correct": [], "incorrect": []}
    for record in _load_jsonl(path):
        if record.get("language") != "py3":
            continue
        code = str(record.get("code") or "")
        digest = sha256_bytes(code.encode("utf-8"))
        label = str(record.get("known_label") or "")
        if digest != record.get("submission_sha256") or label not in by_label:
            raise ValueError(f"malformed replay record: {path}")
        by_label[label].append(Submission(code, label, digest))
    selected = [
        *by_label["correct"][:correct_limit],
        *by_label["incorrect"][:incorrect_limit],
    ]
    if len(by_label["correct"][:correct_limit]) < correct_limit:
        raise ValueError(f"too few py3 correct replays: {path}")
    if len(by_label["incorrect"][:incorrect_limit]) < incorrect_limit:
        raise ValueError(f"too few py3 incorrect replays: {path}")
    return tuple(selected)


def _compile_checker(source: Path, output: Path) -> dict[str, Any]:
    expected = _load_json(source.parent / "task.json")["source_hashes"][
        "checker_cpp_sha256"
    ]
    if sha256_file(source) != expected:
        raise ValueError(f"released checker source hash mismatch: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        shutil.which("g++") or "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        "-I",
        str(ROOT / "third_party/testlib"),
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True, timeout=120)
    output.chmod(0o500)
    return {
        "source_sha256": expected,
        "binary_sha256": sha256_file(output),
        "compile_command": command,
    }


def _load_tasks(
    *,
    slate_root: Path,
    plus_suites_root: Path,
    suite_kind: str,
    build_root: Path,
    correct_limit: int,
    incorrect_limit: int,
    task_filter: frozenset[str],
) -> tuple[list[Task], list[dict[str, Any]]]:
    slate_manifest = _load_json(slate_root / "manifest.json")
    plus_by_problem: dict[str, dict[str, Any]] = {}
    if suite_kind == "plus_5x":
        plus_manifest = _load_json(plus_suites_root / "manifest.json")
        if plus_manifest.get("schema_version") != "constructive-code-plus-5x-suites-v1":
            raise ValueError("unsupported Plus 5x suite manifest")
        plus_by_problem = {
            str(item["source_problem_id"]): item
            for item in plus_manifest["tasks"]
        }
    tasks: list[Task] = []
    builds: list[dict[str, Any]] = []
    registered = set(registered_task_adapters())
    for summary in slate_manifest["tasks"]:
        problem_id = summary["source_problem_id"]
        if task_filter and problem_id not in task_filter:
            continue
        task_dir = slate_root / summary["relative_path"]
        record = _load_json(task_dir / "task.json")
        adapter_key = (problem_id, record["task_adapter"])
        if adapter_key not in registered:
            raise ValueError(f"task adapter is not registered: {adapter_key}")
        checker_binary = build_root / problem_id.lower() / "checker"
        build = _compile_checker(task_dir / "checker.cpp", checker_binary)
        build["problem_id"] = problem_id
        builds.append(build)
        if suite_kind == "overlay":
            suite = record["overlay_suite"]
            suite_path = task_dir / "overlay_inputs.jsonl.gz"
        else:
            try:
                suite = plus_by_problem[problem_id]
            except KeyError as error:
                raise ValueError(
                    f"Plus 5x suite is missing task {problem_id}"
                ) from error
            if suite["problem_key"] != record["problem_key"]:
                raise ValueError(f"Plus problem identity mismatch: {problem_id}")
            if suite["checker_sha256"] != record["source_hashes"][
                "checker_cpp_sha256"
            ]:
                raise ValueError(f"Plus checker identity mismatch: {problem_id}")
            suite_path = (
                plus_suites_root
                / suite["relative_path"]
                / "plus_5x_inputs.jsonl.gz"
            )
        raw_limits = record["limits"]
        cpu_seconds = max(
            1,
            math.ceil(
                raw_limits["time_milliseconds"]
                * PYTHON_CPU_MULTIPLIER
                / 1000
            ),
        )
        tasks.append(
            Task(
                problem_id=problem_id,
                problem_key=record["problem_key"],
                adapter_id=record["task_adapter"],
                witness_family=record["witness_family"],
                suite_id=suite["suite_id"],
                suite_sha256=suite["suite_sha256"],
                checker_sha256=record["source_hashes"]["checker_cpp_sha256"],
                checker_binary=checker_binary,
                tests=_load_tests(suite_path, suite),
                submissions=_select_submissions(
                    task_dir / "python_replays.jsonl",
                    record,
                    correct_limit,
                    incorrect_limit,
                ),
                limits=SandboxLimits(
                    cpu_seconds=cpu_seconds,
                    wall_seconds=float(cpu_seconds + 2),
                    memory_bytes=raw_limits["memory_megabytes"] * 1024 * 1024,
                    output_bytes=OUTPUT_LIMIT_BYTES,
                    file_count=32,
                    source_bytes=256 * 1024,
                ),
            )
        )
    if not tasks:
        raise ValueError("no tasks selected")
    return tasks, builds


def _run_checker(
    checker: Path,
    stdin: bytes,
    output: bytes,
    scratch_root: Path,
) -> CheckerResult:
    with tempfile.TemporaryDirectory(prefix="released-checker-", dir=scratch_root) as raw:
        directory = Path(raw)
        input_path = directory / "input"
        output_path = directory / "output"
        answer_path = directory / "answer"
        input_path.write_bytes(stdin)
        output_path.write_bytes(output)
        answer_path.write_bytes(b"")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [str(checker), str(input_path), str(output_path), str(answer_path)],
                check=False,
                text=True,
                capture_output=True,
                timeout=CHECKER_WALL_SECONDS,
            )
            timed_out = False
            returncode = completed.returncode
            message = (completed.stdout + completed.stderr)[-2000:]
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = -1
            message = str(error)[-2000:]
        elapsed = time.monotonic() - started
    return CheckerResult(
        accepted=returncode == 0 and not timed_out,
        returncode=returncode,
        timed_out=timed_out,
        message=message,
        wall_seconds=elapsed,
    )


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
    candidate_wall_seconds = 0.0
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
        candidate_wall_seconds += candidate.wall_seconds
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
        checker = _run_checker(
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
            first_failure = {
                "stage": "task_adapter",
                "test_index": test.test_index,
                "message": str(error),
            }
            continue
        observations.append(CheckedWitness(decision=decision, witness=witness))

    behavior_key: str | None = None
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
            "candidate_wall_seconds": candidate_wall_seconds,
            "checker_wall_seconds": checker_wall_seconds,
            "executed_tests": executed_tests,
            "suite_tests": len(task.tests),
            "first_failure": first_failure,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _admission_manifest(
    tasks: list[Task],
    correct_limit: int,
    incorrect_limit: int,
    threshold: float,
    builds: list[dict[str, Any]],
    replay_stage: str,
    suite_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": f"{replay_stage}_{suite_kind}_replay",
        "selection": {
            "language": "py3",
            "order": "exact code SHA-256 ascending within known label",
            "correct_per_task": correct_limit,
            "incorrect_per_task": incorrect_limit,
            "suite_kind": suite_kind,
        },
        "execution_limits": {
            "python_cpu_multiplier": PYTHON_CPU_MULTIPLIER,
            "output_bytes": OUTPUT_LIMIT_BYTES,
            "released_checker_wall_seconds": CHECKER_WALL_SECONDS,
        },
        "runtime": {
            "image_sha256": PINNED_IMAGE_SHA256,
            "launcher_source_sha256": sha256_file(
                ROOT / "ops/constructive_code_sandbox.c"
            ),
        },
        "checker_builds": builds,
        "testlib": {
            "revision": TESTLIB_REVISION,
            "testlib_h_sha256": TESTLIB_SHA256,
        },
        "tasks": [
            {
                "problem_key": task.problem_key,
                "source_problem_id": task.problem_id,
                "task_adapter": task.adapter_id,
                "witness_family": task.witness_family,
                "suites": [
                    {
                        "suite_id": task.suite_id,
                        "suite_sha256": task.suite_sha256,
                        "checker_sha256": task.checker_sha256,
                        "test_count": len(task.tests),
                        "required_correct_replays": correct_limit,
                        "required_incorrect_replays": incorrect_limit,
                        "verified_threshold": threshold,
                    }
                ],
            }
            for task in tasks
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate-root", type=Path, default=DEFAULT_SLATE)
    parser.add_argument(
        "--plus-suites-root",
        type=Path,
        default=DEFAULT_PLUS_SUITES,
    )
    parser.add_argument(
        "--suite-kind",
        choices=("overlay", "plus_5x"),
        default="overlay",
    )
    parser.add_argument(
        "--replay-stage",
        choices=("diagnostic", "admission"),
        default="diagnostic",
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--replays-output", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--run-audit-output", type=Path, default=DEFAULT_RUN_AUDIT)
    parser.add_argument("--correct-limit", type=int, default=5)
    parser.add_argument("--incorrect-limit", type=int, default=5)
    parser.add_argument("--verified-threshold", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--task", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.correct_limit < 1 or args.incorrect_limit < 1 or args.workers < 1:
        raise ValueError("replay limits and workers must be positive")
    if not 0.0 <= args.verified_threshold <= 1.0:
        raise ValueError("verified threshold must be in [0, 1]")
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    launcher_sha = build_launcher(ROOT / "ops/constructive_code_sandbox.c", args.launcher)
    runtime_identity = prepare_runtime(args.image, args.runtime_root)
    tasks, builds = _load_tasks(
        slate_root=args.slate_root,
        plus_suites_root=args.plus_suites_root,
        suite_kind=args.suite_kind,
        build_root=args.build_root,
        correct_limit=args.correct_limit,
        incorrect_limit=args.incorrect_limit,
        task_filter=frozenset(args.task),
    )
    manifest = _admission_manifest(
        tasks,
        args.correct_limit,
        args.incorrect_limit,
        args.verified_threshold,
        builds,
        args.replay_stage,
        args.suite_kind,
    )
    _write_json(args.manifest_output, manifest)

    jobs = [
        (task, submission)
        for task in tasks
        for submission in task.submissions
    ]
    started = time.monotonic()
    records: list[dict[str, Any]] = []
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
            ): (task.problem_id, submission.submission_sha256)
            for task, submission in jobs
        }
        for future in as_completed(futures):
            records.append(future.result())
            with lock:
                completed_count += 1
                if completed_count % 20 == 0 or completed_count == len(jobs):
                    print(
                        "[constructive-code-replay] "
                        f"completed={completed_count}/{len(jobs)}",
                        flush=True,
                    )
    elapsed = time.monotonic() - started
    records.sort(
        key=lambda row: (
            row["source_problem_id"],
            row["known_label"],
            row["submission_sha256"],
        )
    )
    _write_jsonl(args.replays_output, records)
    audit = build_checker_equivalence_audit(manifest, records)
    _write_json(args.audit_output, audit)
    total_candidate_seconds = sum(
        row["execution"]["candidate_wall_seconds"] for row in records
    )
    total_checker_seconds = sum(
        row["execution"]["checker_wall_seconds"] for row in records
    )
    total_tests = sum(row["execution"]["executed_tests"] for row in records)
    run_audit = {
        "schema_version": "constructive-code-replay-run-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": audit["status"],
        "task_count": len(tasks),
        "submission_count": len(records),
        "executed_test_count": total_tests,
        "wall_seconds": elapsed,
        "aggregate_candidate_wall_seconds": total_candidate_seconds,
        "aggregate_checker_wall_seconds": total_checker_seconds,
        "effective_tests_per_wall_second": total_tests / elapsed,
        "launcher_binary_sha256": launcher_sha,
        "runtime_identity": asdict(runtime_identity),
        "manifest_sha256": sha256_bytes(_canonical_json_bytes(manifest)),
        "replays_sha256": sha256_file(args.replays_output),
        "checker_equivalence_audit_sha256": sha256_file(args.audit_output),
    }
    _write_json(args.run_audit_output, run_audit)
    print(
        "[constructive-code-replay] "
        f"status={audit['status']} tasks={len(tasks)} submissions={len(records)} "
        f"tests={total_tests} wall_seconds={elapsed:.3f} "
        f"violations={len(audit['violations'])}",
        flush=True,
    )
    raise SystemExit(0 if audit["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
