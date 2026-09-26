#!/usr/bin/env python3
"""Execute the frozen four-task ConstructiveCode overlay replay gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OPS = ROOT / "ops"
for path in (SRC, OPS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    canonicalize_task_output,
)
from oat_drgrpo.constructive_code_sandbox import (  # noqa: E402
    SandboxLimits,
    build_launcher,
    percentile,
    prepare_runtime,
    run_candidate,
    sha256_file,
)


DEFAULT_SLATE = ROOT / "var/data/constructive_code_review_slate_v1"
DEFAULT_IMAGE = ROOT / "var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"
DEFAULT_TESTLIB = (
    ROOT
    / "var/source_data/testlib/1e4e8a24c79c6bad3becbdb5a332ffc352b7d5dd"
    / "testlib.h"
)
DEFAULT_OUTPUT = ROOT / "var/artifacts/constructive_code_overlay_replay_v1"
TESTLIB_SHA256 = "bb323e3c89285214966076e0d23d5a295c5f6126da7ff198c1276ddb95ecb1a0"
SUITE_ID = "codecontests_o_corner_cases_v1"
SELECTED = {
    "482_A": ("482_a", "fixed_integer_sequence_v1", 100, 100),
    "988_A": ("988_a", "status_integer_set_v1", 100, 100),
    "1153_B": ("1153_b", "matrix_assignment_v1", 100, 94),
    "149_C": ("149_c", "two_group_partition_v1", 100, 100),
}
LIMITS = SandboxLimits(
    cpu_seconds=2,
    wall_seconds=4.0,
    memory_bytes=512 * 1024 * 1024,
    output_bytes=1024 * 1024,
    file_count=32,
    source_bytes=256 * 1024,
)
CHECKER_TIMEOUT_SECONDS = 2.0
MAX_MEDIAN_EXECUTION_SECONDS = 0.5
MAX_P95_EXECUTION_SECONDS = 1.0


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} contains a non-object replay")
            rows.append(row)
    return rows


def _load_inputs(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                raw = row["stdin"].encode("utf-8")
                if sha256_bytes(raw) != row["input_sha256"]:
                    raise ValueError(f"{path} input hash mismatch")
                rows.append({**row, "input_bytes_raw": raw})
    if not rows:
        raise ValueError(f"{path} contains no tests")
    return rows


def _compile_checker(
    checker_source: Path,
    testlib: Path,
    output: Path,
) -> dict[str, Any]:
    source = checker_source.read_text(encoding="utf-8")
    if re.search(r"\bans\b", source):
        raise ValueError(f"{checker_source} reads the answer stream")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-pipe",
        "-I",
        str(testlib.parent),
        str(checker_source),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return {
        "source_sha256": sha256_file(checker_source),
        "binary_sha256": sha256_file(output),
        "compiler": subprocess.run(
            ["g++", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0],
        "command": command,
    }


def _run_checker(
    checker: Path,
    input_bytes: bytes,
    output_bytes: bytes,
    scratch_root: Path,
) -> tuple[bool, int, bool]:
    with tempfile.TemporaryDirectory(prefix="checker-", dir=scratch_root) as raw:
        directory = Path(raw)
        input_path = directory / "input"
        output_path = directory / "output"
        answer_path = directory / "answer"
        input_path.write_bytes(input_bytes)
        output_path.write_bytes(output_bytes)
        answer_path.write_bytes(b"")
        try:
            completed = subprocess.run(
                [
                    str(checker),
                    str(input_path),
                    str(output_path),
                    str(answer_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=CHECKER_TIMEOUT_SECONDS,
                check=False,
            )
            return completed.returncode == 0, completed.returncode, False
        except subprocess.TimeoutExpired:
            return False, -1, True


def _replay_one(
    *,
    replay: Mapping[str, Any],
    task: Mapping[str, Any],
    adapter: str,
    inputs: list[dict[str, Any]],
    checker: Path,
    checker_sha256: str,
    launcher: Path,
    runtime_root: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    observations: list[CheckedWitness] = []
    execution_seconds = []
    simulator_executions = 0
    released_all = True
    wrapper_all = True
    first_failure: str | None = None
    adapter_errors = []
    for test in inputs:
        result = run_candidate(
            launcher=launcher,
            runtime_root=runtime_root,
            source=str(replay["code"]),
            stdin=test["input_bytes_raw"],
            limits=LIMITS,
            scratch_root=scratch_root,
            runtime_is_preverified=True,
        )
        simulator_executions += 1
        execution_seconds.append(result.wall_seconds)
        if not result.completed_cleanly:
            released_all = False
            wrapper_all = False
            first_failure = (
                "candidate_timeout"
                if result.timed_out
                else "candidate_output_limit"
                if result.output_limited
                else "candidate_sandbox_violation"
                if result.sandbox_violation
                else "candidate_nonzero"
            )
            break
        accepted, exit_code, checker_timed_out = _run_checker(
            checker,
            test["input_bytes_raw"],
            result.stdout,
            scratch_root,
        )
        decision = ReleasedCheckerDecision(
            checker_sha256=checker_sha256,
            input_sha256=test["input_sha256"],
            output_sha256=sha256_bytes(result.stdout),
            accepted=accepted,
            exit_code=exit_code,
            timed_out=checker_timed_out,
        )
        if not accepted:
            released_all = False
            wrapper_all = False
            first_failure = (
                "checker_timeout" if checker_timed_out else "checker_rejected"
            )
            break
        try:
            witness = canonicalize_task_output(
                adapter,
                test["input_bytes_raw"],
                result.stdout,
                decision,
            )
            observations.append(CheckedWitness(decision=decision, witness=witness))
        except ConstructiveCodeError as error:
            wrapper_all = False
            adapter_errors.append(type(error).__name__ + ":" + str(error))

    behavior_key = None
    if released_all and wrapper_all and len(observations) == len(inputs):
        behavior_key = canonicalize_checked_behavior(
            str(task["problem_key"]),
            str(task["overlay_suite"]["suite_sha256"]),
            observations,
        ).canonical_key
    elif released_all and len(observations) != len(inputs):
        wrapper_all = False
        first_failure = first_failure or "adapter_rejected_checker_accepted_output"
    return {
        "schema_version": REPLAY_SCHEMA,
        "problem_key": task["problem_key"],
        "suite_id": SUITE_ID,
        "submission_sha256": replay["submission_sha256"],
        "checker_sha256": checker_sha256,
        "known_label": replay["known_label"],
        "released_checker_accepted": released_all,
        "wrapper_accepted": released_all and wrapper_all,
        "behavior_key": behavior_key,
        "diagnostics": {
            "tests_executed": simulator_executions,
            "candidate_wall_seconds": execution_seconds,
            "first_failure": first_failure,
            "adapter_errors": adapter_errors[:3],
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate-root", type=Path, default=DEFAULT_SLATE)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--testlib", type=Path, default=DEFAULT_TESTLIB)
    parser.add_argument("--launcher-source", type=Path, default=ROOT / "ops/constructive_code_sandbox.c")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if sha256_file(args.testlib) != TESTLIB_SHA256:
        raise ValueError("testlib.h hash mismatch")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    runtime_identity = prepare_runtime(args.image, args.runtime_root)
    launcher_sha256 = build_launcher(args.launcher_source, args.launcher)

    manifest_tasks = []
    jobs = []
    checker_receipts = {}
    for problem_id, (relative, adapter, required_correct, required_incorrect) in SELECTED.items():
        directory = args.slate_root / relative
        task = json.loads((directory / "task.json").read_text())
        if task["source_problem_id"] != problem_id or task["task_adapter"] != adapter:
            raise ValueError(f"{problem_id} task identity or adapter mismatch")
        inputs = _load_inputs(directory / "overlay_inputs.jsonl.gz")
        if len(inputs) != task["overlay_suite"]["test_count"]:
            raise ValueError(f"{problem_id} overlay test count mismatch")
        replays = _load_jsonl(directory / "python_replays.jsonl")
        counts = {
            label: sum(row["known_label"] == label for row in replays)
            for label in ("correct", "incorrect")
        }
        if counts != {"correct": required_correct, "incorrect": required_incorrect}:
            raise ValueError(f"{problem_id} replay counts differ from preregistration")
        checker = args.output_root / "checkers" / relative / "checker"
        checker_receipt = _compile_checker(
            directory / "checker.cpp", args.testlib, checker
        )
        if checker_receipt["source_sha256"] != task["source_hashes"]["checker_cpp_sha256"]:
            raise ValueError(f"{problem_id} checker source hash mismatch")
        checker_receipts[problem_id] = checker_receipt
        manifest_tasks.append(
            {
                "problem_key": task["problem_key"],
                "source_problem_id": problem_id,
                "witness_family": task["witness_family"],
                "task_adapter": adapter,
                "suites": [
                    {
                        "suite_id": SUITE_ID,
                        "checker_sha256": checker_receipt["source_sha256"],
                        "required_correct_replays": required_correct,
                        "required_incorrect_replays": required_incorrect,
                        "verified_threshold": 0.9,
                    }
                ],
            }
        )
        for replay in replays:
            jobs.append(
                {
                    "replay": replay,
                    "task": task,
                    "adapter": adapter,
                    "inputs": inputs,
                    "checker": checker,
                    "checker_sha256": checker_receipt["source_sha256"],
                }
            )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": SUITE_ID,
        "tasks": manifest_tasks,
    }
    _write_json(args.output_root / "admission_manifest.json", manifest)

    rows = []
    completed_by_task: dict[str, int] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _replay_one,
                **job,
                launcher=args.launcher,
                runtime_root=args.runtime_root,
                scratch_root=args.scratch_root,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            with lock:
                key = row["problem_key"]
                completed_by_task[key] = completed_by_task.get(key, 0) + 1
                count = completed_by_task[key]
                if count % 25 == 0:
                    print(
                        "[constructive-overlay] "
                        f"task={key} replays={count}",
                        flush=True,
                    )
    rows.sort(
        key=lambda row: (
            row["problem_key"],
            row["known_label"],
            row["submission_sha256"],
        )
    )
    _write_jsonl(args.output_root / "replays.jsonl", rows)
    equivalence = build_checker_equivalence_audit(manifest, rows)
    _write_json(args.output_root / "checker_equivalence_audit.json", equivalence)

    durations = [
        duration
        for row in rows
        for duration in row["diagnostics"]["candidate_wall_seconds"]
    ]
    median_seconds = statistics.median(durations)
    p95_seconds = percentile(durations, 0.95)
    throughput_pass = (
        median_seconds <= MAX_MEDIAN_EXECUTION_SECONDS
        and p95_seconds <= MAX_P95_EXECUTION_SECONDS
    )
    status = "pass" if equivalence["status"] == "pass" and throughput_pass else "fail"
    receipt = {
        "schema_version": "constructive-code-overlay-replay-receipt-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": (
            "overlay_executable_gate_passed_plus_5x_pending"
            if status == "pass"
            else "ineligible"
        ),
        "tasks": list(SELECTED),
        "replay_count": len(rows),
        "candidate_execution_count": len(durations),
        "elapsed_seconds": time.monotonic() - started,
        "equivalence_status": equivalence["status"],
        "throughput": {
            "median_candidate_seconds": median_seconds,
            "p95_candidate_seconds": p95_seconds,
            "maximum_median_seconds": MAX_MEDIAN_EXECUTION_SECONDS,
            "maximum_p95_seconds": MAX_P95_EXECUTION_SECONDS,
            "status": "pass" if throughput_pass else "fail",
        },
        "limits": asdict(LIMITS),
        "hashes": {
            "manifest_sha256": _canonical_sha256(manifest),
            "replays_file_sha256": sha256_file(args.output_root / "replays.jsonl"),
            "testlib_sha256": sha256_file(args.testlib),
            "image_sha256": runtime_identity.image_sha256,
            "launcher_source_sha256": sha256_file(args.launcher_source),
            "launcher_binary_sha256": launcher_sha256,
            "adapter_source_sha256": sha256_file(
                ROOT / "src/oat_drgrpo/constructive_code_adapters.py"
            ),
            "canonicalizer_source_sha256": sha256_file(
                ROOT / "src/oat_drgrpo/constructive_code.py"
            ),
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "checker_receipts": checker_receipts,
    }
    _write_json(args.output_root / "receipt.json", receipt)
    print(
        "[constructive-overlay] "
        f"status={status} replays={len(rows)} executions={len(durations)} "
        f"median={median_seconds:.3f}s p95={p95_seconds:.3f}s",
        flush=True,
    )
    raise SystemExit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()
