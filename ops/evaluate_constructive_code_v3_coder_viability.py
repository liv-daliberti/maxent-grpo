#!/usr/bin/env python3
"""Run the frozen development-only ConstructiveCode v3 Coder viability gate."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence


DEVELOPMENT_PROBLEMS = ("359_B", "988_A", "1283_C", "1399_D")
EVALUATION_PROBLEMS = ("361_B", "1294_C", "1408_A", "149_C")
OVERLAY_SUITE = "codecontests_o_corner_cases_v2"
PLUS_SUITE = "codecontests_plus_5x_v2"
SYSTEM_MESSAGE = (
    "Write a complete Python 3 program that solves the problem. Return only "
    "the program source, without Markdown fences or explanation."
)
RECEIPT_SCHEMA = "constructive-code-v3-coder-05b-viability-v1"
EXPECTED_SEED = 77101


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tree_sha256(root: Path, names: set[str] | None = None) -> str:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if names is not None and path.name not in names:
            continue
        records.append((relative, _sha256_file(path.resolve())))
    if not records:
        raise ValueError(f"tree hash selected no files under {root}")
    return _canonical_sha256(records)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _prompt(statement: str) -> str:
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("ConstructiveCode statement must be nonempty public text")
    return (
        "<|im_start|>system\n"
        + SYSTEM_MESSAGE
        + "<|im_end|>\n<|im_start|>user\n"
        + statement
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _strip_exact_surrounding_fence(text: str) -> tuple[str, bool]:
    """Strip only a complete bare or Python Markdown fence, with no exterior text."""

    for opening in ("```python\n", "```python3\n", "```\n"):
        if text.startswith(opening) and text.endswith("\n```"):
            return text[len(opening) : -4], True
    return text, False


def _request_seed(row_index: int, sample_index: int, base_seed: int) -> int:
    if row_index < 0 or sample_index < 0:
        raise ValueError("request indices must be nonnegative")
    return int(base_seed) + 10_000 * int(row_index) + int(sample_index)


def _validate_gate_and_choose_suites(
    gate: Mapping[str, Any], problem_keys: Mapping[str, str]
) -> dict[str, str]:
    if (
        gate.get("status") != "pass"
        or gate.get("expected_replay_count") != 4_800
        or gate.get("observed_replay_count") != 4_800
        or gate.get("violations") not in ([], None)
        or gate.get("checker_equivalence_violations") not in ([], None)
    ):
        raise ValueError("ConstructiveCode v3 executable gate is not an exact pass")
    task_results = gate.get("task_results")
    suite_results = gate.get("suite_results")
    if not isinstance(task_results, list) or len(task_results) != 12:
        raise ValueError("ConstructiveCode v3 gate lacks 12 task results")
    if not isinstance(suite_results, list):
        raise ValueError("ConstructiveCode v3 gate lacks suite results")
    task_status = {
        str(row.get("problem_key")): row.get("status")
        for row in task_results
        if isinstance(row, Mapping)
    }
    suite_status = {
        (str(row.get("problem_key")), str(row.get("suite_id"))): row.get(
            "status"
        )
        for row in suite_results
        if isinstance(row, Mapping)
    }
    selected = {}
    for problem_id in DEVELOPMENT_PROBLEMS:
        problem_key = problem_keys[problem_id]
        if task_status.get(problem_key) != "pass":
            raise ValueError(f"development task did not pass gate: {problem_id}")
        if suite_status.get((problem_key, OVERLAY_SUITE)) == "pass":
            selected[problem_id] = OVERLAY_SUITE
        elif suite_status.get((problem_key, PLUS_SUITE)) == "pass":
            selected[problem_id] = PLUS_SUITE
        else:
            raise ValueError(f"development task has no admitted suite: {problem_id}")
    return selected


def _replay_modules():
    import replay_constructive_code_v3 as replay_v3

    return replay_v3, replay_v3.base, replay_v3.materialize_v3


def _load_development_tasks(
    *,
    slate_root: Path,
    v1_root: Path,
    build_root: Path,
    gate: Mapping[str, Any],
):
    _replay_v3, base, materialize = _replay_modules()
    source_manifest = base._validate_source_manifest(slate_root)
    base._validate_v1_logical_root(source_manifest)
    if base.sha256_file(base.TESTLIB_ROOT / "testlib.h") != base.TESTLIB_SHA256:
        raise ValueError("pinned testlib.h hash drift")
    summaries = {
        str(row["source_problem_id"]): row
        for row in source_manifest["tasks"]
    }
    if set(summaries) != set(materialize.V3_TASKS):
        raise ValueError("ConstructiveCode v3 task manifest drift")
    problem_keys = {}
    records = {}
    for problem_id in DEVELOPMENT_PROBLEMS:
        task_dir = slate_root / str(summaries[problem_id]["relative_path"])
        record = base._load_json(task_dir / "task.json")
        records[problem_id] = (task_dir, record)
        problem_keys[problem_id] = str(record["problem_key"])
    selected_suites = _validate_gate_and_choose_suites(gate, problem_keys)

    registered = set(base.registered_task_adapters())
    tasks = []
    builds = []
    public_rows = []
    for problem_id in DEVELOPMENT_PROBLEMS:
        task_dir, record = records[problem_id]
        summary = summaries[problem_id]
        family, adapter = materialize.V3_TASKS[problem_id]
        unsigned = {
            key: value for key, value in record.items() if key != "task_record_sha256"
        }
        stored_sha = record.get("task_record_sha256")
        if (
            record.get("schema_version") != base.TASK_RECORD_SCHEMA
            or record.get("source_problem_id") != problem_id
            or record.get("witness_family") != family
            or record.get("task_adapter") != adapter
            or stored_sha != base._canonical_sha256(unsigned)
            or stored_sha != summary.get("task_record_sha256")
            or (problem_id, adapter) not in registered
            or materialize.V3_SPLIT_ASSIGNMENT.get(problem_id) != "development"
        ):
            raise ValueError(f"v3 development task identity drift: {problem_id}")
        language = record.get("language_contract")
        if not isinstance(language, Mapping) or (
            language.get("accepted_labels") != sorted(base.PYTHON3_LABELS)
            or language.get("runtime") != "Python 3.10.20"
        ):
            raise ValueError(f"v3 language contract drift: {problem_id}")
        excluded = base._v1_hashes(
            v1_root / str(summary["relative_path"]), problem_id
        )
        ledger_count_field = getattr(
            base,
            "V1_LEDGER_COUNT_FIELD",
            "excluded_v1_hash_count",
        )
        if language.get(ledger_count_field) != len(excluded):
            raise ValueError(f"v3 exclusion ledger drift: {problem_id}")
        replay_path = task_dir / "py3_replays.jsonl"
        if base._sha256_path(replay_path) != record.get("replays", {}).get(
            "jsonl_sha256"
        ):
            raise ValueError(f"v3 replay ledger drift: {problem_id}")
        base._validate_submission_records(base._load_jsonl(replay_path), excluded)

        checker_sha = str(record.get("checker_sha256") or "")
        checker_binary = build_root / problem_id.lower() / "checker"
        build = base._compile_checker(
            task_dir / "checker.cpp", checker_binary, checker_sha
        )
        build["source_problem_id"] = problem_id
        builds.append(build)
        raw_limits = record.get("limits")
        if not isinstance(raw_limits, Mapping):
            raise ValueError(f"v3 limits missing: {problem_id}")
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
            raise ValueError(f"v3 limits malformed: {problem_id}")
        cpu_seconds = max(1, math.ceil(time_ms * base.PYTHON_CPU_MULTIPLIER / 1000))
        limits = base.SandboxLimits(
            cpu_seconds=cpu_seconds,
            wall_seconds=float(cpu_seconds + 2),
            memory_bytes=memory_mb * 1024 * 1024,
            output_bytes=base.OUTPUT_LIMIT_BYTES,
            file_count=32,
            source_bytes=256 * 1024,
        )
        suites = record.get("suites")
        suite_id = selected_suites[problem_id]
        if not isinstance(suites, Mapping) or set(suites) != base.REQUIRED_SUITE_IDS:
            raise ValueError(f"v3 suite metadata drift: {problem_id}")
        suite = suites[suite_id]
        tasks.append(
            base.Task(
                problem_id=problem_id,
                problem_key=str(record["problem_key"]),
                adapter_id=adapter,
                witness_family=family,
                suite_id=suite_id,
                suite_sha256=str(suite["suite_sha256"]),
                checker_sha256=checker_sha,
                checker_binary=checker_binary,
                tests=base._load_tests(task_dir / base.SUITE_FILES[suite_id], suite),
                submissions=(),
                limits=limits,
            )
        )
        statement = record.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"v3 public statement missing: {problem_id}")
        public_rows.append(
            {
                "source_problem_id": problem_id,
                "problem_key": str(record["problem_key"]),
                "statement": statement,
                "statement_sha256": hashlib.sha256(
                    statement.encode("utf-8")
                ).hexdigest(),
                "suite_id": suite_id,
                "suite_sha256": str(suite["suite_sha256"]),
            }
        )
    return tasks, public_rows, builds, source_manifest


def _hard_replay_violations(replay: Mapping[str, Any]) -> list[str]:
    violations = []
    execution = replay.get("execution")
    if not isinstance(execution, Mapping):
        return ["missing execution record"]
    timings = execution.get("candidate_invocation_wall_seconds")
    checker_seconds = execution.get("checker_wall_seconds")
    if (
        not isinstance(timings, list)
        or not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in timings)
        or not isinstance(checker_seconds, (int, float))
        or not math.isfinite(checker_seconds)
        or checker_seconds < 0
    ):
        violations.append("nonfinite execution latency")
    if replay.get("released_checker_accepted") != replay.get("wrapper_accepted"):
        violations.append("checker-wrapper disagreement")
    failure = execution.get("first_failure")
    if isinstance(failure, Mapping):
        stage = failure.get("stage")
        if stage == "candidate" and any(
            bool(failure.get(key))
            for key in ("timed_out", "output_limited", "sandbox_violation")
        ):
            violations.append("candidate execution-bound violation")
        elif stage == "released_checker" and bool(failure.get("timed_out")):
            violations.append("released checker timeout")
        elif stage == "task_adapter":
            violations.append("task canonicalizer rejected checker acceptance")
    if replay.get("released_checker_accepted") is True and replay.get(
        "behavior_key"
    ) is None:
        violations.append("accepted candidate lacks behavior key")
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--gate-identity", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--prefix-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=77101)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--execution-workers", type=int, default=16)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count != 64 or args.prefix_count != 16:
        raise ValueError("frozen ConstructiveCode viability requires 64/16 samples")
    if args.seed != EXPECTED_SEED or args.max_tokens != 1024:
        raise ValueError("frozen ConstructiveCode seed or token budget drift")
    if args.temperature != 1.0 or args.top_p != 1.0:
        raise ValueError("frozen ConstructiveCode sampling distribution drift")
    if args.output.exists():
        raise FileExistsError(f"fresh viability receipt required: {args.output}")
    for path in (
        args.slate_root,
        args.v1_root,
        args.model / "config.json",
        args.gate_audit,
        args.gate_identity,
        args.protocol,
        args.identity,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    import vllm

    _replay_v3, base, _materialize = _replay_modules()
    gate = json.loads(args.gate_audit.read_text(encoding="utf-8"))
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    launcher_sha = base.build_launcher(base.SANDBOX_SOURCE, args.launcher)
    runtime_identity = base.prepare_runtime(args.image, args.runtime_root)
    tasks, public_rows, checker_builds, source_manifest = _load_development_tasks(
        slate_root=args.slate_root,
        v1_root=args.v1_root,
        build_root=args.build_root,
        gate=gate,
    )
    if tuple(row["source_problem_id"] for row in public_rows) != DEVELOPMENT_PROBLEMS:
        raise RuntimeError("development row order drift")

    llm = vllm.LLM(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=0.82,
        swap_space=16.0,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    requests = []
    for row_index, row in enumerate(public_rows):
        prompt = _prompt(row["statement"])
        for sample_index in range(args.sample_count):
            seed = _request_seed(row_index, sample_index, args.seed)
            requests.append(
                {
                    "row_index": row_index,
                    "sample_index": sample_index,
                    "request_seed": seed,
                    "prompt": prompt,
                    "params": vllm.SamplingParams(
                        n=1,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=1024,
                        seed=seed,
                    ),
                }
            )
    generated = []
    for start in range(0, len(requests), args.generation_batch_size):
        batch = requests[start : start + args.generation_batch_size]
        outputs = llm.generate(
            [row["prompt"] for row in batch],
            [row["params"] for row in batch],
            use_tqdm=False,
        )
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM returned the wrong request count")
        for request, output in zip(batch, outputs):
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM returned other than one candidate")
            sample = output.outputs[0]
            emitted = str(sample.text)
            code, stripped = _strip_exact_surrounding_fence(emitted)
            generated.append(
                {
                    **{key: request[key] for key in ("row_index", "sample_index", "request_seed")},
                    "emitted_text_sha256": hashlib.sha256(
                        emitted.encode("utf-8")
                    ).hexdigest(),
                    "executed_source_sha256": hashlib.sha256(
                        code.encode("utf-8")
                    ).hexdigest(),
                    "fence_stripped": stripped,
                    "token_count": len(sample.token_ids),
                    "finish_reason": str(sample.finish_reason),
                    "code": code,
                }
            )
    expected_requests = len(DEVELOPMENT_PROBLEMS) * args.sample_count
    if len(generated) != expected_requests:
        raise RuntimeError(
            f"not all {expected_requests} frozen requests produced a candidate"
        )

    started = time.monotonic()
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    exceptions: dict[tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=args.execution_workers) as executor:
        futures = {}
        for candidate in generated:
            key = (candidate["row_index"], candidate["sample_index"])
            task = tasks[candidate["row_index"]]
            submission = base.Submission(
                candidate["code"],
                "model_sample",
                candidate["executed_source_sha256"],
            )
            futures[
                executor.submit(
                    base._replay_submission,
                    task=task,
                    submission=submission,
                    launcher=args.launcher,
                    runtime_root=args.runtime_root,
                    scratch_root=args.scratch_root,
                )
            ] = key
        for count, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                completed[key] = future.result()
            except BaseException as error:
                exceptions[key] = f"{type(error).__name__}: {error}"
            if count % 32 == 0 or count == len(futures):
                print(
                    f"[constructive-coder-viability] "
                    f"executed={count}/{expected_requests}",
                    flush=True,
                )
    execution_wall_seconds = time.monotonic() - started

    hard_violations = []
    attempts = []
    prompt_results = []
    candidate_timings = []
    for candidate in generated:
        key = (candidate["row_index"], candidate["sample_index"])
        replay = completed.get(key)
        violations = []
        if replay is None:
            violations.append("worker exception: " + exceptions.get(key, "missing result"))
        else:
            violations.extend(_hard_replay_violations(replay))
            candidate_timings.extend(
                float(value)
                for value in replay["execution"]["candidate_invocation_wall_seconds"]
            )
        hard_violations.extend(
            f"r{key[0]}-s{key[1]}: {violation}" for violation in violations
        )
        accepted = bool(
            replay is not None
            and replay.get("released_checker_accepted") is True
            and replay.get("wrapper_accepted") is True
            and replay.get("behavior_key") is not None
        )
        attempts.append(
            {
                **{
                    key_name: candidate[key_name]
                    for key_name in (
                        "row_index",
                        "sample_index",
                        "request_seed",
                        "emitted_text_sha256",
                        "executed_source_sha256",
                        "fence_stripped",
                        "token_count",
                        "finish_reason",
                    )
                },
                "source_problem_id": public_rows[candidate["row_index"]][
                    "source_problem_id"
                ],
                "suite_id": public_rows[candidate["row_index"]]["suite_id"],
                "terminal_worker_record": replay is not None,
                "released_checker_accepted": None
                if replay is None
                else replay["released_checker_accepted"],
                "wrapper_accepted": None
                if replay is None
                else replay["wrapper_accepted"],
                "accepted": accepted,
                "canonical_key": None if replay is None else replay["behavior_key"],
                "execution": None if replay is None else replay["execution"],
                "hard_violations": violations,
            }
        )
    for row_index, row in enumerate(public_rows):
        rows = [attempt for attempt in attempts if attempt["row_index"] == row_index]
        keys = [attempt["canonical_key"] for attempt in rows if attempt["accepted"]]
        prefix = [
            attempt for attempt in rows[: args.prefix_count] if attempt["accepted"]
        ]
        prompt_results.append(
            {
                "row_index": row_index,
                "source_problem_id": row["source_problem_id"],
                "problem_key": row["problem_key"],
                "statement_sha256": row["statement_sha256"],
                "suite_id": row["suite_id"],
                "suite_sha256": row["suite_sha256"],
                "accepted_in_prefix": len(prefix),
                "accepted_in_full_sample": len(keys),
                "distinct_keys_in_full_sample": len(set(keys)),
                "canonical_key_counts": dict(sorted(Counter(keys).items())),
            }
        )
    prefix_success_tasks = sum(row["accepted_in_prefix"] > 0 for row in prompt_results)
    multimode_tasks = sum(
        row["distinct_keys_in_full_sample"] >= 2 for row in prompt_results
    )
    terminal_count = sum(attempt["terminal_worker_record"] for attempt in attempts)
    passed = (
        len(attempts) == expected_requests
        and terminal_count == expected_requests
        and not hard_violations
        and prefix_success_tasks >= 1
        and multimode_tasks >= 1
    )
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
    gate_identity = json.loads(args.gate_identity.read_text(encoding="utf-8"))
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "eligible_for_paired_online_training_smoke"
            if passed
            else "stopped_before_online_training"
        ),
        "job_id": int(args.job_id),
        "model": str(args.model.resolve()),
        "model_tree_sha256": _tree_sha256(args.model),
        "tokenizer_tree_sha256": _tree_sha256(args.model, tokenizer_names),
        "model_config_sha256": _sha256_file(args.model / "config.json"),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": _sha256_file(args.protocol),
        "identity_sha256": _sha256_file(args.identity),
        "gate_audit_sha256": _sha256_file(args.gate_audit),
        "gate_identity_sha256": _sha256_file(args.gate_identity),
        "gate_source_hash": gate_identity.get("source_hash"),
        "gate_execution_hash": gate_identity.get("execution_hash"),
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "checker_builds": checker_builds,
        "runtime": {
            "launcher_binary_sha256": launcher_sha,
            "runtime_identity": asdict(runtime_identity),
        },
        "sampling": {
            "seed": EXPECTED_SEED,
            "request_seed_formula": f"{EXPECTED_SEED} + 10000 * row_index + sample_index",
            "sample_count_per_task": 64,
            "prefix_count": 16,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 1024,
            "max_model_len": args.max_model_len,
            "system_message_sha256": hashlib.sha256(
                SYSTEM_MESSAGE.encode("utf-8")
            ).hexdigest(),
        },
        "criteria": {
            "minimum_prefix_success_tasks": 1,
            "minimum_multimode_tasks": 1,
        },
        "summary": {
            "request_count": len(attempts),
            "terminal_worker_records": terminal_count,
            "accepted_candidates": sum(attempt["accepted"] for attempt in attempts),
            "prefix_success_tasks": prefix_success_tasks,
            "multimode_tasks": multimode_tasks,
            "hard_violation_count": len(hard_violations),
            "execution_wall_seconds": execution_wall_seconds,
            "candidate_invocation_count": len(candidate_timings),
            "candidate_latency_median_seconds": None
            if not candidate_timings
            else base.percentile(candidate_timings, 0.5),
            "candidate_latency_p95_seconds": None
            if not candidate_timings
            else base.percentile(candidate_timings, 0.95),
        },
        "prompt_results": prompt_results,
        "attempts": attempts,
        "hard_violations": hard_violations,
        "information_boundary": {
            "loaded_problem_ids": list(DEVELOPMENT_PROBLEMS),
            "evaluation_problem_ids": list(EVALUATION_PROBLEMS),
            "evaluation_rows_loaded": False,
            "reference_programs_in_context": False,
            "canonical_keys_in_context": False,
            "checker_source_in_context": False,
            "test_inputs_in_context": False,
            "gate_outcomes_in_context": False,
        },
    }
    _atomic_json(args.output, payload)
    print(
        "[constructive-coder-viability] "
        f"status={payload['status']} accepted={payload['summary']['accepted_candidates']}/{expected_requests} "
        f"prefix_tasks={prefix_success_tasks}/{len(DEVELOPMENT_PROBLEMS)} "
        f"multimode_tasks={multimode_tasks}/{len(DEVELOPMENT_PROBLEMS)} "
        f"hard_violations={len(hard_violations)} output={args.output}",
        flush=True,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
