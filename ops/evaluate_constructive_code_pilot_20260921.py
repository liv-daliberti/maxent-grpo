#!/usr/bin/env python3
"""Frozen development-only Coder-7B capability pilot with durable raw receipts.

The imported v6/v3 modules must come from the frozen execution bundle. All
verification, canonicalization, task loading, and sandbox operations are reused
without a fallback. A passing receipt only establishes pilot capability.
"""

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
import time
from typing import Any, Mapping, Sequence

import evaluate_constructive_code_v3_coder_viability as base_eval
import evaluate_constructive_code_v6_coder_viability as v6


SCHEMA = "constructive-code-coder-7b-capability-pilot-20260921-v1"
DEVELOPMENT_PROBLEMS = ("359_B", "988_A", "1399_D")
EXPECTED_REQUESTS = 192
MINIMUM_PCMD_ACCEPTED = 30
MINIMUM_AGGREGATE_ACCURACY = 0.10
MINIMUM_MULTIMODE_TASKS = 2
PASS_K = (1, 8, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "slate-root", "v1-root", "image", "runtime-root", "launcher",
        "build-root", "scratch-root", "model", "gate-audit", "gate-identity",
        "protocol", "identity", "output", "prompt-overlay", "gate-replays",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name, default in (
        ("sample-count", 64), ("prefix-count", 16), ("seed", 77101),
        ("max-tokens", 1024), ("max-model-len", 8192),
        ("generation-batch-size", 16), ("execution-workers", 16),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    for name in ("source-hash", "execution-hash", "job-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--preflight", action="store_true",
        help="Validate frozen tasks, compile original checkers, and prepare sandbox; no model load or generation.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "sample_count": 64, "prefix_count": 16, "seed": 77101,
        "max_tokens": 1024, "temperature": 1.0, "top_p": 1.0,
        "max_model_len": 8192, "generation_batch_size": 16,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise ValueError(f"frozen pilot {name} must equal {expected}")
    if args.execution_workers < 1 or args.execution_workers > 16:
        raise ValueError("execution workers must be in [1,16]")
    for name in ("source_hash", "execution_hash"):
        value = getattr(args, name)
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _receipt_paths(output: Path) -> dict[str, Path]:
    return {
        "responses": output.with_name(output.stem + ".responses.jsonl"),
        "attempts": output.with_name(output.stem + ".attempts.jsonl"),
        "progress": output.with_name(output.stem + ".progress.json"),
    }


def _append_jsonl(handle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _pass_at_k(n: int, correct: int, k: int) -> float:
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("invalid pass@k counts")
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def _mode_metrics(keys: Sequence[str], sample_count: int) -> dict[str, Any]:
    if len(keys) > sample_count or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("invalid accepted canonical keys")
    counts = Counter(keys)
    accepted = len(keys)
    eligible = accepted >= MINIMUM_PCMD_ACCEPTED
    collision_numerator = sum(count * (count - 1) for count in counts.values())
    return {
        "sample_count": sample_count,
        "accepted": accepted,
        "distinct_valid_modes": len(counts),
        "canonical_key_counts": dict(sorted(counts.items())),
        **{f"pass_at_{k}": _pass_at_k(sample_count, accepted, k) for k in PASS_K},
        "pcmd_eligible": eligible,
        "pcmd_accepted_threshold": MINIMUM_PCMD_ACCEPTED,
        "pcmd": 1.0 - collision_numerator / (accepted * (accepted - 1)) if eligible else None,
    }


def _summarize(attempts: Sequence[Mapping[str, Any]], public_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(r, s) for r in range(len(DEVELOPMENT_PROBLEMS)) for s in range(64)}
    observed = [(row["row_index"], row["sample_index"]) for row in attempts]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("final receipt requires exactly the 192 frozen request identities")
    if tuple(row["source_problem_id"] for row in public_rows) != DEVELOPMENT_PROBLEMS:
        raise ValueError("pilot development row order drift")
    prompts = []
    hard_violations = []
    for row_index, public in enumerate(public_rows):
        rows = sorted((a for a in attempts if a["row_index"] == row_index), key=lambda a: a["sample_index"])
        keys = [a["canonical_key"] for a in rows if a["accepted"]]
        prompts.append({
            **{name: public[name] for name in ("source_problem_id", "problem_key", "statement_sha256", "suite_id", "suite_sha256")},
            "row_index": row_index,
            "accepted_in_prefix_16": sum(a["accepted"] for a in rows[:16]),
            **_mode_metrics(keys, len(rows)),
        })
        for attempt in rows:
            hard_violations.extend(f"r{row_index}-s{attempt['sample_index']}: {v}" for v in attempt["hard_violations"])
    accepted = sum(row["accepted"] for row in prompts)
    terminal = sum(bool(row["terminal_worker_record"]) for row in attempts)
    accuracy = accepted / EXPECTED_REQUESTS
    multimode = sum(row["distinct_valid_modes"] >= 2 for row in prompts)
    audit_passed = terminal == EXPECTED_REQUESTS and not hard_violations
    capability_passed = accuracy >= MINIMUM_AGGREGATE_ACCURACY and multimode >= MINIMUM_MULTIMODE_TASKS
    eligible = [row["pcmd"] for row in prompts if row["pcmd_eligible"]]
    status = "pass" if audit_passed and capability_passed else "audit_fail" if not audit_passed else "capability_fail"
    return {
        "status": status,
        "decision": "capability_gate_passed_main_study_not_authorized" if status == "pass" else "stopped_before_training",
        "audit_passed": audit_passed,
        "capability_passed": capability_passed,
        "summary": {
            "request_count": EXPECTED_REQUESTS,
            "terminal_worker_records": terminal,
            "accepted_candidates": accepted,
            "aggregate_accuracy": accuracy,
            "multimode_tasks": multimode,
            "hard_violation_count": len(hard_violations),
            "stability_rechecks": sum(bool(row.get("stability_recheck_required")) for row in attempts),
            **{f"macro_pass_at_{k}": sum(row[f"pass_at_{k}"] for row in prompts) / len(prompts) for k in PASS_K},
            "pcmd_eligible_tasks": len(eligible),
            "pcmd_total_tasks": len(prompts),
            "macro_pcmd_over_eligible_tasks": sum(eligible) / len(eligible) if eligible else None,
        },
        "prompt_results": prompts,
        "hard_violations": hard_violations,
    }


def _attempt(candidate: Mapping[str, Any], task: Any, replay: Mapping[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    violations = ["worker exception: " + (error or "missing result")] if replay is None else base_eval._hard_replay_violations(replay)
    if replay is not None:
        for name, expected in (
            ("source_problem_id", task.problem_id), ("problem_key", task.problem_key),
            ("suite_id", task.suite_id), ("suite_sha256", task.suite_sha256),
            ("checker_sha256", task.checker_sha256),
            ("submission_sha256", candidate["executed_source_sha256"]),
        ):
            if replay.get(name) != expected:
                violations.append(f"replay identity mismatch: {name}")
        if replay.get("released_checker_accepted") is True:
            execution = replay.get("execution", {})
            if execution.get("executed_tests") != len(task.tests) or execution.get("suite_tests") != len(task.tests):
                violations.append("accepted replay did not run the complete frozen suite")
        key = replay.get("behavior_key")
        if key is not None and (not isinstance(key, str) or not key):
            violations.append("invalid behavior key")
    accepted = bool(replay is not None and not violations and replay.get("released_checker_accepted") is True and replay.get("wrapper_accepted") is True and replay.get("behavior_key") is not None)
    return {
        **{name: candidate[name] for name in ("row_index", "sample_index", "request_seed", "emitted_text_sha256", "executed_source_sha256", "fence_stripped", "token_count", "finish_reason")},
        "source_problem_id": task.problem_id,
        "suite_id": task.suite_id,
        "suite_sha256": task.suite_sha256,
        "checker_sha256": task.checker_sha256,
        "terminal_worker_record": replay is not None,
        "accepted": accepted,
        "canonical_key": replay.get("behavior_key") if accepted else None,
        "released_checker_accepted": replay.get("released_checker_accepted") if replay is not None else None,
        "wrapper_accepted": replay.get("wrapper_accepted") if replay is not None else None,
        "execution": replay.get("execution") if replay is not None else None,
        "hard_violations": violations,
        "replay": replay,
    }


def _verify_candidate(base: Any, args: argparse.Namespace, candidate: Mapping[str, Any], task: Any) -> dict[str, Any]:
    submission = base.Submission(candidate["code"], "model_sample", candidate["executed_source_sha256"])
    def replay_once() -> Mapping[str, Any]:
        return base._replay_submission(task=task, submission=submission, launcher=args.launcher, runtime_root=args.runtime_root, scratch_root=args.scratch_root)
    attempt = _attempt(candidate, task, replay_once())
    attempt["stability_recheck_required"] = attempt["accepted"]
    attempt["stability_recheck"] = None
    if attempt["accepted"]:
        try:
            second = _attempt(candidate, task, replay_once())
        except Exception as exc:
            second = _attempt(candidate, task, None, f"{type(exc).__name__}: {exc}")
        attempt["stability_recheck"] = second
        violations = ["stability recheck: " + value for value in second["hard_violations"]]
        if not second["accepted"]:
            violations.append("accepted program failed independent full-suite recheck")
        elif second["canonical_key"] != attempt["canonical_key"]:
            violations.append("accepted program changed canonical mode on independent full-suite recheck")
        if violations:
            attempt["hard_violations"].extend(violations)
            attempt["accepted"] = False
            attempt["canonical_key"] = None
    return attempt


def _configure_frozen_verifier() -> Any:
    base_eval.DEVELOPMENT_PROBLEMS = DEVELOPMENT_PROBLEMS
    base_eval.EVALUATION_PROBLEMS = v6.EVALUATION_PROBLEMS
    base_eval._validate_gate_and_choose_suites = v6._validate_v6_gate_and_choose_suites
    base_eval._replay_modules = v6._replay_modules
    return v6._replay_modules()[1]


def _apply_prompt_overlay(public_rows: Sequence[Mapping[str, Any]], path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if overlay.get("schema_version") != "constructive-code-prompt-overlay-v1" or set(overlay.get("overlays", {})) != {"359_B"}:
        raise ValueError("pilot requires exactly the preregistered 359_B statement overlay")
    results = []
    for row in public_rows:
        updated = dict(row)
        updated["source_statement_sha256"] = row["statement_sha256"]
        change = overlay["overlays"].get(row["source_problem_id"])
        if change is not None:
            statement = row["statement"]
            replacement = change["replacement_statement"]
            if (
                change.get("source_statement_sha256") != hashlib.sha256(statement.encode("utf-8")).hexdigest()
                or change.get("replacement_statement_sha256") != hashlib.sha256(replacement.encode("utf-8")).hexdigest()
                or change.get("replacement_token") != "<image>"
                or statement.count("<image>") != 1
                or statement.replace("<image>", change["replacement_text"]) != replacement
                or change.get("source_image_sha256") != "7386a0480bd5b1bcfdca7729b5a041419b3c3641ce8096ee127fe239efa09060"
                or change.get("verification") != "visually_read_original_image"
            ):
                raise ValueError("359_B public equation overlay identity drift")
            updated["statement"] = replacement
            updated["statement_sha256"] = change["replacement_statement_sha256"]
        results.append(updated)
    return results, {"path": str(path.resolve()), "sha256": base_eval._sha256_file(path), "content": overlay, "historical_359_B_prompt_matches": False}


def _selected_audit_replays(path: Path, gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    selected = [row for problem, suite in v6.v6_gate.SELECTED_SUITES.items() for row in rows if row.get("source_problem_id") == problem and row.get("suite_id") == suite]
    if len(rows) != 2304 or len(selected) != 960 or base_eval._canonical_sha256(selected) != gate.get("selected_replays_sha256"):
        raise ValueError("historical gate replay ledger identity drift")
    return selected


def _sandbox_smoke(args: argparse.Namespace, base: Any, tasks: Sequence[Any], source_manifest: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    historical = _selected_audit_replays(args.gate_replays, gate)
    directories = {row["source_problem_id"]: args.slate_root / row["relative_path"] for row in source_manifest["tasks"]}
    receipts = []
    for task in tasks:
        records = base._load_jsonl(directories[task.problem_id] / "py3_replays.jsonl")
        by_hash = {row["submission_sha256"]: row for row in records}
        for label, expected_accepted in (("correct", True), ("incorrect", False)):
            eligible = sorted((row for row in historical if row["source_problem_id"] == task.problem_id and row["known_label"] == label and row["released_checker_accepted"] is expected_accepted and row["wrapper_accepted"] is expected_accepted), key=lambda row: row["submission_sha256"])
            if not eligible:
                raise ValueError(f"no audited {label} smoke program for {task.problem_id}")
            expected = eligible[0]
            source = by_hash[expected["submission_sha256"]]
            replay = base._replay_submission(task=task, submission=base.Submission(source["code"], label, source["submission_sha256"]), launcher=args.launcher, runtime_root=args.runtime_root, scratch_root=args.scratch_root)
            violations = base_eval._hard_replay_violations(replay)
            for field in ("source_problem_id", "problem_key", "suite_id", "suite_sha256", "checker_sha256", "submission_sha256", "released_checker_accepted", "wrapper_accepted", "behavior_key"):
                if replay.get(field) != expected.get(field):
                    violations.append(f"historical smoke mismatch: {field}")
            receipts.append({"source_problem_id": task.problem_id, "known_label": label, "expected_accepted": expected_accepted, "submission_sha256": source["submission_sha256"], "replay": replay, "violations": violations})
            if violations:
                raise RuntimeError(f"sandbox preflight failed {task.problem_id}/{label}: {violations}")
    return {"status": "pass", "selection": "lexicographically first SHA-256 per task/label whose historical released and wrapper acceptance matches label", "gate_replays_sha256": base_eval._sha256_file(args.gate_replays), "historical_selected_replays_sha256": gate["selected_replays_sha256"], "receipts": receipts}


def _progress(path: Path, stage: str, **values: Any) -> None:
    base_eval._atomic_json(path, {"schema_version": SCHEMA, "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **values})


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    paths = _receipt_paths(args.output)
    for path in (args.output, *paths.values()):
        if path.exists():
            raise FileExistsError(f"fresh pilot receipt required: {path}")
    for path in (args.slate_root, args.v1_root, args.model / "config.json", args.gate_audit, args.gate_identity, args.protocol, args.identity, args.image, args.prompt_overlay, args.gate_replays):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _progress(paths["progress"], "verifier_preflight")
    started = time.monotonic()
    base = _configure_frozen_verifier()
    gate = json.loads(args.gate_audit.read_text(encoding="utf-8"))
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    launcher_sha = base.build_launcher(base.SANDBOX_SOURCE, args.launcher)
    runtime_identity = base.prepare_runtime(args.image, args.runtime_root)
    tasks, public_rows, builds, source_manifest = base_eval._load_development_tasks(
        slate_root=args.slate_root, v1_root=args.v1_root, build_root=args.build_root, gate=gate,
    )
    if tuple(row["source_problem_id"] for row in public_rows) != DEVELOPMENT_PROBLEMS:
        raise RuntimeError("development row order drift")
    public_rows, overlay_receipt = _apply_prompt_overlay(public_rows, args.prompt_overlay)
    smoke = _sandbox_smoke(args, base, tasks, source_manifest, gate)
    gate_identity = json.loads(args.gate_identity.read_text(encoding="utf-8"))
    identity = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "model": str(args.model.resolve()),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        **{name + "_sha256": base_eval._sha256_file(getattr(args, name)) for name in ("protocol", "identity", "gate_audit", "gate_identity")},
        "model_config_sha256": base_eval._sha256_file(args.model / "config.json"),
        "gate_source_hash": gate_identity.get("source_hash"),
        "gate_execution_hash": gate_identity.get("execution_hash"),
        "source_manifest_sha256": base_eval._canonical_sha256(source_manifest),
        "checker_builds": builds,
        "prompt_overlay": overlay_receipt,
        "prompt_statements": [{"source_problem_id": row["source_problem_id"], "source_statement_sha256": row["source_statement_sha256"], "effective_statement_sha256": row["statement_sha256"], "effective_statement": row["statement"]} for row in public_rows],
        "sandbox_preflight": smoke,
        "runtime": {"launcher_binary_sha256": launcher_sha, "runtime_identity": asdict(runtime_identity)},
        "loaded_verifier_modules": {name: {"path": str(Path(module.__file__).resolve()), "sha256": base_eval._sha256_file(Path(module.__file__))} for name, module in (("base_eval", base_eval), ("v6", v6), ("replay_base", base))},
        "information_boundary": {
            "loaded_problem_ids": list(DEVELOPMENT_PROBLEMS),
            "evaluation_problem_ids": list(v6.EVALUATION_PROBLEMS),
            "evaluation_rows_loaded": False,
            **{name: False for name in ("reference_programs_in_context", "canonical_keys_in_context", "checker_source_in_context", "test_inputs_in_context", "gate_outcomes_in_context")},
        },
    }
    if args.preflight:
        payload = {**identity, "status": "preflight_pass", "decision": "no_model_sampling_or_training", "preflight_wall_seconds": time.monotonic() - started, "language_model_sampling": False}
        base_eval._atomic_json(args.output, payload)
        _progress(paths["progress"], "preflight_complete", output=str(args.output))
        return payload

    # Hash the complete local model before allocating a GPU; no downloads occur.
    _progress(paths["progress"], "model_identity")
    identity["model_tree_sha256"] = base_eval._tree_sha256(args.model)
    identity["tokenizer_tree_sha256"] = base_eval._tree_sha256(args.model, {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"})
    import torch
    import vllm

    gpu_count = torch.cuda.device_count()
    if gpu_count != 1:
        raise RuntimeError(f"pilot requires exactly one visible GPU, observed {gpu_count}")
    identity["gpu"] = {"visible_count": gpu_count, "names": [torch.cuda.get_device_name(i) for i in range(gpu_count)], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"), "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS")}
    identity["software"] = {"torch": torch.__version__, "vllm": vllm.__version__}
    _progress(paths["progress"], "model_loading")
    load_started = time.monotonic()
    llm = vllm.LLM(model=str(args.model.resolve()), dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.82, swap_space=16.0, enable_prefix_caching=True)
    model_load_seconds = time.monotonic() - load_started
    requests = []
    for row_index, public in enumerate(public_rows):
        prompt = base_eval._prompt(public["statement"])
        for sample_index in range(64):
            seed = base_eval._request_seed(row_index, sample_index, args.seed)
            requests.append({"row_index": row_index, "sample_index": sample_index, "request_seed": seed, "prompt": prompt})

    generated = []
    generation_batches = []
    generation_started = time.monotonic()
    with paths["responses"].open("x", encoding="utf-8") as handle:
        for offset in range(0, len(requests), args.generation_batch_size):
            batch = requests[offset : offset + args.generation_batch_size]
            batch_started = time.monotonic()
            outputs = llm.generate([row["prompt"] for row in batch], [vllm.SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1024, seed=row["request_seed"]) for row in batch], use_tqdm=False)
            batch_seconds = time.monotonic() - batch_started
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned the wrong request count")
            records = []
            for request, output in zip(batch, outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("vLLM returned other than one candidate")
                sample = output.outputs[0]
                emitted = str(sample.text)
                code, stripped = base_eval._strip_exact_surrounding_fence(emitted)
                records.append({
                    **{name: request[name] for name in ("row_index", "sample_index", "request_seed")},
                    "source_problem_id": public_rows[request["row_index"]]["source_problem_id"],
                    "prompt_sha256": hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest(),
                    "emitted_text": emitted, "code": code,
                    "emitted_text_sha256": hashlib.sha256(emitted.encode("utf-8")).hexdigest(),
                    "executed_source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "fence_stripped": stripped, "token_ids": list(sample.token_ids),
                    "token_count": len(sample.token_ids), "finish_reason": str(sample.finish_reason),
                    "stop_reason": sample.stop_reason,
                })
            _append_jsonl(handle, records)
            generated.extend(records)
            generation_batches.append({"offset": offset, "request_count": len(batch), "wall_seconds": batch_seconds})
            _progress(paths["progress"], "generating", generated=len(generated), expected=EXPECTED_REQUESTS, latest_batch_seconds=batch_seconds)
            print(f"[constructive-pilot] generated={len(generated)}/{EXPECTED_REQUESTS}", flush=True)
    generation_seconds = time.monotonic() - generation_started
    if len(generated) != EXPECTED_REQUESTS:
        raise RuntimeError("not all frozen requests generated")

    attempts = []
    execution_started = time.monotonic()
    with paths["attempts"].open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.execution_workers) as executor:
        futures = {}
        for candidate in generated:
            task = tasks[candidate["row_index"]]
            futures[executor.submit(_verify_candidate, base, args, candidate, task)] = candidate
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                attempt = future.result()
            except Exception as exc:
                attempt = _attempt(candidate, tasks[candidate["row_index"]], None, f"{type(exc).__name__}: {exc}")
            _append_jsonl(handle, [attempt])
            attempts.append(attempt)
            _progress(paths["progress"], "verifying", completed=len(attempts), expected=EXPECTED_REQUESTS, accepted=sum(a["accepted"] for a in attempts))
            if len(attempts) % 16 == 0:
                print(f"[constructive-pilot] executed={len(attempts)}/{EXPECTED_REQUESTS}", flush=True)
    execution_seconds = time.monotonic() - execution_started
    attempts.sort(key=lambda row: (row["row_index"], row["sample_index"]))
    outcome = _summarize(attempts, public_rows)
    tokens = sum(row["token_count"] for row in generated)
    total_seconds = time.monotonic() - started
    payload = {
        **identity, **outcome,
        "sampling": {"seed": args.seed, "request_seed_formula": "77101 + 10000 * row_index + sample_index", "sample_count_per_task": 64, "prefix_count": 16, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1024, "max_model_len": args.max_model_len, "generation_batch_size": args.generation_batch_size, "system_message_sha256": hashlib.sha256(base_eval.SYSTEM_MESSAGE.encode("utf-8")).hexdigest()},
        "criteria": {"minimum_aggregate_accuracy": MINIMUM_AGGREGATE_ACCURACY, "minimum_multimode_tasks": MINIMUM_MULTIMODE_TASKS, "minimum_modes_per_multimode_task": 2, "minimum_accepted_for_pcmd": MINIMUM_PCMD_ACCEPTED, "accepted_program_full_suite_replays": 2, "require_identical_replayed_mode": True, "main_study_authorized": False},
        "metric_definitions": {"pass_at_k": "1 - choose(n-c,k)/choose(n,k), macro mean over all three tasks", "pcmd": "1 - sum_c n_c(n_c-1)/(m(m-1)), reported only if m>=30", "canonical_mode": "unchanged v6 audited canonicalized accepted behavior over the frozen suite; not algorithm identity"},
        "timing": {"model_load_seconds": model_load_seconds, "generation_wall_seconds": generation_seconds, "generation_call_seconds": sum(b["wall_seconds"] for b in generation_batches), "execution_wall_seconds": execution_seconds, "total_evaluator_wall_seconds": total_seconds, "generated_tokens": tokens, "generation_samples_per_second": EXPECTED_REQUESTS / generation_seconds, "generation_tokens_per_second": tokens / generation_seconds, "estimated_allocated_gpu_hours_during_evaluator": gpu_count * total_seconds / 3600, "gpu_hour_scope": "one allocated GPU times evaluator elapsed; excludes scheduler startup and teardown", "generation_batches": generation_batches},
        "artifacts": {name: {"path": str(paths[name].resolve()), "sha256": base_eval._sha256_file(paths[name])} for name in ("responses", "attempts")},
        "attempts": attempts,
    }
    base_eval._atomic_json(args.output, payload)
    _progress(paths["progress"], "complete", status=payload["status"], output=str(args.output), summary=payload["summary"])
    print(f"[constructive-pilot] status={payload['status']} accepted={outcome['summary']['accepted_candidates']}/{EXPECTED_REQUESTS} multimode_tasks={outcome['summary']['multimode_tasks']}/3 output={args.output}", flush=True)
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    raise SystemExit(0 if payload["status"] in ("pass", "preflight_pass") else 1)


if __name__ == "__main__":
    main()
