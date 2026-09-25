#!/usr/bin/env python3
"""CPU-only diagnostic regrading of complete, immutable generated samples.

This does not produce a primary endpoint or claim a new model run. It is used
to reassess a development screen after source-audit verifier corrections.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path
import time

from evaluate_real_domains_20260921 import (
    append_rows, atomic_json, load_task_selection, sha256, summarize, verify_one,
)


def run(original_run: Path, config_path: Path, output: Path, source_exclusions: Path | None = None) -> dict:
    started = time.monotonic()
    original_path = original_run / "evaluation.json"
    original = json.loads(original_path.read_text())
    config = json.loads(config_path.read_text())
    if original["status"] not in ("complete", "audit_fail"):
        raise ValueError("diagnostic regrading requires a terminal receipt and complete raw sampling")
    generation_fields = set(original["config"]) | set(config)
    for key in generation_fields - {"adapter_module", "adapter_config", "execution_workers", "task_ids"}:
        if original["config"].get(key) != config.get(key):
            raise ValueError(f"generation setting changed during CPU regrading: {key}")
    old_ids = original["config"]["task_ids"]
    new_ids = config["task_ids"]
    if not new_ids or len(new_ids) != len(set(new_ids)) or new_ids != [t for t in old_ids if t in set(new_ids)]:
        raise ValueError("task selection must preserve original order without introducing tasks")
    excluded = set(old_ids) - set(new_ids)
    exclusion_record = json.loads(source_exclusions.read_text()) if source_exclusions else None
    if excluded:
        if not exclusion_record or set(exclusion_record.get("excluded_tasks", {})) != excluded:
            raise ValueError("every excluded task requires an explicit source-admission receipt")
        for task_id, exclusion in exclusion_record["excluded_tasks"].items():
            if not isinstance(exclusion, dict) or not exclusion.get("reason") or not exclusion.get("receipt_sha256") or not exclusion.get("receipt_path"):
                raise ValueError("source exclusion reasons and concrete audit receipts are required")
            receipt_path = Path(exclusion["receipt_path"])
            if sha256(receipt_path) != exclusion["receipt_sha256"]:
                raise ValueError("source exclusion audit hash mismatch")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("source_problem_id", receipt.get("problem_id")) != task_id or receipt.get("status") not in ("fail", "quarantined", "source_admission_fail"):
                raise ValueError("source exclusion receipt must name the excluded task and a failed source gate")
    elif exclusion_record:
        raise ValueError("unexpected source-exclusion receipt without exclusions")
    attempts_path = output.with_name(output.stem + ".attempts.jsonl")
    progress_path = output.with_name(output.stem + ".progress.json")
    if any(p.exists() for p in (output, attempts_path, progress_path)):
        raise FileExistsError("choose new diagnostic output paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(progress_path, {"stage": "adapter_preflight", "cpu_only": True})
    frozen_identity_path = config_path.parent / "identity.json"
    frozen_identity = None
    if frozen_identity_path.exists():
        frozen_identity = json.loads(frozen_identity_path.read_text())
        if frozen_identity["config_sha256"] != sha256(config_path):
            raise ValueError("frozen diagnostic configuration changed")
        for row in frozen_identity["files"]:
            if sha256(Path(row["snapshot"])) != row["sha256"]:
                raise ValueError("frozen diagnostic dependency changed")
    adapter, tasks, dataset_identity = load_task_selection(config)
    project_root = Path(__file__).resolve().parent.parent
    source_dependencies = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and any(path.is_relative_to(project_root / directory) for directory in ("ops", "src")):
                source_dependencies[name] = {"path": str(path), "sha256": sha256(path)}
    original_prompts = {r["task_id"]: r for r in original["task_prompts"]}
    if {t.task_id for t in tasks} != set(new_ids) or set(original_prompts) != set(old_ids):
        raise ValueError("task denominators differ from the explicit source-admission selection")
    for task in tasks:
        old = original_prompts[task.task_id]
        if task.prompt != old["prompt"] or hashlib.sha256(task.prompt.encode()).hexdigest() != old["prompt_sha256"]:
            raise ValueError("verifier correction changed a model prompt")
    raw_path = Path(original["artifacts"]["responses"]["path"])
    raw_sha = sha256(raw_path)
    if raw_sha != original["artifacts"]["responses"]["sha256"]:
        raise ValueError("original generated samples changed")
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    expected = {(task_id, i) for task_id in old_ids for i in range(config["samples_per_task"])}
    identities = [(r["task_id"], r["sample_index"]) for r in raw]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("raw sampling denominator is incomplete or duplicated")
    task_positions = {task_id: i for i, task_id in enumerate(old_ids)}
    for row in raw:
        if row["text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("raw text hash mismatch")
        if row["prompt_sha256"] != original_prompts[row["task_id"]]["prompt_sha256"]:
            raise ValueError("raw sample prompt mismatch")
        if row["request_seed"] != config["seed"] + 10000 * task_positions[row["task_id"]] + row["sample_index"]:
            raise ValueError("raw request seed mismatch")
        if row["token_count"] != len(row["token_ids"]):
            raise ValueError("raw token-count mismatch")
    original_raw_count = len(raw)
    raw = [row for row in raw if row["task_id"] in set(new_ids)]
    task_map = {t.task_id: t for t in tasks}
    attempts = []
    workers = int(config.get("execution_workers", 8))
    if not 1 <= workers <= 32:
        raise ValueError("verification worker count outside [1,32]")
    with attempts_path.open("x") as handle, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verify_one, task_map[row["task_id"]], row) for row in raw]
        for future in as_completed(futures):
            row = future.result()
            append_rows(handle, [row])
            attempts.append(row)
            if len(attempts) % 32 == 0 or len(attempts) == len(raw):
                atomic_json(progress_path, {"stage": "verifying", "cpu_only": True, "completed": len(attempts), "expected": len(raw)})
    result = {
        "schema": "real-domains-cpu-diagnostic-regrade-20260921-v1",
        "kind": "diagnostic_only", "cpu_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": sha256(config_path),
        "generation_provenance": {
            "original_evaluation_path": str(original_path.resolve()),
            "original_evaluation_sha256": sha256(original_path),
            "original_runner_sha256": original["runner_sha256"],
            "original_config_sha256": original["config_sha256"],
            "original_job_id": original["job_id"],
            "original_evaluation_status": original["status"],
            "original_hard_violation_count": original.get("summary", {}).get("hard_violation_count"),
            "original_verifier_pass_is_not_assumed": True,
            "responses_path": str(raw_path.resolve()), "responses_sha256": raw_sha,
        },
        "verifier_provenance": {
            "adapter_path": str(Path(adapter.__file__).resolve()),
            "adapter_sha256": sha256(Path(adapter.__file__)),
            "dataset_identity": dataset_identity,
            "regrade_runner_sha256": sha256(Path(__file__)),
            "source_dependencies": source_dependencies,
            "frozen_identity": {"path": str(frozen_identity_path.resolve()), "sha256": sha256(frozen_identity_path)} if frozen_identity else None,
        },
        "task_prompts": [r for r in original["task_prompts"] if r["task_id"] in set(new_ids)],
        "original_sampling_denominator": original_raw_count,
        "source_excluded_samples": original_raw_count - len(raw),
        "source_exclusions": {"path": str(source_exclusions.resolve()), "sha256": sha256(source_exclusions), "record": exclusion_record} if source_exclusions else None,
        **summarize(tasks, attempts, config["samples_per_task"]),
        "artifacts": {"attempts": {"path": str(attempts_path.resolve()), "sha256": sha256(attempts_path)}},
        "cpu_wall_seconds": time.monotonic() - started,
        "primary_endpoint": False,
    }
    atomic_json(output, result)
    atomic_json(progress_path, {"stage": "complete", "status": result["status"], "cpu_only": True})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-exclusions", type=Path)
    args = parser.parse_args()
    result = run(args.original_run, args.config, args.output, args.source_exclusions)
    print(json.dumps(result["summary"], sort_keys=True))
    raise SystemExit(0 if result["status"] == "complete" else 1)
