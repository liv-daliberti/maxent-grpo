#!/usr/bin/env python3
"""CPU-only posthoc stress test of policy programs accepted by frozen reward.

No model calls, reward changes, or replay-bank insertion. Every admitted program
is rechecked twice on the separate diagnostic suite with original checker bytes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace
import build_constructive_code_wider_20260921 as b
from build_constructive_stress_manifest_20260921 import SCHEMA, probe_task


def bound_rows(evaluation):
    rows = {}
    for kind in ("responses", "attempts"):
        meta = evaluation["artifacts"][kind]; path = Path(meta["path"])
        if b.digest(path) != meta["sha256"]:
            raise ValueError(f"{kind} sidecar identity drift")
        values = [json.loads(line) for line in path.read_text().splitlines()]
        mapping = {(r["task_id"], r["sample_index"]): r for r in values}
        if len(mapping) != len(values):
            raise ValueError(f"duplicate {kind} request")
        rows[kind] = mapping
    if rows["responses"].keys() != rows["attempts"].keys():
        raise ValueError("response/attempt request coverage mismatch")
    for key, response in rows["responses"].items():
        attempt = rows["attempts"][key]
        if b.raw_sha256(response["text"]) != response["text_sha256"]:
            raise ValueError("response source-text SHA drift")
        for field in ("task_id", "sample_index", "request_seed", "text_sha256", "prompt_sha256"):
            if response[field] != attempt[field]:
                raise ValueError("response/attempt identity mismatch")
        if type(attempt["accepted"]) is not bool:
            raise ValueError("source acceptance is not boolean")
    return rows


def run(args):
    if args.output.exists() or args.output.with_suffix(".attempts.jsonl").exists():
        raise FileExistsError("stress artifacts already exist")
    evaluation = json.loads(args.evaluation.read_text()); fixture = json.loads(args.stress_manifest.read_text())
    if fixture["schema"] != SCHEMA or fixture["status"] != "reference_audit_pass" or fixture["reward_data_modified"]:
        raise ValueError("stress fixture failed reference preflight")
    if b.digest(args.stress_manifest) != args.stress_manifest_sha256:
        raise ValueError("stress fixture SHA drift")
    rows = bound_rows(evaluation); cases = {r["task_id"]: r for r in fixture["cases"]}
    selected = [row for key,row in rows["responses"].items() if key[0] in cases and rows["attempts"][key]["accepted"] and not rows["attempts"][key]["hard_violations"]]
    config = dict(evaluation["config"]["adapter_config"])
    config.update({field: str(getattr(args, field)) for field in ("build_root", "launcher", "scratch_root", "runtime_root")})
    config["problem_ids"] = [case["task_id"] for case in fixture["cases"] if any(r["task_id"] == case["task_id"] for r in selected)]
    tasks = b.load_tasks(config) if config["problem_ids"] else []
    mapping = {}
    for task in tasks:
        case = cases[task.task_id]
        if task._task.checker_sha256 != case["checker_sha256"] or task._task.suite_sha256 != case["frozen_reward_suite_sha256"]:
            raise ValueError("diagnostic task does not match frozen source checker/reward suite")
        if b.raw_sha256(case["stdin"]) != case["input_sha256"]:
            raise ValueError("diagnostic input SHA drift")
        task._task = probe_task(task._task, case["stdin"]); mapping[task.task_id] = task
    args.output.parent.mkdir(parents=True, exist_ok=True); receipts = []
    receipt_path = args.output.with_suffix(".attempts.jsonl")
    with receipt_path.open("x") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(mapping[row["task_id"]].verify, row["text"]): row for row in selected}
        for future in as_completed(futures):
            source = futures[future]; result = future.result()
            receipt = {key: source[key] for key in ("task_id", "sample_index", "request_seed", "text_sha256")}
            receipt.update({"frozen_reward_accepted": True, "stress": result})
            handle.write(json.dumps(receipt, sort_keys=True) + "\n"); handle.flush(); receipts.append(receipt)
    per_task = []
    for task_id in cases:
        task_rows = [r for r in receipts if r["task_id"] == task_id]; n = len(task_rows); accepted = sum(r["stress"]["accepted"] for r in task_rows)
        per_task.append({"task_id": task_id, "frozen_reward_accepted_programs_tested": n, "stress_accepted": accepted, "stress_rejected": n-accepted, "stress_survival_fraction": accepted/n if n else None, "hard_violation_count": sum(len(r["stress"]["hard_violations"]) for r in task_rows)})
    result = {"schema": "constructive-code-posthoc-stress-result-20260921-v1", "status": "complete" if all(not r["stress"]["hard_violations"] for r in receipts) else "integrity_failure", "out_of_reward_diagnostic_only": True, "model_calls": 0, "source_evaluation_sha256": b.digest(args.evaluation), "stress_manifest_sha256": b.digest(args.stress_manifest), "source_evaluation": str(args.evaluation.resolve()), "tasks": per_task, "tested": len(receipts), "receipts": {"path": str(receipt_path.resolve()), "sha256": b.digest(receipt_path)}}
    b.write_json(args.output, result)
    print(json.dumps({"status": result["status"], "tested": len(receipts), "tasks": per_task}, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True); p.add_argument("--stress-manifest", type=Path, required=True)
    p.add_argument("--stress-manifest-sha256", required=True); p.add_argument("--output", type=Path, required=True)
    for field in ("runtime-root", "launcher", "scratch-root", "build-root"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        p.error("workers must be in[1,32]")
    run(args)
